//! Session-aware vs stateless retrieval A/B on LoCoMo follow-up questions.
//!
//! Controlled experiment for the session pointer bridge: for each LoCoMo QA
//! pair we take the original question (Q_full) and derive an under-specified
//! follow-up (Q_follow) by pronominalizing the speaker name
//! ("When did Melanie paint a sunrise?" -> "When did they paint a sunrise?").
//! Only pairs where the crate's own `is_follow_up` detector fires on
//! Q_follow (and not on Q_full) are kept, so the session mechanism is
//! guaranteed to engage in the session arm.
//!
//! Three arms run against the same index, same gold labels, same top_k:
//!   - `full_stateless`: Q_full with no session (reference: what the full
//!     question retrieves on its own).
//!   - `followup_stateless`: Q_follow with no session (under-specified
//!     baseline).
//!   - `followup_session`: one setup turn (Q_full observed under a fresh
//!     session id), then Q_follow under the same session id. The setup turn
//!     populates conversation state; the follow-up is rewritten against it.
//!
//! Two conversational-ranker arms test ranking improvements for multi-turn
//! retrieval on top of the session rewrite:
//!   - `session_prior`: deep (top-200) follow-up search under a fresh
//!     session, then each candidate turn rescored as
//!     base_score + max turn score in its session. Isolates the
//!     session-concentration effect.
//!   - `conv_rerank`: two-stage. Stage 1 scores sessions from the deep
//!     search; stage 2 scores every turn in the top-10 sessions with
//!     neighbor-context (±2 turns) term overlap, a speaker-match boost when
//!     the query names a person, and an interrogative-turn penalty for
//!     wh-questions.
//!
//! Retrieval goes through the public `MemoryService` / `MemorySearchService`
//! path (prepare -> retrieve -> observe), the same path the MCP server uses.
//! The query cache is disabled so every arm executes retrieval fresh.
//! Sessions are fresh per pair, so there is no cross-pair contamination.

use anyhow::{Context, Result};
use clap::Parser;
use lint_ai::memory_api::{AddRequest, MemoryService, Message, SearchRequest};
use lint_ai::session_prepare::is_follow_up;
use lint_ai::{
    stable_doc_id_from_source, ChunkStrategy, PipelineOptions, Tier1NerProvider,
    Tier1TermRankerKind,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const USER_ID: &str = "ab-user";
const TOP_K: usize = 20;

#[derive(Debug, Parser)]
#[command(name = "session-ab-benchmark")]
#[command(about = "A/B: session-aware vs stateless retrieval on LoCoMo follow-ups")]
struct Args {
    /// Path to locomo10.json from snap-research/locomo.
    #[arg(long)]
    locomo: PathBuf,

    /// Only process the first N conversations (smoke testing).
    #[arg(long)]
    limit_conversations: Option<usize>,

    /// Optional output path for the JSON report.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Weight-tuning mode: skip the fixed arms, build stage-2 candidates per
    /// pair once, then grid-search the conversational-rerank weights with
    /// 5-fold cross-validation. Prints a JSON tune report to stdout.
    #[arg(long)]
    tune: bool,
}

#[derive(Debug, Deserialize)]
struct LocomoConversation {
    sample_id: String,
    qa: Vec<LocomoQa>,
    conversation: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Clone)]
struct LocomoQa {
    question: String,
    category: u8,
    #[serde(default)]
    evidence: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct LocomoTurn {
    speaker: String,
    #[allow(dead_code)]
    dia_id: String,
    text: String,
}

/// Session keys look like `session_3`; date keys like `session_3_date_time`.
fn session_number(key: &str) -> Option<usize> {
    let rest = key.strip_prefix("session_")?;
    if rest.contains('_') {
        return None;
    }
    rest.parse::<usize>().ok()
}

/// Parse `D{session}:{turn}` evidence pointers into (session, 0-based turn)
/// pairs. Tolerates odd entries (e.g. `D8:6; D9:17`, `D:11:26`) the same way
/// the session-level parser did: the first two non-empty numeric pieces.
fn parse_evidence_turns(evidence: &[String]) -> HashSet<(usize, usize)> {
    let mut turns = HashSet::new();
    for entry in evidence {
        for part in entry.split(';') {
            let part = part.trim();
            let after_d = match part.strip_prefix('D') {
                Some(rest) => rest,
                None => continue,
            };
            let nums: Vec<usize> = after_d
                .split(':')
                .filter_map(|p| p.trim().parse::<usize>().ok())
                .collect();
            if nums.len() >= 2 && nums[1] >= 1 {
                turns.insert((nums[0], nums[1] - 1));
            }
        }
    }
    turns
}

/// Replace whole-word mentions of `name` in the question with pronouns:
/// possessive ("Caroline's") -> "their", bare name -> "they".
/// Returns None when the name does not appear as a whole word.
fn pronominalize(question: &str, name: &str) -> Option<String> {
    let lower_name = name.to_lowercase();
    let possessive = format!("{lower_name}'s");
    let possessive_curly = format!("{lower_name}’s");
    let mut found = false;
    let mut out: Vec<String> = Vec::new();
    for tok in question.split_whitespace() {
        // Split trailing punctuation so "Melanie?" still matches.
        let cut = tok
            .char_indices()
            .rev()
            .take_while(|(_, c)| !c.is_alphanumeric() && *c != '\'' && *c != '’')
            .count();
        let (word, punct) = tok.split_at(tok.len() - cut);
        let lower_word = word.to_lowercase();
        let replacement = if lower_word == lower_name {
            found = true;
            Some("they")
        } else if lower_word == possessive || lower_word == possessive_curly {
            found = true;
            Some("their")
        } else {
            None
        };
        match replacement {
            Some(r) => out.push(format!("{r}{punct}")),
            None => out.push(tok.to_string()),
        }
    }
    found.then(|| out.join(" "))
}

/// Parse a "s{session}t{turn}" key back into (session, turn_idx).
fn parse_turn_key(key: &str) -> Option<(usize, usize)> {
    let t = key.find('t')?;
    let s: usize = key[1..t].parse().ok()?;
    let i: usize = key[t + 1..].parse().ok()?;
    Some((s, i))
}

/// Crude tokenizer mirroring the analysis heuristic: lowercase alphanumeric
/// tokens longer than 3 chars.
fn tokenize_simple(s: &str) -> HashSet<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 3)
        .map(|w| w.to_lowercase())
        .collect()
}

const WH_WORDS: &[&str] = &[
    "what", "when", "where", "who", "whom", "whose", "which", "why", "how", "do", "does", "did",
    "is", "are", "was", "were", "can", "could", "would", "have", "has", "will",
];

/// Heuristic turn-role check: interrogative turns rarely contain answers.
/// Matches the validated analysis heuristic (60% of top-5 retrieved non-gold
/// turns are interrogative vs 17% of gold turns).
fn is_interrogative(text: &str) -> bool {
    let t = text.trim();
    if t.ends_with('?') {
        return true;
    }
    t.split_whitespace()
        .next()
        .map(|w| {
            let lw = w.to_lowercase();
            WH_WORDS.iter().any(|wh| *wh == lw)
        })
        .unwrap_or(false)
}

fn is_wh_question(query: &str) -> bool {
    query
        .trim()
        .split_whitespace()
        .next()
        .map(|w| {
            let lw = w
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            matches!(
                lw.as_str(),
                "what" | "when" | "where" | "who" | "whom" | "whose" | "which" | "why" | "how"
            )
        })
        .unwrap_or(false)
}

/// Session-prior rerank: rescore deep-search candidates as
/// base_score + max turn score in the candidate's session, concentrating the
/// ranking inside sessions the base ranker already trusts.
fn rerank_session_prior(broad: &[(String, f32)], top_k: usize) -> Vec<String> {
    let mut sess_max: HashMap<usize, f32> = HashMap::new();
    for (key, score) in broad {
        if let Some((s, _)) = parse_turn_key(key) {
            sess_max
                .entry(s)
                .and_modify(|m| *m = (*m).max(*score))
                .or_insert(*score);
        }
    }
    let mut rescored: Vec<(String, f32)> = broad
        .iter()
        .map(|(key, score)| {
            let prior = parse_turn_key(key)
                .and_then(|(s, _)| sess_max.get(&s).copied())
                .unwrap_or(0.0);
            (key.clone(), score + prior)
        })
        .collect();
    rescored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    rescored.into_iter().take(top_k).map(|(k, _)| k).collect()
}

/// Two-stage conversational rerank.
///
/// Stage 1: score sessions by their best turn in the deep search.
/// Stage 2: score every turn in the top-10 sessions with min-max normalized
/// base score, session score, and neighbor-context (±2 turns) term overlap,
/// plus a speaker-match boost when the query names a person and an
/// interrogative-turn penalty for wh-questions.
#[allow(clippy::too_many_arguments)]
fn rerank_conversational(
    broad: &[(String, f32)],
    turn_lookup: &HashMap<(usize, usize), (String, String)>,
    query_followup: &str,
    person_name: &str,
    top_k: usize,
) -> Vec<String> {
    let (cands, query_is_wh) = conv_candidates(broad, turn_lookup, query_followup, person_name);
    // Tuned weights (2026-09-22, 167-pair LoCoMo A/B, 5-fold CV):
    // [w_base, w_sess, w_ctx, w_spk, w_q] = [0.5, 0.5, 1.5, 0.8, 0.5].
    // Round-1 grid winners were unanimous across folds on w_ctx=1.5 and
    // w_spk=0.8; CV mean test recall@10 0.438 vs 0.394 at the hand-set
    // [1.0, 1.0, 0.6, 0.4, 0.5]. A refinement round around the winners did
    // not move the CV mean (0.435), so tuning stopped there.
    rank_conv_cands(&cands, [0.5, 0.5, 1.5, 0.8, 0.5], query_is_wh, top_k)
}

/// One candidate turn for the conversational rerank, with min-max normalized
/// features (normalization is per query over the candidate set).
#[derive(Debug, Clone)]
struct ConvCand {
    key: String,
    n_base: f32,
    n_sess: f32,
    n_ctx: f32,
    spk: f32,
    q: f32,
}

/// Build the stage-2 candidate set: every turn in the top-10 sessions by
/// stage-1 (deep search) session score, with normalized features.
fn conv_candidates(
    broad: &[(String, f32)],
    turn_lookup: &HashMap<(usize, usize), (String, String)>,
    query_followup: &str,
    person_name: &str,
) -> (Vec<ConvCand>, bool) {
    let mut sess_max: HashMap<usize, f32> = HashMap::new();
    let mut base_of: HashMap<String, f32> = HashMap::new();
    for (key, score) in broad {
        base_of.insert(key.clone(), *score);
        if let Some((s, _)) = parse_turn_key(key) {
            sess_max
                .entry(s)
                .and_modify(|m| *m = (*m).max(*score))
                .or_insert(*score);
        }
    }
    let mut top_sess: Vec<usize> = sess_max.keys().copied().collect();
    top_sess.sort_by(|a, b| {
        sess_max[&b]
            .partial_cmp(&sess_max[&a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    top_sess.truncate(10);

    let mut qterms = tokenize_simple(query_followup);
    for w in person_name.split_whitespace() {
        qterms.insert(w.to_lowercase());
    }
    let query_is_wh = is_wh_question(query_followup);

    struct Raw {
        key: String,
        base: f32,
        sess: f32,
        ctx: f32,
        spk: f32,
        q: f32,
    }
    let mut raw: Vec<Raw> = Vec::new();
    for s in &top_sess {
        let mut idxs: Vec<usize> = turn_lookup
            .keys()
            .filter(|(ss, _)| ss == s)
            .map(|(_, i)| *i)
            .collect();
        idxs.sort_unstable();
        let Some(&max_idx) = idxs.last() else {
            continue;
        };
        for i in &idxs {
            let Some((speaker, text)) = turn_lookup.get(&(*s, *i)) else {
                continue;
            };
            let key = format!("s{s}t{i}");
            let lo = (*i as isize - 2).max(0) as usize;
            let hi = (*i + 2).min(max_idx);
            let mut win = String::new();
            for j in lo..=hi {
                if let Some((_, t)) = turn_lookup.get(&(*s, j)) {
                    win.push_str(t);
                    win.push(' ');
                }
            }
            let wtoks = tokenize_simple(&win);
            let inter = qterms.intersection(&wtoks).count() as f32;
            let ctx = if qterms.is_empty() {
                0.0
            } else {
                inter / qterms.len() as f32
            };
            raw.push(Raw {
                key: key.clone(),
                base: base_of.get(&key).copied().unwrap_or(0.0),
                sess: sess_max[s],
                ctx,
                spk: if !person_name.is_empty() && speaker == person_name {
                    1.0
                } else {
                    0.0
                },
                q: if is_interrogative(text) { 1.0 } else { 0.0 },
            });
        }
    }
    let (mut bmin, mut bmax) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut smin, mut smax) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut cmin, mut cmax) = (f32::INFINITY, f32::NEG_INFINITY);
    for c in &raw {
        bmin = bmin.min(c.base);
        bmax = bmax.max(c.base);
        smin = smin.min(c.sess);
        smax = smax.max(c.sess);
        cmin = cmin.min(c.ctx);
        cmax = cmax.max(c.ctx);
    }
    let norm = |v: f32, lo: f32, hi: f32| {
        if hi > lo {
            (v - lo) / (hi - lo)
        } else {
            0.0
        }
    };
    let cands = raw
        .into_iter()
        .map(|c| ConvCand {
            key: c.key,
            n_base: norm(c.base, bmin, bmax),
            n_sess: norm(c.sess, smin, smax),
            n_ctx: norm(c.ctx, cmin, cmax),
            spk: c.spk,
            q: c.q,
        })
        .collect();
    (cands, query_is_wh)
}

/// Score stage-2 candidates with a weight vector
/// [w_base, w_sess, w_ctx, w_spk, w_q] and return the top-k turn keys.
fn rank_conv_cands(
    cands: &[ConvCand],
    w: [f32; 5],
    query_is_wh: bool,
    top_k: usize,
) -> Vec<String> {
    let mut scored: Vec<(String, f32)> = cands
        .iter()
        .map(|c| {
            let score = w[0] * c.n_base + w[1] * c.n_sess + w[2] * c.n_ctx + w[3] * c.spk
                - if query_is_wh { w[4] * c.q } else { 0.0 };
            (c.key.clone(), score)
        })
        .collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.into_iter().take(top_k).map(|(k, _)| k).collect()
}

// ---------------------------------------------------------------------------
// Weight tuning for the conversational rerank.
// ---------------------------------------------------------------------------

/// One pair's precomputed stage-2 data for tuning: normalized candidates,
/// gold turn keys, and whether the follow-up is a wh-question.
#[derive(Debug, Clone)]
struct TunePair {
    cands: Vec<ConvCand>,
    relevant: HashSet<String>,
    query_is_wh: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct TuneWeights {
    w_base: f32,
    w_sess: f32,
    w_ctx: f32,
    w_spk: f32,
    w_q: f32,
}

impl TuneWeights {
    fn as_array(self) -> [f32; 5] {
        [self.w_base, self.w_sess, self.w_ctx, self.w_spk, self.w_q]
    }
}

/// (recall@10, mrr) for a ranked turn-key list, mirroring `score_arms`.
fn tune_rank_metrics(ranked: &[String], relevant: &HashSet<String>) -> (f64, f64) {
    if relevant.is_empty() {
        return (0.0, 0.0);
    }
    let hits_at_10 = ranked
        .iter()
        .take(10)
        .filter(|id| relevant.contains(*id))
        .count();
    let recall_at_10 = hits_at_10 as f64 / relevant.len() as f64;
    let mrr = ranked
        .iter()
        .position(|id| relevant.contains(id))
        .map(|i| 1.0 / (i as f64 + 1.0))
        .unwrap_or(0.0);
    (recall_at_10, mrr)
}

fn eval_tune_weights(pairs: &[TunePair], w: TuneWeights) -> (f64, f64) {
    let arr = w.as_array();
    let mut r10 = 0.0;
    let mut mrr = 0.0;
    for p in pairs {
        let ranked = rank_conv_cands(&p.cands, arr, p.query_is_wh, TOP_K);
        let (r, m) = tune_rank_metrics(&ranked, &p.relevant);
        r10 += r;
        mrr += m;
    }
    let n = pairs.len().max(1) as f64;
    (r10 / n, mrr / n)
}

fn tune_grid() -> Vec<TuneWeights> {
    // Refinement grid centered on the round-1 winners
    // (w_ctx=1.5 and w_spk=0.8 hit the round-1 ceiling in all 5 folds).
    let mut grid = Vec::new();
    for &w_base in &[0.25f32, 0.5, 1.0] {
        for &w_sess in &[0.25f32, 0.5, 1.0] {
            for &w_ctx in &[1.0f32, 1.5, 2.0, 2.5] {
                for &w_spk in &[0.4f32, 0.8, 1.2] {
                    for &w_q in &[0.25f32, 0.5, 1.0] {
                        grid.push(TuneWeights {
                            w_base,
                            w_sess,
                            w_ctx,
                            w_spk,
                            w_q,
                        });
                    }
                }
            }
        }
    }
    grid
}

/// Best grid weights on `train` by mean recall@10 (ties broken by MRR, then
/// first in grid order for determinism).
fn grid_search_best(train: &[TunePair], grid: &[TuneWeights]) -> (TuneWeights, f64, f64) {
    let mut best: Option<(TuneWeights, f64, f64)> = None;
    for &w in grid {
        let (r10, mrr) = eval_tune_weights(train, w);
        let better = match &best {
            None => true,
            Some((_, br10, bmrr)) => r10 > *br10 || (r10 == *br10 && mrr > *bmrr),
        };
        if better {
            best = Some((w, r10, mrr));
        }
    }
    best.expect("grid is non-empty")
}

#[derive(Debug, Serialize)]
struct TuneFoldReport {
    fold: usize,
    test_pairs: usize,
    best_weights: TuneWeights,
    train_recall_at_10: f64,
    train_mrr: f64,
    test_recall_at_10: f64,
    test_mrr: f64,
}

#[derive(Debug, Serialize)]
struct TuneReport {
    mode: String,
    pairs: usize,
    grid_size: usize,
    baseline_weights: TuneWeights,
    baseline_full_recall_at_10: f64,
    baseline_full_mrr: f64,
    folds: Vec<TuneFoldReport>,
    cv_mean_test_recall_at_10: f64,
    cv_mean_test_mrr: f64,
    full_data_best_weights: TuneWeights,
    full_data_best_recall_at_10: f64,
    full_data_best_mrr: f64,
}

fn run_tune_report(pairs: Vec<TunePair>) -> TuneReport {
    const FOLDS: usize = 5;
    let grid = tune_grid();
    let baseline = TuneWeights {
        w_base: 1.0,
        w_sess: 1.0,
        w_ctx: 0.6,
        w_spk: 0.4,
        w_q: 0.5,
    };
    let (b_r10, b_mrr) = eval_tune_weights(&pairs, baseline);

    let mut folds = Vec::new();
    for fold in 0..FOLDS {
        let train: Vec<TunePair> = pairs
            .iter()
            .enumerate()
            .filter(|(i, _)| i % FOLDS != fold)
            .map(|(_, p)| p.clone())
            .collect();
        let test: Vec<TunePair> = pairs
            .iter()
            .enumerate()
            .filter(|(i, _)| i % FOLDS == fold)
            .map(|(_, p)| p.clone())
            .collect();
        let (best_w, train_r10, train_mrr) = grid_search_best(&train, &grid);
        let (test_r10, test_mrr) = eval_tune_weights(&test, best_w);
        folds.push(TuneFoldReport {
            fold,
            test_pairs: test.len(),
            best_weights: best_w,
            train_recall_at_10: train_r10,
            train_mrr: train_mrr,
            test_recall_at_10: test_r10,
            test_mrr: test_mrr,
        });
    }
    let cv_mean_test_recall_at_10 =
        folds.iter().map(|f| f.test_recall_at_10).sum::<f64>() / FOLDS as f64;
    let cv_mean_test_mrr = folds.iter().map(|f| f.test_mrr).sum::<f64>() / FOLDS as f64;

    let (full_best, full_r10, full_mrr) = grid_search_best(&pairs, &grid);

    TuneReport {
        mode: "tune".to_string(),
        pairs: pairs.len(),
        grid_size: grid.len(),
        baseline_weights: baseline,
        baseline_full_recall_at_10: b_r10,
        baseline_full_mrr: b_mrr,
        folds,
        cv_mean_test_recall_at_10,
        cv_mean_test_mrr,
        full_data_best_weights: full_best,
        full_data_best_recall_at_10: full_r10,
        full_data_best_mrr: full_mrr,
    }
}

#[derive(Debug, Clone, Serialize)]
struct ArmMetrics {
    recall_at_5: f64,
    recall_at_10: f64,
    recall_any_at_10: f64,
    mrr: f64,
    ndcg_at_10: f64,
    latency_ms: f64,
}

fn score_arms(retrieved: &[String], relevant: &HashSet<String>, latency_ms: f64) -> ArmMetrics {
    let recall = |k: usize| {
        if relevant.is_empty() {
            return 0.0;
        }
        let hits = retrieved
            .iter()
            .take(k.min(retrieved.len()))
            .filter(|id| relevant.contains(*id))
            .count();
        hits as f64 / relevant.len() as f64
    };
    let recall_any = |k: usize| {
        if relevant.is_empty() {
            return 0.0;
        }
        f64::from(
            retrieved
                .iter()
                .take(k.min(retrieved.len()))
                .any(|id| relevant.contains(id)),
        )
    };
    let mrr = retrieved
        .iter()
        .position(|id| relevant.contains(id))
        .map(|i| 1.0 / (i as f64 + 1.0))
        .unwrap_or(0.0);
    let mut dcg = 0.0;
    for (idx, id) in retrieved.iter().take(10).enumerate() {
        if relevant.contains(id) {
            dcg += 1.0 / ((idx as f64 + 2.0).log2());
        }
    }
    let idcg: f64 = (0..relevant.len().min(10))
        .map(|idx| 1.0 / ((idx as f64 + 2.0).log2()))
        .sum();
    ArmMetrics {
        recall_at_5: recall(5),
        recall_at_10: recall(10),
        recall_any_at_10: recall_any(10),
        mrr,
        ndcg_at_10: if idcg == 0.0 { 0.0 } else { dcg / idcg },
        latency_ms,
    }
}

#[derive(Debug, Clone, Serialize)]
struct PairResult {
    id: String,
    conversation: String,
    category: u8,
    question_full: String,
    question_followup: String,
    relevant_turns: Vec<String>,
    retrieved_full: Vec<String>,
    retrieved_followup_stateless: Vec<String>,
    retrieved_followup_session: Vec<String>,
    retrieved_session_prior: Vec<String>,
    retrieved_conv_rerank: Vec<String>,
    full_stateless: ArmMetrics,
    followup_stateless: ArmMetrics,
    followup_session: ArmMetrics,
    session_prior: ArmMetrics,
    conv_rerank: ArmMetrics,
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Paired t-statistic for per-pair deltas (descriptive; not a claim of
/// significance on its own).
fn paired_t(deltas: &[f64]) -> f64 {
    let n = deltas.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let m = mean(deltas);
    let var = deltas.iter().map(|d| (d - m).powi(2)).sum::<f64>() / (n - 1.0);
    if var == 0.0 {
        return 0.0;
    }
    m / (var / n).sqrt()
}

#[derive(Debug, Serialize, Default)]
struct ArmAggregate {
    pairs: usize,
    recall_at_5: f64,
    recall_at_10: f64,
    recall_any_at_10: f64,
    mrr: f64,
    ndcg_at_10: f64,
    mean_latency_ms: f64,
}

#[derive(Debug, Serialize)]
struct DeltaSummary {
    mean_delta: f64,
    paired_t: f64,
    improved: usize,
    tied: usize,
    degraded: usize,
}

#[derive(Debug, Serialize)]
struct Report {
    dataset: String,
    commit: String,
    conversations: usize,
    pairs: usize,
    top_k: usize,
    arms: HashMap<String, ArmAggregate>,
    /// followup_session minus followup_stateless, per metric.
    session_vs_stateless: HashMap<String, DeltaSummary>,
    /// followup_session minus full_stateless: how much of full-question
    /// performance the session recovers on the under-specified follow-up.
    session_vs_full: HashMap<String, DeltaSummary>,
    /// session_prior minus followup_session: the session-concentration effect.
    sessionprior_vs_session: HashMap<String, DeltaSummary>,
    /// conv_rerank minus followup_session: the full two-stage conversational
    /// ranker effect.
    conv_vs_session: HashMap<String, DeltaSummary>,
    by_category: HashMap<String, HashMap<String, ArmAggregate>>,
    per_pair: Vec<PairResult>,
}

fn aggregate_arm(pairs: &[PairResult], pick: fn(&PairResult) -> &ArmMetrics) -> ArmAggregate {
    let ms: Vec<&ArmMetrics> = pairs.iter().map(pick).collect();
    ArmAggregate {
        pairs: pairs.len(),
        recall_at_5: mean(&ms.iter().map(|m| m.recall_at_5).collect::<Vec<_>>()),
        recall_at_10: mean(&ms.iter().map(|m| m.recall_at_10).collect::<Vec<_>>()),
        recall_any_at_10: mean(&ms.iter().map(|m| m.recall_any_at_10).collect::<Vec<_>>()),
        mrr: mean(&ms.iter().map(|m| m.mrr).collect::<Vec<_>>()),
        ndcg_at_10: mean(&ms.iter().map(|m| m.ndcg_at_10).collect::<Vec<_>>()),
        mean_latency_ms: mean(&ms.iter().map(|m| m.latency_ms).collect::<Vec<_>>()),
    }
}

fn delta_summary(
    pairs: &[PairResult],
    metric: fn(&ArmMetrics) -> f64,
    a: fn(&PairResult) -> &ArmMetrics,
    b: fn(&PairResult) -> &ArmMetrics,
) -> DeltaSummary {
    let deltas: Vec<f64> = pairs.iter().map(|p| metric(b(p)) - metric(a(p))).collect();
    let improved = deltas.iter().filter(|d| **d > 0.0).count();
    let degraded = deltas.iter().filter(|d| **d < 0.0).count();
    DeltaSummary {
        mean_delta: mean(&deltas),
        paired_t: paired_t(&deltas),
        improved,
        tied: deltas.len() - improved - degraded,
        degraded,
    }
}

fn category_label(category: u8) -> &'static str {
    match category {
        1 => "multi-hop",
        2 => "temporal",
        3 => "open-domain",
        4 => "single-hop",
        5 => "adversarial",
        _ => "unknown",
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    // Every arm must execute retrieval; never serve one arm from another's
    // cache entry.
    std::env::set_var("LINT_AI_DISABLE_QUERY_CACHE", "1");

    let data = fs::read_to_string(&args.locomo)
        .with_context(|| format!("failed to read {}", args.locomo.display()))?;
    let conversations: Vec<LocomoConversation> =
        serde_json::from_str(&data).context("failed to parse LoCoMo JSON")?;
    let conversations: Vec<LocomoConversation> = conversations
        .into_iter()
        .take(args.limit_conversations.unwrap_or(usize::MAX))
        .collect();
    eprintln!("processing {} conversations...", conversations.len());

    let options = PipelineOptions {
        ner_provider: Tier1NerProvider::Heuristic,
        spacy_model: "en_core_web_sm".to_string(),
        term_ranker: Tier1TermRankerKind::Yake,
        chunk_strategy: ChunkStrategy::Heading,
        chunk_lines: 40,
        chunk_overlap: 10,
        chunk_target_tokens: 450,
        chunk_max_tokens: 800,
        ..PipelineOptions::default()
    };

    let mut per_pair: Vec<PairResult> = Vec::new();
    let mut tune_pairs: Vec<TunePair> = Vec::new();

    for conv in &conversations {
        let speaker_a = conv
            .conversation
            .get("speaker_a")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let speaker_b = conv
            .conversation
            .get("speaker_b")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Ingest every session as one AddRequest; group_id = session id.
        let mut session_nums: Vec<usize> = conv
            .conversation
            .keys()
            .filter_map(|k| session_number(k))
            .collect();
        session_nums.sort_unstable();

        let mut service = MemoryService::in_memory(options.clone());
        // doc_id -> "s{session}t{turn_idx}" turn key for turn-level scoring.
        let mut doc_to_turn: HashMap<String, String> = HashMap::new();
        // (session_n, turn_idx) -> (speaker, text) for the conversational
        // rerank arms (neighbor context, speaker match, interrogative check).
        let mut turn_lookup: HashMap<(usize, usize), (String, String)> = HashMap::new();
        let mut add_requests: Vec<AddRequest> = Vec::new();
        for n in &session_nums {
            let key = format!("session_{n}");
            let turns: Vec<LocomoTurn> = serde_json::from_value(
                conv.conversation
                    .get(&key)
                    .cloned()
                    .unwrap_or(serde_json::Value::Array(vec![])),
            )
            .with_context(|| format!("failed to parse turns for {key}"))?;
            let request_id = format!("ab-{}-s{n}", conv.sample_id);
            let group_id = format!("{}::session_{n}", conv.sample_id);
            let mut messages: Vec<Message> = Vec::new();
            for (turn_idx, turn) in turns.iter().enumerate() {
                let role = if turn.speaker == speaker_a {
                    "user"
                } else {
                    "assistant"
                };
                messages.push(Message {
                    role: role.to_string(),
                    timestamp: None,
                    // Keep the speaker name in the text so entity analysis
                    // sees the same surface form as the questions.
                    content: format!("{}: {}", turn.speaker, turn.text),
                    expires_at_ms: None,
                    supersedes_id: None,
                });
                let doc_id =
                    stable_doc_id_from_source(&format!("{USER_ID}:{request_id}:{turn_idx}"));
                doc_to_turn.insert(doc_id, format!("s{n}t{turn_idx}"));
                turn_lookup.insert((*n, turn_idx), (turn.speaker.clone(), turn.text.clone()));
            }
            if messages.is_empty() {
                continue;
            }
            add_requests.push(AddRequest {
                request_id,
                messages,
                user_id: USER_ID.to_string(),
                session_id: group_id,
            });
        }
        service.add_batch(add_requests)?;
        let searcher = service.published_search();
        eprintln!(
            "conversation {} indexed ({} sessions)",
            conv.sample_id,
            session_nums.len()
        );

        let search_scored = |query: &str,
                             session_id: Option<String>,
                             top_k: usize|
         -> Result<(Vec<(String, f32)>, f64)> {
            let start = Instant::now();
            let response = searcher.search(SearchRequest {
                query: query.to_string(),
                options: None,
                user_id: USER_ID.to_string(),
                top_k,
                session_id,
            })?;
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            let mut seen = HashSet::new();
            let mut out = Vec::new();
            for m in &response.data {
                if let Some(tk) = doc_to_turn.get(&m.id) {
                    if seen.insert(tk.clone()) {
                        out.push((tk.clone(), m.score));
                    }
                }
            }
            Ok((out, latency_ms))
        };

        let search = |query: &str, session_id: Option<String>| -> Result<(Vec<String>, f64)> {
            let (scored, latency_ms) = search_scored(query, session_id, TOP_K)?;
            Ok((scored.into_iter().map(|(k, _)| k).collect(), latency_ms))
        };

        let names: Vec<&str> = [speaker_a, speaker_b]
            .into_iter()
            .filter(|n| !n.is_empty())
            .collect();

        for (q_idx, q) in conv.qa.iter().enumerate() {
            if q.category == 5 {
                continue; // adversarial/abstention: recall is the wrong lens
            }
            let q_full = q.question.clone();
            if is_follow_up(&q_full) {
                continue;
            }
            // Find the first speaker name mentioned (left to right) and
            // pronominalize it.
            let mut q_follow: Option<String> = None;
            'names: for name in &names {
                // crude left-to-right: check each token for the name
                for tok in q_full.split_whitespace() {
                    let word =
                        tok.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '’');
                    let lw = word.to_lowercase();
                    let ln = name.to_lowercase();
                    if lw == ln || lw == format!("{ln}'s") || lw == format!("{ln}’s") {
                        q_follow = pronominalize(&q_full, name);
                        break 'names;
                    }
                }
            }
            let q_follow = match q_follow {
                Some(f) if f != q_full && is_follow_up(&f) => f,
                _ => continue,
            };
            // The speaker name that was pronominalized (for the speaker-match
            // signal in the conversational rerank arm).
            let q_name = names
                .iter()
                .find(|n| {
                    q_full.split_whitespace().any(|tok| {
                        let word = tok
                            .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '’');
                        let lw = word.to_lowercase();
                        let ln = n.to_lowercase();
                        lw == ln || lw == format!("{ln}'s") || lw == format!("{ln}’s")
                    })
                })
                .map(|s| s.to_string())
                .unwrap_or_default();

            let relevant: HashSet<String> = parse_evidence_turns(&q.evidence)
                .into_iter()
                .map(|(s, t)| format!("s{s}t{t}"))
                .collect();
            if relevant.is_empty() {
                continue;
            }
            let mut relevant_sorted: Vec<String> = relevant.iter().cloned().collect();
            relevant_sorted.sort();

            if args.tune {
                // Tune mode: build stage-2 candidates once per pair; the
                // grid search runs after all pairs are collected.
                let sid_cv = format!("abq-tune-{}-{q_idx}", conv.sample_id);
                let _ = search(&q_full, Some(sid_cv.clone()))?;
                let (broad_cv, _) = search_scored(&q_follow, Some(sid_cv), 200)?;
                let (cands, query_is_wh) =
                    conv_candidates(&broad_cv, &turn_lookup, &q_follow, &q_name);
                tune_pairs.push(TunePair {
                    cands,
                    relevant,
                    query_is_wh,
                });
                continue;
            }

            // Arm 1: full question, stateless (reference).
            let (g_full, t_full) = search(&q_full, None)?;
            // Arm 2: follow-up, stateless (under-specified baseline).
            let (g_a, t_a) = search(&q_follow, None)?;
            // Arm 3: setup turn populates session state, then the follow-up.
            let sid = format!("abq-{}-{q_idx}", conv.sample_id);
            let _ = search(&q_full, Some(sid.clone()))?;
            let (g_b, t_b) = search(&q_follow, Some(sid))?;

            // Arm 4: session-prior rerank over a deep (top-200) candidate pool.
            let sid_sp = format!("abq-sp-{}-{q_idx}", conv.sample_id);
            let _ = search(&q_full, Some(sid_sp.clone()))?;
            let sp_start = Instant::now();
            let (broad_sp, _) = search_scored(&q_follow, Some(sid_sp), 200)?;
            let g_sp = rerank_session_prior(&broad_sp, TOP_K);
            let t_sp = sp_start.elapsed().as_secs_f64() * 1000.0;

            // Arm 5: two-stage conversational rerank.
            let sid_cv = format!("abq-cv-{}-{q_idx}", conv.sample_id);
            let _ = search(&q_full, Some(sid_cv.clone()))?;
            let cv_start = Instant::now();
            let (broad_cv, _) = search_scored(&q_follow, Some(sid_cv), 200)?;
            let g_cv = rerank_conversational(&broad_cv, &turn_lookup, &q_follow, &q_name, TOP_K);
            let t_cv = cv_start.elapsed().as_secs_f64() * 1000.0;

            per_pair.push(PairResult {
                id: format!("{}_q{q_idx}", conv.sample_id),
                conversation: conv.sample_id.clone(),
                category: q.category,
                question_full: q_full,
                question_followup: q_follow,
                relevant_turns: relevant_sorted,
                retrieved_full: g_full.clone(),
                retrieved_followup_stateless: g_a.clone(),
                retrieved_followup_session: g_b.clone(),
                retrieved_session_prior: g_sp.clone(),
                retrieved_conv_rerank: g_cv.clone(),
                full_stateless: score_arms(&g_full, &relevant, t_full),
                followup_stateless: score_arms(&g_a, &relevant, t_a),
                followup_session: score_arms(&g_b, &relevant, t_b),
                session_prior: score_arms(&g_sp, &relevant, t_sp),
                conv_rerank: score_arms(&g_cv, &relevant, t_cv),
            });
        }
        eprintln!(
            "conversation {} done ({} pairs so far)",
            conv.sample_id,
            per_pair.len()
        );
    }

    if args.tune {
        let report = run_tune_report(tune_pairs);
        println!(
            "{}",
            serde_json::to_string_pretty(&report).context("failed to serialize tune report")?
        );
        return Ok(());
    }

    fn pick_full(p: &PairResult) -> &ArmMetrics {
        &p.full_stateless
    }
    fn pick_a(p: &PairResult) -> &ArmMetrics {
        &p.followup_stateless
    }
    fn pick_b(p: &PairResult) -> &ArmMetrics {
        &p.followup_session
    }
    fn pick_sp(p: &PairResult) -> &ArmMetrics {
        &p.session_prior
    }
    fn pick_cv(p: &PairResult) -> &ArmMetrics {
        &p.conv_rerank
    }
    let mut arms: HashMap<String, ArmAggregate> = HashMap::new();
    arms.insert(
        "full_stateless".to_string(),
        aggregate_arm(&per_pair, pick_full),
    );
    arms.insert(
        "followup_stateless".to_string(),
        aggregate_arm(&per_pair, pick_a),
    );
    arms.insert(
        "followup_session".to_string(),
        aggregate_arm(&per_pair, pick_b),
    );
    arms.insert(
        "session_prior".to_string(),
        aggregate_arm(&per_pair, pick_sp),
    );
    arms.insert("conv_rerank".to_string(), aggregate_arm(&per_pair, pick_cv));

    let metrics: Vec<(&str, fn(&ArmMetrics) -> f64)> = vec![
        ("recall_at_5", |m: &ArmMetrics| m.recall_at_5),
        ("recall_at_10", |m: &ArmMetrics| m.recall_at_10),
        ("recall_any_at_10", |m: &ArmMetrics| m.recall_any_at_10),
        ("mrr", |m: &ArmMetrics| m.mrr),
        ("ndcg_at_10", |m: &ArmMetrics| m.ndcg_at_10),
    ];
    let mut session_vs_stateless: HashMap<String, DeltaSummary> = HashMap::new();
    let mut session_vs_full: HashMap<String, DeltaSummary> = HashMap::new();
    let mut sessionprior_vs_session: HashMap<String, DeltaSummary> = HashMap::new();
    let mut conv_vs_session: HashMap<String, DeltaSummary> = HashMap::new();
    for (name, metric) in &metrics {
        session_vs_stateless.insert(
            name.to_string(),
            delta_summary(&per_pair, *metric, pick_a, pick_b),
        );
        session_vs_full.insert(
            name.to_string(),
            delta_summary(&per_pair, *metric, pick_full, pick_b),
        );
        sessionprior_vs_session.insert(
            name.to_string(),
            delta_summary(&per_pair, *metric, pick_b, pick_sp),
        );
        conv_vs_session.insert(
            name.to_string(),
            delta_summary(&per_pair, *metric, pick_b, pick_cv),
        );
    }

    let mut by_category: HashMap<String, HashMap<String, ArmAggregate>> = HashMap::new();
    {
        let mut cats: HashMap<u8, Vec<PairResult>> = HashMap::new();
        for p in per_pair.iter().cloned() {
            cats.entry(p.category).or_default().push(p);
        }
        let mut cat_ids: Vec<u8> = cats.keys().cloned().collect();
        cat_ids.sort_unstable();
        for cat in cat_ids {
            let qs = &cats[&cat];
            let mut m: HashMap<String, ArmAggregate> = HashMap::new();
            m.insert("full_stateless".to_string(), aggregate_arm(qs, pick_full));
            m.insert("followup_stateless".to_string(), aggregate_arm(qs, pick_a));
            m.insert("followup_session".to_string(), aggregate_arm(qs, pick_b));
            m.insert("session_prior".to_string(), aggregate_arm(qs, pick_sp));
            m.insert("conv_rerank".to_string(), aggregate_arm(qs, pick_cv));
            by_category.insert(category_label(cat).to_string(), m);
        }
    }

    // Sort per-pair rows for determinism.
    per_pair.sort_by(|a, b| a.id.cmp(&b.id));

    let report = Report {
        dataset: "LoCoMo (snap-research/locomo locomo10.json), pronominalized follow-up pairs, turn-level scoring".to_string(),
        commit: env!("CARGO_PKG_VERSION").to_string(),
        conversations: conversations.len(),
        pairs: per_pair.len(),
        top_k: TOP_K,
        arms,
        session_vs_stateless,
        session_vs_full,
        sessionprior_vs_session,
        conv_vs_session,
        by_category,
        per_pair,
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(out) = args.out {
        fs::write(&out, &json)
            .with_context(|| format!("failed to write report to {}", out.display()))?;
        println!("wrote session A/B report to {}", out.display());
    } else {
        println!("{json}");
    }
    Ok(())
}
