//! Conversational rerank for multi-turn retrieval.
//!
//! When a query arrives as a follow-up inside a session, the base retrieval
//! often finds the right *session* (document group) but misses the exact
//! turn: follow-ups are pronominalized ("when did they paint it?") and the
//! gold turn rarely shares query terms. This module implements the two-stage
//! rerank validated on 167 synthetic LoCoMo follow-up pairs (2026-09-22):
//!
//! 1. **Stage 1 (deep search).** Retrieve the top [`RERANK_DEEP_TOP_K`]
//!    turns and rank document groups by their best turn score.
//! 2. **Stage 2 (pinpoint).** Score *every* turn in the top
//!    [`RERANK_TOP_GROUPS`] groups with min-max normalized features:
//!    base retrieval score, group score, ±[`RERANK_NEIGHBOR_WINDOW`]-turn
//!    neighbor-context term overlap with the query, a speaker-match boost
//!    when the turn's speaker is one of the session's resolved entities,
//!    and an interrogative-turn penalty for wh-questions.
//!
//! The weights are the 5-fold cross-validated winners from that study:
//! `[w_base, w_group, w_ctx, w_spk, w_q] = [0.5, 0.5, 1.5, 0.8, 0.5]`.
//! Neighbor-context overlap is the strongest signal; the group score gets
//! half weight (session concentration alone barely helps); the base turn
//! score is down-weighted once the right sessions are selected.
//!
//! The rerank fires only for follow-up queries inside a session
//! ([`crate::session_prepare::is_follow_up`]); all other queries keep the
//! existing ranking untouched.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::index::{ScoreBreakdown, SearchResult};

/// Tuned stage-2 weights: `[w_base, w_group, w_ctx, w_spk, w_q]`.
pub const RERANK_WEIGHTS: [f32; 5] = [0.5, 0.5, 1.5, 0.8, 0.5];

/// Retrieval depth feeding stage 1.
pub const RERANK_DEEP_TOP_K: usize = 200;

/// Number of top groups rescored in stage 2.
pub const RERANK_TOP_GROUPS: usize = 10;

/// Neighbor window (in group order) for the context-overlap feature.
pub const RERANK_NEIGHBOR_WINDOW: usize = 2;

/// A document as the reranker sees it.
#[derive(Debug, Clone)]
pub struct RerankDocView {
    /// Full indexed text.
    pub text: String,
    /// RFC3339 timestamp when present; used to order turns in a group.
    pub timestamp: Option<String>,
    /// Turn speaker (the indexed author/role).
    pub speaker: Option<String>,
}

/// Source of document text for the reranker: the live store or a published
/// snapshot, both of which can list a group's member documents.
///
/// `group_member_ids` must honor the same `filters` the deep search ran
/// with (user/provider isolation): stage-2 expansion must never surface a
/// document the filtered search could not have returned.
pub trait RerankDocSource {
    fn rerank_doc(&self, doc_id: &str) -> Option<RerankDocView>;
    fn group_member_ids(&self, group_id: &str, filters: &BTreeMap<String, String>) -> Vec<String>;
}

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

/// Strip the indexer's structured "{role}: " prefix so turn-level heuristics
/// see the message text, not the attribution wrapper.
fn strip_role_prefix<'a>(text: &'a str, speaker: Option<&str>) -> &'a str {
    if let Some(speaker) = speaker {
        let prefix = format!("{speaker}:");
        if let Some(rest) = text.strip_prefix(&prefix) {
            return rest.trim_start();
        }
    }
    text
}

/// Heuristic turn-role check: interrogative turns rarely contain answers.
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

/// True when the query opens with a wh-word.
pub fn is_wh_question(query: &str) -> bool {
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

/// Two-stage conversational rerank of deep-search candidates.
///
/// `query` is the raw follow-up text, `entities` the session's resolved
/// entities (the "who is this about" proxy the speaker feature matches
/// against), and `filters` the same filters the deep search ran with.
/// Returns the rescored candidates, best first; callers truncate to their
/// `top_k`. Emitted scores are the combined rerank scores.
pub fn conversational_rerank<S: RerankDocSource>(
    candidates: Vec<SearchResult>,
    source: &S,
    query: &str,
    entities: &[String],
    filters: &BTreeMap<String, String>,
    weights: [f32; 5],
) -> Vec<SearchResult> {
    // Stage 1: rank groups by their best candidate turn score.
    let mut group_max: HashMap<String, f32> = HashMap::new();
    let mut base_of: HashMap<String, f32> = HashMap::new();
    let mut result_of: HashMap<String, SearchResult> = HashMap::new();
    for r in candidates {
        // Documents without a group form singleton groups keyed by doc id so
        // they stay eligible without gaining a group-concentration boost.
        let group = r.group_id.clone().unwrap_or_else(|| r.doc_id.clone());
        base_of.insert(r.doc_id.clone(), r.score);
        group_max
            .entry(group)
            .and_modify(|m| *m = (*m).max(r.score))
            .or_insert(r.score);
        result_of.insert(r.doc_id.clone(), r);
    }
    let mut top_groups: Vec<String> = group_max.keys().cloned().collect();
    top_groups.sort_by(|a, b| {
        group_max[b]
            .partial_cmp(&group_max[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(b))
    });
    top_groups.truncate(RERANK_TOP_GROUPS);

    let mut qterms = tokenize_simple(query);
    for entity in entities {
        qterms.extend(tokenize_simple(entity));
    }
    let entity_names: HashSet<String> = entities.iter().map(|e| e.to_lowercase()).collect();
    let query_is_wh = is_wh_question(query);

    struct Raw {
        doc_id: String,
        group: String,
        base: f32,
        group_score: f32,
        ctx: f32,
        spk: f32,
        q: f32,
    }
    let mut raw: Vec<Raw> = Vec::new();
    for group in &top_groups {
        let mut members = source.group_member_ids(group, filters);
        // Order turns in the group by timestamp, falling back to doc id for
        // documents without one (deterministic, if arbitrary).
        members.sort_by_cached_key(|id| {
            let view = source.rerank_doc(id);
            (
                view.as_ref()
                    .and_then(|v| v.timestamp.clone())
                    .unwrap_or_default(),
                id.clone(),
            )
        });
        let positions: HashMap<&str, usize> = members
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect();
        let texts: Vec<Option<String>> = members
            .iter()
            .map(|id| source.rerank_doc(id).map(|v| v.text))
            .collect();
        for id in &members {
            let view = source.rerank_doc(id);
            let text = view.as_ref().map(|v| v.text.as_str()).unwrap_or("");
            let pos = positions[id.as_str()];
            let lo = pos.saturating_sub(RERANK_NEIGHBOR_WINDOW);
            let hi = (pos + RERANK_NEIGHBOR_WINDOW).min(members.len().saturating_sub(1));
            let mut window = String::new();
            for t in texts[lo..=hi].iter().flatten() {
                window.push_str(t);
                window.push(' ');
            }
            let wtoks = tokenize_simple(&window);
            let inter = qterms.intersection(&wtoks).count() as f32;
            let ctx = if qterms.is_empty() {
                0.0
            } else {
                inter / qterms.len() as f32
            };
            let speaker = view.as_ref().and_then(|v| v.speaker.clone());
            let bare_text = strip_role_prefix(text, speaker.as_deref());
            raw.push(Raw {
                doc_id: id.clone(),
                group: group.clone(),
                base: base_of.get(id).copied().unwrap_or(0.0),
                group_score: group_max[group],
                ctx,
                spk: match speaker {
                    Some(s)
                        if !entity_names.is_empty()
                            && entity_names.contains(&s.trim().to_lowercase()) =>
                    {
                        1.0
                    }
                    _ => 0.0,
                },
                q: if is_interrogative(bare_text) {
                    1.0
                } else {
                    0.0
                },
            });
        }
    }
    if raw.is_empty() {
        return Vec::new();
    }

    let (mut bmin, mut bmax) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut gmin, mut gmax) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut cmin, mut cmax) = (f32::INFINITY, f32::NEG_INFINITY);
    for c in &raw {
        bmin = bmin.min(c.base);
        bmax = bmax.max(c.base);
        gmin = gmin.min(c.group_score);
        gmax = gmax.max(c.group_score);
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
    let [w_base, w_group, w_ctx, w_spk, w_q] = weights;
    let mut scored: Vec<(String, f32)> = raw
        .iter()
        .map(|c| {
            let score = w_base * norm(c.base, bmin, bmax)
                + w_group * norm(c.group_score, gmin, gmax)
                + w_ctx * norm(c.ctx, cmin, cmax)
                + w_spk * c.spk
                - if query_is_wh { w_q * c.q } else { 0.0 };
            (c.doc_id.clone(), score)
        })
        .collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    scored
        .into_iter()
        .map(|(doc_id, score)| {
            let mut result = result_of.remove(&doc_id).unwrap_or_else(|| {
                // A group member the deep search never returned: synthesize
                // a minimal result so it can still surface on context alone.
                let group = raw
                    .iter()
                    .find(|c| c.doc_id == doc_id)
                    .map(|c| c.group.clone());
                SearchResult {
                    doc_id: doc_id.clone(),
                    source: String::new(),
                    group_id: group,
                    score: 0.0,
                    score_breakdown: ScoreBreakdown::default(),
                    matched_entities: Vec::new(),
                    matched_terms: Vec::new(),
                    probable_topic: None,
                    doc_type_guess: None,
                    semantic_status: None,
                    superseded_by: None,
                    relation_confidence: None,
                    relation_evidence: Vec::new(),
                }
            });
            result.score = score;
            result
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MapSource {
        docs: HashMap<String, (String, Option<String>, Option<String>, String)>,
    }

    impl RerankDocSource for MapSource {
        fn rerank_doc(&self, doc_id: &str) -> Option<RerankDocView> {
            self.docs
                .get(doc_id)
                .map(|(text, ts, spk, _)| RerankDocView {
                    text: text.clone(),
                    timestamp: ts.clone(),
                    speaker: spk.clone(),
                })
        }

        fn group_member_ids(
            &self,
            group_id: &str,
            _filters: &BTreeMap<String, String>,
        ) -> Vec<String> {
            self.docs
                .iter()
                .filter(|(_, (_, _, _, g))| g == group_id)
                .map(|(id, _)| id.clone())
                .collect()
        }
    }

    fn search_result(doc_id: &str, group: &str, score: f32) -> SearchResult {
        SearchResult {
            doc_id: doc_id.to_string(),
            source: "test".to_string(),
            group_id: Some(group.to_string()),
            score,
            score_breakdown: ScoreBreakdown::default(),
            matched_entities: Vec::new(),
            matched_terms: Vec::new(),
            probable_topic: None,
            doc_type_guess: None,
            semantic_status: None,
            superseded_by: None,
            relation_confidence: None,
            relation_evidence: Vec::new(),
        }
    }

    fn fixture() -> MapSource {
        // Two groups; the gold turn (g1d3) is NOT in the deep candidates but
        // sits between turns that share query terms, and its speaker matches
        // the session entity.
        let mut docs = HashMap::new();
        let add = |docs: &mut HashMap<String, (String, Option<String>, Option<String>, String)>,
                   id: &str,
                   text: &str,
                   speaker: &str,
                   group: &str,
                   ts: usize| {
            docs.insert(
                id.to_string(),
                (
                    text.to_string(),
                    Some(format!("2026-09-22T00:00:{ts:02}Z")),
                    Some(speaker.to_string()),
                    group.to_string(),
                ),
            );
        };
        add(
            &mut docs,
            "g1d1",
            "user: when did the launch happen",
            "user",
            "g1",
            1,
        );
        add(
            &mut docs,
            "g1d2",
            "assistant: the sunrise painting launch was lovely",
            "assistant",
            "g1",
            2,
        );
        add(
            &mut docs,
            "g1d3",
            "Melanie: I painted the sunrise in March",
            "Melanie",
            "g1",
            3,
        );
        add(
            &mut docs,
            "g1d4",
            "user: what about the budget",
            "user",
            "g1",
            4,
        );
        add(&mut docs, "g2d1", "user: sunrise photos", "user", "g2", 1);
        add(
            &mut docs,
            "g2d2",
            "assistant: here they are",
            "assistant",
            "g2",
            2,
        );
        MapSource { docs }
    }

    #[test]
    fn promotes_context_match_missing_from_deep_search() {
        let source = fixture();
        // Deep search found g2 (higher base) and g1d2, but not the gold g1d3.
        let candidates = vec![
            search_result("g2d1", "g2", 0.9),
            search_result("g2d2", "g2", 0.8),
            search_result("g1d2", "g1", 0.5),
        ];
        let ranked = conversational_rerank(
            candidates,
            &source,
            "when did Melanie paint a sunrise",
            &["Melanie".to_string()],
            &BTreeMap::new(),
            RERANK_WEIGHTS,
        );
        assert!(!ranked.is_empty());
        // The gold turn wins on neighbor-context overlap + speaker match
        // despite a zero base score.
        assert_eq!(ranked[0].doc_id, "g1d3");
    }

    #[test]
    fn wh_penalty_demotes_interrogative_turns() {
        let source = fixture();
        let candidates = vec![
            search_result("g1d1", "g1", 0.7),
            search_result("g1d3", "g1", 0.6),
        ];
        let ranked = conversational_rerank(
            candidates,
            &source,
            "when did Melanie paint it",
            &["Melanie".to_string()],
            &BTreeMap::new(),
            RERANK_WEIGHTS,
        );
        let pos_gold = ranked.iter().position(|r| r.doc_id == "g1d3").unwrap();
        let pos_q = ranked.iter().position(|r| r.doc_id == "g1d1").unwrap();
        assert!(pos_gold < pos_q);
    }

    #[test]
    fn empty_candidates_yield_empty_ranking() {
        let source = fixture();
        let ranked = conversational_rerank(
            vec![],
            &source,
            "when",
            &[],
            &BTreeMap::new(),
            RERANK_WEIGHTS,
        );
        assert!(ranked.is_empty());
    }

    #[test]
    fn detects_wh_questions() {
        assert!(is_wh_question("when did they paint it?"));
        assert!(is_wh_question("What about the budget"));
        assert!(!is_wh_question("tell me more"));
        assert!(!is_wh_question(""));
    }

    #[test]
    fn strips_role_prefix_for_interrogative_check() {
        assert_eq!(
            strip_role_prefix("user: when did it happen", Some("user")),
            "when did it happen"
        );
        assert_eq!(
            strip_role_prefix("assistant: sure thing", Some("assistant")),
            "sure thing"
        );
        // No prefix match: text unchanged.
        assert_eq!(
            strip_role_prefix("when did it happen", Some("user")),
            "when did it happen"
        );
        assert_eq!(
            strip_role_prefix("user: when did it happen", None),
            "user: when did it happen"
        );
        // The bare text recovers the bin's interrogative signal.
        assert!(is_interrogative(strip_role_prefix(
            "user: when did the launch happen",
            Some("user")
        )));
        assert!(!is_interrogative(strip_role_prefix(
            "assistant: the launch was lovely",
            Some("assistant")
        )));
    }
}
