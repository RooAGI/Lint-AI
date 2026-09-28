//! LoCoMo retrieval benchmark adapter for Lint-AI.
//!
//! Ingests each LoCoMo conversation (snap-research/locomo, `locomo10.json`)
//! as turn-level documents grouped by session, builds one query snapshot per
//! conversation, and scores per-question retrieval against the evidence
//! session pointers (`D{session}:{turn}`). Metrics mirror the
//! haystack-scoped LongMemEval benchmark: fractional and any-hit recall@k,
//! MRR, NDCG@10, reported overall and per question category.
//!
//! Category labels follow the upstream numbering: 1 multi-hop, 2 temporal,
//! 3 open-domain, 4 single-hop, 5 adversarial (abstention; reported
//! separately since retrieval recall is not the right lens for it).

use anyhow::{Context, Result};
use clap::Parser;
use lint_ai::{
    analyze_query, build_query_snapshot_from_source_documents, ChunkStrategy, PipelineOptions,
    SourceDocument, TemporalQueryContext, Tier1NerProvider, Tier1TermRankerKind,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Parser)]
#[command(name = "locomo-benchmark")]
#[command(about = "Run the LoCoMo retrieval benchmark against Lint-AI")]
struct Args {
    /// Path to locomo10.json from snap-research/locomo.
    #[arg(long)]
    locomo: PathBuf,

    /// Top-K values to evaluate. Repeat the flag to add multiple K values.
    #[arg(long = "k", default_values_t = vec![1usize, 3, 5, 10, 20])]
    ks: Vec<usize>,

    /// Only process the first N conversations (smoke testing).
    #[arg(long)]
    limit_conversations: Option<usize>,

    /// Optional output path for JSON results.
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct LocomoConversation {
    sample_id: String,
    qa: Vec<LocomoQa>,
    conversation: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Clone, Serialize)]
struct QueryMetrics {
    id: String,
    conversation: String,
    question: String,
    category: u8,
    category_label: String,
    retrieved_session_ids: Vec<String>,
    relevant_session_ids: Vec<String>,
    recall_at_k: HashMap<usize, f64>,
    recall_any_at_k: HashMap<usize, f64>,
    mrr: f64,
    ndcg_at_10: f64,
    analysis_ms: f64,
    total_ms: f64,
}

#[derive(Debug, Serialize, Default)]
struct Aggregate {
    question_count: usize,
    recall_at_k: HashMap<usize, f64>,
    recall_any_at_k: HashMap<usize, f64>,
    mrr: f64,
    ndcg_at_10: f64,
    mean_latency_ms: f64,
    p50_latency_ms: f64,
    p95_latency_ms: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    dataset: String,
    conversations: usize,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    by_category: HashMap<String, Aggregate>,
    aggregate: Aggregate,
    aggregate_no_adversarial: Aggregate,
    per_query: Vec<QueryMetrics>,
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

/// Parse `D{session}:{turn}` pointers out of evidence strings, returning the
/// session numbers. Tolerates odd entries (e.g. `D8:6; D9:17`, `D:11:26`).
fn parse_evidence_sessions(evidence: &[String]) -> HashSet<usize> {
    let mut sessions = HashSet::new();
    for entry in evidence {
        for part in entry.split(';') {
            let part = part.trim();
            let after_d = match part.strip_prefix('D') {
                Some(rest) => rest,
                None => continue,
            };
            let mut pieces = after_d.split(':');
            // Session number is the first non-empty numeric piece.
            for piece in pieces.by_ref() {
                let piece = piece.trim();
                if piece.is_empty() {
                    continue;
                }
                if let Ok(n) = piece.parse::<usize>() {
                    sessions.insert(n);
                }
                break;
            }
        }
    }
    sessions
}

/// Session keys look like `session_3`; date keys like `session_3_date_time`.
fn session_number(key: &str) -> Option<usize> {
    let rest = key.strip_prefix("session_")?;
    if rest.contains('_') {
        return None;
    }
    rest.parse::<usize>().ok()
}

fn build_conversation_docs(conv: &LocomoConversation) -> Result<Vec<SourceDocument>> {
    let mut session_nums: Vec<usize> = conv
        .conversation
        .keys()
        .filter_map(|k| session_number(k))
        .collect();
    session_nums.sort_unstable();

    let mut docs = Vec::new();
    for n in session_nums {
        let key = format!("session_{n}");
        let date_key = format!("session_{n}_date_time");
        let date = conv
            .conversation
            .get(&date_key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let turns: Vec<LocomoTurn> = serde_json::from_value(
            conv.conversation
                .get(&key)
                .cloned()
                .unwrap_or(serde_json::Value::Array(vec![])),
        )
        .with_context(|| format!("failed to parse turns for {key}"))?;
        let group_id = format!("{}::session_{n}", conv.sample_id);
        for (turn_idx, turn) in turns.iter().enumerate() {
            docs.push(SourceDocument {
                doc_id: format!("{group_id}::turn{turn_idx}"),
                source: format!("locomo/{}/session/{n}/turn/{turn_idx}", conv.sample_id),
                content: format!("{}: {}", turn.speaker, turn.text),
                concept: "locomo-turn".to_string(),
                group_id: Some(group_id.clone()),
                filters: BTreeMap::new(),
                headings: vec![format!("session:{group_id}")],
                links: vec![],
                timestamp: date.clone(),
                doc_length: turn.text.len(),
                author_agent: None,
                key_phrases: Vec::new(),
                key_phrase_extraction_hash: String::new(),
            });
        }
    }
    Ok(docs)
}

fn evaluation_group_id(doc_id: &str) -> String {
    doc_id.split("::turn").next().unwrap_or(doc_id).to_string()
}

fn dedupe_preserve_order(items: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for item in items {
        if seen.insert(item.clone()) {
            out.push(item);
        }
    }
    out
}

fn recall_at_k_fn(retrieved: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let limit = k.min(retrieved.len());
    let hits = retrieved
        .iter()
        .take(limit)
        .filter(|id| relevant.contains(*id))
        .count();
    hits as f64 / relevant.len() as f64
}

fn recall_any_at_k_fn(retrieved: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let limit = k.min(retrieved.len());
    if retrieved.iter().take(limit).any(|id| relevant.contains(id)) {
        1.0
    } else {
        0.0
    }
}

fn reciprocal_rank(retrieved: &[String], relevant: &HashSet<String>) -> f64 {
    for (idx, id) in retrieved.iter().enumerate() {
        if relevant.contains(id) {
            return 1.0 / (idx as f64 + 1.0);
        }
    }
    0.0
}

fn ndcg_at_k(retrieved: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    let mut dcg = 0.0;
    for (idx, id) in retrieved.iter().take(k).enumerate() {
        if relevant.contains(id) {
            dcg += 1.0 / ((idx as f64 + 2.0).log2());
        }
    }
    let ideal = relevant.len().min(k);
    let idcg = (0..ideal)
        .map(|idx| 1.0 / ((idx as f64 + 2.0).log2()))
        .sum::<f64>();
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p / 100.0 * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn aggregate_metrics(queries: &[QueryMetrics], ks: &[usize]) -> Aggregate {
    let n = queries.len() as f64;
    let mut agg = Aggregate {
        question_count: queries.len(),
        ..Default::default()
    };
    if queries.is_empty() {
        return agg;
    }
    for k in ks {
        agg.recall_at_k.insert(
            *k,
            queries.iter().map(|q| q.recall_at_k[k]).sum::<f64>() / n,
        );
        agg.recall_any_at_k.insert(
            *k,
            queries.iter().map(|q| q.recall_any_at_k[k]).sum::<f64>() / n,
        );
    }
    agg.mrr = queries.iter().map(|q| q.mrr).sum::<f64>() / n;
    agg.ndcg_at_10 = queries.iter().map(|q| q.ndcg_at_10).sum::<f64>() / n;
    let mut lat: Vec<f64> = queries.iter().map(|q| q.total_ms).collect();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    agg.mean_latency_ms = lat.iter().sum::<f64>() / n;
    agg.p50_latency_ms = percentile(&lat, 50.0);
    agg.p95_latency_ms = percentile(&lat, 95.0);
    agg
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut ks = args
        .ks
        .into_iter()
        .filter(|k| *k > 0)
        .collect::<Vec<usize>>();
    ks.sort_unstable();
    ks.dedup();
    if ks.is_empty() {
        anyhow::bail!("at least one positive --k value is required");
    }
    let max_k = ks.iter().copied().max().unwrap_or(10).max(10);

    eprintln!("loading LoCoMo data...");
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

    let mut per_query = Vec::new();
    for conv in &conversations {
        let docs = build_conversation_docs(conv)?;
        let index = build_query_snapshot_from_source_documents(
            &docs,
            &options.ner_provider,
            &options.spacy_model,
            &options.term_ranker,
            &options.chunk_strategy,
            options.chunk_lines,
            options.chunk_overlap,
            options.chunk_target_tokens,
            options.chunk_max_tokens,
            options.text_rerank_ngram,
            options.text_rerank_lcs,
        )?;

        for (q_idx, q) in conv.qa.iter().enumerate() {
            let analysis_start = Instant::now();
            let analysis = analyze_query(&q.question);
            let analysis_ms = analysis_start.elapsed().as_secs_f64() * 1000.0;
            let temporal = TemporalQueryContext {
                query_routing_intent: analysis.query_routing_intent,
                ..Default::default()
            };
            let (results, timings, _diagnostics) =
                index.query_with_temporal_context(&analysis.augmented_query, max_k, temporal);
            let retrieved: Vec<String> = dedupe_preserve_order(
                results
                    .into_iter()
                    .map(|r| evaluation_group_id(&r.doc_id))
                    .collect(),
            );
            let relevant: HashSet<String> = parse_evidence_sessions(&q.evidence)
                .into_iter()
                .map(|n| format!("{}::session_{n}", conv.sample_id))
                .collect();

            let mut recall_at_k = HashMap::new();
            let mut recall_any_at_k = HashMap::new();
            for k in &ks {
                recall_at_k.insert(*k, recall_at_k_fn(&retrieved, &relevant, *k));
                recall_any_at_k.insert(*k, recall_any_at_k_fn(&retrieved, &relevant, *k));
            }
            let mut relevant_sorted: Vec<String> = relevant.into_iter().collect();
            relevant_sorted.sort();
            per_query.push(QueryMetrics {
                id: format!("{}_q{q_idx}", conv.sample_id),
                conversation: conv.sample_id.clone(),
                question: q.question.clone(),
                category: q.category,
                category_label: category_label(q.category).to_string(),
                retrieved_session_ids: retrieved,
                relevant_session_ids: relevant_sorted,
                recall_at_k,
                recall_any_at_k,
                mrr: 0.0,        // filled below
                ndcg_at_10: 0.0, // filled below
                analysis_ms,
                total_ms: timings.total_ms,
            });
            // Fill mrr/ndcg now that the struct owns the retrieved list.
            let last = per_query.last_mut().expect("just pushed");
            let rel: HashSet<String> = last.relevant_session_ids.iter().cloned().collect();
            last.mrr = reciprocal_rank(&last.retrieved_session_ids, &rel);
            last.ndcg_at_10 = ndcg_at_k(&last.retrieved_session_ids, &rel, 10);
        }
        eprintln!(
            "conversation {} done ({} questions)",
            conv.sample_id,
            conv.qa.len()
        );
    }

    let by_category: HashMap<String, Aggregate> = {
        let mut map: HashMap<String, Vec<QueryMetrics>> = HashMap::new();
        for q in per_query.drain(..) {
            map.entry(q.category_label.clone()).or_default().push(q);
        }
        let mut out = HashMap::new();
        let mut drained = Vec::new();
        for (label, qs) in &map {
            out.insert(label.clone(), aggregate_metrics(qs, &ks));
        }
        for (_, mut qs) in map {
            drained.append(&mut qs);
        }
        // restore per_query order: sort by id for determinism
        drained.sort_by(|a, b| a.id.cmp(&b.id));
        per_query = drained;
        out
    };

    let report = Report {
        dataset: "LoCoMo (snap-research/locomo locomo10.json)".to_string(),
        conversations: conversations.len(),
        aggregate: aggregate_metrics(&per_query, &ks),
        aggregate_no_adversarial: aggregate_metrics(
            &per_query
                .iter()
                .filter(|q| q.category != 5)
                .cloned()
                .collect::<Vec<_>>(),
            &ks,
        ),
        by_category,
        per_query,
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(out) = args.out {
        fs::write(&out, &json)
            .with_context(|| format!("failed to write report to {}", out.display()))?;
        println!("wrote LoCoMo report to {}", out.display());
    } else {
        println!("{json}");
    }
    Ok(())
}
