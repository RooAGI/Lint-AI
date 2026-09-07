use anyhow::{Context, Result};
use clap::Parser;
use lint_ai::query_plan::PreparedQuery;
use lint_ai::{
    build_index_store, ChunkStrategy, PipelineOptions, SearchResult, SourceDocument,
    Tier1NerProvider, Tier1TermRankerKind,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Parser)]
#[command(name = "haystack-indexstore-benchmark")]
#[command(about = "Run LongMemEval-S through Lint-AI's normal IndexStore query path")]
struct Args {
    /// Path to the cleaned LongMemEval-S dataset.
    #[arg(long)]
    longmemeval: PathBuf,

    /// Top-K values to evaluate. Repeat the flag to add multiple K values.
    #[arg(long = "k", default_values_t = vec![1usize, 3, 5, 10])]
    ks: Vec<usize>,

    /// Limit the number of queries to evaluate.
    #[arg(long)]
    limit: Option<usize>,

    /// Only evaluate one question type, for example `knowledge-update`.
    #[arg(long)]
    question_type: Option<String>,

    /// Optional output path for JSON results.
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
struct LongMemEvalEntry {
    question_id: String,
    question_type: String,
    question: String,
    question_date: String,
    #[serde(default)]
    answer_session_ids: Vec<String>,
    #[serde(default)]
    haystack_session_ids: Vec<String>,
    #[serde(default)]
    haystack_dates: Vec<String>,
    #[serde(default)]
    haystack_sessions: Vec<Vec<LongMemEvalTurn>>,
}

#[derive(Debug, Clone, Deserialize)]
struct LongMemEvalTurn {
    role: String,
    content: String,
    #[serde(default)]
    _has_answer: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
struct QueryMetrics {
    id: String,
    query: String,
    question_type: String,
    question_date: String,
    candidate_session_ids: Vec<String>,
    relevant_session_ids: Vec<String>,
    retrieved_session_ids: Vec<String>,
    recall_at_k: HashMap<usize, f64>,
    recall_any_at_k: HashMap<usize, f64>,
    mrr: f64,
    ndcg_at_10: f64,
    query_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct AggregateMetrics {
    query_count: usize,
    recall_at_k: HashMap<usize, f64>,
    recall_any_at_k: HashMap<usize, f64>,
    mrr: f64,
    ndcg_at_10: f64,
    average_query_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct TypeMetrics {
    query_count: usize,
    recall_at_k: HashMap<usize, f64>,
    recall_any_at_k: HashMap<usize, f64>,
    mrr: f64,
    ndcg_at_10: f64,
    average_query_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkReport {
    query_path: &'static str,
    aggregate: AggregateMetrics,
    by_question_type: BTreeMap<String, TypeMetrics>,
    per_query: Vec<QueryMetrics>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut ks = args.ks.into_iter().filter(|k| *k > 0).collect::<Vec<_>>();
    ks.sort_unstable();
    ks.dedup();
    if ks.is_empty() {
        anyhow::bail!("at least one positive --k value is required");
    }

    let data = fs::read_to_string(&args.longmemeval)
        .with_context(|| format!("failed to read {}", args.longmemeval.display()))?;
    let raw: Vec<LongMemEvalEntry> =
        serde_json::from_str(&data).context("failed to parse LongMemEval JSON")?;

    let report = run_benchmark(raw, args.limit, args.question_type.as_deref(), &ks)?;
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(out) = args.out {
        fs::write(&out, &json).with_context(|| format!("failed to write {}", out.display()))?;
        println!("wrote benchmark report to {}", out.display());
    } else {
        println!("{json}");
    }
    Ok(())
}

fn run_benchmark(
    raw: Vec<LongMemEvalEntry>,
    limit: Option<usize>,
    question_type: Option<&str>,
    ks: &[usize],
) -> Result<BenchmarkReport> {
    let abstention_types = HashSet::from([
        "single-session-user_abs".to_string(),
        "multi-session_abs".to_string(),
        "knowledge-update_abs".to_string(),
        "temporal-reasoning_abs".to_string(),
    ]);
    let entries = raw
        .into_iter()
        .filter(|entry| !abstention_types.contains(&entry.question_type))
        .filter(|entry| question_type.is_none_or(|wanted| entry.question_type == wanted))
        .take(limit.unwrap_or(usize::MAX))
        .collect::<Vec<_>>();

    eprintln!(
        "running {} LongMemEval-S questions through IndexStore::query_prepared...",
        entries.len()
    );

    let max_k = ks.iter().copied().max().unwrap_or(10).max(10);
    let total = entries.len();
    let mut per_query = Vec::with_capacity(total);

    for (idx, entry) in entries.into_iter().enumerate() {
        let candidate_session_ids = entry.haystack_session_ids.clone();
        let source_docs = build_scoped_source_docs(&entry);
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

        // Deliberately use the same public query path as a normal caller. We do not
        // inject benchmark-specific temporal context or call MemoryIndex directly.
        let mut store = build_index_store(&source_docs, &options)?;
        let question_anchor = normalize_longmemeval_date(&entry.question_date);
        let prepared = PreparedQuery::new_at(&entry.question, &question_anchor);
        let query_start = Instant::now();
        let results = store.query_prepared(&prepared, max_k, &BTreeMap::new())?;
        let query_ms = query_start.elapsed().as_secs_f64() * 1000.0;

        let retrieved_session_ids = dedupe_preserve_order(
            results
                .iter()
                .map(|result| evaluation_group_id(result))
                .collect(),
        );
        let relevant = candidate_session_ids
            .iter()
            .filter(|session_id| entry.answer_session_ids.contains(session_id))
            .cloned()
            .collect::<HashSet<_>>();
        let mut relevant_session_ids = relevant.iter().cloned().collect::<Vec<_>>();
        relevant_session_ids.sort();

        let mut recall_at_k = HashMap::new();
        let mut recall_any_at_k = HashMap::new();
        for k in ks {
            recall_at_k.insert(*k, recall_at_k_fn(&retrieved_session_ids, &relevant, *k));
            recall_any_at_k.insert(
                *k,
                recall_any_at_k_fn(&retrieved_session_ids, &relevant, *k),
            );
        }

        per_query.push(QueryMetrics {
            id: entry.question_id,
            query: entry.question,
            question_type: entry.question_type,
            question_date: entry.question_date,
            candidate_session_ids,
            relevant_session_ids,
            retrieved_session_ids,
            recall_at_k,
            recall_any_at_k,
            mrr: reciprocal_rank(per_query_retrieved_placeholder(), &HashSet::new()),
            ndcg_at_10: 0.0,
            query_ms,
        });

        // Compute rank metrics after insertion without cloning the whole result record.
        let last = per_query.last_mut().expect("query metrics should exist");
        let relevant = last
            .relevant_session_ids
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        last.mrr = reciprocal_rank(&last.retrieved_session_ids, &relevant);
        last.ndcg_at_10 = ndcg_at_k(&last.retrieved_session_ids, &relevant, 10);

        eprintln!(
            "[{}/{}] {} type={} query={:.2}ms retrieved_sessions={}",
            idx + 1,
            total,
            last.id,
            last.question_type,
            last.query_ms,
            last.retrieved_session_ids.len()
        );
    }

    Ok(BenchmarkReport {
        query_path: "IndexStore::query_prepared",
        aggregate: aggregate_metrics(&per_query, ks),
        by_question_type: aggregate_by_question_type(&per_query, ks),
        per_query,
    })
}

// Used only to initialize fields before they are filled from the just-inserted query.
fn per_query_retrieved_placeholder() -> &'static [String] {
    &[]
}

fn build_scoped_source_docs(entry: &LongMemEvalEntry) -> Vec<SourceDocument> {
    let mut docs = Vec::new();
    for (idx, (session_id, turns)) in entry
        .haystack_session_ids
        .iter()
        .cloned()
        .zip(entry.haystack_sessions.iter())
        .enumerate()
    {
        let session_date = entry
            .haystack_dates
            .get(idx)
            .cloned()
            .unwrap_or_else(|| entry.question_date.clone());
        for (turn_idx, turn) in turns.iter().enumerate() {
            docs.push(SourceDocument {
                doc_id: format!("{session_id}::turn{turn_idx}"),
                source: format!("longmemeval/session/{session_id}/turn/{turn_idx}"),
                content: format!("{}: {}", turn.role, turn.content),
                concept: "longmemeval-turn".to_string(),
                group_id: Some(session_id.clone()),
                filters: BTreeMap::new(),
                headings: vec![format!("session:{session_id}")],
                links: vec![],
                timestamp: Some(normalize_longmemeval_date(&session_date)),
                doc_length: turn.content.len(),
                author_agent: None,
            });
        }
    }
    docs
}

fn normalize_longmemeval_date(input: &str) -> String {
    let bytes = input.as_bytes();
    if bytes.len() >= 10 && bytes[4] == b'/' && bytes[7] == b'/' {
        return format!("{}-{}-{}", &input[0..4], &input[5..7], &input[8..10]);
    }
    input.to_string()
}

fn evaluation_group_id(result: &SearchResult) -> String {
    result.group_id.clone().unwrap_or_else(|| {
        result
            .doc_id
            .split("::turn")
            .next()
            .unwrap_or(&result.doc_id)
            .to_string()
    })
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
    let hits = retrieved
        .iter()
        .take(k)
        .filter(|session_id| relevant.contains(*session_id))
        .count();
    hits as f64 / relevant.len() as f64
}

fn recall_any_at_k_fn(retrieved: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if retrieved
        .iter()
        .take(k)
        .any(|session_id| relevant.contains(session_id))
    {
        1.0
    } else {
        0.0
    }
}

fn reciprocal_rank(retrieved: &[String], relevant: &HashSet<String>) -> f64 {
    retrieved
        .iter()
        .position(|session_id| relevant.contains(session_id))
        .map(|idx| 1.0 / (idx as f64 + 1.0))
        .unwrap_or(0.0)
}

fn ndcg_at_k(retrieved: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    let dcg = retrieved
        .iter()
        .take(k)
        .enumerate()
        .filter(|(_, session_id)| relevant.contains(*session_id))
        .map(|(idx, _)| 1.0 / ((idx as f64 + 2.0).log2()))
        .sum::<f64>();
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

fn aggregate_metrics(per_query: &[QueryMetrics], ks: &[usize]) -> AggregateMetrics {
    AggregateMetrics {
        query_count: per_query.len(),
        recall_at_k: average_k_metric(per_query, ks, |q| &q.recall_at_k),
        recall_any_at_k: average_k_metric(per_query, ks, |q| &q.recall_any_at_k),
        mrr: average(per_query.iter().map(|q| q.mrr)),
        ndcg_at_10: average(per_query.iter().map(|q| q.ndcg_at_10)),
        average_query_ms: average(per_query.iter().map(|q| q.query_ms)),
    }
}

fn aggregate_by_question_type(
    per_query: &[QueryMetrics],
    ks: &[usize],
) -> BTreeMap<String, TypeMetrics> {
    let mut grouped: BTreeMap<String, Vec<&QueryMetrics>> = BTreeMap::new();
    for query in per_query {
        grouped
            .entry(query.question_type.clone())
            .or_default()
            .push(query);
    }
    grouped
        .into_iter()
        .map(|(question_type, queries)| {
            let recall_at_k = average_k_metric_refs(&queries, ks, |q| &q.recall_at_k);
            let recall_any_at_k = average_k_metric_refs(&queries, ks, |q| &q.recall_any_at_k);
            let metrics = TypeMetrics {
                query_count: queries.len(),
                recall_at_k,
                recall_any_at_k,
                mrr: average(queries.iter().map(|q| q.mrr)),
                ndcg_at_10: average(queries.iter().map(|q| q.ndcg_at_10)),
                average_query_ms: average(queries.iter().map(|q| q.query_ms)),
            };
            (question_type, metrics)
        })
        .collect()
}

fn average_k_metric<F>(queries: &[QueryMetrics], ks: &[usize], metric: F) -> HashMap<usize, f64>
where
    F: Fn(&QueryMetrics) -> &HashMap<usize, f64>,
{
    ks.iter()
        .map(|k| {
            let value = average(queries.iter().filter_map(|q| metric(q).get(k).copied()));
            (*k, value)
        })
        .collect()
}

fn average_k_metric_refs<F>(
    queries: &[&QueryMetrics],
    ks: &[usize],
    metric: F,
) -> HashMap<usize, f64>
where
    F: Fn(&QueryMetrics) -> &HashMap<usize, f64>,
{
    ks.iter()
        .map(|k| {
            let value = average(queries.iter().filter_map(|q| metric(q).get(k).copied()));
            (*k, value)
        })
        .collect()
}

fn average(values: impl Iterator<Item = f64>) -> f64 {
    let values = values.collect::<Vec<_>>();
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}
