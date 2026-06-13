use anyhow::{Context, Result};
use clap::{ArgAction, Parser};
use lint_ai::index::TemporalQueryHint;
use lint_ai::{
    aggregation::{build_aggregate_output, AggregateOutput},
    build_query_snapshot_from_source_documents,
    query_semantics::{analyze_query, QueryTimeHint},
    ChunkStrategy, QueryDiagnostics, QueryTimings, SearchResult, SourceDocument,
    TemporalQueryContext, Tier1NerProvider, Tier1TermRankerKind,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Parser)]
#[command(name = "haystack-scoped-benchmark")]
#[command(about = "Run a question-scoped LongMemEval haystack benchmark against Lint-AI")]
struct Args {
    /// Path to the raw LongMemEval-S dataset.
    #[arg(long)]
    longmemeval: PathBuf,

    /// Top-K values to evaluate. Repeat the flag to add multiple K values.
    #[arg(long = "k", default_values_t = vec![1usize, 3, 5, 10])]
    ks: Vec<usize>,

    /// Limit the number of queries to evaluate.
    #[arg(long)]
    limit: Option<usize>,

    /// Only evaluate one question type, for example `multi-session`.
    #[arg(long)]
    question_type: Option<String>,

    /// Optional output path for JSON results.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Enable n-gram text reranking on the top rerank window.
    #[arg(long, action = ArgAction::SetTrue)]
    text_rerank_ngram: bool,

    /// Enable LCS text reranking on the top rerank window.
    #[arg(long, action = ArgAction::SetTrue)]
    text_rerank_lcs: bool,
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
    question_type: Option<String>,
    question_date: Option<String>,
    analysis_ms: f64,
    candidate_session_ids: Vec<String>,
    retrieved_session_ids: Vec<String>,
    aggregation: Option<AggregateOutput>,
    recall_at_k: HashMap<usize, f64>,
    recall_any_at_k: HashMap<usize, f64>,
    mrr: f64,
    ndcg_at_10: f64,
    timings: QueryTimings,
    diagnostics: QueryDiagnostics,
}

#[derive(Debug, Clone, Serialize)]
struct AggregateMetrics {
    query_count: usize,
    analysis_ms: f64,
    recall_at_k: HashMap<usize, f64>,
    recall_any_at_k: HashMap<usize, f64>,
    mrr: f64,
    ndcg_at_10: f64,
    timings: QueryTimings,
}

#[derive(Debug, Clone, Serialize)]
struct TypeMetrics {
    query_count: usize,
    analysis_ms: f64,
    recall_at_k: HashMap<usize, f64>,
    recall_any_at_k: HashMap<usize, f64>,
    mrr: f64,
    ndcg_at_10: f64,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkReport {
    aggregate: AggregateMetrics,
    by_question_type: HashMap<String, TypeMetrics>,
    per_query: Vec<QueryMetrics>,
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

    eprintln!("loading raw LongMemEval data...");
    let data = fs::read_to_string(&args.longmemeval)
        .with_context(|| format!("failed to read {}", args.longmemeval.display()))?;
    let raw: Vec<LongMemEvalEntry> =
        serde_json::from_str(&data).context("failed to parse raw LongMemEval JSON")?;

    let report = run_scoped_benchmark(
        raw,
        args.limit,
        args.question_type.as_deref(),
        &ks,
        args.text_rerank_ngram,
        args.text_rerank_lcs,
    )?;
    let json = serde_json::to_string_pretty(&report)?;

    if let Some(out) = args.out {
        fs::write(&out, &json)
            .with_context(|| format!("failed to write benchmark report to {}", out.display()))?;
        println!("wrote benchmark report to {}", out.display());
    } else {
        println!("{}", json);
    }

    Ok(())
}

fn run_scoped_benchmark(
    raw: Vec<LongMemEvalEntry>,
    limit: Option<usize>,
    question_type: Option<&str>,
    ks: &[usize],
    text_rerank_ngram: bool,
    text_rerank_lcs: bool,
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

    if let Some(question_type) = question_type {
        eprintln!(
            "running {} scoped questions for question_type={}...",
            entries.len(),
            question_type
        );
    } else {
        eprintln!("running {} scoped questions...", entries.len());
    }
    let max_k = ks.iter().copied().max().unwrap_or(10).max(10);
    let mut per_query = Vec::with_capacity(entries.len());

    for (idx, entry) in entries.into_iter().enumerate() {
        let analysis_start = Instant::now();
        let analysis = analyze_query(&entry.question);
        let analysis_ms = analysis_start.elapsed().as_secs_f64() * 1000.0;
        let query_text = analysis.augmented_query.clone();
        let candidate_session_ids = entry.haystack_session_ids.clone();
        let source_docs = build_scoped_source_docs(&entry);
        let index = build_query_snapshot_from_source_documents(
            &source_docs,
            &Tier1NerProvider::Heuristic,
            "en_core_web_sm",
            &Tier1TermRankerKind::Yake,
            &ChunkStrategy::Heading,
            40,
            10,
            450,
            800,
            text_rerank_ngram,
            text_rerank_lcs,
        )?;
        let temporal = TemporalQueryContext {
            starts_from: None,
            ends_at: Some(entry.question_date.as_str()),
            window_days: 7,
            hard_filter: false,
            time_hint: analysis
                .time_hint
                .filter(|_| analysis.temporal.is_some())
                .map(|hint| match hint {
                    QueryTimeHint::Past => TemporalQueryHint::Past,
                    QueryTimeHint::Present => TemporalQueryHint::Present,
                    QueryTimeHint::Ongoing => TemporalQueryHint::Ongoing,
                    QueryTimeHint::Mixed => TemporalQueryHint::Mixed,
                }),
            allowed_doc_ids: None,
            query_routing_intent: analysis.query_routing_intent,
            has_explicit_temporal: analysis.temporal.is_some(),
        };
        let (results, timings, diagnostics) =
            index.query_with_temporal_context(&query_text, max_k, temporal);
        let aggregation = build_aggregate_output(&index, &entry.question, &results, max_k);
        assert_group_diversity(&results, 2, &entry.question_id);
        let retrieved_session_ids = results
            .into_iter()
            .map(|r| evaluation_group_id(&r.doc_id))
            .collect::<Vec<_>>();
        let retrieved_session_ids = dedupe_preserve_order(retrieved_session_ids);
        let relevant = candidate_session_ids
            .iter()
            .filter(|session_id| entry.answer_session_ids.contains(session_id))
            .cloned()
            .collect::<HashSet<_>>();

        let mut recall_at_k = HashMap::new();
        let mut recall_any_at_k = HashMap::new();
        for k in ks {
            recall_at_k.insert(*k, recall_at_k_fn(&retrieved_session_ids, &relevant, *k));
            recall_any_at_k.insert(
                *k,
                recall_any_at_k_fn(&retrieved_session_ids, &relevant, *k),
            );
        }

        let mrr = reciprocal_rank(&retrieved_session_ids, &relevant);
        let ndcg_at_10 = ndcg_at_k(&retrieved_session_ids, &relevant, 10);

        per_query.push(QueryMetrics {
            id: entry.question_id,
            query: entry.question,
            question_type: Some(entry.question_type),
            question_date: Some(entry.question_date),
            analysis_ms,
            candidate_session_ids,
            retrieved_session_ids,
            aggregation,
            recall_at_k,
            recall_any_at_k,
            mrr,
            ndcg_at_10,
            timings,
            diagnostics,
        });

        let last = per_query.last().expect("query metrics should exist");
        eprintln!(
            "[{}/{}] {} candidates={} q={} analysis={:.2}ms total={:.2}ms snapshot={:.2}ms rerank={:.2}ms sparse={:.2}ms lex_merge={:.2}ms post={:.2}ms routing={:.2}ms seq_rerank={:.2}ms evidence={:.2}ms group_build={:.2}ms group_sort={:.2}ms",
            idx + 1,
            per_query.len(),
            last.id,
            last.candidate_session_ids.len(),
            last.diagnostics.query_terms,
            last.analysis_ms,
            last.timings.total_ms,
            last.timings.snapshot_query_ms,
            last.timings.rerank_ms,
            last.timings.sparse_scoring_ms,
            last.timings.lexical_merge_ms,
            last.timings.posting_scoring_ms,
            last.timings.routing_seed_ms,
            last.timings.sequence_rerank_ms,
            last.timings.evidence_ms,
            last.timings.group_build_ms,
            last.timings.group_sort_ms,
        );
    }

    Ok(BenchmarkReport {
        aggregate: aggregate_metrics(&per_query, ks),
        by_question_type: aggregate_by_question_type(&per_query, ks),
        per_query,
    })
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
            let doc_id = format!("{}::turn{}", session_id, turn_idx);
            docs.push(SourceDocument {
                doc_id,
                source: format!("longmemeval/session/{session_id}/turn/{turn_idx}"),
                content: format!("{}: {}", turn.role, turn.content),
                concept: "longmemeval-turn".to_string(),
                group_id: Some(session_id.clone()),
                filters: std::collections::BTreeMap::new(),
                headings: vec![format!("session:{session_id}")],
                links: vec![],
                timestamp: Some(session_date.clone()),
                doc_length: turn.content.len(),
                author_agent: None,
            });
        }
    }
    docs
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
        .filter(|doc_id| relevant.contains(*doc_id))
        .count();
    hits as f64 / relevant.len() as f64
}

fn recall_any_at_k_fn(retrieved: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let limit = k.min(retrieved.len());
    if retrieved
        .iter()
        .take(limit)
        .any(|doc_id| relevant.contains(doc_id))
    {
        1.0
    } else {
        0.0
    }
}

fn reciprocal_rank(retrieved: &[String], relevant: &HashSet<String>) -> f64 {
    for (idx, doc_id) in retrieved.iter().enumerate() {
        if relevant.contains(doc_id) {
            return 1.0 / (idx as f64 + 1.0);
        }
    }
    0.0
}

fn ndcg_at_k(retrieved: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    let mut dcg = 0.0;
    for (idx, doc_id) in retrieved.iter().take(k).enumerate() {
        if relevant.contains(doc_id) {
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

fn assert_group_diversity(results: &[SearchResult], cap: usize, query_id: &str) {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for result in results {
        if let Some(group_id) = result.group_id.as_ref() {
            *counts.entry(group_id.clone()).or_default() += 1;
        }
    }
    let max_count = counts.values().copied().max().unwrap_or(0);
    eprintln!("{} group_counts={:?}", query_id, counts);
    assert!(
        max_count <= cap,
        "query {} exceeded group cap {} with counts {:?}",
        query_id,
        cap,
        counts
    );
}

fn aggregate_metrics(per_query: &[QueryMetrics], ks: &[usize]) -> AggregateMetrics {
    let n = per_query.len();
    if n == 0 {
        return AggregateMetrics {
            query_count: 0,
            analysis_ms: 0.0,
            recall_at_k: ks.iter().copied().map(|k| (k, 0.0)).collect(),
            recall_any_at_k: ks.iter().copied().map(|k| (k, 0.0)).collect(),
            mrr: 0.0,
            ndcg_at_10: 0.0,
            timings: QueryTimings::default(),
        };
    }

    let mut recall_at_k = HashMap::new();
    let mut recall_any_at_k = HashMap::new();
    for k in ks {
        let avg = per_query
            .iter()
            .map(|q| q.recall_at_k.get(k).copied().unwrap_or(0.0))
            .sum::<f64>()
            / n as f64;
        recall_at_k.insert(*k, avg);
        let avg_any = per_query
            .iter()
            .map(|q| q.recall_any_at_k.get(k).copied().unwrap_or(0.0))
            .sum::<f64>()
            / n as f64;
        recall_any_at_k.insert(*k, avg_any);
    }

    let timings = QueryTimings {
        total_ms: per_query.iter().map(|q| q.timings.total_ms).sum::<f64>() / n as f64,
        refresh_ms: per_query.iter().map(|q| q.timings.refresh_ms).sum::<f64>() / n as f64,
        lexical_bm25_ms: per_query
            .iter()
            .map(|q| q.timings.lexical_bm25_ms)
            .sum::<f64>()
            / n as f64,
        snapshot_query_ms: per_query
            .iter()
            .map(|q| q.timings.snapshot_query_ms)
            .sum::<f64>()
            / n as f64,
        rerank_ms: per_query.iter().map(|q| q.timings.rerank_ms).sum::<f64>() / n as f64,
        parse_ms: per_query.iter().map(|q| q.timings.parse_ms).sum::<f64>() / n as f64,
        sparse_scoring_ms: per_query
            .iter()
            .map(|q| q.timings.sparse_scoring_ms)
            .sum::<f64>()
            / n as f64,
        lexical_merge_ms: per_query
            .iter()
            .map(|q| q.timings.lexical_merge_ms)
            .sum::<f64>()
            / n as f64,
        posting_scoring_ms: per_query
            .iter()
            .map(|q| q.timings.posting_scoring_ms)
            .sum::<f64>()
            / n as f64,
        routing_seed_ms: per_query
            .iter()
            .map(|q| q.timings.routing_seed_ms)
            .sum::<f64>()
            / n as f64,
        candidate_accumulation_ms: per_query
            .iter()
            .map(|q| q.timings.candidate_accumulation_ms)
            .sum::<f64>()
            / n as f64,
        candidate_rank_ms: per_query
            .iter()
            .map(|q| q.timings.candidate_rank_ms)
            .sum::<f64>()
            / n as f64,
        metadata_ms: per_query.iter().map(|q| q.timings.metadata_ms).sum::<f64>() / n as f64,
        graph_ms: per_query.iter().map(|q| q.timings.graph_ms).sum::<f64>() / n as f64,
        entity_graph_ms: per_query
            .iter()
            .map(|q| q.timings.entity_graph_ms)
            .sum::<f64>()
            / n as f64,
        sequence_rerank_ms: per_query
            .iter()
            .map(|q| q.timings.sequence_rerank_ms)
            .sum::<f64>()
            / n as f64,
        evidence_ms: per_query.iter().map(|q| q.timings.evidence_ms).sum::<f64>() / n as f64,
        group_build_ms: per_query
            .iter()
            .map(|q| q.timings.group_build_ms)
            .sum::<f64>()
            / n as f64,
        group_sort_ms: per_query
            .iter()
            .map(|q| q.timings.group_sort_ms)
            .sum::<f64>()
            / n as f64,
        ranking_ms: per_query.iter().map(|q| q.timings.ranking_ms).sum::<f64>() / n as f64,
    };

    AggregateMetrics {
        query_count: n,
        analysis_ms: per_query.iter().map(|q| q.analysis_ms).sum::<f64>() / n as f64,
        recall_at_k,
        recall_any_at_k,
        mrr: per_query.iter().map(|q| q.mrr).sum::<f64>() / n as f64,
        ndcg_at_10: per_query.iter().map(|q| q.ndcg_at_10).sum::<f64>() / n as f64,
        timings,
    }
}

fn aggregate_by_question_type(
    per_query: &[QueryMetrics],
    ks: &[usize],
) -> HashMap<String, TypeMetrics> {
    let mut buckets: HashMap<String, Vec<&QueryMetrics>> = HashMap::new();
    for q in per_query {
        let key = q
            .question_type
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        buckets.entry(key).or_default().push(q);
    }

    let mut out = HashMap::new();
    for (question_type, items) in buckets {
        let n = items.len();
        let mut recall_at_k = HashMap::new();
        let mut recall_any_at_k = HashMap::new();
        for k in ks {
            let avg = items
                .iter()
                .map(|q| q.recall_at_k.get(k).copied().unwrap_or(0.0))
                .sum::<f64>()
                / n as f64;
            recall_at_k.insert(*k, avg);
            let avg_any = items
                .iter()
                .map(|q| q.recall_any_at_k.get(k).copied().unwrap_or(0.0))
                .sum::<f64>()
                / n as f64;
            recall_any_at_k.insert(*k, avg_any);
        }
        out.insert(
            question_type,
            TypeMetrics {
                query_count: n,
                analysis_ms: items.iter().map(|q| q.analysis_ms).sum::<f64>() / n as f64,
                recall_at_k,
                recall_any_at_k,
                mrr: items.iter().map(|q| q.mrr).sum::<f64>() / n as f64,
                ndcg_at_10: items.iter().map(|q| q.ndcg_at_10).sum::<f64>() / n as f64,
            },
        );
    }
    out
}
