use anyhow::{Context, Result};
use clap::{ArgAction, Parser, ValueEnum};
use lint_ai::index::TemporalQueryHint;
use lint_ai::{
    aggregation::{build_aggregate_output, AggregateOutput},
    build_index_store, build_query_snapshot_from_source_documents,
    query_expansion::normalize_for_index,
    query_semantics::{analyze_query, QueryTimeHint},
    segments::{SegmentQueryDiagnostics, SegmentRoutingStrategy, SegmentedMemoryIndex},
    ChunkStrategy, PipelineOptions, QueryDiagnostics, QueryTimings, SearchResult, SourceDocument,
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

    /// Include experimental segmented MemoryIndex comparison metrics.
    #[arg(long, action = ArgAction::SetTrue)]
    segment_compare: bool,

    /// Number of routed segments to query for the top-N segmented variant.
    #[arg(long, default_value_t = 3)]
    segment_top_n: usize,

    /// Segment routing strategy to use for segmented comparison modes.
    #[arg(long, value_enum, default_value_t = SegmentRouterArg::Sparse)]
    segment_router: SegmentRouterArg,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SegmentRouterArg {
    Sparse,
    Kl,
    Local,
}

impl From<SegmentRouterArg> for SegmentRoutingStrategy {
    fn from(value: SegmentRouterArg) -> Self {
        match value {
            SegmentRouterArg::Sparse => SegmentRoutingStrategy::SparseOverlap,
            SegmentRouterArg::Kl => SegmentRoutingStrategy::KlDivergence,
            SegmentRouterArg::Local => SegmentRoutingStrategy::LocalDistinctiveness,
        }
    }
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
    segment_comparison: Option<SegmentComparisonMetrics>,
}

#[derive(Debug, Clone, Serialize)]
struct SegmentComparisonMetrics {
    segment_count: usize,
    top_n: usize,
    global: SegmentVariantMetrics,
    top_1: SegmentVariantMetrics,
    top_3_segments: SegmentVariantMetrics,
    top_5_segments: SegmentVariantMetrics,
    top_n_segments: SegmentVariantMetrics,
    all_segments: SegmentVariantMetrics,
    top_n_connection: MultiSessionConnectionDiagnostics,
    top_n_rewrite_stability: QueryRewriteStability,
}

#[derive(Debug, Clone, Serialize)]
struct SegmentVariantMetrics {
    latency_ms: f64,
    retrieved_session_ids: Vec<String>,
    recall_at_k: HashMap<usize, f64>,
    recall_any_at_k: HashMap<usize, f64>,
    mrr: f64,
    ndcg_at_10: f64,
    diagnostics: Option<SegmentQueryDiagnostics>,
    router_miss: Option<bool>,
    missing_relevant_segments: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct MultiSessionConnectionDiagnostics {
    selected_sessions: Vec<String>,
    correct_sessions: Vec<String>,
    correct_sessions_selected: Vec<String>,
    shared_terms: Vec<String>,
    shared_term_count: usize,
    time_signal: bool,
    connection_types: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct QueryRewriteStability {
    rewrites: Vec<QueryRewriteDiagnostics>,
    average_selected_session_jaccard: f64,
    average_local_memory_jaccard: f64,
    stable_correct_session_coverage: bool,
    stable_local_memory_evidence: bool,
}

#[derive(Debug, Clone, Serialize)]
struct QueryRewriteDiagnostics {
    rewrite: String,
    selected_sessions: Vec<String>,
    selected_session_jaccard_with_base: f64,
    local_memory_jaccard_with_base: f64,
    correct_sessions_selected: Vec<String>,
    covered_query_terms: Vec<String>,
    uncovered_query_terms: Vec<String>,
    local_memory_terms: Vec<String>,
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
        args.segment_compare,
        args.segment_top_n,
        args.segment_router.into(),
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
    segment_compare: bool,
    segment_top_n: usize,
    segment_router: SegmentRoutingStrategy,
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
        let options = PipelineOptions {
            ner_provider: Tier1NerProvider::Heuristic,
            spacy_model: "en_core_web_sm".to_string(),
            term_ranker: Tier1TermRankerKind::Yake,
            chunk_strategy: ChunkStrategy::Heading,
            chunk_lines: 40,
            chunk_overlap: 10,
            chunk_target_tokens: 450,
            chunk_max_tokens: 800,
            text_rerank_ngram,
            text_rerank_lcs,
            ..PipelineOptions::default()
        };
        let index = build_query_snapshot_from_source_documents(
            &source_docs,
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
        let global_results = results.clone();
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
        let segment_comparison = if segment_compare {
            Some(build_segment_comparison(
                &source_docs,
                &options,
                &query_text,
                max_k,
                segment_top_n,
                segment_router,
                ks,
                &relevant,
                timings.total_ms,
                &global_results,
            )?)
        } else {
            None
        };

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
            segment_comparison,
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

fn build_segment_comparison(
    source_docs: &[SourceDocument],
    options: &PipelineOptions,
    query_text: &str,
    max_k: usize,
    segment_top_n: usize,
    segment_router: SegmentRoutingStrategy,
    ks: &[usize],
    relevant: &HashSet<String>,
    global_latency_ms: f64,
    global_results: &[SearchResult],
) -> Result<SegmentComparisonMetrics> {
    let index_store = build_index_store(source_docs, options)?;
    let records = index_store
        .records()
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);
    let top_n = segment_top_n.max(1);

    let top_1_start = Instant::now();
    let top_1 = segmented.query_with_diagnostics_and_strategy(query_text, max_k, 1, segment_router);
    let top_1_latency_ms = top_1_start.elapsed().as_secs_f64() * 1000.0;

    let top_3_start = Instant::now();
    let top_3_output =
        segmented.query_with_diagnostics_and_strategy(query_text, max_k, 3, segment_router);
    let top_3_latency_ms = top_3_start.elapsed().as_secs_f64() * 1000.0;

    let top_5_start = Instant::now();
    let top_5_output =
        segmented.query_with_diagnostics_and_strategy(query_text, max_k, 5, segment_router);
    let top_5_latency_ms = top_5_start.elapsed().as_secs_f64() * 1000.0;

    let top_n_start = Instant::now();
    let top_n_output =
        segmented.query_with_diagnostics_and_strategy(query_text, max_k, top_n, segment_router);
    let top_n_latency_ms = top_n_start.elapsed().as_secs_f64() * 1000.0;
    let top_n_connection = connection_diagnostics(&records, &top_n_output.diagnostics, relevant);
    let top_n_rewrite_stability = rewrite_stability_diagnostics(
        &segmented,
        query_text,
        max_k,
        top_n,
        segment_router,
        relevant,
        &top_n_output.diagnostics,
    );

    let all_start = Instant::now();
    let all_output = segmented.query_all_segments_with_diagnostics(query_text, max_k);
    let all_latency_ms = all_start.elapsed().as_secs_f64() * 1000.0;

    Ok(SegmentComparisonMetrics {
        segment_count: segmented.len(),
        top_n,
        global: variant_metrics(global_results, global_latency_ms, ks, relevant, None),
        top_1: variant_metrics(
            &top_1.results,
            top_1_latency_ms,
            ks,
            relevant,
            Some(top_1.diagnostics),
        ),
        top_3_segments: variant_metrics(
            &top_3_output.results,
            top_3_latency_ms,
            ks,
            relevant,
            Some(top_3_output.diagnostics),
        ),
        top_5_segments: variant_metrics(
            &top_5_output.results,
            top_5_latency_ms,
            ks,
            relevant,
            Some(top_5_output.diagnostics),
        ),
        top_n_segments: variant_metrics(
            &top_n_output.results,
            top_n_latency_ms,
            ks,
            relevant,
            Some(top_n_output.diagnostics),
        ),
        all_segments: variant_metrics(
            &all_output.results,
            all_latency_ms,
            ks,
            relevant,
            Some(all_output.diagnostics),
        ),
        top_n_connection,
        top_n_rewrite_stability,
    })
}

fn variant_metrics(
    results: &[SearchResult],
    latency_ms: f64,
    ks: &[usize],
    relevant: &HashSet<String>,
    diagnostics: Option<SegmentQueryDiagnostics>,
) -> SegmentVariantMetrics {
    let retrieved_session_ids = dedupe_preserve_order(
        results
            .iter()
            .map(|result| evaluation_group_id(&result.doc_id))
            .collect(),
    );
    let mut recall_at_k = HashMap::new();
    let mut recall_any_at_k = HashMap::new();
    for k in ks {
        recall_at_k.insert(*k, recall_at_k_fn(&retrieved_session_ids, relevant, *k));
        recall_any_at_k.insert(*k, recall_any_at_k_fn(&retrieved_session_ids, relevant, *k));
    }
    let missing_relevant_segments = diagnostics
        .as_ref()
        .map(|diagnostics| {
            let selected = diagnostics
                .selected_segments
                .iter()
                .map(|route| route.segment_id.as_str())
                .collect::<HashSet<_>>();
            let mut missing = relevant
                .iter()
                .filter(|segment_id| !selected.contains(segment_id.as_str()))
                .cloned()
                .collect::<Vec<_>>();
            missing.sort();
            missing
        })
        .unwrap_or_default();
    let router_miss = diagnostics
        .as_ref()
        .map(|_| !relevant.is_empty() && !missing_relevant_segments.is_empty());

    SegmentVariantMetrics {
        latency_ms,
        mrr: reciprocal_rank(&retrieved_session_ids, relevant),
        ndcg_at_10: ndcg_at_k(&retrieved_session_ids, relevant, 10),
        retrieved_session_ids,
        recall_at_k,
        recall_any_at_k,
        diagnostics,
        router_miss,
        missing_relevant_segments,
    }
}

fn connection_diagnostics(
    records: &[lint_ai::index::DocRecord],
    diagnostics: &SegmentQueryDiagnostics,
    relevant: &HashSet<String>,
) -> MultiSessionConnectionDiagnostics {
    let selected_sessions = diagnostics
        .selected_segments
        .iter()
        .map(|route| route.segment_id.clone())
        .collect::<Vec<_>>();
    let selected_set = selected_sessions.iter().cloned().collect::<HashSet<_>>();
    let mut correct_sessions = relevant.iter().cloned().collect::<Vec<_>>();
    correct_sessions.sort();
    let mut correct_sessions_selected = correct_sessions
        .iter()
        .filter(|session_id| selected_set.contains(*session_id))
        .cloned()
        .collect::<Vec<_>>();
    correct_sessions_selected.sort();

    let mut term_session_counts: HashMap<String, HashSet<String>> = HashMap::new();
    let mut time_signal = false;
    for record in records {
        let Some(session_id) = record.group_id.as_ref() else {
            continue;
        };
        if !selected_set.contains(session_id) {
            continue;
        }
        if record.timestamp.is_some() || !record.temporal_terms.is_empty() {
            time_signal = true;
        }
        for term in record_connection_terms(record) {
            term_session_counts
                .entry(term)
                .or_default()
                .insert(session_id.clone());
        }
    }

    let mut shared_terms = term_session_counts
        .into_iter()
        .filter_map(|(term, sessions)| (sessions.len() >= 2).then_some(term))
        .collect::<Vec<_>>();
    shared_terms.sort();
    let shared_term_count = shared_terms.len();
    shared_terms.truncate(20);

    let mut connection_types = Vec::new();
    if selected_sessions.len() <= 1 {
        connection_types.push("single_session".to_string());
    }
    if !shared_terms.is_empty() {
        connection_types.push("shared_terms".to_string());
    }
    if time_signal {
        connection_types.push("temporal_signal".to_string());
    }
    if !correct_sessions_selected.is_empty() {
        connection_types.push("correct_session_overlap".to_string());
    }
    if connection_types.is_empty() {
        connection_types.push("no_obvious_connection".to_string());
    }

    MultiSessionConnectionDiagnostics {
        selected_sessions,
        correct_sessions,
        correct_sessions_selected,
        shared_term_count,
        shared_terms,
        time_signal,
        connection_types,
    }
}

fn rewrite_stability_diagnostics(
    segmented: &SegmentedMemoryIndex,
    query_text: &str,
    max_k: usize,
    segment_limit: usize,
    strategy: SegmentRoutingStrategy,
    relevant: &HashSet<String>,
    base_diagnostics: &SegmentQueryDiagnostics,
) -> QueryRewriteStability {
    let base_selected = base_diagnostics
        .selected_segments
        .iter()
        .map(|route| route.segment_id.clone())
        .collect::<Vec<_>>();
    let base_selected_set = base_selected.iter().cloned().collect::<HashSet<_>>();
    let base_correct = sorted_intersection(&base_selected_set, relevant);
    let base_local_memory_terms = local_memory_terms(base_diagnostics)
        .into_iter()
        .collect::<HashSet<_>>();
    let rewrites = query_rewrites(query_text);
    let mut rewrite_diagnostics = Vec::new();
    for rewrite in rewrites {
        let output =
            segmented.query_with_diagnostics_and_strategy(&rewrite, max_k, segment_limit, strategy);
        let selected_sessions = output
            .diagnostics
            .selected_segments
            .iter()
            .map(|route| route.segment_id.clone())
            .collect::<Vec<_>>();
        let selected_set = selected_sessions.iter().cloned().collect::<HashSet<_>>();
        let local_memory_terms = local_memory_terms(&output.diagnostics);
        let local_memory_set = local_memory_terms.iter().cloned().collect::<HashSet<_>>();
        rewrite_diagnostics.push(QueryRewriteDiagnostics {
            rewrite,
            selected_session_jaccard_with_base: jaccard(&base_selected_set, &selected_set),
            local_memory_jaccard_with_base: jaccard(&base_local_memory_terms, &local_memory_set),
            correct_sessions_selected: sorted_intersection(&selected_set, relevant),
            selected_sessions,
            covered_query_terms: output.diagnostics.covered_query_terms,
            uncovered_query_terms: output.diagnostics.uncovered_query_terms,
            local_memory_terms,
        });
    }

    let average_selected_session_jaccard = if rewrite_diagnostics.is_empty() {
        1.0
    } else {
        rewrite_diagnostics
            .iter()
            .map(|rewrite| rewrite.selected_session_jaccard_with_base)
            .sum::<f64>()
            / rewrite_diagnostics.len() as f64
    };
    let average_local_memory_jaccard = if rewrite_diagnostics.is_empty() {
        1.0
    } else {
        rewrite_diagnostics
            .iter()
            .map(|rewrite| rewrite.local_memory_jaccard_with_base)
            .sum::<f64>()
            / rewrite_diagnostics.len() as f64
    };
    let stable_correct_session_coverage = rewrite_diagnostics
        .iter()
        .all(|rewrite| rewrite.correct_sessions_selected == base_correct);
    let stable_local_memory_evidence = rewrite_diagnostics
        .iter()
        .all(|rewrite| rewrite.local_memory_jaccard_with_base >= 0.5);

    QueryRewriteStability {
        rewrites: rewrite_diagnostics,
        average_selected_session_jaccard,
        average_local_memory_jaccard,
        stable_correct_session_coverage,
        stable_local_memory_evidence,
    }
}

fn local_memory_terms(diagnostics: &SegmentQueryDiagnostics) -> Vec<String> {
    let mut terms = diagnostics
        .local_evidence
        .iter()
        .flat_map(|segment| segment.differentiators.iter())
        .filter(|differentiator| {
            differentiator
                .evidence_types
                .iter()
                .any(|evidence_type| evidence_type == "local_memory")
        })
        .map(|differentiator| differentiator.term.clone())
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

fn record_connection_terms(record: &lint_ai::index::DocRecord) -> HashSet<String> {
    let mut terms = HashSet::new();
    for term in &record.important_terms {
        terms.extend(normalized_tokens(&term.term));
    }
    for entity in &record.key_entities {
        terms.extend(normalized_tokens(&entity.text));
    }
    if let Some(topic) = &record.probable_topic {
        terms.extend(normalized_tokens(topic));
    }
    for temporal in &record.temporal_terms {
        terms.extend(normalized_tokens(temporal));
    }
    terms
}

fn query_rewrites(query: &str) -> Vec<String> {
    let mut rewrites = Vec::new();
    push_unique_rewrite(&mut rewrites, query.trim().to_string());

    let keyword_terms = raw_query_terms(query);
    let keyword_rewrite = keyword_terms.join(" ");
    push_unique_rewrite(&mut rewrites, keyword_rewrite);

    let mut dropped_one = keyword_terms;
    if dropped_one.len() > 2 {
        dropped_one.pop();
        push_unique_rewrite(&mut rewrites, dropped_one.join(" "));
    }

    rewrites
}

fn push_unique_rewrite(rewrites: &mut Vec<String>, rewrite: String) {
    if !rewrite.is_empty() && !rewrites.iter().any(|existing| existing == &rewrite) {
        rewrites.push(rewrite);
    }
}

fn normalized_tokens(text: &str) -> HashSet<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .map(normalize_for_index)
        .filter(|token| token.len() > 1)
        .filter(|token| !is_query_stopword(token))
        .collect()
}

fn raw_query_terms(text: &str) -> Vec<String> {
    let mut terms = text
        .split(|ch: char| !ch.is_alphanumeric())
        .map(|token| token.to_lowercase())
        .filter(|token| token.len() > 1)
        .filter(|token| !is_query_stopword(token))
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

fn is_query_stopword(token: &str) -> bool {
    matches!(
        token,
        "a" | "an"
            | "and"
            | "are"
            | "can"
            | "did"
            | "do"
            | "does"
            | "for"
            | "from"
            | "i"
            | "in"
            | "is"
            | "it"
            | "me"
            | "my"
            | "of"
            | "on"
            | "or"
            | "that"
            | "the"
            | "to"
            | "was"
            | "what"
            | "when"
            | "where"
            | "which"
            | "who"
            | "with"
            | "you"
    )
}

fn sorted_intersection(left: &HashSet<String>, right: &HashSet<String>) -> Vec<String> {
    let mut out = left.intersection(right).cloned().collect::<Vec<String>>();
    out.sort();
    out
}

fn jaccard(left: &HashSet<String>, right: &HashSet<String>) -> f64 {
    if left.is_empty() && right.is_empty() {
        return 1.0;
    }
    let intersection = left.intersection(right).count();
    let union = left.union(right).count();
    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
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
