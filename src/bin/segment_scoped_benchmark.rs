use anyhow::{bail, Context, Result};
use clap::{ArgAction, Parser, ValueEnum};
use lint_ai::index::{DocRecord, TemporalQueryHint};
use lint_ai::{
    aggregation::{build_aggregate_output, AggregateOutput},
    build_index_store, build_query_snapshot_from_source_documents,
    query_expansion::normalize_for_index,
    query_semantics::{analyze_query, QueryTimeHint},
    segments::{
        SegmentQueryDiagnostics, SegmentRoute, SegmentRoutingStrategy,
        SegmentSpecificEnrichmentDiagnostics, SegmentedMemoryIndex,
    },
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

    /// Maximum routed segments for adaptive top-N expansion. Defaults to fixed top-N when unset.
    #[arg(long, default_value_t = 0)]
    adaptive_segment_max_n: usize,

    /// Segment routing strategy to use for segmented comparison modes.
    #[arg(long, value_enum, default_value_t = SegmentRouterArg::Sparse)]
    segment_router: SegmentRouterArg,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SegmentRouterArg {
    Sparse,
    Kl,
    Local,
    CoverageLocal,
    TeamCoverageLocal,
    CoverageTeam,
    TypedEvidence,
}

impl From<SegmentRouterArg> for SegmentRoutingStrategy {
    fn from(value: SegmentRouterArg) -> Self {
        match value {
            SegmentRouterArg::Sparse => SegmentRoutingStrategy::SparseOverlap,
            SegmentRouterArg::Kl => SegmentRoutingStrategy::KlDivergence,
            SegmentRouterArg::Local => SegmentRoutingStrategy::LocalDistinctiveness,
            SegmentRouterArg::CoverageLocal => SegmentRoutingStrategy::CoverageLocalDistinctiveness,
            SegmentRouterArg::TeamCoverageLocal => {
                SegmentRoutingStrategy::TeamCoverageLocalDistinctiveness
            }
            SegmentRouterArg::CoverageTeam => SegmentRoutingStrategy::CoverageTeamSelection,
            SegmentRouterArg::TypedEvidence => SegmentRoutingStrategy::TypedEvidence,
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
    router_miss_failure: Option<RouterMissFailureReport>,
    global: SegmentVariantMetrics,
    top_1: SegmentVariantMetrics,
    top_3_segments: SegmentVariantMetrics,
    top_5_segments: SegmentVariantMetrics,
    top_n_segments: SegmentVariantMetrics,
    top_n_segment_enriched: SegmentVariantMetrics,
    top_n_segment_enriched_reranked: SegmentVariantMetrics,
    top_n_segment_enriched_session_aggregated: SegmentVariantMetrics,
    adaptive_top_n_segment_enriched: SegmentVariantMetrics,
    adaptive_top_n_segment_enriched_reranked: SegmentVariantMetrics,
    connected_top_n_segment_enriched: SegmentVariantMetrics,
    missing_coverage_recovered_segment_enriched: SegmentVariantMetrics,
    top_n_temporal_path_enriched: SegmentVariantMetrics,
    all_segments: SegmentVariantMetrics,
    top_n_connection: MultiSessionConnectionDiagnostics,
    top_n_rewrite_stability: QueryRewriteStability,
    top_n_segment_enrichment: SegmentSpecificEnrichmentDiagnostics,
    adaptive_top_n_segment_enrichment: SegmentSpecificEnrichmentDiagnostics,
    top_n_temporal_path_enrichment: SegmentSpecificEnrichmentDiagnostics,
}

#[derive(Debug, Clone, Serialize)]
struct SegmentComparisonAggregate {
    query_count: usize,
    average_segment_count: f64,
    top_n: usize,
    global: SegmentVariantAggregate,
    top_1: SegmentVariantAggregate,
    top_3_segments: SegmentVariantAggregate,
    top_5_segments: SegmentVariantAggregate,
    top_n_segments: SegmentVariantAggregate,
    top_n_segment_enriched: SegmentVariantAggregate,
    top_n_segment_enriched_reranked: SegmentVariantAggregate,
    top_n_segment_enriched_session_aggregated: SegmentVariantAggregate,
    adaptive_top_n_segment_enriched: SegmentVariantAggregate,
    adaptive_top_n_segment_enriched_reranked: SegmentVariantAggregate,
    connected_top_n_segment_enriched: SegmentVariantAggregate,
    missing_coverage_recovered_segment_enriched: SegmentVariantAggregate,
    top_n_temporal_path_enriched: SegmentVariantAggregate,
    all_segments: SegmentVariantAggregate,
}

#[derive(Debug, Clone, Serialize)]
struct SegmentVariantAggregate {
    latency_ms: f64,
    recall_at_k: HashMap<usize, f64>,
    recall_any_at_k: HashMap<usize, f64>,
    mrr: f64,
    ndcg_at_10: f64,
    router_miss_count: usize,
    average_missing_relevant_segments: f64,
    average_routed_relevant_segment_recall: Option<f64>,
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
    selected_relevant_segments: Vec<String>,
    missing_relevant_segments: Vec<String>,
    routed_relevant_segment_recall: Option<f64>,
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
    segment_aggregate: Option<SegmentComparisonAggregate>,
    router_miss_failures: Vec<RouterMissFailureReport>,
    router_miss_aggregate: RouterMissAggregate,
    by_question_type: HashMap<String, TypeMetrics>,
    per_query: Vec<QueryMetrics>,
}

#[derive(Debug, Clone, Serialize)]
struct RouterMissAggregate {
    failure_query_count: usize,
    missed_gold_session_count: usize,
    average_global_route_rank: Option<f64>,
    route_rank_buckets: HashMap<String, usize>,
    evidence_kind_match_counts: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize)]
struct RouterMissFailureReport {
    id: String,
    query: String,
    question_type: Option<String>,
    gold_sessions: Vec<String>,
    selected_sessions: Vec<String>,
    missed_gold_sessions: Vec<MissedGoldSessionReport>,
}

#[derive(Debug, Clone, Serialize)]
struct MissedGoldSessionReport {
    session_id: String,
    global_route_rank: Option<usize>,
    global_route_score: Option<f32>,
    evidence: Vec<RouterMissEvidenceBucket>,
}

#[derive(Debug, Clone, Serialize)]
struct RouterMissEvidenceBucket {
    kind: String,
    matched_query_terms: Vec<String>,
    selected_session_overlap: Vec<String>,
    missing_query_terms: Vec<String>,
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
        args.adaptive_segment_max_n,
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
    adaptive_segment_max_n: usize,
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
                &entry.question_id,
                &entry.question,
                Some(entry.question_type.as_str()),
                &query_text,
                max_k,
                segment_top_n,
                adaptive_segment_max_n,
                segment_router,
                ks,
                &relevant,
                timings.total_ms,
                &global_results,
                temporal,
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

    let router_miss_failures = router_miss_failure_reports(&per_query);

    Ok(BenchmarkReport {
        aggregate: aggregate_metrics(&per_query, ks),
        segment_aggregate: aggregate_segment_comparison(&per_query, ks),
        router_miss_aggregate: aggregate_router_miss_failures(&router_miss_failures),
        router_miss_failures,
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
    query_id: &str,
    original_query: &str,
    question_type: Option<&str>,
    query_text: &str,
    max_k: usize,
    segment_top_n: usize,
    adaptive_segment_max_n: usize,
    segment_router: SegmentRoutingStrategy,
    ks: &[usize],
    relevant: &HashSet<String>,
    global_latency_ms: f64,
    global_results: &[SearchResult],
    temporal: TemporalQueryContext<'_>,
) -> Result<SegmentComparisonMetrics> {
    let index_store = build_index_store(source_docs, options)?;
    let records = index_store
        .records()
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);
    verify_segment_group_alignment(&records, &segmented)?;
    let top_n = segment_top_n.max(1);
    let all_routed_segments =
        segmented.route_with_temporal_context_and_strategy(query_text, segment_router, temporal);
    let adaptive_max_n = if adaptive_segment_max_n == 0 {
        top_n
    } else {
        adaptive_segment_max_n.max(top_n)
    };

    let top_1_start = Instant::now();
    let top_1 = segmented.query_with_temporal_context_and_diagnostics_and_strategy(
        query_text,
        max_k,
        1,
        segment_router,
        temporal,
    );
    let top_1_latency_ms = top_1_start.elapsed().as_secs_f64() * 1000.0;

    let top_3_start = Instant::now();
    let top_3_output = segmented.query_with_temporal_context_and_diagnostics_and_strategy(
        query_text,
        max_k,
        3,
        segment_router,
        temporal,
    );
    let top_3_latency_ms = top_3_start.elapsed().as_secs_f64() * 1000.0;

    let top_5_start = Instant::now();
    let top_5_output = segmented.query_with_temporal_context_and_diagnostics_and_strategy(
        query_text,
        max_k,
        5,
        segment_router,
        temporal,
    );
    let top_5_latency_ms = top_5_start.elapsed().as_secs_f64() * 1000.0;

    let top_n_start = Instant::now();
    let top_n_output = segmented.query_with_temporal_context_and_diagnostics_and_strategy(
        query_text,
        max_k,
        top_n,
        segment_router,
        temporal,
    );
    let top_n_latency_ms = top_n_start.elapsed().as_secs_f64() * 1000.0;
    let segment_enriched_start = Instant::now();
    let (segment_enriched_output, top_n_segment_enrichment) = segmented
        .query_with_segment_enrichment_temporal_context_and_strategy(
            query_text,
            max_k,
            top_n,
            segment_router,
            temporal,
        );
    let segment_enriched_latency_ms = segment_enriched_start.elapsed().as_secs_f64() * 1000.0;
    let segment_enriched_reranked_start = Instant::now();
    let (segment_enriched_reranked_output, _) = segmented
        .query_with_route_aware_segment_enrichment_temporal_context_and_strategy(
            query_text,
            max_k,
            top_n,
            segment_router,
            temporal,
        );
    let segment_enriched_reranked_latency_ms =
        segment_enriched_reranked_start.elapsed().as_secs_f64() * 1000.0;
    let segment_session_aggregated_start = Instant::now();
    let (segment_session_aggregated_output, _) = segmented
        .query_with_session_aggregated_segment_enrichment_temporal_context_and_strategy(
            query_text,
            max_k,
            top_n,
            segment_router,
            temporal,
        );
    let segment_session_aggregated_latency_ms =
        segment_session_aggregated_start.elapsed().as_secs_f64() * 1000.0;
    let adaptive_segment_enriched_start = Instant::now();
    let (adaptive_segment_enriched_output, adaptive_top_n_segment_enrichment) = segmented
        .query_with_adaptive_segment_enrichment_temporal_context_and_strategy(
            query_text,
            max_k,
            top_n,
            adaptive_max_n,
            segment_router,
            temporal,
        );
    let adaptive_segment_enriched_latency_ms =
        adaptive_segment_enriched_start.elapsed().as_secs_f64() * 1000.0;
    let adaptive_segment_enriched_reranked_start = Instant::now();
    let (adaptive_segment_enriched_reranked_output, _) = segmented
        .query_with_adaptive_route_aware_segment_enrichment_temporal_context_and_strategy(
            query_text,
            max_k,
            top_n,
            adaptive_max_n,
            segment_router,
            temporal,
        );
    let adaptive_segment_enriched_reranked_latency_ms = adaptive_segment_enriched_reranked_start
        .elapsed()
        .as_secs_f64()
        * 1000.0;
    let connected_segment_enriched_start = Instant::now();
    let (connected_segment_enriched_output, _) = segmented
        .query_with_connected_segment_enrichment_and_strategy(
            query_text,
            max_k,
            top_n,
            segment_router,
            temporal,
        );
    let connected_segment_enriched_latency_ms =
        connected_segment_enriched_start.elapsed().as_secs_f64() * 1000.0;
    let missing_coverage_recovered_start = Instant::now();
    let (missing_coverage_recovered_output, _) = segmented
        .query_with_missing_coverage_recovery_segment_enrichment_and_strategy(
            query_text,
            max_k,
            top_n,
            segment_router,
            temporal,
        );
    let missing_coverage_recovered_latency_ms =
        missing_coverage_recovered_start.elapsed().as_secs_f64() * 1000.0;
    let temporal_path_start = Instant::now();
    let (temporal_path_output, top_n_temporal_path_enrichment) = segmented
        .query_with_temporal_path_enrichment_and_strategy(
            query_text,
            max_k,
            top_n,
            segment_router,
            temporal,
        );
    let temporal_path_latency_ms = temporal_path_start.elapsed().as_secs_f64() * 1000.0;
    let top_n_connection = connection_diagnostics(&records, &top_n_output.diagnostics, relevant);
    let router_miss_failure = router_miss_failure_report(
        query_id,
        original_query,
        question_type,
        &records,
        &top_n_output.diagnostics,
        &all_routed_segments,
        relevant,
    );
    let top_n_rewrite_stability = rewrite_stability_diagnostics(
        &segmented,
        query_text,
        max_k,
        top_n,
        segment_router,
        relevant,
        &top_n_output.diagnostics,
        temporal,
    );

    let all_start = Instant::now();
    let all_output = segmented
        .query_all_segments_with_temporal_context_and_diagnostics(query_text, max_k, temporal);
    let all_latency_ms = all_start.elapsed().as_secs_f64() * 1000.0;

    Ok(SegmentComparisonMetrics {
        segment_count: segmented.len(),
        top_n,
        router_miss_failure,
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
        top_n_segment_enriched: variant_metrics(
            &segment_enriched_output.results,
            segment_enriched_latency_ms,
            ks,
            relevant,
            Some(segment_enriched_output.diagnostics),
        ),
        top_n_segment_enriched_reranked: variant_metrics(
            &segment_enriched_reranked_output.results,
            segment_enriched_reranked_latency_ms,
            ks,
            relevant,
            Some(segment_enriched_reranked_output.diagnostics),
        ),
        top_n_segment_enriched_session_aggregated: variant_metrics(
            &segment_session_aggregated_output.results,
            segment_session_aggregated_latency_ms,
            ks,
            relevant,
            Some(segment_session_aggregated_output.diagnostics),
        ),
        adaptive_top_n_segment_enriched: variant_metrics(
            &adaptive_segment_enriched_output.results,
            adaptive_segment_enriched_latency_ms,
            ks,
            relevant,
            Some(adaptive_segment_enriched_output.diagnostics),
        ),
        adaptive_top_n_segment_enriched_reranked: variant_metrics(
            &adaptive_segment_enriched_reranked_output.results,
            adaptive_segment_enriched_reranked_latency_ms,
            ks,
            relevant,
            Some(adaptive_segment_enriched_reranked_output.diagnostics),
        ),
        connected_top_n_segment_enriched: variant_metrics(
            &connected_segment_enriched_output.results,
            connected_segment_enriched_latency_ms,
            ks,
            relevant,
            Some(connected_segment_enriched_output.diagnostics),
        ),
        missing_coverage_recovered_segment_enriched: variant_metrics(
            &missing_coverage_recovered_output.results,
            missing_coverage_recovered_latency_ms,
            ks,
            relevant,
            Some(missing_coverage_recovered_output.diagnostics),
        ),
        top_n_temporal_path_enriched: variant_metrics(
            &temporal_path_output.results,
            temporal_path_latency_ms,
            ks,
            relevant,
            Some(temporal_path_output.diagnostics),
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
        top_n_segment_enrichment,
        adaptive_top_n_segment_enrichment,
        top_n_temporal_path_enrichment,
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
    let selected_relevant_segments = diagnostics
        .as_ref()
        .map(|diagnostics| {
            let selected = diagnostics
                .selected_segments
                .iter()
                .map(|route| route.segment_id.as_str())
                .collect::<HashSet<_>>();
            let mut selected_relevant = relevant
                .iter()
                .filter(|segment_id| selected.contains(segment_id.as_str()))
                .cloned()
                .collect::<Vec<_>>();
            selected_relevant.sort();
            selected_relevant
        })
        .unwrap_or_default();
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
    let routed_relevant_segment_recall = diagnostics.as_ref().and_then(|_| {
        if relevant.is_empty() {
            None
        } else {
            Some(selected_relevant_segments.len() as f64 / relevant.len() as f64)
        }
    });

    SegmentVariantMetrics {
        latency_ms,
        mrr: reciprocal_rank(&retrieved_session_ids, relevant),
        ndcg_at_10: ndcg_at_k(&retrieved_session_ids, relevant, 10),
        retrieved_session_ids,
        recall_at_k,
        recall_any_at_k,
        diagnostics,
        router_miss,
        selected_relevant_segments,
        missing_relevant_segments,
        routed_relevant_segment_recall,
    }
}

fn router_miss_failure_reports(per_query: &[QueryMetrics]) -> Vec<RouterMissFailureReport> {
    per_query
        .iter()
        .filter_map(|query| {
            query
                .segment_comparison
                .as_ref()
                .and_then(|comparison| comparison.router_miss_failure.clone())
        })
        .collect()
}

fn aggregate_router_miss_failures(reports: &[RouterMissFailureReport]) -> RouterMissAggregate {
    let mut route_rank_buckets = HashMap::new();
    let mut evidence_kind_match_counts = HashMap::new();
    let mut rank_sum = 0usize;
    let mut ranked_count = 0usize;
    let mut missed_gold_session_count = 0usize;

    for report in reports {
        for missed in &report.missed_gold_sessions {
            missed_gold_session_count += 1;
            let bucket = match missed.global_route_rank {
                Some(rank @ 1..=5) => {
                    rank_sum += rank;
                    ranked_count += 1;
                    "top_5"
                }
                Some(rank @ 6..=10) => {
                    rank_sum += rank;
                    ranked_count += 1;
                    "rank_6_to_10"
                }
                Some(rank @ 11..=20) => {
                    rank_sum += rank;
                    ranked_count += 1;
                    "rank_11_to_20"
                }
                Some(rank) => {
                    rank_sum += rank;
                    ranked_count += 1;
                    "rank_gt_20"
                }
                None => "not_routed",
            };
            *route_rank_buckets.entry(bucket.to_string()).or_insert(0) += 1;

            for bucket in &missed.evidence {
                if !bucket.matched_query_terms.is_empty() {
                    *evidence_kind_match_counts
                        .entry(bucket.kind.clone())
                        .or_insert(0) += 1;
                }
            }
        }
    }

    RouterMissAggregate {
        failure_query_count: reports.len(),
        missed_gold_session_count,
        average_global_route_rank: (ranked_count > 0)
            .then_some(rank_sum as f64 / ranked_count as f64),
        route_rank_buckets,
        evidence_kind_match_counts,
    }
}

fn router_miss_failure_report(
    query_id: &str,
    query: &str,
    question_type: Option<&str>,
    records: &[DocRecord],
    diagnostics: &SegmentQueryDiagnostics,
    all_routed_segments: &[SegmentRoute],
    relevant: &HashSet<String>,
) -> Option<RouterMissFailureReport> {
    if question_type != Some("multi-session") {
        return None;
    }

    let selected_sessions = diagnostics
        .selected_segments
        .iter()
        .map(|route| route.segment_id.clone())
        .collect::<Vec<_>>();
    let selected_set = selected_sessions.iter().cloned().collect::<HashSet<_>>();
    let mut gold_sessions = relevant.iter().cloned().collect::<Vec<_>>();
    gold_sessions.sort();
    let mut missing_sessions = relevant
        .iter()
        .filter(|session_id| !selected_set.contains(*session_id))
        .cloned()
        .collect::<Vec<_>>();
    missing_sessions.sort();
    if missing_sessions.is_empty() {
        return None;
    }

    let query_terms = normalized_tokens(query);
    let selected_profile = evidence_profile_for_sessions(records, &selected_set);
    let route_by_session = all_routed_segments
        .iter()
        .enumerate()
        .map(|(idx, route)| (route.segment_id.as_str(), (idx + 1, route.score)))
        .collect::<HashMap<_, _>>();
    let missed_gold_sessions = missing_sessions
        .into_iter()
        .map(|session_id| {
            let profile =
                evidence_profile_for_sessions(records, &HashSet::from([session_id.clone()]));
            let (global_route_rank, global_route_score) = route_by_session
                .get(session_id.as_str())
                .copied()
                .map(|(rank, score)| (Some(rank), Some(score)))
                .unwrap_or((None, None));
            MissedGoldSessionReport {
                session_id,
                global_route_rank,
                global_route_score,
                evidence: evidence_buckets(&query_terms, &profile, &selected_profile),
            }
        })
        .collect::<Vec<_>>();

    Some(RouterMissFailureReport {
        id: query_id.to_string(),
        query: query.to_string(),
        question_type: question_type.map(str::to_string),
        gold_sessions,
        selected_sessions,
        missed_gold_sessions,
    })
}

#[derive(Debug, Clone, Default)]
struct BenchmarkEvidenceProfile {
    people: HashSet<String>,
    subjects: HashSet<String>,
    times: HashSet<String>,
    actions: HashSet<String>,
    objects: HashSet<String>,
}

fn evidence_profile_for_sessions(
    records: &[DocRecord],
    sessions: &HashSet<String>,
) -> BenchmarkEvidenceProfile {
    let mut profile = BenchmarkEvidenceProfile::default();
    for record in records {
        let Some(group_id) = record.group_id.as_ref() else {
            continue;
        };
        if !sessions.contains(group_id) {
            continue;
        }
        for entity in &record.key_entities {
            let terms = normalized_tokens(&entity.text);
            let label = entity.label.to_ascii_lowercase();
            if label.contains("person") || label == "per" {
                profile.people.extend(terms.clone());
            }
            profile.subjects.extend(terms);
        }
        if let Some(topic) = &record.probable_topic {
            profile.subjects.extend(normalized_tokens(topic));
        }
        for heading in &record.headings {
            profile.subjects.extend(normalized_tokens(heading));
        }
        for term in &record.important_terms {
            for token in normalized_tokens(&term.term) {
                if looks_like_benchmark_action(&token) {
                    profile.actions.insert(token);
                } else {
                    profile.objects.insert(token);
                }
            }
        }
        for token in normalized_tokens(&record.content) {
            if looks_like_benchmark_action(&token) {
                profile.actions.insert(token);
            }
        }
        if let Some(timestamp) = &record.timestamp {
            profile
                .times
                .insert(timestamp.get(..10).unwrap_or(timestamp).to_string());
        }
        for temporal in &record.temporal_terms {
            profile.times.extend(normalized_tokens(temporal));
        }
    }
    profile
}

fn evidence_buckets(
    query_terms: &HashSet<String>,
    missed: &BenchmarkEvidenceProfile,
    selected: &BenchmarkEvidenceProfile,
) -> Vec<RouterMissEvidenceBucket> {
    vec![
        evidence_bucket("people", query_terms, &missed.people, &selected.people),
        evidence_bucket("subject", query_terms, &missed.subjects, &selected.subjects),
        evidence_bucket("time", query_terms, &missed.times, &selected.times),
        evidence_bucket("action", query_terms, &missed.actions, &selected.actions),
        evidence_bucket("object", query_terms, &missed.objects, &selected.objects),
    ]
}

fn evidence_bucket(
    kind: &str,
    query_terms: &HashSet<String>,
    missed_terms: &HashSet<String>,
    selected_terms: &HashSet<String>,
) -> RouterMissEvidenceBucket {
    let mut matched_query_terms = query_terms
        .intersection(missed_terms)
        .cloned()
        .collect::<Vec<_>>();
    matched_query_terms.sort();
    let mut selected_session_overlap = missed_terms
        .intersection(selected_terms)
        .cloned()
        .collect::<Vec<_>>();
    selected_session_overlap.sort();
    selected_session_overlap.truncate(12);
    let mut missing_query_terms = query_terms
        .difference(missed_terms)
        .cloned()
        .collect::<Vec<_>>();
    missing_query_terms.sort();
    missing_query_terms.truncate(12);
    RouterMissEvidenceBucket {
        kind: kind.to_string(),
        matched_query_terms,
        selected_session_overlap,
        missing_query_terms,
    }
}

fn looks_like_benchmark_action(token: &str) -> bool {
    token.ends_with("ed")
        || token.ends_with("ing")
        || matches!(
            token,
            "ask"
                | "asked"
                | "bought"
                | "buy"
                | "call"
                | "called"
                | "decid"
                | "discuss"
                | "discussed"
                | "find"
                | "found"
                | "go"
                | "need"
                | "plan"
                | "planned"
                | "schedule"
                | "scheduled"
                | "sent"
                | "share"
                | "shared"
                | "tell"
                | "told"
                | "visit"
                | "visited"
                | "want"
                | "went"
        )
}

fn verify_segment_group_alignment(
    records: &[DocRecord],
    segmented: &SegmentedMemoryIndex,
) -> Result<()> {
    let group_by_doc_id = records
        .iter()
        .filter_map(|record| {
            record
                .group_id
                .as_ref()
                .map(|group_id| (record.doc_id.as_str(), group_id.as_str()))
        })
        .collect::<HashMap<_, _>>();

    for segment in &segmented.segments {
        for doc_id in &segment.doc_ids {
            let Some(group_id) = group_by_doc_id.get(doc_id.as_str()) else {
                continue;
            };
            if *group_id != segment.segment_id {
                bail!(
                    "segment/group alignment mismatch: segment_id={} doc_id={} group_id={}",
                    segment.segment_id,
                    doc_id,
                    group_id
                );
            }
        }
    }

    Ok(())
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
    temporal: TemporalQueryContext<'_>,
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
        let output = segmented.query_with_temporal_context_and_diagnostics_and_strategy(
            &rewrite,
            max_k,
            segment_limit,
            strategy,
            temporal,
        );
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
            | "how"
            | "i"
            | "in"
            | "is"
            | "it"
            | "many"
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

fn aggregate_segment_comparison(
    per_query: &[QueryMetrics],
    ks: &[usize],
) -> Option<SegmentComparisonAggregate> {
    let comparisons = per_query
        .iter()
        .filter_map(|query| query.segment_comparison.as_ref())
        .collect::<Vec<_>>();
    if comparisons.is_empty() {
        return None;
    }

    let query_count = comparisons.len();
    let average_segment_count = comparisons
        .iter()
        .map(|comparison| comparison.segment_count as f64)
        .sum::<f64>()
        / query_count as f64;
    let top_n = comparisons
        .last()
        .map(|comparison| comparison.top_n)
        .unwrap_or_default();

    Some(SegmentComparisonAggregate {
        query_count,
        average_segment_count,
        top_n,
        global: aggregate_segment_variant(&comparisons, ks, |comparison| &comparison.global),
        top_1: aggregate_segment_variant(&comparisons, ks, |comparison| &comparison.top_1),
        top_3_segments: aggregate_segment_variant(&comparisons, ks, |comparison| {
            &comparison.top_3_segments
        }),
        top_5_segments: aggregate_segment_variant(&comparisons, ks, |comparison| {
            &comparison.top_5_segments
        }),
        top_n_segments: aggregate_segment_variant(&comparisons, ks, |comparison| {
            &comparison.top_n_segments
        }),
        top_n_segment_enriched: aggregate_segment_variant(&comparisons, ks, |comparison| {
            &comparison.top_n_segment_enriched
        }),
        top_n_segment_enriched_reranked: aggregate_segment_variant(
            &comparisons,
            ks,
            |comparison| &comparison.top_n_segment_enriched_reranked,
        ),
        top_n_segment_enriched_session_aggregated: aggregate_segment_variant(
            &comparisons,
            ks,
            |comparison| &comparison.top_n_segment_enriched_session_aggregated,
        ),
        adaptive_top_n_segment_enriched: aggregate_segment_variant(
            &comparisons,
            ks,
            |comparison| &comparison.adaptive_top_n_segment_enriched,
        ),
        adaptive_top_n_segment_enriched_reranked: aggregate_segment_variant(
            &comparisons,
            ks,
            |comparison| &comparison.adaptive_top_n_segment_enriched_reranked,
        ),
        connected_top_n_segment_enriched: aggregate_segment_variant(
            &comparisons,
            ks,
            |comparison| &comparison.connected_top_n_segment_enriched,
        ),
        missing_coverage_recovered_segment_enriched: aggregate_segment_variant(
            &comparisons,
            ks,
            |comparison| &comparison.missing_coverage_recovered_segment_enriched,
        ),
        top_n_temporal_path_enriched: aggregate_segment_variant(&comparisons, ks, |comparison| {
            &comparison.top_n_temporal_path_enriched
        }),
        all_segments: aggregate_segment_variant(&comparisons, ks, |comparison| {
            &comparison.all_segments
        }),
    })
}

fn aggregate_segment_variant<F>(
    comparisons: &[&SegmentComparisonMetrics],
    ks: &[usize],
    select: F,
) -> SegmentVariantAggregate
where
    F: Fn(&SegmentComparisonMetrics) -> &SegmentVariantMetrics,
{
    let n = comparisons.len();
    let variants = comparisons
        .iter()
        .map(|comparison| select(comparison))
        .collect::<Vec<_>>();

    let mut recall_at_k = HashMap::new();
    let mut recall_any_at_k = HashMap::new();
    for k in ks {
        let avg = variants
            .iter()
            .map(|variant| variant.recall_at_k.get(k).copied().unwrap_or(0.0))
            .sum::<f64>()
            / n as f64;
        recall_at_k.insert(*k, avg);
        let avg_any = variants
            .iter()
            .map(|variant| variant.recall_any_at_k.get(k).copied().unwrap_or(0.0))
            .sum::<f64>()
            / n as f64;
        recall_any_at_k.insert(*k, avg_any);
    }

    SegmentVariantAggregate {
        latency_ms: variants
            .iter()
            .map(|variant| variant.latency_ms)
            .sum::<f64>()
            / n as f64,
        recall_at_k,
        recall_any_at_k,
        mrr: variants.iter().map(|variant| variant.mrr).sum::<f64>() / n as f64,
        ndcg_at_10: variants
            .iter()
            .map(|variant| variant.ndcg_at_10)
            .sum::<f64>()
            / n as f64,
        router_miss_count: variants
            .iter()
            .filter(|variant| variant.router_miss.unwrap_or(false))
            .count(),
        average_missing_relevant_segments: variants
            .iter()
            .map(|variant| variant.missing_relevant_segments.len() as f64)
            .sum::<f64>()
            / n as f64,
        average_routed_relevant_segment_recall: average_present(
            variants
                .iter()
                .filter_map(|variant| variant.routed_relevant_segment_recall),
        ),
    }
}

fn average_present(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value;
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
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
