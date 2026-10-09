use crate::memory_api::{AddRequest, MemoryService, Message, SearchRequest};
use crate::{
    parse_reference_date, segments::SegmentRoutingStrategy, AggregateOutput, ChunkStrategy,
    MemoryIndexLayout, PipelineOptions, QueryDiagnostics, QueryTimings, Tier1NerProvider,
    Tier1TermRankerKind,
};
use anyhow::{Context, Result};
use clap::{ArgAction, Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Fixed user ID for all benchmark documents. The MemoryService production
/// path filters by user ownership, so every haystack session is indexed
/// under this user and every query searches as this user.
const BENCHMARK_USER_ID: &str = "benchmark-user";

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

    /// Enable beKIND semantic tags (occupation/activity hypernyms).
    #[arg(long)]
    bekind: bool,

    /// Enable CombSUM fusion: run multiple query variants and sum scores.
    #[arg(long)]
    combsum: bool,

    /// Enable RRF fusion: Reciprocal Rank Fusion across query variants.
    #[arg(long)]
    rrf: bool,

    /// Enable PRF: Pseudo-Relevance Feedback query expansion.
    #[arg(long)]
    prf: bool,

    /// Enable RM3: Relevance Model 3 query expansion.
    #[arg(long)]
    rm3: bool,

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
    ///
    /// NOTE: Segment comparison requires the direct index API
    /// (SegmentedMemoryIndex over a separately built IndexStore) and does
    /// not apply to the MemoryService production flow. This flag is
    /// accepted for CLI compatibility but the comparison is not produced;
    /// `segment_comparison` is always null in the report.
    #[arg(long, action = ArgAction::SetTrue)]
    segment_compare: bool,

    /// Number of routed segments to query for the top-N segmented variant.
    #[arg(long, default_value_t = 3)]
    segment_top_n: usize,

    /// Segment routing strategy to use for segmented comparison modes.
    #[arg(long, value_enum, default_value_t = SegmentRouterArg::TypedEvidenceMultiplicative)]
    segment_router: SegmentRouterArg,

    /// Tier1 NER backend. `heuristic` reproduces the published
    /// docs/benchmark.md numbers; `spacy` is the current default.
    #[arg(long, value_enum, default_value_t = Tier1NerProvider::Spacy)]
    ner_provider: Tier1NerProvider,

    /// Index layout mode: `single` uses one global tantivy index;
    /// `segmented` routes across per-session segments;
    /// `adaptive` starts routed and expands when evidence is thin.
    #[arg(long, value_enum, default_value_t = IndexModeArg::Single)]
    index_mode: IndexModeArg,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum IndexModeArg {
    Single,
    Segmented,
    Adaptive,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SegmentRouterArg {
    Sparse,
    Kl,
    Local,
    TypedEvidenceMultiplicative,
}

impl From<SegmentRouterArg> for SegmentRoutingStrategy {
    fn from(value: SegmentRouterArg) -> Self {
        match value {
            SegmentRouterArg::Sparse => SegmentRoutingStrategy::SparseOverlap,
            SegmentRouterArg::Kl => SegmentRoutingStrategy::KlDivergence,
            SegmentRouterArg::Local => SegmentRoutingStrategy::LocalDistinctiveness,
            SegmentRouterArg::TypedEvidenceMultiplicative => {
                SegmentRoutingStrategy::TypedEvidenceMultiplicative
            }
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

// Segment-comparison metric structs are kept so the JSON report format is
// unchanged (`segment_comparison` serializes as null). The comparison itself
// is not produced in the MemoryService flow; see the note on Args::segment_compare.
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
    diagnostics: Option<crate::segments::SegmentQueryDiagnostics>,
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
    ner_provider: Tier1NerProvider,
    aggregate: AggregateMetrics,
    by_question_type: HashMap<String, TypeMetrics>,
    per_query: Vec<QueryMetrics>,
}

/// CombSUM fusion (Luyi 2026-10-09): run multiple query variants and sum scores.
/// Variants: (1) original query, (2) query without temporal phrases (broader recall).
/// Returns fused SearchResponse with top_k results by summed score.
fn combsum_search(
    service: &mut crate::MemoryService,
    question: &str,
    top_k: usize,
    question_id: &str,
) -> anyhow::Result<crate::SearchResponse> {
    use std::collections::HashMap;

    // Generate variants
    let variants = vec![
        question.to_string(),                    // Original
        strip_temporal_phrases(question),        // No temporal (broader)
    ];

    // Run each variant, collect scores per doc_id
    let mut score_sums: HashMap<String, f32> = HashMap::new();
    let mut doc_map: HashMap<String, crate::SearchMemory> = HashMap::new();

    for variant in &variants {
        let resp = service
            .search(crate::SearchRequest {
                query: variant.clone(),
                user_id: BENCHMARK_USER_ID.to_string(),
                top_k,
                session_id: None,
                scope: None,
                filters: None,
                options: None,
            })
            .with_context(|| format!("combsum search failed for {}", question_id))?;
        for result in resp.data {
            *score_sums.entry(result.id.clone()).or_insert(0.0) += result.score;
            doc_map.entry(result.id.clone()).or_insert(result);
        }
    }

    // Sort by summed score, take top_k
    let mut fused: Vec<(String, f32)> = score_sums.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(top_k);

    let data: Vec<crate::SearchMemory> = fused
        .into_iter()
        .filter_map(|(doc_id, sum_score)| {
            doc_map.get(&doc_id).map(|r| {
                let mut fused_result = r.clone();
                fused_result.score = sum_score;
                fused_result
            })
        })
        .collect();

    Ok(crate::SearchResponse { data })
}

/// RRF fusion (Luyi 2026-10-09): Reciprocal Rank Fusion across query variants.
/// score(d) = sum over variants of 1 / (k + rank(d)), with k=60.
/// More robust than CombSUM as it uses ranks, not raw scores.
fn rrf_search(
    service: &mut crate::MemoryService,
    question: &str,
    top_k: usize,
    question_id: &str,
) -> anyhow::Result<crate::SearchResponse> {
    use std::collections::HashMap;

    const RRF_K: f32 = 60.0;

    let variants = vec![
        question.to_string(),
        strip_temporal_phrases(question),
    ];

    let mut rrf_scores: HashMap<String, f32> = HashMap::new();
    let mut doc_map: HashMap<String, crate::SearchMemory> = HashMap::new();

    for variant in &variants {
        let resp = service
            .search(crate::SearchRequest {
                query: variant.clone(),
                user_id: BENCHMARK_USER_ID.to_string(),
                top_k,
                session_id: None,
                scope: None,
                filters: None,
                options: None,
            })
            .with_context(|| format!("rrf search failed for {}", question_id))?;
        for (rank, result) in resp.data.iter().enumerate() {
            let rrf = 1.0 / (RRF_K + rank as f32 + 1.0);
            *rrf_scores.entry(result.id.clone()).or_insert(0.0) += rrf;
            doc_map.entry(result.id.clone()).or_insert_with(|| result.clone());
        }
    }

    let mut fused: Vec<(String, f32)> = rrf_scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(top_k);

    let data: Vec<crate::SearchMemory> = fused
        .into_iter()
        .filter_map(|(doc_id, rrf_score)| {
            doc_map.get(&doc_id).map(|r| {
                let mut fused_result = r.clone();
                fused_result.score = rrf_score;
                fused_result
            })
        })
        .collect();

    Ok(crate::SearchResponse { data })
}

/// PRF (Luyi 2026-10-09): Pseudo-Relevance Feedback.
/// 1. Search with original query, get top-10 docs
/// 2. Extract top terms from those docs (by frequency, excluding stopwords/query terms)
/// 3. Add top 20 terms to query, search again
fn prf_search(
    service: &mut crate::MemoryService,
    question: &str,
    top_k: usize,
    question_id: &str,
) -> anyhow::Result<crate::SearchResponse> {
    use std::collections::HashMap;

    // Step 1: Initial retrieval
    let initial = service
        .search(crate::SearchRequest {
            query: question.to_string(),
            user_id: BENCHMARK_USER_ID.to_string(),
            top_k: 10,
            session_id: None,
            scope: None,
            filters: None,
            options: None,
        })
        .with_context(|| format!("prf initial search failed for {}", question_id))?;

    // Step 2: Extract terms from top docs
    let query_terms: std::collections::HashSet<String> = question
        .to_lowercase()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let stopwords: std::collections::HashSet<&str> = [
        "the", "a", "an", "is", "was", "were", "are", "be", "been", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should", "what",
        "which", "who", "whom", "how", "when", "where", "why", "i", "my", "you",
    ]
    .into_iter()
    .collect();

    let mut term_freq: HashMap<String, usize> = HashMap::new();
    for doc in &initial.data {
        for term in doc.content.to_lowercase().split_whitespace() {
            let clean: String = term.chars().filter(|c| c.is_alphanumeric()).collect();
            if clean.len() > 2 && !stopwords.contains(clean.as_str()) && !query_terms.contains(&clean) {
                *term_freq.entry(clean).or_insert(0) += 1;
            }
        }
    }

    // Step 3: Take top 20 expansion terms
    let mut terms: Vec<(String, usize)> = term_freq.into_iter().collect();
    terms.sort_by(|a, b| b.1.cmp(&a.1));
    terms.truncate(20);
    let expansion: Vec<String> = terms.into_iter().map(|(t, _)| t).collect();

    // Step 4: Expanded query and final search
    let expanded_query = format!("{} {}", question, expansion.join(" "));
    service
        .search(crate::SearchRequest {
            query: expanded_query,
            user_id: BENCHMARK_USER_ID.to_string(),
            top_k,
            session_id: None,
            scope: None,
            filters: None,
            options: None,
        })
        .with_context(|| format!("prf expanded search failed for {}", question_id))
}

/// RM3 (Luyi 2026-10-09): Relevance Model 3 (simplified).
/// 1. Search with original query, get top docs with scores
/// 2. Build relevance model: weight terms by doc score
/// 3. Interpolate: expanded = original + top RM terms (lambda=0.5)
/// 4. Search with expanded query
fn rm3_search(
    service: &mut crate::MemoryService,
    question: &str,
    top_k: usize,
    question_id: &str,
) -> anyhow::Result<crate::SearchResponse> {
    use std::collections::HashMap;

    // Step 1: Initial retrieval with scores
    let initial = service
        .search(crate::SearchRequest {
            query: question.to_string(),
            user_id: BENCHMARK_USER_ID.to_string(),
            top_k: 10,
            session_id: None,
            scope: None,
            filters: None,
            options: None,
        })
        .with_context(|| format!("rm3 initial search failed for {}", question_id))?;

    // Step 2: Build relevance model P(w|R) weighted by doc scores
    let query_terms: std::collections::HashSet<String> = question
        .to_lowercase()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let stopwords: std::collections::HashSet<&str> = [
        "the", "a", "an", "is", "was", "were", "are", "be", "been", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
    ]
    .into_iter()
    .collect();

    let mut rm_weights: HashMap<String, f32> = HashMap::new();
    for doc in &initial.data {
        let weight = doc.score.max(0.01); // Avoid zero weights
        for term in doc.content.to_lowercase().split_whitespace() {
            let clean: String = term.chars().filter(|c| c.is_alphanumeric()).collect();
            if clean.len() > 2 && !stopwords.contains(clean.as_str()) && !query_terms.contains(&clean) {
                *rm_weights.entry(clean).or_insert(0.0) += weight;
            }
        }
    }

    // Step 3: Top RM terms (lambda interpolation simplified: just append)
    let mut terms: Vec<(String, f32)> = rm_weights.into_iter().collect();
    terms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    terms.truncate(15);
    let expansion: Vec<String> = terms.into_iter().map(|(t, _)| t).collect();

    // Step 4: Expanded query (original + RM terms)
    let expanded_query = format!("{} {}", question, expansion.join(" "));
    service
        .search(crate::SearchRequest {
            query: expanded_query,
            user_id: BENCHMARK_USER_ID.to_string(),
            top_k,
            session_id: None,
            scope: None,
            filters: None,
            options: None,
        })
        .with_context(|| format!("rm3 expanded search failed for {}", question_id))
}

/// Strip temporal phrases like "two weeks ago", "last Tuesday", "past month".
/// Simple heuristic: remove common temporal patterns.
fn strip_temporal_phrases(question: &str) -> String {
    let temporal_patterns = [
        "two weeks ago", "three weeks ago", "four weeks ago",
        "a week ago", "two months ago", "three months ago",
        "last week", "last month", "last Tuesday", "last Friday", "last Saturday",
        "past month", "past three months", "a couple of days ago",
        "10 days ago", "5 days ago",
    ];
    let mut result = question.to_string();
    for pattern in &temporal_patterns {
        result = result.replace(pattern, "");
    }
    // Clean up extra whitespace
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(crate) fn main() -> Result<()> {
    let args = Args::parse();
    // Enable beKIND semantic tags if requested (Luyi 2026-10-08).
    if args.bekind {
        crate::behood_query::set_enabled(true);
    }
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
        args.ner_provider.clone(),
        args.index_mode,
        args.combsum,
        args.rrf,
        args.prf,
        args.rm3,
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

#[allow(clippy::too_many_arguments)]
fn run_scoped_benchmark(
    raw: Vec<LongMemEvalEntry>,
    limit: Option<usize>,
    question_type: Option<&str>,
    ks: &[usize],
    text_rerank_ngram: bool,
    text_rerank_lcs: bool,
    _segment_compare: bool,
    segment_top_n: usize,
    segment_router: SegmentRoutingStrategy,
    ner_provider: Tier1NerProvider,
    index_mode: IndexModeArg,
    combsum: bool,
    rrf: bool,
    prf: bool,
    rm3: bool,
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

    // Per-question fresh MemoryService, like the old scoped benchmark:
    // each question gets its own index with only its haystack sessions.
    // This preserves the LongMemEval methodology (isolated per-question).
    //
    // PipelineOptions mirror the old direct-index benchmark so the
    // --ner-provider / --text-rerank-* flags keep their meaning. The
    // MemoryService applies them on both the add() and search() paths.
    // ner_provider is the master Python-free switch: Heuristic (the
    // default) skips all spaCy subprocesses via PipelineOptions::python_free().
    // structured_fact_retrieval stays off for the benchmark: the spaCy
    // dependency-parse extractor is too slow/brittle for 500 questions
    // (120s timeout killed the first attempt) when ner_provider=spacy.
    //
    // Indexing uses add_batch() (one refresh per question) instead of
    // add() in a loop (one refresh per session) — same semantics, ~50x
    // fewer behood round-trips per question.
    let mut per_query = Vec::with_capacity(entries.len());

    for (idx, entry) in entries.into_iter().enumerate() {
        let memory_index_layout = match index_mode {
            IndexModeArg::Single => MemoryIndexLayout::Single,
            IndexModeArg::Segmented => MemoryIndexLayout::Segmented {
                query_top_n: segment_top_n,
                routing_strategy: segment_router.clone(),
            },
            IndexModeArg::Adaptive => MemoryIndexLayout::AdaptiveSegmented {
                query_top_n: segment_top_n,
                max_query_n: segment_top_n.max(1) * 2,
                routing_strategy: segment_router.clone(),
            },
        };
        let options = PipelineOptions {
            ner_provider: ner_provider.clone(),
            spacy_model: "en_core_web_sm".to_string(),
            term_ranker: Tier1TermRankerKind::Yake,
            chunk_strategy: ChunkStrategy::Heading,
            chunk_lines: 40,
            chunk_overlap: 10,
            chunk_target_tokens: 450,
            chunk_max_tokens: 800,
            text_rerank_ngram,
            text_rerank_lcs,
            structured_fact_retrieval: false,
            memory_index_layout,
            ..PipelineOptions::default()
        };
        let mut service = MemoryService::in_memory(options);

        // Index this question's haystack sessions via add_batch (one
        // refresh). Dedupe: the dataset can list the same session twice
        // in one haystack; MemoryService rejects repeated request_ids.
        // Filter empty message content (validation rejects it).
        let mut seen = std::collections::HashSet::new();
        let mut add_requests = Vec::new();
        for (sess_idx, (session_id, turns)) in entry
            .haystack_session_ids
            .iter()
            .zip(entry.haystack_sessions.iter())
            .enumerate()
        {
            if turns.is_empty() || !seen.insert(session_id.clone()) {
                continue;
            }
            let session_date = entry
                .haystack_dates
                .get(sess_idx)
                .cloned()
                .unwrap_or_else(|| entry.question_date.clone());
            let timestamp_ms = parse_reference_date(&session_date)
                .and_then(|d| d.and_hms_opt(0, 0, 0))
                .map(|dt| dt.and_utc().timestamp_millis());
            let messages: Vec<Message> = turns
                .iter()
                .filter(|turn| !turn.content.trim().is_empty())
                .map(|turn| Message {
                    role: turn.role.clone(),
                    content: turn.content.clone(),
                    timestamp: timestamp_ms,
                    expires_at_ms: None,
                    supersedes_id: None,
                })
                .collect();
            if messages.is_empty() {
                continue;
            }
            add_requests.push(AddRequest {
                request_id: format!("bench-{session_id}"),
                messages,
                user_id: BENCHMARK_USER_ID.to_string(),
                session_id: session_id.clone(),
            });
        }
        if !add_requests.is_empty() {
            service
                .add_batch(add_requests)
                .with_context(|| format!("failed to index haystack for {}", entry.question_id))?;
        }

        // candidate_session_ids is this question's haystack (for reporting).
        let candidate_session_ids = entry.haystack_session_ids.clone();

        // Production search. session_id/scope/filters are None: stateless
        // search as the benchmark user, no conversation-state scoping.
        // Fusion/expansion (Luyi 2026-10-09):
        // --combsum: multiple variants, sum scores
        // --rrf: multiple variants, reciprocal rank fusion
        // --prf: pseudo-relevance feedback expansion
        // --rm3: relevance model 3 expansion
        let search_start = Instant::now();
        let response = if combsum {
            combsum_search(&mut service, &entry.question, max_k, &entry.question_id)?
        } else if rrf {
            rrf_search(&mut service, &entry.question, max_k, &entry.question_id)?
        } else if prf {
            prf_search(&mut service, &entry.question, max_k, &entry.question_id)?
        } else if rm3 {
            rm3_search(&mut service, &entry.question, max_k, &entry.question_id)?
        } else {
            service
                .search(SearchRequest {
                    query: entry.question.clone(),
                    user_id: BENCHMARK_USER_ID.to_string(),
                    top_k: max_k,
                    session_id: None,
                    scope: None,
                    filters: None,
                    options: None,
                })
                .with_context(|| format!("search failed for {}", entry.question_id))?
        };
        let search_ms = search_start.elapsed().as_secs_f64() * 1000.0;

        // Session IDs come from SearchMemory.session_id (Option<String>).
        // Dedupe preserving rank order, mirroring the old evaluation_group_id
        // collapsing of per-turn doc_ids to their session.
        let retrieved_session_ids = dedupe_preserve_order(
            response
                .data
                .iter()
                .filter_map(|m| m.session_id.clone())
                .collect(),
        );

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
            // MemoryService performs query analysis internally during
            // search(); there is no separate analysis step to time.
            analysis_ms: 0.0,
            candidate_session_ids,
            retrieved_session_ids,
            // build_aggregate_output needs a direct MemoryIndex handle, which
            // the MemoryService API does not expose.
            aggregation: None,
            recall_at_k,
            recall_any_at_k,
            mrr,
            ndcg_at_10,
            timings: QueryTimings {
                total_ms: search_ms,
                ..QueryTimings::default()
            },
            diagnostics: QueryDiagnostics::default(),
            // Segment comparison needs the direct index API and does not
            // apply to the MemoryService flow; always null (see Args note).
            segment_comparison: None,
        });

        let last = per_query.last().expect("query metrics should exist");
        eprintln!(
            "[{}/{}] {} candidates={} retrieved={} search={:.2}ms mrr={:.3}",
            idx + 1,
            per_query.len(),
            last.id,
            last.candidate_session_ids.len(),
            last.retrieved_session_ids.len(),
            last.timings.total_ms,
            last.mrr,
        );
    }

    Ok(BenchmarkReport {
        ner_provider,
        aggregate: aggregate_metrics(&per_query, ks),
        by_question_type: aggregate_by_question_type(&per_query, ks),
        per_query,
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
