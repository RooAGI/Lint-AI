use crate::index::{
    GlobalBm25Statistics, MemoryIndex, SearchResult, TemporalQueryContext, TemporalQueryHint,
};
use chrono::NaiveDate;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use super::catalog::*;
use super::diagnostics::*;
use super::model::*;
use super::routing::*;

pub(crate) fn segment_candidate_limit(top_k: usize) -> usize {
    top_k
        .saturating_mul(SEGMENT_CANDIDATE_OVERSAMPLE)
        .max(top_k)
}

fn query_local_segment(
    segment: &MemoryIndexSegment,
    query: &str,
    top_k: usize,
    temporal: TemporalQueryContext<'_>,
    reference_date: Option<&str>,
    statistics: &GlobalBm25Statistics,
) -> Result<Vec<SearchResult>, String> {
    Ok(segment
        .index
        .query_with_temporal_context_at_and_statistics(
            query,
            top_k,
            temporal,
            reference_date,
            Some(statistics),
        )
        .0)
}

struct LocalSegmentResult {
    segment_id: String,
    result: Result<Vec<SearchResult>, String>,
}

/// Execute the selected shard plan with bounded local fan-out. The coordinator
/// owns scheduling and failure accounting; reduction happens only after every
/// planned shard has produced a result.
fn execute_selected_segments(
    selected_segments: &[SegmentRoute],
    segments: &[MemoryIndexSegment],
    query: &str,
    candidate_limit: usize,
    temporal: TemporalQueryContext<'_>,
    reference_date: Option<&str>,
    statistics: &GlobalBm25Statistics,
) -> Vec<LocalSegmentResult> {
    let segments_by_id = segments
        .iter()
        .map(|segment| (segment.segment_id.as_str(), segment))
        .collect::<HashMap<_, _>>();
    let mut executed = Vec::with_capacity(selected_segments.len());
    for route_batch in selected_segments.chunks(MAX_LOCAL_SEGMENT_QUERY_CONCURRENCY) {
        let batch = route_batch
            .par_iter()
            .map(|route| {
                let segment = segments_by_id.get(route.segment_id.as_str()).copied();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    segment
                        .map(|segment| {
                            let mut local_bitmap = temporal
                                .allowed_segment_doc_bitmaps
                                .and_then(|maps| maps.get(&segment.segment_id).cloned())
                                .or_else(|| {
                                    temporal
                                        .allowed_doc_ids
                                        .map(|ids| segment.index.doc_bitmap_for_ids(ids))
                                });
                            if let (Some(bitmap), Some(ids)) =
                                (local_bitmap.as_mut(), temporal.allowed_doc_ids)
                            {
                                *bitmap &= segment.index.doc_bitmap_for_ids(ids);
                            }
                            let local_temporal = TemporalQueryContext {
                                allowed_doc_ids: None,
                                allowed_doc_bitmap: local_bitmap.as_ref(),
                                ..temporal
                            };
                            query_local_segment(
                                segment,
                                query,
                                candidate_limit,
                                local_temporal,
                                reference_date,
                                statistics,
                            )
                        })
                        .unwrap_or_else(|| {
                            Err("segment is not present in the query snapshot".to_string())
                        })
                }))
                .unwrap_or_else(|_| Err("local segment query panicked".to_string()));
                LocalSegmentResult {
                    segment_id: route.segment_id.clone(),
                    result,
                }
            })
            .collect::<Vec<_>>();
        executed.extend(batch);
    }
    executed
}

pub(crate) fn query_top_segment(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
) -> Vec<SearchResult> {
    query_top_segments(query, top_k, segments, 1)
}

pub(crate) fn query_top_segments(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
) -> Vec<SearchResult> {
    query_top_segments_with_diagnostics(query, top_k, segments, segment_limit).results
}

fn query_all_segments(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
) -> Vec<SearchResult> {
    query_all_segments_with_diagnostics(query, top_k, segments).results
}

fn query_top_segments_with_diagnostics(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
) -> SegmentQueryOutput {
    let corpus_stats = SegmentCorpusStats::from_segments(segments, 0);
    query_top_segments_with_corpus_stats_and_strategy(
        query,
        top_k,
        segments,
        segment_limit,
        false,
        SegmentRoutingStrategy::SparseOverlap,
        TemporalQueryContext::default(),
        None,
        &corpus_stats,
        None,
        0,
    )
}

fn query_top_segments_with_diagnostics_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    strategy: SegmentRoutingStrategy,
) -> SegmentQueryOutput {
    let corpus_stats = SegmentCorpusStats::from_segments(segments, 0);
    query_top_segments_with_corpus_stats_and_strategy(
        query,
        top_k,
        segments,
        segment_limit,
        false,
        strategy,
        TemporalQueryContext::default(),
        None,
        &corpus_stats,
        None,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn query_top_segments_with_corpus_stats_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    execute_all_eligible: bool,
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    reference_date: Option<&str>,
    corpus_stats: &SegmentCorpusStats,
    cached_statistics: Option<&GlobalBm25Statistics>,
    snapshot_generation: u64,
) -> SegmentQueryOutput {
    let profile = std::env::var_os("LINT_AI_QUERY_TIMINGS").is_some();
    let coordinator_started = std::time::Instant::now();
    let query_terms = query_tokens(query);
    if top_k == 0 || segment_limit == 0 {
        return SegmentQueryOutput {
            results: Vec::new(),
            diagnostics: SegmentQueryDiagnostics {
                snapshot_generation,
                query_terms: sorted_terms(&query_terms),
                uncovered_query_terms: sorted_terms(&query_terms),
                ..SegmentQueryDiagnostics::default()
            },
        };
    }

    let segments_by_id = segments
        .iter()
        .map(|segment| (segment.segment_id.as_str(), segment))
        .collect::<HashMap<_, _>>();
    let mut routes = route_segments_with_corpus_stats(query, segments, strategy, corpus_stats)
        .into_iter()
        .filter(|route| {
            segments_by_id
                .get(route.segment_id.as_str())
                .copied()
                .is_some_and(|segment| {
                    segment_has_allowed_documents(segment, temporal.allowed_doc_ids)
                })
        })
        .collect::<Vec<_>>();
    if execute_all_eligible {
        let mut present = routes
            .iter()
            .map(|route| route.segment_id.clone())
            .collect::<HashSet<_>>();
        for segment_id in &corpus_stats.ordered_segment_ids {
            if present.contains(segment_id.as_str()) {
                continue;
            }
            let Some(segment) = segments_by_id.get(segment_id.as_str()).copied() else {
                continue;
            };
            if segment_has_allowed_documents(segment, temporal.allowed_doc_ids) {
                routes.push(SegmentRoute {
                    segment_id: segment_id.clone(),
                    score: 0.0,
                    fallback: false,
                });
                present.insert(segment_id.clone());
            }
        }
    }
    let signal_routes = routes
        .iter()
        .filter(|route| route_has_signal(route, strategy, &query_terms))
        .cloned()
        .collect::<Vec<_>>();
    let selected_segments = signal_routes
        .iter()
        .take(segment_limit)
        .cloned()
        .collect::<Vec<_>>();
    let fallback_limit = if execute_all_eligible {
        segment_limit.saturating_sub(selected_segments.len())
    } else if selected_segments.is_empty() {
        segment_limit
    } else {
        0
    };
    let fallback_segments = routes
        .iter()
        .filter(|route| !route_has_signal(route, strategy, &query_terms))
        .take(fallback_limit)
        .map(|route| SegmentRoute {
            segment_id: route.segment_id.clone(),
            score: route.score,
            fallback: true,
        })
        .collect::<Vec<_>>();
    let execution_segments = if execute_all_eligible || selected_segments.is_empty() {
        selected_segments
            .iter()
            .chain(fallback_segments.iter())
            .cloned()
            .collect::<Vec<_>>()
    } else {
        selected_segments.clone()
    };
    let diagnostic_selected_segments = if execute_all_eligible {
        execution_segments.clone()
    } else {
        selected_segments.clone()
    };
    let routing_fallback_reason = if query_terms.is_empty() {
        Some("empty_query_terms".to_string())
    } else if selected_segments.is_empty() && !routes.is_empty() {
        Some("no_signal_routes".to_string())
    } else if selected_segments.len() < segment_limit && !fallback_segments.is_empty() {
        Some("insufficient_signal_routes".to_string())
    } else {
        None
    };
    let local_evidence = selected_segments
        .iter()
        .filter_map(|route| {
            segments_by_id
                .get(route.segment_id.as_str())
                .copied()
                .map(|segment| segment.local_evidence(&query_terms, corpus_stats))
        })
        .collect::<Vec<_>>();
    let mut merged = Vec::new();
    let mut per_segment_result_counts = HashMap::new();
    let mut covered_query_terms = HashSet::new();
    let computed_statistics;
    let global_statistics = if let Some(cached_statistics) = cached_statistics {
        debug_assert_eq!(cached_statistics.generation(), snapshot_generation);
        cached_statistics
    } else {
        computed_statistics = GlobalBm25Statistics::from_indexes(
            segments.iter().map(|segment| segment.index.as_ref()),
        );
        &computed_statistics
    };
    let candidate_limit = segment_candidate_limit(top_k);
    let mut shard_completeness = ShardQueryCompleteness {
        expected_segments: execution_segments
            .iter()
            .map(|route| route.segment_id.clone())
            .collect(),
        ..ShardQueryCompleteness::default()
    };
    let execution_results = execute_selected_segments(
        &execution_segments,
        segments,
        query,
        candidate_limit,
        temporal,
        reference_date,
        global_statistics,
    );
    if profile {
        eprintln!(
            "query_timing default_segment_execution_ms={:.3} segments={}",
            coordinator_started.elapsed().as_secs_f64() * 1000.0,
            execution_segments.len()
        );
    }
    let queried_segment_count = execution_results.len();
    for execution in execution_results {
        if let Some(segment) = segments_by_id.get(execution.segment_id.as_str()).copied() {
            for term in &query_terms {
                if corpus_stats.summary(&segment.segment_id).covers_term(term) {
                    covered_query_terms.insert(term.clone());
                }
            }
        }
        match execution.result {
            Ok(results) => {
                shard_completeness
                    .successful_segments
                    .push(execution.segment_id.clone());
                per_segment_result_counts.insert(execution.segment_id, results.len());
                merged.extend(results);
            }
            Err(message) => {
                per_segment_result_counts.insert(execution.segment_id.clone(), 0);
                shard_completeness.failures.push(ShardQueryFailure {
                    segment_id: execution.segment_id,
                    message,
                });
            }
        }
    }
    if let Some(allowed_doc_ids) = temporal.allowed_doc_ids {
        for segment in segments {
            if !segment_has_allowed_documents(segment, Some(allowed_doc_ids)) {
                per_segment_result_counts
                    .entry(segment.segment_id.clone())
                    .or_insert(0);
            }
        }
    }
    let (merged, merged_result_count, final_result_count) =
        finalize_segment_results(merged, top_k, true);
    let segments_with_results = execution_segments
        .iter()
        .filter_map(|route| {
            per_segment_result_counts
                .get(&route.segment_id)
                .copied()
                .filter(|count| *count > 0)
                .map(|_| route.segment_id.clone())
        })
        .collect::<Vec<_>>();
    let uncovered_query_terms = query_terms
        .difference(&covered_query_terms)
        .cloned()
        .collect::<HashSet<_>>();
    SegmentQueryOutput {
        results: merged,
        diagnostics: SegmentQueryDiagnostics {
            snapshot_generation,
            selected_segments: diagnostic_selected_segments,
            fallback_segments,
            routing_fallback: routing_fallback_reason.is_some(),
            routing_fallback_reason,
            local_evidence,
            queried_segment_count,
            per_segment_result_counts,
            merged_result_count,
            final_result_count,
            query_terms: sorted_terms(&query_terms),
            covered_query_terms: sorted_terms(&covered_query_terms),
            uncovered_query_terms: sorted_terms(&uncovered_query_terms),
            segments_with_results,
            shard_completeness: Some(shard_completeness),
        },
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn query_top_segments_with_segment_enrichment_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    corpus_stats: &SegmentCorpusStats,
    cached_statistics: Option<&GlobalBm25Statistics>,
    snapshot_generation: u64,
) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
    let query_terms = query_tokens(query);
    if top_k == 0 || segment_limit == 0 {
        return (
            SegmentQueryOutput {
                results: Vec::new(),
                diagnostics: SegmentQueryDiagnostics {
                    snapshot_generation,
                    query_terms: sorted_terms(&query_terms),
                    uncovered_query_terms: sorted_terms(&query_terms),
                    ..SegmentQueryDiagnostics::default()
                },
            },
            SegmentSpecificEnrichmentDiagnostics {
                selected_segments: Vec::new(),
                average_added_terms: 0.0,
                temporal_expanded_segments: Vec::new(),
                connected_expanded_segments: Vec::new(),
            },
        );
    }

    let selected_segments = route_segments_with_temporal_context_and_corpus_stats(
        query,
        segments,
        strategy,
        temporal,
        corpus_stats,
    )
    .into_iter()
    .take(segment_limit)
    .collect::<Vec<_>>();
    query_selected_segments_with_enrichment(
        query,
        top_k,
        segments,
        selected_segments,
        temporal,
        corpus_stats,
        Vec::new(),
        Vec::new(),
        false,
        false,
        None,
        cached_statistics,
        snapshot_generation,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn query_top_segments_with_route_aware_segment_enrichment_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    corpus_stats: &SegmentCorpusStats,
    cached_statistics: Option<&GlobalBm25Statistics>,
    snapshot_generation: u64,
) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
    let query_terms = query_tokens(query);
    if top_k == 0 || segment_limit == 0 {
        return (
            SegmentQueryOutput {
                results: Vec::new(),
                diagnostics: SegmentQueryDiagnostics {
                    snapshot_generation,
                    query_terms: sorted_terms(&query_terms),
                    uncovered_query_terms: sorted_terms(&query_terms),
                    ..SegmentQueryDiagnostics::default()
                },
            },
            SegmentSpecificEnrichmentDiagnostics {
                selected_segments: Vec::new(),
                average_added_terms: 0.0,
                temporal_expanded_segments: Vec::new(),
                connected_expanded_segments: Vec::new(),
            },
        );
    }

    let selected_segments = route_segments_with_temporal_context_and_corpus_stats(
        query,
        segments,
        strategy,
        temporal,
        corpus_stats,
    )
    .into_iter()
    .take(segment_limit)
    .collect::<Vec<_>>();
    query_selected_segments_with_enrichment(
        query,
        top_k,
        segments,
        selected_segments,
        temporal,
        corpus_stats,
        Vec::new(),
        Vec::new(),
        true,
        false,
        None,
        cached_statistics,
        snapshot_generation,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn query_top_segments_with_session_aggregated_segment_enrichment_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    corpus_stats: &SegmentCorpusStats,
    cached_statistics: Option<&GlobalBm25Statistics>,
    snapshot_generation: u64,
) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
    let query_terms = query_tokens(query);
    if top_k == 0 || segment_limit == 0 {
        return (
            SegmentQueryOutput {
                results: Vec::new(),
                diagnostics: SegmentQueryDiagnostics {
                    snapshot_generation,
                    query_terms: sorted_terms(&query_terms),
                    uncovered_query_terms: sorted_terms(&query_terms),
                    ..SegmentQueryDiagnostics::default()
                },
            },
            SegmentSpecificEnrichmentDiagnostics {
                selected_segments: Vec::new(),
                average_added_terms: 0.0,
                temporal_expanded_segments: Vec::new(),
                connected_expanded_segments: Vec::new(),
            },
        );
    }

    let selected_segments = route_segments_with_temporal_context_and_corpus_stats(
        query,
        segments,
        strategy,
        temporal,
        corpus_stats,
    )
    .into_iter()
    .take(segment_limit)
    .collect::<Vec<_>>();
    query_selected_segments_with_enrichment(
        query,
        top_k,
        segments,
        selected_segments,
        temporal,
        corpus_stats,
        Vec::new(),
        Vec::new(),
        false,
        true,
        None,
        cached_statistics,
        snapshot_generation,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn query_top_segments_with_temporal_path_enrichment_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    corpus_stats: &SegmentCorpusStats,
    cached_statistics: Option<&GlobalBm25Statistics>,
    snapshot_generation: u64,
) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
    let query_terms = query_tokens(query);
    if top_k == 0 || segment_limit == 0 {
        return (
            SegmentQueryOutput {
                results: Vec::new(),
                diagnostics: SegmentQueryDiagnostics {
                    snapshot_generation,
                    query_terms: sorted_terms(&query_terms),
                    uncovered_query_terms: sorted_terms(&query_terms),
                    ..SegmentQueryDiagnostics::default()
                },
            },
            SegmentSpecificEnrichmentDiagnostics {
                selected_segments: Vec::new(),
                average_added_terms: 0.0,
                temporal_expanded_segments: Vec::new(),
                connected_expanded_segments: Vec::new(),
            },
        );
    }

    let routed_segments = route_segments_with_temporal_context_and_corpus_stats(
        query,
        segments,
        strategy,
        temporal,
        corpus_stats,
    )
    .into_iter()
    .take(segment_limit)
    .collect::<Vec<_>>();
    let (expanded_segments, temporal_expanded_segments) =
        expand_temporal_path_segments(&routed_segments, segments, segment_limit, temporal);
    query_selected_segments_with_enrichment(
        query,
        top_k,
        segments,
        expanded_segments,
        temporal,
        corpus_stats,
        temporal_expanded_segments,
        Vec::new(),
        false,
        false,
        None,
        cached_statistics,
        snapshot_generation,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn query_top_segments_with_connected_segment_enrichment_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    corpus_stats: &SegmentCorpusStats,
    cached_statistics: Option<&GlobalBm25Statistics>,
    snapshot_generation: u64,
) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
    let query_terms = query_tokens(query);
    if top_k == 0 || segment_limit == 0 {
        return (
            SegmentQueryOutput {
                results: Vec::new(),
                diagnostics: SegmentQueryDiagnostics {
                    snapshot_generation,
                    query_terms: sorted_terms(&query_terms),
                    uncovered_query_terms: sorted_terms(&query_terms),
                    ..SegmentQueryDiagnostics::default()
                },
            },
            SegmentSpecificEnrichmentDiagnostics {
                selected_segments: Vec::new(),
                average_added_terms: 0.0,
                temporal_expanded_segments: Vec::new(),
                connected_expanded_segments: Vec::new(),
            },
        );
    }

    let routes = route_segments_with_temporal_context_and_corpus_stats(
        query,
        segments,
        strategy,
        temporal,
        corpus_stats,
    );
    let routed_segments = routes
        .iter()
        .take(segment_limit)
        .cloned()
        .collect::<Vec<_>>();
    let (expanded_segments, connected_expanded_segments) = expand_connected_segments(
        &routed_segments,
        &routes,
        segments,
        segment_limit,
        temporal,
        corpus_stats,
    );
    query_selected_segments_with_enrichment(
        query,
        top_k,
        segments,
        expanded_segments,
        temporal,
        corpus_stats,
        Vec::new(),
        connected_expanded_segments,
        false,
        true,
        None,
        cached_statistics,
        snapshot_generation,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn query_top_segments_with_missing_coverage_recovery_segment_enrichment_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    corpus_stats: &SegmentCorpusStats,
    cached_statistics: Option<&GlobalBm25Statistics>,
    snapshot_generation: u64,
) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
    let query_terms = query_tokens(query);
    if top_k == 0 || segment_limit == 0 {
        return (
            SegmentQueryOutput {
                results: Vec::new(),
                diagnostics: SegmentQueryDiagnostics {
                    snapshot_generation,
                    query_terms: sorted_terms(&query_terms),
                    uncovered_query_terms: sorted_terms(&query_terms),
                    ..SegmentQueryDiagnostics::default()
                },
            },
            SegmentSpecificEnrichmentDiagnostics {
                selected_segments: Vec::new(),
                average_added_terms: 0.0,
                temporal_expanded_segments: Vec::new(),
                connected_expanded_segments: Vec::new(),
            },
        );
    }

    let routes = route_segments_with_temporal_context_and_corpus_stats(
        query,
        segments,
        strategy,
        temporal,
        corpus_stats,
    );
    let routed_segments = routes
        .iter()
        .take(segment_limit)
        .cloned()
        .collect::<Vec<_>>();
    let (recovered_segments, recovery_events) = recover_missing_coverage_segments(
        &query_terms,
        &routed_segments,
        &routes,
        segments,
        corpus_stats,
    );
    query_selected_segments_with_enrichment(
        query,
        top_k,
        segments,
        recovered_segments,
        temporal,
        corpus_stats,
        Vec::new(),
        recovery_events,
        false,
        false,
        None,
        cached_statistics,
        snapshot_generation,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn query_top_segments_with_adaptive_segment_enrichment_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    base_segment_limit: usize,
    max_segment_limit: usize,
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    reference_date: Option<&str>,
    corpus_stats: &SegmentCorpusStats,
    cached_statistics: Option<&GlobalBm25Statistics>,
    snapshot_generation: u64,
) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
    let query_terms = query_tokens(query);
    if top_k == 0 || base_segment_limit == 0 || max_segment_limit == 0 {
        return (
            SegmentQueryOutput {
                results: Vec::new(),
                diagnostics: SegmentQueryDiagnostics {
                    snapshot_generation,
                    query_terms: sorted_terms(&query_terms),
                    uncovered_query_terms: sorted_terms(&query_terms),
                    ..SegmentQueryDiagnostics::default()
                },
            },
            SegmentSpecificEnrichmentDiagnostics {
                selected_segments: Vec::new(),
                average_added_terms: 0.0,
                temporal_expanded_segments: Vec::new(),
                connected_expanded_segments: Vec::new(),
            },
        );
    }

    let routes = route_segments_with_temporal_context_and_corpus_stats(
        query,
        segments,
        strategy,
        temporal,
        corpus_stats,
    );
    let selected_segments = adaptive_segment_routes(
        &query_terms,
        &routes,
        segments,
        corpus_stats,
        base_segment_limit,
        max_segment_limit,
    );
    query_selected_segments_with_enrichment(
        query,
        top_k,
        segments,
        selected_segments,
        temporal,
        corpus_stats,
        Vec::new(),
        Vec::new(),
        false,
        false,
        reference_date,
        cached_statistics,
        snapshot_generation,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn query_top_segments_with_adaptive_route_aware_segment_enrichment_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    base_segment_limit: usize,
    max_segment_limit: usize,
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    reference_date: Option<&str>,
    corpus_stats: &SegmentCorpusStats,
    cached_statistics: Option<&GlobalBm25Statistics>,
    snapshot_generation: u64,
) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
    let query_terms = query_tokens(query);
    if top_k == 0 || base_segment_limit == 0 || max_segment_limit == 0 {
        return (
            SegmentQueryOutput {
                results: Vec::new(),
                diagnostics: SegmentQueryDiagnostics {
                    snapshot_generation,
                    query_terms: sorted_terms(&query_terms),
                    uncovered_query_terms: sorted_terms(&query_terms),
                    ..SegmentQueryDiagnostics::default()
                },
            },
            SegmentSpecificEnrichmentDiagnostics {
                selected_segments: Vec::new(),
                average_added_terms: 0.0,
                temporal_expanded_segments: Vec::new(),
                connected_expanded_segments: Vec::new(),
            },
        );
    }

    let routes = route_segments_with_temporal_context_and_corpus_stats(
        query,
        segments,
        strategy,
        temporal,
        corpus_stats,
    );
    let selected_segments = adaptive_segment_routes(
        &query_terms,
        &routes,
        segments,
        corpus_stats,
        base_segment_limit,
        max_segment_limit,
    );
    query_selected_segments_with_enrichment(
        query,
        top_k,
        segments,
        selected_segments,
        temporal,
        corpus_stats,
        Vec::new(),
        Vec::new(),
        true,
        false,
        reference_date,
        cached_statistics,
        snapshot_generation,
    )
}

pub(crate) fn recover_missing_coverage_segments(
    query_terms: &HashSet<String>,
    selected_routes: &[SegmentRoute],
    routes: &[SegmentRoute],
    segments: &[MemoryIndexSegment],
    corpus_stats: &SegmentCorpusStats,
) -> (Vec<SegmentRoute>, Vec<ConnectedSegmentExpansion>) {
    if query_terms.is_empty() || selected_routes.is_empty() {
        return (selected_routes.to_vec(), Vec::new());
    }

    let segment_by_id = segments
        .iter()
        .map(|segment| (segment.segment_id.as_str(), segment))
        .collect::<HashMap<_, _>>();
    let mut selected = selected_routes.to_vec();
    let selected_ids = selected
        .iter()
        .map(|route| route.segment_id.clone())
        .collect::<HashSet<_>>();
    let mut recovery_events = Vec::new();

    let covered_terms = selected_query_terms(query_terms, &selected, segments, corpus_stats);
    let missing_terms = query_terms
        .difference(&covered_terms)
        .cloned()
        .collect::<HashSet<_>>();
    let weakest = selected
        .iter()
        .enumerate()
        .filter_map(|(idx, route)| {
            let segment = segment_by_id.get(route.segment_id.as_str()).copied()?;
            let score = weak_segment_score(
                query_terms,
                route,
                segment,
                &selected,
                segments,
                corpus_stats,
            );
            Some((idx, score))
        })
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right)
                .unwrap_or(Ordering::Equal)
                .then_with(|| Ordering::Equal)
        });

    if !missing_terms.is_empty() {
        if let Some((replace_idx, weak_score)) = weakest {
            let candidate_pool_limit = MISSING_COVERAGE_RECOVERY_POOL_LIMIT.min(routes.len());
            let replacement = (weak_score >= MISSING_COVERAGE_MIN_WEAK_SCORE)
                .then(|| {
                    routes
                        .iter()
                        .take(candidate_pool_limit)
                        .filter(|route| !selected_ids.contains(route.segment_id.as_str()))
                        .filter_map(|route| {
                            let segment = segment_by_id.get(route.segment_id.as_str()).copied()?;
                            let gain = missing_coverage_recovery_gain(
                                &missing_terms,
                                query_terms,
                                route,
                                segment,
                                corpus_stats,
                            );
                            (gain >= MISSING_COVERAGE_MIN_GAIN).then_some((route.clone(), gain))
                        })
                        .max_by(|(left_route, left_gain), (right_route, right_gain)| {
                            left_gain
                                .partial_cmp(right_gain)
                                .unwrap_or(Ordering::Equal)
                                .then_with(|| {
                                    left_route
                                        .score
                                        .partial_cmp(&right_route.score)
                                        .unwrap_or(Ordering::Equal)
                                })
                                .then_with(|| right_route.segment_id.cmp(&left_route.segment_id))
                        })
                })
                .flatten();

            if let Some((mut replacement, gain)) = replacement {
                let replaced = selected.swap_remove(replace_idx);
                replacement.score += gain * 0.05;
                let shared_subjects = sorted_missing_terms_covered(
                    &missing_terms,
                    &replacement,
                    segments,
                    corpus_stats,
                );
                recovery_events.push(ConnectedSegmentExpansion {
                    segment_id: replacement.segment_id.clone(),
                    source_segment_id: replaced.segment_id.clone(),
                    score: gain,
                    connection_types: vec!["missing_query_coverage".to_string()],
                    shared_people: Vec::new(),
                    shared_subjects,
                    shared_time: Vec::new(),
                    shared_actions: Vec::new(),
                    shared_objects: Vec::new(),
                    action: format!(
                        "recovered_missing_coverage:swapped_out:{}",
                        replaced.segment_id
                    ),
                });
                selected.push(replacement);
            }
        }
    }

    selected.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.segment_id.cmp(&b.segment_id))
    });

    (selected, recovery_events)
}

fn missing_coverage_recovery_gain(
    missing_terms: &HashSet<String>,
    query_terms: &HashSet<String>,
    route: &SegmentRoute,
    segment: &MemoryIndexSegment,
    corpus_stats: &SegmentCorpusStats,
) -> f32 {
    if missing_terms.is_empty() {
        return 0.0;
    }

    let mut gain = 0.0;
    let mut covered_missing = 0usize;
    for term in missing_terms {
        if !corpus_stats.summary(&segment.segment_id).covers_term(term) {
            continue;
        }
        covered_missing += 1;
        let idf = corpus_stats.idf(term);
        gain += corpus_stats
            .summary(&segment.segment_id)
            .local_term_weight(term)
            * idf
            * corpus_stats
                .summary(&segment.segment_id)
                .coverage_evidence_multiplier(term);
        gain += idf * 0.45;
    }
    if covered_missing == 0 {
        return 0.0;
    }

    let coverage_ratio = covered_missing as f32 / query_terms.len().max(1) as f32;
    gain + coverage_ratio + route.score.max(0.0) * 0.08
}

fn weak_segment_score(
    query_terms: &HashSet<String>,
    route: &SegmentRoute,
    segment: &MemoryIndexSegment,
    selected_routes: &[SegmentRoute],
    segments: &[MemoryIndexSegment],
    corpus_stats: &SegmentCorpusStats,
) -> f32 {
    let covered_terms = query_terms
        .iter()
        .filter(|term| corpus_stats.summary(&segment.segment_id).covers_term(term))
        .cloned()
        .collect::<HashSet<_>>();
    if covered_terms.is_empty() {
        return 3.0;
    }

    let other_covered_terms = selected_routes
        .iter()
        .filter(|selected| selected.segment_id != route.segment_id)
        .filter_map(|selected| {
            segments
                .iter()
                .find(|segment| segment.segment_id == selected.segment_id)
        })
        .flat_map(|segment| {
            query_terms
                .iter()
                .filter(|term| corpus_stats.summary(&segment.segment_id).covers_term(term))
                .cloned()
                .collect::<Vec<_>>()
        })
        .collect::<HashSet<_>>();
    let unique_coverage = covered_terms.difference(&other_covered_terms).count();
    let evidence_count = covered_terms
        .iter()
        .map(|term| {
            corpus_stats
                .summary(&segment.segment_id)
                .evidence_types_for_term(term)
                .len()
        })
        .sum::<usize>();
    let local_score = covered_terms
        .iter()
        .map(|term| {
            corpus_stats
                .summary(&segment.segment_id)
                .local_term_weight(term)
        })
        .sum::<f32>();

    let mut weakness = 0.0;
    if unique_coverage == 0 {
        weakness += 1.6;
    } else {
        weakness -= unique_coverage as f32 * 1.1;
    }
    if covered_terms.len() <= 1 {
        weakness += 0.8;
    }
    if evidence_count <= covered_terms.len() {
        weakness += 0.5;
    }
    if local_score < 0.75 {
        weakness += 0.7;
    }
    weakness - route.score.max(0.0) * 0.03
}

fn sorted_missing_terms_covered(
    missing_terms: &HashSet<String>,
    route: &SegmentRoute,
    segments: &[MemoryIndexSegment],
    corpus_stats: &SegmentCorpusStats,
) -> Vec<String> {
    let Some(segment) = segments
        .iter()
        .find(|segment| segment.segment_id == route.segment_id)
    else {
        return Vec::new();
    };
    let mut terms = missing_terms
        .iter()
        .filter(|term| corpus_stats.summary(&segment.segment_id).covers_term(term))
        .cloned()
        .collect::<Vec<_>>();
    terms.sort();
    terms
}

pub(crate) fn selected_query_terms(
    query_terms: &HashSet<String>,
    selected_routes: &[SegmentRoute],
    segments: &[MemoryIndexSegment],
    corpus_stats: &SegmentCorpusStats,
) -> HashSet<String> {
    let segments_by_id = segments
        .iter()
        .map(|segment| (segment.segment_id.as_str(), segment))
        .collect::<HashMap<_, _>>();
    let mut covered = HashSet::new();
    for route in selected_routes {
        let Some(segment) = segments_by_id.get(route.segment_id.as_str()).copied() else {
            continue;
        };
        for term in query_terms {
            if corpus_stats.summary(&segment.segment_id).covers_term(term) {
                covered.insert(term.clone());
            }
        }
    }
    covered
}

#[allow(clippy::too_many_arguments)]
fn query_selected_segments_with_enrichment(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    selected_segments: Vec<SegmentRoute>,
    temporal: TemporalQueryContext<'_>,
    corpus_stats: &SegmentCorpusStats,
    temporal_expanded_segments: Vec<TemporalSegmentExpansion>,
    connected_expanded_segments: Vec<ConnectedSegmentExpansion>,
    route_aware_rerank: bool,
    session_aggregate: bool,
    reference_date: Option<&str>,
    cached_statistics: Option<&GlobalBm25Statistics>,
    snapshot_generation: u64,
) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
    let profile = std::env::var_os("LINT_AI_QUERY_TIMINGS").is_some();
    let coordinator_started = std::time::Instant::now();
    let query_terms = query_tokens(query);
    let mut merged = Vec::new();
    let mut rerank_candidates = Vec::new();
    let mut seen_doc_ids = HashSet::new();
    let mut per_segment_result_counts = HashMap::new();
    let mut queried_segment_count = 0usize;
    let mut segments_with_results = Vec::new();
    let mut covered_query_terms = HashSet::new();
    let mut local_evidence = Vec::new();
    let mut enrichment_diagnostics = Vec::new();
    let mut shard_completeness = ShardQueryCompleteness {
        expected_segments: selected_segments
            .iter()
            .map(|route| route.segment_id.clone())
            .collect(),
        ..ShardQueryCompleteness::default()
    };
    let segments_by_id = segments
        .iter()
        .map(|segment| (segment.segment_id.as_str(), segment))
        .collect::<HashMap<_, _>>();
    let computed_statistics;
    let global_statistics = if let Some(cached_statistics) = cached_statistics {
        debug_assert_eq!(cached_statistics.generation(), snapshot_generation);
        cached_statistics
    } else {
        computed_statistics = GlobalBm25Statistics::from_indexes(
            segments.iter().map(|segment| segment.index.as_ref()),
        );
        &computed_statistics
    };
    let candidate_limit = segment_candidate_limit(top_k);
    let max_route_score = selected_segments
        .iter()
        .map(|route| route.score.max(0.0))
        .fold(0.0f32, f32::max);
    // Enrichment is independent for each routed segment. Compute it in
    // parallel before the deterministic reduction below.
    let enrichments = selected_segments
        .par_iter()
        .filter_map(|route| {
            segments_by_id
                .get(route.segment_id.as_str())
                .map(|segment| {
                    (
                        route.segment_id.clone(),
                        segment.enriched_query(query, &query_terms, temporal, corpus_stats),
                    )
                })
        })
        .collect::<HashMap<_, _>>();
    if profile {
        eprintln!(
            "query_timing enrichment_ms={:.3}",
            coordinator_started.elapsed().as_secs_f64() * 1000.0
        );
    }
    let bm25_started = std::time::Instant::now();
    let segment_results = selected_segments
        .par_iter()
        .filter_map(|route| {
            let segment = segments_by_id.get(route.segment_id.as_str())?;
            let enrichment = enrichments.get(route.segment_id.as_str())?;
            let results = segment
                .index
                .query_with_temporal_context_at_and_statistics_with_local_terms(
                    query,
                    candidate_limit,
                    temporal,
                    reference_date,
                    Some(global_statistics),
                    &enrichment.added_terms,
                )
                .0;
            Some((route.segment_id.clone(), results))
        })
        .collect::<HashMap<_, _>>();
    if profile {
        eprintln!(
            "query_timing segment_bm25_ms={:.3}",
            bm25_started.elapsed().as_secs_f64() * 1000.0
        );
    }
    let reduction_started = std::time::Instant::now();

    for route in selected_segments.iter().filter(|route| {
        if !segments_by_id.contains_key(route.segment_id.as_str()) {
            shard_completeness.failures.push(ShardQueryFailure {
                segment_id: route.segment_id.clone(),
                message: "segment is not present in the query snapshot".to_string(),
            });
            false
        } else {
            true
        }
    }) {
        let segment = segments_by_id[route.segment_id.as_str()];
        queried_segment_count += 1;
        shard_completeness
            .successful_segments
            .push(segment.segment_id.clone());
        for term in &query_terms {
            if corpus_stats.summary(&segment.segment_id).covers_term(term) {
                covered_query_terms.insert(term.clone());
            }
        }
        let enrichment = enrichments
            .get(route.segment_id.as_str())
            .expect("enrichment computed for every routed segment");
        let segment_query_results = segment_results
            .get(route.segment_id.as_str())
            .expect("results computed for every routed segment");
        let max_segment_result_score = segment_query_results
            .iter()
            .map(|result| result.score.max(0.0))
            .fold(0.0f32, f32::max);
        let mut segment_results = 0usize;
        for mut result in segment_query_results.iter().cloned() {
            if seen_doc_ids.insert(result.doc_id.clone()) {
                segment_results += 1;
                if route_aware_rerank {
                    rerank_candidates.push(RouteAwareCandidate::new(
                        result,
                        segment,
                        route,
                        &query_terms,
                        &enrichment,
                        temporal,
                        max_segment_result_score,
                        max_route_score,
                        corpus_stats,
                    ));
                } else {
                    if session_aggregate {
                        result.score +=
                            normalize_positive_score(route.score, max_route_score) * 0.25;
                    }
                    merged.push(result);
                }
            }
        }
        if segment_results > 0 {
            segments_with_results.push(segment.segment_id.clone());
        }
        per_segment_result_counts.insert(segment.segment_id.clone(), segment_results);
        local_evidence.push(SegmentLocalEvidence {
            segment_id: segment.segment_id.clone(),
            differentiators: enrichment
                .added_terms
                .iter()
                .filter_map(|term| {
                    enrichment
                        .term_weights
                        .get(term)
                        .map(|weight| LocalDifferentiator {
                            term: term.clone(),
                            weight: *weight,
                            evidence_types: enrichment
                                .term_evidence_types
                                .get(term)
                                .cloned()
                                .unwrap_or_default(),
                        })
                })
                .collect(),
        });
        enrichment_diagnostics.push(SegmentEnrichedQueryDiagnostics {
            segment_id: segment.segment_id.clone(),
            base_query: query.to_string(),
            enriched_query: enrichment.enriched_query.clone(),
            added_terms: enrichment.added_terms.clone(),
            evidence_types: enrichment.evidence_types.clone(),
            temporal_added_terms: enrichment.temporal_added_terms.clone(),
            temporal_evidence: enrichment.temporal_evidence.clone(),
            temporal_signal: enrichment.temporal_signal,
        });
    }

    if route_aware_rerank {
        merged = select_route_aware_top_k(rerank_candidates, top_k);
    }
    let (merged, merged_result_count, final_result_count) =
        finalize_segment_results(merged, top_k, session_aggregate);
    if profile {
        eprintln!(
            "query_timing reduction_ms={:.3} coordinator_ms={:.3}",
            reduction_started.elapsed().as_secs_f64() * 1000.0,
            coordinator_started.elapsed().as_secs_f64() * 1000.0
        );
    }
    let uncovered_query_terms = query_terms
        .difference(&covered_query_terms)
        .cloned()
        .collect::<HashSet<_>>();
    let average_added_terms = if enrichment_diagnostics.is_empty() {
        0.0
    } else {
        enrichment_diagnostics
            .iter()
            .map(|diagnostic| diagnostic.added_terms.len())
            .sum::<usize>() as f64
            / enrichment_diagnostics.len() as f64
    };

    (
        SegmentQueryOutput {
            results: merged,
            diagnostics: SegmentQueryDiagnostics {
                snapshot_generation,
                selected_segments,
                fallback_segments: Vec::new(),
                routing_fallback: false,
                routing_fallback_reason: None,
                local_evidence,
                queried_segment_count,
                per_segment_result_counts,
                merged_result_count,
                final_result_count,
                query_terms: sorted_terms(&query_terms),
                covered_query_terms: sorted_terms(&covered_query_terms),
                uncovered_query_terms: sorted_terms(&uncovered_query_terms),
                segments_with_results,
                shard_completeness: Some(shard_completeness),
            },
        },
        SegmentSpecificEnrichmentDiagnostics {
            selected_segments: enrichment_diagnostics,
            average_added_terms,
            temporal_expanded_segments,
            connected_expanded_segments,
        },
    )
}

/// Applies the common coordinator reduction after shard-local queries have
/// produced candidates. Keeping this in one place makes the single-result
/// and enrichment paths agree on grouping, deterministic ordering, and the
/// final result window.
fn finalize_segment_results(
    mut results: Vec<SearchResult>,
    top_k: usize,
    session_aggregate: bool,
) -> (Vec<SearchResult>, usize, usize) {
    if session_aggregate {
        results = aggregate_segment_results_by_session(results);
    } else {
        results.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.doc_id.cmp(&right.doc_id))
        });
    }
    let result_count = results.len();
    results.truncate(top_k);
    let final_result_count = results.len();
    (results, result_count, final_result_count)
}

pub(crate) fn aggregate_segment_results_by_session(
    results: Vec<SearchResult>,
) -> Vec<SearchResult> {
    let mut grouped: HashMap<String, Vec<SearchResult>> = HashMap::new();
    for result in results {
        grouped
            .entry(session_key_for_result(&result))
            .or_default()
            .push(result);
    }

    let mut ranked_groups = grouped
        .into_iter()
        .map(|(session_id, mut items)| {
            items.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.doc_id.cmp(&b.doc_id))
            });

            let best_score = items.first().map(|result| result.score).unwrap_or_default();
            let support_score = items
                .iter()
                .skip(1)
                .take(4)
                .map(|result| result.score.max(0.0) * 0.35)
                .sum::<f32>();
            let unique_terms = items
                .iter()
                .flat_map(|result| result.matched_terms.iter().cloned())
                .collect::<HashSet<_>>()
                .len()
                .min(8) as f32;
            let unique_entities = items
                .iter()
                .flat_map(|result| result.matched_entities.iter().cloned())
                .collect::<HashSet<_>>()
                .len()
                .min(6) as f32;
            let supporting_docs = items.len().min(4) as f32;
            let graph_support = items
                .iter()
                .map(|result| {
                    result.score_breakdown.graph_link_score
                        + result.score_breakdown.entity_graph_score
                })
                .sum::<f32>()
                .min(1.5);
            let session_score = best_score
                + support_score
                + unique_terms * 0.025
                + unique_entities * 0.05
                + supporting_docs * 0.04
                + graph_support * 0.20;

            for item in &mut items {
                item.score = session_score;
            }
            (session_id, session_score, items)
        })
        .collect::<Vec<_>>();

    ranked_groups.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    ranked_groups
        .into_iter()
        .flat_map(|(_, _, items)| items.into_iter())
        .collect()
}

fn session_key_for_result(result: &SearchResult) -> String {
    result.group_id.clone().unwrap_or_else(|| {
        result
            .doc_id
            .split("::turn")
            .next()
            .unwrap_or(&result.doc_id)
            .to_string()
    })
}

pub(crate) fn result_evidence_terms(
    result: &SearchResult,
    segment: &MemoryIndexSegment,
) -> HashSet<String> {
    let mut terms = HashSet::new();
    for value in result
        .matched_terms
        .iter()
        .chain(result.matched_entities.iter())
    {
        terms.extend(query_tokens(value));
    }
    if let Some(topic) = &result.probable_topic {
        terms.extend(query_tokens(topic));
    }
    if let Some(doc_type) = &result.doc_type_guess {
        terms.extend(query_tokens(doc_type));
    }
    if let Some(record) = segment.index.docs.get(&result.doc_id) {
        for term in &record.important_terms {
            terms.extend(query_tokens(&term.term));
        }
        for entity in &record.key_entities {
            terms.extend(query_tokens(&entity.text));
        }
        if let Some(topic) = &record.probable_topic {
            terms.extend(query_tokens(topic));
        }
        if let Some(doc_type) = &record.doc_type_guess {
            terms.extend(query_tokens(doc_type));
        }
        for term in &record.temporal_terms {
            terms.extend(query_tokens(term));
        }
    }
    terms
}

pub(crate) fn result_temporal_score(
    result: &SearchResult,
    segment: &MemoryIndexSegment,
    temporal: TemporalQueryContext<'_>,
) -> f32 {
    if !temporal.has_explicit_temporal && temporal.time_hint.is_none() && temporal.ends_at.is_none()
    {
        return 0.0;
    }
    let Some(record) = segment.index.docs.get(&result.doc_id) else {
        return 0.0;
    };
    let mut score = if record.temporal_terms.is_empty() {
        0.0
    } else {
        0.4
    };
    let Some(record_date) = record.timestamp.as_deref().and_then(parse_iso_date) else {
        return score;
    };
    let Some(query_date) = temporal.ends_at.and_then(parse_iso_date) else {
        return score + 0.2;
    };
    let window_days = temporal.window_days.max(1);
    let delta_days = record_date.signed_duration_since(query_date).num_days();
    let distance = delta_days.abs();
    if distance <= window_days {
        score += 1.0 - (distance as f32 / window_days as f32).clamp(0.0, 1.0);
    }
    match temporal.time_hint {
        Some(TemporalQueryHint::Past) if delta_days <= 0 => score += 0.4,
        Some(TemporalQueryHint::Present) | Some(TemporalQueryHint::Ongoing)
            if distance <= window_days =>
        {
            score += 0.4
        }
        Some(TemporalQueryHint::Mixed) if distance <= window_days * 4 => score += 0.2,
        _ => {}
    }
    score
}

fn expand_temporal_path_segments(
    routed_segments: &[SegmentRoute],
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    temporal: TemporalQueryContext<'_>,
) -> (Vec<SegmentRoute>, Vec<TemporalSegmentExpansion>) {
    if !temporal_path_active(temporal) || routed_segments.is_empty() {
        return (routed_segments.to_vec(), Vec::new());
    }

    let selected_ids = routed_segments
        .iter()
        .map(|route| route.segment_id.clone())
        .collect::<HashSet<_>>();
    let selected_dates = routed_segments
        .iter()
        .filter_map(|route| {
            segments
                .iter()
                .find(|segment| segment.segment_id == route.segment_id)
                .map(|segment| (route.segment_id.clone(), segment_dates(segment)))
        })
        .flat_map(|(segment_id, dates)| {
            dates
                .into_iter()
                .map(move |date| (segment_id.clone(), date))
        })
        .collect::<Vec<_>>();
    if selected_dates.is_empty() {
        return (routed_segments.to_vec(), Vec::new());
    }

    let window_days = temporal.window_days.max(1);
    let expansion_limit = segment_limit.max(1);
    let query_date = temporal.ends_at.and_then(parse_iso_date);
    let mut candidates = Vec::new();

    for segment in segments {
        if selected_ids.contains(&segment.segment_id) {
            continue;
        }
        for segment_date in segment_dates(segment) {
            let mut best: Option<(f32, TemporalSegmentExpansion)> = None;
            for (source_segment_id, anchor_date) in &selected_dates {
                let delta_days = segment_date.signed_duration_since(*anchor_date).num_days();
                let distance = delta_days.abs();
                let relation = temporal_path_relation(delta_days, distance, window_days);
                let Some(relation) = relation else {
                    continue;
                };
                if !temporal_relation_matches_hint(&relation, temporal, segment_date, query_date) {
                    continue;
                }
                let score = temporal_path_score(distance, window_days, &relation);
                let expansion = TemporalSegmentExpansion {
                    segment_id: segment.segment_id.clone(),
                    source_segment_id: source_segment_id.clone(),
                    relation,
                    days_apart: delta_days,
                    anchor_date: anchor_date.to_string(),
                    segment_date: segment_date.to_string(),
                };
                if best
                    .as_ref()
                    .map(|(best_score, _)| score > *best_score)
                    .unwrap_or(true)
                {
                    best = Some((score, expansion));
                }
            }
            if let Some(best) = best {
                candidates.push(best);
                break;
            }
        }
    }

    candidates.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.1.segment_id.cmp(&b.1.segment_id))
    });

    let mut expanded_routes = routed_segments.to_vec();
    let mut expanded_ids = selected_ids;
    let mut expansions = Vec::new();
    for (score, expansion) in candidates.into_iter().take(expansion_limit) {
        if !expanded_ids.insert(expansion.segment_id.clone()) {
            continue;
        }
        expanded_routes.push(SegmentRoute {
            segment_id: expansion.segment_id.clone(),
            score,
            fallback: false,
        });
        expansions.push(expansion);
    }

    (expanded_routes, expansions)
}

fn temporal_path_active(temporal: TemporalQueryContext<'_>) -> bool {
    temporal.has_explicit_temporal || temporal.time_hint.is_some() || temporal.starts_from.is_some()
}

fn segment_dates(segment: &MemoryIndexSegment) -> Vec<NaiveDate> {
    let mut dates = segment
        .index
        .docs
        .values()
        .filter_map(|record| record.timestamp.as_deref().and_then(parse_iso_date))
        .collect::<Vec<_>>();
    dates.sort();
    dates.dedup();
    dates
}

fn temporal_path_relation(delta_days: i64, distance: i64, window_days: i64) -> Option<String> {
    if distance <= window_days {
        return Some("near".to_string());
    }
    if delta_days < 0 && distance <= window_days * 4 {
        return Some("before".to_string());
    }
    if delta_days > 0 && distance <= window_days * 4 {
        return Some("after".to_string());
    }
    None
}

fn temporal_relation_matches_hint(
    relation: &str,
    temporal: TemporalQueryContext<'_>,
    segment_date: NaiveDate,
    query_date: Option<NaiveDate>,
) -> bool {
    match temporal.time_hint {
        Some(TemporalQueryHint::Past) => {
            relation == "before"
                || relation == "near"
                || query_date.map(|date| segment_date <= date).unwrap_or(false)
        }
        Some(TemporalQueryHint::Present) | Some(TemporalQueryHint::Ongoing) => relation == "near",
        Some(TemporalQueryHint::Mixed) | None => true,
    }
}

fn temporal_path_score(distance: i64, window_days: i64, relation: &str) -> f32 {
    let proximity = 1.0 - (distance.min(window_days * 4) as f32 / (window_days * 4) as f32);
    let relation_boost = match relation {
        "near" => 1.0,
        "before" | "after" => 0.65,
        _ => 0.4,
    };
    (proximity.max(0.0) * 1.5) + relation_boost
}

fn query_all_segments_with_diagnostics(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
) -> SegmentQueryOutput {
    let corpus_stats = SegmentCorpusStats::from_segments(segments, 0);
    query_top_segments_with_corpus_stats_and_strategy(
        query,
        top_k,
        segments,
        segments.len(),
        true,
        SegmentRoutingStrategy::SparseOverlap,
        TemporalQueryContext::default(),
        None,
        &corpus_stats,
        None,
        0,
    )
}

#[derive(Debug)]
pub(crate) struct SegmentQueryEnrichment {
    pub(crate) enriched_query: String,
    pub(crate) added_terms: Vec<String>,
    pub(crate) evidence_types: Vec<String>,
    pub(crate) temporal_added_terms: Vec<String>,
    pub(crate) temporal_evidence: Vec<String>,
    pub(crate) temporal_signal: bool,
    pub(crate) term_weights: HashMap<String, f32>,
    pub(crate) term_evidence_types: HashMap<String, Vec<String>>,
}

pub(crate) fn add_local_frequency_candidates(
    candidates: &mut HashMap<String, (f32, HashSet<String>)>,
    index: &MemoryIndex,
    query_terms: &HashSet<String>,
) {
    for source in query_terms {
        let Some(postings) = index.term_to_docs.get(source) else {
            continue;
        };
        let source_document_count = postings.len().max(1) as f32;
        for posting in postings.iter().take(LOCAL_FREQUENCY_MAX_SOURCE_POSTINGS) {
            let Some(record) = index.docs.get(&posting.doc_id) else {
                continue;
            };
            for target in &record.important_terms {
                // `MemoryIndex.term_to_docs` and `DocRecord.important_terms` are
                // populated from the normalized index vocabulary. Re-normalizing
                // every co-occurring term here would turn local enrichment into
                // a per-query stemming pass over the selected shards.
                let target_term = &target.term;
                if target_term.is_empty() || query_terms.contains(target_term) {
                    continue;
                }
                let entry = candidates.entry(target_term.clone()).or_default();
                entry.0 += target.score.max(0.1) / source_document_count * 0.55;
                entry.1.insert("local_frequency".to_string());
            }
        }
    }
}
