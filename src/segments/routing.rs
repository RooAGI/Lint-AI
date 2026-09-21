use crate::index::{SearchResult, TemporalQueryContext};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use super::catalog::*;
use super::model::*;
use super::query::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentRoutingStrategy {
    SparseOverlap,
    KlDivergence,
    LocalDistinctiveness,
    CoverageLocalDistinctiveness,
    TeamCoverageLocalDistinctiveness,
    CoverageTeamSelection,
    TypedEvidence,
}

pub(crate) fn route_segments(query: &str, segments: &[MemoryIndexSegment]) -> Vec<SegmentRoute> {
    route_segments_with_strategy(query, segments, SegmentRoutingStrategy::SparseOverlap)
}

pub(crate) fn route_segments_with_strategy(
    query: &str,
    segments: &[MemoryIndexSegment],
    strategy: SegmentRoutingStrategy,
) -> Vec<SegmentRoute> {
    let corpus_stats = SegmentCorpusStats::from_segments(segments, 0);
    route_segments_with_corpus_stats(query, segments, strategy, &corpus_stats)
}

pub(crate) fn route_segments_with_corpus_stats(
    query: &str,
    segments: &[MemoryIndexSegment],
    strategy: SegmentRoutingStrategy,
    corpus_stats: &SegmentCorpusStats,
) -> Vec<SegmentRoute> {
    let query_terms = query_tokens(query);
    if strategy == SegmentRoutingStrategy::TeamCoverageLocalDistinctiveness {
        return route_segments_by_team_coverage(&query_terms, segments, corpus_stats);
    }
    if strategy == SegmentRoutingStrategy::CoverageTeamSelection {
        return route_segments_by_coverage_team_selection(&query_terms, segments, corpus_stats);
    }
    if strategy == SegmentRoutingStrategy::TypedEvidence {
        return route_segments_by_typed_evidence(query, &query_terms, corpus_stats);
    }

    let candidate_segment_ids = corpus_stats.bounded_candidate_segment_ids(&query_terms);
    // Coverage/team/typed strategies compare candidates across the whole corpus
    // and therefore require the exhaustive route pool. Sparse and local routing
    // can safely use the inverted catalog directly.
    let supports_inverted_candidates = matches!(
        strategy,
        SegmentRoutingStrategy::SparseOverlap
            | SegmentRoutingStrategy::LocalDistinctiveness
            | SegmentRoutingStrategy::CoverageLocalDistinctiveness
    );
    let use_inverted_candidates = supports_inverted_candidates && !candidate_segment_ids.is_empty();
    let mut routes = if use_inverted_candidates {
        candidate_segment_ids
            .into_iter()
            .map(|segment_id| SegmentRoute {
                score: corpus_stats
                    .summary(&segment_id)
                    .score_query_terms_with_strategy(&query_terms, strategy, corpus_stats),
                segment_id,
                fallback: false,
            })
            .collect::<Vec<_>>()
    } else if supports_inverted_candidates {
        corpus_stats
            .ordered_segment_ids
            .iter()
            .map(|segment_id| SegmentRoute {
                segment_id: segment_id.clone(),
                score: 0.0,
                fallback: false,
            })
            .collect::<Vec<_>>()
    } else {
        segments
            .iter()
            .map(|segment| SegmentRoute {
                segment_id: segment.segment_id.clone(),
                score: corpus_stats
                    .summary(&segment.segment_id)
                    .score_query_terms_with_strategy(&query_terms, strategy, corpus_stats),
                fallback: false,
            })
            .collect::<Vec<_>>()
    };
    routes.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.segment_id.cmp(&b.segment_id))
    });
    routes
}

fn route_segments_by_typed_evidence(
    query: &str,
    query_terms: &HashSet<String>,
    corpus_stats: &SegmentCorpusStats,
) -> Vec<SegmentRoute> {
    let query_profile = query_connection_profile(query);
    let mut candidate_ids = corpus_stats.typed_candidate_segment_ids(&query_profile);
    candidate_ids.extend(corpus_stats.bounded_candidate_segment_ids(query_terms));
    let route_ids = if candidate_ids.is_empty() {
        corpus_stats.ordered_segment_ids.clone()
    } else {
        candidate_ids.into_iter().collect()
    };
    let mut routes = route_ids
        .into_iter()
        .filter_map(|segment_id| {
            corpus_stats.segment_positions.get(&segment_id)?;
            let base_score = corpus_stats
                .summary(&segment_id)
                .coverage_local_distinctiveness_score(query_terms, corpus_stats);
            let segment_profile = corpus_stats.connection_profiles.get(&segment_id)?;
            let typed_score = typed_evidence_route_score(&query_profile, &segment_profile);
            Some(SegmentRoute {
                segment_id,
                score: base_score + typed_score * TYPED_EVIDENCE_ROUTE_WEIGHT,
                fallback: false,
            })
        })
        .collect::<Vec<_>>();
    routes.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.segment_id.cmp(&b.segment_id))
    });
    routes
}

fn route_segments_by_team_coverage(
    query_terms: &HashSet<String>,
    segments: &[MemoryIndexSegment],
    corpus_stats: &SegmentCorpusStats,
) -> Vec<SegmentRoute> {
    let eligible_segments = bounded_routing_candidates(segments, query_terms, corpus_stats);
    let mut selected = Vec::with_capacity(eligible_segments.len());
    let mut selected_segment_ids = HashSet::new();
    let mut covered_terms = HashSet::new();
    let base_scores = eligible_segments
        .iter()
        .map(|segment| {
            (
                segment.segment_id.as_str(),
                corpus_stats
                    .summary(&segment.segment_id)
                    .coverage_local_distinctiveness_score(query_terms, corpus_stats),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut base_order = eligible_segments;
    base_order.sort_by(|left, right| {
        base_scores
            .get(right.segment_id.as_str())
            .copied()
            .unwrap_or_default()
            .partial_cmp(
                &base_scores
                    .get(left.segment_id.as_str())
                    .copied()
                    .unwrap_or_default(),
            )
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.segment_id.cmp(&right.segment_id))
    });

    while selected.len() < base_order.len() {
        let candidate_pool_limit = ((selected.len() + 1) * 3).max(8).min(base_order.len());
        let candidate_pool = base_order
            .iter()
            .take(candidate_pool_limit)
            .copied()
            .filter(|segment| !selected_segment_ids.contains(&segment.segment_id))
            .collect::<Vec<_>>();
        let candidate_segments = if candidate_pool.is_empty() {
            base_order
                .iter()
                .copied()
                .filter(|segment| !selected_segment_ids.contains(&segment.segment_id))
                .collect::<Vec<_>>()
        } else {
            candidate_pool
        };

        let Some((segment, score)) = candidate_segments
            .into_iter()
            .filter(|segment| !selected_segment_ids.contains(&segment.segment_id))
            .map(|segment| {
                let marginal_score = corpus_stats
                    .summary(&segment.segment_id)
                    .team_coverage_gain(query_terms, corpus_stats, &covered_terms);
                let base_score = base_scores
                    .get(segment.segment_id.as_str())
                    .copied()
                    .unwrap_or_default();
                (segment, marginal_score + (base_score * 2.0))
            })
            .max_by(|(left_segment, left_score), (right_segment, right_score)| {
                left_score
                    .partial_cmp(right_score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| right_segment.segment_id.cmp(&left_segment.segment_id))
            })
        else {
            break;
        };

        selected_segment_ids.insert(segment.segment_id.clone());
        for term in query_terms {
            if corpus_stats.summary(&segment.segment_id).covers_term(term) {
                covered_terms.insert(term.clone());
            }
        }
        selected.push(SegmentRoute {
            segment_id: segment.segment_id.clone(),
            score,
            fallback: false,
        });
    }

    selected
}

fn bounded_routing_candidates<'a>(
    segments: &'a [MemoryIndexSegment],
    query_terms: &HashSet<String>,
    corpus_stats: &SegmentCorpusStats,
) -> Vec<&'a MemoryIndexSegment> {
    let mut candidate_ids = corpus_stats.bounded_candidate_segment_ids(query_terms);
    if candidate_ids.is_empty() {
        candidate_ids.extend(
            corpus_stats
                .ordered_segment_ids
                .iter()
                .take(ROUTING_CANDIDATE_POOL_LIMIT)
                .cloned(),
        );
    }
    candidate_ids
        .into_iter()
        .filter_map(|segment_id| {
            corpus_stats
                .segment_positions
                .get(&segment_id)
                .and_then(|position| segments.get(*position))
        })
        .collect()
}

fn route_segments_by_coverage_team_selection(
    query_terms: &HashSet<String>,
    segments: &[MemoryIndexSegment],
    corpus_stats: &SegmentCorpusStats,
) -> Vec<SegmentRoute> {
    let eligible_segments = bounded_routing_candidates(segments, query_terms, corpus_stats);
    let mut selected = Vec::with_capacity(eligible_segments.len());
    let mut selected_segment_ids = HashSet::new();
    let mut coverage = TeamCoverageState::default();
    let base_scores = eligible_segments
        .iter()
        .map(|segment| {
            (
                segment.segment_id.as_str(),
                corpus_stats
                    .summary(&segment.segment_id)
                    .coverage_local_distinctiveness_score(query_terms, corpus_stats),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut base_order = eligible_segments;
    base_order.sort_by(|left, right| {
        base_scores
            .get(right.segment_id.as_str())
            .copied()
            .unwrap_or_default()
            .partial_cmp(
                &base_scores
                    .get(left.segment_id.as_str())
                    .copied()
                    .unwrap_or_default(),
            )
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.segment_id.cmp(&right.segment_id))
    });

    while selected.len() < base_order.len() {
        let candidate_pool_limit = ((selected.len() + 1) * 5).max(16).min(base_order.len());
        let candidate_pool = base_order
            .iter()
            .take(candidate_pool_limit)
            .copied()
            .filter(|segment| !selected_segment_ids.contains(&segment.segment_id))
            .collect::<Vec<_>>();
        let candidate_segments = if candidate_pool.is_empty() {
            base_order
                .iter()
                .copied()
                .filter(|segment| !selected_segment_ids.contains(&segment.segment_id))
                .collect::<Vec<_>>()
        } else {
            candidate_pool
        };

        let Some((segment, score)) = candidate_segments
            .into_iter()
            .filter(|segment| !selected_segment_ids.contains(&segment.segment_id))
            .map(|segment| {
                let marginal_score =
                    segment.team_coverage_gain(query_terms, corpus_stats, &coverage);
                let base_score = base_scores
                    .get(segment.segment_id.as_str())
                    .copied()
                    .unwrap_or_default();
                (segment, marginal_score + (base_score * 1.6))
            })
            .max_by(|(left_segment, left_score), (right_segment, right_score)| {
                left_score
                    .partial_cmp(right_score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| right_segment.segment_id.cmp(&left_segment.segment_id))
            })
        else {
            break;
        };

        selected_segment_ids.insert(segment.segment_id.clone());
        coverage.add_segment(segment, query_terms, corpus_stats);
        selected.push(SegmentRoute {
            segment_id: segment.segment_id.clone(),
            score,
            fallback: false,
        });
    }

    selected
}

#[derive(Debug, Default)]
pub(crate) struct TeamCoverageState {
    pub(crate) covered_terms: HashSet<String>,
    pub(crate) covered_evidence_keys: HashSet<String>,
    pub(crate) covered_connection_terms: HashSet<String>,
    pub(crate) has_temporal_signal: bool,
}

impl TeamCoverageState {
    fn add_segment(
        &mut self,
        segment: &MemoryIndexSegment,
        query_terms: &HashSet<String>,
        corpus_stats: &SegmentCorpusStats,
    ) {
        for term in query_terms {
            if corpus_stats.summary(&segment.segment_id).covers_term(term) {
                self.covered_terms.insert(term.clone());
                for evidence_type in corpus_stats
                    .summary(&segment.segment_id)
                    .evidence_types_for_term(term)
                {
                    self.covered_evidence_keys
                        .insert(format!("{evidence_type}:{term}"));
                }
            }
        }
        self.covered_connection_terms
            .extend(segment.connection_terms(corpus_stats));
        self.has_temporal_signal |= segment.has_temporal_signal();
    }
}

pub(crate) fn route_segments_with_temporal_context_and_corpus_stats(
    query: &str,
    segments: &[MemoryIndexSegment],
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    corpus_stats: &SegmentCorpusStats,
) -> Vec<SegmentRoute> {
    let mut routes = route_segments_with_corpus_stats(query, segments, strategy, corpus_stats)
        .into_iter()
        .filter_map(|mut route| {
            let segment = corpus_stats
                .segment_positions
                .get(&route.segment_id)
                .and_then(|position| segments.get(*position))?;
            if !segment_has_allowed_documents(segment, temporal.allowed_doc_ids) {
                return None;
            }
            route.score += segment_temporal_route_boost(segment, temporal);
            Some(route)
        })
        .collect::<Vec<_>>();
    routes.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.segment_id.cmp(&b.segment_id))
    });
    routes
}

pub(crate) fn segment_has_allowed_documents(
    segment: &MemoryIndexSegment,
    allowed_doc_ids: Option<&HashSet<String>>,
) -> bool {
    allowed_doc_ids.is_none_or(|allowed| {
        segment
            .doc_ids
            .iter()
            .any(|doc_id| allowed.contains(doc_id))
    })
}

pub(crate) fn adaptive_segment_routes(
    query_terms: &HashSet<String>,
    routes: &[SegmentRoute],
    segments: &[MemoryIndexSegment],
    corpus_stats: &SegmentCorpusStats,
    base_segment_limit: usize,
    max_segment_limit: usize,
) -> Vec<SegmentRoute> {
    let base_limit = base_segment_limit.max(1).min(routes.len());
    let max_limit = max_segment_limit.max(base_limit).min(routes.len());
    let mut selected = routes.iter().take(base_limit).cloned().collect::<Vec<_>>();
    if selected.len() >= max_limit || query_terms.is_empty() {
        return selected;
    }

    loop {
        if selected.len() >= max_limit {
            break;
        }
        let covered_terms = selected_query_terms(query_terms, &selected, segments, corpus_stats);
        let coverage = covered_terms.len() as f32 / query_terms.len() as f32;
        let cutoff_score = selected.last().map(|route| route.score).unwrap_or_default();
        let selected_ids = selected
            .iter()
            .map(|route| route.segment_id.as_str())
            .collect::<HashSet<_>>();

        let next = routes
            .iter()
            .take(max_limit)
            .filter(|route| !selected_ids.contains(route.segment_id.as_str()))
            .filter_map(|route| {
                let segment = segments
                    .iter()
                    .find(|segment| segment.segment_id == route.segment_id)?;
                let added_terms = query_terms
                    .iter()
                    .filter(|term| !covered_terms.contains(*term))
                    .filter(|term| corpus_stats.summary(&segment.segment_id).covers_term(term))
                    .count();
                let close = route_score_is_close(route.score, cutoff_score);
                let should_expand =
                    added_terms > 0 && (coverage < ADAPTIVE_MIN_QUERY_COVERAGE || close);
                should_expand.then_some((route, added_terms, close))
            })
            .max_by(
                |(left_route, left_added, left_close), (right_route, right_added, right_close)| {
                    left_added
                        .cmp(right_added)
                        .then_with(|| left_close.cmp(right_close))
                        .then_with(|| {
                            left_route
                                .score
                                .partial_cmp(&right_route.score)
                                .unwrap_or(Ordering::Equal)
                        })
                        .then_with(|| right_route.segment_id.cmp(&left_route.segment_id))
                },
            );

        let Some((route, _, _)) = next else {
            break;
        };
        selected.push(route.clone());
    }

    selected
}

fn route_score_is_close(candidate_score: f32, cutoff_score: f32) -> bool {
    if cutoff_score > 0.0 {
        candidate_score >= cutoff_score * ADAPTIVE_CLOSE_SCORE_RATIO
    } else {
        (candidate_score - cutoff_score).abs() <= 0.05
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RouteAwareCandidate {
    pub(crate) result: SearchResult,
    pub(crate) segment_id: String,
    pub(crate) base_score: f32,
    pub(crate) evidence_terms: HashSet<String>,
}

impl RouteAwareCandidate {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        mut result: SearchResult,
        segment: &MemoryIndexSegment,
        route: &SegmentRoute,
        query_terms: &HashSet<String>,
        enrichment: &SegmentQueryEnrichment,
        temporal: TemporalQueryContext<'_>,
        max_segment_result_score: f32,
        max_route_score: f32,
        corpus_stats: &SegmentCorpusStats,
    ) -> Self {
        let result_terms = result_evidence_terms(&result, segment);
        let normalized_result_score =
            normalize_positive_score(result.score, max_segment_result_score);
        let normalized_route_score = normalize_positive_score(route.score, max_route_score);
        let query_matches = query_terms
            .iter()
            .filter(|term| result_terms.contains(*term))
            .count() as f32;
        let enriched_matches = enrichment
            .added_terms
            .iter()
            .filter(|term| result_terms.contains(*term))
            .count() as f32;
        // Sum in sorted-term order via sorted_query_terms: HashSet iteration
        // order is nondeterministic and float summation is order-sensitive at
        // the last ULP. Without this, two identical rebuilds can produce
        // base_scores differing by 1 ULP, which defeats the doc_id tie-break
        // in select_route_aware_top_k and makes query results nondeterministic.
        let local_evidence_score = sorted_query_terms(&result_terms)
            .iter()
            .map(|term| {
                let local_weight = corpus_stats
                    .summary(&segment.segment_id)
                    .local_term_weight(term);
                if local_weight <= 0.0 {
                    0.0
                } else {
                    local_weight
                        * corpus_stats
                            .summary(&segment.segment_id)
                            .coverage_evidence_multiplier(term)
                }
            })
            .sum::<f32>();
        let temporal_score = result_temporal_score(&result, segment, temporal);
        let common_only_penalty = if query_matches == 0.0 && enriched_matches == 0.0 {
            RERANK_COMMON_ONLY_PENALTY
        } else {
            0.0
        };
        let base_score = (normalized_result_score * RERANK_NORMALIZED_RESULT_WEIGHT)
            + (normalized_route_score * RERANK_ROUTE_WEIGHT)
            + (query_matches * RERANK_QUERY_EVIDENCE_WEIGHT)
            + (enriched_matches * RERANK_ENRICHED_EVIDENCE_WEIGHT)
            + (local_evidence_score.min(3.0) * RERANK_LOCAL_EVIDENCE_WEIGHT)
            + (temporal_score * RERANK_TEMPORAL_WEIGHT)
            - common_only_penalty;

        let mut evidence_terms = query_terms
            .iter()
            .filter(|term| result_terms.contains(*term))
            .cloned()
            .collect::<HashSet<_>>();
        evidence_terms.extend(
            enrichment
                .added_terms
                .iter()
                .filter(|term| result_terms.contains(*term))
                .cloned(),
        );
        result.score = base_score;

        Self {
            result,
            segment_id: segment.segment_id.clone(),
            base_score,
            evidence_terms,
        }
    }
}

pub(crate) fn select_route_aware_top_k(
    mut candidates: Vec<RouteAwareCandidate>,
    top_k: usize,
) -> Vec<SearchResult> {
    let mut selected = Vec::new();
    let mut covered_terms = HashSet::new();
    let mut represented_segments = HashSet::new();

    while selected.len() < top_k && !candidates.is_empty() {
        let best_index = candidates
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                route_aware_selection_score(left, &covered_terms, &represented_segments)
                    .partial_cmp(&route_aware_selection_score(
                        right,
                        &covered_terms,
                        &represented_segments,
                    ))
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| right.result.doc_id.cmp(&left.result.doc_id))
            })
            .map(|(index, _)| index);
        let Some(best_index) = best_index else {
            break;
        };
        let mut candidate = candidates.swap_remove(best_index);
        let coverage_gain = candidate.evidence_terms.difference(&covered_terms).count() as f32
            * RERANK_COVERAGE_GAIN_WEIGHT;
        let segment_gain = if represented_segments.contains(&candidate.segment_id) {
            0.0
        } else {
            RERANK_SEGMENT_COVERAGE_WEIGHT
        };
        candidate.result.score = candidate.base_score + coverage_gain + segment_gain;
        covered_terms.extend(candidate.evidence_terms.iter().cloned());
        represented_segments.insert(candidate.segment_id);
        selected.push(candidate.result);
    }

    selected
}

fn route_aware_selection_score(
    candidate: &RouteAwareCandidate,
    covered_terms: &HashSet<String>,
    represented_segments: &HashSet<String>,
) -> f32 {
    candidate.base_score
        + candidate.evidence_terms.difference(covered_terms).count() as f32
            * RERANK_COVERAGE_GAIN_WEIGHT
        + if represented_segments.contains(&candidate.segment_id) {
            0.0
        } else {
            RERANK_SEGMENT_COVERAGE_WEIGHT
        }
}

pub(crate) fn normalize_positive_score(score: f32, max_score: f32) -> f32 {
    if max_score > 0.0 {
        (score.max(0.0) / max_score).clamp(0.0, 1.0)
    } else {
        score.max(0.0)
    }
}
