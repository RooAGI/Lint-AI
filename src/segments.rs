use crate::index::{DocRecord, MemoryIndex, SearchResult, TemporalQueryContext, TemporalQueryHint};
use crate::query_expansion::normalize_for_index;
use chrono::NaiveDate;
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

const KL_SMOOTHING: f32 = 1.0e-6;
const LOCAL_IDF_FLOOR: f32 = 1.0;
const LOCAL_MEMORY_DECAY: f32 = 0.82;
const LOCAL_MEMORY_PRUNE_BELOW: f32 = 0.015;
const LOCAL_MEMORY_BASE_SIGNAL: f32 = 0.04;
const LOCAL_MEMORY_IMPORTANT_TERM_WEIGHT: f32 = 1.0;
const LOCAL_MEMORY_ENTITY_WEIGHT: f32 = 1.4;
const LOCAL_MEMORY_TOPIC_WEIGHT: f32 = 0.8;
const LOCAL_MEMORY_NEARBY_REINFORCEMENT: f32 = 0.08;
const LOCAL_MEMORY_PROFILE_WEIGHT: f32 = 0.35;
const LOCAL_MEMORY_RECORD_TERM_LIMIT: usize = 24;
const LOCAL_MEMORY_SEGMENT_TERM_LIMIT: usize = 96;
const SEGMENT_ENRICHMENT_TERM_LIMIT: usize = 6;
const ADAPTIVE_MIN_QUERY_COVERAGE: f32 = 0.80;
const ADAPTIVE_CLOSE_SCORE_RATIO: f32 = 0.80;
const RERANK_NORMALIZED_RESULT_WEIGHT: f32 = 1.0;
const RERANK_ROUTE_WEIGHT: f32 = 0.35;
const RERANK_QUERY_EVIDENCE_WEIGHT: f32 = 0.18;
const RERANK_ENRICHED_EVIDENCE_WEIGHT: f32 = 0.14;
const RERANK_LOCAL_EVIDENCE_WEIGHT: f32 = 0.12;
const RERANK_TEMPORAL_WEIGHT: f32 = 0.10;
const RERANK_COVERAGE_GAIN_WEIGHT: f32 = 0.16;
const RERANK_SEGMENT_COVERAGE_WEIGHT: f32 = 0.28;
const RERANK_COMMON_ONLY_PENALTY: f32 = 0.18;
const CONNECTED_EXPANSION_POOL_MULTIPLIER: usize = 3;
const CONNECTED_EXPANSION_MIN_SCORE: f32 = 1.4;
const CONNECTED_EXPANSION_MAX_SWAP_PENALTY: f32 = 0.35;
const TYPED_EVIDENCE_ROUTE_WEIGHT: f32 = 1.15;
const MISSING_COVERAGE_RECOVERY_POOL_LIMIT: usize = 20;
const MISSING_COVERAGE_MIN_GAIN: f32 = 1.8;
const MISSING_COVERAGE_MIN_WEAK_SCORE: f32 = 1.5;

pub struct MemoryIndexSegment {
    pub segment_id: String,
    pub doc_ids: Vec<String>,
    pub profile: SegmentProfile,
    pub index: MemoryIndex,
}

pub struct SegmentedMemoryIndex {
    pub segments: Vec<MemoryIndexSegment>,
    pub global_index: Option<MemoryIndex>,
    corpus_stats: SegmentCorpusStats,
}

#[derive(Debug, Clone, Default)]
pub struct SegmentProfile {
    pub terms: HashMap<String, f32>,
    pub entities: HashMap<String, f32>,
    pub topics: HashMap<String, f32>,
    pub local_memory: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SegmentRoute {
    pub segment_id: String,
    pub score: f32,
    pub fallback: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SegmentLocalEvidence {
    pub segment_id: String,
    pub differentiators: Vec<LocalDifferentiator>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LocalDifferentiator {
    pub term: String,
    pub weight: f32,
    pub evidence_types: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SegmentQueryDiagnostics {
    pub selected_segments: Vec<SegmentRoute>,
    pub fallback_segments: Vec<SegmentRoute>,
    pub routing_fallback: bool,
    pub routing_fallback_reason: Option<String>,
    pub local_evidence: Vec<SegmentLocalEvidence>,
    pub queried_segment_count: usize,
    pub per_segment_result_counts: HashMap<String, usize>,
    pub merged_result_count: usize,
    pub final_result_count: usize,
    pub query_terms: Vec<String>,
    pub covered_query_terms: Vec<String>,
    pub uncovered_query_terms: Vec<String>,
    pub segments_with_results: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SegmentQueryOutput {
    pub results: Vec<SearchResult>,
    pub diagnostics: SegmentQueryDiagnostics,
}

#[derive(Debug, Clone, Serialize)]
pub struct SegmentSpecificEnrichmentDiagnostics {
    pub selected_segments: Vec<SegmentEnrichedQueryDiagnostics>,
    pub average_added_terms: f64,
    pub temporal_expanded_segments: Vec<TemporalSegmentExpansion>,
    pub connected_expanded_segments: Vec<ConnectedSegmentExpansion>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SegmentEnrichedQueryDiagnostics {
    pub segment_id: String,
    pub base_query: String,
    pub enriched_query: String,
    pub added_terms: Vec<String>,
    pub evidence_types: Vec<String>,
    pub temporal_added_terms: Vec<String>,
    pub temporal_evidence: Vec<String>,
    pub temporal_signal: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TemporalSegmentExpansion {
    pub segment_id: String,
    pub source_segment_id: String,
    pub relation: String,
    pub days_apart: i64,
    pub anchor_date: String,
    pub segment_date: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectedSegmentExpansion {
    pub segment_id: String,
    pub source_segment_id: String,
    pub score: f32,
    pub connection_types: Vec<String>,
    pub shared_people: Vec<String>,
    pub shared_subjects: Vec<String>,
    pub shared_time: Vec<String>,
    pub shared_actions: Vec<String>,
    pub shared_objects: Vec<String>,
    pub action: String,
}

impl SegmentedMemoryIndex {
    pub fn from_segments(segments: Vec<MemoryIndexSegment>) -> Self {
        let records = segments
            .iter()
            .flat_map(|segment| segment.index.docs.values().cloned())
            .collect::<Vec<_>>();
        let global_index = (!records.is_empty()).then(|| MemoryIndex::from_records(records));
        let corpus_stats = SegmentCorpusStats::from_segments(&segments);
        Self {
            segments,
            global_index,
            corpus_stats,
        }
    }

    pub fn from_records_by_group_id(records: &[DocRecord]) -> Self {
        Self::from_records_by_group_id_with_global_index(
            records,
            MemoryIndex::from_records(records.to_vec()),
        )
    }

    pub fn from_records_by_group_id_with_global_index(
        records: &[DocRecord],
        global_index: MemoryIndex,
    ) -> Self {
        let segments = build_segments_by_group_id(records);
        let corpus_stats = SegmentCorpusStats::from_segments(&segments);
        Self {
            segments,
            global_index: Some(global_index),
            corpus_stats,
        }
    }

    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn route(&self, query: &str) -> Vec<SegmentRoute> {
        route_segments_with_corpus_stats(
            query,
            &self.segments,
            SegmentRoutingStrategy::SparseOverlap,
            &self.corpus_stats,
        )
    }

    pub fn route_with_strategy(
        &self,
        query: &str,
        strategy: SegmentRoutingStrategy,
    ) -> Vec<SegmentRoute> {
        route_segments_with_corpus_stats(query, &self.segments, strategy, &self.corpus_stats)
    }

    pub fn route_with_temporal_context_and_strategy(
        &self,
        query: &str,
        strategy: SegmentRoutingStrategy,
        temporal: TemporalQueryContext<'_>,
    ) -> Vec<SegmentRoute> {
        route_segments_with_temporal_context_and_corpus_stats(
            query,
            &self.segments,
            strategy,
            temporal,
            &self.corpus_stats,
        )
    }

    pub fn query(&self, query: &str, top_k: usize, segment_limit: usize) -> Vec<SearchResult> {
        self.query_with_diagnostics(query, top_k, segment_limit)
            .results
    }

    pub fn query_with_diagnostics(
        &self,
        query: &str,
        top_k: usize,
        segment_limit: usize,
    ) -> SegmentQueryOutput {
        self.query_with_diagnostics_and_strategy(
            query,
            top_k,
            segment_limit,
            SegmentRoutingStrategy::SparseOverlap,
        )
    }

    pub fn query_with_diagnostics_and_strategy(
        &self,
        query: &str,
        top_k: usize,
        segment_limit: usize,
        strategy: SegmentRoutingStrategy,
    ) -> SegmentQueryOutput {
        query_top_segments_with_corpus_stats_and_strategy(
            query,
            top_k,
            &self.segments,
            segment_limit,
            strategy,
            TemporalQueryContext::default(),
            self.global_index.as_ref(),
            &self.corpus_stats,
        )
    }

    pub fn query_with_temporal_context_and_diagnostics_and_strategy(
        &self,
        query: &str,
        top_k: usize,
        segment_limit: usize,
        strategy: SegmentRoutingStrategy,
        temporal: TemporalQueryContext<'_>,
    ) -> SegmentQueryOutput {
        query_top_segments_with_corpus_stats_and_strategy(
            query,
            top_k,
            &self.segments,
            segment_limit,
            strategy,
            temporal,
            self.global_index.as_ref(),
            &self.corpus_stats,
        )
    }

    pub fn query_with_segment_enrichment_and_strategy(
        &self,
        query: &str,
        top_k: usize,
        segment_limit: usize,
        strategy: SegmentRoutingStrategy,
    ) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
        self.query_with_segment_enrichment_temporal_context_and_strategy(
            query,
            top_k,
            segment_limit,
            strategy,
            TemporalQueryContext::default(),
        )
    }

    pub fn query_with_segment_enrichment_temporal_context_and_strategy(
        &self,
        query: &str,
        top_k: usize,
        segment_limit: usize,
        strategy: SegmentRoutingStrategy,
        temporal: TemporalQueryContext<'_>,
    ) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
        query_top_segments_with_segment_enrichment_and_strategy(
            query,
            top_k,
            &self.segments,
            segment_limit,
            strategy,
            temporal,
            &self.corpus_stats,
        )
    }

    pub fn query_with_route_aware_segment_enrichment_temporal_context_and_strategy(
        &self,
        query: &str,
        top_k: usize,
        segment_limit: usize,
        strategy: SegmentRoutingStrategy,
        temporal: TemporalQueryContext<'_>,
    ) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
        query_top_segments_with_route_aware_segment_enrichment_and_strategy(
            query,
            top_k,
            &self.segments,
            segment_limit,
            strategy,
            temporal,
            &self.corpus_stats,
        )
    }

    pub fn query_with_session_aggregated_segment_enrichment_temporal_context_and_strategy(
        &self,
        query: &str,
        top_k: usize,
        segment_limit: usize,
        strategy: SegmentRoutingStrategy,
        temporal: TemporalQueryContext<'_>,
    ) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
        query_top_segments_with_session_aggregated_segment_enrichment_and_strategy(
            query,
            top_k,
            &self.segments,
            segment_limit,
            strategy,
            temporal,
            &self.corpus_stats,
        )
    }

    pub fn query_with_adaptive_segment_enrichment_temporal_context_and_strategy(
        &self,
        query: &str,
        top_k: usize,
        base_segment_limit: usize,
        max_segment_limit: usize,
        strategy: SegmentRoutingStrategy,
        temporal: TemporalQueryContext<'_>,
    ) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
        query_top_segments_with_adaptive_segment_enrichment_and_strategy(
            query,
            top_k,
            &self.segments,
            base_segment_limit,
            max_segment_limit,
            strategy,
            temporal,
            &self.corpus_stats,
        )
    }

    pub fn query_with_adaptive_route_aware_segment_enrichment_temporal_context_and_strategy(
        &self,
        query: &str,
        top_k: usize,
        base_segment_limit: usize,
        max_segment_limit: usize,
        strategy: SegmentRoutingStrategy,
        temporal: TemporalQueryContext<'_>,
    ) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
        query_top_segments_with_adaptive_route_aware_segment_enrichment_and_strategy(
            query,
            top_k,
            &self.segments,
            base_segment_limit,
            max_segment_limit,
            strategy,
            temporal,
            &self.corpus_stats,
        )
    }

    pub fn query_with_temporal_path_enrichment_and_strategy(
        &self,
        query: &str,
        top_k: usize,
        segment_limit: usize,
        strategy: SegmentRoutingStrategy,
        temporal: TemporalQueryContext<'_>,
    ) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
        query_top_segments_with_temporal_path_enrichment_and_strategy(
            query,
            top_k,
            &self.segments,
            segment_limit,
            strategy,
            temporal,
            &self.corpus_stats,
        )
    }

    pub fn query_with_connected_segment_enrichment_and_strategy(
        &self,
        query: &str,
        top_k: usize,
        segment_limit: usize,
        strategy: SegmentRoutingStrategy,
        temporal: TemporalQueryContext<'_>,
    ) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
        query_top_segments_with_connected_segment_enrichment_and_strategy(
            query,
            top_k,
            &self.segments,
            segment_limit,
            strategy,
            temporal,
            &self.corpus_stats,
        )
    }

    pub fn query_with_missing_coverage_recovery_segment_enrichment_and_strategy(
        &self,
        query: &str,
        top_k: usize,
        segment_limit: usize,
        strategy: SegmentRoutingStrategy,
        temporal: TemporalQueryContext<'_>,
    ) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
        query_top_segments_with_missing_coverage_recovery_segment_enrichment_and_strategy(
            query,
            top_k,
            &self.segments,
            segment_limit,
            strategy,
            temporal,
            &self.corpus_stats,
        )
    }

    pub fn query_all_segments(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        self.query_all_segments_with_diagnostics(query, top_k)
            .results
    }

    pub fn query_all_segments_with_diagnostics(
        &self,
        query: &str,
        top_k: usize,
    ) -> SegmentQueryOutput {
        self.query_all_segments_with_temporal_context_and_diagnostics(
            query,
            top_k,
            TemporalQueryContext::default(),
        )
    }

    pub fn query_all_segments_with_temporal_context_and_diagnostics(
        &self,
        query: &str,
        top_k: usize,
        temporal: TemporalQueryContext<'_>,
    ) -> SegmentQueryOutput {
        query_top_segments_with_corpus_stats_and_strategy(
            query,
            top_k,
            &self.segments,
            self.segments.len(),
            SegmentRoutingStrategy::SparseOverlap,
            temporal,
            self.global_index.as_ref(),
            &self.corpus_stats,
        )
    }
}

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

pub fn build_segments_by_group_id(records: &[DocRecord]) -> Vec<MemoryIndexSegment> {
    let mut grouped: HashMap<String, Vec<DocRecord>> = HashMap::new();
    for record in records {
        let segment_id = record
            .group_id
            .clone()
            .unwrap_or_else(|| "ungrouped".to_string());
        grouped.entry(segment_id).or_default().push(record.clone());
    }

    let mut segments = grouped
        .into_iter()
        .map(|(segment_id, mut segment_records)| {
            segment_records.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
            let doc_ids = segment_records
                .iter()
                .map(|record| record.doc_id.clone())
                .collect::<Vec<_>>();
            let profile = SegmentProfile::from_records(&segment_records);
            let index = MemoryIndex::from_records(segment_records);
            MemoryIndexSegment {
                segment_id,
                doc_ids,
                profile,
                index,
            }
        })
        .collect::<Vec<_>>();
    segments.sort_by(|a, b| a.segment_id.cmp(&b.segment_id));
    segments
}

pub fn route_segments(query: &str, segments: &[MemoryIndexSegment]) -> Vec<SegmentRoute> {
    route_segments_with_strategy(query, segments, SegmentRoutingStrategy::SparseOverlap)
}

pub fn route_segments_with_strategy(
    query: &str,
    segments: &[MemoryIndexSegment],
    strategy: SegmentRoutingStrategy,
) -> Vec<SegmentRoute> {
    let corpus_stats = SegmentCorpusStats::from_segments(segments);
    route_segments_with_corpus_stats(query, segments, strategy, &corpus_stats)
}

fn route_segments_with_corpus_stats(
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
        return route_segments_by_typed_evidence(query, &query_terms, segments, corpus_stats);
    }

    let mut routes = segments
        .iter()
        .map(|segment| SegmentRoute {
            segment_id: segment.segment_id.clone(),
            score: segment.profile.score_query_terms_with_strategy(
                &query_terms,
                strategy,
                corpus_stats,
            ),
            fallback: false,
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

fn route_segments_by_typed_evidence(
    query: &str,
    query_terms: &HashSet<String>,
    segments: &[MemoryIndexSegment],
    corpus_stats: &SegmentCorpusStats,
) -> Vec<SegmentRoute> {
    let query_profile = query_connection_profile(query);
    let mut routes = segments
        .iter()
        .map(|segment| {
            let base_score = segment
                .profile
                .coverage_local_distinctiveness_score(query_terms, corpus_stats);
            let segment_profile =
                segment_connection_profile(segment, TemporalQueryContext::default());
            let typed_score = typed_evidence_route_score(&query_profile, &segment_profile);
            SegmentRoute {
                segment_id: segment.segment_id.clone(),
                score: base_score + typed_score * TYPED_EVIDENCE_ROUTE_WEIGHT,
                fallback: false,
            }
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
    let mut selected = Vec::with_capacity(segments.len());
    let mut selected_segment_ids = HashSet::new();
    let mut covered_terms = HashSet::new();
    let base_scores = segments
        .iter()
        .map(|segment| {
            (
                segment.segment_id.as_str(),
                segment
                    .profile
                    .coverage_local_distinctiveness_score(query_terms, corpus_stats),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut base_order = segments.iter().collect::<Vec<_>>();
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

    while selected.len() < segments.len() {
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
                let marginal_score =
                    segment
                        .profile
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
            if segment.profile.covers_term(term) {
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

fn route_segments_by_coverage_team_selection(
    query_terms: &HashSet<String>,
    segments: &[MemoryIndexSegment],
    corpus_stats: &SegmentCorpusStats,
) -> Vec<SegmentRoute> {
    let mut selected = Vec::with_capacity(segments.len());
    let mut selected_segment_ids = HashSet::new();
    let mut coverage = TeamCoverageState::default();
    let base_scores = segments
        .iter()
        .map(|segment| {
            (
                segment.segment_id.as_str(),
                segment
                    .profile
                    .coverage_local_distinctiveness_score(query_terms, corpus_stats),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut base_order = segments.iter().collect::<Vec<_>>();
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

    while selected.len() < segments.len() {
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
        coverage.add_segment(segment, query_terms);
        selected.push(SegmentRoute {
            segment_id: segment.segment_id.clone(),
            score,
            fallback: false,
        });
    }

    selected
}

#[derive(Debug, Default)]
struct TeamCoverageState {
    covered_terms: HashSet<String>,
    covered_evidence_keys: HashSet<String>,
    covered_connection_terms: HashSet<String>,
    has_temporal_signal: bool,
}

impl TeamCoverageState {
    fn add_segment(&mut self, segment: &MemoryIndexSegment, query_terms: &HashSet<String>) {
        for term in query_terms {
            if segment.profile.covers_term(term) {
                self.covered_terms.insert(term.clone());
                for evidence_type in segment.profile.evidence_types_for_term(term) {
                    self.covered_evidence_keys
                        .insert(format!("{evidence_type}:{term}"));
                }
            }
        }
        self.covered_connection_terms
            .extend(segment.connection_terms());
        self.has_temporal_signal |= segment.has_temporal_signal();
    }
}

fn route_segments_with_temporal_context_and_corpus_stats(
    query: &str,
    segments: &[MemoryIndexSegment],
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    corpus_stats: &SegmentCorpusStats,
) -> Vec<SegmentRoute> {
    let mut routes = route_segments_with_corpus_stats(query, segments, strategy, corpus_stats)
        .into_iter()
        .map(|mut route| {
            if let Some(segment) = segments
                .iter()
                .find(|segment| segment.segment_id == route.segment_id)
            {
                route.score += segment_temporal_route_boost(segment, temporal);
            }
            route
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

pub fn query_top_segment(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
) -> Vec<SearchResult> {
    query_top_segments(query, top_k, segments, 1)
}

pub fn query_top_segments(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
) -> Vec<SearchResult> {
    query_top_segments_with_diagnostics(query, top_k, segments, segment_limit).results
}

pub fn query_all_segments(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
) -> Vec<SearchResult> {
    query_top_segments(query, top_k, segments, segments.len())
}

pub fn query_top_segments_with_diagnostics(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
) -> SegmentQueryOutput {
    let corpus_stats = SegmentCorpusStats::from_segments(segments);
    query_top_segments_with_corpus_stats_and_strategy(
        query,
        top_k,
        segments,
        segment_limit,
        SegmentRoutingStrategy::SparseOverlap,
        TemporalQueryContext::default(),
        None,
        &corpus_stats,
    )
}

pub fn query_top_segments_with_diagnostics_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    strategy: SegmentRoutingStrategy,
) -> SegmentQueryOutput {
    let corpus_stats = SegmentCorpusStats::from_segments(segments);
    query_top_segments_with_corpus_stats_and_strategy(
        query,
        top_k,
        segments,
        segment_limit,
        strategy,
        TemporalQueryContext::default(),
        None,
        &corpus_stats,
    )
}

fn query_top_segments_with_corpus_stats_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    global_index: Option<&MemoryIndex>,
    corpus_stats: &SegmentCorpusStats,
) -> SegmentQueryOutput {
    let query_terms = query_tokens(query);
    if top_k == 0 || segment_limit == 0 {
        return SegmentQueryOutput {
            results: Vec::new(),
            diagnostics: SegmentQueryDiagnostics {
                query_terms: sorted_terms(&query_terms),
                uncovered_query_terms: sorted_terms(&query_terms),
                ..SegmentQueryDiagnostics::default()
            },
        };
    }

    let routes = route_segments_with_corpus_stats(query, segments, strategy, corpus_stats);
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
    let fallback_segments = routes
        .iter()
        .filter(|route| !route_has_signal(route, strategy, &query_terms))
        .take(segment_limit.saturating_sub(selected_segments.len()))
        .map(|route| SegmentRoute {
            segment_id: route.segment_id.clone(),
            score: route.score,
            fallback: true,
        })
        .collect::<Vec<_>>();
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
            segments
                .iter()
                .find(|segment| segment.segment_id == route.segment_id)
                .map(|segment| segment.local_evidence(&query_terms, corpus_stats))
        })
        .collect::<Vec<_>>();
    let mut merged = Vec::new();
    let mut per_segment_result_counts = HashMap::new();
    let mut queried_segment_count = 0usize;
    let mut covered_query_terms = HashSet::new();
    let mut selected_doc_ids = HashSet::new();
    for route in selected_segments.iter() {
        let Some(segment) = segments
            .iter()
            .find(|segment| segment.segment_id == route.segment_id)
        else {
            continue;
        };
        queried_segment_count += 1;
        for term in &query_terms {
            if segment.profile.covers_term(term) {
                covered_query_terms.insert(term.clone());
            }
        }
        for doc_id in &segment.doc_ids {
            selected_doc_ids.insert(doc_id.clone());
        }
        per_segment_result_counts.insert(segment.segment_id.clone(), 0);
    }

    if !selected_doc_ids.is_empty() {
        if let Some(global_index) = global_index {
            let allowed_doc_ids =
                intersect_allowed_doc_ids(&selected_doc_ids, temporal.allowed_doc_ids);
            let scoped_temporal = TemporalQueryContext {
                allowed_doc_ids: Some(&allowed_doc_ids),
                ..temporal
            };
            merged = global_index
                .query_with_temporal_context(query, top_k, scoped_temporal)
                .0;
        } else if let Some(global_index) = build_global_index_from_segments(segments) {
            let allowed_doc_ids =
                intersect_allowed_doc_ids(&selected_doc_ids, temporal.allowed_doc_ids);
            let scoped_temporal = TemporalQueryContext {
                allowed_doc_ids: Some(&allowed_doc_ids),
                ..temporal
            };
            merged = global_index
                .query_with_temporal_context(query, top_k, scoped_temporal)
                .0;
        }
    }

    let selected_segment_ids = selected_segments
        .iter()
        .map(|route| route.segment_id.as_str())
        .collect::<HashSet<_>>();
    let doc_id_to_segment_id = segments
        .iter()
        .filter(|segment| selected_segment_ids.contains(segment.segment_id.as_str()))
        .flat_map(|segment| {
            segment
                .doc_ids
                .iter()
                .map(|doc_id| (doc_id.as_str(), segment.segment_id.as_str()))
        })
        .collect::<HashMap<_, _>>();
    for result in &merged {
        if let Some(segment_id) = doc_id_to_segment_id.get(result.doc_id.as_str()) {
            *per_segment_result_counts
                .entry((*segment_id).to_string())
                .or_default() += 1;
        }
    }
    let segments_with_results = selected_segments
        .iter()
        .filter_map(|route| {
            per_segment_result_counts
                .get(&route.segment_id)
                .copied()
                .filter(|count| *count > 0)
                .map(|_| route.segment_id.clone())
        })
        .collect::<Vec<_>>();
    let merged_result_count = merged.len();
    let final_result_count = merged.len();
    let uncovered_query_terms = query_terms
        .difference(&covered_query_terms)
        .cloned()
        .collect::<HashSet<_>>();
    SegmentQueryOutput {
        results: merged,
        diagnostics: SegmentQueryDiagnostics {
            selected_segments,
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
        },
    }
}

fn build_global_index_from_segments(segments: &[MemoryIndexSegment]) -> Option<MemoryIndex> {
    let records = segments
        .iter()
        .flat_map(|segment| segment.index.docs.values().cloned())
        .collect::<Vec<_>>();
    (!records.is_empty()).then(|| MemoryIndex::from_records(records))
}

fn intersect_allowed_doc_ids(
    selected_doc_ids: &HashSet<String>,
    existing_allowed_doc_ids: Option<&HashSet<String>>,
) -> HashSet<String> {
    match existing_allowed_doc_ids {
        Some(existing) => selected_doc_ids
            .intersection(existing)
            .cloned()
            .collect::<HashSet<_>>(),
        None => selected_doc_ids.clone(),
    }
}

fn query_top_segments_with_segment_enrichment_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    corpus_stats: &SegmentCorpusStats,
) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
    let query_terms = query_tokens(query);
    if top_k == 0 || segment_limit == 0 {
        return (
            SegmentQueryOutput {
                results: Vec::new(),
                diagnostics: SegmentQueryDiagnostics {
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
        Vec::new(),
        Vec::new(),
        false,
        false,
    )
}

fn query_top_segments_with_route_aware_segment_enrichment_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    corpus_stats: &SegmentCorpusStats,
) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
    let query_terms = query_tokens(query);
    if top_k == 0 || segment_limit == 0 {
        return (
            SegmentQueryOutput {
                results: Vec::new(),
                diagnostics: SegmentQueryDiagnostics {
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
        Vec::new(),
        Vec::new(),
        true,
        false,
    )
}

fn query_top_segments_with_session_aggregated_segment_enrichment_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    corpus_stats: &SegmentCorpusStats,
) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
    let query_terms = query_tokens(query);
    if top_k == 0 || segment_limit == 0 {
        return (
            SegmentQueryOutput {
                results: Vec::new(),
                diagnostics: SegmentQueryDiagnostics {
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
        Vec::new(),
        Vec::new(),
        false,
        true,
    )
}

fn query_top_segments_with_temporal_path_enrichment_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    corpus_stats: &SegmentCorpusStats,
) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
    let query_terms = query_tokens(query);
    if top_k == 0 || segment_limit == 0 {
        return (
            SegmentQueryOutput {
                results: Vec::new(),
                diagnostics: SegmentQueryDiagnostics {
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
        temporal_expanded_segments,
        Vec::new(),
        false,
        false,
    )
}

fn query_top_segments_with_connected_segment_enrichment_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    corpus_stats: &SegmentCorpusStats,
) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
    let query_terms = query_tokens(query);
    if top_k == 0 || segment_limit == 0 {
        return (
            SegmentQueryOutput {
                results: Vec::new(),
                diagnostics: SegmentQueryDiagnostics {
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
    let (expanded_segments, connected_expanded_segments) =
        expand_connected_segments(&routed_segments, &routes, segments, segment_limit, temporal);
    query_selected_segments_with_enrichment(
        query,
        top_k,
        segments,
        expanded_segments,
        temporal,
        Vec::new(),
        connected_expanded_segments,
        false,
        true,
    )
}

fn query_top_segments_with_missing_coverage_recovery_segment_enrichment_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    corpus_stats: &SegmentCorpusStats,
) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
    let query_terms = query_tokens(query);
    if top_k == 0 || segment_limit == 0 {
        return (
            SegmentQueryOutput {
                results: Vec::new(),
                diagnostics: SegmentQueryDiagnostics {
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
    let (recovered_segments, recovery_events) =
        recover_missing_coverage_segments(&query_terms, &routed_segments, &routes, segments);
    query_selected_segments_with_enrichment(
        query,
        top_k,
        segments,
        recovered_segments,
        temporal,
        Vec::new(),
        recovery_events,
        false,
        false,
    )
}

fn query_top_segments_with_adaptive_segment_enrichment_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    base_segment_limit: usize,
    max_segment_limit: usize,
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    corpus_stats: &SegmentCorpusStats,
) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
    let query_terms = query_tokens(query);
    if top_k == 0 || base_segment_limit == 0 || max_segment_limit == 0 {
        return (
            SegmentQueryOutput {
                results: Vec::new(),
                diagnostics: SegmentQueryDiagnostics {
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
        base_segment_limit,
        max_segment_limit,
    );
    query_selected_segments_with_enrichment(
        query,
        top_k,
        segments,
        selected_segments,
        temporal,
        Vec::new(),
        Vec::new(),
        false,
        false,
    )
}

fn query_top_segments_with_adaptive_route_aware_segment_enrichment_and_strategy(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    base_segment_limit: usize,
    max_segment_limit: usize,
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    corpus_stats: &SegmentCorpusStats,
) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
    let query_terms = query_tokens(query);
    if top_k == 0 || base_segment_limit == 0 || max_segment_limit == 0 {
        return (
            SegmentQueryOutput {
                results: Vec::new(),
                diagnostics: SegmentQueryDiagnostics {
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
        base_segment_limit,
        max_segment_limit,
    );
    query_selected_segments_with_enrichment(
        query,
        top_k,
        segments,
        selected_segments,
        temporal,
        Vec::new(),
        Vec::new(),
        true,
        false,
    )
}

fn adaptive_segment_routes(
    query_terms: &HashSet<String>,
    routes: &[SegmentRoute],
    segments: &[MemoryIndexSegment],
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
        let covered_terms = selected_query_terms(query_terms, &selected, segments);
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
                    .filter(|term| segment.profile.covers_term(term))
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

fn recover_missing_coverage_segments(
    query_terms: &HashSet<String>,
    selected_routes: &[SegmentRoute],
    routes: &[SegmentRoute],
    segments: &[MemoryIndexSegment],
) -> (Vec<SegmentRoute>, Vec<ConnectedSegmentExpansion>) {
    if query_terms.is_empty() || selected_routes.is_empty() {
        return (selected_routes.to_vec(), Vec::new());
    }

    let corpus_stats = SegmentCorpusStats::from_segments(segments);
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

    let covered_terms = selected_query_terms(query_terms, &selected, segments);
    let missing_terms = query_terms
        .difference(&covered_terms)
        .cloned()
        .collect::<HashSet<_>>();
    let weakest = selected
        .iter()
        .enumerate()
        .filter_map(|(idx, route)| {
            let segment = segment_by_id.get(route.segment_id.as_str()).copied()?;
            let score = weak_segment_score(query_terms, route, segment, &selected, segments);
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
                                &corpus_stats,
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
                let shared_subjects =
                    sorted_missing_terms_covered(&missing_terms, &replacement, segments);
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
        if !segment.profile.covers_term(term) {
            continue;
        }
        covered_missing += 1;
        let idf = corpus_stats.idf(term);
        gain += segment.profile.local_term_weight(term)
            * idf
            * segment.profile.coverage_evidence_multiplier(term);
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
) -> f32 {
    let covered_terms = query_terms
        .iter()
        .filter(|term| segment.profile.covers_term(term))
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
                .filter(|term| segment.profile.covers_term(term))
                .cloned()
                .collect::<Vec<_>>()
        })
        .collect::<HashSet<_>>();
    let unique_coverage = covered_terms.difference(&other_covered_terms).count();
    let evidence_count = covered_terms
        .iter()
        .map(|term| segment.profile.evidence_types_for_term(term).len())
        .sum::<usize>();
    let local_score = covered_terms
        .iter()
        .map(|term| segment.profile.local_term_weight(term))
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
) -> Vec<String> {
    let Some(segment) = segments
        .iter()
        .find(|segment| segment.segment_id == route.segment_id)
    else {
        return Vec::new();
    };
    let mut terms = missing_terms
        .iter()
        .filter(|term| segment.profile.covers_term(term))
        .cloned()
        .collect::<Vec<_>>();
    terms.sort();
    terms
}

fn selected_query_terms(
    query_terms: &HashSet<String>,
    selected_routes: &[SegmentRoute],
    segments: &[MemoryIndexSegment],
) -> HashSet<String> {
    let selected_ids = selected_routes
        .iter()
        .map(|route| route.segment_id.as_str())
        .collect::<HashSet<_>>();
    let mut covered = HashSet::new();
    for segment in segments {
        if !selected_ids.contains(segment.segment_id.as_str()) {
            continue;
        }
        for term in query_terms {
            if segment.profile.covers_term(term) {
                covered.insert(term.clone());
            }
        }
    }
    covered
}

fn route_score_is_close(candidate_score: f32, cutoff_score: f32) -> bool {
    if cutoff_score > 0.0 {
        candidate_score >= cutoff_score * ADAPTIVE_CLOSE_SCORE_RATIO
    } else {
        (candidate_score - cutoff_score).abs() <= 0.05
    }
}

fn query_selected_segments_with_enrichment(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    selected_segments: Vec<SegmentRoute>,
    temporal: TemporalQueryContext<'_>,
    temporal_expanded_segments: Vec<TemporalSegmentExpansion>,
    connected_expanded_segments: Vec<ConnectedSegmentExpansion>,
    route_aware_rerank: bool,
    session_aggregate: bool,
) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
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
    let max_route_score = selected_segments
        .iter()
        .map(|route| route.score.max(0.0))
        .fold(0.0f32, f32::max);

    for route in &selected_segments {
        let Some(segment) = segments
            .iter()
            .find(|segment| segment.segment_id == route.segment_id)
        else {
            continue;
        };
        queried_segment_count += 1;
        for term in &query_terms {
            if segment.profile.covers_term(term) {
                covered_query_terms.insert(term.clone());
            }
        }
        let enrichment = segment.enriched_query(query, &query_terms, temporal);
        let segment_query_results = segment
            .index
            .query_with_temporal_context(&enrichment.enriched_query, top_k, temporal)
            .0;
        let max_segment_result_score = segment_query_results
            .iter()
            .map(|result| result.score.max(0.0))
            .fold(0.0f32, f32::max);
        let mut segment_results = 0usize;
        for mut result in segment_query_results {
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
            enriched_query: enrichment.enriched_query,
            added_terms: enrichment.added_terms,
            evidence_types: enrichment.evidence_types,
            temporal_added_terms: enrichment.temporal_added_terms,
            temporal_evidence: enrichment.temporal_evidence,
            temporal_signal: enrichment.temporal_signal,
        });
    }

    if route_aware_rerank {
        merged = select_route_aware_top_k(rerank_candidates, top_k);
    } else if session_aggregate {
        merged = aggregate_segment_results_by_session(merged);
    } else {
        merged.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
    }
    let merged_result_count = merged.len();
    merged.truncate(top_k);
    let final_result_count = merged.len();
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

fn aggregate_segment_results_by_session(results: Vec<SearchResult>) -> Vec<SearchResult> {
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

#[derive(Debug, Clone)]
struct RouteAwareCandidate {
    result: SearchResult,
    segment_id: String,
    base_score: f32,
    evidence_terms: HashSet<String>,
}

impl RouteAwareCandidate {
    fn new(
        mut result: SearchResult,
        segment: &MemoryIndexSegment,
        route: &SegmentRoute,
        query_terms: &HashSet<String>,
        enrichment: &SegmentQueryEnrichment,
        temporal: TemporalQueryContext<'_>,
        max_segment_result_score: f32,
        max_route_score: f32,
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
        let local_evidence_score = result_terms
            .iter()
            .map(|term| {
                let local_weight = segment.profile.local_term_weight(term);
                if local_weight <= 0.0 {
                    0.0
                } else {
                    local_weight * segment.profile.coverage_evidence_multiplier(term)
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

fn select_route_aware_top_k(
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

fn normalize_positive_score(score: f32, max_score: f32) -> f32 {
    if max_score > 0.0 {
        (score.max(0.0) / max_score).clamp(0.0, 1.0)
    } else {
        score.max(0.0)
    }
}

fn result_evidence_terms(result: &SearchResult, segment: &MemoryIndexSegment) -> HashSet<String> {
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

fn result_temporal_score(
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

#[derive(Debug, Clone, Default)]
struct SegmentConnectionProfile {
    people: HashSet<String>,
    subjects: HashSet<String>,
    times: HashSet<String>,
    actions: HashSet<String>,
    objects: HashSet<String>,
}

#[derive(Debug, Clone)]
struct SegmentConnectionMatch {
    score: f32,
    shared_people: Vec<String>,
    shared_subjects: Vec<String>,
    shared_time: Vec<String>,
    shared_actions: Vec<String>,
    shared_objects: Vec<String>,
}

fn query_connection_profile(query: &str) -> SegmentConnectionProfile {
    let mut profile = SegmentConnectionProfile::default();
    for raw_token in query.split(|ch: char| !ch.is_alphanumeric()) {
        if raw_token.len() <= 1 {
            continue;
        }
        let token = normalize_for_index(raw_token);
        if token.len() <= 1 || is_routing_stopword(&token) {
            continue;
        }
        if raw_token
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_uppercase())
            && !looks_like_time_token(&token)
        {
            profile.people.insert(token.clone());
        }
        if looks_like_time_token(&token) || raw_token.chars().any(|ch| ch.is_ascii_digit()) {
            profile.times.insert(token);
        } else if looks_like_action(&token) {
            profile.actions.insert(token);
        } else {
            profile.subjects.insert(token.clone());
            profile.objects.insert(token);
        }
    }
    profile
}

fn typed_evidence_route_score(
    query: &SegmentConnectionProfile,
    segment: &SegmentConnectionProfile,
) -> f32 {
    let shared_people = sorted_set_intersection(&query.people, &segment.people, 8);
    let shared_subjects = sorted_set_intersection(&query.subjects, &segment.subjects, 8);
    let shared_time = sorted_set_intersection(&query.times, &segment.times, 8);
    let shared_actions = sorted_set_intersection(&query.actions, &segment.actions, 8);
    let shared_objects = sorted_set_intersection(&query.objects, &segment.objects, 8);
    let bucket_count = [
        !shared_people.is_empty(),
        !shared_subjects.is_empty(),
        !shared_time.is_empty(),
        !shared_actions.is_empty(),
        !shared_objects.is_empty(),
    ]
    .into_iter()
    .filter(|present| *present)
    .count();

    let has_person_pair = !shared_people.is_empty() && bucket_count >= 2;
    let has_time_pair = !shared_time.is_empty() && bucket_count >= 2;
    let has_subject_action_pair = !shared_subjects.is_empty() && !shared_actions.is_empty();
    let has_subject_object_pair = !shared_subjects.is_empty() && !shared_objects.is_empty();
    let has_action_object_pair = !shared_actions.is_empty() && !shared_objects.is_empty();
    if bucket_count < 2
        || !(has_person_pair
            || has_time_pair
            || has_subject_action_pair
            || has_subject_object_pair
            || has_action_object_pair)
    {
        return 0.0;
    }

    let bucket_score = shared_people.len().min(3) as f32 * 1.4
        + shared_subjects.len().min(4) as f32 * 0.75
        + shared_time.len().min(2) as f32 * 1.0
        + shared_actions.len().min(3) as f32 * 0.7
        + shared_objects.len().min(4) as f32 * 0.55;
    let pair_score = if has_person_pair { 1.0 } else { 0.0 }
        + if has_time_pair { 0.8 } else { 0.0 }
        + if has_subject_action_pair { 0.75 } else { 0.0 }
        + if has_subject_object_pair { 0.55 } else { 0.0 }
        + if has_action_object_pair { 0.9 } else { 0.0 };
    bucket_score + pair_score + (bucket_count.saturating_sub(1) as f32 * 0.35)
}

fn expand_connected_segments(
    routed_segments: &[SegmentRoute],
    routes: &[SegmentRoute],
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    temporal: TemporalQueryContext<'_>,
) -> (Vec<SegmentRoute>, Vec<ConnectedSegmentExpansion>) {
    if routed_segments.is_empty() || segment_limit == 0 {
        return (routed_segments.to_vec(), Vec::new());
    }

    let profiles = segments
        .iter()
        .map(|segment| {
            (
                segment.segment_id.as_str(),
                segment_connection_profile(segment, temporal),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut selected = routed_segments.to_vec();
    let mut selected_ids = selected
        .iter()
        .map(|route| route.segment_id.clone())
        .collect::<HashSet<_>>();
    let candidate_pool_limit = segment_limit
        .saturating_mul(CONNECTED_EXPANSION_POOL_MULTIPLIER)
        .max(segment_limit + 5)
        .min(routes.len());
    let mut expansions = Vec::new();
    let mut candidates = routes
        .iter()
        .take(candidate_pool_limit)
        .filter(|route| !selected_ids.contains(route.segment_id.as_str()))
        .filter_map(|route| {
            let candidate_profile = profiles.get(route.segment_id.as_str())?;
            let best = selected
                .iter()
                .filter_map(|selected_route| {
                    let selected_profile = profiles.get(selected_route.segment_id.as_str())?;
                    connected_expansion_evidence(
                        route,
                        selected_route,
                        candidate_profile,
                        selected_profile,
                    )
                })
                .max_by(|left, right| {
                    left.score
                        .partial_cmp(&right.score)
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| right.segment_id.cmp(&left.segment_id))
                })?;
            (best.score >= CONNECTED_EXPANSION_MIN_SCORE).then_some((route.clone(), best))
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|a, b| {
        b.1.score
            .partial_cmp(&a.1.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.0.score.partial_cmp(&a.0.score).unwrap_or(Ordering::Equal))
            .then_with(|| a.0.segment_id.cmp(&b.0.segment_id))
    });

    for (mut route, mut expansion) in candidates {
        if selected_ids.contains(route.segment_id.as_str()) {
            continue;
        }
        route.score += expansion.score * 0.1;
        if selected.len() < segment_limit {
            selected_ids.insert(route.segment_id.clone());
            expansion.action = "added".to_string();
            selected.push(route);
            expansions.push(expansion);
            continue;
        }

        let Some(replace_idx) = weakest_connected_expansion_swap_index(&selected, &profiles) else {
            continue;
        };
        let replace_score = selected[replace_idx].score;
        if route.score + CONNECTED_EXPANSION_MAX_SWAP_PENALTY < replace_score
            && expansion.score < CONNECTED_EXPANSION_MIN_SCORE * 1.8
        {
            continue;
        }
        let replaced = selected.swap_remove(replace_idx);
        selected_ids.remove(replaced.segment_id.as_str());
        selected_ids.insert(route.segment_id.clone());
        expansion.action = format!("swapped_out:{}", replaced.segment_id);
        selected.push(route);
        expansions.push(expansion);
        break;
    }

    selected.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.segment_id.cmp(&b.segment_id))
    });

    (selected, expansions)
}

fn weakest_connected_expansion_swap_index(
    selected: &[SegmentRoute],
    profiles: &HashMap<&str, SegmentConnectionProfile>,
) -> Option<usize> {
    selected
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            let left_profile = profiles.get(left.segment_id.as_str());
            let right_profile = profiles.get(right.segment_id.as_str());
            let left_support = selected_connection_support(left, selected, left_profile, profiles);
            let right_support =
                selected_connection_support(right, selected, right_profile, profiles);
            left_support
                .partial_cmp(&right_support)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    left.score
                        .partial_cmp(&right.score)
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| right.segment_id.cmp(&left.segment_id))
        })
        .map(|(idx, _)| idx)
}

fn selected_connection_support(
    route: &SegmentRoute,
    selected: &[SegmentRoute],
    profile: Option<&SegmentConnectionProfile>,
    profiles: &HashMap<&str, SegmentConnectionProfile>,
) -> f32 {
    let Some(profile) = profile else {
        return 0.0;
    };
    selected
        .iter()
        .filter(|other| other.segment_id != route.segment_id)
        .filter_map(|other| {
            profiles
                .get(other.segment_id.as_str())
                .and_then(|other_profile| connection_match(profile, other_profile))
                .map(|connection| connection.score)
        })
        .fold(0.0f32, f32::max)
}

fn connected_expansion_evidence(
    route: &SegmentRoute,
    selected_route: &SegmentRoute,
    candidate: &SegmentConnectionProfile,
    selected: &SegmentConnectionProfile,
) -> Option<ConnectedSegmentExpansion> {
    let connection = connection_match(candidate, selected)?;
    let mut connection_types = Vec::new();
    if !connection.shared_people.is_empty() {
        connection_types.push("person".to_string());
    }
    if !connection.shared_subjects.is_empty() {
        connection_types.push("subject".to_string());
    }
    if !connection.shared_time.is_empty() {
        connection_types.push("time".to_string());
    }
    if !connection.shared_actions.is_empty() {
        connection_types.push("action".to_string());
    }
    if !connection.shared_objects.is_empty() {
        connection_types.push("object".to_string());
    }

    Some(ConnectedSegmentExpansion {
        segment_id: route.segment_id.clone(),
        source_segment_id: selected_route.segment_id.clone(),
        score: connection.score,
        connection_types,
        shared_people: connection.shared_people,
        shared_subjects: connection.shared_subjects,
        shared_time: connection.shared_time,
        shared_actions: connection.shared_actions,
        shared_objects: connection.shared_objects,
        action: String::new(),
    })
}

fn connection_match(
    left: &SegmentConnectionProfile,
    right: &SegmentConnectionProfile,
) -> Option<SegmentConnectionMatch> {
    let shared_people = sorted_set_intersection(&left.people, &right.people, 8);
    let shared_subjects = sorted_set_intersection(&left.subjects, &right.subjects, 8);
    let shared_time = sorted_set_intersection(&left.times, &right.times, 8);
    let shared_actions = sorted_set_intersection(&left.actions, &right.actions, 8);
    let shared_objects = sorted_set_intersection(&left.objects, &right.objects, 8);
    let evidence_type_count = [
        !shared_people.is_empty(),
        !shared_subjects.is_empty(),
        !shared_time.is_empty(),
        !shared_actions.is_empty(),
        !shared_objects.is_empty(),
    ]
    .into_iter()
    .filter(|present| *present)
    .count();

    let has_person_or_time_pair =
        (!shared_people.is_empty() || !shared_time.is_empty()) && evidence_type_count >= 2;
    let has_strong_pair = has_person_or_time_pair
        || (!shared_subjects.is_empty() && !shared_objects.is_empty())
        || (!shared_actions.is_empty() && !shared_objects.is_empty());
    if evidence_type_count < 2 || !has_strong_pair {
        return None;
    }

    let score = shared_people.len().min(3) as f32 * 1.4
        + shared_subjects.len().min(4) as f32 * 0.9
        + shared_time.len().min(2) as f32 * 1.1
        + shared_actions.len().min(3) as f32 * 0.6
        + shared_objects.len().min(4) as f32 * 0.5;
    Some(SegmentConnectionMatch {
        score,
        shared_people,
        shared_subjects,
        shared_time,
        shared_actions,
        shared_objects,
    })
}

fn segment_connection_profile(
    segment: &MemoryIndexSegment,
    temporal: TemporalQueryContext<'_>,
) -> SegmentConnectionProfile {
    let mut profile = SegmentConnectionProfile::default();
    for record in segment.index.docs.values() {
        for entity in &record.key_entities {
            let tokens = query_tokens(&entity.text);
            let label = entity.label.to_ascii_lowercase();
            if label.contains("person") || label == "per" {
                profile.people.extend(tokens.clone());
            }
            profile.subjects.extend(tokens);
        }
        if let Some(topic) = &record.probable_topic {
            profile.subjects.extend(query_tokens(topic));
        }
        for heading in &record.headings {
            profile.subjects.extend(query_tokens(heading));
        }
        for term in &record.important_terms {
            for token in query_tokens(&term.term) {
                if looks_like_action(&token) {
                    profile.actions.insert(token);
                } else if is_segment_enrichment_candidate(&token) {
                    profile.objects.insert(token);
                }
            }
        }
        for token in ordered_tokens(&record.content) {
            if looks_like_action(&token) {
                profile.actions.insert(token);
            }
        }
        if let Some(timestamp) = record.timestamp.as_deref().and_then(parse_iso_date) {
            profile.times.insert(timestamp.to_string());
            if let Some(query_date) = temporal.ends_at.and_then(parse_iso_date) {
                let days = timestamp.signed_duration_since(query_date).num_days().abs();
                if days <= temporal.window_days.max(1) {
                    profile.times.insert("near_query_date".to_string());
                }
            }
        }
        for temporal_term in &record.temporal_terms {
            profile.times.extend(query_tokens(temporal_term));
        }
    }
    retain_hashset(&mut profile.people, 24);
    retain_hashset(&mut profile.subjects, 48);
    retain_hashset(&mut profile.times, 24);
    retain_hashset(&mut profile.actions, 48);
    retain_hashset(&mut profile.objects, 48);
    profile
}

fn looks_like_action(token: &str) -> bool {
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

fn looks_like_time_token(token: &str) -> bool {
    matches!(
        token,
        "after"
            | "before"
            | "date"
            | "day"
            | "month"
            | "near"
            | "past"
            | "present"
            | "recent"
            | "today"
            | "tomorrow"
            | "week"
            | "yesterday"
            | "year"
    )
}

fn sorted_set_intersection(
    left: &HashSet<String>,
    right: &HashSet<String>,
    limit: usize,
) -> Vec<String> {
    let mut out = left.intersection(right).cloned().collect::<Vec<_>>();
    out.sort();
    out.truncate(limit);
    out
}

fn retain_hashset(values: &mut HashSet<String>, limit: usize) {
    if values.len() <= limit {
        return;
    }
    let mut sorted = values.iter().cloned().collect::<Vec<_>>();
    sorted.sort();
    sorted.truncate(limit);
    values.retain(|value| sorted.contains(value));
}

pub fn query_all_segments_with_diagnostics(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
) -> SegmentQueryOutput {
    query_top_segments_with_diagnostics(query, top_k, segments, segments.len())
}

impl SegmentProfile {
    pub fn from_records(records: &[DocRecord]) -> Self {
        let mut profile = Self::default();
        for record in records {
            for term in &record.important_terms {
                add_weight(&mut profile.terms, &term.term, term.score.max(0.1));
            }
            for entity in &record.key_entities {
                add_weight(
                    &mut profile.entities,
                    &entity.text,
                    entity.score.unwrap_or(1.0),
                );
            }
            if let Some(topic) = &record.probable_topic {
                add_weight(&mut profile.topics, topic, 1.0);
            }
            merge_distribution(
                &mut profile.local_memory,
                &gated_local_memory_profile(record),
                LOCAL_MEMORY_PROFILE_WEIGHT,
            );
        }
        normalize_distribution(&mut profile.terms);
        normalize_distribution(&mut profile.entities);
        normalize_distribution(&mut profile.topics);
        normalize_distribution(&mut profile.local_memory);
        retain_top_terms(&mut profile.local_memory, LOCAL_MEMORY_SEGMENT_TERM_LIMIT);
        normalize_distribution(&mut profile.local_memory);
        profile
    }

    fn score_query_terms_with_strategy(
        &self,
        query_terms: &HashSet<String>,
        strategy: SegmentRoutingStrategy,
        corpus_stats: &SegmentCorpusStats,
    ) -> f32 {
        match strategy {
            SegmentRoutingStrategy::SparseOverlap => self.sparse_overlap_score(query_terms),
            SegmentRoutingStrategy::KlDivergence => self.kl_divergence_score(query_terms),
            SegmentRoutingStrategy::LocalDistinctiveness => {
                self.local_distinctiveness_score(query_terms, corpus_stats)
            }
            SegmentRoutingStrategy::CoverageLocalDistinctiveness => {
                self.coverage_local_distinctiveness_score(query_terms, corpus_stats)
            }
            SegmentRoutingStrategy::TeamCoverageLocalDistinctiveness => {
                self.coverage_local_distinctiveness_score(query_terms, corpus_stats)
            }
            SegmentRoutingStrategy::CoverageTeamSelection => {
                self.coverage_local_distinctiveness_score(query_terms, corpus_stats)
            }
            SegmentRoutingStrategy::TypedEvidence => {
                self.coverage_local_distinctiveness_score(query_terms, corpus_stats)
            }
        }
    }

    fn sparse_overlap_score(&self, query_terms: &HashSet<String>) -> f32 {
        if query_terms.is_empty() {
            return 0.0;
        }
        let term_score = overlap_score(query_terms, &self.terms);
        let entity_score = overlap_score(query_terms, &self.entities);
        let topic_score = overlap_score(query_terms, &self.topics);
        term_score + (1.3 * entity_score) + (0.7 * topic_score)
    }

    fn kl_divergence_score(&self, query_terms: &HashSet<String>) -> f32 {
        if query_terms.is_empty() {
            return 0.0;
        }

        let segment_distribution = self.combined_distribution();
        if segment_distribution.is_empty() {
            return -f32::INFINITY;
        }

        let query_probability = 1.0 / query_terms.len() as f32;
        let divergence = query_terms
            .iter()
            .map(|term| {
                let segment_probability = segment_distribution
                    .get(term)
                    .copied()
                    .unwrap_or(KL_SMOOTHING)
                    .max(KL_SMOOTHING);
                query_probability * (query_probability / segment_probability).ln()
            })
            .sum::<f32>();

        -divergence
    }

    fn local_distinctiveness_score(
        &self,
        query_terms: &HashSet<String>,
        corpus_stats: &SegmentCorpusStats,
    ) -> f32 {
        if query_terms.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;
        for term in query_terms {
            let local_weight = self.local_term_weight(term);
            if local_weight <= 0.0 {
                continue;
            }
            score += local_weight * corpus_stats.idf(term);
        }
        score
    }

    fn coverage_local_distinctiveness_score(
        &self,
        query_terms: &HashSet<String>,
        corpus_stats: &SegmentCorpusStats,
    ) -> f32 {
        if query_terms.is_empty() {
            return 0.0;
        }

        let total_query_idf = query_terms
            .iter()
            .map(|term| corpus_stats.idf(term))
            .sum::<f32>()
            .max(f32::EPSILON);
        let mut weighted_local_score = 0.0;
        let mut covered_idf = 0.0;
        let mut covered_count = 0usize;

        for term in query_terms {
            let local_weight = self.local_term_weight(term);
            if local_weight <= 0.0 {
                continue;
            }

            let idf = corpus_stats.idf(term);
            let evidence_multiplier = self.coverage_evidence_multiplier(term);
            weighted_local_score += local_weight * idf * evidence_multiplier;
            covered_idf += idf;
            covered_count += 1;
        }

        if covered_count == 0 {
            return 0.0;
        }

        let rare_term_coverage = covered_idf / total_query_idf;
        let term_coverage = covered_count as f32 / query_terms.len() as f32;
        let multi_term_bonus = (covered_count.saturating_sub(1) as f32).sqrt() * 0.75;

        weighted_local_score * (1.0 + rare_term_coverage)
            + rare_term_coverage * 2.0
            + term_coverage
            + multi_term_bonus
    }

    fn team_coverage_gain(
        &self,
        query_terms: &HashSet<String>,
        corpus_stats: &SegmentCorpusStats,
        already_covered_terms: &HashSet<String>,
    ) -> f32 {
        if query_terms.is_empty() {
            return 0.0;
        }

        let total_query_idf = query_terms
            .iter()
            .map(|term| corpus_stats.idf(term))
            .sum::<f32>()
            .max(f32::EPSILON);
        let mut gain = 0.0;
        let mut newly_covered_idf = 0.0;
        let mut newly_covered_count = 0usize;

        for term in query_terms {
            if already_covered_terms.contains(term) {
                continue;
            }

            let local_weight = self.local_term_weight(term);
            if local_weight <= 0.0 {
                continue;
            }

            let idf = corpus_stats.idf(term);
            gain += local_weight * idf * self.coverage_evidence_multiplier(term);
            newly_covered_idf += idf;
            newly_covered_count += 1;
        }

        if newly_covered_count == 0 {
            return 0.0;
        }

        let rare_coverage_gain = newly_covered_idf / total_query_idf;
        let term_coverage_gain = newly_covered_count as f32 / query_terms.len() as f32;
        gain * (1.0 + rare_coverage_gain) + rare_coverage_gain * 2.5 + term_coverage_gain
    }

    fn combined_distribution(&self) -> HashMap<String, f32> {
        let mut combined = HashMap::new();
        merge_distribution(&mut combined, &self.terms, 1.0);
        merge_distribution(&mut combined, &self.entities, 1.3);
        merge_distribution(&mut combined, &self.topics, 0.7);
        merge_distribution(&mut combined, &self.local_memory, 0.5);
        normalize_distribution(&mut combined);
        combined
    }

    fn covers_term(&self, term: &str) -> bool {
        self.terms.contains_key(term)
            || self.entities.contains_key(term)
            || self.topics.contains_key(term)
            || self.local_memory.contains_key(term)
    }

    fn local_term_weight(&self, term: &str) -> f32 {
        let term_weight = self.terms.get(term).copied().unwrap_or_default();
        let entity_weight = self.entities.get(term).copied().unwrap_or_default() * 1.8;
        let topic_weight = self.topics.get(term).copied().unwrap_or_default() * 1.2;
        let local_memory_weight = self.local_memory.get(term).copied().unwrap_or_default() * 1.5;
        term_weight + entity_weight + topic_weight + local_memory_weight
    }

    fn coverage_evidence_multiplier(&self, term: &str) -> f32 {
        let mut multiplier = 1.0;
        if self.entities.contains_key(term) {
            multiplier += 0.45;
        }
        if self.local_memory.contains_key(term) {
            multiplier += 0.35;
        }
        if self.topics.contains_key(term) {
            multiplier += 0.15;
        }
        multiplier
    }

    fn evidence_types_for_term(&self, term: &str) -> Vec<String> {
        let mut evidence_types = Vec::new();
        if self.entities.contains_key(term) {
            evidence_types.push("entity".to_string());
        }
        if self.terms.contains_key(term) {
            evidence_types.push("term".to_string());
        }
        if self.topics.contains_key(term) {
            evidence_types.push("topic".to_string());
        }
        if self.local_memory.contains_key(term) {
            evidence_types.push("local_memory".to_string());
        }
        evidence_types
    }
}

impl MemoryIndexSegment {
    fn team_coverage_gain(
        &self,
        query_terms: &HashSet<String>,
        corpus_stats: &SegmentCorpusStats,
        coverage: &TeamCoverageState,
    ) -> f32 {
        if query_terms.is_empty() {
            return 0.0;
        }

        let total_query_idf = query_terms
            .iter()
            .map(|term| corpus_stats.idf(term))
            .sum::<f32>()
            .max(f32::EPSILON);
        let mut gain = 0.0;
        let mut newly_covered_idf = 0.0;
        let mut newly_covered_count = 0usize;
        let mut new_evidence_keys = 0usize;

        for term in query_terms {
            let local_weight = self.profile.local_term_weight(term);
            if local_weight <= 0.0 {
                continue;
            }

            let idf = corpus_stats.idf(term);
            let evidence_multiplier = self.profile.coverage_evidence_multiplier(term);
            if !coverage.covered_terms.contains(term) {
                gain += local_weight * idf * evidence_multiplier * 1.35;
                newly_covered_idf += idf;
                newly_covered_count += 1;
            } else {
                gain += local_weight * idf * evidence_multiplier * 0.18;
            }

            for evidence_type in self.profile.evidence_types_for_term(term) {
                let evidence_key = format!("{evidence_type}:{term}");
                if coverage.covered_evidence_keys.contains(&evidence_key) {
                    continue;
                }
                new_evidence_keys += 1;
                gain += match evidence_type.as_str() {
                    "entity" => 0.75,
                    "local_memory" => 0.65,
                    "topic" => 0.45,
                    _ => 0.30,
                } * idf;
            }
        }

        let new_connection_terms = self
            .connection_terms()
            .into_iter()
            .filter(|term| !coverage.covered_connection_terms.contains(term))
            .take(4)
            .count();
        let temporal_gain = if self.has_temporal_signal() && !coverage.has_temporal_signal {
            0.9
        } else {
            0.0
        };

        if newly_covered_count == 0 && new_evidence_keys == 0 && new_connection_terms == 0 {
            return temporal_gain;
        }

        let rare_coverage_gain = newly_covered_idf / total_query_idf;
        let term_coverage_gain = newly_covered_count as f32 / query_terms.len() as f32;
        gain * (1.0 + rare_coverage_gain)
            + rare_coverage_gain * 2.8
            + term_coverage_gain
            + (new_evidence_keys as f32 * 0.45)
            + (new_connection_terms as f32 * 0.20)
            + temporal_gain
    }

    fn connection_terms(&self) -> HashSet<String> {
        let mut terms = HashSet::new();
        terms.extend(top_weighted_keys(&self.profile.entities, 16));
        terms.extend(top_weighted_keys(&self.profile.topics, 8));
        terms.extend(
            top_weighted_keys(&self.profile.local_memory, 16)
                .into_iter()
                .filter(|term| is_segment_enrichment_candidate(term)),
        );
        terms
    }

    fn has_temporal_signal(&self) -> bool {
        self.index
            .docs
            .values()
            .any(|record| record.timestamp.is_some() || !record.temporal_terms.is_empty())
    }

    fn enriched_query(
        &self,
        query: &str,
        query_terms: &HashSet<String>,
        temporal: TemporalQueryContext<'_>,
    ) -> SegmentQueryEnrichment {
        let mut candidates: HashMap<String, (f32, HashSet<String>)> = HashMap::new();
        collect_profile_candidates(
            &mut candidates,
            &self.profile.local_memory,
            "local_memory",
            1.6,
        );
        collect_profile_candidates(&mut candidates, &self.profile.entities, "entity", 1.4);
        collect_profile_candidates(&mut candidates, &self.profile.topics, "topic", 1.1);
        collect_profile_candidates(&mut candidates, &self.profile.terms, "term", 0.8);
        let temporal_signal =
            collect_temporal_candidates(&mut candidates, self, temporal, query_terms);

        let mut ranked = candidates
            .into_iter()
            .filter(|(term, _)| !query_terms.contains(term))
            .filter(|(term, _)| is_segment_enrichment_candidate(term))
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| {
            b.1 .0
                .partial_cmp(&a.1 .0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let mut added_terms = ranked
            .iter()
            .take(SEGMENT_ENRICHMENT_TERM_LIMIT)
            .map(|(term, _)| term.clone())
            .collect::<Vec<_>>();
        added_terms.sort();

        let mut evidence_types = ranked
            .iter()
            .take(SEGMENT_ENRICHMENT_TERM_LIMIT)
            .flat_map(|(_, (_, evidence))| evidence.iter().cloned())
            .collect::<Vec<_>>();
        evidence_types.sort();
        evidence_types.dedup();
        let temporal_added_terms = ranked
            .iter()
            .take(SEGMENT_ENRICHMENT_TERM_LIMIT)
            .filter(|(_, (_, evidence))| evidence.contains("temporal"))
            .map(|(term, _)| term.clone())
            .collect::<Vec<_>>();
        let mut temporal_evidence = ranked
            .iter()
            .take(SEGMENT_ENRICHMENT_TERM_LIMIT)
            .flat_map(|(_, (_, evidence))| {
                evidence
                    .iter()
                    .filter(|evidence_type| evidence_type.starts_with("temporal"))
                    .cloned()
            })
            .collect::<Vec<_>>();
        temporal_evidence.sort();
        temporal_evidence.dedup();

        let term_weights = ranked
            .iter()
            .take(SEGMENT_ENRICHMENT_TERM_LIMIT)
            .map(|(term, (weight, _))| (term.clone(), *weight))
            .collect::<HashMap<_, _>>();
        let term_evidence_types = ranked
            .iter()
            .take(SEGMENT_ENRICHMENT_TERM_LIMIT)
            .map(|(term, (_, evidence))| {
                let mut evidence = evidence.iter().cloned().collect::<Vec<_>>();
                evidence.sort();
                (term.clone(), evidence)
            })
            .collect::<HashMap<_, _>>();

        let enriched_query = if added_terms.is_empty() {
            query.to_string()
        } else {
            format!("{query} {}", added_terms.join(" "))
        };

        SegmentQueryEnrichment {
            enriched_query,
            added_terms,
            evidence_types,
            temporal_added_terms,
            temporal_evidence,
            temporal_signal,
            term_weights,
            term_evidence_types,
        }
    }

    fn local_evidence(
        &self,
        query_terms: &HashSet<String>,
        corpus_stats: &SegmentCorpusStats,
    ) -> SegmentLocalEvidence {
        let mut differentiators = query_terms
            .iter()
            .filter_map(|term| {
                let local_weight = self.profile.local_term_weight(term);
                (local_weight > 0.0).then(|| LocalDifferentiator {
                    term: term.clone(),
                    weight: local_weight * corpus_stats.idf(term),
                    evidence_types: self.profile.evidence_types_for_term(term),
                })
            })
            .collect::<Vec<_>>();
        differentiators.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.term.cmp(&b.term))
        });
        differentiators.truncate(8);
        SegmentLocalEvidence {
            segment_id: self.segment_id.clone(),
            differentiators,
        }
    }
}

#[derive(Debug)]
struct SegmentQueryEnrichment {
    enriched_query: String,
    added_terms: Vec<String>,
    evidence_types: Vec<String>,
    temporal_added_terms: Vec<String>,
    temporal_evidence: Vec<String>,
    temporal_signal: bool,
    term_weights: HashMap<String, f32>,
    term_evidence_types: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone)]
struct SegmentCorpusStats {
    segment_count: usize,
    term_segment_counts: HashMap<String, usize>,
}

impl SegmentCorpusStats {
    fn from_segments(segments: &[MemoryIndexSegment]) -> Self {
        let mut term_segment_counts = HashMap::new();
        for segment in segments {
            let mut segment_terms = HashSet::new();
            segment_terms.extend(segment.profile.terms.keys().cloned());
            segment_terms.extend(segment.profile.entities.keys().cloned());
            segment_terms.extend(segment.profile.topics.keys().cloned());
            segment_terms.extend(segment.profile.local_memory.keys().cloned());
            for term in segment_terms {
                *term_segment_counts.entry(term).or_default() += 1;
            }
        }
        Self {
            segment_count: segments.len(),
            term_segment_counts,
        }
    }

    fn idf(&self, term: &str) -> f32 {
        let segment_count = self.segment_count as f32;
        if segment_count <= 0.0 {
            return LOCAL_IDF_FLOOR;
        }
        let containing_segments = self
            .term_segment_counts
            .get(term)
            .copied()
            .unwrap_or_default() as f32;
        ((segment_count + 1.0) / (containing_segments + 1.0)).ln() + LOCAL_IDF_FLOOR
    }
}

fn query_tokens(query: &str) -> HashSet<String> {
    query
        .split(|ch: char| !ch.is_alphanumeric())
        .map(normalize_for_index)
        .filter(|token| token.len() > 1)
        .filter(|token| !is_routing_stopword(token))
        .collect()
}

fn add_weight(distribution: &mut HashMap<String, f32>, text: &str, weight: f32) {
    for token in query_tokens(text) {
        *distribution.entry(token).or_default() += weight;
    }
}

fn collect_profile_candidates(
    candidates: &mut HashMap<String, (f32, HashSet<String>)>,
    profile_terms: &HashMap<String, f32>,
    evidence_type: &str,
    weight_multiplier: f32,
) {
    for (term, weight) in profile_terms {
        for token in query_tokens(term) {
            if !is_segment_enrichment_candidate(&token) {
                continue;
            }
            let entry = candidates
                .entry(token)
                .or_insert_with(|| (0.0, HashSet::new()));
            entry.0 += *weight * weight_multiplier;
            entry.1.insert(evidence_type.to_string());
        }
    }
}

fn collect_temporal_candidates(
    candidates: &mut HashMap<String, (f32, HashSet<String>)>,
    segment: &MemoryIndexSegment,
    temporal: TemporalQueryContext<'_>,
    query_terms: &HashSet<String>,
) -> bool {
    let temporal_active = temporal.has_explicit_temporal
        || temporal.time_hint.is_some()
        || temporal.starts_from.is_some();
    if !temporal_active {
        return false;
    }
    let Some(query_date) = temporal.ends_at.and_then(parse_iso_date) else {
        return collect_segment_temporal_terms(candidates, segment, temporal_active, query_terms);
    };

    let mut signal =
        collect_segment_temporal_terms(candidates, segment, temporal_active, query_terms);
    if let Some(hint) = temporal.time_hint {
        add_candidate(
            candidates,
            temporal_hint_label(hint),
            0.6,
            "temporal_query_hint",
        );
        signal = true;
    }

    for record in segment.index.docs.values() {
        let Some(record_date) = record.timestamp.as_deref().and_then(parse_iso_date) else {
            continue;
        };
        let delta_days = record_date.signed_duration_since(query_date).num_days();
        let distance = delta_days.abs();
        let window_days = temporal.window_days.max(1);
        if distance <= window_days {
            add_candidate(candidates, "near", 2.4, "temporal_near_query_date");
            add_candidate(candidates, "recent", 1.2, "temporal_near_query_date");
            signal = true;
        }
        if delta_days < 0 {
            add_candidate(candidates, "before", 0.9, "temporal_before_query_date");
        } else if delta_days > 0 {
            add_candidate(candidates, "after", 0.9, "temporal_after_query_date");
        } else {
            add_candidate(candidates, "same", 1.0, "temporal_same_query_date");
        }
    }
    signal
}

fn segment_temporal_route_boost(
    segment: &MemoryIndexSegment,
    temporal: TemporalQueryContext<'_>,
) -> f32 {
    if !temporal.has_explicit_temporal && temporal.time_hint.is_none() {
        return 0.0;
    }
    let mut boost = 0.0f32;
    let query_date = temporal.ends_at.and_then(parse_iso_date);
    let window_days = temporal.window_days.max(1);

    for record in segment.index.docs.values() {
        if !record.temporal_terms.is_empty() {
            boost += 0.08;
        }
        let Some(record_date) = record.timestamp.as_deref().and_then(parse_iso_date) else {
            continue;
        };
        let Some(query_date) = query_date else {
            boost += 0.05;
            continue;
        };
        let delta_days = record_date.signed_duration_since(query_date).num_days();
        let distance = delta_days.abs();
        if distance <= window_days {
            let proximity = 1.0 - (distance as f32 / window_days as f32);
            boost += proximity.clamp(0.0, 1.0) * 1.25;
        }
        match temporal.time_hint {
            Some(TemporalQueryHint::Past) if delta_days <= 0 => boost += 0.18,
            Some(TemporalQueryHint::Present) | Some(TemporalQueryHint::Ongoing)
                if distance <= 30 =>
            {
                boost += 0.22
            }
            Some(TemporalQueryHint::Mixed) if distance <= 30 => boost += 0.12,
            _ => {}
        }
    }
    boost.min(2.0)
}

fn collect_segment_temporal_terms(
    candidates: &mut HashMap<String, (f32, HashSet<String>)>,
    segment: &MemoryIndexSegment,
    temporal_active: bool,
    query_terms: &HashSet<String>,
) -> bool {
    let mut signal = false;
    let weight = if temporal_active { 1.9 } else { return false };
    for record in segment.index.docs.values() {
        if record.timestamp.is_some() {
            signal = true;
        }
        for term in &record.temporal_terms {
            signal = true;
            for token in query_tokens(term) {
                if query_terms.contains(&token) || !is_segment_enrichment_candidate(&token) {
                    continue;
                }
                add_candidate(candidates, &token, weight, "temporal");
            }
        }
    }
    signal
}

fn add_candidate(
    candidates: &mut HashMap<String, (f32, HashSet<String>)>,
    token: &str,
    weight: f32,
    evidence_type: &str,
) {
    if !is_segment_enrichment_candidate(token) {
        return;
    }
    let entry = candidates
        .entry(token.to_string())
        .or_insert_with(|| (0.0, HashSet::new()));
    entry.0 += weight;
    entry.1.insert(evidence_type.to_string());
    if evidence_type.starts_with("temporal_") {
        entry.1.insert("temporal".to_string());
    }
}

fn temporal_hint_label(hint: TemporalQueryHint) -> &'static str {
    match hint {
        TemporalQueryHint::Past => "past",
        TemporalQueryHint::Present => "present",
        TemporalQueryHint::Ongoing => "ongoing",
        TemporalQueryHint::Mixed => "mixed",
    }
}

fn parse_iso_date(value: &str) -> Option<NaiveDate> {
    let date = value.get(..10).unwrap_or(value);
    NaiveDate::parse_from_str(date, "%Y-%m-%d").ok()
}

fn is_segment_enrichment_candidate(token: &str) -> bool {
    token.len() > 2
        && token.len() <= 24
        && token.chars().all(|ch| ch.is_ascii_lowercase())
        && !is_routing_stopword(token)
        && !matches!(
            token,
            "answer"
                | "assistant"
                | "content"
                | "custom"
                | "longmemev"
                | "longmemeval"
                | "session"
                | "sharegpt"
                | "turn"
                | "ultrachat"
                | "user"
        )
}

fn gated_local_memory_profile(record: &DocRecord) -> HashMap<String, f32> {
    let important_terms = weighted_terms(
        record
            .important_terms
            .iter()
            .map(|term| (term.term.as_str(), term.score.max(0.1))),
    );
    let entity_terms = weighted_terms(
        record
            .key_entities
            .iter()
            .map(|entity| (entity.text.as_str(), entity.score.unwrap_or(1.0))),
    );
    let topic_terms = record
        .probable_topic
        .as_deref()
        .map(|topic| weighted_terms(std::iter::once((topic, 1.0))))
        .unwrap_or_default();

    let mut active_memory: HashMap<String, f32> = HashMap::new();
    let mut retained_memory: HashMap<String, f32> = HashMap::new();
    for token in ordered_tokens(&record.content) {
        for value in active_memory.values_mut() {
            *value *= LOCAL_MEMORY_DECAY;
        }
        active_memory.retain(|_, value| *value >= LOCAL_MEMORY_PRUNE_BELOW);

        let importance = LOCAL_MEMORY_BASE_SIGNAL
            + important_terms.get(&token).copied().unwrap_or_default()
                * LOCAL_MEMORY_IMPORTANT_TERM_WEIGHT
            + entity_terms.get(&token).copied().unwrap_or_default() * LOCAL_MEMORY_ENTITY_WEIGHT
            + topic_terms.get(&token).copied().unwrap_or_default() * LOCAL_MEMORY_TOPIC_WEIGHT;

        if importance > LOCAL_MEMORY_BASE_SIGNAL {
            for value in active_memory.values_mut() {
                *value += importance * LOCAL_MEMORY_NEARBY_REINFORCEMENT;
            }
        }

        *active_memory.entry(token).or_default() += importance;
        for (active_token, value) in &active_memory {
            *retained_memory.entry(active_token.clone()).or_default() += *value;
        }
    }

    normalize_distribution(&mut retained_memory);
    retain_top_terms(&mut retained_memory, LOCAL_MEMORY_RECORD_TERM_LIMIT);
    normalize_distribution(&mut retained_memory);
    retained_memory
}

fn weighted_terms<'a>(terms: impl Iterator<Item = (&'a str, f32)>) -> HashMap<String, f32> {
    let mut weights = HashMap::new();
    for (text, weight) in terms {
        for token in query_tokens(text) {
            *weights.entry(token).or_default() += weight;
        }
    }
    weights
}

fn ordered_tokens(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .map(normalize_for_index)
        .filter(|token| token.len() > 1)
        .filter(|token| !is_routing_stopword(token))
        .collect()
}

fn normalize_distribution(distribution: &mut HashMap<String, f32>) {
    let total = distribution.values().copied().sum::<f32>();
    if total <= f32::EPSILON {
        return;
    }
    for value in distribution.values_mut() {
        *value /= total;
    }
}

fn retain_top_terms(distribution: &mut HashMap<String, f32>, limit: usize) {
    if distribution.len() <= limit {
        return;
    }
    let mut ranked = distribution
        .iter()
        .map(|(term, weight)| (term.clone(), *weight))
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let keep = ranked
        .into_iter()
        .take(limit)
        .map(|(term, _)| term)
        .collect::<HashSet<_>>();
    distribution.retain(|term, _| keep.contains(term));
}

fn top_weighted_keys(distribution: &HashMap<String, f32>, limit: usize) -> Vec<String> {
    let mut ranked = distribution
        .iter()
        .map(|(term, weight)| (term.clone(), *weight))
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked
        .into_iter()
        .take(limit)
        .map(|(term, _)| term)
        .collect()
}

fn overlap_score(query_terms: &HashSet<String>, distribution: &HashMap<String, f32>) -> f32 {
    query_terms
        .iter()
        .filter_map(|term| distribution.get(term))
        .copied()
        .sum()
}

fn merge_distribution(
    target: &mut HashMap<String, f32>,
    source: &HashMap<String, f32>,
    weight: f32,
) {
    for (term, value) in source {
        *target.entry(term.clone()).or_default() += value * weight;
    }
}

fn sorted_terms(terms: &HashSet<String>) -> Vec<String> {
    let mut out = terms.iter().cloned().collect::<Vec<_>>();
    out.sort();
    out
}

fn route_has_signal(
    route: &SegmentRoute,
    strategy: SegmentRoutingStrategy,
    query_terms: &HashSet<String>,
) -> bool {
    if query_terms.is_empty() {
        return false;
    }
    match strategy {
        SegmentRoutingStrategy::KlDivergence => route.score.is_finite(),
        SegmentRoutingStrategy::SparseOverlap
        | SegmentRoutingStrategy::LocalDistinctiveness
        | SegmentRoutingStrategy::CoverageLocalDistinctiveness
        | SegmentRoutingStrategy::TeamCoverageLocalDistinctiveness
        | SegmentRoutingStrategy::CoverageTeamSelection
        | SegmentRoutingStrategy::TypedEvidence => route.score > 0.0,
    }
}

fn is_routing_stopword(token: &str) -> bool {
    matches!(
        token,
        "a" | "an"
            | "and"
            | "are"
            | "can"
            | "did"
            | "do"
            | "doe"
            | "for"
            | "from"
            | "had"
            | "have"
            | "how"
            | "i"
            | "in"
            | "is"
            | "it"
            | "many"
            | "mani"
            | "me"
            | "my"
            | "of"
            | "on"
            | "or"
            | "that"
            | "the"
            | "thi"
            | "this"
            | "to"
            | "wa"
            | "what"
            | "when"
            | "where"
            | "which"
            | "who"
            | "with"
            | "you"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{DocRecord, Provenance, ScoreBreakdown};
    use crate::tier1::{RankedTerm, Tier1Entity};

    fn record(doc_id: &str, group_id: &str, content: &str, terms: &[&str]) -> DocRecord {
        DocRecord {
            doc_id: doc_id.to_string(),
            source: format!("memory://{doc_id}"),
            content: content.to_string(),
            timestamp: None,
            doc_length: content.len(),
            author_agent: None,
            group_id: Some(group_id.to_string()),
            filters: std::collections::BTreeMap::new(),
            probable_topic: terms.first().map(|term| (*term).to_string()),
            doc_type_guess: None,
            headings: vec![group_id.to_string()],
            doc_links: vec![],
            temporal_terms: vec![],
            key_entities: terms
                .iter()
                .map(|term| Tier1Entity {
                    text: (*term).to_string(),
                    label: "KEY".to_string(),
                    start: 0,
                    end: term.len(),
                    score: Some(1.0),
                    source: "test".to_string(),
                })
                .collect(),
            important_terms: terms
                .iter()
                .map(|term| RankedTerm {
                    term: (*term).to_string(),
                    score: 1.0,
                    source: "test".to_string(),
                })
                .collect(),
            section_chunks: vec![],
            embedding: None,
            top_claims: vec![],
            provenance: Provenance {
                source: "test".to_string(),
                timestamp: None,
                ner_provider: "test".to_string(),
                term_ranker: "test".to_string(),
                index_version: "test".to_string(),
            },
        }
    }

    fn search_result(doc_id: &str, score: f32) -> SearchResult {
        SearchResult {
            doc_id: doc_id.to_string(),
            source: format!("memory://{doc_id}"),
            group_id: None,
            score,
            score_breakdown: ScoreBreakdown::default(),
            matched_entities: Vec::new(),
            matched_terms: Vec::new(),
            probable_topic: None,
            doc_type_guess: None,
        }
    }

    #[test]
    fn routes_and_queries_one_group_segment() {
        let records = vec![
            record(
                "doc-a",
                "session-a",
                "docker install guide for linux",
                &["docker", "install", "linux"],
            ),
            record(
                "doc-b",
                "session-b",
                "kubernetes cluster operations",
                &["kubernetes", "cluster"],
            ),
        ];
        let segments = build_segments_by_group_id(&records);

        assert_eq!(segments.len(), 2);
        let routes = route_segments("docker install", &segments);
        assert_eq!(routes[0].segment_id, "session-a");

        let results = query_top_segment("docker install", 5, &segments);
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "doc-a");
    }

    #[test]
    fn queries_top_n_segments_and_merges_results() {
        let records = vec![
            record(
                "doc-a",
                "session-a",
                "docker install guide for linux",
                &["docker", "install", "linux"],
            ),
            record(
                "doc-b",
                "session-b",
                "docker compose troubleshooting",
                &["docker", "compose", "troubleshooting"],
            ),
            record(
                "doc-c",
                "session-c",
                "kubernetes cluster operations",
                &["kubernetes", "cluster"],
            ),
        ];
        let segments = build_segments_by_group_id(&records);

        let one_segment = query_top_segments("docker", 5, &segments, 1);
        assert_eq!(one_segment.len(), 1);

        let two_segments = query_top_segments("docker", 5, &segments, 2);
        let doc_ids = two_segments
            .iter()
            .map(|result| result.doc_id.as_str())
            .collect::<HashSet<_>>();
        assert_eq!(two_segments.len(), 2);
        assert!(doc_ids.contains("doc-a"));
        assert!(doc_ids.contains("doc-b"));
        assert!(!doc_ids.contains("doc-c"));
    }

    #[test]
    fn segmented_memory_index_returns_diagnostics() {
        let records = vec![
            record(
                "doc-a",
                "session-a",
                "docker install guide for linux",
                &["docker", "install", "linux"],
            ),
            record(
                "doc-b",
                "session-b",
                "kubernetes cluster operations",
                &["kubernetes", "cluster"],
            ),
        ];
        let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

        assert_eq!(segmented.len(), 2);
        let routes = segmented.route("docker install");
        assert_eq!(routes[0].segment_id, "session-a");

        let output = segmented.query_with_diagnostics("docker install", 5, 1);
        assert_eq!(output.diagnostics.queried_segment_count, 1);
        assert_eq!(
            output.diagnostics.selected_segments[0].segment_id,
            "session-a"
        );
        assert_eq!(output.diagnostics.final_result_count, output.results.len());
        assert!(output.diagnostics.merged_result_count >= output.diagnostics.final_result_count);
        assert_eq!(
            output
                .diagnostics
                .per_segment_result_counts
                .get("session-a")
                .copied(),
            Some(output.results.len())
        );
        assert_eq!(
            output.diagnostics.query_terms,
            vec!["docker".to_string(), "instal".to_string()]
        );
        assert_eq!(
            output.diagnostics.covered_query_terms,
            vec!["docker".to_string(), "instal".to_string()]
        );
        assert!(output.diagnostics.uncovered_query_terms.is_empty());
        assert_eq!(
            output.diagnostics.segments_with_results,
            vec!["session-a".to_string()]
        );
    }

    #[test]
    fn diagnostics_report_query_coverage_across_more_segments() {
        let records = vec![
            record("doc-a", "session-a", "docker install guide", &["docker"]),
            record(
                "doc-b",
                "session-b",
                "compose troubleshooting guide",
                &["compose"],
            ),
            record(
                "doc-c",
                "session-c",
                "kubernetes cluster operations",
                &["kubernetes"],
            ),
        ];
        let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

        let top_one = segmented.query_with_diagnostics("docker compose", 5, 1);
        assert_eq!(
            top_one.diagnostics.covered_query_terms,
            vec!["docker".to_string()]
        );
        assert_eq!(
            top_one.diagnostics.uncovered_query_terms,
            vec!["compos".to_string()]
        );
        assert_eq!(
            top_one.diagnostics.segments_with_results,
            vec!["session-a".to_string()]
        );

        let top_two = segmented.query_with_diagnostics("docker compose", 5, 2);
        assert_eq!(
            top_two.diagnostics.covered_query_terms,
            vec!["compos".to_string(), "docker".to_string()]
        );
        assert!(top_two.diagnostics.uncovered_query_terms.is_empty());
        assert_eq!(
            top_two.diagnostics.segments_with_results,
            vec!["session-a".to_string(), "session-b".to_string()]
        );
    }

    #[test]
    fn kl_router_prefers_closest_segment_distribution() {
        let records = vec![
            record(
                "doc-a",
                "session-a",
                "docker compose install troubleshooting",
                &["docker", "compose", "install"],
            ),
            record(
                "doc-b",
                "session-b",
                "kubernetes cluster node scheduling",
                &["kubernetes", "cluster", "node"],
            ),
        ];
        let segments = build_segments_by_group_id(&records);

        let routes = route_segments_with_strategy(
            "docker compose",
            &segments,
            SegmentRoutingStrategy::KlDivergence,
        );

        assert_eq!(routes[0].segment_id, "session-a");
        assert!(routes[0].score > routes[1].score);
    }

    #[test]
    fn local_router_prefers_segment_specific_differentiators() {
        let records = vec![
            record(
                "doc-a",
                "session-a",
                "general study planning notes",
                &["study", "notes", "planning"],
            ),
            record(
                "doc-b",
                "session-b",
                "undergraduate graduate GPA transcript",
                &["study", "undergraduate", "graduate", "GPA"],
            ),
            record(
                "doc-c",
                "session-c",
                "study schedule and reading list",
                &["study", "schedule", "reading"],
            ),
        ];
        let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

        let output = segmented.query_with_diagnostics_and_strategy(
            "what is my average GPA from undergraduate and graduate studies",
            5,
            1,
            SegmentRoutingStrategy::LocalDistinctiveness,
        );

        assert_eq!(
            output.diagnostics.selected_segments[0].segment_id,
            "session-b"
        );
        let local_terms = output.diagnostics.local_evidence[0]
            .differentiators
            .iter()
            .map(|differentiator| differentiator.term.as_str())
            .collect::<HashSet<_>>();
        assert!(local_terms.contains("gpa"));
        assert!(local_terms.contains("undergradu"));
        assert!(local_terms.contains("graduat"));
        assert!(!output.diagnostics.query_terms.contains(&"what".to_string()));
        assert!(!output.diagnostics.query_terms.contains(&"my".to_string()));
    }

    #[test]
    fn coverage_local_router_prefers_rare_term_coverage() {
        let records = vec![
            record(
                "doc-a",
                "session-a",
                "GPA GPA GPA GPA GPA academic score notes",
                &["GPA"],
            ),
            record(
                "doc-b",
                "session-b",
                "undergraduate graduate transcript academic record",
                &["undergraduate", "graduate", "transcript"],
            ),
            record(
                "doc-c",
                "session-c",
                "general academic planning and study notes",
                &["academic", "planning"],
            ),
        ];
        let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

        let output = segmented.query_with_diagnostics_and_strategy(
            "GPA undergraduate graduate transcript",
            5,
            1,
            SegmentRoutingStrategy::CoverageLocalDistinctiveness,
        );

        assert_eq!(
            output.diagnostics.selected_segments[0].segment_id,
            "session-b"
        );
        let local_terms = output.diagnostics.local_evidence[0]
            .differentiators
            .iter()
            .map(|differentiator| differentiator.term.as_str())
            .collect::<HashSet<_>>();
        assert!(local_terms.contains("undergradu"));
        assert!(local_terms.contains("graduat"));
        assert!(local_terms.contains("transcript"));
    }

    #[test]
    fn team_coverage_router_prefers_complementary_segments() {
        let records = vec![
            record(
                "doc-a",
                "session-a",
                "GPA undergraduate academic record",
                &["GPA", "undergraduate"],
            ),
            record(
                "doc-b",
                "session-b",
                "GPA undergraduate admission note",
                &["GPA", "undergraduate"],
            ),
            record(
                "doc-c",
                "session-c",
                "graduate jewelry repair appointment",
                &["graduate", "jewelry", "appointment"],
            ),
        ];
        let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

        let output = segmented.query_with_diagnostics_and_strategy(
            "GPA undergraduate graduate jewelry appointment",
            5,
            2,
            SegmentRoutingStrategy::TeamCoverageLocalDistinctiveness,
        );
        let selected = output
            .diagnostics
            .selected_segments
            .iter()
            .map(|route| route.segment_id.as_str())
            .collect::<HashSet<_>>();

        assert_eq!(selected.len(), 2);
        assert!(selected.contains("session-c"));
        assert!(selected.contains("session-a") || selected.contains("session-b"));
        assert!(output
            .diagnostics
            .covered_query_terms
            .contains(&"appoint".to_string()));
        assert!(output
            .diagnostics
            .covered_query_terms
            .contains(&"gpa".to_string()));
        assert!(output
            .diagnostics
            .covered_query_terms
            .contains(&"jewelri".to_string()));
    }

    #[test]
    fn adaptive_segment_enrichment_expands_when_query_coverage_is_low() {
        let records = vec![
            record(
                "doc-a",
                "session-a",
                "GPA undergraduate academic record",
                &["GPA", "undergraduate"],
            ),
            record(
                "doc-b",
                "session-b",
                "graduate jewelry repair appointment",
                &["graduate", "jewelry", "appointment"],
            ),
        ];
        let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

        let (output, _) = segmented
            .query_with_adaptive_segment_enrichment_temporal_context_and_strategy(
                "GPA undergraduate graduate jewelry appointment",
                5,
                1,
                2,
                SegmentRoutingStrategy::CoverageLocalDistinctiveness,
                TemporalQueryContext::default(),
            );
        let selected = output
            .diagnostics
            .selected_segments
            .iter()
            .map(|route| route.segment_id.as_str())
            .collect::<HashSet<_>>();

        assert_eq!(selected.len(), 2);
        assert!(selected.contains("session-a"));
        assert!(selected.contains("session-b"));
        assert!(output.diagnostics.uncovered_query_terms.is_empty());
    }

    #[test]
    fn route_aware_rerank_prefers_new_evidence_coverage() {
        let candidates = vec![
            RouteAwareCandidate {
                result: search_result("doc-common", 1.0),
                segment_id: "session-common".to_string(),
                base_score: 1.0,
                evidence_terms: HashSet::from(["appointment".to_string()]),
            },
            RouteAwareCandidate {
                result: search_result("doc-specific", 0.9),
                segment_id: "session-specific".to_string(),
                base_score: 0.9,
                evidence_terms: HashSet::from(["gpa".to_string(), "jewelri".to_string()]),
            },
        ];

        let selected = select_route_aware_top_k(candidates, 1);

        assert_eq!(selected[0].doc_id, "doc-specific");
        assert!(selected[0].score > 0.9);
    }

    #[test]
    fn session_aggregation_combines_segment_evidence_by_group() {
        let mut session_a_primary = search_result("session-a::turn0", 1.0);
        session_a_primary.group_id = Some("session-a".to_string());
        session_a_primary.matched_terms = vec!["gpa".to_string()];
        let mut session_a_support = search_result("session-a::turn1", 0.8);
        session_a_support.group_id = Some("session-a".to_string());
        session_a_support.matched_entities = vec!["jewelry".to_string()];
        let mut session_b = search_result("session-b::turn0", 1.1);
        session_b.group_id = Some("session-b".to_string());

        let aggregated = aggregate_segment_results_by_session(vec![
            session_b,
            session_a_support,
            session_a_primary,
        ]);

        assert_eq!(aggregated[0].group_id.as_deref(), Some("session-a"));
        assert!(aggregated[0].score > 1.1);
    }

    #[test]
    fn local_memory_router_uses_rolling_content_window() {
        let records = vec![
            record(
                "doc-a",
                "session-a",
                "the day before my doctor appointment I went to bed early",
                &["appointment"],
            ),
            record(
                "doc-b",
                "session-b",
                "appointment calendar reminder and scheduling notes",
                &["appointment"],
            ),
        ];
        let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

        let output = segmented.query_with_diagnostics_and_strategy(
            "what time did I go to bed on the day before the doctor appointment",
            5,
            1,
            SegmentRoutingStrategy::LocalDistinctiveness,
        );

        assert_eq!(
            output.diagnostics.selected_segments[0].segment_id,
            "session-a"
        );
        let local_evidence = &output.diagnostics.local_evidence[0].differentiators;
        assert!(local_evidence
            .iter()
            .any(|differentiator| differentiator.term == "bed"
                && differentiator
                    .evidence_types
                    .contains(&"local_memory".to_string())));
        assert!(local_evidence
            .iter()
            .any(|differentiator| differentiator.term == "doctor"
                && differentiator
                    .evidence_types
                    .contains(&"local_memory".to_string())));
    }

    #[test]
    fn segment_enriched_query_uses_each_selected_segments_local_context() {
        let records = vec![
            record(
                "doc-a",
                "session-a",
                "doctor appointment clinic notes before going to bed early",
                &["appointment", "doctor", "clinic"],
            ),
            record(
                "doc-b",
                "session-b",
                "appointment calendar reminder for product planning meeting",
                &["appointment", "calendar", "meeting"],
            ),
        ];
        let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

        let (output, enrichment) = segmented.query_with_segment_enrichment_and_strategy(
            "appointment",
            5,
            2,
            SegmentRoutingStrategy::LocalDistinctiveness,
        );

        assert_eq!(output.diagnostics.queried_segment_count, 2);
        assert_eq!(enrichment.selected_segments.len(), 2);
        let by_segment = enrichment
            .selected_segments
            .iter()
            .map(|diagnostic| (diagnostic.segment_id.as_str(), diagnostic))
            .collect::<HashMap<_, _>>();
        let session_a = by_segment.get("session-a").expect("session-a enrichment");
        let session_b = by_segment.get("session-b").expect("session-b enrichment");

        assert_ne!(session_a.enriched_query, session_b.enriched_query);
        assert!(session_a.added_terms.iter().any(|term| term == "doctor"));
        assert!(session_b.added_terms.iter().any(|term| term == "calendar"));
        assert!(enrichment.average_added_terms > 0.0);
        assert!(output
            .diagnostics
            .local_evidence
            .iter()
            .any(|evidence| !evidence.differentiators.is_empty()));
    }

    #[test]
    fn segment_enriched_query_reports_temporal_local_context() {
        let mut older = record(
            "doc-a",
            "session-a",
            "doctor appointment notes from last week",
            &["appointment", "doctor"],
        );
        older.timestamp = Some("2024-05-01".to_string());
        older.temporal_terms = vec!["date 2024-05-01".to_string(), "last week".to_string()];
        let mut current = record(
            "doc-b",
            "session-b",
            "doctor appointment notes from today",
            &["appointment", "doctor"],
        );
        current.timestamp = Some("2024-05-10".to_string());
        current.temporal_terms = vec!["date 2024-05-10".to_string(), "today".to_string()];
        let segmented = SegmentedMemoryIndex::from_records_by_group_id(&[older, current]);

        let (_output, enrichment) = segmented
            .query_with_segment_enrichment_temporal_context_and_strategy(
                "what happened at the appointment today",
                5,
                2,
                SegmentRoutingStrategy::LocalDistinctiveness,
                TemporalQueryContext {
                    ends_at: Some("2024-05-10"),
                    time_hint: Some(TemporalQueryHint::Present),
                    has_explicit_temporal: true,
                    ..TemporalQueryContext::default()
                },
            );

        let current = enrichment
            .selected_segments
            .iter()
            .find(|diagnostic| diagnostic.segment_id == "session-b")
            .expect("session-b enrichment");
        assert!(current.temporal_signal);
        assert!(!current.temporal_added_terms.is_empty());
        assert!(current
            .temporal_evidence
            .iter()
            .any(|evidence| evidence == "temporal_near_query_date"));
    }

    #[test]
    fn temporal_path_enrichment_expands_to_nearby_before_segment() {
        let mut anchor = record(
            "doc-anchor",
            "session-anchor",
            "doctor appointment happened at the clinic",
            &["doctor", "appointment", "clinic"],
        );
        anchor.timestamp = Some("2024-05-10".to_string());
        anchor.temporal_terms = vec!["date 2024-05-10".to_string()];
        let mut before = record(
            "doc-before",
            "session-before",
            "bought toothpaste before the visit",
            &["toothpaste", "visit"],
        );
        before.timestamp = Some("2024-05-09".to_string());
        before.temporal_terms = vec!["date 2024-05-09".to_string(), "before".to_string()];
        let mut unrelated = record(
            "doc-unrelated",
            "session-unrelated",
            "weekly grocery list and errands",
            &["grocery", "errands"],
        );
        unrelated.timestamp = Some("2024-04-01".to_string());
        let segmented =
            SegmentedMemoryIndex::from_records_by_group_id(&[anchor, before, unrelated]);

        let (output, enrichment) = segmented.query_with_temporal_path_enrichment_and_strategy(
            "what happened before the doctor appointment",
            5,
            1,
            SegmentRoutingStrategy::LocalDistinctiveness,
            TemporalQueryContext {
                ends_at: Some("2024-05-10"),
                window_days: 2,
                time_hint: Some(TemporalQueryHint::Past),
                has_explicit_temporal: true,
                ..TemporalQueryContext::default()
            },
        );

        assert!(output
            .diagnostics
            .selected_segments
            .iter()
            .any(|route| route.segment_id == "session-before"));
        assert!(enrichment
            .temporal_expanded_segments
            .iter()
            .any(|expansion| expansion.segment_id == "session-before"
                && expansion.source_segment_id == "session-anchor"));
        assert!(output
            .results
            .iter()
            .any(|result| result.doc_id == "doc-before"));
    }

    #[test]
    fn connected_segment_enrichment_swaps_in_explicitly_related_session() {
        let anchor = record(
            "doc-anchor",
            "session-anchor",
            "Alice mentioned toothpaste during the doctor appointment",
            &["Alice", "doctor", "appointment", "toothpaste"],
        );
        let connected = record(
            "doc-connected",
            "session-connected",
            "Alice bought toothpaste before the clinic visit",
            &["Alice", "bought", "toothpaste", "visit"],
        );
        let decoy = record(
            "doc-decoy",
            "session-decoy",
            "calendar reminder and weekly planning",
            &["calendar", "planning"],
        );
        let segmented = SegmentedMemoryIndex::from_records_by_group_id(&[anchor, connected, decoy]);

        let (output, enrichment) = segmented.query_with_connected_segment_enrichment_and_strategy(
            "doctor appointment",
            5,
            1,
            SegmentRoutingStrategy::CoverageLocalDistinctiveness,
            TemporalQueryContext::default(),
        );

        assert_eq!(output.diagnostics.selected_segments.len(), 1);
        assert_eq!(
            output.diagnostics.selected_segments[0].segment_id,
            "session-connected"
        );
        let expansion = enrichment
            .connected_expanded_segments
            .iter()
            .find(|expansion| expansion.segment_id == "session-connected")
            .expect("connected expansion");
        assert!(expansion.action.starts_with("swapped_out:"));
        assert!(expansion
            .shared_subjects
            .iter()
            .any(|term| term == "alic" || term == "alice"));
        assert!(expansion
            .shared_objects
            .iter()
            .any(|term| term == "toothpast"));
    }

    #[test]
    fn missing_coverage_recovery_replaces_weak_segment() {
        let strong = record(
            "doc-strong",
            "session-strong",
            "GPA application admissions notes",
            &["GPA", "application", "admissions"],
        );
        let weak = record(
            "doc-weak",
            "session-weak",
            "GPA application general notes repeated",
            &["GPA", "application"],
        );
        let recovered = record(
            "doc-recovered",
            "session-recovered",
            "undergraduate graduate transcript details",
            &["undergraduate", "graduate", "transcript"],
        );
        let segments = build_segments_by_group_id(&[strong, weak, recovered]);
        let query_terms = query_tokens("GPA undergraduate graduate application");
        let routes = vec![
            SegmentRoute {
                segment_id: "session-strong".to_string(),
                score: 10.0,
                fallback: false,
            },
            SegmentRoute {
                segment_id: "session-weak".to_string(),
                score: 2.0,
                fallback: false,
            },
            SegmentRoute {
                segment_id: "session-recovered".to_string(),
                score: 1.5,
                fallback: false,
            },
        ];

        let (selected, events) =
            recover_missing_coverage_segments(&query_terms, &routes[..2], &routes, &segments);
        let selected_ids = selected
            .iter()
            .map(|route| route.segment_id.as_str())
            .collect::<HashSet<_>>();

        assert_eq!(selected.len(), 2);
        assert!(selected_ids.contains("session-strong"));
        assert!(selected_ids.contains("session-recovered"));
        assert!(!selected_ids.contains("session-weak"));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].source_segment_id, "session-weak");
        assert!(events[0]
            .shared_subjects
            .iter()
            .any(|term| term == "undergradu"));
        assert!(events[0]
            .action
            .starts_with("recovered_missing_coverage:swapped_out:"));
    }

    #[test]
    fn typed_evidence_score_requires_explicit_multi_bucket_match() {
        let query = query_connection_profile("Alice bought toothpaste today");
        let mut loose = SegmentConnectionProfile::default();
        loose.subjects.insert("toothpast".to_string());

        let mut explicit = loose.clone();
        explicit.objects.insert("toothpast".to_string());
        explicit.people.insert("alic".to_string());
        explicit.actions.insert("bought".to_string());
        explicit.times.insert("today".to_string());

        assert_eq!(typed_evidence_route_score(&query, &loose), 0.0);
        assert!(typed_evidence_route_score(&query, &explicit) > 0.0);
    }

    #[test]
    fn sparse_router_does_not_query_zero_signal_padding_segments() {
        let records = vec![
            record("doc-a", "session-a", "docker install guide", &["docker"]),
            record("doc-b", "session-b", "kubernetes cluster", &["kubernetes"]),
            record("doc-c", "session-c", "postgres index tuning", &["postgres"]),
        ];
        let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

        let output = segmented.query_with_diagnostics("docker", 5, 3);

        assert_eq!(output.diagnostics.selected_segments.len(), 1);
        assert_eq!(
            output.diagnostics.selected_segments[0].segment_id,
            "session-a"
        );
        assert_eq!(output.diagnostics.queried_segment_count, 1);
        assert_eq!(output.diagnostics.fallback_segments.len(), 2);
        assert!(output.diagnostics.routing_fallback);
        assert_eq!(
            output.diagnostics.routing_fallback_reason.as_deref(),
            Some("insufficient_signal_routes")
        );
        assert_eq!(
            output
                .results
                .iter()
                .map(|result| result.doc_id.as_str())
                .collect::<Vec<_>>(),
            vec!["doc-a"]
        );
        assert!(output
            .diagnostics
            .fallback_segments
            .iter()
            .all(|route| route.fallback));
    }

    #[test]
    fn sparse_router_reports_no_signal_without_querying_lexicographic_first_segment() {
        let records = vec![
            record("doc-a", "session-a", "docker install guide", &["docker"]),
            record("doc-b", "session-b", "kubernetes cluster", &["kubernetes"]),
        ];
        let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

        let output = segmented.query_with_diagnostics("espresso grinder", 5, 1);

        assert!(output.results.is_empty());
        assert!(output.diagnostics.selected_segments.is_empty());
        assert_eq!(output.diagnostics.queried_segment_count, 0);
        assert_eq!(output.diagnostics.fallback_segments.len(), 1);
        assert!(output.diagnostics.routing_fallback);
        assert_eq!(
            output.diagnostics.routing_fallback_reason.as_deref(),
            Some("no_signal_routes")
        );
    }

    #[test]
    fn segmented_query_passes_temporal_allowed_doc_ids_to_inner_indexes() {
        let records = vec![
            record(
                "doc-a",
                "session-a",
                "docker install guide for linux",
                &["docker", "install", "linux"],
            ),
            record(
                "doc-b",
                "session-b",
                "docker compose troubleshooting",
                &["docker", "compose", "troubleshooting"],
            ),
        ];
        let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);
        let allowed_doc_ids = HashSet::from(["doc-b".to_string()]);

        let output = segmented.query_with_temporal_context_and_diagnostics_and_strategy(
            "docker",
            5,
            2,
            SegmentRoutingStrategy::SparseOverlap,
            TemporalQueryContext {
                allowed_doc_ids: Some(&allowed_doc_ids),
                ..TemporalQueryContext::default()
            },
        );

        assert_eq!(output.results.len(), 1);
        assert_eq!(output.results[0].doc_id, "doc-b");
        assert_eq!(
            output
                .diagnostics
                .per_segment_result_counts
                .get("session-a")
                .copied(),
            Some(0)
        );
        assert_eq!(
            output
                .diagnostics
                .per_segment_result_counts
                .get("session-b")
                .copied(),
            Some(1)
        );
    }

    #[test]
    fn all_segment_query_matches_global_memory_index_on_asymmetric_corpus() {
        let records = vec![
            record(
                "doc-a1",
                "session-a",
                "docker install guide for linux",
                &["docker", "install", "linux"],
            ),
            record(
                "doc-a2",
                "session-a",
                "docker setup notes and troubleshooting",
                &["docker", "setup", "troubleshooting"],
            ),
            record(
                "doc-a3",
                "session-a",
                "linux package manager notes",
                &["linux", "package"],
            ),
            record(
                "doc-b1",
                "session-b",
                "docker compose production incident",
                &["docker", "compose", "production"],
            ),
        ];
        let global = MemoryIndex::from_records(records.clone());
        let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

        let global_doc_ids = global
            .query("docker compose troubleshooting", 4)
            .into_iter()
            .map(|result| result.doc_id)
            .collect::<Vec<_>>();
        let segmented_doc_ids = segmented
            .query_all_segments("docker compose troubleshooting", 4)
            .into_iter()
            .map(|result| result.doc_id)
            .collect::<Vec<_>>();

        assert_eq!(segmented_doc_ids, global_doc_ids);
    }

    #[test]
    fn all_segment_query_reconstructs_global_index_when_missing() {
        let records = vec![
            record(
                "doc-a",
                "session-a",
                "docker install guide for linux",
                &["docker", "install", "linux"],
            ),
            record(
                "doc-b",
                "session-b",
                "docker compose troubleshooting",
                &["docker", "compose", "troubleshooting"],
            ),
            record(
                "doc-c",
                "session-c",
                "postgres index tuning",
                &["postgres", "index", "tuning"],
            ),
        ];
        let segmented_with_global = SegmentedMemoryIndex::from_records_by_group_id(&records);
        let segments = build_segments_by_group_id(&records);
        let segmented_without_global = SegmentedMemoryIndex {
            corpus_stats: SegmentCorpusStats::from_segments(&segments),
            segments,
            global_index: None,
        };

        let expected = segmented_with_global
            .query_all_segments("docker compose troubleshooting", 3)
            .into_iter()
            .map(|result| result.doc_id)
            .collect::<Vec<_>>();
        let actual = segmented_without_global
            .query_all_segments_with_diagnostics("docker compose troubleshooting", 3)
            .results
            .into_iter()
            .map(|result| result.doc_id)
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);
    }

    #[test]
    fn sparse_router_marks_empty_query_terms_as_fallback() {
        let records = vec![
            record("doc-a", "session-a", "docker install guide", &["docker"]),
            record("doc-b", "session-b", "kubernetes cluster", &["kubernetes"]),
        ];
        let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

        let output = segmented.query_with_diagnostics("the and to", 5, 2);

        assert!(output.results.is_empty());
        assert!(output.diagnostics.selected_segments.is_empty());
        assert_eq!(output.diagnostics.queried_segment_count, 0);
        assert!(output.diagnostics.routing_fallback);
        assert_eq!(
            output.diagnostics.routing_fallback_reason.as_deref(),
            Some("empty_query_terms")
        );
        assert_eq!(output.diagnostics.fallback_segments.len(), 2);
    }
}
