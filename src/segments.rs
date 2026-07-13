use crate::index::{DocRecord, MemoryIndex, SearchResult, TemporalQueryContext};
use crate::query_expansion::normalize_for_index;
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
        let segments = build_segments_by_group_id(records);
        let corpus_stats = SegmentCorpusStats::from_segments(&segments);
        Self {
            segments,
            global_index: Some(MemoryIndex::from_records(records.to_vec())),
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
        self.query_with_temporal_context_and_diagnostics_and_strategy(
            query,
            top_k,
            segment_limit,
            strategy,
            TemporalQueryContext::default(),
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
    temporal: TemporalQueryContext<'_>,
    global_index: Option<&MemoryIndex>,
) -> SegmentQueryOutput {
    let corpus_stats = SegmentCorpusStats::from_segments(segments);
    query_top_segments_with_corpus_stats_and_strategy(
        query,
        top_k,
        segments,
        segment_limit,
        strategy,
        temporal,
        global_index,
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
        } else {
            let mut seen_doc_ids = HashSet::new();
            for route in selected_segments.iter() {
                let Some(segment) = segments
                    .iter()
                    .find(|segment| segment.segment_id == route.segment_id)
                else {
                    continue;
                };
                let segment_doc_ids = segment.doc_ids.iter().cloned().collect();
                let segment_allowed_doc_ids =
                    intersect_allowed_doc_ids(&segment_doc_ids, temporal.allowed_doc_ids);
                let scoped_temporal = TemporalQueryContext {
                    allowed_doc_ids: Some(&segment_allowed_doc_ids),
                    ..temporal
                };
                for result in segment
                    .index
                    .query_with_temporal_context(query, top_k, scoped_temporal)
                    .0
                {
                    if seen_doc_ids.insert(result.doc_id.clone()) {
                        merged.push(result);
                    }
                }
            }
            merged.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.doc_id.cmp(&b.doc_id))
            });
            merged.truncate(top_k);
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

pub fn query_all_segments_with_diagnostics(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
) -> SegmentQueryOutput {
    query_all_segments_with_temporal_context_and_diagnostics(
        query,
        top_k,
        segments,
        TemporalQueryContext::default(),
    )
}

pub fn query_all_segments_with_temporal_context_and_diagnostics(
    query: &str,
    top_k: usize,
    segments: &[MemoryIndexSegment],
    temporal: TemporalQueryContext<'_>,
) -> SegmentQueryOutput {
    let corpus_stats = SegmentCorpusStats::from_segments(segments);
    query_top_segments_with_corpus_stats_and_strategy(
        query,
        top_k,
        segments,
        segments.len(),
        SegmentRoutingStrategy::SparseOverlap,
        temporal,
        None,
        &corpus_stats,
    )
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
        SegmentRoutingStrategy::SparseOverlap | SegmentRoutingStrategy::LocalDistinctiveness => {
            route.score > 0.0
        }
        SegmentRoutingStrategy::KlDivergence => route.score.is_finite(),
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
    use crate::index::{DocRecord, Provenance};
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
        let segmented_without_global = SegmentedMemoryIndex {
            segments: build_segments_by_group_id(&records),
            global_index: None,
            corpus_stats: SegmentCorpusStats::from_segments(&build_segments_by_group_id(&records)),
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
