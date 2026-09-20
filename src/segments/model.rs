use crate::index::{DocRecord, MemoryIndex, TemporalQueryContext};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::catalog::*;
use super::query::*;
use super::routing::*;

pub(crate) const KL_SMOOTHING: f32 = 1.0e-6;
pub(crate) const LOCAL_IDF_FLOOR: f32 = 1.0;
pub(crate) const LOCAL_MEMORY_DECAY: f32 = 0.82;
pub(crate) const LOCAL_MEMORY_PRUNE_BELOW: f32 = 0.015;
pub(crate) const LOCAL_MEMORY_BASE_SIGNAL: f32 = 0.04;
pub(crate) const LOCAL_MEMORY_IMPORTANT_TERM_WEIGHT: f32 = 1.0;
pub(crate) const LOCAL_MEMORY_ENTITY_WEIGHT: f32 = 1.4;
pub(crate) const LOCAL_MEMORY_TOPIC_WEIGHT: f32 = 0.8;
pub(crate) const LOCAL_MEMORY_NEARBY_REINFORCEMENT: f32 = 0.08;
pub(crate) const LOCAL_MEMORY_PROFILE_WEIGHT: f32 = 0.35;
pub(crate) const LOCAL_MEMORY_RECORD_TERM_LIMIT: usize = 24;
pub(crate) const LOCAL_FREQUENCY_MAX_SOURCE_POSTINGS: usize = 64;
pub(crate) const LOCAL_MEMORY_SEGMENT_TERM_LIMIT: usize = 96;
pub(crate) const SEGMENT_ENRICHMENT_TERM_LIMIT: usize = 6;
pub(crate) const ADAPTIVE_MIN_QUERY_COVERAGE: f32 = 0.80;
pub(crate) const ADAPTIVE_CLOSE_SCORE_RATIO: f32 = 0.80;
pub(crate) const RERANK_NORMALIZED_RESULT_WEIGHT: f32 = 1.0;
pub(crate) const RERANK_ROUTE_WEIGHT: f32 = 0.35;
pub(crate) const RERANK_QUERY_EVIDENCE_WEIGHT: f32 = 0.18;
pub(crate) const RERANK_ENRICHED_EVIDENCE_WEIGHT: f32 = 0.14;
pub(crate) const RERANK_LOCAL_EVIDENCE_WEIGHT: f32 = 0.12;
pub(crate) const RERANK_TEMPORAL_WEIGHT: f32 = 0.10;
pub(crate) const RERANK_COVERAGE_GAIN_WEIGHT: f32 = 0.16;
pub(crate) const RERANK_SEGMENT_COVERAGE_WEIGHT: f32 = 0.28;
pub(crate) const RERANK_COMMON_ONLY_PENALTY: f32 = 0.18;
pub(crate) const CONNECTED_EXPANSION_POOL_MULTIPLIER: usize = 3;
pub(crate) const CONNECTED_NEIGHBOR_CANDIDATE_LIMIT: usize = 128;
pub(crate) const CONNECTED_EXPANSION_MIN_SCORE: f32 = 1.4;
pub(crate) const CONNECTED_EXPANSION_MAX_SWAP_PENALTY: f32 = 0.35;
pub(crate) const TYPED_EVIDENCE_ROUTE_WEIGHT: f32 = 1.15;
pub(crate) const MISSING_COVERAGE_RECOVERY_POOL_LIMIT: usize = 20;
pub(crate) const MISSING_COVERAGE_MIN_GAIN: f32 = 1.8;
pub(crate) const MISSING_COVERAGE_MIN_WEAK_SCORE: f32 = 1.5;
pub(crate) const MAX_LOCAL_SEGMENT_QUERY_CONCURRENCY: usize = 8;
pub(crate) const SEGMENT_CANDIDATE_OVERSAMPLE: usize = 2;
pub(crate) const ROUTING_POSTINGS_PER_TERM: usize = 16;
pub(crate) const ROUTING_CANDIDATE_POOL_LIMIT: usize = 64;

pub struct MemoryIndexSegment {
    pub segment_id: String,
    pub doc_ids: Vec<String>,
    /// Shared by value across snapshots: a segment's index is immutable once
    /// built, so refreshes reuse unchanged segments instead of rebuilding them.
    pub index: Arc<MemoryIndex>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentManifest {
    pub generation: u64,
    pub segments: Vec<SegmentManifestEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentManifestEntry {
    pub segment_id: String,
    pub doc_ids: Vec<String>,
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

#[derive(Debug, Clone, Serialize)]
pub struct ShardQueryFailure {
    pub segment_id: String,
    pub message: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ShardQueryCompleteness {
    pub expected_segments: Vec<String>,
    pub successful_segments: Vec<String>,
    pub failures: Vec<ShardQueryFailure>,
}

impl ShardQueryCompleteness {
    pub fn is_complete(&self) -> bool {
        self.failures.is_empty() && self.successful_segments.len() == self.expected_segments.len()
    }
}

pub(crate) fn validate_segments(segments: &[MemoryIndexSegment]) -> Result<(), String> {
    let mut segment_ids = HashSet::new();
    let mut assigned_doc_ids = HashSet::new();
    for segment in segments {
        if segment.segment_id.trim().is_empty() {
            return Err("segment id must not be empty".to_string());
        }
        if !segment_ids.insert(segment.segment_id.as_str()) {
            return Err(format!("duplicate segment id: {}", segment.segment_id));
        }
        if segment.doc_ids.is_empty() || segment.index.docs.is_empty() {
            return Err(format!("segment is empty: {}", segment.segment_id));
        }

        let mut local_doc_ids = HashSet::new();
        for doc_id in &segment.doc_ids {
            if !local_doc_ids.insert(doc_id.as_str()) {
                return Err(format!(
                    "duplicate document id in segment {}: {doc_id}",
                    segment.segment_id
                ));
            }
            if !assigned_doc_ids.insert(doc_id.as_str()) {
                return Err(format!(
                    "document is assigned to multiple segments: {doc_id}"
                ));
            }
        }
        let indexed_doc_ids = segment
            .index
            .docs
            .keys()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        if local_doc_ids != indexed_doc_ids {
            return Err(format!(
                "segment doc_ids do not match indexed documents: {}",
                segment.segment_id
            ));
        }
    }
    Ok(())
}

pub(crate) fn build_segments_by_group_id(records: &[DocRecord]) -> Vec<MemoryIndexSegment> {
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
        .map(|(segment_id, segment_records)| {
            build_memory_index_segment(segment_id, segment_records)
        })
        .collect::<Vec<_>>();
    segments.sort_by(|a, b| a.segment_id.cmp(&b.segment_id));
    segments
}

pub(crate) fn build_memory_index_segment(
    segment_id: String,
    mut segment_records: Vec<DocRecord>,
) -> MemoryIndexSegment {
    segment_records.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
    let doc_ids = segment_records
        .iter()
        .map(|record| record.doc_id.clone())
        .collect::<Vec<_>>();
    let index = Arc::new(MemoryIndex::from_records(segment_records));
    MemoryIndexSegment {
        segment_id,
        doc_ids,
        index,
    }
}

impl MemoryIndexSegment {
    pub(crate) fn team_coverage_gain(
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
            let summary = corpus_stats.summary(&self.segment_id);
            let local_weight = summary.local_term_weight(term);
            if local_weight <= 0.0 {
                continue;
            }

            let idf = corpus_stats.idf(term);
            let evidence_multiplier = summary.coverage_evidence_multiplier(term);
            if !coverage.covered_terms.contains(term) {
                gain += local_weight * idf * evidence_multiplier * 1.35;
                newly_covered_idf += idf;
                newly_covered_count += 1;
            } else {
                gain += local_weight * idf * evidence_multiplier * 0.18;
            }

            for evidence_type in summary.evidence_types_for_term(term) {
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
            .connection_terms(corpus_stats)
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

    pub(crate) fn connection_terms(&self, corpus_stats: &SegmentCorpusStats) -> HashSet<String> {
        let summary = corpus_stats.summary(&self.segment_id);
        let mut terms = HashSet::new();
        terms.extend(top_weighted_keys(&summary.entities, 16));
        terms.extend(top_weighted_keys(&summary.topics, 8));
        terms.extend(
            top_weighted_keys(&summary.local_memory, 16)
                .into_iter()
                .filter(|term| is_segment_enrichment_candidate(term)),
        );
        terms
    }

    pub(crate) fn has_temporal_signal(&self) -> bool {
        self.index
            .docs
            .values()
            .any(|record| record.timestamp.is_some() || !record.temporal_terms.is_empty())
    }

    pub(crate) fn enriched_query(
        &self,
        query: &str,
        query_terms: &HashSet<String>,
        temporal: TemporalQueryContext<'_>,
        corpus_stats: &SegmentCorpusStats,
    ) -> SegmentQueryEnrichment {
        let summary = corpus_stats.summary(&self.segment_id);
        let mut candidates: HashMap<String, (f32, HashSet<String>)> = HashMap::new();
        add_local_frequency_candidates(&mut candidates, &self.index, query_terms);
        collect_profile_candidates(&mut candidates, &summary.local_memory, "local_memory", 1.6);
        collect_profile_candidates(&mut candidates, &summary.entities, "entity", 1.4);
        collect_profile_candidates(&mut candidates, &summary.topics, "topic", 1.1);
        collect_profile_candidates(&mut candidates, &summary.terms, "term", 0.8);
        let temporal_signal =
            collect_temporal_candidates(&mut candidates, self, temporal, query_terms);

        let mut ranked = candidates
            .into_iter()
            .filter(|(term, _)| !query_terms.contains(term))
            .filter(|(term, _)| is_segment_enrichment_candidate(term))
            .filter(|(term, (_, evidence))| {
                // Temporal labels describe query-time context rather than reusable
                // lexical enrichment. Keep those signals even when their words
                // occur in multiple segments; all lexical/local candidates must
                // be unique to this segment.
                evidence.iter().all(|kind| kind.starts_with("temporal"))
                    || corpus_stats.is_unique_to_segment(term, &self.segment_id)
            })
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

    pub(crate) fn local_evidence(
        &self,
        query_terms: &HashSet<String>,
        corpus_stats: &SegmentCorpusStats,
    ) -> SegmentLocalEvidence {
        let mut differentiators = query_terms
            .iter()
            .filter_map(|term| {
                let summary = corpus_stats.summary(&self.segment_id);
                let local_weight = summary.local_term_weight(term);
                (local_weight > 0.0).then(|| LocalDifferentiator {
                    term: term.clone(),
                    weight: local_weight * corpus_stats.idf(term),
                    evidence_types: summary.evidence_types_for_term(term),
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
