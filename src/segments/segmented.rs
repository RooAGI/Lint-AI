use crate::index::{
    DocRecord, GlobalBm25Statistics, QueryTimings, SearchResult, TemporalQueryContext,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::catalog::*;
use super::diagnostics::*;
use super::model::*;
use super::query::*;
use super::routing::*;

pub struct SegmentedMemoryIndex {
    pub segments: Vec<MemoryIndexSegment>,
    pub(crate) catalog: SegmentCatalog,
    pub(crate) global_statistics: GlobalBm25Statistics,
    pub(crate) generation: u64,
}

impl SegmentedMemoryIndex {
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    pub fn query_single_segment(
        &self,
        query: &str,
        top_k: usize,
        temporal: TemporalQueryContext<'_>,
        reference_date: Option<&str>,
    ) -> Option<(Vec<SearchResult>, QueryTimings)> {
        let segment = self.segments.first()?;
        let (results, timings, _) =
            segment
                .index
                .query_with_temporal_context_at(query, top_k, temporal, reference_date);
        Some((results, timings))
    }
    /// Returns document IDs matching a generic filter across all logical
    /// segments. Filtering remains owned by the underlying MemoryIndex; this
    /// method only unions the per-segment results for the snapshot.
    pub fn doc_ids_matching_filters(
        &self,
        filters: &std::collections::BTreeMap<String, String>,
    ) -> Option<HashSet<String>> {
        if filters.is_empty() {
            return None;
        }
        Some(
            self.segments
                .iter()
                .flat_map(|segment| {
                    segment
                        .index
                        .doc_ids_matching_filters(filters)
                        .into_iter()
                        .flatten()
                })
                .collect(),
        )
    }

    pub fn doc_bitmaps_matching_filters(
        &self,
        filters: &std::collections::BTreeMap<String, String>,
    ) -> HashMap<String, roaring::RoaringBitmap> {
        self.segments
            .iter()
            .filter_map(|segment| {
                segment
                    .index
                    .doc_bitmap_matching_filters(filters)
                    .map(|bitmap| (segment.segment_id.clone(), bitmap))
            })
            .collect()
    }

    pub fn from_segments(segments: Vec<MemoryIndexSegment>) -> Result<Self, String> {
        Self::from_segments_with_generation(segments, 0)
    }

    pub fn from_segments_with_generation(
        segments: Vec<MemoryIndexSegment>,
        generation: u64,
    ) -> Result<Self, String> {
        validate_segments(&segments)?;
        let catalog = SegmentCatalog::from_segments(&segments, generation);
        let global_statistics = GlobalBm25Statistics::from_indexes_with_generation(
            segments.iter().map(|segment| segment.index.as_ref()),
            generation,
        );
        Ok(Self {
            segments,
            catalog,
            global_statistics,
            generation,
        })
    }

    pub fn from_records_by_group_id(records: &[DocRecord]) -> Self {
        Self::from_records_by_group_id_with_generation(records, 0)
    }

    pub fn from_records_by_group_id_with_generation(
        records: &[DocRecord],
        generation: u64,
    ) -> Self {
        let segments = build_segments_by_group_id(records);
        Self::from_segments_with_generation(segments, generation)
            .expect("grouped records must produce structurally valid segments")
    }

    pub fn from_records_by_manifest(
        records: &[DocRecord],
        manifest: &SegmentManifest,
        generation: u64,
    ) -> Result<Self, String> {
        let records_by_id = records
            .iter()
            .map(|record| (record.doc_id.as_str(), record))
            .collect::<HashMap<_, _>>();
        let mut seen_segment_ids = HashSet::new();
        let mut seen_doc_ids = HashSet::new();
        let mut segments = Vec::with_capacity(manifest.segments.len());

        for entry in &manifest.segments {
            if !seen_segment_ids.insert(entry.segment_id.as_str()) {
                return Err(format!(
                    "duplicate segment id in manifest: {}",
                    entry.segment_id
                ));
            }
            if entry.doc_ids.is_empty() {
                return Err(format!("manifest segment is empty: {}", entry.segment_id));
            }
            let mut segment_records = Vec::with_capacity(entry.doc_ids.len());
            for doc_id in &entry.doc_ids {
                if !seen_doc_ids.insert(doc_id.as_str()) {
                    return Err(format!("duplicate document id in manifest: {doc_id}"));
                }
                let Some(record) = records_by_id.get(doc_id.as_str()) else {
                    return Err(format!("manifest references missing document: {doc_id}"));
                };
                segment_records.push((*record).clone());
            }
            segments.push(build_memory_index_segment(
                entry.segment_id.clone(),
                segment_records,
            ));
        }

        if seen_doc_ids.len() != records.len() {
            return Err("manifest does not assign every document".to_string());
        }
        Self::from_segments_with_generation(segments, generation)
    }

    /// Rebuild a snapshot from the previous one, reusing every segment whose
    /// document set is unchanged and none of whose documents were
    /// re-processed since the previous snapshot. Reuse clones the segment's
    /// shared index, so it is O(1) per segment regardless of readers.
    ///
    /// Correctness rests on one invariant: a segment's index is a pure
    /// function of its documents' records, and records only change for
    /// re-processed documents. A segment is therefore reusable exactly when
    /// its document id set is unchanged and none of its documents appear in
    /// `reprocessed_doc_ids`.
    ///
    /// Only segments that are actually rebuilt pay for record clones: the
    /// grouping and the reuse check work on borrowed document ids, so a
    /// single-write refresh never clones the corpus.
    pub fn refresh_incremental(
        previous: &Self,
        records: &HashMap<String, DocRecord>,
        reprocessed_doc_ids: &HashSet<String>,
        generation: u64,
    ) -> Result<Self, String> {
        let mut grouped: HashMap<&str, Vec<&str>> = HashMap::new();
        for record in records.values() {
            let segment_id = record.group_id.as_deref().unwrap_or("ungrouped");
            grouped
                .entry(segment_id)
                .or_default()
                .push(record.doc_id.as_str());
        }

        let mut segment_ids: Vec<&str> = grouped.keys().copied().collect();
        segment_ids.sort_unstable();
        let mut segments = Vec::with_capacity(segment_ids.len());
        for segment_id in segment_ids {
            let mut new_doc_ids = grouped.remove(segment_id).unwrap_or_default();
            new_doc_ids.sort_unstable();
            let reusable = previous
                .segments
                .iter()
                .find(|segment| segment.segment_id.as_str() == segment_id)
                .filter(|segment| {
                    segment.doc_ids.len() == new_doc_ids.len()
                        && segment
                            .doc_ids
                            .iter()
                            .map(String::as_str)
                            .eq(new_doc_ids.iter().copied())
                        && !segment
                            .doc_ids
                            .iter()
                            .any(|doc_id| reprocessed_doc_ids.contains(doc_id))
                });
            match reusable {
                Some(previous_segment) => segments.push(MemoryIndexSegment {
                    segment_id: segment_id.to_string(),
                    doc_ids: previous_segment.doc_ids.clone(),
                    index: Arc::clone(&previous_segment.index),
                    record_dates: std::sync::OnceLock::new(),
                }),
                None => {
                    let mut segment_records = Vec::with_capacity(new_doc_ids.len());
                    for doc_id in &new_doc_ids {
                        segment_records.push(records.get(*doc_id).cloned().ok_or_else(|| {
                            format!("grouped document missing from records: {doc_id}")
                        })?);
                    }
                    segments.push(build_memory_index_segment(
                        segment_id.to_string(),
                        segment_records,
                    ))
                }
            }
        }
        validate_segments(&segments)?;
        let catalog = SegmentCatalog::refresh_incremental(
            &previous.catalog,
            &previous.segments,
            &segments,
            generation,
        );
        let global_statistics = GlobalBm25Statistics::from_indexes_with_generation(
            segments.iter().map(|segment| segment.index.as_ref()),
            generation,
        );
        Ok(Self {
            segments,
            catalog,
            global_statistics,
            generation,
        })
    }

    pub fn manifest(&self) -> SegmentManifest {
        SegmentManifest {
            generation: self.generation,
            segments: self
                .segments
                .iter()
                .map(|segment| SegmentManifestEntry {
                    segment_id: segment.segment_id.clone(),
                    doc_ids: segment.doc_ids.clone(),
                })
                .collect(),
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn routing_summary_counts(&self, segment_id: &str) -> (usize, usize, usize, usize) {
        debug_assert_eq!(self.catalog.generation, self.generation);
        let summary = self.catalog.summary(segment_id);
        (
            summary.terms.len(),
            summary.entities.len(),
            summary.topics.len(),
            summary.local_memory.len(),
        )
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
            &self.catalog,
        )
    }

    pub fn route_with_strategy(
        &self,
        query: &str,
        strategy: SegmentRoutingStrategy,
    ) -> Vec<SegmentRoute> {
        route_segments_with_corpus_stats(query, &self.segments, strategy, &self.catalog)
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
            &self.catalog,
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
            false,
            strategy,
            TemporalQueryContext::default(),
            None,
            &self.catalog,
            Some(&self.global_statistics),
            self.generation,
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
        self.query_with_temporal_context_at_and_diagnostics_and_strategy(
            query,
            top_k,
            segment_limit,
            strategy,
            temporal,
            None,
        )
    }

    pub fn query_with_temporal_context_at_and_diagnostics_and_strategy(
        &self,
        query: &str,
        top_k: usize,
        segment_limit: usize,
        strategy: SegmentRoutingStrategy,
        temporal: TemporalQueryContext<'_>,
        reference_date: Option<&str>,
    ) -> SegmentQueryOutput {
        query_top_segments_with_corpus_stats_and_strategy(
            query,
            top_k,
            &self.segments,
            segment_limit,
            false,
            strategy,
            temporal,
            reference_date,
            &self.catalog,
            Some(&self.global_statistics),
            self.generation,
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
        query_top_segments_with_enrichment_and_strategy(
            query,
            top_k,
            &self.segments,
            segment_limit,
            strategy,
            temporal,
            &self.catalog,
            Some(&self.global_statistics),
            self.generation,
            SegmentEnrichmentKind::Segment,
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
        query_top_segments_with_enrichment_and_strategy(
            query,
            top_k,
            &self.segments,
            segment_limit,
            strategy,
            temporal,
            &self.catalog,
            Some(&self.global_statistics),
            self.generation,
            SegmentEnrichmentKind::RouteAware,
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
        query_top_segments_with_enrichment_and_strategy(
            query,
            top_k,
            &self.segments,
            segment_limit,
            strategy,
            temporal,
            &self.catalog,
            Some(&self.global_statistics),
            self.generation,
            SegmentEnrichmentKind::SessionAggregated,
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
        reference_date: Option<&str>,
    ) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
        query_top_segments_with_enrichment_and_strategy(
            query,
            top_k,
            &self.segments,
            max_segment_limit,
            strategy,
            temporal,
            &self.catalog,
            Some(&self.global_statistics),
            self.generation,
            SegmentEnrichmentKind::Adaptive {
                base_segment_limit,
                max_segment_limit,
                reference_date,
            },
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
        reference_date: Option<&str>,
    ) -> (SegmentQueryOutput, SegmentSpecificEnrichmentDiagnostics) {
        query_top_segments_with_enrichment_and_strategy(
            query,
            top_k,
            &self.segments,
            max_segment_limit,
            strategy,
            temporal,
            &self.catalog,
            Some(&self.global_statistics),
            self.generation,
            SegmentEnrichmentKind::AdaptiveRouteAware {
                base_segment_limit,
                max_segment_limit,
                reference_date,
            },
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
        query_top_segments_with_enrichment_and_strategy(
            query,
            top_k,
            &self.segments,
            segment_limit,
            strategy,
            temporal,
            &self.catalog,
            Some(&self.global_statistics),
            self.generation,
            SegmentEnrichmentKind::TemporalPath,
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
        query_top_segments_with_enrichment_and_strategy(
            query,
            top_k,
            &self.segments,
            segment_limit,
            strategy,
            temporal,
            &self.catalog,
            Some(&self.global_statistics),
            self.generation,
            SegmentEnrichmentKind::ConnectedSegment,
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
        query_top_segments_with_enrichment_and_strategy(
            query,
            top_k,
            &self.segments,
            segment_limit,
            strategy,
            temporal,
            &self.catalog,
            Some(&self.global_statistics),
            self.generation,
            SegmentEnrichmentKind::MissingCoverageRecovery,
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
            true,
            SegmentRoutingStrategy::SparseOverlap,
            temporal,
            None,
            &self.catalog,
            Some(&self.global_statistics),
            self.generation,
        )
    }
}

/// Uniform reciprocal rank fusion over several retrieval modes' ranked
/// lists (Cormack et al., SIGIR 2009): `score(doc) = sum_modes 1 / (60 +
/// rank)`. Label-free — uses only ranks, never scores or gold labels — so
/// cross-mode score-scale mismatch cannot crowd out any single mode's
/// ranking. Deterministic: ties broken by `doc_id` ascending.
///
/// This is how the corpus-wide ("global") arm stays part of the query path:
/// the routed segment arm's ranking is fused with an all-segments ranking,
/// so a router miss (wrong segments selected) can still surface the
/// relevant document through the global arm's rank.
pub fn reciprocal_rank_fusion(mode_results: &[&[SearchResult]], top_k: usize) -> Vec<SearchResult> {
    const RRF_K: f64 = 60.0;
    let mut fused: HashMap<&str, (SearchResult, f64)> = HashMap::new();
    for results in mode_results {
        for (rank, result) in results.iter().enumerate() {
            let entry = fused
                .entry(result.doc_id.as_str())
                .or_insert_with(|| (result.clone(), 0.0));
            entry.1 += 1.0 / (RRF_K + rank as f64 + 1.0);
        }
    }
    let mut merged: Vec<(SearchResult, f64)> = fused.into_values().collect();
    merged.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.doc_id.cmp(&right.0.doc_id))
    });
    merged.truncate(top_k);
    merged.into_iter().map(|(result, _)| result).collect()
}
