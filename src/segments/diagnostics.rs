use crate::index::SearchResult;
use serde::Serialize;
use std::collections::HashMap;

use super::model::*;

#[derive(Debug, Clone, Default, Serialize)]
pub struct SegmentQueryDiagnostics {
    pub snapshot_generation: u64,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shard_completeness: Option<ShardQueryCompleteness>,
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
