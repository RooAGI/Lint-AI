//! Versioned wire types for cross-process distributed querying.
//!
//! These types describe the semantic contract independently of gRPC, HTTP, or
//! any other transport. The local segmented coordinator does not require them;
//! they are the boundary used when a worker is placed behind `lint-service`.

use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet};

pub const REMOTE_QUERY_PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteQueryRequest {
    pub protocol_version: u32,
    pub request_id: String,
    pub deadline_unix_ms: u64,
    pub coordinator_generation: u64,
    pub query: String,
    /// Kept bounded and fixed-width because this is a wire field.
    pub top_k: u32,
    #[serde(default)]
    pub filters: BTreeMap<String, String>,
    #[serde(default)]
    pub temporal: Option<RemoteTemporalContext>,
    #[serde(default)]
    pub reference_date: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteTemporalContext {
    #[serde(default)]
    pub starts_from: Option<String>,
    #[serde(default)]
    pub ends_at: Option<String>,
    pub window_days: i64,
    pub hard_filter: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteStatisticsRequest {
    pub base: RemoteQueryRequest,
    pub terms: Vec<String>,
    pub fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteStatisticsResponse {
    pub protocol_version: u32,
    pub request_id: String,
    pub worker_id: String,
    pub worker_generation: u64,
    pub statistics_generation: String,
    pub fields: BTreeMap<String, RemoteFieldStatistics>,
    pub terms: BTreeMap<String, BTreeMap<String, u64>>,
    pub completeness: RemoteQueryCompleteness,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteFieldStatistics {
    pub documents: u64,
    pub tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoteCandidateRequest {
    pub base: RemoteQueryRequest,
    pub statistics_generation: String,
    pub statistics: RemoteStatisticsSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoteStatisticsSnapshot {
    pub fields: BTreeMap<String, RemoteFieldStatistics>,
    pub terms: BTreeMap<String, BTreeMap<String, u64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoteCandidateResponse {
    pub protocol_version: u32,
    pub request_id: String,
    pub worker_id: String,
    pub worker_generation: u64,
    pub statistics_generation: String,
    pub results: Vec<RemoteSearchResult>,
    pub completeness: RemoteQueryCompleteness,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoteSearchResult {
    pub doc_id: String,
    pub source: String,
    pub group_id: Option<String>,
    pub score: f32,
    pub score_breakdown: BTreeMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteQueryCompleteness {
    pub expected_segments: Vec<String>,
    pub successful_segments: Vec<String>,
    pub failures: Vec<RemoteQueryFailure>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteQueryFailure {
    pub segment_id: String,
    pub message: String,
}

impl From<&crate::index::SearchResult> for RemoteSearchResult {
    fn from(result: &crate::index::SearchResult) -> Self {
        let breakdown = &result.score_breakdown;
        Self {
            doc_id: result.doc_id.clone(),
            source: result.source.clone(),
            group_id: result.group_id.clone(),
            score: result.score,
            score_breakdown: BTreeMap::from([
                ("lexical_score".to_string(), breakdown.lexical_score),
                ("entity_score".to_string(), breakdown.entity_score),
                ("term_score".to_string(), breakdown.term_score),
                ("claim_score".to_string(), breakdown.claim_score),
                ("semantic_score".to_string(), breakdown.semantic_score),
                ("topic_score".to_string(), breakdown.topic_score),
                ("doc_type_score".to_string(), breakdown.doc_type_score),
                ("recency_score".to_string(), breakdown.recency_score),
                ("graph_link_score".to_string(), breakdown.graph_link_score),
                (
                    "entity_graph_score".to_string(),
                    breakdown.entity_graph_score,
                ),
                (
                    "sequence_rerank_score".to_string(),
                    breakdown.sequence_rerank_score,
                ),
            ]),
        }
    }
}

impl From<&crate::segments::ShardQueryCompleteness> for RemoteQueryCompleteness {
    fn from(completeness: &crate::segments::ShardQueryCompleteness) -> Self {
        Self {
            expected_segments: completeness.expected_segments.clone(),
            successful_segments: completeness.successful_segments.clone(),
            failures: completeness
                .failures
                .iter()
                .map(|failure| RemoteQueryFailure {
                    segment_id: failure.segment_id.clone(),
                    message: failure.message.clone(),
                })
                .collect(),
        }
    }
}

impl RemoteQueryRequest {
    pub fn validate(&self, now_unix_ms: u64) -> Result<(), String> {
        if self.protocol_version != REMOTE_QUERY_PROTOCOL_VERSION {
            return Err(format!(
                "unsupported remote query protocol version: {}",
                self.protocol_version
            ));
        }
        if self.request_id.trim().is_empty() {
            return Err("remote query request_id must not be empty".to_string());
        }
        if self.deadline_unix_ms <= now_unix_ms {
            return Err("remote query deadline has expired".to_string());
        }
        if self.query.trim().is_empty() {
            return Err("remote query query must not be empty".to_string());
        }
        if !(1..=100).contains(&self.top_k) {
            return Err("remote query top_k must be between 1 and 100".to_string());
        }
        Ok(())
    }
}

impl RemoteStatisticsRequest {
    pub fn validate(&self, now_unix_ms: u64) -> Result<(), String> {
        self.base.validate(now_unix_ms)?;
        validate_unique_names("terms", &self.terms)?;
        validate_unique_names("fields", &self.fields)?;
        Ok(())
    }
}

impl RemoteCandidateRequest {
    pub fn validate(&self, now_unix_ms: u64) -> Result<(), String> {
        self.base.validate(now_unix_ms)?;
        if self.statistics_generation.trim().is_empty() {
            return Err("statistics_generation must not be empty".to_string());
        }
        Ok(())
    }
}

impl RemoteStatisticsResponse {
    pub fn validate_against(&self, request: &RemoteStatisticsRequest) -> Result<(), String> {
        validate_response_identity(
            self.protocol_version,
            &self.request_id,
            &self.worker_id,
            &request.base,
        )?;
        if self.statistics_generation.trim().is_empty() {
            return Err("statistics_generation must not be empty".to_string());
        }
        self.completeness.validate()?;
        Ok(())
    }
}

impl RemoteCandidateResponse {
    pub fn validate_against(&self, request: &RemoteCandidateRequest) -> Result<(), String> {
        validate_response_identity(
            self.protocol_version,
            &self.request_id,
            &self.worker_id,
            &request.base,
        )?;
        if self.statistics_generation != request.statistics_generation {
            return Err(format!(
                "statistics generation mismatch: response={}, request={}",
                self.statistics_generation, request.statistics_generation
            ));
        }
        self.completeness.validate()?;
        Ok(())
    }
}

impl RemoteQueryCompleteness {
    pub fn validate(&self) -> Result<(), String> {
        let expected: BTreeSet<_> = self.expected_segments.iter().collect();
        if expected.len() != self.expected_segments.len()
            || self
                .expected_segments
                .iter()
                .any(|segment| segment.trim().is_empty())
        {
            return Err("completeness expected_segments must be unique and non-empty".to_string());
        }
        let successful: BTreeSet<_> = self.successful_segments.iter().collect();
        if successful.len() != self.successful_segments.len()
            || successful.iter().any(|segment| !expected.contains(segment))
        {
            return Err("completeness successful_segments must be unique and expected".to_string());
        }
        let mut failures = BTreeSet::new();
        for failure in &self.failures {
            if failure.segment_id.trim().is_empty() || failure.message.trim().is_empty() {
                return Err("completeness failures must have segment IDs and messages".to_string());
            }
            if !expected.contains(&failure.segment_id) {
                return Err("completeness failure must refer to an expected segment".to_string());
            }
            if !failures.insert((&failure.segment_id, &failure.message)) {
                return Err("completeness failures must be unique".to_string());
            }
            if successful.contains(&failure.segment_id) {
                return Err("a segment cannot be both successful and failed".to_string());
            }
        }
        Ok(())
    }

    pub fn is_complete(&self) -> bool {
        self.failures.is_empty()
            && self
                .expected_segments
                .iter()
                .all(|segment| self.successful_segments.contains(segment))
    }

    fn merge<I>(responses: I) -> Self
    where
        I: IntoIterator,
        I::Item: Borrow<RemoteQueryCompleteness>,
    {
        let mut expected = BTreeSet::new();
        let mut successful = BTreeSet::new();
        let mut failures = BTreeMap::new();
        for completeness in responses {
            let completeness = completeness.borrow();
            expected.extend(completeness.expected_segments.iter().cloned());
            successful.extend(completeness.successful_segments.iter().cloned());
            for failure in &completeness.failures {
                failures
                    .entry((failure.segment_id.clone(), failure.message.clone()))
                    .or_insert_with(|| failure.clone());
            }
        }
        Self {
            expected_segments: expected.into_iter().collect(),
            successful_segments: successful.into_iter().collect(),
            failures: failures.into_values().collect(),
        }
    }
}

/// Sums worker-local statistics into the frozen view used by candidate scoring.
///
/// Responses must belong to one request and must have unique worker IDs. The
/// returned completeness is the union of every worker's segment-level report.
pub fn aggregate_statistics(
    request: &RemoteStatisticsRequest,
    responses: &[RemoteStatisticsResponse],
) -> Result<(RemoteStatisticsSnapshot, RemoteQueryCompleteness), String> {
    if responses.is_empty() {
        return Err("cannot aggregate an empty statistics response set".to_string());
    }
    let mut worker_ids = BTreeSet::new();
    let mut statistics_generation: Option<&str> = None;
    let mut fields = BTreeMap::<String, RemoteFieldStatistics>::new();
    let mut terms = BTreeMap::<String, BTreeMap<String, u64>>::new();

    for response in responses {
        response.validate_against(request)?;
        if !worker_ids.insert(response.worker_id.as_str()) {
            return Err(format!(
                "duplicate remote worker_id: {}",
                response.worker_id
            ));
        }
        if let Some(expected) = statistics_generation {
            if expected != response.statistics_generation {
                return Err("statistics generations do not match".to_string());
            }
        } else {
            statistics_generation = Some(&response.statistics_generation);
        }
        for (field, value) in &response.fields {
            let total = fields
                .entry(field.clone())
                .or_insert(RemoteFieldStatistics {
                    documents: 0,
                    tokens: 0,
                });
            total.documents = total.documents.saturating_add(value.documents);
            total.tokens = total.tokens.saturating_add(value.tokens);
        }
        for (term, field_counts) in &response.terms {
            let totals = terms.entry(term.clone()).or_default();
            for (field, count) in field_counts {
                let total = totals.entry(field.clone()).or_default();
                *total = total.saturating_add(*count);
            }
        }
    }

    Ok((
        RemoteStatisticsSnapshot { fields, terms },
        RemoteQueryCompleteness::merge(responses.iter().map(|response| &response.completeness)),
    ))
}

/// Validates and deterministically reduces candidate responses from workers.
///
/// Results are deduplicated by stable document ID, sorted by descending score
/// and then ascending document ID, and truncated only after the merge.
pub fn reduce_candidates(
    request: &RemoteCandidateRequest,
    responses: &[RemoteCandidateResponse],
) -> Result<(Vec<RemoteSearchResult>, RemoteQueryCompleteness), String> {
    if responses.is_empty() {
        return Err("cannot reduce an empty candidate response set".to_string());
    }
    let mut worker_ids = BTreeSet::new();
    let mut by_doc_id = BTreeMap::<String, RemoteSearchResult>::new();
    for response in responses {
        response.validate_against(request)?;
        if !worker_ids.insert(response.worker_id.as_str()) {
            return Err(format!(
                "duplicate remote worker_id: {}",
                response.worker_id
            ));
        }
        for result in &response.results {
            let replace = match by_doc_id.get(&result.doc_id) {
                None => true,
                Some(existing) => {
                    result.score.total_cmp(&existing.score).is_gt()
                        || (result.score.total_cmp(&existing.score).is_eq()
                            && canonical_result_key(result) > canonical_result_key(existing))
                }
            };
            if replace {
                by_doc_id.insert(result.doc_id.clone(), result.clone());
            }
        }
    }
    let mut results: Vec<_> = by_doc_id.into_values().collect();
    results.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.doc_id.cmp(&right.doc_id))
    });
    results.truncate(request.base.top_k as usize);
    Ok((
        results,
        RemoteQueryCompleteness::merge(responses.iter().map(|response| &response.completeness)),
    ))
}

fn canonical_result_key(result: &RemoteSearchResult) -> (String, Option<String>, String) {
    (
        result.source.clone(),
        result.group_id.clone(),
        serde_json::to_string(&result.score_breakdown).unwrap_or_default(),
    )
}

fn validate_response_identity(
    protocol_version: u32,
    request_id: &str,
    worker_id: &str,
    request: &RemoteQueryRequest,
) -> Result<(), String> {
    if protocol_version != REMOTE_QUERY_PROTOCOL_VERSION {
        return Err(format!(
            "unsupported remote query protocol version: {protocol_version}"
        ));
    }
    if request_id != request.request_id {
        return Err("remote response request_id does not match request".to_string());
    }
    if worker_id.trim().is_empty() {
        return Err("remote response worker_id must not be empty".to_string());
    }
    Ok(())
}

fn validate_unique_names(kind: &str, names: &[String]) -> Result<(), String> {
    let mut unique = BTreeSet::new();
    for name in names {
        if name.trim().is_empty() || !unique.insert(name) {
            return Err(format!("remote query {kind} must be unique and non-empty"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> RemoteQueryRequest {
        RemoteQueryRequest {
            protocol_version: REMOTE_QUERY_PROTOCOL_VERSION,
            request_id: "q-1".to_string(),
            deadline_unix_ms: 2_000,
            coordinator_generation: 7,
            query: "deployment routing".to_string(),
            top_k: 5,
            filters: BTreeMap::new(),
            temporal: None,
            reference_date: None,
        }
    }

    #[test]
    fn query_request_round_trips_as_json() {
        let request = request();
        let encoded = serde_json::to_string(&request).unwrap();
        assert_eq!(
            serde_json::from_str::<RemoteQueryRequest>(&encoded).unwrap(),
            request
        );
    }

    #[test]
    fn request_validation_rejects_expired_and_invalid_requests() {
        let mut request = request();
        assert!(request.validate(1_999).is_ok());
        assert!(request.validate(2_000).is_err());
        request.top_k = 0;
        assert!(request.validate(1_000).is_err());
    }

    #[test]
    fn phase_requests_validate_dimensions_and_generation() {
        let stats = RemoteStatisticsRequest {
            base: request(),
            terms: vec!["routing".to_string(), "routing".to_string()],
            fields: vec!["content".to_string()],
        };
        assert!(stats.validate(1_000).is_err());

        let candidate = RemoteCandidateRequest {
            base: request(),
            statistics_generation: " ".to_string(),
            statistics: RemoteStatisticsSnapshot {
                fields: BTreeMap::new(),
                terms: BTreeMap::new(),
            },
        };
        assert!(candidate.validate(1_000).is_err());
    }

    #[test]
    fn candidate_response_rejects_statistics_generation_mismatch() {
        let base = request();
        let request = RemoteCandidateRequest {
            base,
            statistics_generation: "q-1:stats".to_string(),
            statistics: RemoteStatisticsSnapshot {
                fields: BTreeMap::new(),
                terms: BTreeMap::new(),
            },
        };
        let response = RemoteCandidateResponse {
            protocol_version: REMOTE_QUERY_PROTOCOL_VERSION,
            request_id: "q-1".to_string(),
            worker_id: "worker-a".to_string(),
            worker_generation: 42,
            statistics_generation: "q-1:stale".to_string(),
            results: vec![],
            completeness: RemoteQueryCompleteness {
                expected_segments: vec![],
                successful_segments: vec![],
                failures: vec![],
            },
        };
        assert!(response.validate_against(&request).is_err());
    }

    fn completeness(segment_id: &str) -> RemoteQueryCompleteness {
        RemoteQueryCompleteness {
            expected_segments: vec![segment_id.to_string()],
            successful_segments: vec![segment_id.to_string()],
            failures: vec![],
        }
    }

    fn stats_response(worker_id: &str, segment_id: &str) -> RemoteStatisticsResponse {
        RemoteStatisticsResponse {
            protocol_version: REMOTE_QUERY_PROTOCOL_VERSION,
            request_id: "q-1".to_string(),
            worker_id: worker_id.to_string(),
            worker_generation: 1,
            statistics_generation: "q-1:stats".to_string(),
            fields: BTreeMap::from([(
                "content".to_string(),
                RemoteFieldStatistics {
                    documents: 2,
                    tokens: 10,
                },
            )]),
            terms: BTreeMap::from([(
                "routing".to_string(),
                BTreeMap::from([("content".to_string(), 1)]),
            )]),
            completeness: completeness(segment_id),
        }
    }

    #[test]
    fn statistics_aggregation_sums_values_and_completeness() {
        let request = RemoteStatisticsRequest {
            base: request(),
            terms: vec!["routing".to_string()],
            fields: vec!["content".to_string()],
        };
        let (snapshot, completeness) = aggregate_statistics(
            &request,
            &[stats_response("a", "a1"), stats_response("b", "b1")],
        )
        .unwrap();
        assert_eq!(snapshot.fields["content"].documents, 4);
        assert_eq!(snapshot.terms["routing"]["content"], 2);
        assert_eq!(completeness.expected_segments, vec!["a1", "b1"]);
        assert!(completeness.is_complete());
    }

    fn candidate_response(worker_id: &str, doc_id: &str, score: f32) -> RemoteCandidateResponse {
        RemoteCandidateResponse {
            protocol_version: REMOTE_QUERY_PROTOCOL_VERSION,
            request_id: "q-1".to_string(),
            worker_id: worker_id.to_string(),
            worker_generation: 1,
            statistics_generation: "q-1:stats".to_string(),
            results: vec![RemoteSearchResult {
                doc_id: doc_id.to_string(),
                source: format!("{doc_id}.md"),
                group_id: None,
                score,
                score_breakdown: BTreeMap::new(),
            }],
            completeness: completeness(worker_id),
        }
    }

    #[test]
    fn candidate_reduction_is_order_independent_and_truncates_after_merge() {
        let request = RemoteCandidateRequest {
            base: request(),
            statistics_generation: "q-1:stats".to_string(),
            statistics: RemoteStatisticsSnapshot {
                fields: BTreeMap::new(),
                terms: BTreeMap::new(),
            },
        };
        let a = candidate_response("a", "same", 2.0);
        let b = candidate_response("b", "other", 3.0);
        let (first, first_completeness) =
            reduce_candidates(&request, &[a.clone(), b.clone()]).unwrap();
        let (second, second_completeness) = reduce_candidates(&request, &[b, a]).unwrap();
        assert_eq!(first, second);
        assert_eq!(first_completeness, second_completeness);
        assert_eq!(first[0].doc_id, "other");
        assert_eq!(first.len(), 2);
    }

    #[test]
    fn malformed_completeness_is_rejected() {
        let invalid = RemoteQueryCompleteness {
            expected_segments: vec!["segment-a".to_string()],
            successful_segments: vec!["segment-a".to_string()],
            failures: vec![RemoteQueryFailure {
                segment_id: "segment-a".to_string(),
                message: "also failed".to_string(),
            }],
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn local_completeness_maps_to_remote_contract() {
        let local = crate::segments::ShardQueryCompleteness {
            expected_segments: vec!["segment-a".to_string()],
            successful_segments: vec![],
            failures: vec![crate::segments::ShardQueryFailure {
                segment_id: "segment-a".to_string(),
                message: "deadline".to_string(),
            }],
        };
        let remote = RemoteQueryCompleteness::from(&local);
        assert_eq!(remote.expected_segments, vec!["segment-a"]);
        assert_eq!(remote.failures[0].message, "deadline");
        assert!(!remote.is_complete());
    }
}
