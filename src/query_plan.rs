//! Shared query preparation.
//!
//! Turning a raw user query into "what we actually search for" has two parts:
//! analyzing the query (intent, temporal hints, augmentation) and turning that
//! analysis into a [`TemporalQueryContext`]. Every caller that accepts a raw
//! query from a user. The CLI and memory server should go through here, so
//! the two cannot drift apart.
//!
//! Callers that have already built a context, or that deliberately want the
//! default one, keep using `query_with_temporal_context` directly.

use crate::index::{
    MemoryIndex, QueryDiagnostics, QueryTimings, SearchResult, TemporalQueryContext,
    TemporalQueryHint,
};
use crate::query_semantics::{
    analyze_query, parse_reference_date, resolve_anchor_window, resolve_temporal_anchor,
    temporal_anchor_is_span, QueryAnalysis, QueryTimeHint,
};
use crate::semantic_relations::{is_historical_query, SemanticRelationStore, SemanticStatus};
use chrono::{NaiveDate, TimeDelta, Utc};
use std::collections::HashSet;

/// A raw query plus the analysis derived from it.
///
/// Holds the analysis so the borrowed [`TemporalQueryContext`] it hands out
/// stays valid; build one per query and keep it alive for the search.
pub struct PreparedQuery {
    analysis: QueryAnalysis,
    reference_date: Option<String>,
    resolved_anchor_date: Option<String>,
    resolved_start: Option<String>,
    resolved_end: Option<String>,
    resolved_window: Option<(NaiveDate, NaiveDate)>,
}

impl PreparedQuery {
    pub fn new(query: &str) -> Self {
        Self::build(analyze_query(query), None)
    }

    /// Builds a query whose relative temporal language is resolved against
    /// `reference_date` instead of the machine clock.
    pub fn new_at(query: &str, reference_date: &str) -> Self {
        Self::build(analyze_query(query), Some(reference_date.to_string()))
    }

    fn build(analysis: QueryAnalysis, reference_date: Option<String>) -> Self {
        let resolution_date = reference_date
            .as_deref()
            .and_then(parse_reference_date)
            .unwrap_or_else(|| Utc::now().date_naive());
        let resolved = analysis.temporal.as_ref().and_then(|temporal| {
            if temporal.source == "session-anchor" {
                return None;
            }
            let anchor = resolve_temporal_anchor(&temporal.phrase, resolution_date)?;
            let is_span = temporal_anchor_is_span(&temporal.phrase, &analysis.original_query);
            let window =
                resolve_anchor_window(&temporal.phrase, &analysis.original_query, resolution_date)?;
            Some((anchor, is_span, window))
        });
        let (resolved_anchor_date, resolved_start, resolved_end, resolved_window) =
            if let Some((anchor, is_span, window)) = resolved {
                (
                    Some(
                        if is_span { window.1 } else { anchor }
                            .format("%Y-%m-%d")
                            .to_string(),
                    ),
                    is_span.then(|| window.0.format("%Y-%m-%d").to_string()),
                    is_span.then(|| window.1.format("%Y-%m-%d").to_string()),
                    Some(window),
                )
            } else {
                (None, None, None, None)
            };
        Self {
            analysis,
            reference_date,
            resolved_anchor_date,
            resolved_start,
            resolved_end,
            resolved_window,
        }
    }

    pub fn reference_date(&self) -> Option<&str> {
        self.reference_date.as_deref()
    }

    /// Reuses an analysis the caller already computed, so no query is analyzed
    /// twice on paths that need the analysis for other reasons too.
    pub(crate) fn from_analysis(analysis: QueryAnalysis) -> Self {
        Self::build(analysis, None)
    }

    pub fn analysis(&self) -> &QueryAnalysis {
        &self.analysis
    }

    fn into_analysis(self) -> QueryAnalysis {
        self.analysis
    }

    /// The text to search with: the augmented query, not the raw input.
    pub fn search_query(&self) -> &str {
        &self.analysis.augmented_query
    }

    /// The context implied by the analysis. `allowed_doc_ids` is left unset;
    /// callers that scope a search assign it themselves.
    pub fn temporal_context(&self) -> TemporalQueryContext<'_> {
        // A seeded session anchor ("temporal anchor: YYYY-MM-DD") carries an
        // absolute date, so it needs no reference clock: center temporal
        // scoring on it and pre-filter routing to its ±7-day point window,
        // mirroring resolve_anchor_window's point behavior. Relative phrases
        // are resolved once when this PreparedQuery is built.
        let session_anchor = self
            .analysis
            .temporal
            .as_ref()
            .filter(|temporal| temporal.source == "session-anchor")
            .and_then(|temporal| {
                let resolved_at = temporal.resolved_at.as_deref()?;
                let date_str = resolved_at.get(..10)?;
                let date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d").ok()?;
                let window = (
                    date.checked_sub_signed(TimeDelta::days(7))?,
                    date.checked_add_signed(TimeDelta::days(7))?,
                );
                Some((resolved_at, window))
            });
        let anchor_date = session_anchor
            .map(|(date, _)| date)
            .or(self.resolved_anchor_date.as_deref());
        let anchor_window = session_anchor
            .map(|(_, window)| window)
            .or(self.resolved_window);
        TemporalQueryContext {
            starts_from: self.resolved_start.as_deref(),
            ends_at: self.resolved_end.as_deref(),
            window_days: match self.analysis.time_hint {
                Some(QueryTimeHint::Past) => 365,
                Some(QueryTimeHint::Present) => 30,
                Some(QueryTimeHint::Ongoing) => 14,
                Some(QueryTimeHint::Mixed) => 30,
                None => 7,
            },
            // A resolved closed interval is an explicit query constraint.
            // Enforce it during candidate collection so out-of-window
            // documents cannot consume the ranking budget. Point dates keep
            // their existing proximity behavior because their surrounding
            // memories may still carry useful event context.
            hard_filter: self.resolved_start.is_some() && self.resolved_end.is_some(),
            time_hint: self
                .analysis
                .time_hint
                .map(|hint| match hint {
                    QueryTimeHint::Past => TemporalQueryHint::Past,
                    QueryTimeHint::Present => TemporalQueryHint::Present,
                    QueryTimeHint::Ongoing => TemporalQueryHint::Ongoing,
                    QueryTimeHint::Mixed => TemporalQueryHint::Mixed,
                })
                .filter(|_| self.analysis.temporal.is_some()),
            query_routing_intent: self.analysis.query_routing_intent,
            evidence_requirements: self.analysis.evidence_requirements,
            has_explicit_temporal: self.analysis.temporal.is_some(),
            anchor_date,
            anchor_window,
            allowed_doc_ids: None,
            allowed_doc_bitmap: None,
            allowed_segment_doc_bitmaps: None,
        }
    }

    /// Executes this prepared query against a single memory index.
    ///
    /// This is the canonical adapter from query analysis into index execution:
    /// callers provide only document scoping. Temporal context, query routing,
    /// augmented query text, and the optional reference clock always come from
    /// the same `PreparedQuery` instance.
    pub fn execute_on_index(
        &self,
        index: &MemoryIndex,
        top_k: usize,
        allowed_doc_ids: Option<&HashSet<String>>,
    ) -> (Vec<SearchResult>, QueryTimings, QueryDiagnostics) {
        let mut context = self.temporal_context();
        context.allowed_doc_ids = allowed_doc_ids;
        index.query_with_temporal_context_at(
            self.search_query(),
            top_k,
            context,
            self.reference_date(),
        )
    }

    /// Executes against a precomputed numeric filter bitmap. This avoids
    /// rebuilding a string-ID allow-list when the index already owns the
    /// corresponding bitmap postings.
    pub fn execute_on_index_with_bitmap(
        &self,
        index: &MemoryIndex,
        top_k: usize,
        allowed_doc_bitmap: Option<&roaring::RoaringBitmap>,
    ) -> (Vec<SearchResult>, QueryTimings, QueryDiagnostics) {
        let mut context = self.temporal_context();
        context.allowed_doc_bitmap = allowed_doc_bitmap;
        index.query_with_temporal_context_at(
            self.search_query(),
            top_k,
            context,
            self.reference_date(),
        )
    }

    /// Intersects caller-provided document filters with current-state semantic policy.
    /// Historical queries retain superseded documents; ordinary queries suppress them.
    pub fn semantic_allowed_doc_ids(
        &self,
        base_allowed_doc_ids: Option<HashSet<String>>,
        semantic_relations: &SemanticRelationStore,
        document_ids: &[String],
    ) -> Option<HashSet<String>> {
        let semantic_allowed = if semantic_relations.is_empty()
            || is_historical_query(&self.analysis.original_query)
        {
            None
        } else {
            let has_superseded = document_ids.iter().any(|doc_id| {
                semantic_relations.document_state(doc_id).status == Some(SemanticStatus::Superseded)
            });
            has_superseded.then(|| {
                document_ids
                    .iter()
                    .filter(|doc_id| {
                        semantic_relations.document_state(doc_id).status
                            != Some(SemanticStatus::Superseded)
                    })
                    .cloned()
                    .collect()
            })
        };
        intersect_allowed_doc_ids(base_allowed_doc_ids, semantic_allowed)
    }

    pub(crate) fn semantic_visibility_requires_doc_ids(
        &self,
        semantic_relations: &SemanticRelationStore,
    ) -> bool {
        !semantic_relations.is_empty()
            && !is_historical_query(&self.analysis.original_query)
            && semantic_relations.has_superseded_documents()
    }

    /// Applies semantic provenance consistently after ranking.
    pub fn annotate_semantic_results(
        &self,
        results: Vec<SearchResult>,
        semantic_relations: &SemanticRelationStore,
        top_k: usize,
    ) -> Vec<SearchResult> {
        let historical = is_historical_query(&self.analysis.original_query);
        results
            .into_iter()
            .map(|mut result| {
                let state = semantic_relations.document_state(&result.doc_id);
                result.semantic_status =
                    if historical && state.status == Some(SemanticStatus::Superseded) {
                        Some(SemanticStatus::Historical)
                    } else {
                        state.status
                    };
                result.superseded_by = state.superseded_by;
                result.relation_confidence = state.relation_confidence;
                result.relation_evidence = state.evidence;
                result
            })
            .take(top_k)
            .collect()
    }

    /// Canonical high-level execution for a prepared query on a single index.
    ///
    /// Both the CLI cache path and `IndexStore` use this method, so temporal
    /// interpretation, semantic suppression, and provenance annotation cannot drift.
    pub fn execute_on_index_with_semantics(
        &self,
        index: &MemoryIndex,
        top_k: usize,
        base_allowed_doc_ids: Option<HashSet<String>>,
        semantic_relations: &SemanticRelationStore,
        document_ids: &[String],
    ) -> (Vec<SearchResult>, QueryTimings, QueryDiagnostics) {
        let allowed_doc_ids =
            self.semantic_allowed_doc_ids(base_allowed_doc_ids, semantic_relations, document_ids);
        let (results, timings, diagnostics) =
            self.execute_on_index(index, top_k, allowed_doc_ids.as_ref());
        (
            self.annotate_semantic_results(results, semantic_relations, top_k),
            timings,
            diagnostics,
        )
    }
}

fn intersect_allowed_doc_ids(
    left: Option<HashSet<String>>,
    right: Option<HashSet<String>>,
) -> Option<HashSet<String>> {
    match (left, right) {
        (None, None) => None,
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (Some(left), Some(right)) => Some(left.intersection(&right).cloned().collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_semantics::QueryRoutingIntent;

    #[test]
    fn count_queries_carry_count_intent() {
        let prepared = PreparedQuery::new("How many projects did I ship?");
        assert_eq!(
            prepared.temporal_context().query_routing_intent,
            Some(QueryRoutingIntent::Count)
        );
    }

    #[test]
    fn search_query_is_the_augmented_query() {
        let prepared = PreparedQuery::new("what database did I pick");
        assert_eq!(
            prepared.search_query(),
            prepared.analysis().augmented_query.as_str()
        );
    }

    #[test]
    fn explicit_reference_date_is_preserved() {
        let prepared = PreparedQuery::new_at("What happened two weeks ago?", "2024-05-10");
        assert_eq!(prepared.reference_date(), Some("2024-05-10"));
        assert!(PreparedQuery::new("anything").reference_date().is_none());
    }

    #[test]
    fn relative_temporal_anchor_uses_explicit_reference_date() {
        let prepared = PreparedQuery::new_at(
            "What was the life event of one of my relatives that I participated in a week ago?",
            "2023-06-22",
        );

        assert_eq!(
            prepared
                .analysis()
                .temporal
                .as_ref()
                .and_then(|temporal| temporal.resolved_at.as_deref()),
            None,
            "relative phrases must not be pre-resolved against the machine clock"
        );
        let context = prepared.temporal_context();
        assert_eq!(context.anchor_date, Some("2023-06-15"));
        assert_eq!(context.starts_from, None);
        assert_eq!(context.ends_at, None);
        assert!(!context.hard_filter);
    }

    #[test]
    fn context_leaves_scoping_to_the_caller() {
        let prepared = PreparedQuery::new("anything");
        assert!(prepared.temporal_context().allowed_doc_ids.is_none());
    }
}
