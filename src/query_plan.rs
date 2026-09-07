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
use crate::query_semantics::{analyze_query, QueryAnalysis, QueryTimeHint};
use crate::semantic_relations::{is_historical_query, SemanticRelationStore, SemanticStatus};
use std::collections::HashSet;

/// A raw query plus the analysis derived from it.
///
/// Holds the analysis so the borrowed [`TemporalQueryContext`] it hands out
/// stays valid; build one per query and keep it alive for the search.
pub struct PreparedQuery {
    analysis: QueryAnalysis,
    reference_date: Option<String>,
}

impl PreparedQuery {
    pub fn new(query: &str) -> Self {
        Self {
            analysis: analyze_query(query),
            reference_date: None,
        }
    }

    /// Builds a query whose relative temporal language is resolved against
    /// `reference_date` instead of the machine clock.
    pub fn new_at(query: &str, reference_date: &str) -> Self {
        Self {
            analysis: analyze_query(query),
            reference_date: Some(reference_date.to_string()),
        }
    }

    pub fn reference_date(&self) -> Option<&str> {
        self.reference_date.as_deref()
    }

    /// Reuses an analysis the caller already computed, so no query is analyzed
    /// twice on paths that need the analysis for other reasons too.
    pub fn from_analysis(analysis: QueryAnalysis) -> Self {
        Self {
            analysis,
            reference_date: None,
        }
    }

    pub fn analysis(&self) -> &QueryAnalysis {
        &self.analysis
    }

    pub fn into_analysis(self) -> QueryAnalysis {
        self.analysis
    }

    /// The text to search with: the augmented query, not the raw input.
    pub fn search_query(&self) -> &str {
        &self.analysis.augmented_query
    }

    /// The context implied by the analysis. `allowed_doc_ids` is left unset;
    /// callers that scope a search assign it themselves.
    pub fn temporal_context(&self) -> TemporalQueryContext<'_> {
        TemporalQueryContext {
            starts_from: None,
            ends_at: None,
            window_days: match self.analysis.time_hint {
                Some(QueryTimeHint::Past) => 365,
                Some(QueryTimeHint::Present) => 30,
                Some(QueryTimeHint::Ongoing) => 14,
                Some(QueryTimeHint::Mixed) => 30,
                None => 7,
            },
            hard_filter: false,
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
            has_explicit_temporal: self.analysis.temporal.is_some(),
            allowed_doc_ids: None,
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

    /// Intersects caller-provided document filters with current-state semantic policy.
    /// Historical queries retain superseded documents; ordinary queries suppress them.
    pub fn semantic_allowed_doc_ids(
        &self,
        base_allowed_doc_ids: Option<HashSet<String>>,
        semantic_relations: &SemanticRelationStore,
        document_ids: &[String],
    ) -> Option<HashSet<String>> {
        let semantic_allowed = if is_historical_query(&self.analysis.original_query) {
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
    fn context_leaves_scoping_to_the_caller() {
        let prepared = PreparedQuery::new("anything");
        assert!(prepared.temporal_context().allowed_doc_ids.is_none());
    }
}
