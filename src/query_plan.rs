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

use crate::index::{TemporalQueryContext, TemporalQueryHint};
use crate::query_semantics::{analyze_query, QueryAnalysis, QueryTimeHint};

/// A raw query plus the analysis derived from it.
///
/// Holds the analysis so the borrowed [`TemporalQueryContext`] it hands out
/// stays valid; build one per query and keep it alive for the search.
pub struct PreparedQuery {
    analysis: QueryAnalysis,
}

impl PreparedQuery {
    pub fn new(query: &str) -> Self {
        Self {
            analysis: analyze_query(query),
        }
    }

    /// Reuses an analysis the caller already computed, so no query is analyzed
    /// twice on paths that need the analysis for other reasons too.
    pub fn from_analysis(analysis: QueryAnalysis) -> Self {
        Self { analysis }
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
    fn context_leaves_scoping_to_the_caller() {
        let prepared = PreparedQuery::new("anything");
        assert!(prepared.temporal_context().allowed_doc_ids.is_none());
    }
}
