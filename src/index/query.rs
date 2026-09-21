use crate::query_expansion::normalize_for_index;
use crate::query_semantics::QueryRoutingIntent;
use crate::temporal::{
    recency_boost, resolve_temporal_target, DEFAULT_RECENCY_HALF_LIFE_DAYS,
    DEFAULT_RECENCY_MAX_BOOST,
};
use chrono::{DateTime, NaiveDate, Utc};
use roaring::RoaringBitmap;
use std::collections::{HashMap, HashSet};
use std::time::{Instant, SystemTime};
use tantivy::query::Bm25StatisticsProvider;

use super::helpers::*;
use super::model::*;
use super::query_terms::*;

impl MemoryIndex {
    pub fn query(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        self.query_with_temporal_context(query, top_k, TemporalQueryContext::default())
            .0
    }

    pub fn query_with_temporal_context(
        &self,
        query: &str,
        top_k: usize,
        temporal: TemporalQueryContext<'_>,
    ) -> (Vec<SearchResult>, QueryTimings, QueryDiagnostics) {
        self.query_with_temporal_context_at(query, top_k, temporal, None)
    }

    pub fn query_with_temporal_context_and_statistics(
        &self,
        query: &str,
        top_k: usize,
        temporal: TemporalQueryContext<'_>,
        statistics: &GlobalBm25Statistics,
    ) -> (Vec<SearchResult>, QueryTimings, QueryDiagnostics) {
        self.query_with_temporal_context_at_and_statistics(
            query,
            top_k,
            temporal,
            None,
            Some(statistics),
        )
    }

    pub fn query_with_temporal_context_at(
        &self,
        query: &str,
        top_k: usize,
        temporal: TemporalQueryContext<'_>,
        reference_date: Option<&str>,
    ) -> (Vec<SearchResult>, QueryTimings, QueryDiagnostics) {
        self.query_with_temporal_context_at_and_statistics(
            query,
            top_k,
            temporal,
            reference_date,
            None,
        )
    }

    pub fn query_with_temporal_context_at_and_statistics(
        &self,
        query: &str,
        top_k: usize,
        temporal: TemporalQueryContext<'_>,
        reference_date: Option<&str>,
        statistics: Option<&GlobalBm25Statistics>,
    ) -> (Vec<SearchResult>, QueryTimings, QueryDiagnostics) {
        self.query_with_temporal_context_at_and_statistics_with_local_terms(
            query,
            top_k,
            temporal,
            reference_date,
            statistics,
            &[],
        )
    }

    pub(crate) fn query_with_temporal_context_at_and_statistics_with_local_terms(
        &self,
        query: &str,
        top_k: usize,
        temporal: TemporalQueryContext<'_>,
        reference_date: Option<&str>,
        statistics: Option<&GlobalBm25Statistics>,
        local_terms: &[String],
    ) -> (Vec<SearchResult>, QueryTimings, QueryDiagnostics) {
        let search_k = match temporal.query_routing_intent {
            Some(_) => top_k.saturating_mul(10).max(25),
            None => top_k.saturating_mul(5).max(20),
        };
        let total_start = Instant::now();
        let (temporal_start, temporal_end) =
            normalize_temporal_bounds(temporal.starts_from, temporal.ends_at);
        let relative_anchor = reference_date
            .or(temporal_start.and(temporal.starts_from))
            .or(temporal_end.and(temporal.ends_at))
            .or(temporal.starts_from)
            .or(temporal.ends_at);
        let target = resolve_temporal_target(query, relative_anchor);
        let apply_recency = !temporal.has_explicit_temporal
            && target.is_none()
            && temporal_start.is_none()
            && temporal_end.is_none()
            && temporal.time_hint.is_none();
        let (mut results, mut timings, diagnostics) = self.query_timed_with_context(
            query,
            search_k,
            temporal,
            apply_recency,
            statistics,
            local_terms,
        );
        if target.is_none()
            && temporal_start.is_none()
            && temporal_end.is_none()
            && temporal.time_hint.is_none()
        {
            timings.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            results.truncate(top_k);
            return (results, timings, diagnostics);
        }

        let window_days = temporal.window_days.max(1);
        let hard_filter = temporal.hard_filter;
        let time_hint = temporal.time_hint;
        let now_date = DateTime::<Utc>::from(SystemTime::now()).date_naive();
        let mut rescored = Vec::with_capacity(results.len());
        for mut result in results.drain(..) {
            let Some(doc) = self.docs.get(&result.doc_id) else {
                continue;
            };
            let Some(doc_date) = doc_temporal_date(doc) else {
                if hard_filter {
                    continue;
                }
                rescored.push(result);
                continue;
            };
            if let Some(start) = temporal_start {
                if doc_date < start && hard_filter {
                    continue;
                }
            }
            if let Some(end) = temporal_end {
                if doc_date > end && hard_filter {
                    continue;
                }
            }
            let within_explicit_range = temporal_start.is_none_or(|start| doc_date >= start)
                && temporal_end.is_none_or(|end| doc_date <= end);
            if within_explicit_range {
                let explicit_boost = TEMPORAL_RANGE_BOOST;
                result.score += explicit_boost;
                result.score_breakdown.recency_score += explicit_boost;
            }
            if let Some(target) = target {
                let delta_days = (doc_date
                    .signed_duration_since(target.target_date)
                    .num_days())
                .abs();
                if hard_filter && delta_days > window_days {
                    continue;
                }
                if delta_days <= window_days {
                    let proximity = 1.0 - (delta_days as f32 / window_days as f32);
                    let boost = proximity.clamp(0.0, 1.0) * TEMPORAL_PROXIMITY_WEIGHT;
                    result.score += boost;
                    result.score_breakdown.recency_score += boost;
                }
            }
            if temporal.has_explicit_temporal
                && target.is_none()
                && temporal_start.is_none()
                && temporal_end.is_none()
            {
                if let Some(hint) = time_hint {
                    let delta_days = (doc_date.signed_duration_since(now_date).num_days()).abs();
                    let boost = match hint {
                        TemporalQueryHint::Past => {
                            if doc_date <= now_date {
                                0.03
                            } else {
                                0.0
                            }
                        }
                        TemporalQueryHint::Present => {
                            if delta_days <= 30 {
                                let proximity = 1.0 - (delta_days as f32 / 30.0);
                                proximity.clamp(0.0, 1.0) * 0.25
                            } else {
                                0.0
                            }
                        }
                        TemporalQueryHint::Ongoing => {
                            if delta_days <= 14 {
                                let proximity = 1.0 - (delta_days as f32 / 14.0);
                                proximity.clamp(0.0, 1.0) * 0.3
                            } else {
                                0.0
                            }
                        }
                        TemporalQueryHint::Mixed => {
                            if delta_days <= 30 {
                                let proximity = 1.0 - (delta_days as f32 / 30.0);
                                proximity.clamp(0.0, 1.0) * 0.15
                            } else {
                                0.0
                            }
                        }
                    };
                    if boost > 0.0 {
                        result.score += boost;
                        result.score_breakdown.recency_score += boost;
                    }
                }
            }
            rescored.push(result);
        }
        rescored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                // Deterministic tie-break on doc id; equal scores must not
                // fall back to HashMap iteration order.
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        rescored.truncate(top_k);
        timings.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        (rescored, timings, diagnostics)
    }

    #[deprecated(
        since = "0.1.9",
        note = "use `query_with_temporal_context`, which also carries query intent; \
                retained for 0.2 compatibility; scheduled for a later breaking release"
    )]
    pub fn query_timed(
        &self,
        query: &str,
        top_k: usize,
    ) -> (Vec<SearchResult>, QueryTimings, QueryDiagnostics) {
        self.query_timed_with_context(
            query,
            top_k,
            TemporalQueryContext::default(),
            true,
            None,
            &[],
        )
    }

    fn query_timed_with_context(
        &self,
        query: &str,
        top_k: usize,
        temporal: TemporalQueryContext<'_>,
        apply_recency: bool,
        statistics: Option<&GlobalBm25Statistics>,
        local_terms: &[String],
    ) -> (Vec<SearchResult>, QueryTimings, QueryDiagnostics) {
        let total_start = Instant::now();
        let lexical_start = Instant::now();
        let lexical_hits = match self.lexical_bm25(
            query,
            top_k.saturating_mul(5).max(20),
            statistics.map(|value| value as &dyn Bm25StatisticsProvider),
        ) {
            Ok(hits) => Some(hits),
            Err(err) => {
                eprintln!("warning: lexical BM25 query component failed: {}", err);
                None
            }
        };
        let lexical_bm25_ms = lexical_start.elapsed().as_secs_f64() * 1000.0;
        let snapshot_start = Instant::now();
        let (results, mut timings, diagnostics) = self.query_with_lexical_hits_timed(
            query,
            top_k,
            lexical_hits.as_ref(),
            temporal.query_routing_intent,
            temporal.has_explicit_temporal,
            temporal.allowed_doc_ids,
            temporal.allowed_doc_bitmap,
            apply_recency,
            local_terms,
        );
        let snapshot_query_ms = snapshot_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        timings.snapshot_query_ms = snapshot_query_ms;
        timings.lexical_bm25_ms = lexical_bm25_ms;
        timings.total_ms = total_ms;
        (results, timings, diagnostics)
    }

    #[deprecated(
        since = "0.1.9",
        note = "use `query_with_temporal_context`; externally supplied lexical hits are \
                no longer needed now that the index scores its own. Retained for 0.2 compatibility"
    )]
    pub fn query_with_lexical_hits(
        &self,
        query: &str,
        top_k: usize,
        lexical_hits: Option<&HashMap<String, f32>>,
    ) -> Vec<SearchResult> {
        self.query_with_lexical_hits_timed(
            query,
            top_k,
            lexical_hits,
            None,
            false,
            None,
            None,
            true,
            &[],
        )
        .0
    }

    /// Query with exact-match field filtering. All supplied filter key-value pairs must match
    /// a document's `filters` map for it to be included in results.
    #[deprecated(
        since = "0.1.9",
        note = "use `query_with_temporal_context` with `TemporalQueryContext::allowed_doc_ids` \
                built from `doc_ids_matching_filters`; retained for 0.2 compatibility"
    )]
    #[allow(deprecated)] // delegates to a compatibility helper
    pub fn query_with_filters(
        &self,
        query: &str,
        top_k: usize,
        filters: &std::collections::BTreeMap<String, String>,
    ) -> Vec<SearchResult> {
        self.query_with_filters_and_lexical(query, top_k, filters, None)
    }

    /// Doc ids whose `filters` map matches every supplied key-value pair.
    ///
    /// Returns `None` when `filters` is empty, meaning "no scoping" rather
    /// than "nothing matched". Callers pass this straight to
    /// `TemporalQueryContext::allowed_doc_ids`.
    pub fn doc_ids_matching_filters(
        &self,
        filters: &std::collections::BTreeMap<String, String>,
    ) -> Option<HashSet<String>> {
        if filters.is_empty() {
            return None;
        }
        let mut matches = filters.iter().filter_map(|(key, value)| {
            self.filter_postings
                .get(key)
                .and_then(|values| values.get(value))
        });
        let Some(first) = matches.next() else {
            return Some(HashSet::new());
        };
        let mut bitmap = first.clone();
        for posting in matches {
            bitmap &= posting;
        }
        Some(
            bitmap
                .iter()
                .filter_map(|doc_u32| self.doc_u32_to_id.get(doc_u32 as usize).cloned())
                .collect(),
        )
    }

    pub fn doc_bitmap_matching_filters(
        &self,
        filters: &std::collections::BTreeMap<String, String>,
    ) -> Option<RoaringBitmap> {
        if filters.is_empty() {
            return None;
        }
        let mut matches = filters
            .iter()
            .filter_map(|(key, value)| self.filter_postings.get(key)?.get(value));
        let Some(first) = matches.next() else {
            return Some(RoaringBitmap::new());
        };
        let mut bitmap = first.clone();
        for posting in matches {
            bitmap &= posting;
        }
        Some(bitmap)
    }

    pub fn doc_bitmap_for_ids(&self, ids: &HashSet<String>) -> RoaringBitmap {
        ids.iter()
            .filter_map(|id| self.doc_id_to_u32.get(id).copied())
            .collect()
    }

    #[deprecated(
        since = "0.1.9",
        note = "use `query_with_temporal_context` with `TemporalQueryContext::allowed_doc_ids` \
                built from `doc_ids_matching_filters`; retained for 0.2 compatibility"
    )]
    #[allow(deprecated)] // delegates to a compatibility helper
    pub fn query_with_filters_and_lexical(
        &self,
        query: &str,
        top_k: usize,
        filters: &std::collections::BTreeMap<String, String>,
        lexical_hits: Option<&HashMap<String, f32>>,
    ) -> Vec<SearchResult> {
        let Some(allowed) = self.doc_ids_matching_filters(filters) else {
            return self.query_with_lexical_hits(query, top_k, lexical_hits);
        };
        self.query_with_lexical_hits_timed(
            query,
            top_k,
            lexical_hits,
            None,
            false,
            Some(&allowed),
            None,
            true,
            &[],
        )
        .0
    }

    /// Multi-term variant: builds the allowed-doc set once from `filters`, then scores each
    /// query against it. At scale (thousands of docs) this avoids rescanning docs N times.
    /// Returns one `Vec<SearchResult>` per input query, in the same order.
    #[deprecated(
        since = "0.1.9",
        note = "use `query_with_temporal_context` with `TemporalQueryContext::allowed_doc_ids` \
                built once from `doc_ids_matching_filters` and reused across queries; \
                retained for 0.2 compatibility; scheduled for a later breaking release"
    )]
    pub fn query_with_filters_multi(
        &self,
        queries: &[&str],
        top_k: usize,
        filters: &std::collections::BTreeMap<String, String>,
        lexical_hits_per_query: &[HashMap<String, f32>],
    ) -> Vec<Vec<SearchResult>> {
        let allowed_opt: Option<HashSet<String>> = if filters.is_empty() {
            None
        } else {
            Some(
                self.docs
                    .iter()
                    .filter(|(_, doc)| {
                        filters
                            .iter()
                            .all(|(k, v)| doc.filters.get(k).map(|dv| dv == v).unwrap_or(false))
                    })
                    .map(|(doc_id, _)| doc_id.clone())
                    .collect(),
            )
        };
        queries
            .iter()
            .zip(lexical_hits_per_query.iter())
            .map(|(q, hits)| {
                self.query_with_lexical_hits_timed(
                    q,
                    top_k,
                    Some(hits),
                    None,
                    false,
                    allowed_opt.as_ref(),
                    None,
                    true,
                    &[],
                )
                .0
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn query_with_lexical_hits_timed(
        &self,
        query: &str,
        top_k: usize,
        lexical_hits: Option<&HashMap<String, f32>>,
        query_routing_intent: Option<QueryRoutingIntent>,
        has_explicit_temporal: bool,
        allowed_doc_ids: Option<&HashSet<String>>,
        allowed_doc_bitmap: Option<&RoaringBitmap>,
        apply_recency: bool,
        local_terms: &[String],
    ) -> (Vec<SearchResult>, QueryTimings, QueryDiagnostics) {
        let rerank_start = Instant::now();
        let parse_start = Instant::now();
        let Some(prepared) = prepare_query_terms(query) else {
            return (
                Vec::new(),
                QueryTimings::default(),
                QueryDiagnostics::default(),
            );
        };
        let q = prepared.normalized;
        let q_terms = prepared.terms;
        let mut expanded_terms = prepared.expanded_terms;
        for term in local_terms {
            if !q_terms.contains(term) && !expanded_terms.contains(term) {
                expanded_terms.push(term.clone());
            }
        }
        let mut query_entities: HashSet<String> = HashSet::new();
        if self.entity_trie.get(&q).is_some() {
            query_entities.insert(q.clone());
        }
        for term in &q_terms {
            if self.entity_trie.get(term).is_some() {
                query_entities.insert(term.clone());
            }
        }
        if self.doc_u32_to_id.is_empty() {
            return (
                Vec::new(),
                QueryTimings::default(),
                QueryDiagnostics::default(),
            );
        }
        let parse_ms = parse_start.elapsed().as_secs_f64() * 1000.0;
        let sparse_start = Instant::now();
        let mut candidates: HashMap<usize, CandidateState> = HashMap::new();
        let query_set: HashSet<String> = q_terms.iter().cloned().collect();
        enum AllowedDocs<'a> {
            Set(&'a HashSet<usize>),
            Bitmap(&'a RoaringBitmap),
        }
        impl AllowedDocs<'_> {
            fn contains(&self, doc_u32: usize) -> bool {
                match self {
                    Self::Set(ids) => ids.contains(&doc_u32),
                    Self::Bitmap(bitmap) => bitmap.contains(doc_u32 as u32),
                }
            }
        }
        let allowed_doc_ids_u32 = allowed_doc_bitmap
            .is_none()
            .then(|| {
                allowed_doc_ids.map(|allowed| {
                    allowed
                        .iter()
                        .filter_map(|id| self.doc_id_to_u32.get(id).copied())
                        .map(|id| id as usize)
                        .collect::<HashSet<_>>()
                })
            })
            .flatten();
        let allowed_doc_u32s = allowed_doc_bitmap
            .map(AllowedDocs::Bitmap)
            .or_else(|| allowed_doc_ids_u32.as_ref().map(AllowedDocs::Set));

        fn score_doc<F>(
            candidates: &mut HashMap<usize, CandidateState>,
            doc_u32: usize,
            delta: f32,
            allowed_doc_u32s: Option<&AllowedDocs<'_>>,
            apply: F,
        ) where
            F: FnOnce(&mut CandidateState),
        {
            if let Some(allowed_doc_u32s) = allowed_doc_u32s {
                if !allowed_doc_u32s.contains(doc_u32) {
                    return;
                }
            }
            let entry = candidates.entry(doc_u32).or_default();
            entry.score += delta;
            apply(entry);
        }

        let lexical_merge_start = Instant::now();
        if let Some(lexical_hits) = lexical_hits {
            for (doc_id, bm25_score) in lexical_hits {
                if let Some(&doc_u32) = self.doc_id_to_u32.get(doc_id) {
                    score_doc(
                        &mut candidates,
                        doc_u32 as usize,
                        *bm25_score,
                        allowed_doc_u32s.as_ref(),
                        |entry| {
                            entry.breakdown.lexical_score += *bm25_score;
                        },
                    );
                }
            }
        }
        let lexical_merge_ms = lexical_merge_start.elapsed().as_secs_f64() * 1000.0;

        let posting_scoring_start = Instant::now();

        // Full-query entity hit.
        if let Some(entity_u32) = self.entity_trie.get(&q) {
            if let Some(postings) = self.entity_postings_doc.get(entity_u32 as usize) {
                for &(doc_u32, post_score) in postings {
                    let mut delta = FULL_QUERY_ENTITY_WEIGHT * post_score;
                    if !query_entities.is_empty() {
                        delta *= 1.25;
                    }
                    score_doc(
                        &mut candidates,
                        doc_u32 as usize,
                        delta,
                        allowed_doc_u32s.as_ref(),
                        |entry| {
                            entry.breakdown.entity_score += delta;
                            entry.matched_entities.push(q.clone());
                        },
                    );
                }
            }
        }

        for term in &q_terms {
            let mut entity_ids = Vec::new();
            if let Some(entity_u32) = self.entity_trie.get(term) {
                entity_ids.push((entity_u32, 1.0f32));
            } else if term.len() >= 4 {
                for entity_u32 in self.entity_trie.prefix_ids(term, 1) {
                    entity_ids.push((entity_u32, ENTITY_PREFIX_MULTIPLIER));
                }
            }
            for (entity_u32, mult) in entity_ids {
                if let Some(postings) = self.entity_postings_doc.get(entity_u32 as usize) {
                    for &(doc_u32, post_score) in postings {
                        let exact_entity_match = query_entities.contains(term);
                        let mut delta = ENTITY_TERM_WEIGHT * post_score * mult;
                        if exact_entity_match {
                            delta *= 1.4;
                        }
                        score_doc(
                            &mut candidates,
                            doc_u32 as usize,
                            delta,
                            allowed_doc_u32s.as_ref(),
                            |entry| {
                                entry.breakdown.entity_score += delta;
                                entry.matched_entities.push(term.clone());
                            },
                        );
                    }
                }
            }
            let mut term_ids = Vec::new();
            if let Some(term_u32) = self.term_trie.get(term) {
                term_ids.push((term_u32, 1.0f32));
            } else if term.len() >= 4 {
                for term_u32 in self.term_trie.prefix_ids(term, 1) {
                    term_ids.push((term_u32, IMPORTANT_TERM_PREFIX_MULTIPLIER));
                }
            }
            for (term_u32, mult) in term_ids {
                if let Some(postings) = self.term_postings_doc.get(term_u32 as usize) {
                    for &(doc_u32, post_score) in postings {
                        let delta = IMPORTANT_TERM_WEIGHT * post_score * mult;
                        score_doc(
                            &mut candidates,
                            doc_u32 as usize,
                            delta,
                            allowed_doc_u32s.as_ref(),
                            |entry| {
                                entry.breakdown.term_score += delta;
                                entry.matched_terms.push(term.clone());
                            },
                        );
                    }
                }
            }
            if self.claim_scoring {
                if let Some(postings) = self.claim_to_docs.get(term) {
                    for posting in postings {
                        if let Some(&doc_u32) = self.doc_id_to_u32.get(&posting.doc_id) {
                            let delta = 0.7 * posting.score;
                            score_doc(
                                &mut candidates,
                                doc_u32 as usize,
                                delta,
                                allowed_doc_u32s.as_ref(),
                                |entry| {
                                    entry.breakdown.claim_score += delta;
                                    entry.matched_terms.push(term.clone());
                                },
                            );
                        }
                    }
                }
            }
        }

        // Expanded semantic terms are lower-weight than original terms.
        for term in &expanded_terms {
            if let Some(entity_u32) = self.entity_trie.get(term) {
                if let Some(postings) = self.entity_postings_doc.get(entity_u32 as usize) {
                    for &(doc_u32, post_score) in postings {
                        let mut delta = EXPANDED_ENTITY_WEIGHT * post_score;
                        if query_entities.contains(term) {
                            delta *= 1.2;
                        }
                        score_doc(
                            &mut candidates,
                            doc_u32 as usize,
                            delta,
                            allowed_doc_u32s.as_ref(),
                            |entry| {
                                entry.breakdown.entity_score += delta;
                            },
                        );
                    }
                }
            }
            if let Some(term_u32) = self.term_trie.get(term) {
                if let Some(postings) = self.term_postings_doc.get(term_u32 as usize) {
                    for &(doc_u32, post_score) in postings {
                        let delta = EXPANDED_TERM_WEIGHT * post_score;
                        score_doc(
                            &mut candidates,
                            doc_u32 as usize,
                            delta,
                            allowed_doc_u32s.as_ref(),
                            |entry| {
                                entry.breakdown.term_score += delta;
                            },
                        );
                    }
                }
            }
            if self.claim_scoring {
                if let Some(postings) = self.claim_to_docs.get(term) {
                    for posting in postings {
                        if let Some(&doc_u32) = self.doc_id_to_u32.get(&posting.doc_id) {
                            let delta = 0.7 * 0.6 * posting.score;
                            score_doc(
                                &mut candidates,
                                doc_u32 as usize,
                                delta,
                                allowed_doc_u32s.as_ref(),
                                |entry| {
                                    entry.breakdown.claim_score += delta;
                                },
                            );
                        }
                    }
                }
            }
        }
        let posting_scoring_ms = posting_scoring_start.elapsed().as_secs_f64() * 1000.0;

        let routing_seed_start = Instant::now();
        if let Some(intent) = query_routing_intent {
            let candidate_doc_ids = candidates.keys().copied().collect::<Vec<_>>();
            seed_routing_candidates(
                self,
                &mut candidates,
                &candidate_doc_ids,
                &q_terms,
                &query_entities,
                intent,
            );
        }
        let routing_seed_ms = routing_seed_start.elapsed().as_secs_f64() * 1000.0;
        let sparse_scoring_ms = sparse_start.elapsed().as_secs_f64() * 1000.0;

        let candidate_accumulation_start = Instant::now();
        let metadata_start = Instant::now();
        let candidate_doc_ids = candidates.keys().copied().collect::<Vec<_>>();
        for doc_u32 in candidate_doc_ids {
            let Some(doc_id) = self.doc_u32_to_id.get(doc_u32) else {
                continue;
            };
            let Some(doc) = self.docs.get(doc_id) else {
                continue;
            };
            let entry = candidates.entry(doc_u32).or_default();
            if let Some(topic) = doc.probable_topic.as_ref() {
                let topic_tokens = tokenize_query_terms(topic);
                let overlap = topic_tokens
                    .iter()
                    .filter(|t| query_set.contains(*t))
                    .count();
                if overlap > 0 {
                    let delta = TOPIC_OVERLAP_WEIGHT * overlap as f32;
                    entry.score += delta;
                    entry.breakdown.topic_score += delta;
                }
            }
            if let Some(dt) = doc.doc_type_guess.as_ref() {
                let dt_tokens = tokenize_query_terms(dt);
                let overlap = dt_tokens.iter().filter(|t| query_set.contains(*t)).count();
                if overlap > 0 {
                    let delta = DOC_TYPE_OVERLAP_WEIGHT * overlap as f32;
                    entry.score += delta;
                    entry.breakdown.doc_type_score += delta;
                }
            }
            if self.claim_scoring && !doc.top_claims.is_empty() {
                let mut best_claim_delta = 0.0f32;
                for claim in &doc.top_claims {
                    let claim_terms = claim_tokens(claim);
                    if claim_terms.is_empty() {
                        continue;
                    }
                    let matched = claim_terms
                        .iter()
                        .filter(|token| query_set.contains(*token))
                        .count();
                    if matched == 0 {
                        continue;
                    }
                    let coverage = matched as f32 / claim_terms.len().max(1) as f32;
                    let mut delta = 0.45 * coverage * claim.confidence.max(0.1);
                    if matched >= 2 {
                        delta += 0.15 * claim.confidence.max(0.1);
                    }
                    if delta > best_claim_delta {
                        best_claim_delta = delta;
                    }
                }
                if best_claim_delta > 0.0 {
                    entry.score += best_claim_delta;
                    entry.breakdown.claim_score += best_claim_delta;
                }
            }
            if apply_recency {
                let delta = recency_boost(
                    doc.section_chunks
                        .iter()
                        .find_map(|chunk| chunk.timestamp.as_deref())
                        .or(doc.timestamp.as_deref()),
                    DateTime::<Utc>::from(SystemTime::now()),
                    DEFAULT_RECENCY_HALF_LIFE_DAYS,
                    DEFAULT_RECENCY_MAX_BOOST,
                )
                .unwrap_or(0.0);
                if delta > 0.0 {
                    entry.score += delta;
                    entry.breakdown.recency_score += delta;
                }
            }
        }
        let candidate_accumulation_ms =
            candidate_accumulation_start.elapsed().as_secs_f64() * 1000.0;
        let metadata_ms = metadata_start.elapsed().as_secs_f64() * 1000.0;

        let candidate_rank_start = Instant::now();
        let mut ranked_docs: Vec<(usize, f32)> = candidates
            .iter()
            .map(|(doc_u32, state)| (*doc_u32, state.score))
            .collect();
        ranked_docs.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                // Deterministic tie-break: `candidates` is a HashMap, so equal
                // scores must not fall back to input (iteration) order, which
                // differs between index builds.
                .then_with(|| {
                    self.doc_u32_to_id
                        .get(a.0)
                        .cmp(&self.doc_u32_to_id.get(b.0))
                })
        });
        let candidate_rank_ms = candidate_rank_start.elapsed().as_secs_f64() * 1000.0;

        // Graph-aware rerank features with bounded contribution.
        // Use current hybrid score as base so graph signals cannot dominate.
        let disable_followup_boosts = matches!(
            query_routing_intent,
            Some(QueryRoutingIntent::Count) | Some(QueryRoutingIntent::Sum)
        );

        let graph_ms;
        let entity_graph_ms;
        if disable_followup_boosts {
            graph_ms = 0.0;
            entity_graph_ms = 0.0;
        } else {
            let graph_start = Instant::now();
            let anchors = ranked_docs.iter().take(5).cloned().collect::<Vec<_>>();

            // 1-hop doc-link proximity boost.
            for (anchor_idx, anchor_score) in &anchors {
                let anchor_doc_id = &self.doc_u32_to_id[*anchor_idx];
                let Some(anchor_doc) = self.docs.get(anchor_doc_id) else {
                    continue;
                };
                let anchor_weight = (*anchor_score / (1.0 + *anchor_score)).clamp(0.0, 1.0);
                for linked in &anchor_doc.doc_links {
                    if let Some(&doc_u32) = self.doc_id_to_u32.get(linked) {
                        let Some(entry) = candidates.get_mut(&(doc_u32 as usize)) else {
                            continue;
                        };
                        let delta = (DOC_LINK_GRAPH_WEIGHT * anchor_weight)
                            .min(GRAPH_MAX_BOOST - entry.breakdown.graph_link_score);
                        if delta > 0.0 {
                            entry.score += delta;
                            entry.breakdown.graph_link_score += delta;
                        }
                    }
                }
            }
            graph_ms = graph_start.elapsed().as_secs_f64() * 1000.0;

            let entity_graph_start = Instant::now();
            // Entity-graph proximity boost: overlap against anchor-entity set.
            let mut anchor_entities: HashSet<String> = HashSet::new();
            for (doc_u32, _) in &anchors {
                if let Some(keys) = self.doc_key_entities.get(*doc_u32) {
                    for key in keys {
                        if !key.is_empty() {
                            anchor_entities.insert(key.clone());
                        }
                    }
                }
            }
            if !anchor_entities.is_empty() {
                let candidate_doc_ids = ranked_docs
                    .iter()
                    .take(ENTITY_GRAPH_MAX_CANDIDATES)
                    .map(|(doc_u32, _)| *doc_u32)
                    .collect::<Vec<_>>();
                for doc_u32 in candidate_doc_ids {
                    let Some(entry) = candidates.get_mut(&doc_u32) else {
                        continue;
                    };
                    let Some(keys) = self.doc_key_entities.get(doc_u32) else {
                        continue;
                    };
                    let overlap = keys.iter().filter(|k| anchor_entities.contains(*k)).count();
                    if overlap > 0 {
                        let delta = (ENTITY_GRAPH_WEIGHT * overlap as f32)
                            .min(GRAPH_MAX_BOOST - entry.breakdown.entity_graph_score);
                        if delta > 0.0 {
                            entry.score += delta;
                            entry.breakdown.entity_graph_score += delta;
                        }
                    }
                }
            }
            entity_graph_ms = entity_graph_start.elapsed().as_secs_f64() * 1000.0;
        }

        let sequence_rerank_start = Instant::now();
        let rerank_window = ranked_docs
            .iter()
            .take(TEXT_RERANK_WINDOW.min(FINAL_RERANK_WINDOW))
            .map(|(doc_u32, _)| *doc_u32)
            .collect::<Vec<_>>();
        let skip_sequence_rerank = query_routing_intent.is_some() && !has_explicit_temporal;
        if !skip_sequence_rerank && !q_terms.is_empty() {
            for doc_u32 in rerank_window {
                let Some(candidate_tokens) = cached_doc_rerank_tokens(self, doc_u32) else {
                    let Some(doc_id) = self.doc_u32_to_id.get(doc_u32) else {
                        continue;
                    };
                    let Some(doc) = self.docs.get(doc_id) else {
                        continue;
                    };
                    let fallback_text = normalize_for_index(&candidate_rerank_text(doc));
                    let owned_tokens = tokenize_query_terms(&fallback_text);
                    if owned_tokens.is_empty() {
                        continue;
                    }
                    let token_overlap = token_overlap_ratio(&q_terms, &owned_tokens);
                    let mut delta = TEXT_RERANK_WEIGHT * token_overlap;
                    if self.text_rerank_ngram {
                        let ngram_overlap =
                            ngram_overlap_ratio(&q_terms, &owned_tokens, TEXT_RERANK_NGRAM_SIZE);
                        delta += TEXT_RERANK_NGRAM_WEIGHT * ngram_overlap;
                    }
                    if self.text_rerank_lcs {
                        let lcs_overlap = lcs_ratio(&q_terms, &owned_tokens);
                        delta += TEXT_RERANK_LCS_WEIGHT * lcs_overlap;
                    }
                    if delta <= 0.0 {
                        continue;
                    }
                    if let Some(entry) = candidates.get_mut(&doc_u32) {
                        entry.score += delta;
                        entry.breakdown.sequence_rerank_score += delta;
                    }
                    continue;
                };
                if candidate_tokens.is_empty() {
                    continue;
                }
                let token_overlap = token_overlap_ratio(&q_terms, candidate_tokens);
                let mut delta = TEXT_RERANK_WEIGHT * token_overlap;
                if self.text_rerank_ngram {
                    let ngram_overlap =
                        ngram_overlap_ratio(&q_terms, candidate_tokens, TEXT_RERANK_NGRAM_SIZE);
                    delta += TEXT_RERANK_NGRAM_WEIGHT * ngram_overlap;
                }
                if self.text_rerank_lcs {
                    let lcs_overlap = lcs_ratio(&q_terms, candidate_tokens);
                    delta += TEXT_RERANK_LCS_WEIGHT * lcs_overlap;
                }
                if delta <= 0.0 {
                    continue;
                }
                if let Some(entry) = candidates.get_mut(&doc_u32) {
                    entry.score += delta;
                    entry.breakdown.sequence_rerank_score += delta;
                }
            }
        }
        let sequence_rerank_ms = sequence_rerank_start.elapsed().as_secs_f64() * 1000.0;

        let count_query = {
            let lower = query.to_lowercase();
            lower.starts_with("how many")
                || lower.starts_with("count ")
                || lower.contains(" number of ")
                || lower.contains(" number of")
                || lower.contains("count the ")
                || lower.contains("times did i")
        };
        let group_build_start = Instant::now();
        let mut ranked_docs: Vec<(usize, f32)> = candidates
            .iter()
            .filter_map(|(doc_u32, state)| {
                if state.score > 0.0 {
                    Some((*doc_u32, state.score))
                } else {
                    None
                }
            })
            .collect();
        ranked_docs.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                // Deterministic tie-break: `candidates` is a HashMap, so equal
                // scores must not fall back to input (iteration) order, which
                // differs between index builds.
                .then_with(|| {
                    self.doc_u32_to_id
                        .get(a.0)
                        .cmp(&self.doc_u32_to_id.get(b.0))
                })
        });
        let ranking_start = Instant::now();
        let ranked_docs = ranked_docs
            .into_iter()
            .take(FINAL_RERANK_WINDOW.min(self.doc_u32_to_id.len()))
            .collect::<Vec<_>>();
        let mut grouped_ranked_docs: HashMap<String, Vec<(usize, f32)>> = HashMap::new();
        let mut group_metadata: HashMap<String, (Option<String>, String)> = HashMap::new();
        for (doc_u32, score) in ranked_docs {
            let Some(doc_id) = self.doc_u32_to_id.get(doc_u32) else {
                continue;
            };
            let Some(doc) = self.docs.get(doc_id) else {
                continue;
            };
            let group_key = doc
                .group_id
                .clone()
                .unwrap_or_else(|| format!("doc:{}", doc.doc_id));
            grouped_ranked_docs
                .entry(group_key.clone())
                .or_default()
                .push((doc_u32, score));
            group_metadata
                .entry(group_key)
                .or_insert_with(|| (doc.group_id.clone(), doc.source.clone()));
        }
        let group_build_ms = group_build_start.elapsed().as_secs_f64() * 1000.0;

        type RankedGroup = (String, f32, Vec<(usize, f32)>);
        let mut ranked_groups: Vec<RankedGroup> = grouped_ranked_docs
            .into_iter()
            .map(|(group_key, mut items)| {
                items.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        // Deterministic tie-break on doc id; see ranked_docs
                        // above.
                        .then_with(|| {
                            self.doc_u32_to_id
                                .get(a.0)
                                .cmp(&self.doc_u32_to_id.get(b.0))
                        })
                });
                let mut unique_entities: HashSet<String> = HashSet::new();
                let mut unique_terms: HashSet<String> = HashSet::new();
                let mut dates: Vec<NaiveDate> = Vec::new();
                let support_threshold = (items[0].1 * 0.35).max(0.15);
                let mut supporting_docs = 0usize;

                for (doc_u32, score) in &items {
                    if let Some(state) = candidates.get(doc_u32) {
                        if *score >= support_threshold {
                            supporting_docs += 1;
                        }
                        for entity in &state.matched_entities {
                            if !entity.is_empty() {
                                unique_entities.insert(entity.clone());
                            }
                        }
                        for term in &state.matched_terms {
                            if !term.is_empty() {
                                unique_terms.insert(term.clone());
                            }
                        }
                    }
                    if let Some(doc_id) = self.doc_u32_to_id.get(*doc_u32) {
                        if let Some(doc) = self.docs.get(doc_id) {
                            if let Some(date) = doc_temporal_date(doc) {
                                dates.push(date);
                            }
                        }
                    }
                }

                let mut score = aggregate_group_score(&items, count_query);
                if !query_entities.is_empty() {
                    let exact_entity_matches = items
                        .iter()
                        .filter(|(doc_u32, _)| {
                            candidates
                                .get(doc_u32)
                                .map(|state| {
                                    state
                                        .matched_entities
                                        .iter()
                                        .any(|entity| query_entities.contains(entity.as_str()))
                                })
                                .unwrap_or(false)
                        })
                        .count();
                    score += (exact_entity_matches.min(3) as f32) * 0.10;
                    score += (unique_entities
                        .iter()
                        .filter(|entity| query_entities.contains(*entity))
                        .count()
                        .min(3) as f32)
                        * 0.08;
                }
                score += (unique_entities.len().min(4) as f32) * 0.04;
                score += (unique_terms.len().min(6) as f32) * 0.015;
                score += (supporting_docs.min(4) as f32) * if count_query { 0.06 } else { 0.03 };
                if dates.len() >= 2 {
                    dates.sort();
                    let span_days = dates
                        .last()
                        .zip(dates.first())
                        .map(|(end, start)| end.signed_duration_since(*start).num_days().abs())
                        .unwrap_or(0);
                    if span_days <= 7 {
                        score += if count_query { 0.12 } else { 0.08 };
                    } else if span_days <= 30 {
                        score += if count_query { 0.06 } else { 0.04 };
                    }
                }
                (group_key, score, items)
            })
            .collect();
        let mut group_sort_ms = 0.0;
        let group_sort_start = Instant::now();
        ranked_groups.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                // Deterministic tie-break on the session key; group order
                // feeds result assembly and truncation.
                .then_with(|| a.0.cmp(&b.0))
        });
        group_sort_ms += group_sort_start.elapsed().as_secs_f64() * 1000.0;
        let evidence_start = Instant::now();
        if let Some(intent) = query_routing_intent {
            let evidence_terms = routing_content_terms(&q_terms);
            let unit_terms = routing_unit_terms(&q_terms);
            for (_group_key, score, items) in ranked_groups.iter_mut() {
                let mut evidence_supporting_docs = 0usize;
                let mut evidence_score = 0.0f32;
                let mut evidence_term_matches = 0usize;
                let mut evidence_numbers = 0usize;
                let mut evidence_units = 0usize;
                let mut evidence_dates = 0usize;
                let mut evidence_predicates = 0usize;
                let mut evidence_penalty = 0.0f32;
                for (doc_u32, _) in items.iter() {
                    if let Some(features) = self.evidence_features_for_doc(
                        *doc_u32,
                        &evidence_terms,
                        &unit_terms,
                        intent,
                    ) {
                        if features.score > 0.0 {
                            evidence_supporting_docs += 1;
                            evidence_score += features.score;
                            evidence_term_matches += features.matched_terms;
                            evidence_numbers += usize::from(features.has_number);
                            evidence_units += usize::from(features.has_unit);
                            evidence_dates +=
                                usize::from(features.has_date || features.has_temporal_terms);
                            evidence_predicates += usize::from(features.predicate_signal);
                            evidence_penalty += features.distractor_penalty;
                        }
                    }
                }
                let evidence_boost = group_evidence_boost(
                    intent,
                    evidence_supporting_docs,
                    evidence_score,
                    evidence_term_matches,
                    evidence_numbers,
                    evidence_units,
                    evidence_dates,
                    evidence_predicates,
                    evidence_penalty,
                );
                *score = *score * 0.80 + evidence_boost;
            }
            let group_sort_start = Instant::now();
            ranked_groups.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    // Deterministic tie-break on the session key; see above.
                    .then_with(|| a.0.cmp(&b.0))
            });
            group_sort_ms += group_sort_start.elapsed().as_secs_f64() * 1000.0;
        }
        let evidence_ms = evidence_start.elapsed().as_secs_f64() * 1000.0;

        let mut per_group_counts: HashMap<String, usize> = HashMap::new();
        let mut results = Vec::new();
        for (group_key, _, items) in ranked_groups {
            let (group_id, _) = group_metadata
                .get(&group_key)
                .cloned()
                .unwrap_or((None, String::new()));
            for (doc_u32, score) in items {
                if results.len() >= top_k.min(self.doc_u32_to_id.len()) {
                    break;
                }
                let Some(doc_id) = self.doc_u32_to_id.get(doc_u32).cloned() else {
                    continue;
                };
                let Some(doc) = self.docs.get(&doc_id) else {
                    continue;
                };
                if let Some(group) = group_id.as_ref() {
                    let count = per_group_counts.entry(group.clone()).or_default();
                    if *count >= MAX_RESULTS_PER_GROUP {
                        continue;
                    }
                    *count += 1;
                }
                let state = candidates.get(&doc_u32);
                results.push(SearchResult {
                    doc_id,
                    source: doc.source.clone(),
                    group_id: group_id.clone(),
                    score,
                    score_breakdown: state.map(|s| s.breakdown.clone()).unwrap_or_default(),
                    matched_entities: state
                        .map(|s| {
                            let mut v = s.matched_entities.clone();
                            v.sort();
                            v.dedup();
                            v
                        })
                        .unwrap_or_default(),
                    matched_terms: state
                        .map(|s| {
                            let mut v = s.matched_terms.clone();
                            v.sort();
                            v.dedup();
                            v
                        })
                        .unwrap_or_default(),
                    probable_topic: doc.probable_topic.clone(),
                    doc_type_guess: doc.doc_type_guess.clone(),
                    semantic_status: None,
                    superseded_by: None,
                    relation_confidence: None,
                    relation_evidence: Vec::new(),
                });
            }
            if results.len() >= top_k.min(self.doc_u32_to_id.len()) {
                break;
            }
        }
        let ranking_ms = ranking_start.elapsed().as_secs_f64() * 1000.0;
        let rerank_ms = rerank_start.elapsed().as_secs_f64() * 1000.0;
        if std::env::var_os("LINT_AI_QUERY_TIMINGS").is_some() {
            eprintln!(
                "index_timing total={:.3} parse={:.3} lexical={:.3} posting={:.3} accumulate={:.3} rank={:.3} sequence={:.3} candidates={}",
                rerank_ms, parse_ms, lexical_merge_ms, posting_scoring_ms,
                candidate_accumulation_ms, candidate_rank_ms, sequence_rerank_ms,
                candidates.len()
            );
        }
        (
            results,
            QueryTimings {
                total_ms: rerank_ms,
                refresh_ms: 0.0,
                lexical_bm25_ms: 0.0,
                snapshot_query_ms: rerank_ms,
                rerank_ms,
                parse_ms,
                sparse_scoring_ms,
                lexical_merge_ms,
                posting_scoring_ms,
                routing_seed_ms,
                candidate_accumulation_ms,
                candidate_rank_ms,
                metadata_ms,
                graph_ms,
                entity_graph_ms,
                sequence_rerank_ms,
                evidence_ms,
                group_build_ms,
                group_sort_ms,
                ranking_ms,
            },
            QueryDiagnostics {
                query_terms: q_terms.len(),
                expanded_terms: expanded_terms.len(),
                lexical_hits: lexical_hits.map_or(0, HashMap::len),
                candidates: candidates.len(),
                ..QueryDiagnostics::default()
            },
        )
    }

    pub fn redacted_for_export(&self) -> RedactedMemoryIndex {
        let docs = self
            .docs
            .iter()
            .map(|(id, d)| {
                (
                    id.clone(),
                    RedactedDocRecord {
                        doc_id: d.doc_id.clone(),
                        source: d.source.clone(),
                        timestamp: d.timestamp.clone(),
                        doc_length: d.doc_length,
                        group_id: d.group_id.clone(),
                        probable_topic: d.probable_topic.clone(),
                        doc_type_guess: d.doc_type_guess.clone(),
                        provenance: d.provenance.clone(),
                    },
                )
            })
            .collect();
        RedactedMemoryIndex {
            docs,
            topic_to_docs: self.topic_to_docs.clone(),
            doc_type_to_docs: self.doc_type_to_docs.clone(),
        }
    }

    pub fn impacted_chunks_for_line_range(
        &self,
        doc_id: &str,
        start_line: usize,
        end_line: usize,
    ) -> Vec<String> {
        let Some(&doc_u32) = self.doc_id_to_u32.get(doc_id) else {
            return Vec::new();
        };
        let Some(tree) = self.doc_interval_trees.get(doc_u32 as usize) else {
            return Vec::new();
        };
        let mut chunk_ids = tree
            .query(start_line.max(1), end_line.max(start_line))
            .into_iter()
            .filter_map(|chunk_u32| {
                self.chunks
                    .get(chunk_u32 as usize)
                    .map(|m| m.chunk_id.clone())
            })
            .collect::<Vec<_>>();
        chunk_ids.sort();
        chunk_ids.dedup();
        chunk_ids
    }
}

impl MemoryIndex {
    fn evidence_features_for_doc(
        &self,
        doc_u32: usize,
        content_terms: &[String],
        unit_terms: &[String],
        intent: QueryRoutingIntent,
    ) -> Option<EvidenceFeatures> {
        let doc_id = self.doc_u32_to_id.get(doc_u32)?;
        let doc = self.docs.get(doc_id)?;
        let normalized = cached_doc_rerank_text(self, doc_u32)?;
        if normalized.is_empty() {
            return None;
        }

        let doc_terms: HashSet<String> = cached_doc_rerank_tokens(self, doc_u32)
            .map(|tokens| tokens.iter().cloned().collect())
            .unwrap_or_else(|| tokenize_query_terms(normalized).into_iter().collect());
        let matched_terms = content_terms
            .iter()
            .filter(|term| doc_terms.contains(term.as_str()))
            .count();
        let has_number = self
            .doc_has_number
            .get(doc_u32)
            .copied()
            .unwrap_or_else(|| contains_number_like(normalized));
        let has_unit = !unit_terms.is_empty()
            && unit_terms
                .iter()
                .any(|unit| doc_terms.contains(unit.as_str()));
        if matched_terms == 0 && !has_number && !has_unit {
            return None;
        }
        let has_date = doc_temporal_date(doc).is_some();
        let has_temporal_terms = !doc.temporal_terms.is_empty();
        let predicate_signal = has_predicate_signal(normalized, intent);
        let distractor_penalty = routing_distractor_penalty(normalized, intent);

        let mut score = matched_terms.min(6) as f32 * 0.18;
        if predicate_signal {
            score += 0.55;
        }
        if has_date || has_temporal_terms {
            score += match intent {
                QueryRoutingIntent::Sequence => 0.45,
                _ => 0.12,
            };
        }
        if has_number {
            score += match intent {
                QueryRoutingIntent::Sum => 0.45,
                QueryRoutingIntent::Sequence => 0.12,
                QueryRoutingIntent::Count => 0.08,
            };
        }
        if has_unit {
            score += match intent {
                QueryRoutingIntent::Sum => 0.30,
                QueryRoutingIntent::Sequence => 0.16,
                QueryRoutingIntent::Count => 0.06,
            };
        }
        score -= distractor_penalty;

        if !predicate_signal && matched_terms < 2 {
            score *= 0.25;
        }

        Some(EvidenceFeatures {
            score: score.max(0.0),
            matched_terms,
            has_number,
            has_unit,
            has_date,
            has_temporal_terms,
            predicate_signal,
            distractor_penalty,
        })
    }
}
