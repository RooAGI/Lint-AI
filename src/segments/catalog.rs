use crate::index::{
    prepare_query_terms, DocRecord, MemoryIndex, TemporalQueryContext, TemporalQueryHint,
};
use crate::query_expansion::normalize_for_index;
use crate::query_semantics::parse_reference_date;
use crate::tier1::BEHOOD_NP_ENTITY_SOURCE;
use crate::tokenizer::{self, TokenizerMode};
use chrono::NaiveDate;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::diagnostics::*;
use super::model::*;
use super::routing::*;

#[derive(Debug, Clone, Default)]
pub(crate) struct SegmentRoutingSummary {
    pub terms: HashMap<String, f32>,
    pub entities: HashMap<String, f32>,
    pub topics: HashMap<String, f32>,
    pub local_memory: HashMap<String, f32>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SegmentCatalog {
    pub(crate) generation: u64,
    segment_count: usize,
    pub(crate) ordered_segment_ids: Vec<String>,
    pub(crate) segment_positions: HashMap<String, usize>,
    pub(crate) summaries: HashMap<String, Arc<SegmentRoutingSummary>>,
    term_segment_counts: HashMap<String, usize>,
    pub(crate) term_to_segments: HashMap<String, Vec<String>>,
    weighted_term_to_segments: HashMap<String, Vec<(String, f32)>>,
    typed_evidence_to_segments: HashMap<String, Vec<String>>,
    pub(crate) connection_profiles: HashMap<String, Arc<SegmentConnectionProfile>>,
}

pub(crate) type SegmentCorpusStats = SegmentCatalog;

#[derive(Debug, Clone, Default)]
pub(crate) struct SegmentConnectionProfile {
    pub(crate) people: HashSet<String>,
    pub(crate) subjects: HashSet<String>,
    pub(crate) times: HashSet<String>,
    pub(crate) actions: HashSet<String>,
    pub(crate) objects: HashSet<String>,
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

pub(crate) fn query_connection_profile(query: &str) -> SegmentConnectionProfile {
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

pub(crate) fn typed_evidence_route_score(
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

pub(crate) fn expand_connected_segments(
    routed_segments: &[SegmentRoute],
    routes: &[SegmentRoute],
    segments: &[MemoryIndexSegment],
    segment_limit: usize,
    temporal: TemporalQueryContext<'_>,
    corpus_stats: &SegmentCorpusStats,
) -> (Vec<SegmentRoute>, Vec<ConnectedSegmentExpansion>) {
    if routed_segments.is_empty() || segment_limit == 0 {
        return (routed_segments.to_vec(), Vec::new());
    }

    let mut selected = routed_segments.to_vec();
    let mut selected_ids = selected
        .iter()
        .map(|route| route.segment_id.clone())
        .collect::<HashSet<_>>();
    let candidate_pool_limit = segment_limit
        .saturating_mul(CONNECTED_EXPANSION_POOL_MULTIPLIER)
        .max(segment_limit + 5)
        .min(CONNECTED_NEIGHBOR_CANDIDATE_LIMIT);
    let routes_by_id = routes
        .iter()
        .map(|route| (route.segment_id.as_str(), route))
        .collect::<HashMap<_, _>>();
    let mut candidate_ids = routes
        .iter()
        .take(candidate_pool_limit)
        .map(|route| route.segment_id.clone())
        .collect::<HashSet<_>>();
    for selected_route in &selected {
        if let Some(profile) = corpus_stats
            .connection_profiles
            .get(&selected_route.segment_id)
        {
            candidate_ids.extend(corpus_stats.typed_candidate_segment_ids(profile));
        }
    }
    candidate_ids.retain(|segment_id| !selected_ids.contains(segment_id));
    let mut candidate_ids = candidate_ids.into_iter().collect::<Vec<_>>();
    candidate_ids.sort();
    candidate_ids.truncate(CONNECTED_NEIGHBOR_CANDIDATE_LIMIT);
    let candidate_routes = candidate_ids
        .iter()
        .map(|segment_id| {
            routes_by_id
                .get(segment_id.as_str())
                .map(|route| (*route).clone())
                .unwrap_or_else(|| SegmentRoute {
                    segment_id: segment_id.clone(),
                    score: 0.0,
                    fallback: false,
                })
        })
        .collect::<Vec<_>>();
    let profile_ids = selected_ids
        .iter()
        .chain(candidate_ids.iter())
        .collect::<HashSet<_>>();
    let profiles = profile_ids
        .into_iter()
        .filter_map(|segment_id| {
            let position = corpus_stats.segment_positions.get(segment_id)?;
            let segment = segments.get(*position)?;
            let mut profile = corpus_stats
                .connection_profiles
                .get(segment_id)?
                .as_ref()
                .clone();
            add_query_temporal_connection_signal(&mut profile, segment, temporal);
            Some((segment.segment_id.as_str(), profile))
        })
        .collect::<HashMap<_, _>>();
    let mut expansions = Vec::new();
    let mut candidates = candidate_routes
        .iter()
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

fn add_query_temporal_connection_signal(
    profile: &mut SegmentConnectionProfile,
    segment: &MemoryIndexSegment,
    temporal: TemporalQueryContext<'_>,
) {
    let Some(query_date) = temporal.ends_at.and_then(parse_iso_date) else {
        return;
    };
    let window_days = temporal.window_days.max(1);
    if segment.index.docs.values().any(|record| {
        record
            .timestamp
            .as_deref()
            .and_then(parse_iso_date)
            .is_some_and(|timestamp| {
                timestamp.signed_duration_since(query_date).num_days().abs() <= window_days
            })
    }) {
        profile.times.insert("near_query_date".to_string());
    }
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

impl SegmentRoutingSummary {
    fn from_index(index: &MemoryIndex) -> Self {
        // Sorted: docs is a HashMap, and float weight accumulation is
        // order-sensitive at the last ULP. Deterministic input order keeps
        // summaries bit-identical across builds from the same records.
        let mut records = index.docs.values().cloned().collect::<Vec<_>>();
        records.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
        Self::from_records(&records)
    }

    pub fn from_records(records: &[DocRecord]) -> Self {
        let mut profile = Self::default();
        for record in records {
            for term in &record.important_terms {
                add_weight(&mut profile.terms, &term.term, term.score.max(0.1));
            }
            for entity in &record.key_entities {
                let weight = entity.score.unwrap_or(1.0);
                add_weight(&mut profile.entities, &entity.text, weight);
                if entity.source == BEHOOD_NP_ENTITY_SOURCE {
                    // Grammar-accepted entity mentions are indexed literally
                    // (in addition to the stemmed form above): the Porter
                    // stem would conflate the phrase head ("conference" ->
                    // "confer"), destroying the discriminative mention.
                    // The 2x score (set in assemble_doc_record) reflects the
                    // higher precision of grammar-accepted mentions.
                    add_weight_literal(&mut profile.entities, &entity.text, weight);
                }
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

    pub(crate) fn score_query_terms_with_strategy(
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
            SegmentRoutingStrategy::TypedEvidenceMultiplicative => {
                self.coverage_local_distinctiveness_score(query_terms, corpus_stats)
            }
            SegmentRoutingStrategy::CoverageTeamTypedMultiplicative => {
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
        let divergence = sorted_query_terms(query_terms)
            .iter()
            .map(|term| {
                let segment_probability = segment_distribution
                    .get(term.as_str())
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

    pub(crate) fn coverage_local_distinctiveness_score(
        &self,
        query_terms: &HashSet<String>,
        corpus_stats: &SegmentCorpusStats,
    ) -> f32 {
        if query_terms.is_empty() {
            return 0.0;
        }

        let total_query_idf = sorted_query_terms(query_terms)
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

    pub(crate) fn team_coverage_gain(
        &self,
        query_terms: &HashSet<String>,
        corpus_stats: &SegmentCorpusStats,
        already_covered_terms: &HashSet<String>,
    ) -> f32 {
        if query_terms.is_empty() {
            return 0.0;
        }

        let total_query_idf = sorted_query_terms(query_terms)
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

    pub(crate) fn covers_term(&self, term: &str) -> bool {
        self.terms.contains_key(term)
            || self.entities.contains_key(term)
            || self.topics.contains_key(term)
            || self.local_memory.contains_key(term)
    }

    pub(crate) fn local_term_weight(&self, term: &str) -> f32 {
        let term_weight = self.terms.get(term).copied().unwrap_or_default();
        let entity_weight = self.entities.get(term).copied().unwrap_or_default() * 1.8;
        let topic_weight = self.topics.get(term).copied().unwrap_or_default() * 1.2;
        let local_memory_weight = self.local_memory.get(term).copied().unwrap_or_default() * 1.5;
        term_weight + entity_weight + topic_weight + local_memory_weight
    }

    pub(crate) fn coverage_evidence_multiplier(&self, term: &str) -> f32 {
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

    pub(crate) fn evidence_types_for_term(&self, term: &str) -> Vec<String> {
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

impl SegmentCatalog {
    pub(crate) fn from_segments(segments: &[MemoryIndexSegment], generation: u64) -> Self {
        let summaries = segments
            .iter()
            .map(|segment| {
                (
                    segment.segment_id.clone(),
                    Arc::new(SegmentRoutingSummary::from_index(&segment.index)),
                )
            })
            .collect::<HashMap<_, _>>();
        let connection_profiles = segments
            .iter()
            .map(|segment| {
                (
                    segment.segment_id.clone(),
                    Arc::new(segment_connection_profile(
                        segment,
                        TemporalQueryContext::default(),
                    )),
                )
            })
            .collect::<HashMap<_, _>>();
        Self::from_parts(segments, summaries, connection_profiles, generation)
    }

    /// Rebuild the catalog reusing per-segment summaries and connection
    /// profiles for segments whose index object is shared with the previous
    /// snapshot (detected by `Arc` identity — a shared index means identical
    /// records, so its summary cannot have changed). Only new or rebuilt
    /// segments pay for summary construction, and the term-level aggregation
    /// maps are updated by delta: terms belonging to unchanged segments keep
    /// their existing (already sorted) postings, so a single-write refresh
    /// costs O(changed terms) instead of O(distinct corpus terms).
    pub(crate) fn refresh_incremental(
        previous: &Self,
        previous_segments: &[MemoryIndexSegment],
        segments: &[MemoryIndexSegment],
        generation: u64,
    ) -> Self {
        let previous_by_id: HashMap<&str, &MemoryIndexSegment> = previous_segments
            .iter()
            .map(|segment| (segment.segment_id.as_str(), segment))
            .collect();
        let mut summaries = HashMap::with_capacity(segments.len());
        let mut connection_profiles = HashMap::with_capacity(segments.len());
        let mut changed_segment_ids: HashSet<String> = HashSet::new();
        for segment in segments {
            let shared = previous_by_id
                .get(segment.segment_id.as_str())
                .map(|previous_segment| Arc::ptr_eq(&previous_segment.index, &segment.index))
                .unwrap_or(false);
            if shared {
                summaries.insert(
                    segment.segment_id.clone(),
                    Arc::clone(
                        previous
                            .summaries
                            .get(&segment.segment_id)
                            .expect("reused segment must have a catalog summary"),
                    ),
                );
                connection_profiles.insert(
                    segment.segment_id.clone(),
                    Arc::clone(
                        previous
                            .connection_profiles
                            .get(&segment.segment_id)
                            .expect("reused segment must have a connection profile"),
                    ),
                );
            } else {
                changed_segment_ids.insert(segment.segment_id.clone());
                let summary = SegmentRoutingSummary::from_index(&segment.index);
                let profile = segment_connection_profile(segment, TemporalQueryContext::default());
                summaries.insert(segment.segment_id.clone(), Arc::new(summary));
                connection_profiles.insert(segment.segment_id.clone(), Arc::new(profile));
            }
        }
        // Segments present in the previous snapshot but gone now also change
        // the term maps: their contributions must be removed.
        for segment_id in previous.summaries.keys() {
            if !summaries.contains_key(segment_id) {
                changed_segment_ids.insert(segment_id.clone());
            }
        }
        Self::from_parts_incremental(
            previous,
            segments,
            summaries,
            connection_profiles,
            &changed_segment_ids,
            generation,
        )
    }

    /// Assemble a catalog from the previous snapshot's derived maps plus
    /// deltas for `changed_segment_ids`. Produces bit-identical maps to
    /// [`Self::from_parts`]: untouched terms keep their existing postings
    /// (built by the same comparator), and every touched term's postings are
    /// re-sorted with that comparator after the delta is applied.
    fn from_parts_incremental(
        previous: &Self,
        segments: &[MemoryIndexSegment],
        summaries: HashMap<String, Arc<SegmentRoutingSummary>>,
        connection_profiles: HashMap<String, Arc<SegmentConnectionProfile>>,
        changed_segment_ids: &HashSet<String>,
        generation: u64,
    ) -> Self {
        let mut term_segment_counts = previous.term_segment_counts.clone();
        let mut term_to_segments = previous.term_to_segments.clone();
        let mut weighted_term_to_segments = previous.weighted_term_to_segments.clone();
        let mut typed_evidence_to_segments = previous.typed_evidence_to_segments.clone();
        for segment_id in changed_segment_ids {
            apply_term_delta(
                &mut term_segment_counts,
                &mut term_to_segments,
                &mut weighted_term_to_segments,
                segment_id,
                previous.summaries.get(segment_id).map(Arc::as_ref),
                summaries.get(segment_id).map(Arc::as_ref),
            );
            apply_typed_evidence_delta(
                &mut typed_evidence_to_segments,
                segment_id,
                previous
                    .connection_profiles
                    .get(segment_id)
                    .map(Arc::as_ref),
                connection_profiles.get(segment_id).map(Arc::as_ref),
            );
        }
        let mut ordered_segment_ids = summaries.keys().cloned().collect::<Vec<_>>();
        ordered_segment_ids.sort();
        let segment_positions = segments
            .iter()
            .enumerate()
            .map(|(position, segment)| (segment.segment_id.clone(), position))
            .collect();
        Self {
            generation,
            segment_count: segments.len(),
            ordered_segment_ids,
            segment_positions,
            summaries,
            term_segment_counts,
            term_to_segments,
            weighted_term_to_segments,
            typed_evidence_to_segments,
            connection_profiles,
        }
    }

    fn from_parts(
        segments: &[MemoryIndexSegment],
        summaries: HashMap<String, Arc<SegmentRoutingSummary>>,
        connection_profiles: HashMap<String, Arc<SegmentConnectionProfile>>,
        generation: u64,
    ) -> Self {
        let mut term_segment_counts = HashMap::new();
        let mut term_to_segments: HashMap<String, Vec<String>> = HashMap::new();
        for (segment_id, summary) in &summaries {
            let mut segment_terms = HashSet::new();
            segment_terms.extend(summary.terms.keys().cloned());
            segment_terms.extend(summary.entities.keys().cloned());
            segment_terms.extend(summary.topics.keys().cloned());
            segment_terms.extend(summary.local_memory.keys().cloned());
            for term in segment_terms {
                *term_segment_counts.entry(term.clone()).or_default() += 1;
                term_to_segments
                    .entry(term)
                    .or_default()
                    .push(segment_id.clone());
            }
        }
        for segment_ids in term_to_segments.values_mut() {
            segment_ids.sort();
        }
        let mut weighted_term_to_segments: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        for (segment_id, summary) in &summaries {
            for term in summary
                .terms
                .keys()
                .chain(summary.entities.keys())
                .chain(summary.topics.keys())
                .chain(summary.local_memory.keys())
            {
                let postings = weighted_term_to_segments.entry(term.clone()).or_default();
                if postings.iter().any(|(existing, _)| existing == segment_id) {
                    continue;
                }
                postings.push((segment_id.clone(), summary.local_term_weight(term)));
            }
        }
        for postings in weighted_term_to_segments.values_mut() {
            postings.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
        }
        let mut typed_evidence_to_segments: HashMap<String, Vec<String>> = HashMap::new();
        for (segment_id, profile) in &connection_profiles {
            for (prefix, terms) in [
                ("person", &profile.people),
                ("subject", &profile.subjects),
                ("time", &profile.times),
                ("action", &profile.actions),
                ("object", &profile.objects),
            ] {
                for term in terms {
                    typed_evidence_to_segments
                        .entry(format!("{prefix}:{term}"))
                        .or_default()
                        .push(segment_id.clone());
                }
            }
        }
        for segment_ids in typed_evidence_to_segments.values_mut() {
            segment_ids.sort();
            segment_ids.dedup();
        }
        let mut ordered_segment_ids = summaries.keys().cloned().collect::<Vec<_>>();
        ordered_segment_ids.sort();
        let segment_positions = segments
            .iter()
            .enumerate()
            .map(|(position, segment)| (segment.segment_id.clone(), position))
            .collect();
        Self {
            generation,
            segment_count: segments.len(),
            ordered_segment_ids,
            segment_positions,
            summaries,
            term_segment_counts,
            term_to_segments,
            weighted_term_to_segments,
            typed_evidence_to_segments,
            connection_profiles,
        }
    }

    pub(crate) fn summary(&self, segment_id: &str) -> &SegmentRoutingSummary {
        self.summaries
            .get(segment_id)
            .map(Arc::as_ref)
            .unwrap_or_else(|| panic!("segment catalog missing summary for {segment_id}"))
    }

    /// Canonical snapshot of the derived routing maps for differential
    /// tests: sorted term counts, weighted postings in postings order with
    /// weights as bits, and sorted typed-evidence postings.
    #[cfg(test)]
    pub(crate) fn derived_maps_snapshot(
        &self,
    ) -> (
        Vec<(String, usize)>,
        Vec<(String, Vec<(String, u32)>)>,
        Vec<(String, Vec<String>)>,
    ) {
        let mut counts: Vec<(String, usize)> = self
            .term_segment_counts
            .iter()
            .map(|(term, count)| (term.clone(), *count))
            .collect();
        counts.sort();
        let mut weighted: Vec<(String, Vec<(String, u32)>)> = self
            .weighted_term_to_segments
            .iter()
            .map(|(term, postings)| {
                (
                    term.clone(),
                    postings
                        .iter()
                        .map(|(segment_id, weight)| (segment_id.clone(), weight.to_bits()))
                        .collect(),
                )
            })
            .collect();
        weighted.sort();
        let mut typed: Vec<(String, Vec<String>)> = self
            .typed_evidence_to_segments
            .iter()
            .map(|(key, segment_ids)| {
                let mut sorted = segment_ids.clone();
                sorted.sort();
                (key.clone(), sorted)
            })
            .collect();
        typed.sort();
        (counts, weighted, typed)
    }

    pub(crate) fn bounded_candidate_segment_ids(
        &self,
        query_terms: &HashSet<String>,
    ) -> Vec<String> {
        let mut scores = HashMap::<String, f32>::new();
        for term in query_terms {
            let Some(postings) = self.weighted_term_to_segments.get(term) else {
                continue;
            };
            for (segment_id, weight) in postings.iter().take(ROUTING_POSTINGS_PER_TERM) {
                *scores.entry(segment_id.clone()).or_default() += *weight * self.idf(term);
            }
        }
        let mut candidates = scores.into_iter().collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        candidates.truncate(ROUTING_CANDIDATE_POOL_LIMIT);
        candidates
            .into_iter()
            .map(|(segment_id, _)| segment_id)
            .collect()
    }

    pub(crate) fn typed_candidate_segment_ids(
        &self,
        profile: &SegmentConnectionProfile,
    ) -> HashSet<String> {
        let mut candidates = HashSet::new();
        for (prefix, terms) in [
            ("person", &profile.people),
            ("subject", &profile.subjects),
            ("time", &profile.times),
            ("action", &profile.actions),
            ("object", &profile.objects),
        ] {
            for term in terms {
                if let Some(segment_ids) = self
                    .typed_evidence_to_segments
                    .get(&format!("{prefix}:{term}"))
                {
                    candidates.extend(segment_ids.iter().take(ROUTING_POSTINGS_PER_TERM).cloned());
                }
            }
        }
        candidates
    }

    pub(crate) fn is_unique_to_segment(&self, term: &str, segment_id: &str) -> bool {
        self.term_segment_counts.get(term).copied() == Some(1)
            && self
                .term_to_segments
                .get(term)
                .is_some_and(|segments| segments.first().is_some_and(|id| id == segment_id))
    }

    pub(crate) fn idf(&self, term: &str) -> f32 {
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

/// Term -> routing weight for one segment summary: the union of the four
/// evidence maps, weighted exactly as [`SegmentRoutingSummary::local_term_weight`].
fn summary_term_weights(summary: &SegmentRoutingSummary) -> HashMap<&str, f32> {
    let mut weights = HashMap::new();
    for term in summary
        .terms
        .keys()
        .chain(summary.entities.keys())
        .chain(summary.topics.keys())
        .chain(summary.local_memory.keys())
    {
        weights
            .entry(term.as_str())
            .or_insert_with(|| summary.local_term_weight(term));
    }
    weights
}

/// The exact postings comparator used by [`SegmentCatalog::from_parts`].
fn sort_weighted_postings(postings: &mut Vec<(String, f32)>) {
    postings.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
}

/// Update the term-level routing maps for one segment whose summary changed
/// from `old` to `new` (either may be absent for removed/added segments).
/// Only terms in the union of the two summaries are touched; every touched
/// postings list ends up exactly as [`SegmentCatalog::from_parts`] would
/// build it.
fn apply_term_delta(
    term_segment_counts: &mut HashMap<String, usize>,
    term_to_segments: &mut HashMap<String, Vec<String>>,
    weighted_term_to_segments: &mut HashMap<String, Vec<(String, f32)>>,
    segment_id: &str,
    old: Option<&SegmentRoutingSummary>,
    new: Option<&SegmentRoutingSummary>,
) {
    let old_weights = old.map(summary_term_weights).unwrap_or_default();
    let new_weights = new.map(summary_term_weights).unwrap_or_default();
    let mut terms: HashSet<&str> = old_weights.keys().copied().collect();
    terms.extend(new_weights.keys().copied());
    for term in terms {
        match (old_weights.get(term), new_weights.get(term)) {
            (None, None) => unreachable!("term came from one of the weight maps"),
            (Some(_), None) => {
                if let Some(segment_ids) = term_to_segments.get_mut(term) {
                    let pos = segment_ids
                        .binary_search_by(|id| id.as_str().cmp(segment_id))
                        .expect("removed segment must be listed for its term");
                    segment_ids.remove(pos);
                    if segment_ids.is_empty() {
                        term_to_segments.remove(term);
                    }
                }
                if let Some(count) = term_segment_counts.get_mut(term) {
                    *count -= 1;
                    if *count == 0 {
                        term_segment_counts.remove(term);
                    }
                } else {
                    unreachable!("removed segment must have been counted for its term");
                }
                if let Some(postings) = weighted_term_to_segments.get_mut(term) {
                    let pos = postings
                        .iter()
                        .position(|(id, _)| id == segment_id)
                        .expect("removed segment must have a weighted posting for its term");
                    postings.remove(pos);
                    if postings.is_empty() {
                        weighted_term_to_segments.remove(term);
                    }
                }
            }
            (None, Some(&weight)) => {
                let segment_ids = term_to_segments.entry(term.to_string()).or_default();
                if let Err(pos) = segment_ids.binary_search_by(|id| id.as_str().cmp(segment_id)) {
                    segment_ids.insert(pos, segment_id.to_string());
                } else {
                    unreachable!("added segment must not already be listed for its term");
                }
                *term_segment_counts.entry(term.to_string()).or_default() += 1;
                let postings = weighted_term_to_segments
                    .entry(term.to_string())
                    .or_default();
                postings.push((segment_id.to_string(), weight));
                sort_weighted_postings(postings);
            }
            (Some(&old_weight), Some(&new_weight)) => {
                if old_weight.to_bits() != new_weight.to_bits() {
                    let postings = weighted_term_to_segments
                        .get_mut(term)
                        .expect("kept term must have weighted postings");
                    let entry = postings
                        .iter_mut()
                        .find(|(id, _)| id == segment_id)
                        .expect("kept segment must have a weighted posting for its term");
                    entry.1 = new_weight;
                    sort_weighted_postings(postings);
                }
            }
        }
    }
}

/// Typed evidence keys (`"prefix:term"`) for one connection profile.
fn typed_evidence_keys(profile: &SegmentConnectionProfile) -> HashSet<String> {
    [
        ("person", &profile.people),
        ("subject", &profile.subjects),
        ("time", &profile.times),
        ("action", &profile.actions),
        ("object", &profile.objects),
    ]
    .iter()
    .flat_map(|(prefix, terms)| terms.iter().map(move |term| format!("{prefix}:{term}")))
    .collect()
}

/// Update the typed-evidence routing map for one segment whose connection
/// profile changed. Postings stay sorted and deduplicated, as in
/// [`SegmentCatalog::from_parts`].
fn apply_typed_evidence_delta(
    typed_evidence_to_segments: &mut HashMap<String, Vec<String>>,
    segment_id: &str,
    old: Option<&SegmentConnectionProfile>,
    new: Option<&SegmentConnectionProfile>,
) {
    let old_keys = old.map(typed_evidence_keys).unwrap_or_default();
    let new_keys = new.map(typed_evidence_keys).unwrap_or_default();
    for key in old_keys.symmetric_difference(&new_keys) {
        if new_keys.contains(key) {
            let segment_ids = typed_evidence_to_segments.entry(key.clone()).or_default();
            if let Err(pos) = segment_ids.binary_search_by(|id| id.as_str().cmp(segment_id)) {
                segment_ids.insert(pos, segment_id.to_string());
            }
        } else if let Some(segment_ids) = typed_evidence_to_segments.get_mut(key) {
            if let Ok(pos) = segment_ids.binary_search_by(|id| id.as_str().cmp(segment_id)) {
                segment_ids.remove(pos);
            }
            if segment_ids.is_empty() {
                typed_evidence_to_segments.remove(key);
            }
        }
    }
}

pub(crate) fn query_tokens(query: &str) -> HashSet<String> {
    tokenizer::tokenize(query, TokenizerMode::Stemmed)
        .into_iter()
        .filter(|token| !is_routing_stopword(token))
        .collect()
}

/// Routing term set: the plain stemmed query tokens plus the lexical
/// expansion terms the per-segment scorer will also match. Routing on the
/// unexpanded tokens alone strands expansion-only queries (e.g. "diploma",
/// which expands to "degree") on arbitrary fallback segments, so the
/// expansion vocabulary never reaches the segment that actually holds it.
pub(crate) fn query_tokens_expanded(query: &str) -> HashSet<String> {
    let mut terms = query_tokens(query);
    if let Some(prepared) = prepare_query_terms(query) {
        // Expanded terms are already index-normalized; split multi-word
        // expansions into their component tokens but never re-stem them
        // (the stemmer is not idempotent: "degre" would become "degr").
        terms.extend(
            prepared
                .expanded_terms
                .iter()
                .flat_map(|term| term.split_whitespace())
                .filter(|token| !is_routing_stopword(token))
                .map(str::to_string),
        );
    }
    terms
}

/// Query terms in sorted order, for order-independent float summation:
/// HashSet iteration order is nondeterministic and float summation is
/// order-sensitive at the last ULP, which would otherwise make route scores
/// (and therefore top-k selection) differ between identical builds.
pub(crate) fn sorted_query_terms(query_terms: &HashSet<String>) -> Vec<&String> {
    let mut terms: Vec<&String> = query_terms.iter().collect();
    terms.sort();
    terms
}

fn add_weight(distribution: &mut HashMap<String, f32>, text: &str, weight: f32) {
    for token in query_tokens(text) {
        *distribution.entry(token).or_default() += weight;
    }
}

/// Add literal (unstemmed, lowercased) tokens to a distribution. Used for
/// grammar-accepted entity mentions, where the Porter stem conflates the
/// phrase head with an unrelated word ("conference" -> "confer", colliding
/// with the verb "confer"). Literal phrase tokens are rare by construction
/// and carry high IDF, so they dominate acronym-only evidence.
fn add_weight_literal(distribution: &mut HashMap<String, f32>, text: &str, weight: f32) {
    for token in literal_query_tokens(text) {
        *distribution.entry(token).or_default() += weight;
    }
}

/// Query tokens without stemming: lowercase alphanumeric tokens (min 3
/// chars), stopwords removed. Matches the entity channel's literal indexing
/// of grammar-accepted phrases.
pub(crate) fn literal_query_tokens(input: &str) -> Vec<String> {
    tokenizer::tokenize(input, TokenizerMode::Unstemmed)
        .into_iter()
        .filter(|token| !tokenizer::is_stopword(token, TokenizerMode::Unstemmed))
        .collect()
}

pub(crate) fn collect_profile_candidates(
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

pub(crate) fn collect_temporal_candidates(
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
    // Center on the resolved relative-time anchor when the query carries one
    // ("last Tuesday" -> that Tuesday's date); otherwise the reference date.
    // `parse_reference_date` tolerates non-ISO separators.
    let Some(query_date) = temporal
        .anchor_date
        .and_then(parse_reference_date)
        .or_else(|| temporal.ends_at.and_then(parse_reference_date))
    else {
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
        let Some(record_date) = record.timestamp.as_deref().and_then(parse_reference_date) else {
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

/// Temporal route signal as an additive nudge in `[0.0, 1.0]`.
///
/// The caller adds the factor to the segment's content route score, so temporal
/// proximity breaks ties toward temporally relevant segments but can never let
/// a content-weak segment overtake a content-strong one. A segment with no
/// content overlap keeps its (zero) score and cannot be elected on temporal
/// evidence alone.
///
/// The factor is derived from the segment's *best* temporal evidence (max,
/// not sum): one same-day record contributes the full proximity weight, and N
/// mediocre in-window records can no longer saturate the boost and drown the
/// content scores.
pub(crate) fn segment_temporal_route_boost(
    segment: &MemoryIndexSegment,
    temporal: TemporalQueryContext<'_>,
) -> f32 {
    if !temporal.has_explicit_temporal && temporal.time_hint.is_none() {
        return 0.0;
    }
    // Center on the resolved relative-time anchor when the query carries one
    // ("last Tuesday" -> that Tuesday's date); otherwise the reference date.
    // `parse_reference_date` tolerates non-ISO separators.
    let query_date = temporal
        .anchor_date
        .and_then(parse_reference_date)
        .or_else(|| temporal.ends_at.and_then(parse_reference_date));
    let window_days = temporal.window_days.max(1);

    let mut best_proximity = 0.0f32;
    let mut best_hint = 0.0f32;
    // Date math runs over the segment's cached sorted record dates. The
    // factor only depends on the set of dates (max/any are order- and
    // duplicate-independent), so no per-query timestamp parsing is needed.
    if let Some(query_date) = query_date {
        for record_date in segment.record_dates() {
            let delta_days = record_date.signed_duration_since(query_date).num_days();
            let distance = delta_days.abs();
            if distance <= window_days {
                let proximity = 1.0 - (distance as f32 / window_days as f32);
                best_proximity = best_proximity.max(proximity.clamp(0.0, 1.0));
            }
            let hint_boost = match temporal.time_hint {
                Some(TemporalQueryHint::Past) if delta_days <= 0 => 0.18,
                Some(TemporalQueryHint::Present) | Some(TemporalQueryHint::Ongoing)
                    if distance <= 30 =>
                {
                    0.22
                }
                Some(TemporalQueryHint::Mixed) if distance <= 30 => 0.12,
                _ => 0.0,
            };
            best_hint = best_hint.max(hint_boost);
        }
    }
    let has_temporal_terms = segment
        .index
        .docs
        .values()
        .any(|record| !record.temporal_terms.is_empty());

    let mut factor = best_proximity + best_hint;
    if has_temporal_terms {
        factor += 0.08;
    }
    factor.min(1.0)
}

/// Whether the segment holds any record whose timestamp falls inside the
/// resolved anchor window.
///
/// Used for anchored temporal pre-filtering: when a query resolves to a
/// concrete date range, routing is first restricted to segments with in-window
/// evidence, then those are ranked by content. A record with no parseable
/// timestamp never counts as in-window evidence.
///
/// The check binary-searches the segment's cached sorted record dates, so it
/// costs O(log n) per query with no timestamp parsing on the hot path.
pub(crate) fn segment_has_record_in_anchor_window(
    segment: &MemoryIndexSegment,
    window: (NaiveDate, NaiveDate),
) -> bool {
    let (start, end) = window;
    let dates = segment.record_dates();
    let idx = dates.partition_point(|date| *date < start);
    idx < dates.len() && dates[idx] <= end
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

pub(crate) fn parse_iso_date(value: &str) -> Option<NaiveDate> {
    let date = value.get(..10).unwrap_or(value);
    NaiveDate::parse_from_str(date, "%Y-%m-%d").ok()
}

pub(crate) fn is_segment_enrichment_candidate(token: &str) -> bool {
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
    // Summed in sorted-key order: HashMap iteration order is nondeterministic
    // and float summation is order-sensitive at the last ULP.
    let mut keys: Vec<&String> = distribution.keys().collect();
    keys.sort();
    let total: f32 = keys.iter().map(|key| distribution[*key]).sum();
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

pub(crate) fn top_weighted_keys(distribution: &HashMap<String, f32>, limit: usize) -> Vec<String> {
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

pub(crate) fn sorted_terms(terms: &HashSet<String>) -> Vec<String> {
    let mut out = terms.iter().cloned().collect::<Vec<_>>();
    out.sort();
    out
}

pub(crate) fn route_has_signal(
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
        | SegmentRoutingStrategy::TypedEvidence
        | SegmentRoutingStrategy::TypedEvidenceMultiplicative
        | SegmentRoutingStrategy::CoverageTeamTypedMultiplicative => route.score > 0.0,
    }
}

fn is_routing_stopword(token: &str) -> bool {
    tokenizer::is_stopword(token, TokenizerMode::Stemmed)
}
