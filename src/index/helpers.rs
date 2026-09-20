use crate::query_expansion::normalize_for_index;
use crate::query_semantics::QueryRoutingIntent;
use crate::temporal::parse_temporal_date;
use crate::tier1::Tier1Entity;
use crate::tokenizer::{self, TokenizerMode};
use chrono::NaiveDate;
use deunicode::deunicode;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use super::model::*;
use super::query_terms::*;

pub(crate) fn tokenize_query_terms(input: &str) -> Vec<String> {
    tokenizer::tokenize(input, TokenizerMode::Unstemmed)
}

pub(crate) fn claim_tokens(claim: &Claim) -> Vec<String> {
    let mut tokens = Vec::new();
    tokens.extend(tokenize_query_terms(&normalize_for_index(&claim.subject)));
    tokens.extend(tokenize_query_terms(&normalize_for_index(&claim.predicate)));
    tokens.extend(tokenize_query_terms(&normalize_for_index(&claim.object)));
    tokens.sort();
    tokens.dedup();
    tokens
}

pub(crate) fn sanitize_bm25_query(query: &str) -> String {
    static FALLBACK_RE: OnceLock<Regex> = OnceLock::new();
    let lowered = deunicode(query).to_lowercase();
    let token_re = FALLBACK_RE.get_or_init(|| {
        Regex::new(r"[A-Za-z0-9][A-Za-z0-9_-]*").expect("valid fallback query regex")
    });
    token_re
        .find_iter(&lowered)
        .map(|m| m.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn candidate_rerank_text(doc: &DocRecord) -> String {
    let mut parts: Vec<String> = Vec::new();
    if !doc.headings.is_empty() {
        parts.push(doc.headings.join(" "));
    }
    if let Some(topic) = doc.probable_topic.as_ref() {
        parts.push(topic.clone());
    }
    if let Some(doc_type) = doc.doc_type_guess.as_ref() {
        parts.push(doc_type.clone());
    }
    if !doc.important_terms.is_empty() {
        parts.push(
            doc.important_terms
                .iter()
                .map(|t| t.term.clone())
                .collect::<Vec<_>>()
                .join(" "),
        );
    }
    if !doc.key_entities.is_empty() {
        parts.push(
            doc.key_entities
                .iter()
                .map(|e| e.text.clone())
                .collect::<Vec<_>>()
                .join(" "),
        );
    }
    if !doc.temporal_terms.is_empty() {
        parts.push(doc.temporal_terms.join(" "));
    }
    let content_snippet: String = doc
        .content
        .chars()
        .take(TEXT_RERANK_CONTENT_CHARS)
        .collect();
    if !content_snippet.is_empty() {
        parts.push(content_snippet);
    }
    parts.join(" ")
}

pub(crate) fn derive_doc_postings(
    chunk_postings: &[Vec<(u32, f32)>],
    chunks: &[ChunkMeta],
    cap: usize,
) -> Vec<Vec<(u32, f32)>> {
    chunk_postings
        .iter()
        .map(|postings| {
            let mut doc_scores: HashMap<u32, f32> = HashMap::new();
            for &(chunk_u32, score) in postings {
                let doc_u32 = chunks[chunk_u32 as usize].doc_u32;
                doc_scores
                    .entry(doc_u32)
                    .and_modify(|s| *s = s.max(score))
                    .or_insert(score);
            }
            let mut out: Vec<(u32, f32)> = doc_scores.into_iter().collect();
            out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            out.truncate(cap);
            out
        })
        .collect()
}

pub(crate) fn build_doc_rerank_cache(doc: &DocRecord) -> (String, Vec<String>) {
    let rerank_text = candidate_rerank_text(doc);
    let normalized = normalize_for_index(&rerank_text);
    let tokens = tokenize_query_terms(&normalized);
    (normalized, tokens)
}

pub(crate) fn cached_doc_rerank_text(index: &MemoryIndex, doc_u32: usize) -> Option<&str> {
    index.doc_rerank_texts.get(doc_u32).map(|s| s.as_str())
}

pub(crate) fn cached_doc_rerank_tokens(index: &MemoryIndex, doc_u32: usize) -> Option<&[String]> {
    index
        .doc_rerank_tokens
        .get(doc_u32)
        .map(|tokens| tokens.as_slice())
}

#[cfg(test)]
pub(crate) fn doc_has_timestamped_chunk(doc: &DocRecord) -> bool {
    doc.section_chunks
        .iter()
        .any(|chunk| chunk.timestamp.is_some())
}

pub(crate) const TEMPORAL_PROXIMITY_WEIGHT: f32 = 0.45;
pub(crate) const TEMPORAL_RANGE_BOOST: f32 = 0.15;

pub(crate) fn doc_temporal_date(doc: &DocRecord) -> Option<NaiveDate> {
    doc.section_chunks
        .iter()
        .find_map(|chunk| parse_temporal_date(chunk.timestamp.as_deref()))
        .or_else(|| parse_temporal_date(doc.timestamp.as_deref()))
}

pub(crate) fn normalize_temporal_bounds(
    starts_from: Option<&str>,
    ends_at: Option<&str>,
) -> (Option<NaiveDate>, Option<NaiveDate>) {
    let start = parse_temporal_date(starts_from);
    let end = parse_temporal_date(ends_at);
    match (start, end) {
        (Some(start), Some(end)) if start > end => (Some(end), Some(start)),
        (start, end) => (start, end),
    }
}

pub(crate) fn seed_routing_candidates(
    index: &MemoryIndex,
    candidates: &mut HashMap<usize, CandidateState>,
    candidate_doc_ids: &[usize],
    q_terms: &[String],
    query_entities: &HashSet<String>,
    intent: QueryRoutingIntent,
) {
    if candidate_doc_ids.is_empty() {
        return;
    }

    let content_terms = routing_content_terms(q_terms);
    let unit_terms = routing_unit_terms(q_terms);

    for doc_u32 in candidate_doc_ids.iter().copied() {
        let Some(doc_id) = index.doc_u32_to_id.get(doc_u32) else {
            continue;
        };
        let Some(doc) = index.docs.get(doc_id) else {
            continue;
        };
        let Some(doc_terms) = cached_doc_rerank_tokens(index, doc_u32) else {
            continue;
        };
        if doc_terms.is_empty() {
            continue;
        }
        let matched_terms = content_terms
            .iter()
            .filter(|term| doc_terms.iter().any(|tok| tok == *term))
            .count() as f32;
        let doc_entities = index
            .doc_key_entities
            .get(doc_u32)
            .map(|keys| keys.as_slice())
            .unwrap_or(&[]);
        let matched_entities = doc_entities
            .iter()
            .filter(|entity| query_entities.contains(entity.as_str()))
            .count() as f32;
        let has_number = index.doc_has_number.get(doc_u32).copied().unwrap_or(false);
        let has_unit = !unit_terms.is_empty()
            && unit_terms
                .iter()
                .any(|unit| doc_terms.iter().any(|tok| tok == unit));
        let has_date = doc_temporal_date(doc).is_some();
        let has_temporal_terms = !doc.temporal_terms.is_empty();

        let mut score = match intent {
            QueryRoutingIntent::Count => {
                matched_terms * 0.24
                    + matched_entities * 0.35
                    + if has_date { 0.05 } else { 0.0 }
                    + if has_temporal_terms { 0.04 } else { 0.0 }
            }
            QueryRoutingIntent::Sum => {
                matched_terms * 0.20
                    + matched_entities * 0.12
                    + if has_number { 0.30 } else { 0.0 }
                    + if has_unit { 0.18 } else { 0.0 }
                    + if has_date { 0.04 } else { 0.0 }
            }
            QueryRoutingIntent::Sequence => {
                matched_terms * 0.20
                    + matched_entities * 0.12
                    + if has_date { 0.28 } else { 0.0 }
                    + if has_temporal_terms { 0.14 } else { 0.0 }
                    + if has_number { 0.08 } else { 0.0 }
            }
        };

        if score <= 0.0 {
            continue;
        }
        if matches!(intent, QueryRoutingIntent::Sequence) && !has_date && !has_temporal_terms {
            continue;
        }
        if matches!(intent, QueryRoutingIntent::Sum) && !has_number && !has_unit {
            continue;
        }

        score = score.min(0.85);
        let entry = candidates.entry(doc_u32).or_default();
        entry.score += score;
        entry.breakdown.semantic_score += score;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn group_evidence_boost(
    intent: QueryRoutingIntent,
    supporting_docs: usize,
    evidence_score: f32,
    evidence_terms: usize,
    evidence_numbers: usize,
    evidence_units: usize,
    evidence_dates: usize,
    evidence_predicates: usize,
    evidence_penalty: f32,
) -> f32 {
    if supporting_docs == 0 {
        return 0.0;
    }

    let mut boost = evidence_score.min(4.0);
    boost += supporting_docs.min(4) as f32 * 0.22;
    boost += evidence_terms.min(8) as f32 * 0.04;
    boost += evidence_predicates.min(3) as f32 * 0.24;

    match intent {
        QueryRoutingIntent::Count => {
            boost += supporting_docs.saturating_sub(1).min(3) as f32 * 0.22;
        }
        QueryRoutingIntent::Sum => {
            boost += evidence_numbers.min(3) as f32 * 0.30;
            boost += evidence_units.min(3) as f32 * 0.18;
        }
        QueryRoutingIntent::Sequence => {
            boost += evidence_dates.min(4) as f32 * 0.30;
            boost += evidence_numbers.min(2) as f32 * 0.12;
        }
    }

    (boost - evidence_penalty.min(1.5)).max(0.0)
}

pub(crate) fn routing_content_terms(q_terms: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    q_terms
        .iter()
        .filter(|term| term.len() >= 3)
        .filter(|term| !tokenizer::is_stopword(term, TokenizerMode::Unstemmed))
        .filter(|term| seen.insert((*term).clone()))
        .take(16)
        .cloned()
        .collect()
}

pub(crate) fn routing_unit_terms(query: &[String]) -> Vec<String> {
    let units = [
        "mile",
        "miles",
        "km",
        "kilometer",
        "kilometers",
        "meter",
        "meters",
        "hour",
        "hours",
        "minute",
        "minutes",
        "day",
        "days",
        "week",
        "weeks",
        "month",
        "months",
        "dollar",
        "dollars",
        "usd",
        "pound",
        "pounds",
        "kg",
        "kilogram",
        "kilograms",
        "screen",
        "time",
    ];
    let unit_set: HashSet<&str> = units.into_iter().collect();
    query
        .iter()
        .filter(|term| unit_set.contains(term.as_str()))
        .cloned()
        .collect()
}

pub(crate) fn contains_number_like(text: &str) -> bool {
    static NUMBER_RE: OnceLock<Regex> = OnceLock::new();
    let re = NUMBER_RE.get_or_init(|| {
        Regex::new(
            r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
        )
        .expect("valid number regex")
    });
    re.is_match(text)
}

pub(crate) fn has_predicate_signal(text: &str, intent: QueryRoutingIntent) -> bool {
    let terms = match intent {
        QueryRoutingIntent::Count => [
            "led",
            "lead",
            "leading",
            "visited",
            "attended",
            "bought",
            "purchased",
            "baked",
            "cooked",
            "tried",
            "used",
            "played",
            "learned",
            "presented",
            "participated",
            "sibling",
            "brother",
            "sister",
            "doctor",
            "project",
        ]
        .as_slice(),
        QueryRoutingIntent::Sum => [
            "total",
            "distance",
            "cost",
            "spent",
            "paid",
            "miles",
            "kilometers",
            "hours",
            "days",
        ]
        .as_slice(),
        QueryRoutingIntent::Sequence => [
            "consecutive",
            "before",
            "after",
            "weekend",
            "days",
            "weeks",
            "earliest",
            "latest",
            "first",
            "last",
        ]
        .as_slice(),
    };
    terms.iter().any(|term| text.contains(term))
}

pub(crate) fn routing_distractor_penalty(text: &str, intent: QueryRoutingIntent) -> f32 {
    let generic_markers = [
        "write a story",
        "fiction",
        "character",
        "captain america",
        "area of",
        "formula",
        "table showing",
        "census",
        "survey",
        "article",
        "research studies",
        "here are some",
        "resources",
    ];
    let mut penalty = generic_markers
        .iter()
        .filter(|marker| text.contains(**marker))
        .count() as f32
        * 0.35;

    if matches!(intent, QueryRoutingIntent::Count) {
        if text.contains("how many") && text.contains("here are") {
            penalty += 0.25;
        }
        if text.contains("formula") || text.contains("approximately") {
            penalty += 0.35;
        }
    }

    penalty.min(1.4)
}

pub(crate) fn aggregate_group_score(items: &[(usize, f32)], count_query: bool) -> f32 {
    if items.is_empty() {
        return 0.0;
    }
    let base = items[0].1;
    let support = items
        .iter()
        .skip(1)
        .take(4)
        .enumerate()
        .map(|(idx, (_, score))| {
            let decay = if count_query {
                match idx {
                    0 => 0.45,
                    1 => 0.28,
                    2 => 0.16,
                    _ => 0.08,
                }
            } else {
                match idx {
                    0 => 0.30,
                    1 => 0.18,
                    2 => 0.10,
                    _ => 0.05,
                }
            };
            decay * *score
        })
        .sum::<f32>();
    let supporting_docs = items
        .iter()
        .skip(1)
        .filter(|(_, score)| *score >= (items[0].1 * 0.35).max(0.15))
        .count();
    let coverage = supporting_docs.min(4) as f32 * if count_query { 0.08 } else { 0.05 };
    if count_query {
        base * 0.75 + support + coverage
    } else {
        base + support + coverage
    }
}

pub(crate) fn normalized_entity_keys(entities: &[Tier1Entity]) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for entity in entities {
        let key = normalize_for_index(&entity.text);
        if key.is_empty() || !seen.insert(key.clone()) {
            continue;
        }
        out.push(key);
    }
    out
}

// normalize_for_index moved to query_expansion.rs and reused here.

pub(crate) fn token_overlap_ratio(query_tokens: &[String], candidate_tokens: &[String]) -> f32 {
    if query_tokens.is_empty() || candidate_tokens.is_empty() {
        return 0.0;
    }
    let candidate_set: HashSet<&str> = candidate_tokens.iter().map(String::as_str).collect();
    let matched = query_tokens
        .iter()
        .map(String::as_str)
        .filter(|t| candidate_set.contains(t))
        .count();
    matched as f32 / query_tokens.len().max(1) as f32
}

pub(crate) fn ngram_overlap_ratio(
    query_tokens: &[String],
    candidate_tokens: &[String],
    n: usize,
) -> f32 {
    if n == 0 || query_tokens.len() < n || candidate_tokens.len() < n {
        return 0.0;
    }
    let query_ngrams = build_ngrams(query_tokens, n);
    if query_ngrams.is_empty() {
        return 0.0;
    }
    let candidate_ngrams = build_ngrams(candidate_tokens, n);
    if candidate_ngrams.is_empty() {
        return 0.0;
    }
    let candidate_set: HashSet<Vec<String>> = candidate_ngrams.into_iter().collect();
    let overlap = query_ngrams
        .into_iter()
        .filter(|g| candidate_set.contains(g))
        .count();
    overlap as f32 / query_tokens.len().saturating_sub(n - 1).max(1) as f32
}

fn build_ngrams(tokens: &[String], n: usize) -> Vec<Vec<String>> {
    if n == 0 || tokens.len() < n {
        return Vec::new();
    }
    tokens.windows(n).map(|win| win.to_vec()).collect()
}

pub(crate) fn lcs_ratio(query_tokens: &[String], candidate_tokens: &[String]) -> f32 {
    if query_tokens.is_empty() || candidate_tokens.is_empty() {
        return 0.0;
    }
    let mut prev = vec![0usize; candidate_tokens.len() + 1];
    let mut cur = vec![0usize; candidate_tokens.len() + 1];
    for q in query_tokens {
        for (j, cand) in candidate_tokens.iter().enumerate() {
            if q == cand {
                cur[j + 1] = prev[j] + 1;
            } else {
                cur[j + 1] = cur[j].max(prev[j + 1]);
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[candidate_tokens.len()] as f32 / query_tokens.len().max(1) as f32
}
