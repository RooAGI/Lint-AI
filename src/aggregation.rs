use crate::index::{MemoryIndex, SearchResult};
use regex::Regex;
use serde::Serialize;
use std::collections::HashSet;
use std::sync::OnceLock;
use text2num::{replace_numbers_in_text, Language};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateIntent {
    Count,
    Sum,
}

#[derive(Debug, Clone, Serialize)]
pub struct AggregateCitation {
    pub doc_id: String,
    pub source: String,
    pub group_id: Option<String>,
    pub score: f32,
    pub evidence_value: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AggregateOutput {
    pub intent: String,
    pub normalized_query: String,
    pub retrieved_candidates: usize,
    pub evidence_count: usize,
    pub answer_value: Option<f64>,
    pub answer_text: String,
    pub citations: Vec<AggregateCitation>,
    pub reasoning: String,
}

pub fn classify_aggregate_intent(query: &str) -> Option<AggregateIntent> {
    let q = query.to_lowercase();
    if q.contains("how many")
        || q.starts_with("count ")
        || q.contains(" number of ")
        || q.contains(" number of")
        || q.contains("count the ")
        || q.contains("times did i")
    {
        return Some(AggregateIntent::Count);
    }
    if q.contains("how much")
        || q.contains(" total ")
        || q.starts_with("total ")
        || q.contains("combined")
        || q.contains("in all")
        || q.contains("sum ")
    {
        return Some(AggregateIntent::Sum);
    }
    None
}

pub fn normalize_number_words(query: &str) -> String {
    replace_numbers_in_text(query, &Language::english(), 0.0)
}

pub fn build_aggregate_output(
    index: &MemoryIndex,
    query: &str,
    results: &[SearchResult],
    candidate_limit: usize,
) -> Option<AggregateOutput> {
    let intent = classify_aggregate_intent(query)?;
    let normalized_query = normalize_number_words(query);
    let limit = candidate_limit.max(1).min(results.len().max(1));
    let relevant = &results[..limit.min(results.len())];
    let citations = match intent {
        AggregateIntent::Count => build_count_citations(index, relevant),
        AggregateIntent::Sum => build_sum_citations(index, relevant),
    };

    let answer_value = match intent {
        AggregateIntent::Count => Some(citations.len() as f64),
        AggregateIntent::Sum => {
            let sum = citations
                .iter()
                .filter_map(|c| c.evidence_value)
                .sum::<f64>();
            if sum > 0.0 {
                Some(sum)
            } else {
                None
            }
        }
    };

    let answer_text = match (intent, answer_value) {
        (AggregateIntent::Count, Some(v)) => format!("{v:.0}"),
        (AggregateIntent::Sum, Some(v)) => format!("{v:.2}"),
        _ => "unknown".to_string(),
    };
    let intent_name = match intent {
        AggregateIntent::Count => "count",
        AggregateIntent::Sum => "sum",
    };
    let reasoning = match intent {
        AggregateIntent::Count => format!(
            "counted {} unique evidence items from the top {} retrieved candidates",
            citations.len(),
            limit
        ),
        AggregateIntent::Sum => format!(
            "summed numeric evidence from {} supporting candidates in the top {} retrieved candidates",
            citations.iter().filter(|c| c.evidence_value.is_some()).count(),
            limit
        ),
    };

    Some(AggregateOutput {
        intent: intent_name.to_string(),
        normalized_query,
        retrieved_candidates: limit,
        evidence_count: citations.len(),
        answer_value,
        answer_text,
        citations,
        reasoning,
    })
}

fn build_count_citations(index: &MemoryIndex, results: &[SearchResult]) -> Vec<AggregateCitation> {
    let mut seen = HashSet::new();
    let mut citations = Vec::new();
    for result in results {
        let key = result
            .group_id
            .clone()
            .unwrap_or_else(|| result.doc_id.clone());
        if !seen.insert(key) {
            continue;
        }
        let citation = AggregateCitation {
            doc_id: result.doc_id.clone(),
            source: result.source.clone(),
            group_id: result.group_id.clone(),
            score: result.score,
            evidence_value: None,
        };
        citations.push(citation);
        if citations.len() >= index.docs.len().min(results.len()) {
            break;
        }
    }
    citations
}

fn build_sum_citations(index: &MemoryIndex, results: &[SearchResult]) -> Vec<AggregateCitation> {
    let mut seen = HashSet::new();
    let mut citations = Vec::new();
    for result in results {
        let key = result
            .group_id
            .clone()
            .unwrap_or_else(|| result.doc_id.clone());
        if !seen.insert(key) {
            continue;
        }
        let Some(doc) = index.docs.get(&result.doc_id) else {
            continue;
        };
        let evidence_text = aggregate_text(doc);
        let evidence_value = extract_numeric_value(&evidence_text);
        let citation = AggregateCitation {
            doc_id: result.doc_id.clone(),
            source: result.source.clone(),
            group_id: result.group_id.clone(),
            score: result.score,
            evidence_value,
        };
        citations.push(citation);
    }
    citations
}

fn aggregate_text(doc: &crate::index::DocRecord) -> String {
    let mut parts = Vec::new();
    parts.push(doc.headings.join(" "));
    parts.push(doc.probable_topic.clone().unwrap_or_default());
    parts.push(doc.doc_type_guess.clone().unwrap_or_default());
    parts.push(doc.temporal_terms.join(" "));
    parts.push(doc.content.clone());
    parts.join("\n")
}

fn extract_numeric_value(text: &str) -> Option<f64> {
    let normalized = normalize_number_words(text);
    static NUMBER_RE: OnceLock<Regex> = OnceLock::new();
    let re = NUMBER_RE
        .get_or_init(|| Regex::new(r"(?i)\b\d+(?:\.\d+)?\b").expect("valid numeric regex"));
    let mut values = Vec::new();
    for mat in re.find_iter(&normalized) {
        if let Ok(v) = mat.as_str().parse::<f64>() {
            values.push(v);
        }
    }
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{Claim, DocRecord, Provenance, ScoreBreakdown, SearchResult, SectionChunk};
    use crate::tier1::{RankedTerm, Tier1Entity};

    fn sample_doc(doc_id: &str, content: &str, group_id: Option<&str>) -> DocRecord {
        DocRecord {
            doc_id: doc_id.to_string(),
            source: format!("source://{}", doc_id),
            content: content.to_string(),
            timestamp: Some("2024-05-10".to_string()),
            doc_length: content.len(),
            author_agent: None,
            group_id: group_id.map(|g| g.to_string()),
            filters: std::collections::BTreeMap::new(),
            probable_topic: Some("hiking".to_string()),
            doc_type_guess: Some("note".to_string()),
            headings: vec!["Overview".to_string()],
            doc_links: vec![],
            temporal_terms: vec!["date friday".to_string()],
            key_entities: vec![Tier1Entity {
                text: "hike".to_string(),
                label: "PROPN".to_string(),
                start: 0,
                end: 4,
                score: Some(1.0),
                source: "heuristic".to_string(),
            }],
            important_terms: vec![RankedTerm {
                term: "hike".to_string(),
                score: 1.0,
                source: "yake".to_string(),
            }],
            section_chunks: vec![SectionChunk {
                chunk_id: format!("{}::chunk0", doc_id),
                heading: "Overview".to_string(),
                content: content.to_string(),
                start_line: 1,
                end_line: 1,
                timestamp: Some("2024-05-10".to_string()),
                key_entities: vec!["hike".to_string()],
                important_terms: vec!["hike".to_string()],
            }],
            embedding: None,
            top_claims: vec![Claim {
                subject: "a".to_string(),
                predicate: "b".to_string(),
                object: "c".to_string(),
                confidence: 1.0,
            }],
            provenance: Provenance {
                source: "source://doc".to_string(),
                timestamp: Some("2024-05-10".to_string()),
                ner_provider: "heuristic".to_string(),
                term_ranker: "yake".to_string(),
                index_version: "v1".to_string(),
            },
        }
    }

    #[test]
    fn classifies_count_and_sum_intents() {
        assert_eq!(
            classify_aggregate_intent("How many hikes did I do?"),
            Some(AggregateIntent::Count)
        );
        assert_eq!(
            classify_aggregate_intent("How much did I spend?"),
            Some(AggregateIntent::Sum)
        );
    }

    #[test]
    fn normalizes_number_words() {
        let out = normalize_number_words("I walked two miles and ate three apples");
        assert!(out.contains("2"));
        assert!(out.contains("3"));
    }

    #[test]
    fn builds_count_aggregation() {
        let index = MemoryIndex::from_records(vec![
            sample_doc("d1", "I walked 2 miles", Some("g1")),
            sample_doc("d2", "I walked 3 miles", Some("g2")),
        ]);
        let results = vec![
            SearchResult {
                doc_id: "d1".to_string(),
                source: "source://d1".to_string(),
                group_id: Some("g1".to_string()),
                score: 1.0,
                score_breakdown: ScoreBreakdown::default(),
                matched_entities: vec![],
                matched_terms: vec![],
                probable_topic: Some("hiking".to_string()),
                doc_type_guess: Some("note".to_string()),
            },
            SearchResult {
                doc_id: "d2".to_string(),
                source: "source://d2".to_string(),
                group_id: Some("g2".to_string()),
                score: 0.9,
                score_breakdown: ScoreBreakdown::default(),
                matched_entities: vec![],
                matched_terms: vec![],
                probable_topic: Some("hiking".to_string()),
                doc_type_guess: Some("note".to_string()),
            },
        ];
        let agg = build_aggregate_output(&index, "How many miles did I walk?", &results, 5)
            .expect("expected aggregation");
        assert_eq!(agg.intent, "count");
        assert!(agg.answer_value.is_some());
        assert!(!agg.citations.is_empty());
    }
}
