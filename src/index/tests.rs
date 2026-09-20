use super::helpers::*;
use super::model::*;

use crate::query_expansion::normalize_for_index;

#[test]
fn text_overlap_prefers_matching_candidate() {
    let query = tokenize_query_terms(&normalize_for_index("What degree did I graduate with?"));
    let good = tokenize_query_terms(&normalize_for_index(
        "I graduated with a computer science degree.",
    ));
    let bad = tokenize_query_terms(&normalize_for_index("I cooked pasta and watched a movie."));

    assert!(token_overlap_ratio(&query, &good) > token_overlap_ratio(&query, &bad));
    assert!(ngram_overlap_ratio(&query, &good, 2) > ngram_overlap_ratio(&query, &bad, 2));
    assert!(lcs_ratio(&query, &good) > lcs_ratio(&query, &bad));
}

#[test]
fn timestamped_chunk_detects_chunk_level_temporal_anchor() {
    let doc = DocRecord {
        doc_id: "doc-1".to_string(),
        source: "source://doc-1".to_string(),
        content: "content".to_string(),
        timestamp: None,
        doc_length: 7,
        author_agent: None,
        group_id: None,
        filters: std::collections::BTreeMap::new(),
        probable_topic: None,
        doc_type_guess: None,
        headings: vec!["Overview".to_string()],
        doc_links: vec![],
        temporal_terms: vec![],
        key_entities: vec![],
        important_terms: vec![],
        section_chunks: vec![SectionChunk {
            chunk_id: "doc-1::0".to_string(),
            heading: "Overview".to_string(),
            content: "content".to_string(),
            start_line: 1,
            end_line: 1,
            timestamp: Some("2024-05-10".to_string()),
            key_entities: vec![],
            important_terms: vec![],
        }],
        embedding: None,
        top_claims: vec![],
        provenance: Provenance {
            source: "source://doc-1".to_string(),
            timestamp: None,
            ner_provider: "heuristic".to_string(),
            term_ranker: "yake".to_string(),
            index_version: "v1".to_string(),
        },
        content_hash: String::new(),
    };

    assert!(doc_has_timestamped_chunk(&doc));
}

#[test]
fn normalize_temporal_bounds_orders_reversed_ranges_and_preserves_same_day() {
    let (start, end) = normalize_temporal_bounds(Some("2024-05-10"), Some("2024-05-01"));
    assert_eq!(
        start,
        Some(chrono::NaiveDate::from_ymd_opt(2024, 5, 1).expect("valid date"))
    );
    assert_eq!(
        end,
        Some(chrono::NaiveDate::from_ymd_opt(2024, 5, 10).expect("valid date"))
    );

    let (same_start, same_end) = normalize_temporal_bounds(Some("2024-05-10"), Some("2024-05-10"));
    assert_eq!(same_start, same_end);
    assert_eq!(
        same_start,
        Some(chrono::NaiveDate::from_ymd_opt(2024, 5, 10).expect("valid date"))
    );
}
