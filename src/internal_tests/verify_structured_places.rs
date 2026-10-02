//! Harness to verify the structured place-seeking path for
//! "What places has Nate met new people?" (conv-42_q1).
//!
//! Verifies one by one:
//! 1. analyze_fact_question returns Some with expect_place=true, persons=["Nate"]
//! 2. extract_activity_verb returns "meet"
//! 3. docs_for_activity returns doc_ids for Nate+meet
//! 4. query_structured returns hits (score 1000+)

use crate::segments::relations::{
    analyze_fact_question, extract_activity_verb, query_structured, RawRelation, RelationIndex,
    RelationTurn,
};

const QUESTION: &str = "What places has Nate met new people?";

fn fixture_turns() -> Vec<RelationTurn> {
    vec![
        RelationTurn {
            speaker: "Nate".to_string(),
            session_id: "conv-42::session_14".to_string(),
            turn_idx: 7,
            doc_id: "doc14_7".to_string(),
            text: "I've been doing great - I just won another regional video game tournament last week! It was so cool, plus I met some new people.".to_string(),
            session_date: Some("2022-06-03".to_string()),
        },
        RelationTurn {
            speaker: "Nate".to_string(),
            session_id: "conv-42::session_23".to_string(),
            turn_idx: 0,
            doc_id: "doc23_0".to_string(),
            text: "I went to a game convention and met new people who weren't from my normal circle.".to_string(),
            session_date: Some("2022-07-15".to_string()),
        },
    ]
}

fn fixture_raw() -> Vec<RawRelation> {
    vec![
        RawRelation {
            subject: "Nate".to_string(),
            predicate: "meet".to_string(),
            object: "some new people".to_string(),
            object_kind: "thing".to_string(),
            is_place: false,
            is_activity: true,
            session_id: "conv-42::session_14".to_string(),
            turn_idx: 7,
            doc_id: "doc14_7".to_string(),
            session_date: Some("2022-06-03".to_string()),
            evidence: String::new(),
            confidence: 1.0,
            coref: None,
        },
        RawRelation {
            subject: "Nate".to_string(),
            predicate: "meet".to_string(),
            object: "new people".to_string(),
            object_kind: "thing".to_string(),
            is_place: false,
            is_activity: true,
            session_id: "conv-42::session_23".to_string(),
            turn_idx: 0,
            doc_id: "doc23_0".to_string(),
            session_date: Some("2022-07-15".to_string()),
            evidence: String::new(),
            confidence: 1.0,
            coref: None,
        },
        RawRelation {
            subject: "Nate".to_string(),
            predicate: "go_to".to_string(),
            object: "game convention".to_string(),
            object_kind: "place".to_string(),
            is_place: true,
            is_activity: true,
            session_id: "conv-42::session_23".to_string(),
            turn_idx: 0,
            doc_id: "doc23_0".to_string(),
            session_date: Some("2022-07-15".to_string()),
            evidence: String::new(),
            confidence: 1.0,
            coref: None,
        },
    ]
}

#[test]
fn step1_analyzer_returns_expect_place() {
    let fq = analyze_fact_question(QUESTION);
    assert!(
        fq.is_some(),
        "STEP 1 FAILED: analyze_fact_question returned None"
    );
    let fq = fq.unwrap();
    println!("persons: {:?}", fq.persons);
    println!("expect_place: {}", fq.expect_place);
    println!("time_window: {:?}", fq.time_window);
    assert_eq!(fq.persons.len(), 1, "STEP 1 FAILED: expected 1 person");
    assert!(fq.expect_place, "STEP 1 FAILED: expect_place is false");
    assert!(
        fq.time_window.is_none(),
        "STEP 1 FAILED: expected no time window"
    );
    println!("STEP 1 PASSED");
}

#[test]
fn step2_activity_verb_is_meet() {
    let verb = extract_activity_verb(QUESTION, "Nate");
    println!("activity verb: {:?}", verb);
    assert_eq!(
        verb.as_deref(),
        Some("meet"),
        "STEP 2 FAILED: expected 'meet'"
    );
    println!("STEP 2 PASSED");
}

#[test]
fn step3_docs_for_activity() {
    let index = RelationIndex::build(&fixture_turns(), &fixture_raw());
    let docs = index.docs_for_activity("Nate", "meet");
    println!("doc_ids: {:?}", docs);
    assert!(!docs.is_empty(), "STEP 3 FAILED: no doc_ids returned");
    assert!(
        docs.contains(&"doc14_7".to_string()),
        "STEP 3 FAILED: missing session_14 doc"
    );
    assert!(
        docs.contains(&"doc23_0".to_string()),
        "STEP 3 FAILED: missing session_23 doc"
    );
    println!("STEP 3 PASSED");
}

#[test]
fn step4_query_structured_returns_hits() {
    let index = RelationIndex::build(&fixture_turns(), &fixture_raw());
    let hits = query_structured(&index, QUESTION);
    println!("hits: {:?}", hits.as_ref().map(|h| h.len()));
    assert!(
        hits.is_some(),
        "STEP 4 FAILED: query_structured returned None"
    );
    let hits = hits.unwrap();
    assert!(!hits.is_empty(), "STEP 4 FAILED: empty hits");
    assert!(
        hits.iter().all(|h| h.score >= 1000.0),
        "STEP 4 FAILED: scores < 1000"
    );
    println!("STEP 4 PASSED: {} hits", hits.len());
}
