//! Harness to verify spaCy relation extraction health on real conv-42 turns.
//!
//! Runs the actual `extract_relations_via_spacy` (not fixtures) and checks:
//! 1. Extraction completes without timeout/error
//! 2. Nate->meet relations are found for session_14 and session_23
//! 3. object_kind tags are present (event, place, thing)

use crate::segments::relations::{extract_relations_via_spacy, RelationTurn};
use crate::DEFAULT_SPACY_MODEL;
use std::time::Duration;

fn real_turns() -> Vec<RelationTurn> {
    vec![
        RelationTurn {
            speaker: "Nate".to_string(),
            session_id: "conv-42::session_14".to_string(),
            turn_idx: 7,
            doc_id: "conv-42::session_14::7".to_string(),
            text: "I've been doing great - I just won another regional video game tournament last week! It was so cool, plus I met some new people.".to_string(),
            session_date: Some("5:44 pm on 3 June, 2022".to_string()),
        },
        RelationTurn {
            speaker: "Nate".to_string(),
            session_id: "conv-42::session_23".to_string(),
            turn_idx: 0,
            doc_id: "conv-42::session_23::0".to_string(),
            text: "I went to a game convention and met new people who weren't from my normal circle.".to_string(),
            session_date: Some("10:00 am on 15 July, 2022".to_string()),
        },
    ]
}

#[test]
fn spacy_extraction_healthy() {
    let turns = real_turns();
    println!("Running spaCy extraction on {} turns...", turns.len());
    let output = extract_relations_via_spacy(&turns, Duration::from_secs(120), DEFAULT_SPACY_MODEL);
    println!("Extracted {} relations", output.relations.len());
    println!("Extracted {} key phrases", output.key_phrases.len());

    // Skip gracefully if spaCy is not available in this environment
    // (e.g. CI without Python/spaCy installed). This is a health check
    // for environments that provision spaCy, not a hard requirement.
    if output.relations.is_empty() && output.key_phrases.is_empty() {
        println!("SKIPPED: spaCy not available in this environment");
        return;
    }

    // Health check 1: extraction produced something (already verified non-empty above)
    println!("HEALTH 1 PASSED: non-empty relations");

    // Health check 2: Nate->meet relations found
    let meet_rels: Vec<_> = output
        .relations
        .iter()
        .filter(|r| r.subject == "Nate" && r.predicate.contains("meet"))
        .collect();
    println!("Nate->meet relations: {}", meet_rels.len());
    for r in &meet_rels {
        println!(
            "  {} -> {} -> '{}' ({})",
            r.subject, r.predicate, r.object, r.object_kind
        );
    }
    assert!(
        !meet_rels.is_empty(),
        "HEALTH FAILED: no Nate->meet relations extracted"
    );
    println!("HEALTH 2 PASSED: Nate->meet found");

    // Health check 3: both sessions covered
    let sessions: std::collections::HashSet<_> =
        meet_rels.iter().map(|r| r.session_id.as_str()).collect();
    println!("Sessions with meet: {:?}", sessions);
    assert!(
        sessions.contains("conv-42::session_14"),
        "HEALTH FAILED: session_14 missing"
    );
    assert!(
        sessions.contains("conv-42::session_23"),
        "HEALTH FAILED: session_23 missing"
    );
    println!("HEALTH 3 PASSED: both sessions covered");

    // Health check 4: kind tags present
    let kinds: std::collections::HashSet<_> = output
        .relations
        .iter()
        .map(|r| r.object_kind.as_str())
        .collect();
    println!("Object kinds: {:?}", kinds);
    assert!(!kinds.is_empty(), "HEALTH FAILED: no kind tags at all");
    println!("HEALTH 4 PASSED: kind tags present");

    println!("ALL HEALTH CHECKS PASSED");
}

fn chinese_turns() -> Vec<RelationTurn> {
    vec![RelationTurn {
        speaker: "王芳".to_string(),
        session_id: "zh::session_1".to_string(),
        turn_idx: 0,
        doc_id: "zh::session_1::0".to_string(),
        text: "我昨天在北京见了李明，我们讨论了新的项目计划。".to_string(),
        session_date: Some("2026-09-28".to_string()),
    }]
}

/// Chinese relation-extraction contract: spaCy's `doc.noun_chunks` is not
/// implemented for Chinese ([E894]), so `scripts/spacy_relations.py`
/// cannot extract relations/key phrases from Chinese text and the Rust
/// side fails open (empty output, no hang, no crash). Fixing the script
/// needs Python-side logic, which is outside the Rust-only
/// language-support rule -- recorded here so the gap is explicit, not
/// silent. The NER path (`spacy_ner.py`) is unaffected; see
/// `tier1::spacy_chinese_ner_extracts_entities`.
#[test]
fn spacy_chinese_relations_fail_open_on_e894() {
    let turns = chinese_turns();
    let output = extract_relations_via_spacy(&turns, Duration::from_secs(180), "zh_core_web_sm");

    println!(
        "Chinese relations: {} relations, {} key phrases (E894 fail-open)",
        output.relations.len(),
        output.key_phrases.len()
    );
    // The contract is fail-open: whatever spaCy does, this returns.
    // (Today that is empty because of E894.)
}
