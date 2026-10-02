// End-to-end: activity↔venue rank boost (lint-ai activity→venue table
// over bekind's activity phrase + place-kind verdicts).
//
// The NeMo pair mem-14 from the 2026-09-28 build:
//   question: "Where does the user like to eat out?"
//             -> activity_phrase "eat out" (bekind, linguistic)
//   fact A:   "The user's favorite restaurant is Din Tai Fung in San Jose."
//             -> "restaurant" is kind=place (bekind, linguistic)
//             -> "eat out" maps to {restaurant, ...} (lint-ai table,
//                world knowledge) -> MATCH -> +25.0
//   fact B:   "The user likes to eat out with friends on weekends."
//             -> names no eat-out venue -> no match. (Lexically stronger
//                than A on purpose — it shares "likes"/"eat"/"out" with
//                the question — so the boost is decisive; the scope boost
//                does not fire here — the question names no temporal
//                scope.)
//
// Requires an activity-phrase-capable bekind binary: set BEHOOD_BIN to it
// (e.g. ~/workspace/bekind/target/release/bekind at commit e89c5cf).
// Without BEHOOD_BIN the test skips loudly instead of failing — the pure
// match/boost logic is covered by unit tests that need no binary.
use crate::memory_api::{AddRequest, MemoryService, Message};
use crate::PipelineOptions;
use std::collections::BTreeMap;

const FACT_A: &str = "The user's favorite restaurant is Din Tai Fung in San Jose.";
// Lexically competitive on purpose (shares "user"/"likes" with the
// question) but names no eat-out venue, so the boost is decisive.
const FACT_B: &str = "The user likes going out with friends on weekends.";
const QUESTION: &str = "Where does the user like to eat out?";
const EXPECTED_BOOST: f32 = 25.0;

fn bekind_bin() -> Option<String> {
    std::env::var("BEHOOD_BIN")
        .ok()
        .filter(|s| !s.trim().is_empty())
}

fn doc_id(user_id: &str, request_id: &str) -> String {
    // Mirrors MemoryService's internal doc_id_for: "{user}:{request}:0".
    crate::memory_api::memory_document_id(user_id, request_id, 0)
}

fn add_fact(service: &mut MemoryService, user_id: &str, request_id: &str, content: &str) {
    service
        .add(AddRequest {
            request_id: request_id.to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                timestamp: None,
                content: content.to_string(),
                expires_at_ms: None,
                supersedes_id: None,
            }],
            user_id: user_id.to_string(),
            session_id: "venue-session".to_string(),
        })
        .expect("add fact");
}

#[test]
fn eat_out_question_boosts_restaurant_fact_above_park_fact() {
    let Some(bin) = bekind_bin() else {
        eprintln!(
            "SKIP activity_venue_boost integration: BEHOOD_BIN not set \
             (needs an activity-phrase-capable bekind binary)"
        );
        return;
    };
    assert!(
        std::path::Path::new(&bin).exists(),
        "BEHOOD_BIN={bin} does not exist"
    );
    // The behood daemon resolves BEHOOD_BIN when it spawns; set it before
    // any analyze call in this test process.
    std::env::set_var("BEHOOD_BIN", &bin);

    let user_id = "venue-user";
    let mut service = MemoryService::in_memory(PipelineOptions::default());
    add_fact(&mut service, user_id, "req-a", FACT_A);
    add_fact(&mut service, user_id, "req-b", FACT_B);

    let mut filters = BTreeMap::new();
    // Literal key, matching internal convention (cf. USER_FILTER and the
    // locomo benchmark server): the ownership filter const is crate-private.
    filters.insert("memory_user_id".to_string(), user_id.to_string());
    let results = service
        .search_with_filters(QUESTION, user_id, None, 10, &filters)
        .expect("search");

    let id_a = doc_id(user_id, "req-a");
    let id_b = doc_id(user_id, "req-b");
    let pos_a = results
        .iter()
        .position(|r| r.doc_id == id_a)
        .expect("fact A retrieved");
    let pos_b = results
        .iter()
        .position(|r| r.doc_id == id_b)
        .expect("fact B retrieved");

    let a = &results[pos_a];
    let b = &results[pos_b];
    eprintln!(
        "activity_venue_boost: A score={:.2} (lexical {:.2} + boost {:.2}) | \
         B score={:.2} (boost {:.2})",
        a.score,
        a.score - a.score_breakdown.activity_venue_boost,
        a.score_breakdown.activity_venue_boost,
        b.score,
        b.score_breakdown.activity_venue_boost,
    );

    assert_eq!(
        a.score_breakdown.activity_venue_boost, EXPECTED_BOOST,
        "fact A (restaurant is an eat-out venue) must get exactly the fixed boost"
    );
    assert_eq!(
        b.score_breakdown.activity_venue_boost, 0.0,
        "fact B (no eat-out venue named) must be untouched: \
         venue mismatch withholds the boost, never penalizes"
    );
    // No semantic tags fire on this pair either: the question names no
    // temporal scope and seeks no admitted kind. (scope/kind are index-time
    // tag matches now, not boost fields.)
    assert!(
        pos_a < pos_b,
        "restaurant fact must rank above the venue-less fact"
    );
    // Decisiveness: without the boost, A would not be strictly ahead of B —
    // i.e. the reported delta is what moved the rank.
    let a_lex = a.score - a.score_breakdown.activity_venue_boost;
    let b_lex = b.score - b.score_breakdown.activity_venue_boost;
    assert!(
        a_lex <= b_lex,
        "boost must be what puts A ahead (A_lex={a_lex:.2} B_lex={b_lex:.2})"
    );
}
