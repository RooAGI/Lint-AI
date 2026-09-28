// End-to-end: temporal-scope rank boost (bekind scope verdicts).
//
// The NeMo pair from the 2026-09-28 build:
//   question: "What is the user's weekend exercise routine?"
//   fact A:   "The user runs 5 kilometers every Saturday morning in Golden Gate Park."
//             -> temporal_words ["weekend"], habitual true  -> MATCH -> +25.0
//   fact B:   "The user swims every Monday evening at the pool."
//             -> temporal_words ["weekday"], habitual true  -> no match
//
// Requires a scope-capable bekind binary: set BEHOOD_BIN to it
// (e.g. ~/workspace/bekind/target/release/bekind at commit 600ca87).
// Without BEHOOD_BIN the test skips loudly instead of failing — the pure
// match/boost logic is covered by unit tests that need no binary.
use lint_ai::memory_api::{AddRequest, MemoryService, Message};
use lint_ai::PipelineOptions;
use std::collections::BTreeMap;

const FACT_A: &str =
    "The user runs 5 kilometers every Saturday morning in Golden Gate Park.";
const FACT_B: &str = "The user swims every Monday evening at the pool.";
const QUESTION: &str = "What is the user's weekend exercise routine?";
const EXPECTED_BOOST: f32 = 25.0;

fn bekind_bin() -> Option<String> {
    std::env::var("BEHOOD_BIN")
        .ok()
        .filter(|s| !s.trim().is_empty())
}

fn doc_id(user_id: &str, request_id: &str) -> String {
    // Mirrors MemoryService's internal doc_id_for: "{user}:{request}:0".
    lint_ai::stable_doc_id_from_source(&format!("{user_id}:{request_id}:0"))
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
            session_id: "scope-session".to_string(),
        })
        .expect("add fact");
}

#[test]
fn weekend_question_boosts_weekend_fact_above_weekday_fact() {
    let Some(bin) = bekind_bin() else {
        eprintln!(
            "SKIP scope_boost integration: BEHOOD_BIN not set \
             (needs a scope-capable bekind binary)"
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

    let user_id = "scope-user";
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
        "scope_boost: A score={:.2} (lexical {:.2} + boost {:.2}) | \
         B score={:.2} (boost {:.2})",
        a.score,
        a.score - a.score_breakdown.scope_boost,
        a.score_breakdown.scope_boost,
        b.score,
        b.score_breakdown.scope_boost,
    );

    assert_eq!(
        a.score_breakdown.scope_boost, EXPECTED_BOOST,
        "fact A (weekend+habitual) must get exactly the fixed boost"
    );
    assert_eq!(
        b.score_breakdown.scope_boost, 0.0,
        "fact B (weekday) must be untouched: scope mismatch withholds \
         the boost, never penalizes"
    );
    assert!(
        pos_a < pos_b,
        "weekend fact must rank above the weekday fact"
    );
    // Decisiveness: without the boost, A would not be strictly ahead of B —
    // i.e. the reported delta is what moved the rank.
    let a_lex = a.score - a.score_breakdown.scope_boost;
    let b_lex = b.score - b.score_breakdown.scope_boost;
    assert!(
        a_lex <= b_lex,
        "boost must be what puts A ahead (A_lex={a_lex:.2} B_lex={b_lex:.2})"
    );
}
