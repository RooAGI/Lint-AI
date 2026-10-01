// End-to-end: definitional temporal tags (bekind scope verdicts).
//
// Luyi 2026-09-28: definitional knowledge is a MATCH, not a bonus. The
// question's closed-set temporal words and habituality become SHOULD
// TermQueries on the index's `semantic_tags` field, scored by BM25 inside
// tantivy. There is no additive boost and score_breakdown.scope_boost stays
// 0.0.
//
// The NeMo pair mem-05:
//   question: "What is the user's weekend exercise routine?"
//             -> query tags ["habitual", "weekend"]
//   fact A:   "The user runs 5 kilometers every Saturday morning in Golden Gate Park."
//             -> index tags ["habitual", "weekend"] (Saturday -> weekend)
//   fact B:   "The user swims every Monday evening at the pool."
//             -> index tags ["habitual", "weekday"]  (Monday -> weekday)
//
// Requires a scope-capable bekind binary: set BEHOOD_BIN to it
// (e.g. ~/workspace/bekind/target/release/bekind at commit e89c5cf).
// Without BEHOOD_BIN the test skips loudly instead of failing — the pure
// tag-emission logic is covered by unit tests that need no binary.
use lint_ai::memory_api::{AddRequest, MemoryService, Message};
use lint_ai::PipelineOptions;
use std::collections::BTreeMap;

const FACT_A: &str =
    "The user runs 5 kilometers every Saturday morning in Golden Gate Park.";
const FACT_B: &str = "The user swims every Monday evening at the pool.";
const QUESTION: &str = "What is the user's weekend exercise routine?";

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

fn search(service: &mut MemoryService, user_id: &str, question: &str) -> Vec<lint_ai::SearchResult> {
    let mut filters = BTreeMap::new();
    // Literal key, matching internal convention (cf. USER_FILTER and the
    // locomo benchmark server): the ownership filter const is crate-private.
    filters.insert("memory_user_id".to_string(), user_id.to_string());
    service
        .search_with_filters(question, user_id, None, 10, &filters)
        .expect("search")
}

#[test]
fn weekend_question_ranks_saturday_fact_above_monday_fact_via_tags() {
    let Some(bin) = bekind_bin() else {
        eprintln!(
            "SKIP semantic_tags integration: BEHOOD_BIN not set \
             (needs a scope-capable bekind binary)"
        );
        return;
    };
    assert!(
        std::path::Path::new(&bin).exists(),
        "BEHOOD_BIN={bin} does not exist"
    );
    // The behood daemon resolves BEHOOD_BIN when it spawns; set it before
    // any analyze call in this test process (index-time tagging runs at
    // snapshot build).
    std::env::set_var("BEHOOD_BIN", &bin);

    let user_id = "scope-user";
    let mut service = MemoryService::in_memory(PipelineOptions::default());
    add_fact(&mut service, user_id, "req-a", FACT_A);
    add_fact(&mut service, user_id, "req-b", FACT_B);

    let results = search(&mut service, user_id, QUESTION);

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
    // There is no score_breakdown.scope_boost field anymore: definitional
    // tags are scored inside tantivy BM25, so a post-hoc bonus has no field
    // to live in.
    eprintln!(
        "semantic_tags: A score={:.2} (lexical {:.2}) | B score={:.2} (lexical {:.2})",
        a.score,
        a.score_breakdown.lexical_score,
        b.score,
        b.score_breakdown.lexical_score,
    );

    assert!(
        pos_a < pos_b,
        "Saturday (weekend-tagged) fact must rank above the Monday (weekday-tagged) fact"
    );
}

#[test]
fn non_temporal_question_is_unaffected_by_tags() {
    let Some(bin) = bekind_bin() else {
        eprintln!("SKIP semantic_tags integration: BEHOOD_BIN not set");
        return;
    };
    assert!(
        std::path::Path::new(&bin).exists(),
        "BEHOOD_BIN={bin} does not exist"
    );
    std::env::set_var("BEHOOD_BIN", &bin);

    let user_id = "scope-user-plain";
    let mut service = MemoryService::in_memory(PipelineOptions::default());
    add_fact(&mut service, user_id, "req-a", FACT_A);
    add_fact(&mut service, user_id, "req-b", FACT_B);

    // No temporal words, no habituality: no tags emitted, so the tag
    // machinery must not move the ranking versus pure lexical. (And no
    // boost field exists for a bonus to hide in: definitional tags score
    // inside tantivy BM25 only.)
    let results = search(&mut service, user_id, "What exercise does the user do?");

    let id_a = doc_id(user_id, "req-a");
    let id_b = doc_id(user_id, "req-b");
    assert!(
        results.iter().any(|r| r.doc_id == id_a),
        "fact A still retrieved without tags"
    );
    assert!(
        results.iter().any(|r| r.doc_id == id_b),
        "fact B still retrieved without tags"
    );
}
