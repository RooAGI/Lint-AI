// End-to-end: definitional kind tags (bekind kind verdicts).
//
// Luyi 2026-09-28: definitional knowledge is a MATCH, not a bonus. An
// admitted closed-set kind ("herb") becomes a SHOULD TermQuery on the
// index's `semantic_tags` field, scored by BM25 inside tantivy. There is
// no additive boost and no score_breakdown.kind_boost field.
//
// The NeMo pair mem-08:
//   question: "Is there an herb the user avoids?"
//             -> query tags ["herb"] (entity analyzer judges "herb" herb-kind)
//   fact A:   "The user dislikes cilantro and always asks for it to be left out."
//             -> index tags ["herb"] (cilantro judged herb-kind; the spaCy
//                ADV mis-tag is recovered via the nominal-dependency slot)
//   fact B:   "The user avoids coffee and always asks for decaf."
//             -> index tags [] (coffee is food-kind, not admitted)
//
// Fact B is lexically strong on purpose ("avoids"/"always asks" overlap
// the question); the herb tag must close the gap.
//
// Requires a kind-capable bekind binary: set BEHOOD_BIN to it
// (e.g. ~/workspace/bekind/target/release/bekind at commit e89c5cf).
// Without BEHOOD_BIN the test skips loudly instead of failing.
use lint_ai::memory_api::{AddRequest, MemoryService, Message};
use lint_ai::PipelineOptions;
use std::collections::BTreeMap;

const FACT_A: &str = "The user dislikes cilantro and always asks for it to be left out.";
const FACT_B: &str = "The user avoids coffee and always asks for decaf.";
const QUESTION: &str = "Is there an herb the user avoids?";

fn bekind_bin() -> Option<String> {
    std::env::var("BEHOOD_BIN")
        .ok()
        .filter(|s| !s.trim().is_empty())
}

fn doc_id(user_id: &str, request_id: &str) -> String {
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
            session_id: "kind-session".to_string(),
        })
        .expect("add fact");
}

#[test]
fn herb_question_ranks_cilantro_fact_above_coffee_fact_via_tags() {
    let Some(bin) = bekind_bin() else {
        eprintln!(
            "SKIP kind_tags integration: BEHOOD_BIN not set \
             (needs a kind-capable bekind binary)"
        );
        return;
    };
    assert!(
        std::path::Path::new(&bin).exists(),
        "BEHOOD_BIN={bin} does not exist"
    );
    std::env::set_var("BEHOOD_BIN", &bin);

    let user_id = "kind-user";
    let mut service = MemoryService::in_memory(PipelineOptions::default());
    add_fact(&mut service, user_id, "req-a", FACT_A);
    add_fact(&mut service, user_id, "req-b", FACT_B);

    let mut filters = BTreeMap::new();
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
    // No kind_boost field exists: the tag scores inside tantivy BM25.
    eprintln!(
        "kind_tags: A score={:.2} (lexical {:.2}) | B score={:.2} (lexical {:.2})",
        a.score,
        a.score_breakdown.lexical_score,
        b.score,
        b.score_breakdown.lexical_score,
    );

    assert!(
        pos_a < pos_b,
        "cilantro (herb-tagged) fact must rank above the coffee fact"
    );
}
