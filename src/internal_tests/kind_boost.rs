// End-to-end: definitional kind tags (bekind kind verdicts).
//
// Luyi 2026-09-28: definitional knowledge is a MATCH, not a bonus. An
// admitted closed-set kind ("herb") becomes a SHOULD TermQuery on the
// index's `semantic_tags` field, scored by plain BM25 inside tantivy.
// Luyi 2026-09-29: no fixed tag multiplier — the old TAG_BOOST = 29.0 was
// calibrated only on the toy set and never validated retrieval-wide.
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
// the question). The herb tag is a vocabulary bridge (herb <-> cilantro)
// scored by BM25 — it contributes to A's score but is NOT expected to flip
// the pair on its own.
//
// Requires a kind-capable bekind binary: set BEHOOD_BIN to it
// (e.g. ~/workspace/bekind/target/release/bekind at commit e89c5cf).
// Without BEHOOD_BIN the test skips loudly instead of failing.
use crate::memory_api::{AddRequest, MemoryService, Message};
use crate::semantic_tags::{batch_doc_semantic_tags, query_semantic_tags};
use crate::PipelineOptions;
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
            session_id: "kind-session".to_string(),
        })
        .expect("add fact");
}

#[test]
fn herb_tag_bridges_query_to_cilantro_fact_without_flipping_rank() {
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

    // The vocabulary bridge itself: the question emits the herb tag, the
    // cilantro fact carries it, the coffee fact does not.
    let query_tags = query_semantic_tags(QUESTION);
    assert!(
        query_tags.contains(&"herb".to_string()),
        "question must emit the herb query tag, got {query_tags:?}"
    );
    let index_tags = batch_doc_semantic_tags(&[FACT_A, FACT_B]);
    assert!(
        index_tags[0].contains(&"herb".to_string()),
        "cilantro fact must carry the herb index tag, got {:?}",
        index_tags[0]
    );
    assert!(
        !index_tags[1].contains(&"herb".to_string()),
        "coffee fact must not carry the herb tag, got {:?}",
        index_tags[1]
    );

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
        a.score, a.score_breakdown.lexical_score, b.score, b.score_breakdown.lexical_score,
    );

    // Tags never filter: both facts are retrieved. The herb tag contributes
    // to A's score as a plain BM25 SHOULD clause (A's score is positive and
    // includes the tag match), but with no fixed multiplier it does not
    // flip this lexically lopsided pair — and must not be expected to.
    assert!(
        a.score > 0.0,
        "herb-tagged fact must score positively via the tag bridge"
    );
    assert!(
        pos_b < pos_a,
        "without the tag multiplier the lexically stronger coffee fact ranks first"
    );
}
