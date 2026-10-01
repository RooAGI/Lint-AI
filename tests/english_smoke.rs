// English regression check: mirrors the Spanish smoke test to confirm
// English add -> search behavior is unchanged by the Spanish work.
use lint_ai::lang::Lang;
use lint_ai::memory_api::{AddRequest, MemoryService, Message};
use lint_ai::PipelineOptions;
use std::collections::BTreeMap;

fn options_for(lang: Lang) -> PipelineOptions {
    let mut opts = PipelineOptions::default();
    opts.lang = lang;
    opts.ner_provider = lint_ai::pipeline::Tier1NerProvider::Heuristic;
    opts
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
            session_id: "en-regress".to_string(),
        })
        .expect("add fact");
}

#[test]
fn english_add_search_regression() {
    let user_id = "en-regress-user";
    let mut opts = options_for(Lang::En);
    let mut service = MemoryService::in_memory(opts);
    add_fact(
        &mut service,
        user_id,
        "req-book",
        "The user bought a book about the history of Madrid yesterday.",
    );
    add_fact(
        &mut service,
        user_id,
        "req-library",
        "The Madrid library is on Alcala street.",
    );

    let id_book = lint_ai::stable_doc_id_from_source(&format!("{user_id}:req-book:0"));
    let id_library = lint_ai::stable_doc_id_from_source(&format!("{user_id}:req-library:0"));

    let mut filters = BTreeMap::new();
    filters.insert("memory_user_id".to_string(), user_id.to_string());

    // "Where is the Madrid library?" must top-hit the library fact.
    let r = service
        .search_with_filters("Where is the Madrid library?", user_id, None, 10, &filters)
        .expect("search");
    let top = r.first().expect("no results");
    eprintln!("en top: {} score={:.2}", top.doc_id, top.score);
    assert_eq!(top.doc_id, id_library, "English where-question regressed");

    // "What did the user buy yesterday?" must retrieve the book fact.
    let r = service
        .search_with_filters("What did the user buy yesterday?", user_id, None, 10, &filters)
        .expect("search");
    assert!(
        r.iter().any(|hit| hit.doc_id == id_book),
        "English what-yesterday question missed the book fact"
    );

    // Spanish-only words must not leak: they stay live in English
    // (not stopwords). (Words like "no"/"son"/"era" are also English
    // stopwords, so they can't test the gate — use Spanish-only forms.)
    for w in ["también", "dónde", "está"] {
        assert!(
            !lint_ai::tokenizer::is_stopword(w, lint_ai::tokenizer::TokenizerMode::Unstemmed),
            "{w} must not be an English stopword"
        );
    }
}
