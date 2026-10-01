// Chinese add -> search smoke test, plus an English regression check.
//
// Adds Chinese memories (one target, two distractors) and searches with a
// partial-overlap query -- not the exact sentence -- so the test proves
// bigram retrieval works rather than exact matching. The English test
// guards against regressions from the CJK tokenizer changes.
use lint_ai::lang::Lang;
use lint_ai::memory_api::{AddRequest, MemoryService, Message, SearchRequest};
use lint_ai::PipelineOptions;
use std::collections::BTreeMap;

fn add_memory(service: &mut MemoryService, user_id: &str, request_id: &str, content: &str) {
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
            session_id: "smoke-session".to_string(),
        })
        .expect("add memory");
}

fn doc_id(user_id: &str, request_id: &str) -> String {
    // Mirrors MemoryService's internal doc_id_for: "{user}:{request}:0".
    lint_ai::stable_doc_id_from_source(&format!("{user_id}:{request_id}:0"))
}

fn search(
    service: &mut MemoryService,
    query: &str,
    user_id: &str,
    lang: Lang,
) -> Vec<lint_ai::memory_api::SearchMemory> {
    let response = service
        .search(SearchRequest {
            query: query.to_string(),
            options: None,
            user_id: user_id.to_string(),
            top_k: 10,
            session_id: None,
            scope: None,
            filters: None,
            lang: Some(lang),
        })
        .expect("search");
    response.data
}

#[test]
fn chinese_add_search_ranks_target_first() {
    let options = PipelineOptions {
        lang: Lang::Zh,
        ..PipelineOptions::default()
    };
    let mut service = MemoryService::in_memory(options);
    let user_id = "zh-smoke-user";
    add_memory(
        &mut service,
        user_id,
        "zh-target",
        "我毕业于清华大学，专业是计算机科学。",
    );
    add_memory(&mut service, user_id, "zh-d1", "我喜欢在周末去公园跑步。");
    add_memory(&mut service, user_id, "zh-d2", "北京的冬天非常寒冷。");

    // Partial overlap: shares bigrams (清华, 华大, 大学, 计算...) with the
    // target but is not the stored sentence.
    let results = search(&mut service, "清华大学计算机专业", user_id, Lang::Zh);
    assert!(
        !results.is_empty(),
        "Chinese query should retrieve the target memory"
    );
    assert_eq!(
        results[0].id,
        doc_id(user_id, "zh-target"),
        "target should rank first, got {:?}",
        results.iter().map(|r| &r.id).collect::<Vec<_>>()
    );
}

#[test]
fn chinese_question_retrieves_answer_memory() {
    let options = PipelineOptions {
        lang: Lang::Zh,
        ..PipelineOptions::default()
    };
    let mut service = MemoryService::in_memory(options);
    let user_id = "zh-smoke-user-q";
    add_memory(
        &mut service,
        user_id,
        "zh-target",
        "我毕业于清华大学，专业是计算机科学。",
    );
    add_memory(&mut service, user_id, "zh-d1", "我喜欢在周末去公园跑步。");

    // Interrogative + partial content overlap.
    let results = search(&mut service, "我在哪所大学毕业的？", user_id, Lang::Zh);
    assert!(
        !results.is_empty(),
        "Chinese question should retrieve the target memory"
    );
    assert_eq!(
        results[0].id,
        doc_id(user_id, "zh-target"),
        "target should rank first, got {:?}",
        results.iter().map(|r| &r.id).collect::<Vec<_>>()
    );
}

#[test]
fn english_add_search_still_ranks_target_first() {
    let options = PipelineOptions {
        lang: Lang::En,
        ..PipelineOptions::default()
    };
    let mut service = MemoryService::in_memory(options);
    let user_id = "en-smoke-user";
    add_memory(
        &mut service,
        user_id,
        "en-target",
        "I graduated from Stanford University with a degree in computer science.",
    );
    add_memory(
        &mut service,
        user_id,
        "en-d1",
        "I enjoy running in the park on weekends.",
    );
    add_memory(
        &mut service,
        user_id,
        "en-d2",
        "Winters in Boston are very cold.",
    );

    let results = search(
        &mut service,
        "Stanford computer science degree",
        user_id,
        Lang::En,
    );
    assert!(
        !results.is_empty(),
        "English query should retrieve the target memory"
    );
    assert_eq!(
        results[0].id,
        doc_id(user_id, "en-target"),
        "target should rank first, got {:?}",
        results.iter().map(|r| &r.id).collect::<Vec<_>>()
    );
}

#[test]
fn auto_detect_routes_chinese_query_to_chinese_memory() {
    // Lang::Auto on both the pipeline and the request: script detection
    // must pick Chinese without any explicit flag.
    let mut service = MemoryService::in_memory(PipelineOptions::default());
    let user_id = "zh-smoke-user-auto";
    add_memory(
        &mut service,
        user_id,
        "zh-target",
        "我毕业于清华大学，专业是计算机科学。",
    );
    add_memory(&mut service, user_id, "zh-d1", "我喜欢在周末去公园跑步。");

    let response = service
        .search(SearchRequest {
            query: "清华大学计算机专业".to_string(),
            options: None,
            user_id: user_id.to_string(),
            top_k: 10,
            session_id: None,
            scope: None,
            filters: Some({
                let mut f = BTreeMap::new();
                f.insert("memory_user_id".to_string(), user_id.to_string());
                f
            }),
            lang: None,
        })
        .expect("search");
    assert!(
        !response.data.is_empty(),
        "auto-detected Chinese query should retrieve"
    );
    assert_eq!(
        response.data[0].id,
        doc_id(user_id, "zh-target"),
        "target should rank first under auto-detect, got {:?}",
        response.data.iter().map(|r| &r.id).collect::<Vec<_>>()
    );
}
