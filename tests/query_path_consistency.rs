use lint_ai::query_plan::PreparedQuery;
use lint_ai::{IndexStore, PipelineOptions, SourceDocument};
use std::collections::BTreeMap;

fn document(id: &str, content: &str, timestamp: &str, group: &str) -> SourceDocument {
    SourceDocument {
        doc_id: id.to_string(),
        source: format!("memory://{id}"),
        content: content.to_string(),
        concept: "deployment memory".to_string(),
        group_id: Some(group.to_string()),
        headings: vec!["Deployment".to_string()],
        links: vec![],
        timestamp: Some(timestamp.to_string()),
        doc_length: content.len(),
        author_agent: None,
        filters: BTreeMap::new(),
    }
}

fn ids(results: &[lint_ai::SearchResult]) -> Vec<&str> {
    results
        .iter()
        .map(|result| result.doc_id.as_str())
        .collect()
}

#[test]
fn index_store_convenience_queries_share_the_prepared_path() {
    let mut store = IndexStore::in_memory(PipelineOptions::default());
    store.upsert(document(
        "alpha",
        "The deployment database is Postgres and the service runs in production.",
        "2026-08-01",
        "service-a",
    ));
    store.upsert(document(
        "beta",
        "The deployment cache is Redis and the service runs in staging.",
        "2026-08-02",
        "service-b",
    ));

    let query = "What database did I pick for the deployment?";
    let filters = BTreeMap::new();
    let prepared = PreparedQuery::new(query);

    let canonical = store.query_prepared(&prepared, 5, &filters).unwrap();
    let plain = store.query(query, 5).unwrap();
    let filtered = store.query_filtered(query, 5, &filters).unwrap();
    let timed = store.query_timed(query, 5).unwrap().0;
    let multi = store.query_filtered_multi(&[query], 5, &filters).unwrap();
    let cached = store.query_prepared_cached(&prepared, 5, &filters).unwrap();

    assert_eq!(ids(&plain), ids(&canonical));
    assert_eq!(ids(&filtered), ids(&canonical));
    assert_eq!(ids(&timed), ids(&canonical));
    assert_eq!(ids(&multi[0]), ids(&canonical));
    assert_eq!(ids(&cached), ids(&canonical));
}

#[test]
fn cli_uses_the_canonical_semantic_executor() {
    let engine = include_str!("../src/engine.rs");
    assert!(engine.contains("execute_on_index_with_semantics"));
    assert!(!engine.contains("semantic_relations::is_historical_query(query)"));
    assert!(!engine.contains("crate::semantic_relations::SemanticStatus::Superseded"));
}
