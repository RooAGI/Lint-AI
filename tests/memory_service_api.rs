//! Exercise the supported API from a separate crate, with no internal imports.
use lint_ai::{
    AddRequest, GetRequest, ListRequest, MemoryService, Message, PipelineOptions, SearchRequest,
    UpdateRequest,
};

#[test]
fn service_persists_searches_updates_and_deletes_without_store_access() -> anyhow::Result<()> {
    let root = std::fs::canonicalize(std::env::temp_dir())?.join(format!(
        "lint-ai-public-service-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    let mut service = MemoryService::at_path(&root, PipelineOptions::default())?;
    service.add(AddRequest {
        request_id: "first".into(),
        user_id: "alice".into(),
        session_id: "session".into(),
        messages: vec![Message {
            role: "user".into(),
            content: "The quartz deployment uses Postgres.".into(),
            timestamp: None,
            expires_at_ms: None,
            supersedes_id: None,
        }],
    })?;
    let list = ListRequest {
        user_id: "alice".into(),
        session_id: None,
        limit: 10,
        cursor: None,
        include_inactive: false,
    };
    let records = service.list(list.clone())?;
    assert_eq!(records.data.len(), 1);
    let id = records.data[0].id.clone();
    drop(service);

    let mut service = MemoryService::at_path(&root, PipelineOptions::default())?;
    let hits = service.search(SearchRequest {
        query: "quartz Postgres".into(),
        options: None,
        user_id: "alice".into(),
        top_k: 5,
        session_id: None,
        scope: None,
        filters: None,
    })?;
    assert!(hits.data.iter().any(|hit| hit.id == id));
    let updated = service.update(UpdateRequest {
        user_id: "alice".into(),
        memory_id: id.clone(),
        content: "The quartz deployment uses SQLite.".into(),
        role: None,
        timestamp: None,
        expires_at_ms: None,
    })?;
    assert!(updated.is_some());
    drop(service);

    let mut service = MemoryService::at_path(&root, PipelineOptions::default())?;
    let record = service
        .get(GetRequest {
            user_id: "alice".into(),
            memory_id: id.clone(),
            include_inactive: false,
        })?
        .expect("updated memory survives reopening");
    assert!(record.content.contains("SQLite"));
    assert!(service.delete("alice", &id)?);
    drop(service);
    let service = MemoryService::at_path(&root, PipelineOptions::default())?;
    assert!(service.list(list)?.data.is_empty());
    drop(service);
    std::fs::remove_dir_all(root)?;
    Ok(())
}
