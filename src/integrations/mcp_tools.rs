use crate::index::SearchResult;
use crate::integrations::mcp_transport::ToolDefinition;
use crate::memory_api::MemoryService;
use crate::source::SourceDocument;
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// Canonical provider values for `filters.provider`, the per-document
/// attribution stamped on every captured memory.
pub(crate) const PROVIDER_FILTER_VALUES: &[&str] =
    &["claude", "codex", "gemini-cli", "agy", "muse"];

/// JSON Schema fragment for the optional `provider` search argument.
pub(crate) fn provider_argument_schema() -> Value {
    json!({
        "type": "string",
        "description": "Restrict results to memories captured by one provider (claude, codex, gemini-cli, agy, muse). Omit to search the shared pool.",
        "enum": PROVIDER_FILTER_VALUES,
    })
}

/// Build the document filters for an MCP search from the optional `provider`
/// argument. An empty map searches the whole shared pool; an unknown provider
/// is a caller error, not an empty result.
pub(crate) fn search_provider_filters(
    arguments: &Value,
) -> Result<BTreeMap<String, String>, String> {
    let provider = arguments
        .get("provider")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|provider| !provider.is_empty());
    let Some(provider) = provider else {
        return Ok(BTreeMap::new());
    };
    if !PROVIDER_FILTER_VALUES.contains(&provider) {
        return Err(format!(
            "unknown provider \"{provider}\"; expected one of: {}",
            PROVIDER_FILTER_VALUES.join(", ")
        ));
    }
    Ok(BTreeMap::from([(
        "provider".to_string(),
        provider.to_string(),
    )]))
}

/// Extract the optional `session_id` search argument. A supplied value must be
/// a non-empty, non-blank identifier; omit the argument for stateless search.
/// A present-but-blank value, or a present non-string value, is rejected
/// rather than silently treated as absent, so callers cannot accidentally
/// lose session state. This is a conversation-state key only, never a corpus
/// filter.
pub(crate) fn search_session_id(arguments: &Value) -> Result<Option<String>, String> {
    let Some(value) = arguments.get("session_id") else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let Some(session_id) = value.as_str() else {
        return Err("session_id must be a string".to_string());
    };
    let session_id = session_id.trim();
    if session_id.is_empty() {
        return Err("session_id must not be blank".to_string());
    }
    if session_id.len() > 256 {
        return Err("session_id must be at most 256 bytes".to_string());
    }
    if session_id.chars().any(|character| character.is_control()) {
        return Err("session_id must not contain control characters".to_string());
    }
    Ok(Some(session_id.to_string()))
}

/// Resolve the effective session id for an MCP search.
///
/// An explicit `session_id` argument always wins. When the argument is
/// absent (or null), the search inherits the session most recently seen
/// active in this workspace — hooks keep that pointer current through the
/// service's conversation-state store, so an agent that never passes
/// `session_id` still gets follow-up resolution against its live
/// conversation. A present-but-blank argument is still rejected rather than
/// falling back, so callers cannot accidentally lose session state. The
/// resolved id (when any) refreshes the pointer, keeping it alive while the
/// conversation continues through MCP searches.
#[cfg(any(
    feature = "claude-code",
    feature = "codex",
    feature = "gemini-cli",
    feature = "agy",
    feature = "muse-code"
))]
pub(crate) fn resolve_search_session_id(
    arguments: &Value,
    service: &MemoryService,
    provider: &str,
) -> Result<Option<String>, String> {
    let session_id = match search_session_id(arguments)? {
        Some(explicit) => Some(explicit),
        None => service.current_session_id(provider),
    };
    if let Some(active) = session_id.as_deref() {
        service.note_active_session(provider, active);
    }
    Ok(session_id)
}

/// Format retrieval hits for an agent. Keep this separate from the internal
/// ranking representation: diagnostics and score components are useful while
/// tuning the index, but distract an agent from the memory itself.
pub fn search_results(service: &MemoryService, results: Vec<SearchResult>) -> Value {
    let results = results
        .into_iter()
        .filter_map(|result| {
            let document = service.source_document_by_id(&result.doc_id)?;
            let content: String = document.content.chars().take(4_000).collect();
            // Anchor relative date expressions ("yesterday", "last month") to
            // an absolute date the agent can see. Fail-open: memories without
            // a timestamp keep their original text.
            let content = match document.timestamp.as_deref() {
                Some(date) => format!("[session date: {date}]\n{content}"),
                None => content,
            };
            Some(json!({
                "id": result.doc_id,
                "source": result.source,
                "content": content,
                "score": result.score,
                "created_at": document.timestamp.clone(),
                "session_id": document.group_id.clone(),
                "matched_terms": result.matched_terms,
                "matched_entities": result.matched_entities,
                "semantic_status": result.semantic_status,
                "superseded_by": result.superseded_by,
                "relation_confidence": result.relation_confidence,
                "relation_evidence": result.relation_evidence,
            }))
        })
        .collect::<Vec<_>>();
    json!({"results": results})
}

/// Return a bounded, provider-neutral view of the indexed memories.
pub(crate) fn list_memories(service: &MemoryService, limit: usize) -> Value {
    let memories = service
        .source_documents()
        .into_iter()
        .filter(|document| is_recorded_memory(document))
        .take(limit.clamp(1, 100))
        .map(|document| {
            json!({
                "source": document.source,
                "document_type": document.concept,
                "group_id": document.group_id,
                "content": document.content.chars().take(2_000).collect::<String>(),
            })
        })
        .collect::<Vec<_>>();
    json!({"count": memories.len(), "memories": memories})
}

fn is_recorded_memory(document: &SourceDocument) -> bool {
    document.source.starts_with("claude://")
        || document.source.starts_with("codex://")
        || document.source.starts_with("gemini-cli://")
        || document.source.starts_with("agy://")
        || document.source.starts_with("muse://")
        || document.source.starts_with("lint-ai://")
        || document
            .filters
            .get("source_type")
            .is_some_and(|source_type| source_type == "recorded-session")
        || document.group_id.as_deref().is_some_and(|group_id| {
            [
                "claude-session:",
                "codex-session:",
                "gemini-cli-session:",
                "agy-session:",
                "muse-session:",
            ]
            .iter()
            .any(|prefix| group_id.starts_with(prefix))
        })
}

pub(crate) fn list_memories_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "list_memories".to_string(),
        description: "List indexed memories with bounded content previews.".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}
            },
            "additionalProperties": false
        }),
    }
}

// Codex validates arguments locally because its protocol already exposes the
// unknown argument name; Claude and Gemini use this shared parser instead.
#[cfg_attr(
    not(any(feature = "claude-code", feature = "gemini-cli")),
    allow(dead_code)
)]
pub(crate) fn parse_list_memories_limit(arguments: &Value) -> Result<usize, &'static str> {
    if arguments
        .as_object()
        .map(|object| object.keys().any(|key| key != "limit"))
        .unwrap_or(false)
    {
        return Err("unknown list_memories argument");
    }
    Ok(arguments
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(20)
        .clamp(1, 100) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{IndexStore, PipelineOptions};
    use crate::source::SourceDocument;
    use std::collections::BTreeMap;

    #[test]
    fn search_provider_filters_accepts_known_providers() {
        let filters = search_provider_filters(&json!({"query": "x", "provider": "codex"})).unwrap();
        assert_eq!(
            filters,
            BTreeMap::from([("provider".to_string(), "codex".to_string())])
        );
    }

    #[test]
    fn search_provider_filters_defaults_to_shared_pool() {
        assert!(search_provider_filters(&json!({"query": "x"}))
            .unwrap()
            .is_empty());
        assert!(
            search_provider_filters(&json!({"query": "x", "provider": ""}))
                .unwrap()
                .is_empty()
        );
        assert!(
            search_provider_filters(&json!({"query": "x", "provider": "  "}))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn search_provider_filters_rejects_unknown_provider() {
        let error =
            search_provider_filters(&json!({"query": "x", "provider": "clippy"})).unwrap_err();
        assert!(error.contains("unknown provider"), "{error}");
        assert!(error.contains("codex"), "{error}");
    }

    #[test]
    fn provider_filter_restricts_search_to_one_providers_memories() {
        use crate::query_plan::PreparedQuery;

        fn memory(doc_id: &str, provider: &str) -> SourceDocument {
            SourceDocument {
                doc_id: doc_id.to_string(),
                source: format!("{provider}://project/session/outcome"),
                content: "The deployment pipeline codename is cobalt".to_string(),
                concept: "outcome".to_string(),
                group_id: Some(format!("{provider}-session:session")),
                filters: BTreeMap::from([("provider".to_string(), provider.to_string())]),
                headings: vec![],
                links: vec![],
                timestamp: None,
                doc_length: 38,
                author_agent: Some(provider.to_string()),
                key_phrases: Vec::new(),
                key_phrase_extraction_hash: String::new(),
            }
        }

        let mut store = IndexStore::in_memory(PipelineOptions::default());
        store.upsert(memory("codex-memory", "codex"));
        store.upsert(memory("claude-memory", "claude"));
        store.refresh().unwrap();

        // Unfiltered search sees the shared pool.
        let unfiltered = store
            .query_prepared(
                &PreparedQuery::new("deployment pipeline codename"),
                10,
                &BTreeMap::new(),
            )
            .unwrap();
        assert_eq!(unfiltered.len(), 2);

        // A provider filter sees only that provider's memories.
        let filters = search_provider_filters(&json!({"query": "x", "provider": "codex"})).unwrap();
        let filtered = store
            .query_prepared(
                &PreparedQuery::new("deployment pipeline codename"),
                10,
                &filters,
            )
            .unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].doc_id, "codex-memory");
    }

    #[test]
    fn list_memories_is_bounded_and_provider_neutral() {
        let mut store = IndexStore::in_memory(PipelineOptions::default());
        store.upsert(SourceDocument {
            doc_id: "memory-1".to_string(),
            source: "codex://project/session-1/outcome".to_string(),
            content: "memory content".to_string(),
            concept: "decision".to_string(),
            group_id: Some("session-1".to_string()),
            filters: BTreeMap::new(),
            headings: vec![],
            links: vec![],
            timestamp: None,
            doc_length: 14,
            author_agent: None,
            key_phrases: Vec::new(),
            key_phrase_extraction_hash: String::new(),
        });
        store.upsert(SourceDocument {
            doc_id: "workspace-file".to_string(),
            source: "file:///workspace/README.md".to_string(),
            content: "workspace content".to_string(),
            concept: "source-file".to_string(),
            group_id: None,
            filters: BTreeMap::new(),
            headings: vec![],
            links: vec![],
            timestamp: None,
            doc_length: 17,
            author_agent: None,
            key_phrases: Vec::new(),
            key_phrase_extraction_hash: String::new(),
        });
        let service = MemoryService::new(store);
        let payload = list_memories(&service, 20);
        assert_eq!(payload["count"], 1);
        assert_eq!(
            payload["memories"][0]["source"],
            "codex://project/session-1/outcome"
        );
        assert_eq!(parse_list_memories_limit(&json!({"limit": 0})).unwrap(), 1);
        assert!(parse_list_memories_limit(&json!({"unexpected": true})).is_err());
    }

    #[test]
    fn search_results_anchor_memories_to_their_session_date() {
        fn memory(doc_id: &str, timestamp: Option<&str>) -> SourceDocument {
            SourceDocument {
                doc_id: doc_id.to_string(),
                source: "codex://project/session-1/outcome".to_string(),
                content: "we deployed yesterday".to_string(),
                concept: "outcome".to_string(),
                group_id: Some("session-1".to_string()),
                filters: BTreeMap::new(),
                headings: vec![],
                links: vec![],
                timestamp: timestamp.map(str::to_string),
                doc_length: 21,
                author_agent: None,
                key_phrases: Vec::new(),
                key_phrase_extraction_hash: String::new(),
            }
        }
        fn hit(doc_id: &str) -> SearchResult {
            SearchResult {
                doc_id: doc_id.to_string(),
                source: "codex://project/session-1/outcome".to_string(),
                group_id: Some("session-1".to_string()),
                score: 1.0,
                score_breakdown: crate::index::ScoreBreakdown::default(),
                matched_entities: Vec::new(),
                matched_terms: Vec::new(),
                probable_topic: None,
                doc_type_guess: None,
                semantic_status: None,
                superseded_by: None,
                relation_confidence: None,
                relation_evidence: Vec::new(),
            }
        }

        let mut store = IndexStore::in_memory(PipelineOptions::default());
        store.upsert(memory("dated", Some("2026-09-20T10:00:00Z")));
        store.upsert(memory("undated", None));
        let service = MemoryService::new(store);
        let payload = search_results(&service, vec![hit("dated"), hit("undated")]);
        let results = payload["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);

        let dated = &results[0];
        assert_eq!(dated["created_at"], "2026-09-20T10:00:00Z");
        assert_eq!(dated["session_id"], "session-1");
        let content = dated["content"].as_str().unwrap();
        assert!(
            content.starts_with("[session date: 2026-09-20T10:00:00Z]\n"),
            "date prefix missing: {content}"
        );
        assert!(content.ends_with("we deployed yesterday"));

        // Fail-open: a memory without a timestamp keeps its original text.
        let undated = &results[1];
        assert!(undated["created_at"].is_null());
        assert_eq!(undated["content"], "we deployed yesterday");
    }

    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    #[test]
    fn resolve_search_session_id_prefers_explicit_over_pointer() {
        let dir = std::env::temp_dir().join(format!(
            "lint-ai-resolve-session-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos())
                .unwrap_or(0),
        ));
        let service =
            MemoryService::at_path(&dir, PipelineOptions::default()).expect("service opens");
        service.note_active_session("claude", "hook-session");
        // Explicit argument wins over the hook-written pointer...
        let resolved = resolve_search_session_id(
            &json!({"query": "q", "session_id": "agent-session"}),
            &service,
            "claude",
        )
        .unwrap();
        assert_eq!(resolved.as_deref(), Some("agent-session"));
        // ...and refreshes the pointer with the explicit id.
        assert_eq!(
            service.current_session_id("claude").as_deref(),
            Some("agent-session")
        );
        // Absent argument falls back to the pointer.
        let resolved =
            resolve_search_session_id(&json!({"query": "q"}), &service, "claude").unwrap();
        assert_eq!(resolved.as_deref(), Some("agent-session"));
        // A present-but-blank argument is still rejected, not silently
        // degraded to the pointer.
        assert!(resolve_search_session_id(
            &json!({"query": "q", "session_id": "  "}),
            &service,
            "claude"
        )
        .is_err());
        // No argument and no pointer means stateless.
        let fresh = MemoryService::at_path(&dir.join("fresh"), PipelineOptions::default())
            .expect("service opens");
        let resolved = resolve_search_session_id(&json!({"query": "q"}), &fresh, "claude").unwrap();
        assert_eq!(resolved, None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
