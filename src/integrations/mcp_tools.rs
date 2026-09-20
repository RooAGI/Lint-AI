use crate::index::SearchResult;
use crate::integrations::mcp_transport::ToolDefinition;
use crate::pipeline::IndexStore;
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

/// Format retrieval hits for an agent. Keep this separate from the internal
/// ranking representation: diagnostics and score components are useful while
/// tuning the index, but distract an agent from the memory itself.
pub(crate) fn search_results(store: &IndexStore, results: Vec<SearchResult>) -> Value {
    let results = results
        .into_iter()
        .filter_map(|result| {
            let document = store.source_document_by_id(&result.doc_id)?;
            Some(json!({
                "id": result.doc_id,
                "source": result.source,
                "content": document.content.chars().take(4_000).collect::<String>(),
                "score": result.score,
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
pub(crate) fn list_memories(store: &IndexStore, limit: usize) -> Value {
    let memories = store
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
    use crate::pipeline::PipelineOptions;
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
        });
        let payload = list_memories(&store, 20);
        assert_eq!(payload["count"], 1);
        assert_eq!(
            payload["memories"][0]["source"],
            "codex://project/session-1/outcome"
        );
        assert_eq!(parse_list_memories_limit(&json!({"limit": 0})).unwrap(), 1);
        assert!(parse_list_memories_limit(&json!({"unexpected": true})).is_err());
    }
}
