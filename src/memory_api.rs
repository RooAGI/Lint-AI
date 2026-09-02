//! Memory Add/Search API backed by Lint-AI's `IndexStore`.

use crate::query_plan::PreparedQuery;
use crate::{IndexStore, SourceDocument};
use chrono::{TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

const USER_FILTER: &str = "memory_user_id";

#[derive(Debug, Deserialize)]
pub struct AddRequest {
    pub request_id: String,
    pub messages: Vec<Message>,
    pub user_id: String,
    pub session_id: String,
}

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: String,
    pub timestamp: Option<i64>,
    pub content: String,
    #[serde(default)]
    pub expires_at_ms: Option<u64>,
    #[serde(default)]
    pub supersedes_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AddResponse {
    pub success: bool,
    pub request_id: String,
    pub user_id: String,
    pub session_id: String,
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[allow(dead_code)]
    pub options: Option<Vec<String>>,
    pub user_id: String,
    pub top_k: usize,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub data: Vec<SearchMemory>,
}

#[derive(Debug, Deserialize)]
pub struct DeleteRequest {
    pub user_id: String,
    pub doc_id: String,
}

#[derive(Debug, Deserialize)]
pub struct SupersedeRequest {
    pub user_id: String,
    pub replacement_id: String,
    pub old_id: String,
}

#[derive(Debug, Serialize)]
pub struct SearchMemory {
    pub id: String,
    pub content: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

pub struct MemoryService {
    store: IndexStore,
    superseded_ids: HashSet<(String, String)>,
}

impl MemoryService {
    pub fn new(store: IndexStore) -> Self {
        let superseded_ids = store
            .source_documents()
            .into_iter()
            .filter_map(|doc| {
                Some((
                    doc.filters.get(USER_FILTER)?.clone(),
                    doc.filters.get("supersedes_id")?.clone(),
                ))
            })
            .collect();
        Self {
            store,
            superseded_ids,
        }
    }

    pub fn add(&mut self, request: AddRequest) -> anyhow::Result<AddResponse> {
        validate_identifier(&request.request_id, "request_id")?;
        validate_identifier(&request.user_id, "user_id")?;
        validate_identifier(&request.session_id, "session_id")?;
        if request.messages.is_empty() {
            anyhow::bail!("messages must not be empty");
        }

        for (message_index, message) in request.messages.iter().enumerate() {
            if message.content.trim().is_empty() {
                anyhow::bail!("messages[{message_index}].content must not be empty");
            }
            if message.role != "user" && message.role != "assistant" {
                anyhow::bail!("messages[{message_index}].role must be user or assistant");
            }

            let source = format!(
                "memory://{}/{}/{}",
                request.user_id, request.session_id, message_index
            );
            let mut filters = BTreeMap::new();
            filters.insert(USER_FILTER.to_string(), request.user_id.clone());
            if let Some(expires_at_ms) = message.expires_at_ms {
                filters.insert("expires_at_ms".to_string(), expires_at_ms.to_string());
            }
            if let Some(supersedes_id) = &message.supersedes_id {
                validate_identifier(supersedes_id, "supersedes_id")?;
                self.superseded_ids
                    .insert((request.user_id.clone(), supersedes_id.clone()));
                filters.insert("supersedes_id".to_string(), supersedes_id.clone());
            }
            let timestamp = message.timestamp.and_then(|millis| {
                Utc.timestamp_millis_opt(millis)
                    .single()
                    .map(|date| date.to_rfc3339())
            });
            self.store.upsert(SourceDocument {
                doc_id: crate::stable_doc_id_from_source(&format!(
                    "{}:{message_index}",
                    request.request_id
                )),
                source,
                content: format!("{}: {}", message.role, message.content),
                concept: "memory".to_string(),
                group_id: Some(request.session_id.clone()),
                headings: vec![],
                links: vec![],
                timestamp,
                doc_length: message.content.len(),
                author_agent: Some(message.role.clone()),
                filters,
            });
        }

        // Add returns only after the memory is searchable.
        self.store.refresh()?;
        Ok(AddResponse {
            success: true,
            request_id: request.request_id,
            user_id: request.user_id,
            session_id: request.session_id,
        })
    }

    pub fn search(&mut self, request: SearchRequest) -> anyhow::Result<SearchResponse> {
        validate_identifier(&request.user_id, "user_id")?;
        if request.query.trim().is_empty() {
            return Ok(SearchResponse { data: vec![] });
        }
        let top_k = request.top_k.min(100);
        let results = self.query_results(&request, top_k)?;
        Ok(self.format_search_response(results))
    }

    /// Search the last refreshed snapshot without taking a mutable service lock.
    /// Writers refresh the snapshot before releasing their lock.
    pub fn search_cached(&self, request: SearchRequest) -> anyhow::Result<SearchResponse> {
        validate_identifier(&request.user_id, "user_id")?;
        if request.query.trim().is_empty() {
            return Ok(SearchResponse { data: vec![] });
        }
        let top_k = request.top_k.min(100);
        let results = self.query_results_cached(&request, top_k)?;
        Ok(self.format_search_response(results))
    }

    fn query_results(
        &mut self,
        request: &SearchRequest,
        top_k: usize,
    ) -> anyhow::Result<Vec<crate::SearchResult>> {
        let mut filters = BTreeMap::new();
        filters.insert(USER_FILTER.to_string(), request.user_id.clone());
        let prepared = PreparedQuery::new(&request.query);
        self.store.query_prepared(&prepared, top_k, &filters)
    }

    fn query_results_cached(
        &self,
        request: &SearchRequest,
        top_k: usize,
    ) -> anyhow::Result<Vec<crate::SearchResult>> {
        let mut filters = BTreeMap::new();
        filters.insert(USER_FILTER.to_string(), request.user_id.clone());
        let prepared = PreparedQuery::new(&request.query);
        self.store.query_prepared_cached(&prepared, top_k, &filters)
    }

    fn format_search_response(&self, results: Vec<crate::SearchResult>) -> SearchResponse {
        let now_ms = unix_time_ms();
        let data = results
            .into_iter()
            .filter_map(|result| {
                self.store
                    .source_document_by_id(&result.doc_id)
                    .filter(|doc| {
                        doc.filters
                            .get(USER_FILTER)
                            .map(|user| (user.clone(), doc.doc_id.clone()))
                            .is_none_or(|key| !self.superseded_ids.contains(&key))
                    })
                    .filter(|doc| {
                        doc.filters
                            .get("expires_at_ms")
                            .and_then(|value| value.parse::<u64>().ok())
                            .is_none_or(|expires_at| expires_at > now_ms)
                    })
                    .map(|doc| SearchMemory {
                        id: result.doc_id,
                        content: doc.content.clone(),
                        score: result.score,
                        created_at: doc.timestamp.clone(),
                    })
            })
            .collect();
        SearchResponse { data }
    }

    /// Delete one memory, returning false for an already absent or foreign id.
    pub fn delete(&mut self, user_id: &str, doc_id: &str) -> anyhow::Result<bool> {
        validate_identifier(user_id, "user_id")?;
        validate_identifier(doc_id, "doc_id")?;
        let owned = self
            .store
            .source_document_by_id(doc_id)
            .is_some_and(|doc| doc.filters.get(USER_FILTER).map(String::as_str) == Some(user_id));
        if !owned {
            return Ok(false);
        }
        self.store.remove(doc_id);
        self.store.refresh()?;
        Ok(true)
    }

    /// Mark an existing memory as replacing another memory in the same user scope.
    pub fn supersede(
        &mut self,
        user_id: &str,
        replacement_id: &str,
        old_id: &str,
    ) -> anyhow::Result<bool> {
        validate_identifier(user_id, "user_id")?;
        validate_identifier(replacement_id, "replacement_id")?;
        validate_identifier(old_id, "old_id")?;
        let Some(mut replacement) = self.store.source_document_by_id(replacement_id).cloned()
        else {
            return Ok(false);
        };
        if replacement.filters.get(USER_FILTER).map(String::as_str) != Some(user_id) {
            return Ok(false);
        }
        replacement
            .filters
            .insert("supersedes_id".to_string(), old_id.to_string());
        self.superseded_ids
            .insert((user_id.to_string(), old_id.to_string()));
        self.store.upsert(replacement);
        self.store.refresh()?;
        Ok(true)
    }

    /// Remove all expired memories owned by a user.
    pub fn expire(&mut self, user_id: &str) -> anyhow::Result<usize> {
        validate_identifier(user_id, "user_id")?;
        let now_ms = unix_time_ms();
        let ids = self
            .store
            .source_documents()
            .into_iter()
            .filter_map(|doc| {
                let owned = doc.filters.get(USER_FILTER).map(String::as_str) == Some(user_id);
                let expired = doc
                    .filters
                    .get("expires_at_ms")
                    .and_then(|v| v.parse::<u64>().ok())
                    .is_some_and(|expires_at| expires_at <= now_ms);
                (owned && expired).then(|| doc.doc_id.clone())
            })
            .collect::<Vec<_>>();
        for id in &ids {
            self.store.remove(id);
        }
        if !ids.is_empty() {
            self.store.refresh()?;
        }
        Ok(ids.len())
    }
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

/// Identifiers reach the index as filter values and as `memory://` source URIs,
/// so they must stay short, single-line, and free of control characters.
const MAX_IDENTIFIER_BYTES: usize = 256;

fn validate_identifier(value: &str, name: &str) -> anyhow::Result<()> {
    if value.trim().is_empty() {
        anyhow::bail!("{name} must not be empty");
    }
    if value.len() > MAX_IDENTIFIER_BYTES {
        anyhow::bail!("{name} must be at most {MAX_IDENTIFIER_BYTES} bytes");
    }
    if value.chars().any(|character| character.is_control()) {
        anyhow::bail!("{name} must not contain control characters");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PipelineOptions;

    fn service() -> MemoryService {
        MemoryService::new(IndexStore::in_memory(PipelineOptions::default()))
    }

    #[test]
    fn add_is_immediately_searchable() {
        let mut service = service();
        service
            .add(AddRequest {
                request_id: "request-1".into(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: None,
                    content: "I prefer dark mode in every editor".into(),
                    expires_at_ms: None,
                    supersedes_id: None,
                }],
                user_id: "user-a".into(),
                session_id: "session-a".into(),
            })
            .unwrap();
        let response = service
            .search(SearchRequest {
                query: "dark mode editor".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 100,
            })
            .unwrap();
        assert_eq!(response.data.len(), 1);
        assert!(response.data[0].content.contains("dark mode"));
    }

    #[test]
    fn search_cannot_cross_user_boundaries() {
        let mut service = service();
        for (request_id, user_id, content) in [
            ("request-a", "user-a", "the secret project uses amber"),
            ("request-b", "user-b", "the secret project uses cobalt"),
        ] {
            service
                .add(AddRequest {
                    request_id: request_id.into(),
                    messages: vec![Message {
                        role: "user".into(),
                        timestamp: None,
                        content: content.into(),
                        expires_at_ms: None,
                        supersedes_id: None,
                    }],
                    user_id: user_id.into(),
                    session_id: "session".into(),
                })
                .unwrap();
        }
        let response = service
            .search(SearchRequest {
                query: "secret project color".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 100,
            })
            .unwrap();
        assert!(response
            .data
            .iter()
            .all(|memory| memory.content.contains("amber")));
    }

    #[test]
    fn expired_memories_are_not_searchable_and_can_be_deleted_idempotently() {
        let mut service = service();
        service
            .add(AddRequest {
                request_id: "request-expired".into(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: None,
                    content: "temporary migration decision".into(),
                    expires_at_ms: Some(1),
                    supersedes_id: None,
                }],
                user_id: "user-a".into(),
                session_id: "session-a".into(),
            })
            .unwrap();
        let response = service
            .search(SearchRequest {
                query: "temporary migration".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 10,
            })
            .unwrap();
        assert!(response.data.is_empty());
        assert!(!service.delete("user-a", "missing").unwrap());
    }

    #[test]
    fn superseded_memory_is_hidden_but_replacement_remains_searchable() {
        let mut service = service();
        let old_id = crate::stable_doc_id_from_source("old:0");
        for (request_id, content, supersedes_id) in [
            ("old", "old deployment decision", None),
            ("new", "new deployment decision", Some(old_id.as_str())),
        ] {
            service
                .add(AddRequest {
                    request_id: request_id.into(),
                    messages: vec![Message {
                        role: "user".into(),
                        timestamp: None,
                        content: content.into(),
                        expires_at_ms: None,
                        supersedes_id: supersedes_id.map(str::to_string),
                    }],
                    user_id: "user-a".into(),
                    session_id: "session-a".into(),
                })
                .unwrap();
        }
        let response = service
            .search(SearchRequest {
                query: "deployment decision".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 10,
            })
            .unwrap();
        assert_eq!(response.data.len(), 1);
        assert!(response.data[0].content.contains("new deployment"));
    }

    #[test]
    fn supersession_is_scoped_to_the_requesting_user() {
        let mut service = service();
        let old_id = crate::stable_doc_id_from_source("old:0");
        service
            .add(AddRequest {
                request_id: "old".into(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: None,
                    content: "shared deployment decision".into(),
                    expires_at_ms: None,
                    supersedes_id: None,
                }],
                user_id: "user-a".into(),
                session_id: "session-a".into(),
            })
            .unwrap();
        service
            .add(AddRequest {
                request_id: "replacement".into(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: None,
                    content: "replacement from another user".into(),
                    expires_at_ms: None,
                    supersedes_id: Some(old_id),
                }],
                user_id: "user-b".into(),
                session_id: "session-b".into(),
            })
            .unwrap();

        let response = service
            .search(SearchRequest {
                query: "shared deployment decision".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 10,
            })
            .unwrap();
        assert_eq!(response.data.len(), 1);
        assert!(response.data[0].content.contains("shared deployment"));
    }
}
