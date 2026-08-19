//! Memory Add/Search API backed by Lint-AI's `IndexStore`.

use crate::query_plan::PreparedQuery;
use crate::{IndexStore, SourceDocument};
use chrono::{TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
}

impl MemoryService {
    pub fn new(store: IndexStore) -> Self {
        Self { store }
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
        let mut filters = BTreeMap::new();
        filters.insert(USER_FILTER.to_string(), request.user_id);
        // Same query treatment as the CLI: intent inference and augmentation,
        // scoped to this user's documents.
        let prepared = PreparedQuery::new(&request.query);
        let results = self.store.query_prepared(&prepared, top_k, &filters)?;
        let data = results
            .into_iter()
            .filter_map(|result| {
                self.store
                    .source_document_by_id(&result.doc_id)
                    .map(|doc| SearchMemory {
                        id: result.doc_id,
                        content: doc.content.clone(),
                        score: result.score,
                        created_at: doc.timestamp.clone(),
                    })
            })
            .collect();
        Ok(SearchResponse { data })
    }
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
}
