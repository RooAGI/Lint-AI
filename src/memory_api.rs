//! Memory Add/Search API backed by Lint-AI's `IndexStore`.

use crate::pipeline::{PipelineOptions, PublishedIndexSnapshot};
use crate::query_plan::PreparedQuery;
use crate::{IndexStore, SourceDocument};
use chrono::{TimeZone, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const USER_FILTER: &str = "memory_user_id";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddRequest {
    pub request_id: String,
    pub messages: Vec<Message>,
    pub user_id: String,
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub timestamp: Option<i64>,
    pub content: String,
    #[serde(default)]
    pub expires_at_ms: Option<u64>,
    #[serde(default)]
    pub supersedes_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddResponse {
    pub success: bool,
    pub request_id: String,
    pub user_id: String,
    pub session_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[allow(dead_code)]
    pub options: Option<Vec<String>>,
    pub user_id: String,
    pub top_k: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub data: Vec<SearchMemory>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeleteRequest {
    pub user_id: String,
    #[serde(default)]
    pub doc_id: String,
}

#[derive(Debug, Deserialize)]
pub struct SupersedeRequest {
    pub user_id: String,
    pub replacement_id: String,
    pub old_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetRequest {
    pub user_id: String,
    #[serde(default)]
    pub memory_id: String,
    #[serde(default)]
    pub include_inactive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListRequest {
    pub user_id: String,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default = "default_list_limit")]
    pub limit: usize,
    #[serde(default)]
    pub cursor: Option<String>,
    #[serde(default)]
    pub include_inactive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResponse {
    pub data: Vec<MemoryRecord>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateRequest {
    pub user_id: String,
    #[serde(default)]
    pub memory_id: String,
    pub content: String,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub timestamp: Option<i64>,
    #[serde(default)]
    pub expires_at_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub content: String,
    pub role: Option<String>,
    pub user_id: String,
    pub session_id: Option<String>,
    pub created_at: Option<String>,
    pub expires_at_ms: Option<u64>,
    pub supersedes_id: Option<String>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMemory {
    pub id: String,
    pub content: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    pub role: Option<String>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
}

pub struct MemoryService {
    store: IndexStore,
    superseded_ids: HashSet<(String, String)>,
    request_fingerprints: HashMap<(String, String), String>,
}

#[derive(Clone)]
pub struct MemorySearchService {
    index: PublishedIndexSnapshot,
    superseded_ids: HashSet<(String, String)>,
    query_cache: Arc<Mutex<HashMap<String, Vec<crate::SearchResult>>>>,
}

impl MemoryService {
    /// Create an in-memory service without exposing the storage implementation
    /// to application callers.
    pub fn in_memory(options: PipelineOptions) -> Self {
        Self::new(IndexStore::in_memory(options))
    }

    /// Open a persistent memory service without exposing `IndexStore` in the
    /// application-facing construction API.
    pub fn at_path(index_root: impl AsRef<Path>, options: PipelineOptions) -> anyhow::Result<Self> {
        Ok(Self::new(IndexStore::at_path(
            index_root.as_ref(),
            options,
        )?))
    }

    fn new(store: IndexStore) -> Self {
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
        let request_fingerprints = store
            .source_documents()
            .into_iter()
            .filter_map(|doc| {
                let persisted = doc.filters.get("request_fingerprint")?;
                let fingerprint = if persisted.starts_with("v2:") {
                    persisted.clone()
                } else {
                    let messages = serde_json::from_str::<Vec<Message>>(persisted).ok()?;
                    request_fingerprint(doc.group_id.as_deref()?, &messages).ok()?
                };
                Some((
                    (
                        doc.filters.get(USER_FILTER)?.clone(),
                        doc.filters.get("request_id")?.clone(),
                    ),
                    fingerprint,
                ))
            })
            .collect();
        Self {
            store,
            superseded_ids,
            request_fingerprints,
        }
    }

    /// Return bounded structural information for operational status views.
    /// Callers that expose this information should sanitize identifiers and
    /// persistence paths before sending it outside the process.
    pub fn inspection(&self) -> crate::pipeline::IndexStoreInspection {
        self.store.inspection()
    }

    /// Publish pending mutations. Normal mutation methods already call this;
    /// this method is provided for hosts that batch lower-level changes.
    pub fn refresh(&mut self) -> anyhow::Result<()> {
        self.store.refresh()
    }

    pub fn add(&mut self, request: AddRequest) -> anyhow::Result<AddResponse> {
        let response = self.add_unpublished(request)?;
        self.store.refresh()?;
        Ok(response)
    }

    /// Adds several requests and publishes one snapshot after all mutations.
    /// Each request retains the normal per-request message limit and validation.
    pub fn add_batch(&mut self, requests: Vec<AddRequest>) -> anyhow::Result<Vec<AddResponse>> {
        if requests.is_empty() {
            anyhow::bail!("requests must not be empty");
        }
        let mut responses = Vec::with_capacity(requests.len());
        for request in requests {
            responses.push(self.add_unpublished(request)?);
        }
        self.store.refresh()?;
        Ok(responses)
    }

    fn add_unpublished(&mut self, request: AddRequest) -> anyhow::Result<AddResponse> {
        validate_identifier(&request.request_id, "request_id")?;
        validate_identifier(&request.user_id, "user_id")?;
        validate_identifier(&request.session_id, "session_id")?;
        if request.messages.is_empty() {
            anyhow::bail!("messages must not be empty");
        }
        if request.messages.len() > MAX_MESSAGES_PER_REQUEST {
            anyhow::bail!("messages must contain at most {MAX_MESSAGES_PER_REQUEST} items");
        }
        for (message_index, message) in request.messages.iter().enumerate() {
            validate_message(message, message_index)?;
        }
        let fingerprint = request_fingerprint(&request.session_id, &request.messages)?;
        let request_key = (request.user_id.clone(), request.request_id.clone());
        if let Some(previous) = self.request_fingerprints.get(&request_key) {
            if previous != &fingerprint {
                anyhow::bail!("request_id was already used with different content");
            }
            return Ok(AddResponse {
                success: true,
                request_id: request.request_id,
                user_id: request.user_id,
                session_id: request.session_id,
            });
        }

        for (message_index, message) in request.messages.iter().enumerate() {
            let source = format!(
                "memory://{}/{}/{}",
                request.user_id, request.session_id, message_index
            );
            let mut filters = BTreeMap::new();
            filters.insert(USER_FILTER.to_string(), request.user_id.clone());
            filters.insert("request_id".to_string(), request.request_id.clone());
            filters.insert("request_fingerprint".to_string(), fingerprint.clone());
            if let Some(expires_at_ms) = message.expires_at_ms {
                filters.insert("expires_at_ms".to_string(), expires_at_ms.to_string());
            }
            if let Some(supersedes_id) = &message.supersedes_id {
                validate_identifier(supersedes_id, "supersedes_id")?;
                self.superseded_ids
                    .insert((request.user_id.clone(), supersedes_id.clone()));
                filters.insert("supersedes_id".to_string(), supersedes_id.clone());
            }
            let timestamp = message
                .timestamp
                .map(|millis| timestamp_to_rfc3339(millis, "timestamp"))
                .transpose()?;
            self.store.upsert(SourceDocument {
                doc_id: crate::stable_doc_id_from_source(&format!(
                    "{}:{}:{message_index}",
                    request.user_id, request.request_id
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

        self.request_fingerprints.insert(request_key, fingerprint);
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

    pub fn get(&self, request: GetRequest) -> anyhow::Result<Option<MemoryRecord>> {
        validate_identifier(&request.user_id, "user_id")?;
        validate_identifier(&request.memory_id, "memory_id")?;
        Ok(self
            .store
            .source_document_by_id(&request.memory_id)
            .filter(|doc| owns_memory(doc, &request.user_id))
            .filter(|doc| request.include_inactive || self.is_visible(doc))
            .map(memory_record))
    }

    pub fn list(&self, request: ListRequest) -> anyhow::Result<ListResponse> {
        validate_identifier(&request.user_id, "user_id")?;
        if let Some(session_id) = &request.session_id {
            validate_identifier(session_id, "session_id")?;
        }
        let limit = request.limit.clamp(1, 100);
        let mut documents = self
            .store
            .source_documents()
            .into_iter()
            .filter(|doc| owns_memory(doc, &request.user_id))
            .filter(|doc| {
                request
                    .session_id
                    .as_deref()
                    .is_none_or(|session| doc.group_id.as_deref() == Some(session))
            })
            .filter(|doc| request.include_inactive || self.is_visible(doc))
            .collect::<Vec<_>>();
        documents.sort_by(|left, right| left.doc_id.cmp(&right.doc_id));
        if let Some(cursor) = &request.cursor {
            validate_identifier(cursor, "cursor")?;
            documents.retain(|doc| doc.doc_id > *cursor);
        }
        let has_more = documents.len() > limit;
        documents.truncate(limit);
        let next_cursor = has_more
            .then(|| documents.last().map(|doc| doc.doc_id.clone()))
            .flatten();
        Ok(ListResponse {
            data: documents.into_iter().map(memory_record).collect(),
            next_cursor,
        })
    }

    pub fn update(&mut self, request: UpdateRequest) -> anyhow::Result<Option<MemoryRecord>> {
        validate_identifier(&request.user_id, "user_id")?;
        validate_identifier(&request.memory_id, "memory_id")?;
        let Some(mut document) = self
            .store
            .source_document_by_id(&request.memory_id)
            .cloned()
        else {
            return Ok(None);
        };
        if !owns_memory(&document, &request.user_id) {
            return Ok(None);
        }
        if !self.is_visible(&document) {
            return Ok(None);
        }
        if request.content.trim().is_empty() || request.content.len() > MAX_MESSAGE_CONTENT_BYTES {
            anyhow::bail!(
                "content must be non-empty and at most {MAX_MESSAGE_CONTENT_BYTES} bytes"
            );
        }
        let role = request.role.unwrap_or_else(|| {
            document
                .author_agent
                .clone()
                .unwrap_or_else(|| "user".to_string())
        });
        if role != "user" && role != "assistant" {
            anyhow::bail!("role must be user or assistant");
        }
        document.content = format!("{role}: {}", request.content);
        document.author_agent = Some(role);
        document.timestamp = request
            .timestamp
            .map(|millis| timestamp_to_rfc3339(millis, "timestamp"))
            .transpose()?;
        match request.expires_at_ms {
            Some(expires_at) => {
                document
                    .filters
                    .insert("expires_at_ms".to_string(), expires_at.to_string());
            }
            None => {
                document.filters.remove("expires_at_ms");
            }
        }
        self.store.upsert(document);
        self.store.refresh()?;
        Ok(self
            .store
            .source_document_by_id(&request.memory_id)
            .map(memory_record))
    }

    pub fn published_search(&self) -> MemorySearchService {
        MemorySearchService {
            index: self.store.published_snapshot(),
            superseded_ids: self.superseded_ids.clone(),
            query_cache: Arc::new(Mutex::new(HashMap::new())),
        }
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

    fn is_visible(&self, doc: &SourceDocument) -> bool {
        let not_superseded = doc
            .filters
            .get(USER_FILTER)
            .map(|user| (user.clone(), doc.doc_id.clone()))
            .is_none_or(|key| !self.superseded_ids.contains(&key));
        let not_expired = doc
            .filters
            .get("expires_at_ms")
            .and_then(|value| value.parse::<u64>().ok())
            .is_none_or(|expires_at| expires_at > unix_time_ms());
        not_superseded && not_expired
    }

    fn format_search_response(&self, results: Vec<crate::SearchResult>) -> SearchResponse {
        let data = results
            .into_iter()
            .filter_map(|result| {
                self.store
                    .source_document_by_id(&result.doc_id)
                    .filter(|doc| self.is_visible(doc))
                    .map(|doc| SearchMemory {
                        id: result.doc_id,
                        content: doc.content.clone(),
                        score: result.score,
                        created_at: doc.timestamp.clone(),
                        role: doc.author_agent.clone(),
                        user_id: doc.filters.get(USER_FILTER).cloned(),
                        session_id: doc.group_id.clone(),
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

fn default_list_limit() -> usize {
    100
}

fn owns_memory(doc: &SourceDocument, user_id: &str) -> bool {
    doc.filters.get(USER_FILTER).map(String::as_str) == Some(user_id)
}

fn memory_record(doc: &SourceDocument) -> MemoryRecord {
    let mut metadata = doc.filters.clone();
    metadata.retain(|key, _| {
        !matches!(
            key.as_str(),
            USER_FILTER | "request_id" | "request_fingerprint" | "expires_at_ms" | "supersedes_id"
        )
    });
    MemoryRecord {
        id: doc.doc_id.clone(),
        content: doc.content.clone(),
        role: doc.author_agent.clone(),
        user_id: doc.filters.get(USER_FILTER).cloned().unwrap_or_default(),
        session_id: doc.group_id.clone(),
        created_at: doc.timestamp.clone(),
        expires_at_ms: doc
            .filters
            .get("expires_at_ms")
            .and_then(|value| value.parse().ok()),
        supersedes_id: doc.filters.get("supersedes_id").cloned(),
        metadata,
    }
}

impl MemorySearchService {
    pub fn search(&self, request: SearchRequest) -> anyhow::Result<SearchResponse> {
        validate_identifier(&request.user_id, "user_id")?;
        if request.query.trim().is_empty() {
            return Ok(SearchResponse { data: vec![] });
        }
        let mut filters = BTreeMap::new();
        let cache_key = format!(
            "{}\0{}\0{}",
            request.user_id,
            request.query,
            request.top_k.min(100)
        );
        filters.insert(USER_FILTER.to_string(), request.user_id);
        let prepared = PreparedQuery::new(&request.query);
        let cached = if std::env::var_os("LINT_AI_DISABLE_QUERY_CACHE").is_some() {
            None
        } else {
            self.query_cache
                .lock()
                .expect("query cache lock poisoned")
                .get(&cache_key)
                .cloned()
        };
        let results = if let Some(cached) = cached {
            cached
        } else {
            let results = self
                .index
                .query_prepared(&prepared, request.top_k.min(100), &filters)?;
            let mut cache = self.query_cache.lock().expect("query cache lock poisoned");
            if cache.len() >= 256 {
                if let Some(key) = cache.keys().next().cloned() {
                    cache.remove(&key);
                }
            }
            cache.insert(cache_key, results.clone());
            results
        };
        let now_ms = unix_time_ms();
        let data = results
            .into_iter()
            .filter_map(|result| {
                self.index
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
                        role: doc.author_agent.clone(),
                        user_id: doc.filters.get(USER_FILTER).cloned(),
                        session_id: doc.group_id.clone(),
                    })
            })
            .collect();
        Ok(SearchResponse { data })
    }
}

fn request_fingerprint(session_id: &str, messages: &[Message]) -> anyhow::Result<String> {
    let canonical = serde_json::to_vec(&(session_id, messages))?;
    let digest = Sha256::digest(canonical);
    Ok(format!("v2:{digest:x}"))
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
const MAX_MESSAGES_PER_REQUEST: usize = 1024;
const MAX_MESSAGE_CONTENT_BYTES: usize = 1024 * 1024;

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

fn validate_message(message: &Message, index: usize) -> anyhow::Result<()> {
    if message.content.trim().is_empty() {
        anyhow::bail!("messages[{index}].content must not be empty");
    }
    if message.content.len() > MAX_MESSAGE_CONTENT_BYTES {
        anyhow::bail!("messages[{index}].content exceeds {MAX_MESSAGE_CONTENT_BYTES} bytes");
    }
    if message.role != "user" && message.role != "assistant" {
        anyhow::bail!("messages[{index}].role must be user or assistant");
    }
    if let Some(supersedes_id) = &message.supersedes_id {
        validate_identifier(supersedes_id, "supersedes_id")?;
    }
    if let Some(millis) = message.timestamp {
        timestamp_to_rfc3339(millis, "timestamp")
            .map_err(|error| anyhow::anyhow!("messages[{index}].{error}"))?;
    }
    Ok(())
}

fn timestamp_to_rfc3339(millis: i64, field: &str) -> anyhow::Result<String> {
    Utc.timestamp_millis_opt(millis)
        .single()
        .map(|date| date.to_rfc3339())
        .ok_or_else(|| anyhow::anyhow!("{field} must be a valid Unix timestamp in milliseconds"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PipelineOptions;

    fn service() -> MemoryService {
        MemoryService::in_memory(PipelineOptions::default())
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
    fn identical_request_ids_are_isolated_between_users() {
        let mut service = service();
        for (user_id, content) in [("user-a", "alpha secret"), ("user-b", "beta secret")] {
            service
                .add(AddRequest {
                    request_id: "same-request".into(),
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
        let a = service
            .search(SearchRequest {
                query: "secret".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 10,
            })
            .unwrap();
        let b = service
            .search(SearchRequest {
                query: "secret".into(),
                options: None,
                user_id: "user-b".into(),
                top_k: 10,
            })
            .unwrap();
        assert_eq!(a.data.len(), 1);
        assert_eq!(b.data.len(), 1);
        assert!(a.data[0].content.contains("alpha"));
        assert!(b.data[0].content.contains("beta"));
    }

    #[test]
    fn request_fingerprint_is_a_digest_and_includes_session_id() {
        let mut service = service();
        let messages = vec![Message {
            role: "user".into(),
            timestamp: None,
            content: "sensitive preference".into(),
            expires_at_ms: None,
            supersedes_id: None,
        }];
        service
            .add(AddRequest {
                request_id: "same-request".into(),
                messages: messages.clone(),
                user_id: "user-a".into(),
                session_id: "session-a".into(),
            })
            .unwrap();

        let persisted = service.store.source_documents()[0]
            .filters
            .get("request_fingerprint")
            .unwrap();
        assert!(persisted.starts_with("v2:"));
        assert!(!persisted.contains("sensitive preference"));

        let error = service
            .add(AddRequest {
                request_id: "same-request".into(),
                messages,
                user_id: "user-a".into(),
                session_id: "session-b".into(),
            })
            .unwrap_err();
        assert!(error.to_string().contains("different content"));
    }

    #[test]
    fn invalid_later_message_does_not_partially_add_request() {
        let mut service = service();
        let error = service
            .add(AddRequest {
                request_id: "atomic-request".into(),
                messages: vec![
                    Message {
                        role: "user".into(),
                        timestamp: None,
                        content: "first valid message".into(),
                        expires_at_ms: None,
                        supersedes_id: None,
                    },
                    Message {
                        role: "system".into(),
                        timestamp: None,
                        content: "invalid role".into(),
                        expires_at_ms: None,
                        supersedes_id: None,
                    },
                ],
                user_id: "user-a".into(),
                session_id: "session-a".into(),
            })
            .unwrap_err();
        assert!(error.to_string().contains("role"));
        assert!(service.store.source_documents().is_empty());
    }

    #[test]
    fn published_search_remains_stable_while_writer_advances() {
        let mut service = service();
        service
            .add(AddRequest {
                request_id: "first".into(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: None,
                    content: "project codename zephyr".into(),
                    expires_at_ms: None,
                    supersedes_id: None,
                }],
                user_id: "user-a".into(),
                session_id: "session-a".into(),
            })
            .unwrap();
        let previous = service.published_search();

        service
            .add(AddRequest {
                request_id: "second".into(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: None,
                    content: "database codename quartz".into(),
                    expires_at_ms: None,
                    supersedes_id: None,
                }],
                user_id: "user-a".into(),
                session_id: "session-b".into(),
            })
            .unwrap();

        let request = || SearchRequest {
            query: "database codename quartz".into(),
            options: None,
            user_id: "user-a".into(),
            top_k: 10,
        };
        assert!(previous
            .search(request())
            .unwrap()
            .data
            .iter()
            .all(|memory| !memory.content.contains("quartz")));
        assert!(service
            .published_search()
            .search(request())
            .unwrap()
            .data
            .iter()
            .any(|memory| memory.content.contains("quartz")));
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
        let old_id = crate::stable_doc_id_from_source("user-a:old:0");
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

    #[test]
    fn list_get_and_update_preserve_identity_and_scope() {
        let mut service = service();
        service
            .add(AddRequest {
                request_id: "request-1".into(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: None,
                    content: "old preference".into(),
                    expires_at_ms: None,
                    supersedes_id: None,
                }],
                user_id: "user-a".into(),
                session_id: "session-a".into(),
            })
            .unwrap();
        let listed = service
            .list(ListRequest {
                user_id: "user-a".into(),
                session_id: Some("session-a".into()),
                limit: 10,
                cursor: None,
                include_inactive: false,
            })
            .unwrap();
        assert_eq!(listed.data.len(), 1);
        let id = listed.data[0].id.clone();
        assert!(service
            .get(GetRequest {
                user_id: "user-b".into(),
                memory_id: id.clone(),
                include_inactive: false,
            })
            .unwrap()
            .is_none());

        let updated = service
            .update(UpdateRequest {
                user_id: "user-a".into(),
                memory_id: id.clone(),
                content: "new preference".into(),
                role: None,
                timestamp: None,
                expires_at_ms: None,
            })
            .unwrap()
            .unwrap();
        assert_eq!(updated.id, id);
        assert!(updated.content.contains("new preference"));
        assert_eq!(updated.session_id.as_deref(), Some("session-a"));

        let invalid_update = service.update(UpdateRequest {
            user_id: "user-a".into(),
            memory_id: id.clone(),
            content: "should not be applied".into(),
            role: None,
            timestamp: Some(i64::MAX),
            expires_at_ms: None,
        });
        assert!(invalid_update.is_err());
        let unchanged = service
            .get(GetRequest {
                user_id: "user-a".into(),
                memory_id: id,
                include_inactive: false,
            })
            .unwrap()
            .unwrap();
        assert!(unchanged.content.contains("new preference"));
    }

    #[test]
    fn list_cursor_is_deterministic_and_excludes_inactive_memories() {
        let mut service = service();
        for request_id in ["a", "b"] {
            service
                .add(AddRequest {
                    request_id: request_id.into(),
                    messages: vec![Message {
                        role: "assistant".into(),
                        timestamp: None,
                        content: format!("memory {request_id}"),
                        expires_at_ms: None,
                        supersedes_id: None,
                    }],
                    user_id: "user-a".into(),
                    session_id: "session-a".into(),
                })
                .unwrap();
        }
        let first = service
            .list(ListRequest {
                user_id: "user-a".into(),
                session_id: None,
                limit: 1,
                cursor: None,
                include_inactive: false,
            })
            .unwrap();
        assert_eq!(first.data.len(), 1);
        let second = service
            .list(ListRequest {
                user_id: "user-a".into(),
                session_id: None,
                limit: 1,
                cursor: first.next_cursor,
                include_inactive: false,
            })
            .unwrap();
        assert_eq!(second.data.len(), 1);
        assert_ne!(first.data[0].id, second.data[0].id);
    }
}
