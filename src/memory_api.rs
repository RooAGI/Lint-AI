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
    /// Optional conversation session. When supplied, the search loads the
    /// bounded prior state for (user_id, session_id) and resolves follow-up
    /// phrasing and temporal anchors against it before retrieval. Absent or
    /// empty means stateless search: the query is analyzed on its own. This is
    /// a state key only; it never becomes an implicit `group_id` filter.
    #[serde(default)]
    pub session_id: Option<String>,
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
    conversation_states: std::sync::Mutex<crate::conversation_state::ConversationStateStore>,
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
            conversation_states: std::sync::Mutex::new(
                crate::conversation_state::ConversationStateStore::new(None),
            ),
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
        if let Some(session_id) = request.session_id.as_deref() {
            validate_identifier(session_id, "session_id")?;
        }
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
        if let Some(session_id) = request.session_id.as_deref() {
            validate_identifier(session_id, "session_id")?;
        }
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
        let prepared = self.prepare_query(request);
        let results = self.store.query_prepared(&prepared, top_k, &filters)?;
        self.observe_search_turn(request, &prepared, &results);
        Ok(results)
    }

    fn query_results_cached(
        &self,
        request: &SearchRequest,
        top_k: usize,
    ) -> anyhow::Result<Vec<crate::SearchResult>> {
        let mut filters = BTreeMap::new();
        filters.insert(USER_FILTER.to_string(), request.user_id.clone());
        let prepared = self.prepare_query(request);
        let results = self
            .store
            .query_prepared_cached(&prepared, top_k, &filters)?;
        self.observe_search_turn(request, &prepared, &results);
        Ok(results)
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

    /// Prepare the query for search. Without a session id this is exactly
    /// `PreparedQuery::new(&request.query)`; with one, the query is rewritten
    /// against the session's bounded prior state (follow-up resolution and
    /// temporal-anchor seeding) before analysis.
    fn prepare_query(&self, request: &SearchRequest) -> PreparedQuery {
        prepare_session_query(
            &self.conversation_states,
            &request.user_id,
            request.session_id.as_deref(),
            &request.query,
        )
    }

    /// Record the search turn in the session state. No-op without a session
    /// id. The original query text is recorded (not the rewritten one), along
    /// with the retrieved document ids, entity mentions from the analysis,
    /// and the temporal anchor when the query set one explicitly.
    fn observe_search_turn(
        &self,
        request: &SearchRequest,
        prepared: &PreparedQuery,
        results: &[crate::SearchResult],
    ) {
        observe_session_search(
            &self.conversation_states,
            &request.user_id,
            request.session_id.as_deref(),
            &request.query,
            prepared,
            results,
        );
    }

    /// Search with caller-supplied filters and an explicit session scope,
    /// going through the full stateful path (prepare → search → observe).
    /// The MCP dispatchers use this: `scope` is the provider name (MCP has
    /// no user id), `session_id` is the conversation key within it.
    pub(crate) fn search_with_filters(
        &mut self,
        query: &str,
        scope: &str,
        session_id: Option<&str>,
        top_k: usize,
        filters: &BTreeMap<String, String>,
    ) -> anyhow::Result<Vec<crate::SearchResult>> {
        let prepared = prepare_session_query(&self.conversation_states, scope, session_id, query);
        let results = self.store.query_prepared(&prepared, top_k, filters)?;
        observe_session_search(
            &self.conversation_states,
            scope,
            session_id,
            query,
            &prepared,
            &results,
        );
        Ok(results)
    }

    /// Sync shared memory documents into the index. Runs before each MCP
    /// search so the workspace sees memories recorded by other providers.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn sync_shared_memory(
        &mut self,
        memory_root: &std::path::Path,
    ) -> anyhow::Result<bool> {
        crate::integrations::mcp_index::sync_memory_documents(memory_root, self)
    }

    /// Refresh the index after syncing, so newly synced documents are
    /// searchable. Only the Gemini CLI path needed this explicitly.
    pub(crate) fn refresh_index(&mut self) -> anyhow::Result<()> {
        self.store.refresh()
    }

    /// Open a persistent store at `index_root`, routing construction through
    /// the service so callers never touch `IndexStore` directly.
    pub fn at_path(
        index_root: &std::path::Path,
        options: crate::pipeline::PipelineOptions,
    ) -> anyhow::Result<Self> {
        Ok(Self::new(crate::IndexStore::at_path(index_root, options)?))
    }

    /// Build an in-memory service, for composed views and diagnostics.
    pub fn in_memory(options: crate::pipeline::PipelineOptions) -> Self {
        Self::new(crate::IndexStore::in_memory(options))
    }

    /// Compose a workspace service with one provider's memory service into a
    /// single in-memory query view. Transfers already-published segments
    /// rather than reprocessing source documents.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn compose_segmented(
        workspace: Self,
        provider_memory: Option<Self>,
    ) -> anyhow::Result<Self> {
        let composed = crate::IndexStore::compose_segmented(
            workspace.store,
            provider_memory.map(|service| service.store),
        )?;
        Ok(Self::new(composed))
    }

    /// True when the store holds no documents. Integration read paths use
    /// this for the same early exit they had on the raw store.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Upsert one source document. Integration capture paths call this, then
    /// [`MemoryService::refresh_index`], exactly as they did on the raw store.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn upsert(&mut self, document: crate::SourceDocument) {
        self.store.upsert(document)
    }

    /// Remove one document by id, returning the removed document if present.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn remove(&mut self, doc_id: &str) -> Option<crate::SourceDocument> {
        self.store.remove(doc_id)
    }

    /// Look up a raw index record by document id, for response formatting in
    /// integration read paths.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn record_by_id(&self, doc_id: &str) -> Option<&crate::index::DocRecord> {
        self.store.record_by_id(doc_id)
    }

    /// All source documents, for migration and sync iteration.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn source_documents(&self) -> Vec<&crate::SourceDocument> {
        self.store.source_documents()
    }

    /// Stateless plain query for integration read paths (hooks, recall).
    /// Runs the service query pipeline with no session: identical to the raw
    /// store's `query`, with no conversation state read or recorded.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn query_plain(
        &mut self,
        query: &str,
        top_k: usize,
    ) -> anyhow::Result<Vec<crate::SearchResult>> {
        self.search_with_filters(query, "integration", None, top_k, &BTreeMap::new())
    }

    /// Raw record access for tests asserting on capture idempotency and
    /// document attribution. Production code uses the typed service methods.
    #[cfg(test)]
    pub(crate) fn records(&self) -> Vec<&crate::index::DocRecord> {
        self.store.records()
    }

    /// Index-structure access for tests asserting on segment layout.
    #[cfg(test)]
    pub(crate) fn memory_index_snapshot(&self) -> Option<&crate::pipeline::MemoryIndexSnapshot> {
        self.store.memory_index_snapshot()
    }

    /// Preferred-document selection for hook retrieval: one document per
    /// `session_id`, preferring `session-summary` over `outcome` over
    /// `checkpoint`, ties broken by timestamp. Ranked hits are deduplicated
    /// by session and rewritten to the preferred record's identity.
    /// This is the shared implementation of the logic previously duplicated
    /// in the Claude Code and Codex hook modules.
    #[cfg(any(feature = "claude-code", feature = "codex"))]
    pub(crate) fn select_session_documents(
        &self,
        results: Vec<crate::SearchResult>,
        limit: usize,
    ) -> Vec<crate::SearchResult> {
        use std::collections::{HashMap, HashSet};
        fn document_type_priority(document_type: &str) -> u8 {
            match document_type {
                "session-summary" => 3,
                "outcome" => 2,
                "checkpoint" => 1,
                _ => 0,
            }
        }
        let mut preferred = HashMap::<String, (u8, &crate::index::DocRecord)>::new();
        for record in self.store.records() {
            let Some(session) = record.filters.get("session_id") else {
                continue;
            };
            let priority = document_type_priority(
                record
                    .filters
                    .get("document_type")
                    .map(String::as_str)
                    .unwrap_or_default(),
            );
            match preferred.get(session) {
                Some((current_priority, current))
                    if (*current_priority, current.timestamp.as_deref())
                        >= (priority, record.timestamp.as_deref()) => {}
                _ => {
                    preferred.insert(session.clone(), (priority, record));
                }
            }
        }

        let mut seen_sessions = HashSet::new();
        let mut selected = Vec::new();
        for mut result in results {
            let Some(record) = self.store.record_by_id(&result.doc_id) else {
                continue;
            };
            let session = record
                .filters
                .get("session_id")
                .cloned()
                .or_else(|| result.group_id.clone())
                .unwrap_or_else(|| result.doc_id.clone());
            if !seen_sessions.insert(session.clone()) {
                continue;
            }
            if let Some((_, preferred_record)) = preferred.get(&session) {
                result.doc_id = preferred_record.doc_id.clone();
                result.source = preferred_record.source.clone();
                result.group_id = preferred_record.group_id.clone();
            }
            selected.push(result);
            if selected.len() == limit {
                break;
            }
        }
        selected
    }

    /// Number of source documents in the index, for the MCP info tool.
    pub(crate) fn docs_count(&self) -> usize {
        self.store.source_documents().len()
    }

    /// Format search results as the MCP search payload.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn search_results_payload(
        &self,
        results: Vec<crate::SearchResult>,
    ) -> serde_json::Value {
        crate::integrations::mcp_tools::search_results(self, results)
    }

    /// Format the memory list as the MCP list_memories payload.
    #[cfg(any(
        feature = "claude-code",
        feature = "codex",
        feature = "gemini-cli",
        feature = "agy",
        feature = "muse-code"
    ))]
    pub(crate) fn list_memories_payload(&self, limit: usize) -> serde_json::Value {
        crate::integrations::mcp_tools::list_memories(self, limit)
    }

    /// Look up a source document by id, for tests that verify indexed
    /// content through the service API.
    pub(crate) fn source_document_by_id(&self, doc_id: &str) -> Option<&crate::SourceDocument> {
        self.store.source_document_by_id(doc_id)
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
/// Prepare a query against an optional conversation session. Without a
/// session id this is exactly `PreparedQuery::new(query)`; with one, the
/// query is rewritten against the session's bounded prior state (follow-up
/// resolution and temporal-anchor seeding) before analysis.
///
/// `scope` namespaces the state (a user id for the API path, a provider
/// name for the MCP path); `session_id` is the conversation key within it.
/// Shared by `MemoryService` and the MCP dispatchers so both paths get the
/// same stateful behavior.
pub(crate) fn prepare_session_query(
    states: &std::sync::Mutex<crate::conversation_state::ConversationStateStore>,
    scope: &str,
    session_id: Option<&str>,
    query: &str,
) -> PreparedQuery {
    let Some(session_id) = session_id else {
        return PreparedQuery::new(query);
    };
    let now_ms = unix_time_ms();
    let rewritten = {
        let mut states = states.lock().expect("conversation state lock poisoned");
        let state = states.get(scope, session_id, now_ms);
        crate::session_prepare::resolve_follow_up(query, state)
    };
    PreparedQuery::new(&rewritten)
}

/// Record a completed search turn in the session state. No-op without a
/// session id. The original query text is recorded (not the rewritten one),
/// along with the retrieved document ids, entity mentions from the analysis,
/// and the temporal anchor when the query set one explicitly.
pub(crate) fn observe_session_search(
    states: &std::sync::Mutex<crate::conversation_state::ConversationStateStore>,
    scope: &str,
    session_id: Option<&str>,
    original_query: &str,
    prepared: &PreparedQuery,
    results: &[crate::SearchResult],
) {
    let Some(session_id) = session_id else {
        return;
    };
    let analysis = prepared.analysis();
    let entities: Vec<String> = analysis
        .spans
        .iter()
        .map(|span| span.text.clone())
        .collect();
    let temporal_anchor = analysis
        .temporal
        .as_ref()
        .and_then(|temporal| temporal.resolved_at.clone());
    let doc_ids: Vec<String> = results.iter().map(|r| r.doc_id.clone()).collect();
    let now_ms = unix_time_ms();
    states
        .lock()
        .expect("conversation state lock poisoned")
        .observe(
            scope,
            session_id,
            original_query,
            &doc_ids,
            &entities,
            temporal_anchor,
            now_ms,
        );
}

impl MemorySearchService {
    pub fn search(&self, request: SearchRequest) -> anyhow::Result<SearchResponse> {
        validate_identifier(&request.user_id, "user_id")?;
        if let Some(session_id) = request.session_id.as_deref() {
            validate_identifier(session_id, "session_id")?;
        }
        if request.query.trim().is_empty() {
            return Ok(SearchResponse { data: vec![] });
        }
        let mut filters = BTreeMap::new();
        let cache_key = format!(
            "{}\0{}\0{}\0{}",
            request.user_id,
            request.session_id.as_deref().unwrap_or(""),
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
                session_id: None,
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
                session_id: None,
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
                session_id: None,
            })
            .unwrap();
        let b = service
            .search(SearchRequest {
                query: "secret".into(),
                options: None,
                user_id: "user-b".into(),
                top_k: 10,
                session_id: None,
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
            session_id: None,
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
                session_id: None,
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
                session_id: None,
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
                session_id: None,
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
    fn group_id_is_an_exact_filter_only_when_explicitly_requested() {
        use std::collections::BTreeMap;

        let mut service = service();
        // Two sessions for the same user; group_id comes from session_id.
        for (session, text) in [
            (
                "session-one",
                "the quartz database uses append-only segments",
            ),
            (
                "session-two",
                "the quartz database uses append-only segments",
            ),
        ] {
            service
                .add(AddRequest {
                    request_id: format!("req-{session}"),
                    messages: vec![Message {
                        role: "user".into(),
                        timestamp: None,
                        content: text.into(),
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
                    session_id: session.to_string(),
                })
                .unwrap();
        }

        // No group filter: both sessions' documents are searchable. A
        // session_id on the *search* is a state key only and must not
        // implicitly scope the corpus.
        let prepared = PreparedQuery::new("quartz database segments");
        let mut filters = BTreeMap::new();
        filters.insert(USER_FILTER.to_string(), "user-a".to_string());
        let unfiltered = service
            .store
            .query_prepared(&prepared, 10, &filters)
            .unwrap();
        assert_eq!(unfiltered.len(), 2);

        // Explicit group_id filter: only the requested session's documents.
        filters.insert("group_id".to_string(), "session-one".to_string());
        let filtered = service
            .store
            .query_prepared(&prepared, 10, &filters)
            .unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].group_id.as_deref(), Some("session-one"));
    }

    #[test]
    fn session_temporal_anchor_carries_into_follow_up_retrieval() {
        use chrono::Duration;

        let options = PipelineOptions {
            memory_index_layout: crate::pipeline::MemoryIndexLayout::Segmented {
                query_top_n: 8,
                routing_strategy: crate::segments::SegmentRoutingStrategy::SparseOverlap,
            },
            ..PipelineOptions::default()
        };
        let mut service = MemoryService::new(IndexStore::in_memory(options));
        let today = Utc::now().date_naive();
        let recent_date = (today - Duration::days(1)).format("%Y-%m-%d").to_string();
        let old_date = (today - Duration::days(30)).format("%Y-%m-%d").to_string();

        for (doc_id, group_id, timestamp, content) in [
            (
                "recent-doc",
                "recent",
                recent_date.as_str(),
                "Budget planning notes: allocate 20 percent to research.",
            ),
            (
                "old-doc",
                "old",
                old_date.as_str(),
                "Budget planning notes: allocate 5 percent to research.",
            ),
        ] {
            let mut doc_filters = BTreeMap::new();
            doc_filters.insert(USER_FILTER.to_string(), "user-a".to_string());
            service.store.upsert(SourceDocument {
                doc_id: doc_id.to_string(),
                source: format!("artifact://{doc_id}"),
                content: content.to_string(),
                concept: doc_id.to_string(),
                group_id: Some(group_id.to_string()),
                headings: vec![],
                links: vec![],
                timestamp: Some(timestamp.to_string()),
                doc_length: content.len(),
                author_agent: None,
                filters: doc_filters,
            });
        }
        service.store.refresh().unwrap();

        // Turn 1 sets a temporal anchor via a relative phrase ("yesterday").
        service
            .search(SearchRequest {
                query: "What did we discuss yesterday?".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 5,
                session_id: Some("s-temporal".into()),
            })
            .unwrap();

        // Turn 2 is a follow-up: the carried anchor must restrict retrieval
        // to the in-window segment.
        let turn2 = service
            .search(SearchRequest {
                query: "what about the budget?".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 5,
                session_id: Some("s-temporal".into()),
            })
            .unwrap();
        let ids: Vec<&str> = turn2.data.iter().map(|memory| memory.id.as_str()).collect();
        assert!(
            ids.contains(&"recent-doc"),
            "follow-up should retrieve in-window doc, got {ids:?}"
        );
        assert!(
            !ids.contains(&"old-doc"),
            "follow-up should not retrieve out-of-window doc, got {ids:?}"
        );

        // Stateless baseline: the same wording without a session gets no
        // carried anchor, so the out-of-window segment is still searchable.
        let baseline = service
            .search(SearchRequest {
                query: "what about the budget?".into(),
                options: None,
                user_id: "user-a".into(),
                top_k: 5,
                session_id: None,
            })
            .unwrap();
        let baseline_ids: Vec<&str> = baseline
            .data
            .iter()
            .map(|memory| memory.id.as_str())
            .collect();
        assert!(
            baseline_ids.contains(&"old-doc"),
            "stateless baseline should still see the out-of-window doc, got {baseline_ids:?}"
        );
    }
}
