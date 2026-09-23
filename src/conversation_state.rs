//! Multi-turn conversation state: bounded, TTL-managed per-session memory of
//! what was asked and retrieved. This is the *working* memory of a live
//! conversation, kept separate from the durable long-term memory index.
//!
//! A [`ConversationState`] is keyed by `(user_id, session_id)` and records the
//! recent queries, retrieved document ids, resolved entity mentions, and the
//! current temporal anchor for that session. The [`ConversationStateStore`]
//! keeps the hot set in memory (bounded LRU) and persists each session to a
//! sidecar directory as JSON, so state survives process restarts without
//! polluting the durable memory store.
//!
//! Sessions expire after [`SESSION_TTL_MS`] of inactivity and are dropped
//! lazily on access plus on explicit sweeps. All bounds are fixed: state can
//! never grow without limit no matter how long a session runs.

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// How many recent queries are kept per session.
pub const MAX_RECENT_QUERIES: usize = 8;
/// How many recent retrieved document ids are kept per session.
pub const MAX_RECENT_DOC_IDS: usize = 32;
/// How many resolved entity mentions are kept per session.
pub const MAX_RESOLVED_ENTITIES: usize = 16;
/// How many sessions are kept in the in-memory hot set.
pub const MAX_SESSIONS_IN_MEMORY: usize = 128;
/// Inactivity TTL after which a session is dropped (24 hours).
pub const SESSION_TTL_MS: u64 = 24 * 60 * 60 * 1000;
/// Queries longer than this are truncated before being recorded.
pub const MAX_RECORDED_QUERY_CHARS: usize = 500;
/// Entity mentions longer than this are truncated before being recorded.
pub const MAX_RECORDED_ENTITY_CHARS: usize = 200;
/// Name of the sidecar directory, relative to the store root.
pub const CONVERSATION_STATE_DIR: &str = "conversation_state";

/// Bounded working memory for one live conversation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationState {
    pub user_id: String,
    pub session_id: String,
    /// Wall-clock millis of the last observed turn in this session.
    pub updated_at_ms: u64,
    /// Number of search turns observed in this session.
    pub turn_count: u64,
    /// Most recent queries first (bounded to [`MAX_RECENT_QUERIES`]).
    pub recent_queries: VecDeque<String>,
    /// Most recent retrieved document ids first (bounded to
    /// [`MAX_RECENT_DOC_IDS`]).
    pub recent_doc_ids: VecDeque<String>,
    /// Entity mentions resolved from recent queries, most recent first
    /// (bounded to [`MAX_RESOLVED_ENTITIES`]).
    pub resolved_entities: Vec<String>,
    /// Temporal anchor carried from the previous turn, e.g. `"2026-09-22"`.
    /// `None` when the session has no active anchor.
    pub temporal_anchor: Option<String>,
}

impl ConversationState {
    fn new(user_id: &str, session_id: &str, now_ms: u64) -> Self {
        Self {
            user_id: user_id.to_string(),
            session_id: session_id.to_string(),
            updated_at_ms: now_ms,
            turn_count: 0,
            recent_queries: VecDeque::new(),
            recent_doc_ids: VecDeque::new(),
            resolved_entities: Vec::new(),
            temporal_anchor: None,
        }
    }

    /// True when the session has been idle longer than the TTL.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        now_ms.saturating_sub(self.updated_at_ms) > SESSION_TTL_MS
    }

    fn push_bounded(deque: &mut VecDeque<String>, value: String, bound: usize) {
        deque.push_front(value);
        while deque.len() > bound {
            deque.pop_back();
        }
    }

    /// Record one search turn: the query text, the retrieved document ids,
    /// entity mentions extracted from the query, and the temporal anchor that
    /// was in effect for the turn (or the new anchor when the query sets one).
    pub fn observe_turn(
        &mut self,
        query: &str,
        doc_ids: &[String],
        entities: &[String],
        temporal_anchor: Option<String>,
        now_ms: u64,
    ) {
        self.turn_count = self.turn_count.saturating_add(1);
        self.updated_at_ms = now_ms;
        let query: String = query.chars().take(MAX_RECORDED_QUERY_CHARS).collect();
        if !query.trim().is_empty() {
            // Avoid recording the exact same query twice in a row (retries).
            let duplicate = self.recent_queries.front().is_some_and(|q| q == &query);
            if !duplicate {
                Self::push_bounded(&mut self.recent_queries, query, MAX_RECENT_QUERIES);
            }
        }
        for doc_id in doc_ids {
            Self::push_bounded(&mut self.recent_doc_ids, doc_id.clone(), MAX_RECENT_DOC_IDS);
        }
        for entity in entities {
            let entity: String = entity.chars().take(MAX_RECORDED_ENTITY_CHARS).collect();
            let entity = entity.trim().to_string();
            if entity.is_empty() || self.resolved_entities.contains(&entity) {
                continue;
            }
            self.resolved_entities.insert(0, entity);
            while self.resolved_entities.len() > MAX_RESOLVED_ENTITIES {
                self.resolved_entities.pop();
            }
        }
        // An explicit new anchor replaces the carried one; otherwise the
        // carried anchor persists across turns.
        if temporal_anchor.is_some() {
            self.temporal_anchor = temporal_anchor;
        }
    }
}

/// Disk-backed, bounded store of [`ConversationState`] keyed by
/// `(user_id, session_id)`.
///
/// The hot set lives in memory with LRU eviction at
/// [`MAX_SESSIONS_IN_MEMORY`] entries. Every mutation is written through to
/// `<dir>/<hash>.json` atomically (temp file + rename), so a crash can never
/// leave a torn session file behind. A `None` dir means memory-only: useful
/// for tests and for callers that do not have a writable root.
pub struct ConversationStateStore {
    dir: Option<PathBuf>,
    states: HashMap<(String, String), ConversationState>,
    /// Least-recently-used first; most-recently-used last.
    access_order: VecDeque<(String, String)>,
}

impl ConversationStateStore {
    /// Open a store rooted at `dir`. The directory is created on first write.
    /// Pass `None` for a memory-only store.
    pub fn new(dir: Option<PathBuf>) -> Self {
        Self {
            dir,
            states: HashMap::new(),
            access_order: VecDeque::new(),
        }
    }

    /// Open a store in the conventional sidecar location under `store_root`.
    pub fn open_under(store_root: &Path) -> Self {
        Self::new(Some(store_root.join(CONVERSATION_STATE_DIR)))
    }

    fn unix_time_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Path of the per-provider "current session" pointer inside the store
    /// directory, e.g. `current_session.claude.json`. The pointer names the
    /// session most recently seen active in this workspace; the MCP search
    /// dispatch reads it back as the default session when the caller did not
    /// pass `session_id` explicitly.
    fn current_session_pointer_path(&self, provider: &str) -> Option<PathBuf> {
        self.dir
            .as_ref()
            .map(|dir| dir.join(format!("current_session.{provider}.json")))
    }

    /// Record the session most recently seen active for `provider`.
    ///
    /// Fail-open and best-effort: blank ids are ignored, the write is atomic
    /// (temp file + rename), and every I/O failure is swallowed, so a
    /// read-only or full disk never breaks a hook or a search.
    pub fn note_active_session(&self, provider: &str, session_id: &str) {
        let session_id = session_id.trim();
        if session_id.is_empty() {
            return;
        }
        let Some(path) = self.current_session_pointer_path(provider) else {
            return;
        };
        if let Some(parent) = path.parent() {
            if std::fs::create_dir_all(parent).is_err() {
                return;
            }
        }
        let Ok(bytes) = serde_json::to_vec(&serde_json::json!({
            "session_id": session_id,
            "updated_at_ms": Self::unix_time_ms(),
        })) else {
            return;
        };
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, &bytes).is_err() {
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        if std::fs::rename(&tmp, &path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }

    /// Read back the session most recently marked active for `provider`, or
    /// `None` when there is no pointer, it is unreadable or malformed, or it
    /// is older than the session TTL. Never fails: every problem degrades to
    /// stateless search.
    pub fn current_session_id(&self, provider: &str) -> Option<String> {
        let bytes = std::fs::read(self.current_session_pointer_path(provider)?).ok()?;
        let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
        let session_id = value.get("session_id")?.as_str()?.trim();
        if session_id.is_empty() {
            return None;
        }
        let updated_at_ms = value.get("updated_at_ms")?.as_u64()?;
        let now_ms = Self::unix_time_ms();
        // Reject pointers from the future (clock skew) and stale ones; the
        // conversation state they name expires on the same TTL.
        if updated_at_ms > now_ms || now_ms - updated_at_ms > SESSION_TTL_MS {
            return None;
        }
        Some(session_id.to_string())
    }

    fn session_path(&self, user_id: &str, session_id: &str) -> Option<PathBuf> {
        let dir = self.dir.as_ref()?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        user_id.hash(&mut hasher);
        session_id.hash(&mut hasher);
        let hash = hasher.finish();
        Some(dir.join(format!("{hash:016x}.json")))
    }

    fn touch(&mut self, key: &(String, String)) {
        self.access_order.retain(|k| k != key);
        self.access_order.push_back(key.clone());
        while self.access_order.len() > MAX_SESSIONS_IN_MEMORY {
            if let Some(evicted) = self.access_order.pop_front() {
                // Only evict from memory; the disk copy stays as the
                // cold-storage fallback and is reloaded on next access.
                self.states.remove(&evicted);
            }
        }
    }

    fn load_from_disk(
        &self,
        user_id: &str,
        session_id: &str,
        now_ms: u64,
    ) -> Option<ConversationState> {
        let path = self.session_path(user_id, session_id)?;
        let bytes = std::fs::read(&path).ok()?;
        let state: ConversationState = serde_json::from_slice(&bytes).ok()?;
        // The filename is a hash; verify the payload really belongs to this
        // session before trusting it.
        if state.user_id != user_id || state.session_id != session_id {
            return None;
        }
        if state.is_expired(now_ms) {
            let _ = std::fs::remove_file(&path);
            return None;
        }
        Some(state)
    }

    fn save_to_disk(&self, state: &ConversationState) {
        let Some(path) = self.session_path(&state.user_id, &state.session_id) else {
            return;
        };
        if let Some(parent) = path.parent() {
            if std::fs::create_dir_all(parent).is_err() {
                return;
            }
        }
        let Ok(bytes) = serde_json::to_vec(state) else {
            return;
        };
        // Atomic write: temp file + rename. A crash mid-write leaves the
        // previous complete file (or nothing) behind, never a torn one.
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, &bytes).is_err() {
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        if std::fs::rename(&tmp, &path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }

    /// Load the state for a session, or `None` when there is no live state.
    /// Expired sessions are dropped and reported as absent.
    pub fn get(
        &mut self,
        user_id: &str,
        session_id: &str,
        now_ms: u64,
    ) -> Option<&ConversationState> {
        let key = (user_id.to_string(), session_id.to_string());
        if !self.states.contains_key(&key) {
            if let Some(state) = self.load_from_disk(user_id, session_id, now_ms) {
                self.states.insert(key.clone(), state);
            }
        }
        // Check expiry for the in-memory copy as well (it may have sat idle
        // past the TTL without being evicted).
        let expired = self
            .states
            .get(&key)
            .is_some_and(|state| state.is_expired(now_ms));
        if expired {
            self.states.remove(&key);
            self.access_order.retain(|k| k != &key);
            if let Some(path) = self.session_path(user_id, session_id) {
                let _ = std::fs::remove_file(path);
            }
            return None;
        }
        if self.states.contains_key(&key) {
            self.touch(&key);
            self.states.get(&key)
        } else {
            None
        }
    }

    /// Record one search turn for a session, creating the session when needed.
    /// No-op when either id is blank (callers validate before reaching here).
    pub fn observe(
        &mut self,
        user_id: &str,
        session_id: &str,
        query: &str,
        doc_ids: &[String],
        entities: &[String],
        temporal_anchor: Option<String>,
        now_ms: u64,
    ) {
        if user_id.trim().is_empty() || session_id.trim().is_empty() {
            return;
        }
        let key = (user_id.to_string(), session_id.to_string());
        if !self.states.contains_key(&key) {
            let state = self
                .load_from_disk(user_id, session_id, now_ms)
                .unwrap_or_else(|| ConversationState::new(user_id, session_id, now_ms));
            self.states.insert(key.clone(), state);
        }
        // Observe under a short mutable borrow, then persist the snapshot
        // after the borrow ends.
        let snapshot = {
            let Some(state) = self.states.get_mut(&key) else {
                return;
            };
            state.observe_turn(query, doc_ids, entities, temporal_anchor, now_ms);
            state.clone()
        };
        self.save_to_disk(&snapshot);
        self.touch(&key);
    }

    /// Drop every session idle longer than the TTL, from memory and disk.
    /// Returns the number of sessions dropped.
    pub fn evict_expired(&mut self, now_ms: u64) -> usize {
        let mut dropped = 0;
        let keys: Vec<(String, String)> = self.states.keys().cloned().collect();
        for key in keys {
            let expired = self
                .states
                .get(&key)
                .is_some_and(|state| state.is_expired(now_ms));
            if expired {
                self.states.remove(&key);
                self.access_order.retain(|k| k != &key);
                if let Some(path) = self.session_path(&key.0, &key.1) {
                    let _ = std::fs::remove_file(path);
                }
                dropped += 1;
            }
        }
        // Sweep cold files that were never loaded into memory.
        if let Some(dir) = self.dir.clone() {
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) != Some("json") {
                        continue;
                    }
                    let expired = std::fs::read(&path)
                        .ok()
                        .and_then(|bytes| serde_json::from_slice::<ConversationState>(&bytes).ok())
                        .is_some_and(|state| state.is_expired(now_ms));
                    if expired {
                        if std::fs::remove_file(&path).is_ok() {
                            dropped += 1;
                        }
                    }
                }
            }
        }
        dropped
    }

    /// Number of sessions currently held in memory (for tests/diagnostics).
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.states.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms() -> u64 {
        1_700_000_000_000
    }

    #[test]
    fn observe_creates_bounded_state() {
        let mut store = ConversationStateStore::new(None);
        for i in 0..40 {
            store.observe(
                "user-a",
                "session-a",
                &format!("query {i} about databases"),
                &[format!("doc-{i}")],
                &[format!("entity-{i}")],
                None,
                ms() + i as u64,
            );
        }
        let state = store.get("user-a", "session-a", ms() + 40).unwrap();
        assert_eq!(state.turn_count, 40);
        assert_eq!(state.recent_queries.len(), MAX_RECENT_QUERIES);
        assert_eq!(state.recent_doc_ids.len(), MAX_RECENT_DOC_IDS);
        assert_eq!(state.resolved_entities.len(), MAX_RESOLVED_ENTITIES);
        // Most recent first.
        assert!(state.recent_queries[0].contains("query 39"));
        assert_eq!(state.recent_doc_ids[0], "doc-39");
    }

    #[test]
    fn sessions_are_isolated_by_user_and_session() {
        let mut store = ConversationStateStore::new(None);
        store.observe("user-a", "s1", "query one", &[], &[], None, ms());
        store.observe("user-a", "s2", "query two", &[], &[], None, ms());
        store.observe("user-b", "s1", "query three", &[], &[], None, ms());
        assert_eq!(
            store.get("user-a", "s1", ms()).unwrap().recent_queries[0],
            "query one"
        );
        assert_eq!(
            store.get("user-a", "s2", ms()).unwrap().recent_queries[0],
            "query two"
        );
        assert_eq!(
            store.get("user-b", "s1", ms()).unwrap().recent_queries[0],
            "query three"
        );
        assert!(store.get("user-a", "nope", ms()).is_none());
    }

    #[test]
    fn expired_sessions_are_dropped() {
        let mut store = ConversationStateStore::new(None);
        store.observe("user-a", "s1", "query one", &[], &[], None, ms());
        assert!(store
            .get("user-a", "s1", ms() + SESSION_TTL_MS + 1)
            .is_none());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn temporal_anchor_carries_until_replaced() {
        let mut store = ConversationStateStore::new(None);
        store.observe(
            "user-a",
            "s1",
            "what happened yesterday",
            &[],
            &[],
            Some("2026-09-21".to_string()),
            ms(),
        );
        // Next turn sets no anchor: the carried one persists.
        store.observe(
            "user-a",
            "s1",
            "and the day before",
            &[],
            &[],
            None,
            ms() + 1,
        );
        let state = store.get("user-a", "s1", ms() + 1).unwrap();
        assert_eq!(state.temporal_anchor.as_deref(), Some("2026-09-21"));
        // An explicit new anchor replaces it.
        store.observe(
            "user-a",
            "s1",
            "what about last week",
            &[],
            &[],
            Some("2026-09-14".to_string()),
            ms() + 2,
        );
        let state = store.get("user-a", "s1", ms() + 2).unwrap();
        assert_eq!(state.temporal_anchor.as_deref(), Some("2026-09-14"));
    }

    #[test]
    fn disk_round_trip_survives_new_store() {
        let dir = std::env::temp_dir().join(format!("lint-ai-convstate-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        {
            let mut store = ConversationStateStore::new(Some(dir.clone()));
            store.observe(
                "user-a",
                "s1",
                "query one",
                &["doc-1".to_string()],
                &["quartz".to_string()],
                Some("2026-09-22".to_string()),
                ms(),
            );
        }
        // A fresh store over the same dir recovers the session.
        let mut store = ConversationStateStore::new(Some(dir.clone()));
        let state = store.get("user-a", "s1", ms() + 1000).unwrap();
        assert_eq!(state.recent_queries[0], "query one");
        assert_eq!(state.recent_doc_ids[0], "doc-1");
        assert_eq!(state.resolved_entities[0], "quartz");
        assert_eq!(state.temporal_anchor.as_deref(), Some("2026-09-22"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lru_evicts_cold_sessions_from_memory_but_disk_reloads() {
        let dir =
            std::env::temp_dir().join(format!("lint-ai-convstate-lru-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = ConversationStateStore::new(Some(dir.clone()));
        for i in 0..(MAX_SESSIONS_IN_MEMORY + 10) {
            store.observe(
                "user-a",
                &format!("s{i}"),
                &format!("query {i}"),
                &[],
                &[],
                None,
                ms(),
            );
        }
        assert!(store.len() <= MAX_SESSIONS_IN_MEMORY);
        // The evicted session reloads transparently from disk.
        let state = store.get("user-a", "s0", ms() + 1).unwrap();
        assert_eq!(state.recent_queries[0], "query 0");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn pointer_test_dir(name: &str) -> PathBuf {
        let base = std::env::temp_dir()
            .canonicalize()
            .unwrap_or_else(|_| std::env::temp_dir());
        base.join(format!(
            "lint-ai-pointer-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos())
                .unwrap_or(0),
        ))
    }

    #[test]
    fn active_session_pointer_roundtrips_per_provider() {
        let dir = pointer_test_dir("roundtrip");
        let store = ConversationStateStore::open_under(&dir);
        store.note_active_session("claude", "session-123");
        assert_eq!(
            store.current_session_id("claude").as_deref(),
            Some("session-123")
        );
        // The pointer is per provider: another provider sees nothing.
        assert_eq!(store.current_session_id("codex"), None);
        // A fresh store over the same directory reads the pointer back.
        let reopened = ConversationStateStore::open_under(&dir);
        assert_eq!(
            reopened.current_session_id("claude").as_deref(),
            Some("session-123")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn active_session_pointer_stale_is_ignored() {
        let dir = pointer_test_dir("stale");
        std::fs::create_dir_all(dir.join(CONVERSATION_STATE_DIR)).unwrap();
        let stale_ms = ConversationStateStore::unix_time_ms().saturating_sub(SESSION_TTL_MS + 1);
        std::fs::write(
            dir.join(CONVERSATION_STATE_DIR)
                .join("current_session.claude.json"),
            serde_json::json!({"session_id": "old-session", "updated_at_ms": stale_ms}).to_string(),
        )
        .unwrap();
        let store = ConversationStateStore::open_under(&dir);
        assert_eq!(store.current_session_id("claude"), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn active_session_pointer_corrupt_or_missing_is_ignored() {
        let dir = pointer_test_dir("corrupt");
        std::fs::create_dir_all(dir.join(CONVERSATION_STATE_DIR)).unwrap();
        std::fs::write(
            dir.join(CONVERSATION_STATE_DIR)
                .join("current_session.claude.json"),
            b"not json",
        )
        .unwrap();
        let store = ConversationStateStore::open_under(&dir);
        assert_eq!(store.current_session_id("claude"), None);
        let absent = ConversationStateStore::open_under(&dir.join("absent"));
        assert_eq!(absent.current_session_id("claude"), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn note_active_session_ignores_blank_ids() {
        let dir = pointer_test_dir("blank");
        let store = ConversationStateStore::open_under(&dir);
        store.note_active_session("claude", "   ");
        assert_eq!(store.current_session_id("claude"), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
