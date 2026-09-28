//! Agent bulletin board: short status updates agents post and read via MCP.
//!
//! A board is scoped to one owner + workspace + caller-supplied key (e.g.
//! `pr-81-review`). Agents discover boards with [`MemoryService::board_list`]
//! or idempotently create-or-get with [`MemoryService::board_open`]; content
//! operations take an optional `board_id`, defaulting to the current
//! session's board.
//!
//! The default board is a session-scoped alias, not a fixed board: the
//! default board *points to* the board whose ID is derived from the
//! conversation session ID (`session_board_id`). Each session therefore
//! gets its own default board automatically — posts from different
//! sessions never intermingle. The alias name is `"default"`; session
//! boards live under the reserved key namespace `"session:"`, which
//! `board_open` rejects for caller-supplied keys.
//!
//! Storage: boards and posts are [`SourceDocument`]s in the shared
//! [`IndexStore`], distinguished by a `doc_kind` filter (`"board"` vs
//! `"board_post"`). No schema change, no separate database. The single
//! writer path (PR #81) orders mutations; `board_post` assigns the next
//! per-board sequence while holding the board lock.
//!
//! Storage: boards and posts are [`SourceDocument`]s in the shared
//! [`IndexStore`], distinguished by a `doc_kind` filter (`"board"` vs
//! `"board_post"`). No schema change, no separate database. The single
//! writer path (PR #81) orders mutations; `board_post` assigns the next
//! per-board sequence while holding the board lock.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Filter key marking a document's kind in the board subsystem.
pub const BOARD_DOC_KIND_FILTER: &str = "doc_kind";
/// `doc_kind` value for board definition documents.
pub const BOARD_DOC_KIND: &str = "board";
/// `doc_kind` value for board post documents.
pub const BOARD_POST_KIND: &str = "board_post";

/// Reserved board key for the per-(owner, workspace) default board.
/// `board_open` accepts this key like any other; it resolves to the same
/// board the content tools use when `board_id` is omitted.
pub const DEFAULT_BOARD_KEY: &str = "default";

/// Reserved key namespace for session boards. `board_open` rejects
/// caller-supplied keys starting with this prefix so user keys can never
/// collide with a session board's key.
pub const SESSION_BOARD_KEY_PREFIX: &str = "session:";

/// Title used when a session board's persistent record is first created.
pub const DEFAULT_BOARD_TITLE: &str = "Default board";

/// Whether a `board_id` argument is the default-board alias (`"default"`).
pub fn is_default_alias(board_id: &str) -> bool {
    board_id.trim() == DEFAULT_BOARD_KEY
}

/// The board key for a session's board. Reserved: `board_open` refuses
/// caller keys with this prefix.
pub fn session_board_key(session_id: &str) -> String {
    format!("{SESSION_BOARD_KEY_PREFIX}{session_id}")
}

/// Deterministic board ID for a session's board.
///
/// Built on [`board_id_for`] with the reserved `session:` key namespace,
/// so it can never equal a user-keyed board ID (`board_open` rejects
/// caller keys starting with `"session:"`, and `:` is escaped in keys).
pub fn session_board_id(owner: &str, workspace: &str, session_id: &str) -> String {
    board_id_for(owner, workspace, &session_board_key(session_id))
}

/// The default board *points to* the current session's board: omitting
/// `board_id` (or passing the `"default"` alias) resolves to this ID.
pub fn default_board_id(owner: &str, workspace: &str, session_id: &str) -> String {
    session_board_id(owner, workspace, session_id)
}

/// Construct the default board value without persisting it. The persistent
/// record is created on first post; until then the board exists
/// conceptually (empty) so reads and `board_list` behave sensibly.
pub fn default_board(owner: &str, workspace: &str, session_id: &str) -> Board {
    Board {
        board_id: default_board_id(owner, workspace, session_id),
        owner: owner.to_string(),
        workspace: workspace.to_string(),
        key: DEFAULT_BOARD_KEY.to_string(),
        title: format!("{DEFAULT_BOARD_TITLE} (session {session_id})"),
        // Synthetic boards predate any persistent record; the real
        // timestamp is set when the record is created.
        created_at: String::new(),
    }
}

/// Whether a board ID is the default board for (owner, workspace,
/// session_id) — i.e. that session's board.
pub fn is_default_board(
    board_id: &str,
    owner: &str,
    workspace: &str,
    session_id: &str,
) -> bool {
    board_id == default_board_id(owner, workspace, session_id)
}

/// Filter keys used on board and post documents.
pub const BOARD_ID_FILTER: &str = "board_id";
pub const BOARD_OWNER_FILTER: &str = "board_owner";
pub const BOARD_WORKSPACE_FILTER: &str = "board_workspace";
pub const BOARD_KEY_FILTER: &str = "board_key";
pub const BOARD_AUTHOR_FILTER: &str = "author_agent_id";
pub const BOARD_PROVIDER_FILTER: &str = "provider";
pub const BOARD_SEQUENCE_FILTER: &str = "sequence";
pub const BOARD_REQUEST_ID_FILTER: &str = "request_id";

/// A bulletin board: shared by a parent agent and its subagents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Board {
    /// Server-assigned, stable for (owner, workspace, key).
    pub board_id: String,
    /// Stable owner of the memory (user_id).
    pub owner: String,
    /// Workspace scope (canonical project root).
    pub workspace: String,
    /// Caller-supplied key, unique within (owner, workspace).
    pub key: String,
    pub title: String,
    pub created_at: String,
}

/// One post on a board.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardPost {
    /// Document ID of the stored post.
    pub post_id: String,
    pub board_id: String,
    pub author_agent_id: String,
    pub provider: String,
    pub content: String,
    /// Increasing within the board, assigned by the service.
    pub sequence: u64,
    pub created_at: String,
}

/// Deterministic board ID for (owner, workspace, key).
///
/// Stable across restarts and processes, so concurrent `board_open` calls
/// with the same key converge on one board without coordination.
pub fn board_id_for(owner: &str, workspace: &str, key: &str) -> String {
    // Sanitize each component so the composite ID is unambiguous.
    fn esc(s: &str) -> String {
        s.replace('%', "%25").replace(':', "%3A")
    }
    format!("board:{}:{}:{}", esc(owner), esc(workspace), esc(key))
}

/// Document ID for a board definition document.
pub fn board_doc_id(board_id: &str) -> String {
    format!("{board_id}:definition")
}

/// Document ID for a post: includes the sequence so IDs are naturally
/// ordered and unique within the board.
pub fn board_post_doc_id(board_id: &str, sequence: u64) -> String {
    format!("{board_id}:post:{sequence:020}")
}

/// Build the [`SourceDocument`] filters for a board definition.
pub fn board_doc_filters(board: &Board) -> BTreeMap<String, String> {
    let mut f = BTreeMap::new();
    f.insert(BOARD_DOC_KIND_FILTER.to_string(), BOARD_DOC_KIND.to_string());
    f.insert(BOARD_ID_FILTER.to_string(), board.board_id.clone());
    f.insert(BOARD_OWNER_FILTER.to_string(), board.owner.clone());
    f.insert(
        BOARD_WORKSPACE_FILTER.to_string(),
        board.workspace.clone(),
    );
    f.insert(BOARD_KEY_FILTER.to_string(), board.key.clone());
    f
}

/// Build the [`SourceDocument`] filters for a post.
pub fn board_post_doc_filters(post: &BoardPost, request_id: &str) -> BTreeMap<String, String> {
    let mut f = BTreeMap::new();
    f.insert(
        BOARD_DOC_KIND_FILTER.to_string(),
        BOARD_POST_KIND.to_string(),
    );
    f.insert(BOARD_ID_FILTER.to_string(), post.board_id.clone());
    f.insert(BOARD_AUTHOR_FILTER.to_string(), post.author_agent_id.clone());
    f.insert(BOARD_PROVIDER_FILTER.to_string(), post.provider.clone());
    f.insert(
        BOARD_SEQUENCE_FILTER.to_string(),
        post.sequence.to_string(),
    );
    if !request_id.is_empty() {
        f.insert(
            BOARD_REQUEST_ID_FILTER.to_string(),
            request_id.to_string(),
        );
    }
    f
}

/// Serialize a board definition into document content (JSON).
pub fn board_doc_content(board: &Board) -> String {
    serde_json::to_string(board).unwrap_or_else(|_| "{}".to_string())
}

/// Parse a board definition back from document content.
pub fn board_from_doc_content(content: &str) -> Option<Board> {
    serde_json::from_str(content).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn board_id_is_stable_and_unambiguous() {
        let a = board_id_for("u", "ws", "k");
        let b = board_id_for("u", "ws", "k");
        assert_eq!(a, b);
        // ':' in components must not collide with separators.
        let tricky = board_id_for("u", "w:s", "k");
        let plain = board_id_for("u", "w", "s:k");
        assert_ne!(tricky, plain);
    }

    #[test]
    fn post_doc_ids_order_lexicographically_by_sequence() {
        let b = "board:u:ws:k";
        assert!(board_post_doc_id(b, 2) < board_post_doc_id(b, 10));
    }

    #[test]
    fn session_board_id_is_stable_and_session_scoped() {
        let a = session_board_id("u", "ws", "s1");
        let b = session_board_id("u", "ws", "s1");
        assert_eq!(a, b);
        // Different sessions get different boards.
        assert_ne!(a, session_board_id("u", "ws", "s2"));
        // The session board ID is exactly the board ID for the reserved
        // "session:{sid}" key — which is why board_open rejects
        // caller-supplied keys with the "session:" prefix.
        assert_eq!(a, board_id_for("u", "ws", "session:s1"));
        // The default alias points at the session board.
        assert_eq!(default_board_id("u", "ws", "s1"), a);
        assert!(is_default_board(&a, "u", "ws", "s1"));
        assert!(!is_default_board(&a, "u", "ws", "s2"));
        assert!(is_default_alias("default"));
        assert!(is_default_alias("  default  "));
        assert!(!is_default_alias(&a));
    }
}
