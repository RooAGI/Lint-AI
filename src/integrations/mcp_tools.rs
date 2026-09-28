use crate::index::SearchResult;
use crate::integrations::mcp_transport::{JsonRpcError, JsonRpcResponse, ToolDefinition};
use crate::memory_api::MemoryService;
use crate::source::SourceDocument;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

/// Canonical provider values for `filters.provider`, the per-document
/// attribution stamped on every captured memory.
pub(crate) const PROVIDER_FILTER_VALUES: &[&str] =
    &["claude", "codex", "gemini-cli", "agy", "muse", "openclaw"];

/// JSON Schema fragment for the optional `provider` search argument.
pub(crate) fn provider_argument_schema() -> Value {
    json!({
        "type": "string",
        "description": "Restrict results to memories captured by one provider (claude, codex, gemini-cli, agy, muse, openclaw). Omit to search the shared pool.",
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
        || document.source.starts_with("agy://")
        || document.source.starts_with("muse://")
        || document.source.starts_with("openclaw://")
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
                "agy-session:",
                "muse-session:",
                "openclaw-session:",
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

// ---------------------------------------------------------------------------
// Agent bulletin board tools.
// ---------------------------------------------------------------------------

/// Tool definitions for the bulletin board. Every content operation takes an
/// explicit `board_id`; boards are discovered via `board_list` or opened
/// idempotently via `board_open` with a caller-supplied key.
pub(crate) fn board_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "board_open".to_string(),
            description: "Open a bulletin board by key, creating it if needed. The key is unique within your workspace; calling board_open twice with the same key returns the same board. Share the returned board_id with subagents so they post to the same board. The key \"default\" is reserved: it opens the current session's board (see board_post).".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Caller-supplied board key, e.g. \"pr-81-review\". Unique within the workspace. \"default\" is reserved for the current session's board; keys starting with \"session:\" are rejected."},
                    "title": {"type": "string", "description": "Human-readable board title."},
                    "session_id": {"type": "string", "description": "Conversation session ID. Defaults to the session most recently seen active in this workspace. Only needed to resolve the default board."},
                },
                "required": ["key", "title"],
                "additionalProperties": false
            }),
        },
        ToolDefinition {
            name: "board_list".to_string(),
            description: "List bulletin boards in this workspace with their IDs, keys, and titles. Includes the \"default\" entry pointing at the current session's board.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Conversation session ID. Defaults to the session most recently seen active in this workspace. Only needed to resolve the default board."},
                },
                "additionalProperties": false
            }),
        },
        ToolDefinition {
            name: "board_info".to_string(),
            description: "Show a board's details.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "board_id": {"type": "string", "description": "Board ID from board_open or board_list. Pass \"default\" for the current session's default board."},
                    "session_id": {"type": "string", "description": "Conversation session ID. Defaults to the session most recently seen active in this workspace. Only needed to resolve the default board."},
                },
                "required": ["board_id"],
                "additionalProperties": false
            }),
        },
        ToolDefinition {
            name: "board_post".to_string(),
            description: "Post a short status update to a board. The result includes the board_id; pass that exact ID to board_read and share it with subagents so everyone uses the same board. Pass a unique request_id so retries are safe (a repeated request_id returns the original post instead of a duplicate). Omitting board_id selects only the current session's default board; each session has its own.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "board_id": {"type": "string", "description": "For a shared board, pass the exact board_id returned by board_open or board_post. Omit (or pass \"default\") only for this session's private default board."},
                    "content": {"type": "string", "description": "Post content, e.g. \"The parser failure comes from the empty input path.\""},
                    "request_id": {"type": "string", "description": "Unique ID for this post attempt; reuse it when retrying."},
                    "author_agent_id": {"type": "string", "description": "Your agent ID (e.g. subagent ID). Defaults to the provider name."},
                    "session_id": {"type": "string", "description": "Conversation session ID. Defaults to the session most recently seen active in this workspace. Only needed to resolve the default board."},
                },
                "required": ["content", "request_id"],
                "additionalProperties": false
            }),
        },
        ToolDefinition {
            name: "board_read".to_string(),
            description: "Read a board's posts in posting order. For reliable readback, pass the same explicit board_id returned by board_open or board_post; omitting it reads only the current session's default board. Pass after_sequence (from the last post you saw) to catch up on new posts only.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "board_id": {"type": "string", "description": "Pass the exact board_id returned by board_open or board_post to read that shared board. Omit (or pass \"default\") only for this session's private default board."},
                    "after_sequence": {"type": "integer", "minimum": 0, "description": "Only return posts after this sequence number. Omit to read from the start."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                    "session_id": {"type": "string", "description": "Conversation session ID. Defaults to the session most recently seen active in this workspace. Only needed to resolve the default board."},
                },
                "required": [],
                "additionalProperties": false
            }),
        },
        ToolDefinition {
            name: "board_get".to_string(),
            description: "Get one complete board post by ID. Omit board_id for the current session's default board.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "board_id": {"type": "string", "description": "Board ID from board_open or board_list. Omit (or pass \"default\") for the current session's default board."},
                    "post_id": {"type": "string", "description": "Post ID from board_post or board_read."},
                    "session_id": {"type": "string", "description": "Conversation session ID. Defaults to the session most recently seen active in this workspace. Only needed to resolve the default board."},
                },
                "required": ["post_id"],
                "additionalProperties": false
            }),
        },
        ToolDefinition {
            name: "board_search".to_string(),
            description: "Search a board's older posts by keyword. Omit board_id to search the current session's default board.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "board_id": {"type": "string", "description": "Board ID from board_open or board_list. Omit (or pass \"default\") for the current session's default board."},
                    "query": {"type": "string", "description": "Search query."},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                    "session_id": {"type": "string", "description": "Conversation session ID. Defaults to the session most recently seen active in this workspace. Only needed to resolve the default board."},
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        },
    ]
}

/// Dispatch a board tool call. Returns the JSON result payload, or an error
/// string for invalid arguments / unknown boards.
///
/// `owner` is the stable memory owner (user_id), `workspace` the canonical
/// project root, `provider` the calling provider ("claude", "codex").
/// Every board_id comes from the agent's arguments and is validated
/// against the store; unknown boards are errors, never implicit creations.
pub(crate) fn dispatch_board_tool(
    tool: &str,
    arguments: &Value,
    service: &mut MemoryService,
    owner: &str,
    workspace: &str,
    provider: &str,
) -> Result<Value, String> {
    use crate::board::BoardPost;

    fn get_str<'a>(args: &'a Value, name: &str) -> Result<&'a str, String> {
        args.get(name)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("missing required argument: {name}"))
    }
    fn board_json(board: &crate::board::Board) -> Value {
        json!({
            "board_id": board.board_id,
            "key": board.key,
            "title": board.title,
            "owner": board.owner,
            "workspace": board.workspace,
            "created_at": board.created_at,
        })
    }
    fn post_json(post: &BoardPost) -> Value {
        json!({
            "post_id": post.post_id,
            "board_id": post.board_id,
            "author_agent_id": post.author_agent_id,
            "provider": post.provider,
            "content": post.content,
            "sequence": post.sequence,
            "created_at": post.created_at,
        })
    }
    // Reject unknown arguments per tool.
    let allowed: &[&str] = match tool {
        "board_open" => &["key", "title", "session_id"],
        "board_list" => &["session_id"],
        "board_info" => &["board_id", "session_id"],
        "board_post" => &["board_id", "content", "request_id", "author_agent_id", "session_id"],
        "board_read" => &["board_id", "after_sequence", "limit", "session_id"],
        "board_get" => &["board_id", "post_id", "session_id"],
        "board_search" => &["board_id", "query", "top_k", "session_id"],
        _ => return Err(format!("unknown board tool: {tool}")),
    };
    if let Some(obj) = arguments.as_object() {
        if let Some(unknown) = obj.keys().find(|k| !allowed.contains(&k.as_str())) {
            return Err(format!("unknown {tool} argument: {unknown}"));
        }
    }
    // Optional board_id: omitted (or the "default" alias) means the
    // current session's board.
    let opt_board_id = arguments
        .get("board_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty());
    // Session for default-board resolution: explicit `session_id` arg,
    // else the session most recently seen active in this workspace
    // (hook-tracked, same as search).

    match tool {
        "board_open" => {
            let key = get_str(arguments, "key")?;
            let title = get_str(arguments, "title")?;
            // The "default" key is an alias for the current session's
            // board, not a separate board.
            let board = if crate::board::is_default_alias(key) {
                let session_id = resolve_search_session_id(arguments, &*service, provider)?
                    .ok_or_else(|| {
                        "no active session: pass session_id to board_open \"default\"".to_string()
                    })?;
                service
                    .board_open_session(owner, workspace, &session_id, title)
                    .map_err(|e| e.to_string())?
            } else {
                service
                    .board_open(owner, workspace, key, title)
                    .map_err(|e| e.to_string())?
            };
            Ok(board_json(&board))
        }
        "board_list" => {
            let session_id = resolve_search_session_id(arguments, &*service, provider)?;
            let boards = service
                .board_list(owner, workspace, session_id.as_deref())
                .map_err(|e| e.to_string())?;
            Ok(json!({ "boards": boards.iter().map(board_json).collect::<Vec<_>>() }))
        }
        "board_info" => {
            let board_id = get_str(arguments, "board_id")?;
            let session_id = resolve_search_session_id(arguments, &*service, provider)?;
            match service
                .board_info(board_id, owner, workspace, session_id.as_deref())
                .map_err(|e| e.to_string())?
            {
                Some(board) => Ok(board_json(&board)),
                None => Err(format!("unknown board_id: {board_id}")),
            }
        }
        "board_post" => {
            let content = get_str(arguments, "content")?;
            let request_id = get_str(arguments, "request_id")?;
            let session_id = resolve_search_session_id(arguments, &*service, provider)?;
            let author_agent_id = arguments
                .get("author_agent_id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or(provider);
            let post = service
                .board_post(
                    opt_board_id,
                    owner,
                    workspace,
                    session_id.as_deref(),
                    author_agent_id,
                    provider,
                    content,
                    request_id,
                )
                .map_err(|e| e.to_string())?;
            Ok(post_json(&post))
        }
        "board_read" => {
            let after_sequence = arguments
                .get("after_sequence")
                .and_then(Value::as_u64);
            let limit = arguments
                .get("limit")
                .and_then(Value::as_u64)
                .unwrap_or(20)
                .clamp(1, 100) as usize;
            let session_id = resolve_search_session_id(arguments, &*service, provider)?;
            let posts = service
                .board_read(
                    opt_board_id,
                    owner,
                    workspace,
                    session_id.as_deref(),
                    after_sequence,
                    limit,
                )
                .map_err(|e| e.to_string())?;
            Ok(json!({ "posts": posts.iter().map(post_json).collect::<Vec<_>>() }))
        }
        "board_get" => {
            let post_id = get_str(arguments, "post_id")?;
            let session_id = resolve_search_session_id(arguments, &*service, provider)?;
            match service
                .board_get(opt_board_id, owner, workspace, session_id.as_deref(), post_id)
                .map_err(|e| e.to_string())?
            {
                Some(post) => Ok(post_json(&post)),
                None => Err(format!("post not found: {post_id}")),
            }
        }
        "board_search" => {
            let query = get_str(arguments, "query")?;
            let top_k = arguments
                .get("top_k")
                .and_then(Value::as_u64)
                .unwrap_or(10)
                .clamp(1, 50) as usize;
            let session_id = resolve_search_session_id(arguments, &*service, provider)?;
            let posts = service
                .board_search(
                    opt_board_id,
                    owner,
                    workspace,
                    session_id.as_deref(),
                    query,
                    top_k,
                )
                .map_err(|e| e.to_string())?;
            Ok(json!({ "posts": posts.iter().map(post_json).collect::<Vec<_>>() }))
        }
        _ => Err(format!("unknown board tool: {tool}")),
    }
}

/// Stable memory owner for agent-added memories over MCP.
const MCP_MEMORY_USER_ID: &str = "mcp";

/// MCP tool definitions for direct memory writes and reads.
pub(crate) fn memory_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "add_memory".to_string(),
            description: "Record a memory: a fact, decision, or observation worth remembering. Returns the request_id. Pass a unique request_id so retries are safe (a repeated request_id with the same content returns success without duplicating).".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The memory content, e.g. \"The API rate limit is 100 requests per minute.\""},
                    "request_id": {"type": "string", "description": "Unique ID for this write; reuse it when retrying."},
                    "session_id": {"type": "string", "description": "Conversation session ID. Defaults to the session most recently seen active in this workspace."},
                    "role": {"type": "string", "description": "Message role: \"user\" or \"assistant\". Defaults to \"assistant\".", "enum": ["user", "assistant"]}
                },
                "required": ["content", "request_id"],
                "additionalProperties": false
            }),
        },
        ToolDefinition {
            name: "get_memory".to_string(),
            description: "Get one stored memory by its memory ID (from search results or add_memory). Returns the full memory record, or an error when the ID is unknown.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory ID to fetch."},
                    "include_inactive": {"type": "boolean", "description": "Also return memories that were superseded or expired. Defaults to false.", "default": false}
                },
                "required": ["memory_id"],
                "additionalProperties": false
            }),
        },
    ]
}

/// Dispatch an `add_memory` / `get_memory` tool call. Returns the JSON
/// result payload, or an error string for invalid arguments.
///
/// These wrap the same `MemoryService::add` / `MemoryService::get` the
/// HTTP server exposes, with the stable `"mcp"` memory owner. Writes run
/// the full enrichment pipeline; reads enforce ownership and visibility.
pub(crate) fn dispatch_memory_tool(
    tool: &str,
    arguments: &Value,
    service: &mut MemoryService,
    provider: &str,
) -> Result<Value, String> {
    use crate::memory_api::{AddRequest, GetRequest, Message};

    fn get_str<'a>(args: &'a Value, name: &str) -> Result<&'a str, String> {
        args.get(name)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("{name} is required"))
    }

    let allowed: &[&str] = match tool {
        "add_memory" => &["content", "request_id", "session_id", "role"],
        "get_memory" => &["memory_id", "include_inactive"],
        _ => return Err(format!("unknown memory tool: {tool}")),
    };
    if let Some(obj) = arguments.as_object() {
        if let Some(unknown) = obj.keys().find(|k| !allowed.contains(&k.as_str())) {
            return Err(format!("unknown {tool} argument: {unknown}"));
        }
    }

    match tool {
        "add_memory" => {
            let content = get_str(arguments, "content")?;
            let request_id = get_str(arguments, "request_id")?;
            let role = arguments
                .get("role")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or("assistant");
            if role != "user" && role != "assistant" {
                return Err("role must be \"user\" or \"assistant\"".to_string());
            }
            // Same session resolution as search: explicit arg, else the
            // session most recently seen active in this workspace.
            let session_id =
                resolve_search_session_id(arguments, &*service, provider)?.ok_or_else(|| {
                    "no active session: pass session_id to add_memory".to_string()
                })?;
            let request = AddRequest {
                request_id: request_id.to_string(),
                messages: vec![Message {
                    role: role.to_string(),
                    // No server-injected wall-clock time: the request fingerprint
                    // covers the whole message, so a fresh timestamp per call
                    // would make every retry of the same request_id look like
                    // a conflicting request. The document timestamp is
                    // display-only (export), not a retrieval input.
                    timestamp: None,
                    content: content.to_string(),
                    expires_at_ms: None,
                    supersedes_id: None,
                }],
                user_id: MCP_MEMORY_USER_ID.to_string(),
                session_id,
            };
            let response = service.add(request).map_err(|e| e.to_string())?;
            serde_json::to_value(&response).map_err(|e| e.to_string())
        }
        "get_memory" => {
            let memory_id = get_str(arguments, "memory_id")?;
            let include_inactive = arguments
                .get("include_inactive")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let request = GetRequest {
                user_id: MCP_MEMORY_USER_ID.to_string(),
                memory_id: memory_id.to_string(),
                include_inactive,
            };
            match service.get(request).map_err(|e| e.to_string())? {
                Some(record) => serde_json::to_value(&record).map_err(|e| e.to_string()),
                None => Err(format!("memory not found: {memory_id}")),
            }
        }
        _ => Err(format!("unknown memory tool: {tool}")),
    }
}

// ---------------------------------------------------------------------------
// Shared tools/call dispatch for the bulletin-board tools and add_memory /
// get_memory, used by every MCP adapter that opts in (claude_code, codex,
// and the gemini_cli-based adapters: gemini, agy, openclaw).
// ---------------------------------------------------------------------------

/// Route one `tools/call` invocation for the bulletin-board tools or
/// `add_memory` / `get_memory`, and wrap the payload in the standard MCP
/// text-content envelope (or a `-32602` JSON-RPC error). `service` must be an
/// initialized memory service; this syncs the shared memory store first.
/// Boards are scoped to the workspace root with the stable `"mcp"` owner,
/// exactly as the claude_code/codex adapters did before consolidation.
///
/// Write tools (`board_open`, `board_post`, `add_memory`) run against the
/// persistent shared store under the cross-process write lock — the composed
/// view is in-memory only, so view writes would vanish on process exit and
/// stay invisible to hooks and other providers. Reads stay on the view.
pub(crate) fn call_board_or_memory_tool(
    tool_name: &str,
    id: Option<Value>,
    arguments: &Value,
    service: &mut MemoryService,
    root: &Path,
    provider: &str,
) -> anyhow::Result<JsonRpcResponse> {
    let shared_root = crate::integrations::mcp_index::shared_memory_root(root);
    service.sync_shared_memory(&shared_root)?;
    // Board owner/workspace: the workspace root scopes boards;
    // "mcp" is the stable owner for agent-posted boards.
    let workspace = root.to_string_lossy().to_string();
    let is_write = matches!(tool_name, "board_open" | "board_post" | "add_memory");
    let result: Result<Value, String> = if is_write {
        match crate::integrations::mcp_index::with_shared_store_write(root, |store| {
            dispatch_write_tool(tool_name, arguments, store, "mcp", &workspace, provider)
                .map_err(anyhow::Error::msg)
        }) {
            Ok(value) => Ok(value),
            Err(error) => Err(format!("{error:#}")),
        }
    } else {
        match tool_name {
            "board_open" | "board_list" | "board_info" | "board_post" | "board_read" | "board_get"
            | "board_search" => {
                dispatch_board_tool(tool_name, arguments, service, "mcp", &workspace, provider)
            }
            "add_memory" | "get_memory" => {
                dispatch_memory_tool(tool_name, arguments, service, provider)
            }
            _ => Err(format!("unknown tool: {tool_name}")),
        }
    };
    // After a successful write, re-sync the view so this process observes
    // its own write without waiting for the next pre-dispatch sync.
    if is_write && result.is_ok() {
        service.sync_shared_memory(&shared_root)?;
    }
    match result {
        Ok(payload) => Ok(JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string_pretty(&payload)?,
                }]
            })),
            error: None,
        }),
        Err(message) => Ok(JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError {
                code: -32602,
                message,
            }),
        }),
    }
}

/// Dispatch one of the write tools (`board_open`, `board_post`, `add_memory`)
/// against an explicitly passed service. Separated from
/// [`call_board_or_memory_tool`] so the write path can target the persistent
/// shared store while reads stay on the composed in-memory view.
fn dispatch_write_tool(
    tool_name: &str,
    arguments: &Value,
    service: &mut MemoryService,
    owner: &str,
    workspace: &str,
    provider: &str,
) -> Result<Value, String> {
    match tool_name {
        "board_open" | "board_post" => {
            dispatch_board_tool(tool_name, arguments, service, owner, workspace, provider)
        }
        "add_memory" => dispatch_memory_tool(tool_name, arguments, service, provider),
        _ => Err(format!("unknown write tool: {tool_name}")),
    }
}

// Codex validates arguments locally because its protocol already exposes the
// unknown argument name; Claude and Gemini use this shared parser instead.
#[cfg_attr(
    not(any(feature = "claude-code", feature = "gemini-cli")),
    allow(dead_code)
)]
pub(crate) fn parse_list_memories_limit(arguments: &Value) -> Result<usize, &'static str> {    if arguments
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

    fn memory_service() -> MemoryService {
        MemoryService::in_memory(PipelineOptions::default())
    }

    #[test]
    fn board_open_default_alias_redirects_to_session_board() {
        let mut service = memory_service();
        // board_open(key="default") goes to the session board, not a
        // literal "default" board.
        let opened = dispatch_board_tool(
            "board_open",
            &json!({"key": "default", "title": "T", "session_id": "sess-9"}),
            &mut service,
            "mcp",
            "/tmp/ws-board-dispatch",
            "claude",
        )
        .unwrap();
        let expected = crate::board::default_board_id("mcp", "/tmp/ws-board-dispatch", "sess-9");
        assert_eq!(opened["board_id"], json!(expected));
        // Omitting board_id on post lands on the same board.
        let posted = dispatch_board_tool(
            "board_post",
            &json!({"content": "hi", "request_id": "r1", "session_id": "sess-9"}),
            &mut service,
            "mcp",
            "/tmp/ws-board-dispatch",
            "claude",
        )
        .unwrap();
        assert_eq!(posted["board_id"], json!(expected));
        // And an explicit "default" board_id agrees too.
        let posted_alias = dispatch_board_tool(
            "board_post",
            &json!({"board_id": "default", "content": "yo", "request_id": "r2",
                    "session_id": "sess-9"}),
            &mut service,
            "mcp",
            "/tmp/ws-board-dispatch",
            "claude",
        )
        .unwrap();
        assert_eq!(posted_alias["board_id"], json!(expected));
        assert_eq!(posted_alias["sequence"], json!(2));
    }

    #[test]
    fn memory_tool_definitions_expose_add_and_get() {
        let defs = memory_tool_definitions();
        let names: Vec<_> = defs.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["add_memory", "get_memory"]);
        // Schemas require only the essentials.
        assert_eq!(defs[0].input_schema["required"], json!(["content", "request_id"]));
        assert_eq!(defs[1].input_schema["required"], json!(["memory_id"]));
    }

    #[test]
    fn dispatch_memory_tool_rejects_unknown_tool_and_args() {
        let mut service = memory_service();
        let err = dispatch_memory_tool("nope", &json!({}), &mut service, "claude")
            .unwrap_err();
        assert!(err.contains("unknown memory tool"), "{err}");
        let err = dispatch_memory_tool(
            "add_memory",
            &json!({"content": "x", "request_id": "r1", "bogus": 1}),
            &mut service,
            "claude",
        )
        .unwrap_err();
        assert!(err.contains("unknown add_memory argument"), "{err}");
    }

    #[test]
    fn add_memory_then_get_memory_round_trip() {
        let mut service = memory_service();
        let payload = dispatch_memory_tool(
            "add_memory",
            &json!({"content": "The API rate limit is 100 requests per minute.",
                    "request_id": "mem-1", "session_id": "s1"}),
            &mut service,
            "claude",
        )
        .unwrap();
        assert_eq!(payload["success"], json!(true));
        assert_eq!(payload["request_id"], json!("mem-1"));

        // The memory ID is deterministic: stable_doc_id("mcp:mem-1:0").
        let memory_id = crate::stable_doc_id_from_source("mcp:mem-1:0");
        let record = dispatch_memory_tool(
            "get_memory",
            &json!({"memory_id": memory_id}),
            &mut service,
            "claude",
        )
        .unwrap();
        assert!(record["content"]
            .as_str()
            .unwrap()
            .contains("rate limit"));

        // Unknown ID is an error, not an empty result.
        let err = dispatch_memory_tool(
            "get_memory",
            &json!({"memory_id": "does-not-exist"}),
            &mut service,
            "claude",
        )
        .unwrap_err();
        assert!(err.contains("memory not found"), "{err}");
    }

    #[test]
    fn add_memory_request_id_retry_is_idempotent() {
        let mut service = memory_service();
        let args = json!({"content": "same content", "request_id": "dup-1",
                          "session_id": "s1"});
        let first = dispatch_memory_tool("add_memory", &args, &mut service, "claude").unwrap();
        let second = dispatch_memory_tool("add_memory", &args, &mut service, "claude").unwrap();
        assert_eq!(first, second);
        // Same request_id with different content is still a conflict.
        let err = dispatch_memory_tool(
            "add_memory",
            &json!({"content": "different content", "request_id": "dup-1",
                    "session_id": "s1"}),
            &mut service,
            "claude",
        )
        .unwrap_err();
        assert!(err.contains("already used with different content"), "{err}");
    }

    #[test]
    fn add_memory_requires_session() {
        let mut service = memory_service();
        // No session_id arg and no hook-tracked session: clear error.
        let err = dispatch_memory_tool(
            "add_memory",
            &json!({"content": "x", "request_id": "r1"}),
            &mut service,
            "claude",
        )
        .unwrap_err();
        assert!(err.contains("session_id"), "{err}");
    }

    #[test]
    fn add_memory_validates_role_and_content() {
        let mut service = memory_service();
        let err = dispatch_memory_tool(
            "add_memory",
            &json!({"content": "x", "request_id": "r1", "session_id": "s1", "role": "system"}),
            &mut service,
            "claude",
        )
        .unwrap_err();
        assert!(err.contains("role"), "{err}");
        let err = dispatch_memory_tool(
            "add_memory",
            &json!({"request_id": "r1", "session_id": "s1"}),
            &mut service,
            "claude",
        )
        .unwrap_err();
        assert!(err.contains("content is required"), "{err}");
    }

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

    fn test_root(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lint-ai-{}-{}-{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos())
                .unwrap_or(0),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Unwrap a successful `tools/call` envelope back to its JSON payload.
    fn tool_payload(response: &JsonRpcResponse) -> Value {
        assert!(
            response.error.is_none(),
            "tool error: {:?}",
            response.error
        );
        let text = response.result.as_ref().expect("result")["content"][0]["text"]
            .as_str()
            .expect("text content");
        serde_json::from_str(text).expect("payload parses")
    }

    /// P1: board and memory writes must land on the persistent shared store,
    /// not the in-memory composed view. Drives `board_open` -> `board_post`
    /// and `add_memory` through the shared dispatch, drops the view entirely
    /// (simulating process exit), then reopens the shared store fresh and
    /// requires the board, post, and memory to be present and readable.
    #[test]
    fn board_and_memory_writes_survive_reopen() {
        use crate::integrations::mcp_index;
        let root = test_root("board-persist");
        // Build the composed in-memory view exactly like the adapters do.
        let mut view = mcp_index::open_workspace_memory_store(
            &root,
            mcp_index::SHARED_MEMORY_DIR,
            &[],
            || Ok(vec![]),
        )
        .expect("view opens");
        let workspace = root.to_string_lossy().to_string();

        let opened = tool_payload(
            &call_board_or_memory_tool(
                "board_open",
                None,
                &json!({"key": "k1", "title": "Board", "session_id": "sess-p1"}),
                &mut view,
                &root,
                "openclaw",
            )
            .expect("board_open dispatches"),
        );
        let board_id = opened["board_id"]
            .as_str()
            .expect("board_id in payload")
            .to_string();

        let posted = tool_payload(
            &call_board_or_memory_tool(
                "board_post",
                None,
                &json!({"board_id": board_id, "content": "hello board",
                        "request_id": "post-1", "session_id": "sess-p1"}),
                &mut view,
                &root,
                "openclaw",
            )
            .expect("board_post dispatches"),
        );
        assert_eq!(posted["content"], json!("hello board"));

        let added = tool_payload(
            &call_board_or_memory_tool(
                "add_memory",
                None,
                &json!({"content": "The API rate limit is 100 requests per minute.",
                        "request_id": "mem-p1", "session_id": "sess-p1"}),
                &mut view,
                &root,
                "openclaw",
            )
            .expect("add_memory dispatches"),
        );
        assert_eq!(added["success"], json!(true));

        // Simulate process exit: the in-memory view is gone.
        drop(view);

        // Reopen the persistent shared store fresh, as another process would.
        let mut store = MemoryService::at_path(
            mcp_index::shared_memory_root(&root),
            mcp_index::segmented_store_options(),
        )
        .expect("shared store reopens");

        let boards = dispatch_board_tool(
            "board_list",
            &json!({"session_id": "sess-p1"}),
            &mut store,
            "mcp",
            &workspace,
            "openclaw",
        )
        .expect("board_list");
        assert!(
            boards["boards"]
                .as_array()
                .unwrap()
                .iter()
                .any(|board| board["board_id"] == board_id),
            "board missing after reopen: {boards}"
        );
        let posts = dispatch_board_tool(
            "board_read",
            &json!({"board_id": board_id, "session_id": "sess-p1"}),
            &mut store,
            "mcp",
            &workspace,
            "openclaw",
        )
        .expect("board_read");
        assert!(
            posts["posts"]
                .as_array()
                .unwrap()
                .iter()
                .any(|post| post["content"] == "hello board"),
            "post missing after reopen: {posts}"
        );
        let memory_id = crate::stable_doc_id_from_source("mcp:mem-p1:0");
        let record = dispatch_memory_tool(
            "get_memory",
            &json!({"memory_id": memory_id}),
            &mut store,
            "openclaw",
        )
        .expect("get_memory");
        assert!(
            record["content"]
                .as_str()
                .unwrap()
                .contains("rate limit"),
            "memory missing after reopen: {record}"
        );

        let _ = std::fs::remove_dir_all(&root);
    }
}
