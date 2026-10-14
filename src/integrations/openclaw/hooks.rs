//! OpenClaw lifecycle-hook adapter.
//!
//! OpenClaw exposes two hook systems (verified against a live 2026.9.6 host):
//!   - internal hooks, which run in-process and can mutate the live event
//!     (used for `agent:bootstrap` → recall and inject `MEMBRICK.md`), and
//!   - typed plugin hooks, which observe lifecycle events (used for
//!     `agent_end` → Outcome capture, `before_reset` → SessionSummary capture;
//!     `session_start`/`session_end`/`shutdown` are acknowledged no-ops).
//!
//! Stateless by design, like the Claude/Codex hooks: every invocation carries
//! what it needs in the event payload, captures go straight to the memory
//! store, and idempotency comes from stable document IDs in the store — there
//! is no cross-invocation hook state, so concurrent hook processes cannot race.
//! Fail-open throughout: any error leaves OpenClaw's behavior unchanged.

use super::document::{OpenClawDocument, OpenClawDocumentType};
use crate::ids::stable_doc_id_from_source;
use crate::integrations::recall::{relevant_excerpt, truncate_utf8};
use crate::integrations::session_recording::{
    membrick_enabled, record_event_if_enabled, RecordingProvider,
};
use crate::memory_api::MemoryService;
use crate::pipeline::{MemoryIndexLayout, PipelineOptions};
use crate::segments::SegmentRoutingStrategy;
use anyhow::{Context, Result};
use chrono::DateTime;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_TOP_K: usize = 5;
const MAX_EXCERPT_BYTES: usize = 800;
const INJECTED_FILE_NAME: &str = "MEMBRICK.md";
/// Fallback recall query when the shim has no user text to correlate
/// (e.g. a fresh session whose first turn has no `message:received` yet).
const DEFAULT_BOOTSTRAP_QUERY: &str = "decisions unresolved work failures implemented changes";

/// Hook kinds, dispatched from the `--openclaw-hook` CLI flag by the JS shims.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenClawHookKind {
    /// Internal `agent:bootstrap` — retrieve and inject memory.
    Bootstrap,
    /// Typed `agent_end` — capture a per-turn Outcome.
    AgentEnd,
    /// Typed `before_reset` — capture the authoritative SessionSummary.
    BeforeReset,
    /// Typed `session_start` — registry bookkeeping.
    SessionStart,
    /// Typed `session_end` — registry bookkeeping.
    SessionEnd,
    /// Gateway shutdown — bounded pending-capture flush.
    Shutdown,
}

impl OpenClawHookKind {
    pub fn from_str(kind: &str) -> Option<Self> {
        Some(match kind {
            "bootstrap" => Self::Bootstrap,
            "agent-end" => Self::AgentEnd,
            "before-reset" => Self::BeforeReset,
            "session-start" => Self::SessionStart,
            "session-end" => Self::SessionEnd,
            "shutdown" => Self::Shutdown,
            _ => return None,
        })
    }

    fn event_name(self) -> &'static str {
        match self {
            Self::Bootstrap => "agent:bootstrap",
            Self::AgentEnd => "agent_end",
            Self::BeforeReset => "before_reset",
            Self::SessionStart => "session_start",
            Self::SessionEnd => "session_end",
            Self::Shutdown => "shutdown",
        }
    }
}

/// stdin envelope written by the JS shims: the raw hook event, the plugin
/// context (typed hooks only), and — for `agent:bootstrap` — the user text
/// correlated from the earlier `message:received` event.
#[derive(Debug, Clone, Default, Deserialize)]
struct OpenClawHookInput {
    #[serde(default)]
    event: Value,
    #[serde(default)]
    ctx: Value,
    #[serde(default)]
    query: String,
}

/// Entry point for `--openclaw-hook <kind>`.
///
/// Reads one JSON value from stdin, runs the handler, writes one JSON value to
/// stdout. Never fails in a way that blocks OpenClaw: on any error a warning
/// goes to stderr and a valid default response is emitted.
pub fn run_hook(kind: OpenClawHookKind, fallback_root: &Path) -> Result<()> {
    let input: OpenClawHookInput = match crate::integrations::read_bounded_json() {
        Ok(input) => input,
        Err(error) => {
            // Unparseable stdin: there is no safe mutation to return (for
            // bootstrap we cannot know the existing files), so emit a sentinel
            // the shims treat as "leave the event untouched".
            eprintln!("warning: Membrick OpenClaw hook received invalid input: {error:#}");
            emit(&json!({ "ok": false, "error": "invalid hook input" }))?;
            return Ok(());
        }
    };
    let output = process_hook(kind, input, fallback_root);
    emit(&output)
}

/// Execute an OpenClaw lifecycle event for either the CLI shim or HTTP API.
/// Errors are deliberately converted to a fail-open response.
pub fn process_http_hook(kind: OpenClawHookKind, value: Value, fallback_root: &Path) -> Value {
    process_http_hook_with_writer(kind, value, fallback_root, |document| {
        let session_id = document
            .filters
            .get("session_id")
            .cloned()
            .unwrap_or_default();
        crate::integrations::mcp_tools::upsert_hook_document(
            fallback_root,
            "openclaw",
            &session_id,
            document,
        )
    })
}

/// Execute an HTTP hook while sending captures to the caller's resident writer.
pub fn process_http_hook_with_writer(
    kind: OpenClawHookKind,
    value: Value,
    fallback_root: &Path,
    mut publish: impl FnMut(crate::SourceDocument) -> Result<()>,
) -> Value {
    let input: OpenClawHookInput = match serde_json::from_value(value) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("warning: Membrick OpenClaw hook received invalid input: {error:#}");
            return json!({ "ok": false, "error": "invalid hook input" });
        }
    };
    // HTTP callers cannot choose the server's workspace via hook fields.
    match fallback_root.canonicalize() {
        Ok(root) => process_hook_at_root(kind, input, &root, &mut publish),
        Err(_) => default_output(kind, &input),
    }
}

fn process_hook(kind: OpenClawHookKind, input: OpenClawHookInput, fallback_root: &Path) -> Value {
    let root = match resolve_root(&input, kind, fallback_root) {
        Ok(root) => root,
        Err(error) => {
            eprintln!("warning: Membrick OpenClaw hook failed open: {error:#}");
            return default_output(kind, &input);
        }
    };
    let mut publish = |document: crate::SourceDocument| {
        let session_id = document
            .filters
            .get("session_id")
            .cloned()
            .unwrap_or_default();
        crate::integrations::mcp_tools::upsert_hook_document(
            &root,
            "openclaw",
            &session_id,
            document,
        )
    };
    process_hook_at_root(kind, input, &root, &mut publish)
}

fn process_hook_at_root(
    kind: OpenClawHookKind,
    input: OpenClawHookInput,
    root: &Path,
    publish: &mut dyn FnMut(crate::SourceDocument) -> Result<()>,
) -> Value {
    let session_id = session_id(&input, kind);
    if let Err(error) = record_event_if_enabled(
        RecordingProvider::OpenClaw,
        &root,
        &session_id,
        kind.event_name(),
        telemetry_payload(kind, &input),
    ) {
        eprintln!("warning: Membrick OpenClaw session recording failed open: {error:#}");
    }
    let output = match handle_hook(kind, &input, &root, publish) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("warning: Membrick OpenClaw hook failed open: {error:#}");
            default_output(kind, &input)
        }
    };
    output
}

fn emit(output: &Value) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, output)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn handle_hook(
    kind: OpenClawHookKind,
    input: &OpenClawHookInput,
    root: &Path,
    publish: &mut dyn FnMut(crate::SourceDocument) -> Result<()>,
) -> Result<Value> {
    match kind {
        OpenClawHookKind::Bootstrap => handle_bootstrap(input, root),
        OpenClawHookKind::AgentEnd => handle_agent_end(input, root, publish),
        OpenClawHookKind::BeforeReset => handle_before_reset(input, root, publish),
        OpenClawHookKind::SessionStart => handle_session_start(input),
        OpenClawHookKind::SessionEnd => handle_session_end(input),
        OpenClawHookKind::Shutdown => handle_shutdown(),
    }
}

/// The default response emitted when a handler fails: keep OpenClaw's
/// behavior exactly as it was (existing bootstrap files untouched; an `ok:
/// false` acknowledgement for capture hooks).
fn default_output(kind: OpenClawHookKind, input: &OpenClawHookInput) -> Value {
    match kind {
        OpenClawHookKind::Bootstrap => {
            let files = input
                .event
                .pointer("/context/bootstrapFiles")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            json!({ "bootstrapFiles": files })
        }
        _ => json!({ "ok": false }),
    }
}

fn telemetry_payload(kind: OpenClawHookKind, input: &OpenClawHookInput) -> Value {
    match kind {
        OpenClawHookKind::Bootstrap => json!({
            "sessionKey": input.event.get("sessionKey"),
            "bootstrapFileCount": input.event.pointer("/context/bootstrapFiles")
                .and_then(Value::as_array).map(Vec::len).unwrap_or(0),
            "query": truncate_utf8(input.query.trim(), 200),
        }),
        _ => json!({
            "hook": input.event.get("hook").or_else(|| input.event.get("type")),
            "runId": input.event.get("runId"),
            "sessionId": input.ctx.get("sessionId"),
        }),
    }
}

/// The project root is the OpenClaw workspace directory when the event
/// carries one; otherwise the CLI `--path` fallback.
fn resolve_root(
    input: &OpenClawHookInput,
    kind: OpenClawHookKind,
    fallback_root: &Path,
) -> Result<PathBuf> {
    let candidate = match kind {
        OpenClawHookKind::Bootstrap => input.event.pointer("/context/workspaceDir"),
        _ => input.ctx.pointer("/workspaceDir"),
    }
    .and_then(Value::as_str)
    .map(str::trim)
    .filter(|s| !s.is_empty());
    let path = candidate
        .map(PathBuf::from)
        .unwrap_or_else(|| fallback_root.to_path_buf());
    path.canonicalize()
        .with_context(|| format!("failed to canonicalize OpenClaw root {}", path.display()))
}

/// Best-effort session identity for telemetry and registry keys.
fn session_id(input: &OpenClawHookInput, kind: OpenClawHookKind) -> String {
    let pointers: &[&str] = match kind {
        OpenClawHookKind::Bootstrap => &["/context/sessionId"],
        _ => &["/sessionId"],
    };
    for pointer in pointers {
        if let Some(id) = input
            .ctx
            .pointer(pointer)
            .or_else(|| input.event.pointer(pointer))
            .and_then(Value::as_str)
        {
            let id = id.trim();
            if !id.is_empty() {
                return id.to_string();
            }
        }
    }
    "unknown".to_string()
}

fn memory_root(root: &Path) -> PathBuf {
    // Shared store (unified on main): hook captures land where the MCP
    // adapter reads, with provider attribution on the documents.
    crate::integrations::mcp_index::shared_memory_root(root)
}

fn open_store(root: &Path) -> Result<MemoryService> {
    let options = PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 3,
            routing_strategy: SegmentRoutingStrategy::LocalDistinctiveness,
        },
        ..PipelineOptions::default()
    };
    MemoryService::at_path(&memory_root(root), options)
}

fn current_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default();
    DateTime::from_timestamp(seconds, 0)
        .map(|timestamp| timestamp.to_rfc3339())
        .unwrap_or_else(|| "1970-01-01T00:00:00+00:00".to_string())
}

// ---------------------------------------------------------------------------
// agent:bootstrap — retrieve and inject.
// ---------------------------------------------------------------------------

/// Recall from the OpenClaw memory store and return the bootstrap-file list
/// with `MEMBRICK.md` appended (replacing any stale entry with the same name so
/// repeated per-turn firings stay idempotent).
fn handle_bootstrap(input: &OpenClawHookInput, root: &Path) -> Result<Value> {
    let existing = input
        .event
        .pointer("/context/bootstrapFiles")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut files: Vec<Value> = existing
        .into_iter()
        .filter(|file| file.get("name").and_then(Value::as_str) != Some(INJECTED_FILE_NAME))
        .collect();

    if membrick_enabled(RecordingProvider::OpenClaw, root)? {
        let query = input.query.trim();
        let query = if query.is_empty() {
            DEFAULT_BOOTSTRAP_QUERY
        } else {
            query
        };
        let memories = retrieve_memories(root, query)?;
        if !memories.trim().is_empty() {
            // The injected file lives at the resolved root (which already
            // fell back to the install root when the event omitted
            // `workspaceDir`); building the path from the raw event field
            // would yield `/MEMBRICK.md` and the recall would not be injected.
            let injected_path = root.join(INJECTED_FILE_NAME);
            files.push(json!({
                "name": INJECTED_FILE_NAME,
                "path": injected_path.to_string_lossy(),
                "content": format!("# {INJECTED_FILE_NAME}\n\nMembrick recalled memories (automatic):\n{memories}"),
                "missing": false,
            }));
        }
    }
    Ok(json!({ "bootstrapFiles": files }))
}

fn retrieve_memories(root: &Path, query: &str) -> Result<String> {
    let results = crate::integrations::mcp_tools::search_hook_documents(
        root,
        "openclaw",
        None,
        query,
        DEFAULT_TOP_K,
    )?;
    let mut output = String::new();
    for result in results
        .get("results")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let content = result
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let source = result
            .get("source")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let matched_terms = result
            .get("matched_terms")
            .and_then(Value::as_array)
            .map(|terms| {
                terms
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let excerpt = relevant_excerpt(content, query, &matched_terms, MAX_EXCERPT_BYTES);
        let excerpt = excerpt.trim();
        if excerpt.is_empty() {
            continue;
        }
        output.push_str(&format!("\n- Source: {}\n  {}\n", source, excerpt));
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Capture helpers.
// ---------------------------------------------------------------------------

/// Extract `(role, text)` pairs from OpenClaw message payloads.
///
/// OpenClaw message content is either a plain string (typed lifecycle events)
/// or an array of blocks (internal events). Custom runtime-context carrier
/// messages are bootstrap context, not conversation, and are skipped.
fn message_texts(messages: &[Value]) -> Vec<(String, String)> {
    messages
        .iter()
        .filter_map(|message| {
            let role = message.get("role")?.as_str()?;
            if role == "custom" && is_internal_context(message) {
                return None;
            }
            let text = content_text(message.get("content")?)?;
            if text.trim().is_empty() {
                return None;
            }
            Some((role.to_string(), text))
        })
        .collect()
}

/// True for OpenClaw's internal runtime-context carrier messages: the typed
/// `customType`/`details.runtimeContextCarrier` markers, or the raw
/// `<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>` envelope (seen in live
/// `agent_end` payloads without any marker fields).
fn is_internal_context(message: &Value) -> bool {
    if message.get("customType").and_then(Value::as_str) == Some("openclaw.runtime-context") {
        return true;
    }
    if message
        .pointer("/details/runtimeContextCarrier")
        .and_then(Value::as_bool)
        == Some(true)
    {
        return true;
    }
    if let Some(content) = message.get("content").and_then(Value::as_str) {
        if content
            .trim_start()
            .starts_with("<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>")
        {
            return true;
        }
    }
    false
}

fn content_text(content: &Value) -> Option<String> {
    match content {
        Value::String(text) => Some(text.clone()),
        Value::Array(blocks) => {
            let mut out = String::new();
            for block in blocks {
                if matches!(
                    block.get("type").and_then(Value::as_str),
                    Some("text" | "input_text" | "output_text")
                ) {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        out.push_str(text);
                    }
                } else {
                    out.push_str(&serde_json::to_string(block).unwrap_or_default());
                }
                if !out.is_empty() {
                    out.push('\n');
                }
            }
            Some(out)
        }
        _ => None,
    }
}

fn ack(skipped: Option<&str>) -> Value {
    let mut map = Map::new();
    map.insert("ok".to_string(), Value::Bool(skipped.is_none()));
    if let Some(reason) = skipped {
        map.insert("skipped".to_string(), Value::String(reason.to_string()));
    }
    Value::Object(map)
}

/// Capture a per-turn Outcome. Idempotency is a store property, not hook
/// state: the document ID is `openclaw://{project}/{session}/outcome/{run_id}`
/// (see `document.rs`), so `store.upsert` on a retried `agent_end` with the
/// same `runId` replaces the identical record in place — a no-op, exactly as
/// the Claude/Codex hooks rely on.
fn capture_outcome(
    root: &Path,
    session_id: &str,
    run_id: &str,
    channel: Option<&str>,
    turns: &[(String, String)],
    publish: &mut dyn FnMut(crate::SourceDocument) -> Result<()>,
) -> Result<Value> {
    if turns.is_empty() {
        return Ok(ack(Some("no conversation text")));
    }
    let content = turns
        .iter()
        .map(|(role, text)| format!("{role}: {text}"))
        .collect::<Vec<_>>()
        .join("\n\n");
    let document = OpenClawDocument {
        event_id: run_id.to_string(),
        session_id: session_id.to_string(),
        document_type: OpenClawDocumentType::Outcome,
        content,
        cwd: root.to_path_buf(),
        timestamp: Some(current_timestamp()),
        channel: channel.map(str::to_string),
    };
    publish(document.into_source_document()?)?;
    Ok(ack(None))
}

// ---------------------------------------------------------------------------
// agent_end — per-turn Outcome capture (idempotent on runId via the store).
// ---------------------------------------------------------------------------

fn handle_agent_end(
    input: &OpenClawHookInput,
    root: &Path,
    publish: &mut dyn FnMut(crate::SourceDocument) -> Result<()>,
) -> Result<Value> {
    let run_id = input
        .event
        .get("runId")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if run_id.is_empty() {
        return Ok(ack(Some("missing runId")));
    }
    if !membrick_enabled(RecordingProvider::OpenClaw, root)? {
        return Ok(ack(Some("recording disabled")));
    }
    let session_id = session_id(input, OpenClawHookKind::AgentEnd);
    let messages = input
        .event
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let turns = message_texts(&messages);
    let channel = input.ctx.get("channel").and_then(Value::as_str);
    capture_outcome(root, &session_id, run_id, channel, &turns, publish)
}

// ---------------------------------------------------------------------------
// before_reset — authoritative SessionSummary capture from the full
// departing transcript (delivered before OpenClaw wipes the session).
// ---------------------------------------------------------------------------

fn handle_before_reset(
    input: &OpenClawHookInput,
    root: &Path,
    publish: &mut dyn FnMut(crate::SourceDocument) -> Result<()>,
) -> Result<Value> {
    let session_id = session_id(input, OpenClawHookKind::BeforeReset);
    if !membrick_enabled(RecordingProvider::OpenClaw, root)? {
        return Ok(ack(Some("recording disabled")));
    }
    let reason = input
        .event
        .get("reason")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let messages = input
        .event
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let turns = message_texts(&messages);
    let transcript = turns
        .iter()
        .map(|(role, text)| format!("{role}: {text}"))
        .collect::<Vec<_>>()
        .join("\n\n");
    let content = format!("Session closed (reason: {reason}).\n\n{}", transcript);
    let event_id = format!(
        "{session_id}:session-summary:{}",
        stable_doc_id_from_source(&content)
    );
    let document = OpenClawDocument {
        event_id,
        session_id: session_id.clone(),
        document_type: OpenClawDocumentType::SessionSummary,
        content,
        cwd: root.to_path_buf(),
        timestamp: Some(current_timestamp()),
        channel: input
            .ctx
            .get("channel")
            .and_then(Value::as_str)
            .map(str::to_string),
    };
    publish(document.into_source_document()?)?;
    Ok(ack(None))
}

// ---------------------------------------------------------------------------
// session_start / session_end — lifecycle acknowledgements.
//
// These carry no capture today (matching the Claude/Codex lifecycle, where
// session bookkeeping is the host's job). Parent↔child linkage needs no
// registry either: `resumedFrom`/`nextSessionId` already arrive in the event
// payloads (verified against the live 2026.9.6 host), so any future capture
// that wants them can read them straight from the event.
// ---------------------------------------------------------------------------

fn handle_session_start(_input: &OpenClawHookInput) -> Result<Value> {
    Ok(ack(None))
}

fn handle_session_end(_input: &OpenClawHookInput) -> Result<Value> {
    Ok(ack(None))
}

// ---------------------------------------------------------------------------
// Shutdown — bounded pending-capture flush.
//
// All captures above write through synchronously, so this is a no-op
// acknowledgement inside OpenClaw's shared two-second drain budget.
// ---------------------------------------------------------------------------

fn handle_shutdown() -> Result<Value> {
    Ok(ack(None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Isolated project root per test (unique dir, cleaned on drop via tests
    /// writing under target/tmp).
    fn test_root() -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let root = std::env::temp_dir().join(format!(
            "membrick-openclaw-hook-test-{}-{}",
            std::process::id(),
            id
        ));
        fs::create_dir_all(&root).unwrap();
        fs::canonicalize(root).unwrap()
    }

    fn test_publish(root: &Path, document: crate::SourceDocument) -> Result<()> {
        MemoryService::with_shared_memory(root, |service| {
            service.upsert(document);
            service.refresh_index()
        })
    }

    fn write_memory_doc(root: &Path, source: &str, content: &str) {
        let mut store = open_store(root).unwrap();
        store.upsert(
            OpenClawDocument {
                event_id: source.to_string(),
                session_id: "seed".to_string(),
                document_type: OpenClawDocumentType::Outcome,
                content: content.to_string(),
                cwd: root.to_path_buf(),
                timestamp: Some(current_timestamp()),
                channel: None,
            }
            .into_source_document()
            .unwrap(),
        );
        store.refresh_index().unwrap();
    }

    fn bootstrap_input(
        workspace_dir: &str,
        query: &str,
        existing: Vec<Value>,
    ) -> OpenClawHookInput {
        OpenClawHookInput {
            event: json!({
                "type": "agent",
                "action": "bootstrap",
                "sessionKey": "agent:main:dashboard:main",
                "context": {
                    "workspaceDir": workspace_dir,
                    "sessionId": "f917502e-0000-4000-8000-000000000000",
                    "agentId": "agent-main",
                    "bootstrapFiles": existing,
                },
            }),
            ctx: Value::Null,
            query: query.to_string(),
        }
    }

    #[test]
    fn bootstrap_injects_membrick_md_when_memories_exist() {
        let root = test_root();
        write_memory_doc(
            &root,
            "seed-1",
            "The team decided to adopt SQLite for the local session store.",
        );
        let existing = vec![
            json!({"name": "SKILL.md", "path": "/ws/SKILL.md", "content": "x", "missing": false}),
        ];
        let input = bootstrap_input(root.to_str().unwrap(), "local session store", existing);
        let output = handle_bootstrap(&input, &root).unwrap();
        let files = output.get("bootstrapFiles").unwrap().as_array().unwrap();
        assert_eq!(files.len(), 2);
        let injected = files.iter().find(|f| f["name"] == "MEMBRICK.md").unwrap();
        assert!(injected["content"].as_str().unwrap().contains("SQLite"));
        assert_eq!(
            injected["path"],
            format!("{}/MEMBRICK.md", root.to_str().unwrap())
        );
        assert_eq!(injected["missing"], false);
        // Pre-existing files survive.
        assert!(files.iter().any(|f| f["name"] == "SKILL.md"));
    }

    #[test]
    fn bootstrap_injection_is_idempotent() {
        let root = test_root();
        write_memory_doc(&root, "seed-1", "Prefer Postgres for analytics workloads.");
        let input = bootstrap_input(root.to_str().unwrap(), "analytics", vec![]);
        let first = handle_bootstrap(&input, &root).unwrap();
        // Simulate the next per-turn firing: the injected file is now present.
        let rerun_input = bootstrap_input(
            root.to_str().unwrap(),
            "analytics",
            first
                .get("bootstrapFiles")
                .unwrap()
                .as_array()
                .unwrap()
                .clone(),
        );
        let second = handle_bootstrap(&rerun_input, &root).unwrap();
        let files = second.get("bootstrapFiles").unwrap().as_array().unwrap();
        assert_eq!(
            files.iter().filter(|f| f["name"] == "MEMBRICK.md").count(),
            1,
            "must not duplicate the injected file"
        );
    }

    #[test]
    fn bootstrap_with_empty_memory_store_leaves_files_untouched() {
        let root = test_root();
        let existing = vec![
            json!({"name": "SKILL.md", "path": "/ws/SKILL.md", "content": "x", "missing": false}),
        ];
        let input = bootstrap_input(root.to_str().unwrap(), "anything", existing);
        let output = handle_bootstrap(&input, &root).unwrap();
        assert_eq!(
            output
                .get("bootstrapFiles")
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert!(output
            .get("bootstrapFiles")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .all(|f| f["name"] != "MEMBRICK.md"));
    }

    #[test]
    fn bootstrap_recall_composes_workspace_index_with_shared_provider_memory() {
        let root = test_root();
        fs::write(
            root.join("README.md"),
            "The sapphire greenhouse uses a weekly humidity calibration checklist.",
        )
        .unwrap();
        write_memory_doc(
            &root,
            "provider-seed",
            "The copper observatory telescope calibration uses a star chart.",
        );

        let workspace_hits =
            retrieve_memories(&root, "sapphire greenhouse humidity calibration").unwrap();
        assert!(
            workspace_hits.contains("sapphire greenhouse"),
            "{workspace_hits}"
        );

        let provider_hits =
            retrieve_memories(&root, "copper observatory telescope calibration").unwrap();
        assert!(
            provider_hits.contains("copper observatory"),
            "{provider_hits}"
        );
    }

    #[test]
    fn bootstrap_uses_default_query_without_user_text() {
        let root = test_root();
        write_memory_doc(&root, "seed-1", "Decisions about the implemented changes were recorded after we resolved the deployment failures.");
        let input = bootstrap_input(root.to_str().unwrap(), "", vec![]);
        let output = handle_bootstrap(&input, &root).unwrap();
        let files = output.get("bootstrapFiles").unwrap().as_array().unwrap();
        assert!(files.iter().any(|f| f["name"] == "MEMBRICK.md"));
    }

    #[test]
    fn bootstrap_without_workspace_dir_uses_resolved_root_for_injected_path() {
        let root = test_root();
        write_memory_doc(
            &root,
            "seed-1",
            "The team standardized on SQLite for local state.",
        );
        // Live 2026.9.6 payloads sometimes omit `context.workspaceDir`;
        // `resolve_root` then falls back to the install root, which is the
        // `root` the handler receives.
        let input = OpenClawHookInput {
            event: json!({
                "type": "agent",
                "action": "bootstrap",
                "sessionKey": "agent:main:dashboard:main",
                "context": {
                    "sessionId": "f917502e-0000-4000-8000-000000000000",
                    "agentId": "agent-main",
                    "bootstrapFiles": [],
                },
            }),
            ctx: Value::Null,
            query: "local state".to_string(),
        };
        let output = handle_bootstrap(&input, &root).unwrap();
        let files = output.get("bootstrapFiles").unwrap().as_array().unwrap();
        let injected = files.iter().find(|f| f["name"] == "MEMBRICK.md").unwrap();
        let path = injected["path"].as_str().unwrap();
        assert_eq!(path, root.join("MEMBRICK.md").to_string_lossy());
        assert_ne!(path, "/MEMBRICK.md");
    }

    fn agent_end_input(run_id: &str, session_id: &str) -> OpenClawHookInput {
        OpenClawHookInput {
            event: json!({
                "messages": [
                    {"role": "user", "content": "Summarize the deployment steps."},
                    {"role": "assistant", "content": [{"type": "text", "text": "The deployment steps are: build, test, push."}]},
                    // Runtime-context carriers are bootstrap context, not conversation.
                    {"role": "custom", "customType": "openclaw.runtime-context", "content": "carrier payload"},
                    {"role": "custom", "customType": "other", "details": {"runtimeContextCarrier": true}, "content": "carrier payload 2"},
                    // Raw internal-context envelope without marker fields (live agent_end shape).
                    {"role": "custom", "content": "<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>\nActive exec sessions:\nnone"},
                    {"role": "custom", "customType": "other", "content": "a real custom message"},
                ],
                "runId": run_id,
            }),
            ctx: json!({"sessionId": session_id, "channel": "webchat", "workspaceDir": "ignored-here"}),
            query: String::new(),
        }
    }

    #[test]
    fn agent_end_replay_is_idempotent_at_the_store_level() {
        // Mirrors the Claude/Codex hooks' "replayed hook must be idempotent":
        // two independent invocations (no shared state, as if two hook
        // processes) with the same runId must leave exactly one Outcome
        // record. Idempotency is a store property: the document ID is a pure
        // function of (project, session, "outcome", run_id), so the second
        // upsert replaces the identical record in place.
        let root = test_root();
        let mut publish = |document| test_publish(&root, document);
        let first =
            handle_agent_end(&agent_end_input("run-1", "sess-1"), &root, &mut publish).unwrap();
        assert_eq!(first["ok"], true);
        // Retry with the same runId: re-runs capture, store dedupes.
        let second =
            handle_agent_end(&agent_end_input("run-1", "sess-1"), &root, &mut publish).unwrap();
        assert_eq!(second["ok"], true);
        // A new runId captures a second record.
        let third =
            handle_agent_end(&agent_end_input("run-2", "sess-1"), &root, &mut publish).unwrap();
        assert_eq!(third["ok"], true);

        let mut store = open_store(&root).unwrap();
        let results = store
            .search_for_hook("deployment steps", "openclaw", None, 10)
            .unwrap();
        let outcomes: Vec<_> = results
            .iter()
            .filter_map(|r| store.record_by_id(&r.doc_id))
            .filter(|record| {
                record.filters.get("document_type").map(String::as_str) == Some("outcome")
            })
            .collect();
        assert_eq!(outcomes.len(), 2);
        let content = &outcomes[0].content;
        assert!(content.contains("user: Summarize the deployment steps."));
        assert!(content.contains("assistant: The deployment steps are: build, test, push."));
        assert!(content.contains("a real custom message"));
        assert!(!content.contains("carrier payload"));
        assert!(!content.contains("BEGIN_OPENCLAW_INTERNAL_CONTEXT"));
    }

    #[test]
    fn agent_end_without_run_id_skips() {
        let root = test_root();
        let input = OpenClawHookInput {
            event: json!({"messages": []}),
            ctx: json!({"sessionId": "s"}),
            query: String::new(),
        };
        let mut publish = |document| test_publish(&root, document);
        let output = handle_agent_end(&input, &root, &mut publish).unwrap();
        assert_eq!(output["skipped"], "missing runId");
    }

    #[test]
    fn parses_sanitized_runtime_transcript_fixture() {
        let input: OpenClawHookInput = serde_json::from_str(include_str!(
            "../../../tests/fixtures/runtime_transcripts/openclaw.json"
        ))
        .unwrap();
        let messages = input
            .event
            .get("messages")
            .and_then(Value::as_array)
            .unwrap();
        let texts = message_texts(messages);
        assert_eq!(texts.len(), 2);
        assert!(texts.iter().any(|(_, text)| text.contains("default route")));
        assert!(texts.iter().any(|(_, text)| text.contains("toolCall")));
    }

    #[test]
    fn before_reset_captures_session_summary_from_full_transcript() {
        let root = test_root();
        let input = OpenClawHookInput {
            event: json!({
                "messages": [
                    {"role": "user", "content": "How do I reset the gateway?"},
                    {"role": "assistant", "content": "Use the reset command from the dashboard."},
                ],
                "reason": "user-initiated",
            }),
            ctx: json!({"sessionId": "sess-9", "channel": "webchat"}),
            query: String::new(),
        };
        let mut publish = |document| test_publish(&root, document);
        let output = handle_before_reset(&input, &root, &mut publish).unwrap();
        assert_eq!(output["ok"], true);

        let mut store = open_store(&root).unwrap();
        let results = store
            .search_for_hook("reset the gateway", "openclaw", None, 10)
            .unwrap();
        let summaries: Vec<_> = results
            .iter()
            .filter_map(|r| store.record_by_id(&r.doc_id))
            .filter(|record| {
                record.filters.get("document_type").map(String::as_str) == Some("session-summary")
            })
            .collect();
        assert_eq!(summaries.len(), 1);
        assert!(summaries[0].content.contains("reason: user-initiated"));
        assert!(summaries[0].content.contains("How do I reset the gateway?"));
    }

    #[test]
    fn session_lifecycle_hooks_ack_without_state() {
        let start = OpenClawHookInput {
            event: json!({"sessionId": "sess-a", "sessionKey": "agent:main:webchat:sess-a", "resumedFrom": "sess-old"}),
            ctx: json!({"agentId": "agent-main"}),
            query: String::new(),
        };
        assert_eq!(handle_session_start(&start).unwrap()["ok"], true);

        let end = OpenClawHookInput {
            event: json!({"sessionId": "sess-a", "reason": "idle-timeout", "nextSessionId": "sess-b"}),
            ctx: Value::Null,
            query: String::new(),
        };
        assert_eq!(handle_session_end(&end).unwrap()["ok"], true);
    }

    #[test]
    fn shutdown_acks_without_state() {
        assert_eq!(handle_shutdown().unwrap()["ok"], true);
    }

    #[test]
    fn default_output_keeps_bootstrap_files_untouched_on_failure() {
        let input = bootstrap_input("/ws", "q", vec![json!({"name": "SKILL.md"})]);
        let output = default_output(OpenClawHookKind::Bootstrap, &input);
        assert_eq!(
            output
                .get("bootstrapFiles")
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            1
        );
        let output = default_output(OpenClawHookKind::AgentEnd, &input);
        assert_eq!(output["ok"], false);
    }
}
