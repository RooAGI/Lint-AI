//! OpenClaw lifecycle-hook adapter.
//!
//! OpenClaw exposes two hook systems (verified against a live 2026.9.6 host):
//!   - internal hooks, which run in-process and can mutate the live event
//!     (used for `agent:bootstrap` → recall and inject `LINTAI.md`), and
//!   - typed plugin hooks, which observe lifecycle events (used for
//!     `agent_end` → Outcome capture, `before_reset` → SessionSummary capture,
//!     `session_start`/`session_end` → session registry, shutdown → flush).
//!
//! The thin JS shims installed by `--openclaw-install` spawn
//! `lint-ai --openclaw-hook <kind>` with the event JSON on stdin and apply
//! the JSON written to stdout. All lifecycle logic lives here in Rust.
//! Fail-open throughout: any error leaves OpenClaw's behavior unchanged.

use super::document::{OpenClawDocument, OpenClawDocumentType};
use crate::ids::stable_doc_id_from_source;
use crate::integrations::recall::{relevant_excerpt, truncate_utf8};
use crate::integrations::session_recording::{
    lint_ai_enabled, record_event_if_enabled, RecordingProvider,
};
use crate::pipeline::{IndexStore, MemoryIndexLayout, PipelineOptions};
use crate::segments::SegmentRoutingStrategy;
use anyhow::{Context, Result};
use chrono::DateTime;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_TOP_K: usize = 5;
const MAX_EXCERPT_BYTES: usize = 800;
const MAX_CAPTURE_BYTES: usize = 32 * 1024;
const MAX_STATE_RUN_IDS: usize = 2000;
const MAX_STATE_SESSIONS: usize = 500;
const INJECTED_FILE_NAME: &str = "LINTAI.md";
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
            eprintln!("warning: Lint-AI OpenClaw hook received invalid input: {error:#}");
            emit(&json!({ "ok": false, "error": "invalid hook input" }))?;
            return Ok(());
        }
    };
    let root = match resolve_root(&input, kind, fallback_root) {
        Ok(root) => root,
        Err(error) => {
            eprintln!("warning: Lint-AI OpenClaw hook failed open: {error:#}");
            emit(&default_output(kind, &input))?;
            return Ok(());
        }
    };
    let session_id = session_id(&input, kind);
    if let Err(error) = record_event_if_enabled(
        RecordingProvider::OpenClaw,
        &root,
        &session_id,
        kind.event_name(),
        telemetry_payload(kind, &input),
    ) {
        eprintln!("warning: Lint-AI OpenClaw session recording failed open: {error:#}");
    }
    let output = match handle_hook(kind, &input, &root) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("warning: Lint-AI OpenClaw hook failed open: {error:#}");
            default_output(kind, &input)
        }
    };
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &output)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn emit(output: &Value) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, output)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn handle_hook(kind: OpenClawHookKind, input: &OpenClawHookInput, root: &Path) -> Result<Value> {
    let mut state = load_state(root)?;
    let output = match kind {
        OpenClawHookKind::Bootstrap => handle_bootstrap(input, root)?,
        OpenClawHookKind::AgentEnd => handle_agent_end(input, root, &mut state)?,
        OpenClawHookKind::BeforeReset => handle_before_reset(input, root, &mut state)?,
        OpenClawHookKind::SessionStart => handle_session_start(input, &mut state)?,
        OpenClawHookKind::SessionEnd => handle_session_end(input, &mut state)?,
        OpenClawHookKind::Shutdown => handle_shutdown(&state)?,
    };
    save_state(root, &state)?;
    Ok(output)
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
    let path = candidate.map(PathBuf::from).unwrap_or_else(|| fallback_root.to_path_buf());
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
    root.join(".lint-ai").join("openclaw-memory")
}

fn open_store(root: &Path) -> Result<IndexStore> {
    let options = PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 3,
            routing_strategy: SegmentRoutingStrategy::LocalDistinctiveness,
        },
        ..PipelineOptions::default()
    };
    IndexStore::at_path(&memory_root(root), options)
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
// Hook state: runId idempotency + session registry.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct SessionRecord {
    #[serde(default)]
    open: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    session_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    resumed_from: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ended_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    next_session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    agent_id: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct HookState {
    #[serde(default)]
    seen_run_ids: VecDeque<String>,
    #[serde(default)]
    sessions: HashMap<String, SessionRecord>,
}

fn state_path(root: &Path) -> PathBuf {
    root.join(".lint-ai").join("openclaw-hooks").join("state.json")
}

fn load_state(root: &Path) -> Result<HookState> {
    let path = state_path(root);
    if !path.is_file() {
        return Ok(HookState::default());
    }
    let bytes = fs::read(&path)
        .with_context(|| format!("failed to read OpenClaw hook state {}", path.display()))?;
    Ok(serde_json::from_slice(&bytes).unwrap_or_default())
}

fn save_state(root: &Path, state: &HookState) -> Result<()> {
    let path = state_path(root);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!("failed to create OpenClaw hook state dir {}", parent.display())
        })?;
    }
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(state)?)
        .with_context(|| format!("failed to write OpenClaw hook state {}", tmp.display()))?;
    fs::rename(&tmp, &path)
        .with_context(|| format!("failed to publish OpenClaw hook state {}", path.display()))?;
    Ok(())
}

/// Returns true when `run_id` was already captured (retries re-fire the same
/// `runId`, so this is the capture idempotency key).
fn already_seen(state: &mut HookState, run_id: &str) -> bool {
    if state.seen_run_ids.iter().any(|seen| seen == run_id) {
        return true;
    }
    state.seen_run_ids.push_back(run_id.to_string());
    while state.seen_run_ids.len() > MAX_STATE_RUN_IDS {
        state.seen_run_ids.pop_front();
    }
    false
}

fn record_session<'a>(state: &'a mut HookState, session_id: &str) -> &'a mut SessionRecord {
    if state.sessions.len() >= MAX_STATE_SESSIONS && !state.sessions.contains_key(session_id) {
        // Bounded registry: drop a closed session before adding a new one.
        if let Some(closed) = state
            .sessions
            .iter()
            .find(|(_, record)| !record.open)
            .map(|(id, _)| id.clone())
        {
            state.sessions.remove(&closed);
        }
    }
    state.sessions.entry(session_id.to_string()).or_default()
}

// ---------------------------------------------------------------------------
// agent:bootstrap — retrieve and inject.
// ---------------------------------------------------------------------------

/// Recall from the OpenClaw memory store and return the bootstrap-file list
/// with `LINTAI.md` appended (replacing any stale entry with the same name so
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

    if lint_ai_enabled(RecordingProvider::OpenClaw, root)? {
        let query = input.query.trim();
        let query = if query.is_empty() {
            DEFAULT_BOOTSTRAP_QUERY
        } else {
            query
        };
        let memories = retrieve_memories(root, query)?;
        if !memories.trim().is_empty() {
            let workspace_dir = input
                .event
                .pointer("/context/workspaceDir")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim_end_matches('/');
            files.push(json!({
                "name": INJECTED_FILE_NAME,
                "path": format!("{workspace_dir}/{INJECTED_FILE_NAME}"),
                "content": format!("# {INJECTED_FILE_NAME}\n\nLint-AI recalled memories (automatic):\n{memories}"),
                "missing": false,
            }));
        }
    }
    Ok(json!({ "bootstrapFiles": files }))
}

fn retrieve_memories(root: &Path, query: &str) -> Result<String> {
    if !memory_root(root).exists() {
        return Ok(String::new());
    }
    let mut store = open_store(root)?;
    if store.is_empty() {
        return Ok(String::new());
    }
    let results = store.query(query, DEFAULT_TOP_K * 3)?;
    let mut seen = HashSet::new();
    let mut output = String::new();
    for result in results.into_iter().take(DEFAULT_TOP_K) {
        let Some(record) = store.record_by_id(&result.doc_id) else {
            continue;
        };
        let excerpt = relevant_excerpt(&record.content, query, &result.matched_terms, MAX_EXCERPT_BYTES);
        let excerpt = excerpt.trim();
        if excerpt.is_empty() || !seen.insert(excerpt.to_string()) {
            continue;
        }
        output.push_str(&format!("\n- Source: {}\n  {}\n", record.source, excerpt));
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
            Some((role.to_string(), truncate_utf8(&text, MAX_CAPTURE_BYTES)))
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
        if content.trim_start().starts_with("<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>") {
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
                if block.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        out.push_str(text);
                        out.push('\n');
                    }
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

fn capture_outcome(
    root: &Path,
    state: &mut HookState,
    session_id: &str,
    run_id: &str,
    channel: Option<&str>,
    turns: &[(String, String)],
) -> Result<Value> {
    if already_seen(state, run_id) {
        return Ok(ack(Some("duplicate runId")));
    }
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
    let mut store = open_store(root)?;
    store.upsert(document.into_source_document()?);
    store.refresh()?;
    Ok(ack(None))
}

// ---------------------------------------------------------------------------
// agent_end — per-turn Outcome capture (idempotent on runId).
// ---------------------------------------------------------------------------

fn handle_agent_end(
    input: &OpenClawHookInput,
    root: &Path,
    state: &mut HookState,
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
    if !lint_ai_enabled(RecordingProvider::OpenClaw, root)? {
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
    let output = capture_outcome(root, state, &session_id, run_id, channel, &turns)?;
    // agent_end is also the heartbeat that keeps the session registry's
    // open/closed bookkeeping honest when session_start arrived without a
    // workspace root.
    let record = record_session(state, &session_id);
    record.open = true;
    if record.session_key.is_none() {
        record.session_key = string_field(&input.ctx, "sessionKey");
    }
    if record.agent_id.is_none() {
        record.agent_id = input.ctx.get("agentId").and_then(Value::as_str).map(str::to_string);
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// before_reset — authoritative SessionSummary capture from the full
// departing transcript (delivered before OpenClaw wipes the session).
// ---------------------------------------------------------------------------

fn handle_before_reset(
    input: &OpenClawHookInput,
    root: &Path,
    state: &mut HookState,
) -> Result<Value> {
    let session_id = session_id(input, OpenClawHookKind::BeforeReset);
    if !lint_ai_enabled(RecordingProvider::OpenClaw, root)? {
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
    let content = format!(
        "Session closed (reason: {reason}).\n\n{}",
        truncate_utf8(&transcript, MAX_CAPTURE_BYTES)
    );
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
        channel: input.ctx.get("channel").and_then(Value::as_str).map(str::to_string),
    };
    let mut store = open_store(root)?;
    store.upsert(document.into_source_document()?);
    store.refresh()?;
    let record = record_session(state, &session_id);
    record.open = false;
    record.ended_reason = Some(reason.to_string());
    Ok(ack(None))
}

// ---------------------------------------------------------------------------
// session_start / session_end — session registry bookkeeping.
// ---------------------------------------------------------------------------

fn string_field(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn handle_session_start(input: &OpenClawHookInput, state: &mut HookState) -> Result<Value> {
    let session_id = session_id(input, OpenClawHookKind::SessionStart);
    let record = record_session(state, &session_id);
    record.open = true;
    record.session_key = string_field(&input.event, "sessionKey");
    record.resumed_from = string_field(&input.event, "resumedFrom");
    record.agent_id = input.ctx.get("agentId").and_then(Value::as_str).map(str::to_string);
    Ok(ack(None))
}

fn handle_session_end(input: &OpenClawHookInput, state: &mut HookState) -> Result<Value> {
    let session_id = session_id(input, OpenClawHookKind::SessionEnd);
    let reason = string_field(&input.event, "reason");
    let next_session_id = string_field(&input.event, "nextSessionId");
    let record = record_session(state, &session_id);
    record.open = false;
    record.ended_reason = reason;
    record.next_session_id = next_session_id;
    Ok(ack(None))
}

// ---------------------------------------------------------------------------
// Shutdown — bounded pending-capture flush.
//
// All captures above write through synchronously, so this is a final
// best-effort state flush inside OpenClaw's shared two-second drain budget.
// ---------------------------------------------------------------------------

fn handle_shutdown(_state: &HookState) -> Result<Value> {
    Ok(ack(None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Isolated project root per test (unique dir, cleaned on drop via tests
    /// writing under target/tmp).
    fn test_root() -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let root = std::env::temp_dir().join(format!(
            "lint-ai-openclaw-hook-test-{}-{}",
            std::process::id(),
            id
        ));
        fs::create_dir_all(&root).unwrap();
        root
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
        store.refresh().unwrap();
    }

    fn bootstrap_input(workspace_dir: &str, query: &str, existing: Vec<Value>) -> OpenClawHookInput {
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
    fn bootstrap_injects_lintai_md_when_memories_exist() {
        let root = test_root();
        write_memory_doc(
            &root,
            "seed-1",
            "The team decided to adopt SQLite for the local session store.",
        );
        let existing = vec![json!({"name": "SKILL.md", "path": "/ws/SKILL.md", "content": "x", "missing": false})];
        let input = bootstrap_input(root.to_str().unwrap(), "local session store", existing);
        let output = handle_bootstrap(&input, &root).unwrap();
        let files = output.get("bootstrapFiles").unwrap().as_array().unwrap();
        assert_eq!(files.len(), 2);
        let injected = files.iter().find(|f| f["name"] == "LINTAI.md").unwrap();
        assert!(injected["content"].as_str().unwrap().contains("SQLite"));
        assert_eq!(injected["path"], format!("{}/LINTAI.md", root.to_str().unwrap()));
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
            first.get("bootstrapFiles").unwrap().as_array().unwrap().clone(),
        );
        let second = handle_bootstrap(&rerun_input, &root).unwrap();
        let files = second.get("bootstrapFiles").unwrap().as_array().unwrap();
        assert_eq!(
            files.iter().filter(|f| f["name"] == "LINTAI.md").count(),
            1,
            "must not duplicate the injected file"
        );
    }

    #[test]
    fn bootstrap_with_empty_memory_store_leaves_files_untouched() {
        let root = test_root();
        let existing = vec![json!({"name": "SKILL.md", "path": "/ws/SKILL.md", "content": "x", "missing": false})];
        let input = bootstrap_input(root.to_str().unwrap(), "anything", existing);
        let output = handle_bootstrap(&input, &root).unwrap();
        assert_eq!(output.get("bootstrapFiles").unwrap().as_array().unwrap().len(), 1);
        assert!(output
            .get("bootstrapFiles")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .all(|f| f["name"] != "LINTAI.md"));
    }

    #[test]
    fn bootstrap_uses_default_query_without_user_text() {
        let root = test_root();
        write_memory_doc(&root, "seed-1", "Decisions about the implemented changes were recorded after we resolved the deployment failures.");
        let input = bootstrap_input(root.to_str().unwrap(), "", vec![]);
        let output = handle_bootstrap(&input, &root).unwrap();
        let files = output.get("bootstrapFiles").unwrap().as_array().unwrap();
        assert!(files.iter().any(|f| f["name"] == "LINTAI.md"));
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
    fn agent_end_captures_outcome_and_dedupes_run_id() {
        let root = test_root();
        let mut state = HookState::default();
        let first = handle_agent_end(&agent_end_input("run-1", "sess-1"), &root, &mut state).unwrap();
        assert_eq!(first["ok"], true);
        // Retry with the same runId captures nothing new.
        let second = handle_agent_end(&agent_end_input("run-1", "sess-1"), &root, &mut state).unwrap();
        assert_eq!(second["ok"], false);
        assert_eq!(second["skipped"], "duplicate runId");
        // A new runId captures.
        let third = handle_agent_end(&agent_end_input("run-2", "sess-1"), &root, &mut state).unwrap();
        assert_eq!(third["ok"], true);

        let mut store = open_store(&root).unwrap();
        let results = store.query("deployment steps", 10).unwrap();
        let outcomes: Vec<_> = results
            .iter()
            .filter_map(|r| store.record_by_id(&r.doc_id))
            .filter(|record| record.filters.get("document_type").map(String::as_str) == Some("outcome"))
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
        let mut state = HookState::default();
        let input = OpenClawHookInput {
            event: json!({"messages": []}),
            ctx: json!({"sessionId": "s"}),
            query: String::new(),
        };
        let output = handle_agent_end(&input, &root, &mut state).unwrap();
        assert_eq!(output["skipped"], "missing runId");
    }

    #[test]
    fn before_reset_captures_session_summary_from_full_transcript() {
        let root = test_root();
        let mut state = HookState::default();
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
        let output = handle_before_reset(&input, &root, &mut state).unwrap();
        assert_eq!(output["ok"], true);

        let mut store = open_store(&root).unwrap();
        let results = store.query("reset the gateway", 10).unwrap();
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

        let record = state.sessions.get("sess-9").unwrap();
        assert!(!record.open);
        assert_eq!(record.ended_reason.as_deref(), Some("user-initiated"));
    }

    #[test]
    fn session_lifecycle_updates_registry() {
        let mut state = HookState::default();
        let start = OpenClawHookInput {
            event: json!({"sessionId": "sess-a", "sessionKey": "agent:main:webchat:sess-a", "resumedFrom": "sess-old"}),
            ctx: json!({"agentId": "agent-main"}),
            query: String::new(),
        };
        assert_eq!(handle_session_start(&start, &mut state).unwrap()["ok"], true);
        let record = state.sessions.get("sess-a").unwrap();
        assert!(record.open);
        assert_eq!(record.resumed_from.as_deref(), Some("sess-old"));
        assert_eq!(record.agent_id.as_deref(), Some("agent-main"));

        let end = OpenClawHookInput {
            event: json!({"sessionId": "sess-a", "reason": "idle-timeout", "nextSessionId": "sess-b"}),
            ctx: Value::Null,
            query: String::new(),
        };
        assert_eq!(handle_session_end(&end, &mut state).unwrap()["ok"], true);
        let record = state.sessions.get("sess-a").unwrap();
        assert!(!record.open);
        assert_eq!(record.ended_reason.as_deref(), Some("idle-timeout"));
        assert_eq!(record.next_session_id.as_deref(), Some("sess-b"));
    }

    #[test]
    fn shutdown_flushes_state_and_acks() {
        let root = test_root();
        let mut state = HookState::default();
        already_seen(&mut state, "run-7");
        save_state(&root, &state).unwrap();
        assert_eq!(handle_shutdown(&state).unwrap()["ok"], true);
        let reloaded = load_state(&root).unwrap();
        assert!(reloaded.seen_run_ids.iter().any(|id| id == "run-7"));
    }

    #[test]
    fn state_round_trip_survives_reload() {
        let root = test_root();
        let mut state = HookState::default();
        assert!(!already_seen(&mut state, "run-1"));
        assert!(already_seen(&mut state, "run-1"));
        save_state(&root, &state).unwrap();
        let reloaded = load_state(&root).unwrap();
        assert!(reloaded.seen_run_ids.iter().any(|id| id == "run-1"));
    }

    #[test]
    fn default_output_keeps_bootstrap_files_untouched_on_failure() {
        let input = bootstrap_input("/ws", "q", vec![json!({"name": "SKILL.md"})]);
        let output = default_output(OpenClawHookKind::Bootstrap, &input);
        assert_eq!(output.get("bootstrapFiles").unwrap().as_array().unwrap().len(), 1);
        let output = default_output(OpenClawHookKind::AgentEnd, &input);
        assert_eq!(output["ok"], false);
    }
}
