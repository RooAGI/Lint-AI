mod protocol;

use super::document::{ClaudeCodeDocument, ClaudeCodeDocumentType};
use crate::ids::stable_doc_id_from_source;
use crate::integrations::recall::{query_terms, relevant_excerpt, truncate_utf8};
use crate::integrations::session_recording::{
    membrick_enabled, provider_state_dir, record_event_if_enabled,
    record_transcript_usage_if_available, RecordingProvider,
};
use crate::integrations::transcript_cursor;
use crate::memory_api::MemoryService;
#[cfg(test)]
use crate::pipeline::{MemoryIndexLayout, PipelineOptions};
#[cfg(test)]
use crate::segments::SegmentRoutingStrategy;
use anyhow::{Context, Result};
use protocol::{ClaudeHookInput, ClaudeHookOutput};
use serde_json::Value;
use std::collections::{HashSet, VecDeque};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_TOP_K: usize = 5;
const DEFAULT_CONTEXT_BYTES: usize = 8_000;
const MAX_EXCERPT_BYTES: usize = 800;
const MIN_TOOL_QUERY_TERMS: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaudeHookKind {
    SessionStart,
    UserPromptSubmit,
    UserPromptExpansion,
    PreToolUse,
    PostToolUse,
    PreCompact,
    Stop,
    SessionEnd,
    SubagentStart,
    SubagentStop,
}

impl ClaudeHookKind {
    pub fn event_name(self) -> &'static str {
        match self {
            Self::SessionStart => "SessionStart",
            Self::UserPromptSubmit => "UserPromptSubmit",
            Self::UserPromptExpansion => "UserPromptExpansion",
            Self::PreToolUse => "PreToolUse",
            Self::PostToolUse => "PostToolUse",
            Self::PreCompact => "PreCompact",
            Self::Stop => "Stop",
            Self::SessionEnd => "SessionEnd",
            Self::SubagentStart => "SubagentStart",
            Self::SubagentStop => "SubagentStop",
        }
    }
}

pub fn run_hook(kind: ClaudeHookKind, fallback_root: &Path) -> Result<()> {
    let raw: Value =
        crate::integrations::read_bounded_json().context("failed to parse Claude hook input")?;
    let input: ClaudeHookInput =
        serde_json::from_value(raw).context("failed to decode Claude hook input")?;
    let root = resolve_root(&input.cwd, fallback_root)?;
    let started = Instant::now();
    let mut output = match run_with_budget(kind, input, fallback_root) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("warning: Membrick Claude hook failed open: {error:#}");
            ClaudeHookOutput::default()
        }
    };
    if matches!(
        kind,
        ClaudeHookKind::SessionStart | ClaudeHookKind::UserPromptSubmit
    ) {
        output.system_message = crate::updater::take_update_notice();
    }
    write_timing_record(
        &root,
        kind,
        started.elapsed().as_secs_f64() * 1000.0,
        &output,
    );
    serde_json::to_writer(std::io::stdout().lock(), &output)?;
    std::io::stdout().lock().write_all(b"\n")?;
    Ok(())
}

fn run_with_budget(
    kind: ClaudeHookKind,
    input: ClaudeHookInput,
    fallback_root: &Path,
) -> Result<ClaudeHookOutput> {
    let timeout_ms = crate::integrations::hook_settings::hook_timeout_ms(fallback_root);
    let fallback_root = fallback_root.to_path_buf();
    let completes_capture = matches!(
        kind,
        ClaudeHookKind::PreCompact
            | ClaudeHookKind::Stop
            | ClaudeHookKind::SessionEnd
            | ClaudeHookKind::SubagentStop
    );
    crate::integrations::hook_settings::run_with_budget(timeout_ms, completes_capture, move || {
        handle_hook(kind, input, &fallback_root)
    })
    .with_context(|| format!("Membrick {} hook", kind.event_name()))
}

fn write_timing_record(
    root: &Path,
    kind: ClaudeHookKind,
    elapsed_ms: f64,
    output: &ClaudeHookOutput,
) {
    let Some(path) = crate::integrations::hook_settings::hook_timings_path(root) else {
        return;
    };
    let operation = match kind {
        ClaudeHookKind::SessionStart
        | ClaudeHookKind::UserPromptSubmit
        | ClaudeHookKind::UserPromptExpansion
        | ClaudeHookKind::PreToolUse
        | ClaudeHookKind::PostToolUse
        | ClaudeHookKind::SubagentStart => "retrieve",
        ClaudeHookKind::PreCompact
        | ClaudeHookKind::Stop
        | ClaudeHookKind::SessionEnd
        | ClaudeHookKind::SubagentStop => "capture",
    };
    let context_bytes = output
        .hook_specific_output
        .as_ref()
        .map(|value| value.additional_context.len())
        .unwrap_or(0);
    let record = serde_json::json!({
        "event_name": kind.event_name(),
        "operation": operation,
        "elapsed_ms": elapsed_ms,
        "context_bytes": context_bytes,
    });
    let Ok(encoded) = serde_json::to_vec(&record) else {
        return;
    };
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = file.write_all(&encoded);
        let _ = file.write_all(b"\n");
    }
}

fn handle_hook(
    kind: ClaudeHookKind,
    input: ClaudeHookInput,
    fallback_root: &Path,
) -> Result<ClaudeHookOutput> {
    if input.hook_event_name != kind.event_name() {
        anyhow::bail!(
            "Claude hook event mismatch: expected {}, got {}",
            kind.event_name(),
            input.hook_event_name
        );
    }
    let root = resolve_root(&input.cwd, fallback_root)?;
    // Keep every filesystem, resident IPC, and transcript operation inside
    // run_with_budget. Otherwise a stalled resident request can bypass the
    // hook timeout before capture/retrieval even starts.
    let session_id = input.session_id.clone();
    if memory_root(&root).exists() {
        MemoryService::note_active_session_at_path(
            memory_root(&root),
            RecordingProvider::Claude.as_str(),
            &session_id,
        );
    }
    let payload = serde_json::json!({
        "hook_event_name": input.hook_event_name,
        "prompt": input.prompt,
        "tool_name": input.tool_name,
        "tool_input": input.tool_input,
        "tool_response": input.tool_response,
        "transcript_path": input.transcript_path,
        "agent_id": input.agent_id,
        "agent_type": input.agent_type,
        "turn_id": input.turn_id,
        "expansion_type": input.expansion_type,
        "command_name": input.command_name,
        "command_args": input.command_args,
        "command_source": input.command_source,
        "extra": input.extra,
    });
    if let Err(error) = record_event_if_enabled(
        RecordingProvider::Claude,
        &root,
        &session_id,
        kind.event_name(),
        payload,
    ) {
        eprintln!("warning: Membrick session recording failed open: {error:#}");
    }
    if matches!(kind, ClaudeHookKind::Stop) && !input.stop_hook_active {
        if let Err(error) = record_transcript_usage_if_available(
            RecordingProvider::Claude,
            &root,
            &session_id,
            input.transcript_path.as_deref(),
            input.turn_id.as_deref(),
            input.agent_id.as_deref(),
            input.agent_type.as_deref(),
            "claude-transcript",
        ) {
            eprintln!("warning: Claude transcript usage capture failed open: {error:#}");
        }
    }
    if !membrick_enabled(RecordingProvider::Claude, &root)? {
        return Ok(ClaudeHookOutput::default());
    }
    let output = match kind {
        ClaudeHookKind::SessionStart => retrieve(
            &root,
            kind.event_name(),
            &input.session_id,
            "decisions unresolved work failures implemented changes",
        ),
        ClaudeHookKind::UserPromptSubmit => retrieve(
            &root,
            kind.event_name(),
            &input.session_id,
            input.prompt.as_deref().unwrap_or_default(),
        ),
        ClaudeHookKind::UserPromptExpansion => {
            let query = [
                input.expansion_type.as_deref(),
                input.command_name.as_deref(),
                input.command_args.as_deref(),
                input.command_source.as_deref(),
                input.prompt.as_deref(),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(" ");
            retrieve(&root, kind.event_name(), &input.session_id, &query)
        }
        ClaudeHookKind::PostToolUse => {
            let query = tool_query(&input);
            if query_terms(&query).len() < MIN_TOOL_QUERY_TERMS {
                return Ok(ClaudeHookOutput::default());
            }
            retrieve(&root, kind.event_name(), &input.session_id, &query)
        }
        ClaudeHookKind::PreToolUse => retrieve(
            &root,
            kind.event_name(),
            &input.session_id,
            &tool_query(&input),
        ),
        ClaudeHookKind::PreCompact => capture(&root, input, ClaudeCodeDocumentType::Checkpoint),
        ClaudeHookKind::Stop if input.stop_hook_active => Ok(ClaudeHookOutput::default()),
        ClaudeHookKind::Stop => capture(&root, input, ClaudeCodeDocumentType::Outcome),
        ClaudeHookKind::SessionEnd => capture(&root, input, ClaudeCodeDocumentType::SessionSummary),
        ClaudeHookKind::SubagentStart => retrieve(
            &root,
            kind.event_name(),
            &input.session_id,
            input.prompt.as_deref().unwrap_or_default(),
        ),
        ClaudeHookKind::SubagentStop => capture(&root, input, ClaudeCodeDocumentType::Outcome),
    }?;
    if kind == ClaudeHookKind::SessionEnd && memory_root(&root).exists() {
        MemoryService::deactivate_session_at_path(
            memory_root(&root),
            RecordingProvider::Claude.as_str(),
            &session_id,
        );
    }
    Ok(output)
}

fn retrieve(
    root: &Path,
    event_name: &str,
    session_id: &str,
    query: &str,
) -> Result<ClaudeHookOutput> {
    if query.trim().is_empty() {
        return Ok(ClaudeHookOutput::default());
    }
    if event_name == "SessionStart" && !crate::integrations::mcp_tools::has_search_store(root) {
        return Ok(ClaudeHookOutput::default());
    }
    let started = std::time::Instant::now();
    let results = crate::integrations::mcp_tools::search_hook_documents(
        root,
        "claude",
        Some(session_id),
        query,
        DEFAULT_TOP_K * 3,
    );
    let _ = crate::telemetry::record_project_query(
        root,
        started.elapsed().as_millis() as u64,
        results.is_err(),
        results.as_ref().is_ok_and(|value| {
            value
                .get("results")
                .and_then(Value::as_array)
                .is_none_or(Vec::is_empty)
        }),
    );
    let results = results?;
    let selected = results
        .get("results")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        // A conflicted capture is a useful search hit for an agent explicitly
        // inspecting memory, but it is unsafe as automatic prompt context.
        .filter(|result| crate::integrations::mcp_tools::automatic_context_allowed(result))
        .take(DEFAULT_TOP_K);
    let mut seen = HashSet::new();
    let mut retrieved_memories = Vec::new();
    let mut context = String::new();
    for result in selected {
        let content = result
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let source = result
            .get("source")
            .and_then(Value::as_str)
            .unwrap_or_default();
        // Provider operational records are excluded, while captured session
        // memories with the matching provider session tag remain injectable.
        if is_operational_memory_result(result) {
            continue;
        }
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
        let normalized = relevant_excerpt(content, query, &matched_terms, MAX_EXCERPT_BYTES);
        let normalized = normalized.trim();
        if normalized.is_empty() || !seen.insert(normalized.to_string()) {
            continue;
        }
        let semantic_status = result
            .get("semantic_status")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let relation_evidence = result
            .get("relation_evidence")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();
        let entry = format!(
            "\n- Source: {}\n  Type: {}\n  Captured: {}\n  Memory status: {}\n  Relation evidence: {}\n  {}\n",
            source,
            result
                .get("document_type")
                .and_then(Value::as_str)
                .unwrap_or("memory"),
            result
                .get("created_at")
                .and_then(Value::as_str)
                .unwrap_or("unknown"),
            semantic_status,
            if relation_evidence.is_empty() {
                "none".to_string()
            } else {
                relation_evidence.join("; ")
            },
            normalized,
        );
        if context.len() + entry.len() > DEFAULT_CONTEXT_BYTES {
            break;
        }
        context.push_str(&entry);
        retrieved_memories.push(serde_json::json!({
            "source": source,
            "type": result.get("document_type").and_then(Value::as_str).unwrap_or("memory"),
            "score": result.get("score"),
            "excerpt": normalized,
        }));
    }
    let _ = crate::telemetry::record_memory_retrieval(
        root,
        "claude",
        session_id,
        query,
        started.elapsed().as_millis() as u64,
        &retrieved_memories,
    );
    if seen.is_empty() {
        return Ok(ClaudeHookOutput::default());
    }
    Ok(ClaudeHookOutput::additional_context(event_name, context))
}

fn is_operational_memory_result(result: &Value) -> bool {
    let source = result
        .get("source")
        .and_then(Value::as_str)
        .unwrap_or_default();
    crate::provider_sources::is_operational_provider_source(
        source,
        result.get("session_id").and_then(Value::as_str),
    )
}

fn capture(
    root: &Path,
    input: ClaudeHookInput,
    document_type: ClaudeCodeDocumentType,
) -> Result<ClaudeHookOutput> {
    let Some(transcript_path) = input.transcript_path.as_deref() else {
        return Ok(ClaudeHookOutput::default());
    };
    // Store only the transcript written since this session's previous
    // capture, so earlier turns are never re-sent.
    let state_dir = provider_state_dir(RecordingProvider::Claude, root);
    let slice =
        transcript_cursor::read_since_last_capture(&state_dir, &input.session_id, transcript_path)?;
    let messages = extract_messages(&slice.text);
    if messages.is_empty() {
        transcript_cursor::mark_captured(&state_dir, &input.session_id, slice.end)?;
        return Ok(ClaudeHookOutput::default());
    }
    let affected_paths = affected_paths(&input.extra);
    let content = structured_memory(&messages, document_type, &affected_paths);
    let captured_session_id = input.session_id.clone();
    let make_document =
        |event_id: String, document_type, content: String, timestamp| ClaudeCodeDocument {
            event_id,
            session_id: input.session_id.clone(),
            document_type,
            content,
            cwd: root.to_path_buf(),
            timestamp,
            command_name: input.command_name.clone(),
            command_args: input.command_args.clone(),
            affected_paths: affected_paths.clone(),
        };
    let mut documents = Vec::with_capacity(messages.len() + 2);
    for (offset, message) in messages.into_iter().enumerate() {
        // Byte offsets grow faster than message counts, so this order stays
        // unique and increasing across captures.
        let order = slice.start as usize + offset;
        let identity = message.message_id.as_deref().unwrap_or("");
        let event_id = if identity.is_empty() {
            format!("observation-{order}")
        } else {
            format!("observation-{identity}")
        };
        let mut observation = make_document(
            event_id,
            ClaudeCodeDocumentType::SessionSummary,
            format!("{}\n{}", message.role, message.content),
            message.timestamp,
        )
        .into_source_document()?;
        observation
            .filters
            .insert("document_type".into(), "session-observation".into());
        observation.filters.insert("speaker".into(), message.role);
        observation
            .filters
            .insert("observation_index".into(), order.to_string());
        observation.concept = "session-observation".into();
        observation.headings = vec!["session-observation".into()];
        documents.push(observation);
    }
    // This capture's text, compared with other sessions at project scope.
    let mut turn = make_document(
        format!("turn-{}", slice.start),
        document_type,
        content,
        Some(current_timestamp()),
    )
    .into_source_document()?;
    if let Some(project_id) = turn.filters.get("project_id").cloned() {
        turn.filters
            .insert("semantic_scope".into(), format!("project:{project_id}"));
    }
    documents.push(turn);
    let mut context = make_document(
        "whole-session".to_string(),
        ClaudeCodeDocumentType::SessionSummary,
        format!(
            "Session {} context. Observations are linked in source order.",
            input.session_id
        ),
        Some(current_timestamp()),
    )
    .into_source_document()?;
    context
        .filters
        .insert("document_type".into(), "session-context".into());
    context.concept = "session-context".into();
    context.headings = vec!["session-context".into()];
    // Link the observations captured in this batch. The server never
    // rebuilds the context from the store, so each capture stays one upsert.
    context.links = documents
        .iter()
        .map(|document| document.doc_id.clone())
        .collect();
    context.doc_id = stable_doc_id_from_source(&format!("{}::whole-session", context.source));
    context.doc_length = context.content.len();
    documents.push(context);
    crate::integrations::mcp_tools::upsert_hook_documents(
        root,
        "claude",
        &captured_session_id,
        documents,
    )?;
    transcript_cursor::mark_captured(&state_dir, &captured_session_id, slice.end)?;
    Ok(ClaudeHookOutput::default())
}

#[cfg(test)]
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

fn memory_root(root: &Path) -> PathBuf {
    crate::integrations::mcp_index::shared_memory_root(root)
}

fn resolve_root(cwd: &Path, fallback_root: &Path) -> Result<PathBuf> {
    let fallback = fallback_root.canonicalize().with_context(|| {
        format!(
            "failed to resolve Claude fallback project root {}",
            fallback_root.display()
        )
    })?;
    let candidate = if cwd.as_os_str().is_empty() {
        fallback.clone()
    } else {
        cwd.canonicalize()
            .with_context(|| format!("failed to resolve Claude project root {}", cwd.display()))?
    };
    if !candidate.starts_with(&fallback) {
        anyhow::bail!(
            "Claude hook cwd {} is outside configured project root {}",
            candidate.display(),
            fallback.display()
        );
    }
    Ok(candidate)
}

#[derive(Debug, Clone)]
struct ConversationMessage {
    role: String,
    content: String,
    timestamp: Option<String>,
    message_id: Option<String>,
}

fn structured_memory(
    messages: &VecDeque<ConversationMessage>,
    document_type: ClaudeCodeDocumentType,
    affected_paths: &[String],
) -> String {
    let mut fields = vec![
        format!("Memory type: {}", document_type.as_str()),
        "Parsed transcript text observations in source order. Credentials are redacted."
            .to_string(),
    ];
    for (index, message) in messages.iter().enumerate() {
        let content = &message.content;
        if content.trim().is_empty() {
            continue;
        }
        fields.push(format!(
            "Observation {} [{}]:\n{}",
            index + 1,
            message.role,
            content
        ));
    }
    if fields.len() == 2 {
        return String::new();
    }
    if !affected_paths.is_empty() {
        fields.push(format!("Affected paths: {}", affected_paths.join(", ")));
    }
    fields.join("\n")
}

#[cfg(test)]
fn extract_structured_memory(
    path: &Path,
    document_type: ClaudeCodeDocumentType,
    affected_paths: &[String],
) -> Result<String> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read Claude transcript {}", path.display()))?;
    Ok(structured_memory(
        &extract_messages(&text),
        document_type,
        affected_paths,
    ))
}

fn extract_messages(text: &str) -> VecDeque<ConversationMessage> {
    let mut messages = VecDeque::new();
    for line in text.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let Some(message) = value.get("message") else {
            continue;
        };
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_else(|| {
                value
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("message")
            });
        let content = message
            .get("content")
            .map(text_blocks)
            .filter(|content| !content.is_empty())
            .or_else(|| serde_json::to_string(message).ok())
            .unwrap_or_default();
        messages.push_back(ConversationMessage {
            role: role.to_string(),
            content: redact(if content.trim().is_empty() {
                "[empty message content]"
            } else {
                &content
            }),
            timestamp: value
                .get("timestamp")
                .or_else(|| message.get("timestamp"))
                .and_then(Value::as_str)
                .map(str::to_string),
            message_id: value
                .get("uuid")
                .or_else(|| message.get("id"))
                .and_then(Value::as_str)
                .map(str::to_string),
        });
    }
    messages
}

fn text_blocks(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .map(|block| {
                let kind = block
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("content");
                match kind {
                    "text" | "input_text" | "output_text" => block
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    _ => format!(
                        "[{kind}] {}",
                        serde_json::to_string_pretty(block)
                            .unwrap_or_else(|_| "[unserializable block]".to_string())
                    ),
                }
            })
            .filter(|block| !block.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn redact(text: &str) -> String {
    let mut in_private_key = false;
    text.lines()
        .map(|line| {
            let lower = line.to_ascii_lowercase();
            let begins_private_key = lower.contains("-----begin") && lower.contains("private key");
            let ends_private_key = lower.contains("-----end") && lower.contains("private key");
            let sensitive = in_private_key
                || begins_private_key
                || [
                    "api_key",
                    "apikey",
                    "api key",
                    "access_token",
                    "access token",
                    "authorization:",
                    "bearer ",
                    "password=",
                    "password:",
                    "token=",
                    "token:",
                ]
                .iter()
                .any(|marker| lower.contains(marker));
            if begins_private_key {
                in_private_key = true;
            }
            if ends_private_key {
                in_private_key = false;
            }
            if sensitive {
                "[REDACTED]"
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn affected_paths(extra: &serde_json::Map<String, Value>) -> Vec<String> {
    extra
        .get("affected_paths")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect()
}

fn tool_query(input: &ClaudeHookInput) -> String {
    let mut parts = Vec::new();
    if let Some(value) = input
        .turn_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        parts.push(value.to_string());
    }
    if let Some(value) = input
        .agent_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        parts.push(value.to_string());
    }
    if let Some(value) = input
        .agent_type
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        parts.push(value.to_string());
    }
    if let Some(value) = input
        .tool_name
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        parts.push(value.to_string());
    }
    if let Some(value) = input.tool_input.as_ref() {
        parts.push(truncate_utf8(&value.to_string(), MAX_EXCERPT_BYTES));
    }
    if let Some(value) = input.tool_response.as_ref() {
        parts.push(truncate_utf8(&value.to_string(), MAX_EXCERPT_BYTES));
    }
    parts.join(" ")
}

fn current_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default();
    chrono::DateTime::from_timestamp(seconds, 0)
        .map(|timestamp| timestamp.to_rfc3339())
        .unwrap_or_else(|| "1970-01-01T00:00:00+00:00".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrations::session_recording::set_membrick_state;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("membrick-claude-{name}-{nanos}"));
        fs::create_dir_all(&path).unwrap();
        path.canonicalize().unwrap()
    }

    fn input(root: &Path, transcript: Option<PathBuf>, event: &str) -> ClaudeHookInput {
        serde_json::from_value(serde_json::json!({
            "session_id": "session-1",
            "transcript_path": transcript,
            "cwd": root,
            "hook_event_name": event
        }))
        .unwrap()
    }

    #[test]
    fn codex_session_memories_are_not_treated_as_operational_records() {
        let codex_session = serde_json::json!({
            "source": "codex://project/session/outcome",
            "session_id": "codex-session:project:session"
        });
        let codex_operational_record = serde_json::json!({
            "source": "codex://tool-event",
            "session_id": null
        });

        assert!(!is_operational_memory_result(&codex_session));
        assert!(is_operational_memory_result(&codex_operational_record));
    }

    #[test]
    fn session_start_does_not_create_memory_store() {
        let root = temp_dir("session-start");
        let output = handle_hook(
            ClaudeHookKind::SessionStart,
            input(&root, None, "SessionStart"),
            &root,
        )
        .unwrap();

        assert!(output.hook_specific_output.is_none());
        assert!(!memory_root(&root).exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn empty_prompt_does_not_open_memory_store() {
        let root = temp_dir("empty-prompt");
        let output = handle_hook(
            ClaudeHookKind::UserPromptSubmit,
            input(&root, None, "UserPromptSubmit"),
            &root,
        )
        .unwrap();
        assert!(output.hook_specific_output.is_none());
        assert!(!memory_root(&root).exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn disabled_memory_skips_retrieval_without_affecting_hook_success() {
        let root = temp_dir("memory-disabled");
        set_membrick_state(RecordingProvider::Claude, &root, false).unwrap();
        let mut prompt = input(&root, None, "UserPromptSubmit");
        prompt.prompt = Some("retrieve prior routing decision".to_string());
        let output = handle_hook(ClaudeHookKind::UserPromptSubmit, prompt, &root).unwrap();
        assert!(output.hook_specific_output.is_none());
        assert!(!memory_root(&root).join("index.json").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn active_stop_hook_does_not_capture() {
        let root = temp_dir("active-stop");
        let transcript = root.join("transcript.jsonl");
        fs::write(
            &transcript,
            r#"{"message":{"role":"assistant","content":"Do not capture recursively"}}"#,
        )
        .unwrap();
        let mut stop = input(&root, Some(transcript), "Stop");
        stop.stop_hook_active = true;
        let output = handle_hook(ClaudeHookKind::Stop, stop, &root).unwrap();
        assert!(output.hook_specific_output.is_none());
        assert!(!memory_root(&root).exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stop_capture_persists_and_prompt_retrieves_memory() {
        let root = temp_dir("capture");
        let transcript = root.join("transcript.jsonl");
        fs::write(
            &transcript,
            concat!(
                r#"{"type":"user","message":{"role":"user","content":"Fix docker routing"}}"#,
                "\n",
                r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Implemented docker segment routing and tests"}]}}"#,
                "\n"
            ),
        )
        .unwrap();

        let stop_input = input(&root, Some(transcript.clone()), "Stop");
        handle_hook(ClaudeHookKind::Stop, stop_input.clone(), &root).unwrap();
        let mut stored = open_store(&root).unwrap();
        stored.refresh_index().unwrap();
        assert_eq!(
            stored.records().len(),
            4,
            "capture should store two observations, one outcome, and one session context"
        );
        let records_after_first_capture = stored.records().len();
        drop(stored);

        handle_hook(ClaudeHookKind::Stop, stop_input, &root).unwrap();
        let mut stored = open_store(&root).unwrap();
        stored.refresh_index().unwrap();
        assert_eq!(
            stored.records().len(),
            records_after_first_capture,
            "replayed hook must be idempotent"
        );
        assert_eq!(
            stored.memory_index_snapshot().unwrap().segment_count(),
            1,
            "the first capture lazily creates one session segment"
        );
        drop(stored);

        let mut prompt = input(&root, None, "UserPromptSubmit");
        prompt.prompt = Some("docker routing".to_string());
        let output = handle_hook(ClaudeHookKind::UserPromptSubmit, prompt, &root).unwrap();
        let context = output
            .hook_specific_output
            .expect("captured memory should be retrieved")
            .additional_context;
        assert!(context.contains("Implemented docker segment routing"));

        let mut expansion = input(&root, None, "UserPromptExpansion");
        expansion.command_name = Some("review".to_string());
        expansion.command_args = Some("docker routing".to_string());
        expansion.prompt = Some("/review docker routing".to_string());
        let output = handle_hook(ClaudeHookKind::UserPromptExpansion, expansion, &root).unwrap();
        assert!(output
            .hook_specific_output
            .unwrap()
            .additional_context
            .contains("Implemented docker segment routing"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn parses_sanitized_runtime_transcript_fixture() {
        let transcript =
            include_str!("../../../../tests/fixtures/runtime_transcripts/claude.jsonl");
        let messages = extract_messages(transcript);
        assert!(messages.len() >= 4);
        assert!(messages
            .iter()
            .any(|message| message.content.contains("tool_use")));
        assert!(messages
            .iter()
            .any(|message| message.content.contains("No rule match")));
    }

    #[test]
    fn transcript_extraction_preserves_tool_blocks_and_redacts_secret_lines() {
        let root = temp_dir("transcript");
        let transcript = root.join("transcript.jsonl");
        fs::write(
            &transcript,
            concat!(
                r#"{"message":{"role":"assistant","content":[{"type":"tool_use","name":"Bash","input":{"command":"echo safe"}},{"type":"text","text":"api_key=secret\nFinished work"}]}}"#,
                "\n"
            ),
        )
        .unwrap();

        let content =
            extract_structured_memory(&transcript, ClaudeCodeDocumentType::Outcome, &[]).unwrap();
        assert!(content.contains("Bash"));
        assert!(content.contains("echo safe"));
        assert!(!content.contains("secret"));
        assert!(content.contains("[REDACTED]"));
        assert!(content.contains("Finished work"));
        assert!(content.contains("Memory type: outcome"));
        assert!(content.contains("Observation 1 [assistant]"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn transcript_extraction_scans_past_oversized_jsonl_record() {
        let root = temp_dir("oversized-transcript");
        let transcript = root.join("transcript.jsonl");
        let useful = r#"{"message":{"role":"user","content":"Remember cobalt routing"}}"#;
        let oversized = serde_json::json!({
            "message": {
                "role": "user",
                "content": [{ "type": "tool_result", "content": "x".repeat(96 * 1024) }]
            }
        });
        fs::write(&transcript, format!("{useful}\n{oversized}\n")).unwrap();
        let content =
            extract_structured_memory(&transcript, ClaudeCodeDocumentType::Outcome, &[]).unwrap();
        assert!(content.contains("Remember cobalt routing"));
        assert!(content.contains(&"x".repeat(96 * 1024)));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn redacts_auth_tokens_and_private_key_blocks() {
        let content = redact(concat!(
            "Authorization: Bearer secret\n",
            "-----BEGIN RSA PRIVATE KEY-----\n",
            "secret-key-material\n",
            "-----END RSA PRIVATE KEY-----\n",
            "safe line"
        ));
        assert!(!content.contains("secret"));
        assert_eq!(content.lines().last(), Some("safe line"));
    }

    #[test]
    fn relevant_excerpt_selects_matching_lines_and_enforces_byte_limit() {
        let content = format!(
            "Memory type: outcome\nRequest: update unrelated documentation\nResult: {}\nNoise: {}",
            "Implemented LocalDistinctiveness routing for IndexStore",
            "x".repeat(2_000)
        );
        let excerpt = relevant_excerpt(&content, "IndexStore routing", &[], MAX_EXCERPT_BYTES);
        assert!(excerpt.contains("LocalDistinctiveness routing"));
        assert!(!excerpt.contains("unrelated documentation"));
        assert!(excerpt.len() <= MAX_EXCERPT_BYTES + 3);
    }

    #[test]
    fn compact_stop_and_session_end_capture_distinct_document_types() {
        let root = temp_dir("capture-types");
        let transcript = root.join("transcript.jsonl");
        fs::write(
            &transcript,
            concat!(
                r#"{"type":"assistant","message":{"role":"assistant","content":"Completed durable routing work"}}"#,
                "\n"
            ),
        )
        .unwrap();

        for (index, (kind, event)) in [
            (ClaudeHookKind::PreCompact, "PreCompact"),
            (ClaudeHookKind::Stop, "Stop"),
            (ClaudeHookKind::SessionEnd, "SessionEnd"),
        ]
        .into_iter()
        .enumerate()
        {
            if index > 0 {
                append_line(
                    &transcript,
                    &format!(
                        r#"{{"type":"assistant","message":{{"role":"assistant","content":"Routing step {index}"}}}}"#
                    ),
                );
            }
            handle_hook(kind, input(&root, Some(transcript.clone()), event), &root).unwrap();
        }

        let store = open_store(&root).unwrap();
        let document_types = store
            .records()
            .into_iter()
            .filter_map(|record| record.filters.get("document_type"))
            .cloned()
            .collect::<HashSet<_>>();
        assert_eq!(
            document_types,
            HashSet::from([
                "checkpoint".to_string(),
                "outcome".to_string(),
                "session-summary".to_string(),
                "session-context".to_string(),
                "session-observation".to_string(),
            ])
        );
        assert_eq!(
            store.records().len(),
            7,
            "three observations, three turn documents, one session context"
        );

        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn post_tool_use_retrieves_relevant_memory_by_tool_name() {
        let root = temp_dir("post-tool-use");
        let transcript = root.join("transcript.jsonl");
        fs::write(
            &transcript,
            concat!(
                r#"{"type":"user","message":{"role":"user","content":"Run the search tests"}}"#,
                "\n",
                r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Completed search index integration tests"}]}}"#,
                "\n"
            ),
        )
        .unwrap();

        handle_hook(
            ClaudeHookKind::Stop,
            input(&root, Some(transcript), "Stop"),
            &root,
        )
        .unwrap();

        let mut post_tool = input(&root, None, "PostToolUse");
        post_tool.tool_name = Some("Bash".to_string());
        post_tool.tool_input = Some(serde_json::json!({"command": "cargo test search"}));
        let output = handle_hook(ClaudeHookKind::PostToolUse, post_tool, &root).unwrap();
        assert!(output.hook_specific_output.is_some());
        let context = output.hook_specific_output.unwrap().additional_context;
        assert!(context.contains("search"));
        assert!(context.contains("Memory status:"));
        assert!(context.contains("Relation evidence:"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn subagent_start_retrieves_memory_by_prompt() {
        let root = temp_dir("subagent-start");
        let transcript = root.join("transcript.jsonl");
        fs::write(
            &transcript,
            concat!(
                r#"{"type":"user","message":{"role":"user","content":"Review the auth module"}}"#,
                "\n",
                r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Auth module uses JWT with RS256 signing"}]}}"#,
                "\n"
            ),
        )
        .unwrap();

        handle_hook(
            ClaudeHookKind::Stop,
            input(&root, Some(transcript), "Stop"),
            &root,
        )
        .unwrap();

        let mut subagent = input(&root, None, "SubagentStart");
        subagent.prompt = Some("auth module JWT signing".to_string());
        let output = handle_hook(ClaudeHookKind::SubagentStart, subagent, &root).unwrap();
        let context = output
            .hook_specific_output
            .expect("relevant memory should be retrieved")
            .additional_context;
        assert!(context.contains("JWT"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn subagent_stop_captures_outcome_and_is_idempotent() {
        let root = temp_dir("subagent-stop");
        let transcript = root.join("transcript.jsonl");
        fs::write(
            &transcript,
            concat!(
                r#"{"type":"assistant","message":{"role":"assistant","content":"Delegated auth review complete"}}"#,
                "\n"
            ),
        )
        .unwrap();

        let stop_input = input(&root, Some(transcript.clone()), "SubagentStop");
        handle_hook(ClaudeHookKind::SubagentStop, stop_input.clone(), &root).unwrap();
        let records_after_first_capture = open_store(&root).unwrap().records().len();
        assert_eq!(records_after_first_capture, 3);
        handle_hook(ClaudeHookKind::SubagentStop, stop_input, &root).unwrap();

        let store = open_store(&root).unwrap();
        assert_eq!(
            store.records().len(),
            records_after_first_capture,
            "replayed subagent-stop must be idempotent"
        );
        let doc_types = store
            .records()
            .into_iter()
            .filter_map(|r| r.filters.get("document_type").cloned())
            .collect::<HashSet<_>>();
        assert_eq!(
            doc_types,
            HashSet::from([
                "outcome".to_string(),
                "session-context".to_string(),
                "session-observation".to_string(),
            ])
        );
        fs::remove_dir_all(root).unwrap();
    }

    fn append_line(path: &Path, line: &str) {
        use std::io::Write as _;
        let mut file = fs::OpenOptions::new().append(true).open(path).unwrap();
        writeln!(file, "{line}").unwrap();
    }

    #[test]
    fn each_stop_stores_only_the_transcript_since_the_previous_stop() {
        let root = temp_dir("incremental");
        let transcript = root.join("transcript.jsonl");
        fs::write(
            &transcript,
            concat!(
                r#"{"type":"user","message":{"role":"user","content":"Choose the cobalt cache"}}"#,
                "\n"
            ),
        )
        .unwrap();
        let stop = input(&root, Some(transcript.clone()), "Stop");
        handle_hook(ClaudeHookKind::Stop, stop.clone(), &root).unwrap();
        append_line(
            &transcript,
            r#"{"type":"assistant","message":{"role":"assistant","content":"Switched to the amber cache"}}"#,
        );
        handle_hook(ClaudeHookKind::Stop, stop, &root).unwrap();

        let store = open_store(&root).unwrap();
        let records = store.source_documents();
        let of_type = |kind: &str| {
            records
                .iter()
                .filter(|record| {
                    record.filters.get("document_type").map(String::as_str) == Some(kind)
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(of_type("session-observation").len(), 2);
        let outcomes = of_type("outcome");
        assert_eq!(outcomes.len(), 2, "one outcome per stop");
        let second = outcomes
            .iter()
            .find(|record| record.content.contains("amber"))
            .expect("second stop outcome");
        assert!(
            !second.content.contains("cobalt"),
            "second stop must not repeat the first turn"
        );
        let contexts = of_type("session-context");
        assert_eq!(contexts.len(), 1, "one session-wide context");
        assert!(!contexts[0].content.contains("cobalt"));
        // The context links the observations of the latest capture.
        let latest = of_type("session-observation")
            .into_iter()
            .find(|record| record.content.contains("amber"))
            .expect("second stop observation");
        assert!(contexts[0].links.contains(&latest.doc_id));
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn shrunk_transcript_is_captured_again_from_the_start() {
        let root = temp_dir("restart");
        let transcript = root.join("transcript.jsonl");
        fs::write(
            &transcript,
            concat!(
                r#"{"type":"user","message":{"role":"user","content":"First version of a long request"}}"#,
                "\n",
                r#"{"type":"assistant","message":{"role":"assistant","content":"First version of a long answer"}}"#,
                "\n"
            ),
        )
        .unwrap();
        let stop = input(&root, Some(transcript.clone()), "Stop");
        handle_hook(ClaudeHookKind::Stop, stop.clone(), &root).unwrap();
        fs::write(
            &transcript,
            concat!(
                r#"{"type":"user","message":{"role":"user","content":"Short"}}"#,
                "\n"
            ),
        )
        .unwrap();
        handle_hook(ClaudeHookKind::Stop, stop, &root).unwrap();

        let store = open_store(&root).unwrap();
        let observations = store
            .records()
            .into_iter()
            .filter(|record| {
                record.filters.get("document_type").map(String::as_str)
                    == Some("session-observation")
            })
            .collect::<Vec<_>>();
        // Captures are upserts only: the rewritten first line replaces the
        // observation at the same position, and later ones remain as history.
        assert!(observations
            .iter()
            .any(|record| record.content.contains("Short")));
        assert!(!observations
            .iter()
            .any(|record| record.content.contains("long request")));
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }
}
