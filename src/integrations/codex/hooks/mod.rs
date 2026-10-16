pub(crate) mod protocol;

use super::document::{CodexDocument, CodexDocumentType};
use crate::ids::stable_doc_id_from_source;
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
use protocol::{CodexHookInput, CodexHookOutput};
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
pub enum CodexHookKind {
    SessionStart,
    UserPromptSubmit,
    PreToolUse,
    PostToolUse,
    UserPromptExpansion,
    PreCompact,
    PostCompact,
    Stop,
    SessionEnd,
    SubagentStart,
    SubagentStop,
}

impl CodexHookKind {
    pub fn event_name(self) -> &'static str {
        match self {
            Self::SessionStart => "SessionStart",
            Self::UserPromptSubmit => "UserPromptSubmit",
            Self::PreToolUse => "PreToolUse",
            Self::PostToolUse => "PostToolUse",
            Self::UserPromptExpansion => "UserPromptExpansion",
            Self::PreCompact => "PreCompact",
            Self::PostCompact => "PostCompact",
            Self::Stop => "Stop",
            Self::SessionEnd => "SessionEnd",
            Self::SubagentStart => "SubagentStart",
            Self::SubagentStop => "SubagentStop",
        }
    }
}

pub fn run_hook(kind: CodexHookKind, fallback_root: &Path) -> Result<()> {
    let raw: Value =
        crate::integrations::read_bounded_json().context("failed to parse Codex hook input")?;
    let input: CodexHookInput =
        serde_json::from_value(raw).context("failed to decode Codex hook input")?;
    let root = resolve_root(&input.cwd, fallback_root)?;
    let started = Instant::now();
    let mut output = match run_with_budget(kind, input, fallback_root) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("warning: Membrick Codex hook failed open: {error:#}");
            CodexHookOutput::default()
        }
    };
    if matches!(
        kind,
        CodexHookKind::SessionStart | CodexHookKind::UserPromptSubmit
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
    kind: CodexHookKind,
    input: CodexHookInput,
    fallback_root: &Path,
) -> Result<CodexHookOutput> {
    let timeout_ms = crate::integrations::hook_settings::hook_timeout_ms(fallback_root);
    let fallback_root = fallback_root.to_path_buf();
    let completes_capture = matches!(
        kind,
        CodexHookKind::PreCompact
            | CodexHookKind::PostCompact
            | CodexHookKind::Stop
            | CodexHookKind::SessionEnd
            | CodexHookKind::SubagentStop
    );
    crate::integrations::hook_settings::run_with_budget(timeout_ms, completes_capture, move || {
        handle_hook(kind, input, &fallback_root)
    })
    .with_context(|| format!("Membrick {} hook", kind.event_name()))
}

fn write_timing_record(
    root: &Path,
    kind: CodexHookKind,
    elapsed_ms: f64,
    output: &CodexHookOutput,
) {
    let Some(path) = crate::integrations::hook_settings::hook_timings_path(root) else {
        return;
    };
    let operation = match kind {
        CodexHookKind::SessionStart
        | CodexHookKind::UserPromptSubmit
        | CodexHookKind::PreToolUse
        | CodexHookKind::PostToolUse
        | CodexHookKind::UserPromptExpansion
        | CodexHookKind::SubagentStart => "retrieve",
        CodexHookKind::PreCompact
        | CodexHookKind::PostCompact
        | CodexHookKind::Stop
        | CodexHookKind::SessionEnd
        | CodexHookKind::SubagentStop => "capture",
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
    kind: CodexHookKind,
    input: CodexHookInput,
    fallback_root: &Path,
) -> Result<CodexHookOutput> {
    if input.hook_event_name != kind.event_name() {
        anyhow::bail!(
            "Codex hook event mismatch: expected {}, got {}",
            kind.event_name(),
            input.hook_event_name
        );
    }
    let root = resolve_root(&input.cwd, fallback_root)?;
    // Keep every filesystem, resident IPC, and transcript operation inside
    // run_with_budget so a stalled resident request cannot bypass the hook
    // timeout before capture or retrieval starts.
    let session_id = input.session_id.clone();
    if memory_root(&root).exists() {
        MemoryService::note_active_session_at_path(
            memory_root(&root),
            RecordingProvider::Codex.as_str(),
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
        "permission_mode": input.permission_mode,
        "trigger": input.trigger,
        "reason": input.reason,
        "extra": input.extra,
    });
    if let Err(error) = record_event_if_enabled(
        RecordingProvider::Codex,
        &root,
        &session_id,
        kind.event_name(),
        payload,
    ) {
        eprintln!("warning: Membrick session recording failed open: {error:#}");
    }
    if matches!(kind, CodexHookKind::Stop) && !input.stop_hook_active {
        if let Err(error) = record_transcript_usage_if_available(
            RecordingProvider::Codex,
            &root,
            &session_id,
            input.transcript_path.as_deref(),
            input.turn_id.as_deref(),
            input.agent_id.as_deref(),
            input.agent_type.as_deref(),
            "codex-transcript",
        ) {
            eprintln!("warning: Codex transcript usage capture failed open: {error:#}");
        }
    }
    if !membrick_enabled(RecordingProvider::Codex, &root)? {
        return Ok(CodexHookOutput::default());
    }
    let output = match kind {
        CodexHookKind::SessionStart => retrieve(
            &root,
            kind.event_name(),
            &input.session_id,
            "decisions unresolved work failures implemented changes",
        ),
        CodexHookKind::UserPromptSubmit => retrieve(
            &root,
            kind.event_name(),
            &input.session_id,
            input.prompt.as_deref().unwrap_or_default(),
        ),
        CodexHookKind::PreToolUse => retrieve(
            &root,
            kind.event_name(),
            &input.session_id,
            &tool_query(&input),
        ),
        CodexHookKind::PostToolUse => {
            let query = tool_query(&input);
            if query_terms(&query).len() < MIN_TOOL_QUERY_TERMS {
                return Ok(CodexHookOutput::default());
            }
            retrieve(&root, kind.event_name(), &input.session_id, &query)
        }
        CodexHookKind::UserPromptExpansion => {
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
        CodexHookKind::PreCompact => capture(&root, input, CodexDocumentType::Checkpoint),
        CodexHookKind::PostCompact => capture(&root, input, CodexDocumentType::Checkpoint),
        CodexHookKind::Stop if input.stop_hook_active => Ok(CodexHookOutput::default()),
        CodexHookKind::Stop => capture(&root, input, CodexDocumentType::Outcome),
        // Codex limits SessionEnd hooks to three seconds. A fresh MemoryService
        // open and durable index refresh can exceed that; Stop already captures
        // the completed turn, so avoid starting a write that Codex may kill.
        CodexHookKind::SessionEnd => Ok(CodexHookOutput::default()),
        CodexHookKind::SubagentStart => retrieve(
            &root,
            kind.event_name(),
            &input.session_id,
            input.prompt.as_deref().unwrap_or_default(),
        ),
        CodexHookKind::SubagentStop => capture(&root, input, CodexDocumentType::Outcome),
    }?;
    if kind == CodexHookKind::SessionEnd && memory_root(&root).exists() {
        MemoryService::deactivate_session_at_path(
            memory_root(&root),
            RecordingProvider::Codex.as_str(),
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
) -> Result<CodexHookOutput> {
    if query.trim().is_empty() {
        return Ok(CodexHookOutput::default());
    }
    if event_name == "SessionStart" && !crate::integrations::mcp_tools::has_search_store(root) {
        return Ok(CodexHookOutput::default());
    }
    let started = std::time::Instant::now();
    let results = crate::integrations::mcp_tools::search_hook_documents(
        root,
        "codex",
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
        // Keep ambiguous captures available to explicit search, but don't
        // silently promote one of their conflicting claims into prompt data.
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
        let normalized = relevant_excerpt(content, query, &matched_terms);
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
        "codex",
        session_id,
        query,
        started.elapsed().as_millis() as u64,
        &retrieved_memories,
    );
    if seen.is_empty() {
        return Ok(CodexHookOutput::default());
    }
    Ok(CodexHookOutput::additional_context(event_name, context))
}

fn tool_query(input: &CodexHookInput) -> String {
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
    if let Some(value) = input
        .permission_mode
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        parts.push(value.to_string());
    }
    if let Some(value) = input
        .trigger
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        parts.push(value.to_string());
    }
    if let Some(value) = input
        .reason
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

fn relevant_excerpt(content: &str, query: &str, matched_terms: &[String]) -> String {
    let mut terms = query_terms(query);
    terms.extend(matched_terms.iter().flat_map(|term| query_terms(term)));
    let mut lines = content
        .lines()
        .enumerate()
        .map(|(index, line)| {
            let lower = line.to_ascii_lowercase();
            let score = terms.iter().filter(|term| lower.contains(*term)).count();
            (index, score, line.trim())
        })
        .filter(|(_, _, line)| !line.is_empty())
        .collect::<Vec<_>>();
    lines.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));

    let mut selected = lines
        .iter()
        .filter(|(_, score, _)| *score > 0)
        .take(3)
        .cloned()
        .collect::<Vec<_>>();
    if selected.is_empty() {
        selected.extend(lines.into_iter().take(2));
    }
    selected.sort_by_key(|(index, _, _)| *index);
    truncate_utf8(
        &selected
            .into_iter()
            .map(|(_, _, line)| line)
            .collect::<Vec<_>>()
            .join("\n"),
        MAX_EXCERPT_BYTES,
    )
}

fn query_terms(value: &str) -> HashSet<String> {
    const STOP_WORDS: &[&str] = &[
        "about", "after", "again", "also", "and", "are", "before", "from", "have", "into", "our",
        "that", "the", "their", "this", "was", "what", "when", "which", "with", "without",
    ];
    value
        .split(|character: char| {
            !character.is_alphanumeric() && character != '-' && character != '_'
        })
        .map(str::to_ascii_lowercase)
        .filter(|term| term.len() >= 3 && !STOP_WORDS.contains(&term.as_str()))
        .collect()
}

fn truncate_utf8(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", value[..end].trim_end())
}

fn capture(
    root: &Path,
    input: CodexHookInput,
    document_type: CodexDocumentType,
) -> Result<CodexHookOutput> {
    let Some(transcript_path) = transcript_path(&input, root) else {
        return Ok(CodexHookOutput::default());
    };
    if crate::integrations::mcp_tools::resident_backend_enabled() {
        let mut capture_input = serde_json::to_value(&input)?;
        capture_input["transcript_path"] = serde_json::to_value(transcript_path.canonicalize()?)?;
        capture_input["cwd"] = serde_json::to_value(root.canonicalize()?)?;
        crate::integrations::mcp_tools::submit_transcript_capture(
            root,
            "codex",
            document_type.as_str(),
            capture_input,
        )?;
        return Ok(CodexHookOutput::default());
    }
    let state_dir = provider_state_dir(RecordingProvider::Codex, root);
    transcript_cursor::replay_batches(&state_dir, &input.session_id, &transcript_path, |slice| {
        capture_slice(root, &input, document_type, slice)
    })?;
    Ok(CodexHookOutput::default())
}

fn capture_slice(
    root: &Path,
    input: &CodexHookInput,
    document_type: CodexDocumentType,
    slice: &transcript_cursor::TranscriptSlice,
) -> Result<()> {
    let documents = build_capture_documents(root, input, document_type, slice)?;
    if !documents.is_empty() {
        crate::integrations::mcp_tools::upsert_hook_documents(
            root,
            "codex",
            &input.session_id,
            documents,
        )?;
    }
    Ok(())
}

pub(crate) fn capture_documents_from_request(
    root: &Path,
    input: &Value,
    document_type: &str,
    slice: &transcript_cursor::TranscriptSlice,
) -> Result<Vec<crate::SourceDocument>> {
    let document_type = match document_type {
        "checkpoint" => CodexDocumentType::Checkpoint,
        "outcome" => CodexDocumentType::Outcome,
        "session-summary" => CodexDocumentType::SessionSummary,
        _ => anyhow::bail!("invalid capture document type"),
    };
    build_capture_documents(
        root,
        &serde_json::from_value(input.clone())?,
        document_type,
        slice,
    )
}

fn build_capture_documents(
    root: &Path,
    input: &CodexHookInput,
    document_type: CodexDocumentType,
    slice: &transcript_cursor::TranscriptSlice,
) -> Result<Vec<crate::SourceDocument>> {
    let messages = extract_messages(&slice.text);
    if messages.is_empty() {
        return Ok(Vec::new());
    }
    // A hook's turn ID describes its current turn, not every older turn in
    // an initial backfill. Prefer transcript attribution and only use the hook
    // fallback for a capture that contains at most one user message.
    let capture_turn_id = if messages.iter().all(|message| message.turn_id.is_none())
        && messages
            .iter()
            .filter(|message| message.role == "user")
            .count()
            <= 1
    {
        input.turn_id.clone()
    } else {
        messages
            .front()
            .and_then(|message| message.turn_id.clone())
            .filter(|id| {
                messages
                    .iter()
                    .all(|message| message.turn_id.as_ref() == Some(id))
            })
    };
    let affected_paths = affected_paths(&input.extra);
    let content = structured_memory(&messages, document_type, &affected_paths);
    let make_document =
        |event_id: String, document_type, content: String, timestamp| CodexDocument {
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
    for message in messages {
        // The actual JSONL line position stays stable across batch boundaries.
        let order = slice.start as usize + message.source_offset;
        let turn_id = message.turn_id.as_deref().or(capture_turn_id.as_deref());
        let identity = message.message_id.as_deref().unwrap_or("");
        let event_id = if identity.is_empty() {
            format!("observation-{order}")
        } else {
            format!("observation-{identity}")
        };
        let mut observation = make_document(
            event_id,
            CodexDocumentType::SessionSummary,
            format!("{}\n{}", message.role, message.content),
            message.timestamp,
        )
        .into_source_document()?;
        transcript_cursor::link_provider_turn(&mut observation, turn_id);
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
    transcript_cursor::link_provider_turn(&mut turn, capture_turn_id.as_deref());
    if let Some(project_id) = turn.filters.get("project_id").cloned() {
        turn.filters
            .insert("semantic_scope".into(), format!("project:{project_id}"));
    }
    documents.push(turn);
    let mut context = make_document(
        "whole-session".to_string(),
        CodexDocumentType::SessionSummary,
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
    Ok(documents)
}

fn transcript_path(input: &CodexHookInput, project_root: &Path) -> Option<PathBuf> {
    if let Some(path) = input.transcript_path.as_ref().filter(|path| path.exists()) {
        return Some(path.clone());
    }
    let settings = std::fs::read_to_string(project_root.join(".membrick/settings.toml")).ok()?;
    let settings = settings.parse::<toml::Value>().ok()?;
    let transcript_dir = settings
        .get("hooks")?
        .get("codex_transcript_dir")?
        .as_str()?;
    let transcript_dir = PathBuf::from(transcript_dir);
    let transcript_dir = if transcript_dir.is_absolute() {
        transcript_dir
    } else {
        project_root.join(transcript_dir)
    };
    find_session_transcript(&transcript_dir, &input.session_id)
}

fn find_session_transcript(sessions_root: &Path, session_id: &str) -> Option<PathBuf> {
    let expected_suffix = format!("-{session_id}.jsonl");
    let mut pending = VecDeque::from([sessions_root.to_path_buf()]);
    let mut visited = 0usize;
    while let Some(directory) = pending.pop_front() {
        let entries = std::fs::read_dir(directory).ok()?;
        for entry in entries.flatten() {
            visited += 1;
            if visited > 10_000 {
                return None;
            }
            let path = entry.path();
            if path.is_dir() {
                pending.push_back(path);
            } else if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with("rollout-") && name.ends_with(&expected_suffix)
                })
            {
                return Some(path);
            }
        }
    }
    None
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
            "failed to resolve Codex fallback project root {}",
            fallback_root.display()
        )
    })?;
    let candidate = if cwd.as_os_str().is_empty() {
        fallback.clone()
    } else {
        cwd.canonicalize()
            .with_context(|| format!("failed to resolve Codex project root {}", cwd.display()))?
    };
    if !candidate.starts_with(&fallback) {
        anyhow::bail!(
            "Codex hook cwd {} is outside configured project root {}",
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
    turn_id: Option<String>,
    source_offset: usize,
}

fn structured_memory(
    messages: &VecDeque<ConversationMessage>,
    document_type: CodexDocumentType,
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
    document_type: CodexDocumentType,
    affected_paths: &[String],
) -> Result<String> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read Codex transcript {}", path.display()))?;
    Ok(structured_memory(
        &extract_messages(&text),
        document_type,
        affected_paths,
    ))
}

fn extract_messages(text: &str) -> VecDeque<ConversationMessage> {
    let mut messages = VecDeque::new();
    let mut source_offset = 0;
    let mut context_turn_id = None;
    for line in text.split_inclusive('\n') {
        let line_start = source_offset;
        source_offset += line.len();
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if value.get("type").and_then(Value::as_str) == Some("turn_context") {
            context_turn_id = transcript_cursor::transcript_turn_id(&value).map(str::to_owned);
        }
        let message = value.get("message").or_else(|| {
            value
                .get("payload")
                .filter(|_| value.get("type").and_then(Value::as_str) == Some("response_item"))
        });
        let Some(message) = message else {
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
            turn_id: transcript_cursor::transcript_turn_id(&value)
                .map(str::to_owned)
                .or_else(|| context_turn_id.clone()),
            source_offset: line_start,
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
                || crate::integrations::session_recording::contains_credential_material(line)
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

    fn input(root: &Path, transcript: Option<PathBuf>, event: &str) -> CodexHookInput {
        serde_json::from_value(serde_json::json!({
            "session_id": "session-1",
            "transcript_path": transcript,
            "cwd": root,
            "hook_event_name": event
        }))
        .unwrap()
    }

    #[test]
    fn batch_replay_persists_every_codex_observation_and_checkpoints_each_batch() {
        let root = temp_dir("batch-replay");
        let transcript = root.join("transcript.jsonl");
        let rows = [
            ("turn-1", "user", "First full request"),
            ("turn-1", "assistant", "First full answer"),
            ("turn-2", "user", "Second full request"),
        ];
        let contents = rows
            .iter()
            .map(|(turn_id, role, content)| {
                serde_json::json!({
                    "turn_id": turn_id,
                    "id": format!("{turn_id}-{role}"),
                    "message": {"role": role, "content": content}
                })
                .to_string()
                    + "\n"
            })
            .collect::<String>();
        fs::write(&transcript, &contents).unwrap();
        let hook_input = input(&root, Some(transcript.clone()), "Stop");
        let state_dir = provider_state_dir(RecordingProvider::Codex, &root);

        let batches = transcript_cursor::replay_batches_with_limit(
            &state_dir,
            &hook_input.session_id,
            &transcript,
            1,
            |slice| capture_slice(&root, &hook_input, CodexDocumentType::Outcome, slice),
        )
        .unwrap();

        assert_eq!(batches, rows.len());
        assert_eq!(
            transcript_cursor::load_offset(&state_dir, &hook_input.session_id),
            contents.len() as u64
        );
        let store = open_store(&root).unwrap();
        let observations = store
            .source_documents()
            .into_iter()
            .filter(|doc| {
                doc.filters.get("document_type").map(String::as_str) == Some("session-observation")
            })
            .collect::<Vec<_>>();
        assert_eq!(observations.len(), rows.len());
        for (_, _, content) in rows {
            assert!(observations.iter().any(|doc| doc.content.contains(content)));
        }
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn finds_persisted_codex_session_transcript_by_session_id() {
        let root = temp_dir("session-transcript");
        let sessions = root.join("sessions/2026/08/03");
        fs::create_dir_all(&sessions).unwrap();
        let transcript = sessions.join("rollout-2026-08-03T12-00-00-session-1.jsonl");
        fs::write(&transcript, "{}\n").unwrap();

        assert_eq!(
            find_session_transcript(&root.join("sessions"), "session-1"),
            Some(transcript)
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn provider_turn_id_preserves_backfill_attribution() {
        let root = temp_dir("turn-backfill");
        let transcript = root.join("transcript.jsonl");
        fs::write(&transcript, concat!(
            "{\"turn_id\":\"earlier\",\"message\":{\"role\":\"user\",\"content\":\"Earlier request\"}}\n",
            "{\"turn_id\":\"latest\",\"message\":{\"role\":\"user\",\"content\":\"Latest request\"}}\n"
        )).unwrap();
        let mut stop = input(&root, Some(transcript), "Stop");
        stop.turn_id = Some("latest".into());
        handle_hook(CodexHookKind::Stop, stop, &root).unwrap();
        let store = open_store(&root).unwrap();
        let documents = store.source_documents();
        for (text, expected) in [("Earlier request", "earlier"), ("Latest request", "latest")] {
            let doc = documents
                .iter()
                .find(|doc| {
                    doc.filters.get("document_type").map(String::as_str)
                        == Some("session-observation")
                        && doc.content.contains(text)
                })
                .unwrap();
            assert_eq!(
                doc.filters.get("turn_id").map(String::as_str),
                Some(expected)
            );
        }
        let capture = documents
            .iter()
            .find(|doc| doc.filters.get("document_type").map(String::as_str) == Some("outcome"))
            .unwrap();
        assert!(
            !capture.filters.contains_key("turn_id"),
            "a capture spanning turns must not be mislabeled as the latest turn"
        );
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn provider_turn_id_replay_keeps_message_ids_across_capture_boundaries() {
        let root = temp_dir("turn-replay");
        let transcript = root.join("transcript.jsonl");
        fs::write(
            &transcript,
            "{\"turn_id\":\"turn-a\",\"message\":{\"role\":\"user\",\"content\":\"A request\"}}\n",
        )
        .unwrap();
        let mut stop = input(&root, Some(transcript.clone()), "Stop");
        stop.turn_id = Some("turn-a".into());
        handle_hook(CodexHookKind::Stop, stop.clone(), &root).unwrap();
        append_line(
            &transcript,
            r#"{"turn_id":"turn-a","message":{"role":"assistant","content":"An answer"}}"#,
        );
        handle_hook(CodexHookKind::Stop, stop.clone(), &root).unwrap();
        let before = {
            let store = open_store(&root).unwrap();
            store
                .source_documents()
                .into_iter()
                .filter(|doc| {
                    doc.filters.get("document_type").map(String::as_str)
                        == Some("session-observation")
                })
                .map(|doc| doc.doc_id.clone())
                .collect::<HashSet<_>>()
        };
        fs::remove_dir_all(provider_state_dir(RecordingProvider::Codex, &root)).unwrap();
        handle_hook(CodexHookKind::Stop, stop, &root).unwrap();
        let store = open_store(&root).unwrap();
        let after = store
            .source_documents()
            .into_iter()
            .filter(|doc| {
                doc.filters.get("document_type").map(String::as_str) == Some("session-observation")
            })
            .map(|doc| doc.doc_id.clone())
            .collect::<HashSet<_>>();
        assert_eq!(before.len(), 2);
        assert_eq!(
            before, after,
            "replaying with a missing cursor must upsert the same observations"
        );
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn provider_turn_id_links_capture_documents_without_overwriting_other_turns() {
        let root = temp_dir("provider-turn-identity");
        let transcript = root.join("transcript.jsonl");
        fs::write(
            &transcript,
            "{\"message\":{\"role\":\"assistant\",\"content\":\"Turn result\"}}\n",
        )
        .unwrap();
        for turn_id in ["turn-a", "turn-b"] {
            let state_dir = provider_state_dir(RecordingProvider::Codex, &root);
            if state_dir.exists() {
                fs::remove_dir_all(&state_dir).unwrap();
            }
            let mut stop = input(&root, Some(transcript.clone()), "Stop");
            stop.turn_id = Some(turn_id.into());
            handle_hook(CodexHookKind::Stop, stop, &root).unwrap();
        }
        let store = open_store(&root).unwrap();
        let documents = store.source_documents();
        let outcomes = documents
            .iter()
            .filter(|doc| doc.filters.get("document_type").map(String::as_str) == Some("outcome"))
            .collect::<Vec<_>>();
        assert_eq!(
            outcomes.len(),
            2,
            "distinct provider turns must have distinct doc_ids"
        );
        for turn_id in ["turn-a", "turn-b"] {
            assert!(
                outcomes
                    .iter()
                    .any(|doc| doc.filters.get("turn_id").map(String::as_str) == Some(turn_id)),
                "provider turn ID must be retained on its document"
            );
        }
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn session_start_does_not_create_memory_store() {
        let root = temp_dir("session-start");
        let output = handle_hook(
            CodexHookKind::SessionStart,
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
            CodexHookKind::UserPromptSubmit,
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
        set_membrick_state(RecordingProvider::Codex, &root, false).unwrap();
        let mut prompt = input(&root, None, "UserPromptSubmit");
        prompt.prompt = Some("retrieve prior routing decision".to_string());
        let output = handle_hook(CodexHookKind::UserPromptSubmit, prompt, &root).unwrap();
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
        let output = handle_hook(CodexHookKind::Stop, stop, &root).unwrap();
        assert!(output.hook_specific_output.is_none());
        assert!(!memory_root(&root).exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn session_end_does_not_start_a_memory_capture_under_codex_deadline() {
        let root = temp_dir("session-end");
        let transcript = root.join("transcript.jsonl");
        fs::write(
            &transcript,
            r#"{"type":"assistant","message":{"role":"assistant","content":"Final session outcome"}}"#,
        )
        .unwrap();

        let output = handle_hook(
            CodexHookKind::SessionEnd,
            input(&root, Some(transcript), "SessionEnd"),
            &root,
        )
        .unwrap();

        assert!(output.hook_specific_output.is_none());
        assert!(
            !memory_root(&root).exists(),
            "Codex SessionEnd has a hard three-second budget; Stop already captures the final outcome"
        );
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
        handle_hook(CodexHookKind::Stop, stop_input.clone(), &root).unwrap();
        let mut stored = open_store(&root).unwrap();
        stored.refresh_index().unwrap();
        assert_eq!(
            stored.records().len(),
            4,
            "capture should store two observations, one outcome, and one session context"
        );
        let records_after_first_capture = stored.records().len();
        drop(stored);
        handle_hook(CodexHookKind::Stop, stop_input, &root).unwrap();

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
        let output = handle_hook(CodexHookKind::UserPromptSubmit, prompt, &root).unwrap();
        let context = output
            .hook_specific_output
            .expect("captured memory should be retrieved")
            .additional_context;
        assert!(context.contains("Implemented docker segment routing"));
        assert!(context.contains("Memory status:"));
        assert!(context.contains("Relation evidence:"));

        let mut expansion = input(&root, None, "UserPromptExpansion");
        expansion.command_name = Some("review".to_string());
        expansion.command_args = Some("docker routing".to_string());
        expansion.prompt = Some("/review docker routing".to_string());
        let output = handle_hook(CodexHookKind::UserPromptExpansion, expansion, &root).unwrap();
        assert!(output
            .hook_specific_output
            .unwrap()
            .additional_context
            .contains("Implemented docker segment routing"));

        fs::remove_dir_all(root).unwrap();
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
            extract_structured_memory(&transcript, CodexDocumentType::Outcome, &[]).unwrap();
        assert!(content.contains("Bash"));
        assert!(content.contains("echo safe"));
        assert!(!content.contains("secret"));
        assert!(content.contains("[REDACTED]"));
        assert!(content.contains("Finished work"));
        assert!(content.contains("Memory type: outcome"));
        assert!(content.contains("Observation 1 [assistant]:"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn parses_sanitized_runtime_transcript_fixture() {
        let transcript = include_str!("../../../../tests/fixtures/runtime_transcripts/codex.jsonl");
        let messages = extract_messages(transcript);
        assert_eq!(messages.len(), 4);
        assert!(messages
            .iter()
            .any(|message| message.content.contains("function_call")));
        assert!(messages
            .iter()
            .any(|message| message.content.contains("default_route")));
    }

    #[test]
    fn transcript_extraction_supports_codex_rollout_messages() {
        let root = temp_dir("codex-rollout");
        let transcript = root.join("rollout.jsonl");
        fs::write(
            &transcript,
            concat!(
                r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Remember cobalt routing"}]}}"#,
                "\n",
                r#"{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Implemented cobalt routing"}]}}"#,
                "\n"
            ),
        )
        .unwrap();

        let content =
            extract_structured_memory(&transcript, CodexDocumentType::Outcome, &[]).unwrap();
        assert!(content.contains("Remember cobalt routing"));
        assert!(content.contains("Implemented cobalt routing"));
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
            extract_structured_memory(&transcript, CodexDocumentType::Outcome, &[]).unwrap();
        assert!(content.contains("Remember cobalt routing"));
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
        let excerpt = relevant_excerpt(&content, "IndexStore routing", &[]);
        assert!(excerpt.contains("LocalDistinctiveness routing"));
        assert!(!excerpt.contains("unrelated documentation"));
        assert!(excerpt.len() <= MAX_EXCERPT_BYTES + 3);
    }

    #[test]
    fn compact_and_stop_capture_distinct_types_without_session_end_write() {
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
            (CodexHookKind::PreCompact, "PreCompact"),
            (CodexHookKind::Stop, "Stop"),
            (CodexHookKind::SessionEnd, "SessionEnd"),
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
                "session-context".to_string(),
                "session-observation".to_string(),
            ])
        );
        assert_eq!(
            store.records().len(),
            5,
            "two observations, two turn documents, one session context; SessionEnd writes nothing"
        );

        drop(store);
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
        handle_hook(CodexHookKind::Stop, stop.clone(), &root).unwrap();
        append_line(
            &transcript,
            r#"{"type":"assistant","message":{"role":"assistant","content":"Switched to the amber cache"}}"#,
        );
        handle_hook(CodexHookKind::Stop, stop, &root).unwrap();

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
        handle_hook(CodexHookKind::Stop, stop.clone(), &root).unwrap();
        fs::write(
            &transcript,
            concat!(
                r#"{"type":"user","message":{"role":"user","content":"Short"}}"#,
                "\n"
            ),
        )
        .unwrap();
        handle_hook(CodexHookKind::Stop, stop, &root).unwrap();

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
