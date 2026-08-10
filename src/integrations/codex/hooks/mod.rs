mod protocol;

use super::document::{CodexDocument, CodexDocumentType};
use crate::integrations::session_recording::{
    lint_ai_enabled, record_event_if_enabled, record_transcript_usage_if_available,
    RecordingProvider,
};
use crate::pipeline::{IndexStore, MemoryIndexLayout, PipelineOptions};
use crate::segments::SegmentRoutingStrategy;
use anyhow::{Context, Result};
use protocol::{CodexHookInput, CodexHookOutput};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_TOP_K: usize = 5;
const DEFAULT_CONTEXT_BYTES: usize = 8_000;
const INITIAL_TRANSCRIPT_BYTES: u64 = 64 * 1024;
const MAX_TRANSCRIPT_BYTES: u64 = 4 * 1024 * 1024;
const MAX_CAPTURED_MESSAGES: usize = 6;
const MAX_MEMORY_FIELD_BYTES: usize = 1_500;
const MAX_EXCERPT_BYTES: usize = 800;
const MIN_TOOL_QUERY_TERMS: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexHookKind {
    SessionStart,
    UserPromptSubmit,
    PreToolUse,
    PermissionRequest,
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
            Self::PermissionRequest => "PermissionRequest",
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
    let raw: Value = serde_json::from_reader(std::io::stdin().lock())
        .context("failed to parse Codex hook input")?;
    let input: CodexHookInput =
        serde_json::from_value(raw).context("failed to decode Codex hook input")?;
    let root = resolve_root(&input.cwd, fallback_root)?;
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
        &input.session_id,
        kind.event_name(),
        payload,
    ) {
        eprintln!("warning: Lint-AI session recording failed open: {error:#}");
    }
    if matches!(kind, CodexHookKind::Stop) && !input.stop_hook_active {
        if let Err(error) = record_transcript_usage_if_available(
            RecordingProvider::Codex,
            &root,
            &input.session_id,
            input.transcript_path.as_deref(),
            input.turn_id.as_deref(),
            "codex-transcript",
        ) {
            eprintln!("warning: Codex transcript usage capture failed open: {error:#}");
        }
    }
    let started = Instant::now();
    let output = match handle_hook(kind, input, fallback_root) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("warning: Lint-AI Codex hook failed open: {error:#}");
            CodexHookOutput::default()
        }
    };
    write_timing_record(kind, started.elapsed().as_secs_f64() * 1000.0, &output);
    serde_json::to_writer(std::io::stdout().lock(), &output)?;
    std::io::stdout().lock().write_all(b"\n")?;
    Ok(())
}

fn write_timing_record(kind: CodexHookKind, elapsed_ms: f64, output: &CodexHookOutput) {
    let Ok(path) = std::env::var("LINT_AI_HOOK_TIMINGS_PATH") else {
        return;
    };
    let operation = match kind {
        CodexHookKind::SessionStart
        | CodexHookKind::UserPromptSubmit
        | CodexHookKind::PreToolUse
        | CodexHookKind::PermissionRequest
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
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
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
    if !lint_ai_enabled(RecordingProvider::Codex, &root)? {
        return Ok(CodexHookOutput::default());
    }
    match kind {
        CodexHookKind::SessionStart => retrieve(
            &root,
            kind.event_name(),
            "decisions unresolved work failures implemented changes",
        ),
        CodexHookKind::UserPromptSubmit => retrieve(
            &root,
            kind.event_name(),
            input.prompt.as_deref().unwrap_or_default(),
        ),
        CodexHookKind::PreToolUse | CodexHookKind::PermissionRequest => {
            retrieve(&root, kind.event_name(), &tool_query(&input))
        }
        CodexHookKind::PostToolUse => {
            let query = tool_query(&input);
            if query_terms(&query).len() < MIN_TOOL_QUERY_TERMS {
                return Ok(CodexHookOutput::default());
            }
            retrieve(&root, kind.event_name(), &query)
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
            retrieve(&root, kind.event_name(), &query)
        }
        CodexHookKind::PreCompact => capture(&root, input, CodexDocumentType::Checkpoint),
        CodexHookKind::PostCompact => capture(&root, input, CodexDocumentType::Checkpoint),
        CodexHookKind::Stop if input.stop_hook_active => Ok(CodexHookOutput::default()),
        CodexHookKind::Stop => capture(&root, input, CodexDocumentType::Outcome),
        CodexHookKind::SessionEnd => capture(&root, input, CodexDocumentType::SessionSummary),
        CodexHookKind::SubagentStart => retrieve(
            &root,
            kind.event_name(),
            input.prompt.as_deref().unwrap_or_default(),
        ),
        CodexHookKind::SubagentStop => capture(&root, input, CodexDocumentType::Outcome),
    }
}

fn retrieve(root: &Path, event_name: &str, query: &str) -> Result<CodexHookOutput> {
    if query.trim().is_empty() || !memory_root(root).exists() {
        return Ok(CodexHookOutput::default());
    }
    let mut store = open_store(root)?;
    if store.is_empty() {
        return Ok(CodexHookOutput::default());
    }
    let results = store.query(query, DEFAULT_TOP_K * 3)?;
    let selected = select_session_documents(&store, results, DEFAULT_TOP_K);
    let mut seen = HashSet::new();
    let current_revision = git_value(root, &["rev-parse", "HEAD"]);
    let current_branch = git_value(root, &["branch", "--show-current"]);
    let mut context = String::new();
    for result in selected {
        let Some(record) = store.record_by_id(&result.doc_id) else {
            continue;
        };
        let normalized = relevant_excerpt(&record.content, query, &result.matched_terms);
        let normalized = normalized.trim();
        if normalized.is_empty() || !seen.insert(normalized.to_string()) {
            continue;
        }
        let captured_revision = record.filters.get("revision").map(String::as_str);
        let status = revision_status(root, captured_revision, current_revision.as_deref());
        let entry = format!(
            "\n- Source: {}\n  Type: {}\n  Captured: {}\n  Captured branch: {}\n  Captured revision: {}\n  Current branch: {}\n  Current revision: {}\n  Revision status: {}\n  {}\n",
            record.source,
            record
                .filters
                .get("document_type")
                .map(String::as_str)
                .unwrap_or("memory"),
            record.timestamp.as_deref().unwrap_or("unknown"),
            record
                .filters
                .get("branch")
                .map(String::as_str)
                .unwrap_or("unknown"),
            captured_revision.unwrap_or("unknown"),
            current_branch.as_deref().unwrap_or("unknown"),
            current_revision.as_deref().unwrap_or("unknown"),
            status,
            normalized,
        );
        if context.len() + entry.len() > DEFAULT_CONTEXT_BYTES {
            break;
        }
        context.push_str(&entry);
    }
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

fn select_session_documents(
    store: &IndexStore,
    results: Vec<crate::index::SearchResult>,
    limit: usize,
) -> Vec<crate::index::SearchResult> {
    let mut preferred = HashMap::<String, (u8, &crate::index::DocRecord)>::new();
    for record in store.records() {
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
        let Some(record) = store.record_by_id(&result.doc_id) else {
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

fn document_type_priority(document_type: &str) -> u8 {
    match document_type {
        "session-summary" => 3,
        "outcome" => 2,
        "checkpoint" => 1,
        _ => 0,
    }
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
    let Some(transcript_path) = transcript_path(&input) else {
        return Ok(CodexHookOutput::default());
    };
    let affected_paths = affected_paths(&input.extra);
    let content = extract_structured_memory(&transcript_path, document_type, &affected_paths)?;
    if content.trim().is_empty() {
        return Ok(CodexHookOutput::default());
    }
    let event_id = format!(
        "{}:{}:{}",
        input.session_id,
        document_type.as_str(),
        crate::ids::stable_doc_id_from_source(&content)
    );
    let document = CodexDocument {
        event_id,
        session_id: input.session_id,
        document_type,
        content,
        cwd: root.to_path_buf(),
        timestamp: Some(current_timestamp()),
        command_name: input.command_name,
        command_args: input.command_args,
        affected_paths,
        branch: git_value(root, &["branch", "--show-current"]),
        revision: git_value(root, &["rev-parse", "HEAD"]),
    };
    let mut store = open_store(root)?;
    store.upsert(document.into_source_document()?);
    store.refresh()?;
    Ok(CodexHookOutput::default())
}

fn transcript_path(input: &CodexHookInput) -> Option<PathBuf> {
    if let Some(path) = input.transcript_path.as_ref().filter(|path| path.exists()) {
        return Some(path.clone());
    }
    let codex_home = std::env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".codex")))?;
    find_session_transcript(&codex_home.join("sessions"), &input.session_id)
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

fn memory_root(root: &Path) -> PathBuf {
    root.join(".lint-ai").join("codex-memory")
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
}

fn extract_structured_memory(
    path: &Path,
    document_type: CodexDocumentType,
    affected_paths: &[String],
) -> Result<String> {
    let messages = extract_recent_messages(path)?;
    let request = messages
        .iter()
        .rev()
        .find(|message| message.role == "user" && !is_task_notification(&message.content))
        .map(|message| compact_memory_field(&message.content));
    let result = messages
        .iter()
        .rev()
        .find(|message| message.role == "assistant" && !looks_like_tool_trace(&message.content))
        .map(|message| compact_memory_field(&message.content));

    let mut fields = vec![format!("Memory type: {}", document_type.as_str())];
    if let Some(request) = request.filter(|value| !value.is_empty()) {
        fields.push(format!("Request: {request}"));
    }
    if let Some(result) = result.filter(|value| !value.is_empty()) {
        fields.push(format!("Result: {result}"));
    }
    if !affected_paths.is_empty() {
        fields.push(format!("Affected paths: {}", affected_paths.join(", ")));
    }
    if fields.len() == 1 {
        return Ok(String::new());
    }
    Ok(fields.join("\n"))
}

fn extract_recent_messages(path: &Path) -> Result<VecDeque<ConversationMessage>> {
    let mut file = File::open(path)
        .with_context(|| format!("failed to open Codex transcript {}", path.display()))?;
    let len = file.metadata()?.len();
    let mut scan_bytes = INITIAL_TRANSCRIPT_BYTES.min(len);
    loop {
        let start = len.saturating_sub(scan_bytes);
        file.seek(SeekFrom::Start(start))?;
        let mut text = String::new();
        file.read_to_string(&mut text)?;
        let text = if start > 0 {
            text.split_once('\n').map(|(_, tail)| tail).unwrap_or("")
        } else {
            &text
        };
        let messages = extract_messages(text);
        if !messages.is_empty() || start == 0 || scan_bytes >= MAX_TRANSCRIPT_BYTES {
            return Ok(messages);
        }
        scan_bytes = (scan_bytes.saturating_mul(2))
            .min(len)
            .min(MAX_TRANSCRIPT_BYTES);
    }
}

fn extract_messages(text: &str) -> VecDeque<ConversationMessage> {
    let mut messages = VecDeque::new();
    for line in text.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let message = value.get("message").or_else(|| {
            value
                .get("payload")
                .filter(|payload| payload.get("type").and_then(Value::as_str) == Some("message"))
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
        let content = message.get("content").map(text_blocks).unwrap_or_default();
        if !content.trim().is_empty() {
            messages.push_back(ConversationMessage {
                role: role.to_string(),
                content: redact(&content),
            });
            if messages.len() > MAX_CAPTURED_MESSAGES {
                messages.pop_front();
            }
        }
    }
    messages
}

fn compact_memory_field(value: &str) -> String {
    let normalized = value
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if normalized.len() <= MAX_MEMORY_FIELD_BYTES {
        return normalized;
    }
    let mut end = MAX_MEMORY_FIELD_BYTES;
    while !normalized.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", normalized[..end].trim_end())
}

fn is_task_notification(value: &str) -> bool {
    value.trim_start().starts_with("<task-notification>")
}

fn looks_like_tool_trace(value: &str) -> bool {
    let value = value.trim_start();
    value.starts_with("**Tool:") || value.starts_with("<tool-")
}

fn revision_status(root: &Path, captured: Option<&str>, current: Option<&str>) -> &'static str {
    let (Some(captured), Some(current)) = (captured, current) else {
        return "unknown";
    };
    if captured == current {
        return "exact-match";
    }
    let status = std::process::Command::new("git")
        .args(["merge-base", "--is-ancestor", captured, current])
        .current_dir(root)
        .status();
    match status {
        Ok(status) if status.success() => "captured-revision-is-ancestor",
        Ok(status) if status.code() == Some(1) => "diverged",
        _ => "unknown",
    }
}

fn text_blocks(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter(|block| {
                matches!(
                    block.get("type").and_then(Value::as_str),
                    Some("text" | "input_text" | "output_text")
                )
            })
            .filter_map(|block| block.get("text").and_then(Value::as_str))
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

fn git_value(root: &Path, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
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
    use crate::integrations::session_recording::set_lint_ai_state;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("lint-ai-claude-{name}-{nanos}"));
        fs::create_dir_all(&path).unwrap();
        path
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
        set_lint_ai_state(RecordingProvider::Codex, &root, false).unwrap();
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
        handle_hook(CodexHookKind::Stop, stop_input, &root).unwrap();

        let mut stored = open_store(&root).unwrap();
        stored.refresh().unwrap();
        assert_eq!(
            stored.records().len(),
            1,
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
    fn transcript_extraction_ignores_tool_blocks_and_redacts_secret_lines() {
        let root = temp_dir("transcript");
        let transcript = root.join("transcript.jsonl");
        fs::write(
            &transcript,
            concat!(
                r#"{"message":{"role":"assistant","content":[{"type":"tool_use","name":"Bash","input":{"command":"secret"}},{"type":"text","text":"api_key=secret\nFinished work"}]}}"#,
                "\n"
            ),
        )
        .unwrap();

        let content =
            extract_structured_memory(&transcript, CodexDocumentType::Outcome, &[]).unwrap();
        assert!(!content.contains("Bash"));
        assert!(!content.contains("secret"));
        assert!(content.contains("[REDACTED]"));
        assert!(content.contains("Finished work"));
        assert!(content.contains("Memory type: outcome"));
        assert!(content.contains("Result:"));
        fs::remove_dir_all(root).unwrap();
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
    fn exact_revision_provenance_does_not_require_repository_revalidation() {
        let root = std::env::current_dir().unwrap();
        let revision = git_value(&root, &["rev-parse", "HEAD"]).unwrap();
        assert_eq!(
            revision_status(&root, Some(&revision), Some(&revision)),
            "exact-match"
        );
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

        for (kind, event) in [
            (CodexHookKind::PreCompact, "PreCompact"),
            (CodexHookKind::Stop, "Stop"),
            (CodexHookKind::SessionEnd, "SessionEnd"),
        ] {
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
            ])
        );

        drop(store);
        let output = retrieve(&root, "UserPromptSubmit", "durable routing").unwrap();
        let context = output
            .hook_specific_output
            .expect("captured memory should be retrieved")
            .additional_context;
        assert_eq!(context.matches("- Source:").count(), 1);
        assert!(context.contains("Type: session-summary"));
        fs::remove_dir_all(root).unwrap();
    }
}
