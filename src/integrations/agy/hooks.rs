//! Antigravity lifecycle-hook adapter.

use crate::integrations::session_recording::{lint_ai_enabled, record_event_if_enabled, RecordingProvider};
use crate::pipeline::{IndexStore, MemoryIndexLayout, PipelineOptions};
use crate::segments::SegmentRoutingStrategy;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgyHookKind { PreToolUse, PostToolUse, PreInvocation, PostInvocation, Stop }

impl AgyHookKind {
    pub fn event_name(self) -> &'static str { match self {
        Self::PreToolUse => "PreToolUse", Self::PostToolUse => "PostToolUse",
        Self::PreInvocation => "PreInvocation", Self::PostInvocation => "PostInvocation",
        Self::Stop => "Stop",
    }}
}

#[derive(Debug, Clone, Deserialize)]
struct AgyHookInput {
    #[serde(default, rename = "conversationId")] conversation_id: String,
    #[serde(default, rename = "workspacePaths")] workspace_paths: Vec<PathBuf>,
    #[serde(default)] tool_call: Option<Value>,
    #[serde(flatten)] extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize)]
struct AgyHookOutput {
    #[serde(rename = "injectSteps", skip_serializing_if = "Option::is_none")]
    inject_steps: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decision: Option<String>,
}

pub fn run_hook(kind: AgyHookKind, fallback_root: &Path) -> Result<()> {
    let raw: Value = serde_json::from_reader(std::io::stdin().lock())
        .context("failed to parse AGY hook input")?;
    let input: AgyHookInput = serde_json::from_value(raw.clone())
        .context("failed to decode AGY hook input")?;
    let root = resolve_root(&input, fallback_root)?;
    let session_id = if input.conversation_id.is_empty() { "unknown" } else { &input.conversation_id };
    if let Err(error) = record_event_if_enabled(RecordingProvider::Agy, &root, session_id, kind.event_name(), raw) {
        eprintln!("warning: Lint-AI AGY session recording failed open: {error:#}");
    }
    let output = handle_hook(kind, &input, &root).unwrap_or_else(|error| {
        eprintln!("warning: Lint-AI AGY hook failed open: {error:#}");
        AgyHookOutput::default()
    });
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &output)?;
    stdout.write_all(b"\n")?;
    Ok(())
}

fn handle_hook(kind: AgyHookKind, input: &AgyHookInput, root: &Path) -> Result<AgyHookOutput> {
    if !lint_ai_enabled(RecordingProvider::Agy, root)? { return Ok(AgyHookOutput::default()); }
    if matches!(kind, AgyHookKind::PostToolUse | AgyHookKind::PostInvocation | AgyHookKind::Stop) {
        return Ok(AgyHookOutput::default());
    }
    let query = input.extra.get("prompt").and_then(Value::as_str)
        .or_else(|| input.extra.get("userPrompt").and_then(Value::as_str))
        .or_else(|| input.tool_call.as_ref().and_then(|v| v.get("name")).and_then(Value::as_str))
        .unwrap_or("");
    if query.trim().is_empty() { return Ok(AgyHookOutput::default()); }
    let memory = root.join(".lint-ai/agy-memory");
    if !memory.exists() { return Ok(AgyHookOutput::default()); }
    let mut store = IndexStore::at_path(&memory, PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented { query_top_n: 3, routing_strategy: SegmentRoutingStrategy::LocalDistinctiveness },
        ..PipelineOptions::default()
    })?;
    if store.is_empty() { return Ok(AgyHookOutput::default()); }
    let mut context = String::new();
    for result in store.query(query, 5)? {
        if let Some(record) = store.record_by_id(&result.doc_id) {
            let excerpt = record.content.lines().take(8).collect::<Vec<_>>().join("\n");
            if !excerpt.trim().is_empty() { context.push_str(&format!("\n- Source: {}\n  {}\n", record.source, excerpt)); }
            if context.len() > 8_000 { break; }
        }
    }
    if context.is_empty() { return Ok(AgyHookOutput::default()); }
    Ok(AgyHookOutput { inject_steps: Some(vec![format!("Relevant Lint-AI memory:\n{context}")]), decision: None })
}

fn resolve_root(input: &AgyHookInput, fallback: &Path) -> Result<PathBuf> {
    let fallback = fallback.canonicalize().with_context(|| format!("failed to resolve AGY project root {}", fallback.display()))?;
    let candidate = input.workspace_paths.first().map(PathBuf::from).unwrap_or_else(|| fallback.clone()).canonicalize()?;
    if !candidate.starts_with(&fallback) { anyhow::bail!("AGY workspace path is outside configured project root") }
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uses_current_agy_event_names() {
        assert_eq!(AgyHookKind::PreToolUse.event_name(), "PreToolUse");
        assert_eq!(AgyHookKind::Stop.event_name(), "Stop");
    }
}
