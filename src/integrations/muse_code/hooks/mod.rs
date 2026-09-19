//! Capture-only session recording hooks for Muse Code.
//!
//! Muse invokes hooks as `lint-ai --muse-hook <event>` with a JSON payload on
//! stdin using the schema documented in `protocol.rs`. These hooks record the
//! session lifecycle (start, prompts, stops, end) plus tool use into Lint-AI's
//! session store so a Muse session can later be replayed. Recording is
//! capture-only: it never injects memory into the Muse context and never
//! blocks the session. Every failure path warns to stderr and exits zero.

mod protocol;

use crate::integrations::session_recording::{record_event_if_enabled, RecordingProvider};
use anyhow::{Context, Result};
use protocol::MuseHookInput;
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MuseHookKind {
    SessionStart,
    UserPromptSubmit,
    PreToolUse,
    PostToolUse,
    PostToolUseFailure,
    Stop,
    SessionEnd,
}

impl MuseHookKind {
    pub fn event_name(self) -> &'static str {
        match self {
            Self::SessionStart => "SessionStart",
            Self::UserPromptSubmit => "UserPromptSubmit",
            Self::PreToolUse => "PreToolUse",
            Self::PostToolUse => "PostToolUse",
            Self::PostToolUseFailure => "PostToolUseFailure",
            Self::Stop => "Stop",
            Self::SessionEnd => "SessionEnd",
        }
    }

    pub fn cli_name(self) -> &'static str {
        match self {
            Self::SessionStart => "session-start",
            Self::UserPromptSubmit => "user-prompt-submit",
            Self::PreToolUse => "pre-tool-use",
            Self::PostToolUse => "post-tool-use",
            Self::PostToolUseFailure => "post-tool-use-failure",
            Self::Stop => "stop",
            Self::SessionEnd => "session-end",
        }
    }
}

/// All events we install hooks for.
pub const HOOK_EVENTS: &[MuseHookKind] = &[
    MuseHookKind::SessionStart,
    MuseHookKind::UserPromptSubmit,
    MuseHookKind::PreToolUse,
    MuseHookKind::PostToolUse,
    MuseHookKind::PostToolUseFailure,
    MuseHookKind::Stop,
    MuseHookKind::SessionEnd,
];

pub fn run_hook(kind: MuseHookKind, fallback_root: &Path) -> Result<()> {
    // Fail open on every path: a broken hook invocation must never break a
    // Muse session, so even an unreadable payload only warns on stderr.
    match run_hook_inner(kind, fallback_root) {
        Ok(()) => Ok(()),
        Err(error) => {
            eprintln!("warning: Lint-AI Muse hook failed open: {error:#}");
            Ok(())
        }
    }
}

fn run_hook_inner(kind: MuseHookKind, fallback_root: &Path) -> Result<()> {
    let raw: Value =
        crate::integrations::read_bounded_json().context("failed to parse Muse hook input")?;
    let input: MuseHookInput =
        serde_json::from_value(raw).context("failed to decode Muse hook input")?;
    handle_hook(kind, &input, fallback_root)?;
    Ok(())
}

/// Resolve the project root from the hook payload's `cwd` and record the
/// event. Split out of `run_hook` so tests can exercise it without stdin.
pub fn handle_hook(kind: MuseHookKind, input: &MuseHookInput, fallback_root: &Path) -> Result<()> {
    let root = resolve_root(&input.cwd, fallback_root)?;
    let payload = serde_json::json!({
        "hook_event_name": input.hook_event_name,
        "model": input.model,
        "permission_mode": input.permission_mode,
        "source": input.source,
        "reason": input.reason,
        "prompt": input.prompt,
        "turn_id": input.turn_id,
        "last_assistant_message": input.last_assistant_message,
        "stop_hook_active": input.stop_hook_active,
        "tool_input": input.tool_input,
        "tool_use_id": input.tool_use_id,
        "tool_response": input.tool_response,
        "is_interrupt": input.is_interrupt,
        "duration_ms": input.duration_ms,
        "extra": input.extra,
    });
    record_event_if_enabled(
        RecordingProvider::Muse,
        &root,
        &input.session_id,
        kind.event_name(),
        payload,
    )
    .context("failed to record Muse hook event")?;
    Ok(())
}

fn resolve_root(cwd: &Path, fallback_root: &Path) -> Result<PathBuf> {
    if cwd.as_os_str().is_empty() {
        return fallback_root
            .canonicalize()
            .with_context(|| format!("failed to canonicalize {}", fallback_root.display()));
    }
    cwd.canonicalize()
        .with_context(|| format!("failed to canonicalize {}", cwd.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrations::session_recording::set_recording_state;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("lint-ai-muse-hooks-{name}-{nanos}"));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn input(root: &Path, event: &str) -> MuseHookInput {
        serde_json::from_value(serde_json::json!({
            "session_id": "session-1",
            "transcript_path": null,
            "cwd": root,
            "hook_event_name": event,
            "model": "muse-spark",
            "permission_mode": "default",
            "source": "submit"
        }))
        .unwrap()
    }

    fn recorded_events(root: &Path) -> Vec<Value> {
        let path = root.join(".lint-ai/muse-sessions/session-1/events.jsonl");
        if !path.exists() {
            return Vec::new();
        }
        fs::read_to_string(&path)
            .unwrap()
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn session_start_is_recorded_with_muse_provider() {
        let root = temp_dir("start");
        set_recording_state(RecordingProvider::Muse, &root, true).unwrap();
        let input = input(&root, "SessionStart");
        handle_hook(MuseHookKind::SessionStart, &input, &root).unwrap();
        let events = recorded_events(&root);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["kind"].as_str(), Some("session_start"));
        assert!(root.join(".lint-ai/muse-sessions/session-1").is_dir());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn user_prompt_records_prompt_text() {
        let root = temp_dir("prompt");
        set_recording_state(RecordingProvider::Muse, &root, true).unwrap();
        let mut hook = input(&root, "UserPromptSubmit");
        hook.prompt = Some("what does this repo do".to_string());
        handle_hook(MuseHookKind::UserPromptSubmit, &hook, &root).unwrap();
        let events = recorded_events(&root);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["kind"].as_str(), Some("user_prompt"));
        assert_eq!(
            events[0]["payload"]["prompt"].as_str(),
            Some("what does this repo do")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn recording_is_skipped_when_lint_ai_is_disabled() {
        let root = temp_dir("disabled");
        set_recording_state(RecordingProvider::Muse, &root, false).unwrap();
        let input = input(&root, "SessionStart");
        handle_hook(MuseHookKind::SessionStart, &input, &root).unwrap();
        assert!(recorded_events(&root).is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn empty_cwd_falls_back_to_install_root() {
        let root = temp_dir("fallback");
        set_recording_state(RecordingProvider::Muse, &root, true).unwrap();
        let mut hook = input(&root, "Stop");
        hook.cwd = PathBuf::new();
        handle_hook(MuseHookKind::Stop, &hook, &root).unwrap();
        assert_eq!(recorded_events(&root).len(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn tool_events_capture_tool_fields() {
        let root = temp_dir("tool");
        set_recording_state(RecordingProvider::Muse, &root, true).unwrap();
        let mut hook = input(&root, "PostToolUse");
        hook.tool_use_id = Some("tool-9".to_string());
        hook.tool_response = Some(serde_json::json!({"ok": true}));
        hook.duration_ms = Some(412);
        handle_hook(MuseHookKind::PostToolUse, &hook, &root).unwrap();
        let events = recorded_events(&root);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["payload"]["tool_use_id"].as_str(), Some("tool-9"));
        assert_eq!(events[0]["payload"]["duration_ms"].as_u64(), Some(412));
        fs::remove_dir_all(root).unwrap();
    }
}
