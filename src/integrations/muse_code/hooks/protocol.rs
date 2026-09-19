//! Muse Code hook payload parsing.
//!
//! Field shapes below were validated against a live Muse Code 1.3.0 binary
//! (2026-09-18). Hook configs use `"type": "command"`, `"command"` as a
//! single shell string, and `"timeout"` in seconds. Tool payloads
//! (`PreToolUse`/`PostToolUse`) were observed only in the binary's embedded
//! documentation, not from a live tool-calling run, so every tool field is
//! optional and parsed defensively.

use serde::Deserialize;
use serde_json::{Map, Value};
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
pub struct MuseHookInput {
    pub session_id: String,
    #[serde(default)]
    pub transcript_path: Option<PathBuf>,
    #[serde(default)]
    pub cwd: PathBuf,
    pub hook_event_name: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub permission_mode: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub turn_id: Option<String>,
    #[serde(default)]
    pub last_assistant_message: Option<String>,
    #[serde(default)]
    pub stop_hook_active: bool,
    // Tool events were not observed live (echo provider makes no tool calls),
    // so all of these stay optional.
    #[serde(default)]
    pub tool_input: Option<Value>,
    #[serde(default)]
    pub tool_use_id: Option<String>,
    #[serde(default)]
    pub tool_response: Option<Value>,
    #[serde(default)]
    pub is_interrupt: Option<bool>,
    #[serde(default)]
    pub duration_ms: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_live_session_start_payload() {
        let input: MuseHookInput = serde_json::from_value(serde_json::json!({
            "schema_version": 1,
            "session_id": "6be68b6b-d95c-41ac-b4c0-6f8b31a4c9c4",
            "transcript_path": null,
            "cwd": "/tmp/muse-probe/work",
            "hook_event_name": "SessionStart",
            "model": "muse-spark",
            "permission_mode": "default",
            "source": "startup"
        }))
        .unwrap();
        assert_eq!(input.hook_event_name, "SessionStart");
        assert_eq!(input.model.as_deref(), Some("muse-spark"));
        assert_eq!(input.permission_mode.as_deref(), Some("default"));
        assert_eq!(input.source.as_deref(), Some("startup"));
        assert!(input.prompt.is_none());
    }

    #[test]
    fn parses_live_session_end_payload() {
        let input: MuseHookInput = serde_json::from_value(serde_json::json!({
            "session_id": "6be68b6b-d95c-41ac-b4c0-6f8b31a4c9c4",
            "transcript_path": null,
            "cwd": "/tmp/muse-probe/work",
            "hook_event_name": "SessionEnd",
            "model": "muse-spark",
            "permission_mode": "default",
            "reason": "exit",
            "source": "exit"
        }))
        .unwrap();
        assert_eq!(input.reason.as_deref(), Some("exit"));
    }

    #[test]
    fn parses_live_user_prompt_submit_payload() {
        let input: MuseHookInput = serde_json::from_value(serde_json::json!({
            "session_id": "6be68b6b-d95c-41ac-b4c0-6f8b31a4c9c4",
            "transcript_path": null,
            "cwd": "/tmp/muse-probe/work",
            "hook_event_name": "UserPromptSubmit",
            "model": "muse-spark",
            "permission_mode": "default",
            "prompt": "what does this repo do",
            "turn_id": "5d47f60b-9e33-4b6d-9a02-8d47a9d2aab3",
            "source": "submit"
        }))
        .unwrap();
        assert_eq!(input.prompt.as_deref(), Some("what does this repo do"));
        assert!(input.turn_id.is_some());
    }

    #[test]
    fn parses_live_stop_payload() {
        let input: MuseHookInput = serde_json::from_value(serde_json::json!({
            "session_id": "6be68b6b-d95c-41ac-b4c0-6f8b31a4c9c4",
            "transcript_path": null,
            "cwd": "/tmp/muse-probe/work",
            "hook_event_name": "Stop",
            "model": "muse-spark",
            "permission_mode": "default",
            "source": "submit",
            "stop_hook_active": false,
            "last_assistant_message": "It parses JSON files.",
            "turn_id": "5d47f60b-9e33-4b6d-9a02-8d47a9d2aab3"
        }))
        .unwrap();
        assert_eq!(
            input.last_assistant_message.as_deref(),
            Some("It parses JSON files.")
        );
        assert!(!input.stop_hook_active);
    }

    #[test]
    fn ignores_unknown_fields_and_accepts_minimal_payload() {
        let input: MuseHookInput = serde_json::from_value(serde_json::json!({
            "session_id": "abc",
            "cwd": "/tmp",
            "hook_event_name": "SessionStart",
            "future_field": {"nested": true}
        }))
        .unwrap();
        assert_eq!(input.session_id, "abc");
        assert!(input.extra.contains_key("future_field"));
    }
}
