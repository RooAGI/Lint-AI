use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
pub struct ClaudeHookInput {
    pub session_id: String,
    #[serde(default)]
    pub transcript_path: Option<PathBuf>,
    #[serde(default)]
    pub cwd: PathBuf,
    pub hook_event_name: String,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub tool_input: Option<Value>,
    #[serde(default)]
    pub tool_response: Option<Value>,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub agent_type: Option<String>,
    #[serde(default)]
    pub turn_id: Option<String>,
    #[serde(default)]
    pub expansion_type: Option<String>,
    #[serde(default)]
    pub command_name: Option<String>,
    #[serde(default)]
    pub command_args: Option<String>,
    #[serde(default)]
    pub command_source: Option<String>,
    #[serde(default)]
    pub stop_hook_active: bool,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ClaudeHookOutput {
    #[serde(rename = "hookSpecificOutput", skip_serializing_if = "Option::is_none")]
    pub hook_specific_output: Option<ClaudeHookSpecificOutput>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClaudeHookSpecificOutput {
    #[serde(rename = "hookEventName")]
    pub hook_event_name: String,
    #[serde(rename = "additionalContext")]
    pub additional_context: String,
}

impl ClaudeHookOutput {
    pub fn additional_context(event_name: &str, context: String) -> Self {
        if context.trim().is_empty() {
            return Self::default();
        }
        Self {
            hook_specific_output: Some(ClaudeHookSpecificOutput {
                hook_event_name: event_name.to_string(),
                additional_context: context,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_user_prompt_expansion_fields() {
        let input: ClaudeHookInput = serde_json::from_value(serde_json::json!({
            "session_id": "session-1",
            "transcript_path": "/tmp/transcript.jsonl",
            "cwd": "/tmp",
            "hook_event_name": "UserPromptExpansion",
            "expansion_type": "slash_command",
            "command_name": "review",
            "command_args": "auth",
            "command_source": "plugin",
            "prompt": "/review auth"
        }))
        .unwrap();

        assert_eq!(input.command_name.as_deref(), Some("review"));
        assert_eq!(input.command_args.as_deref(), Some("auth"));
        assert_eq!(input.expansion_type.as_deref(), Some("slash_command"));
    }

    #[test]
    fn empty_context_serializes_to_empty_object() {
        assert_eq!(
            serde_json::to_value(ClaudeHookOutput::default()).unwrap(),
            serde_json::json!({})
        );
    }

    #[test]
    fn missing_cwd_uses_empty_fallback_path() {
        let input: ClaudeHookInput = serde_json::from_value(serde_json::json!({
            "session_id": "session-1",
            "hook_event_name": "SessionStart"
        }))
        .unwrap();
        assert!(input.cwd.as_os_str().is_empty());
    }

    #[test]
    fn parses_post_tool_use_fields() {
        let input: ClaudeHookInput = serde_json::from_value(serde_json::json!({
            "session_id": "session-1",
            "cwd": "/tmp",
            "hook_event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "cargo test"},
            "tool_response": {"output": "test result: ok"},
            "turn_id": "turn-42"
        }))
        .unwrap();

        assert_eq!(input.tool_name.as_deref(), Some("Bash"));
        assert_eq!(input.turn_id.as_deref(), Some("turn-42"));
        assert!(input.tool_input.is_some());
        assert!(input.tool_response.is_some());
    }

    #[test]
    fn parses_subagent_fields() {
        let input: ClaudeHookInput = serde_json::from_value(serde_json::json!({
            "session_id": "session-1",
            "cwd": "/tmp",
            "hook_event_name": "SubagentStart",
            "agent_id": "agent-7",
            "agent_type": "claude",
            "prompt": "review the auth module"
        }))
        .unwrap();

        assert_eq!(input.agent_id.as_deref(), Some("agent-7"));
        assert_eq!(input.agent_type.as_deref(), Some("claude"));
        assert_eq!(input.prompt.as_deref(), Some("review the auth module"));
    }
}
