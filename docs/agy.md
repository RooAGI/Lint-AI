# Antigravity CLI integration

Lint-AI supports the Antigravity CLI (`agy`) through its MCP and lifecycle-hook
integration.

Build and install the integration:

```bash
cargo install --path . --features agy
lint-ai --agy-install /path/to/project
```

The installer writes the MCP server to `~/.gemini/config/mcp_config.json` and
the hook commands to `~/.gemini/config/hooks.json`. Explicit files
can be supplied with `--agy-config` and `--agy-settings`.
The installer also adds the project-scoped `lint-ai-memory` skill under
`.agents/skills/lint-ai-memory/SKILL.md`, which AGY loads as a workspace skill.
User-modified skills are preserved; use `--agy-force-skill` to replace one
intentionally.

AGY receives the same `search`, `info`, `record_session`, `enable_lint_ai`,
`disable_lint_ai`, and `lint_ai_status` MCP tools as the other integrations.
AGY sessions are recorded under `.lint-ai/agy-sessions/` and use
`.lint-ai/agy-memory/` for provider-specific memory hooks.

## Supported hooks

The installer registers the following AGY lifecycle hooks:

| Hook | Purpose | Lint-AI behavior |
| --- | --- | --- |
| `PreToolUse` | A tool call is about to run | Records the event and may inject relevant memory steps. |
| `PostToolUse` | A tool call has completed | Records the event and result. |
| `PreInvocation` | An agent invocation is about to run | Records the event and may inject relevant memory steps. |
| `PostInvocation` | An agent invocation has completed | Records the event. |
| `Stop` | AGY is stopping | Records the terminal event. |

### Hook input

Lint-AI accepts a JSON object on standard input. The integration recognizes
these fields:

```json
{
  "session_id": "agy-session-id",
  "cwd": "/path/to/project",
  "hook_event_name": "BeforeTool",
  "prompt": "optional user or agent prompt",
  "tool_name": "optional tool name",
  "tool_input": {},
  "tool_response": {}
}
```

`session_id`, `cwd`, and `hook_event_name` identify the event. The prompt and
tool fields are optional; additional AGY fields are preserved in the recorded
event payload.

### Hook output

The hook writes one JSON object to standard output. When relevant memory is
found, it returns an `additionalContext` value under `hookSpecificOutput`:

```json
{
  "hookSpecificOutput": {
    "hookEventName": "BeforeAgent",
    "additionalContext": "Relevant Lint-AI memory:\n..."
  }
}
```

`SessionStart` can also return a `systemMessage` confirming that hooks are
active. Other events return an empty JSON object when no context is available
or when Lint-AI is disabled.

### Recording and memory behavior

- Every supported hook is recorded when session recording is enabled.
- Enabling Lint-AI also enables recording by default.
- `record_session` can start or stop recording without ending the AGY session.
- Disabling Lint-AI stops memory injection but does not erase existing sessions.
- Hook failures are fail-open and do not block AGY tool calls or model turns.
- Hook working directories must remain inside the project root configured by
  `--agy-install`; outside paths are rejected.

The adapter is intentionally fail-open: an unavailable Lint-AI hook does not
interrupt an AGY session. AGY authentication and model usage remain managed by
AGY itself.

## Permission troubleshooting

The normal installation does not require broad command permissions. If a
particular AGY version refuses to execute configured lifecycle hooks, these
options can be used temporarily while diagnosing that installation:

```json
{
  "permissions": {
    "allow": ["command(*)"]
  }
}
```

Run the CLI with:

```bash
agy --dangerously-skip-permissions ...
```

`command(*)` allows every command and `--dangerously-skip-permissions` bypasses
AGY permission checks. They are intentionally not part of the normal AGY
benchmark or recommended default configuration. Use them only for a short,
isolated diagnostic run, do not use AGY concurrently, and restore the original
settings afterward.
