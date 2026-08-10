# Gemini CLI integration

Lint-AI can integrate with Gemini CLI through its command hooks. Gemini sends
structured JSON to the hook process, so Lint-AI can record session events and
inject relevant project memory without wrapping or replacing the Gemini CLI.

## Install

Build with the Gemini feature and install the project hooks:

```sh
cargo install --path . --features gemini-cli
lint-ai --gemini-cli-install .
```

The installer updates `~/.gemini/settings.json`, preserving unrelated settings
and hooks. It registers both the `lint-ai` MCP server and lifecycle hooks. Use
`--gemini-cli-config` or `--gemini-cli-settings` to target explicit files.
Re-running the installer is safe and replaces only Lint-AI's own entries.

The installed hooks cover `SessionStart`, `BeforeAgent`, `AfterAgent`,
`BeforeModel`, `BeforeToolSelection`, `BeforeTool`, `AfterTool`, `PreCompress`,
and `SessionEnd`.

## Recording and memory

Gemini session events are stored under:

```text
.lint-ai/gemini-cli-sessions/
```

The Gemini memory index is `.lint-ai/gemini-cli-memory/`. The shared recording
controls work with `--session-provider gemini` for promotion and replay. Hook
execution is fail-open: recording, indexing, or retrieval failures are
reported diagnostically and do not block Gemini CLI from continuing its
session. Sensitive payloads use the same redaction and size limits as Claude
and Codex recordings.

Gemini CLI's hook protocol does not guarantee token usage in every lifecycle
payload. Token metrics should therefore be treated as available only when the
CLI supplies usage metadata or telemetry for that event.

## MCP note

The Gemini MCP server exposes `search`, `info`, `record_session`,
`enable_lint_ai`, `disable_lint_ai`, and `lint_ai_status`. It uses the shared
JSON-RPC transport and persistent memory/session-control components.
