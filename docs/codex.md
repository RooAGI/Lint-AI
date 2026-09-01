# Codex Integration

Lint-AI can run as a Codex memory layer and provide persistent, segmented
project memory through Codex lifecycle hooks.

Codex support is isolated behind a non-default Cargo feature. Build a local
integration-enabled binary with:

```bash
cargo build --release --features codex
```

The default core library and binary do not expose Codex-specific protocol
types, commands, or configuration behavior. Published standalone CLI release
assets can enable the feature explicitly.

## Install

From the repository root:

```bash
./lint-ai --codex-install /path/to/repo
```

By default this should:

- merge a `mcp_servers.lint-ai` entry into `~/.codex/config.toml`
- enable Codex's stable `[features].hooks = true` gate while preserving other
  feature flags
- merge Lint-AI commands into `~/.codex/hooks.json` for the supported Codex
  lifecycle events
- merge the Lint-AI memory policy into the project's `AGENTS.md`, which Codex
  uses for standing project instructions
- preserve unrelated MCP servers, hooks, and settings

Codex's built-in TUI status line currently accepts only Codex-defined item
identifiers, so installation does not inject an unsupported custom item. The
Lint-AI state is available inside Codex through
`mcp__lint-ai__lint_ai_status`, which returns both memory and recording state
plus compact display text such as `Lint-AI:ON | Record:OFF`. External
terminal/status-bar integrations can use the provider-owned
`--codex-statusline` command.

Codex project memory should be persisted under:

```text
<project>/.lint-ai/codex-memory/
```

Hook execution is fail-open: recording, indexing, or retrieval failures are
reported diagnostically and do not block Codex from continuing its session.

After installation, restart Codex Desktop so its app-server reloads
`config.toml` and `hooks.json`. Desktop versions that enforce hook trust may
also require approving the installed commands before they become runnable.

`SessionStart`, `UserPromptSubmit`, `UserPromptExpansion`, `PreToolUse`,
`PermissionRequest`, `PostToolUse`, and `SubagentStart` retrieve context.
`PreCompact`, `PostCompact`, `Stop`, `SessionEnd`, and `SubagentStop` capture
bounded session memory. A new session segment is created lazily by the first
capture hook, not by `SessionStart`.

Supported Codex hooks:

Hook execution is fail-open and bounded by a 2-second budget by default. Set
`LINT_AI_HOOK_TIMEOUT_MS` to tune it; values are clamped to 100–30,000 ms.
Timeouts are reported on stderr while the provider receives valid fallback JSON.

- `SessionStart`
- `UserPromptSubmit`
- `PreToolUse`
- `PermissionRequest`
- `PostToolUse`
- `UserPromptExpansion`
- `PreCompact`
- `PostCompact`
- `Stop`
- `SessionEnd`
- `SubagentStart`
- `SubagentStop`

Durable captures should be compact structured records rather than raw
conversation transcripts. Retrieved records should include capture/current Git
revisions and an exact, ancestor, diverged, or unknown revision status.

Retrieval should inject at most one preferred document per session and use
bounded query-relevant excerpts instead of complete records.

## Runtime controls and session recording

The Codex MCP server exposes the same provider-neutral control tools as the
Claude integration:

| Tool | Purpose |
|---|---|
| `record_session` | Start, stop, or inspect local capture-only recording |
| `enable_lint_ai` | Enable memory retrieval/capture and recording by default |
| `disable_lint_ai` | Disable Lint-AI memory behavior without changing recording |
| `lint_ai_status` | Return `Lint-AI:ON/OFF` and `Record:ON/OFF` |

Inside Codex, call `mcp__lint-ai__record_session` with `start`, `stop`, or
`status`:

```json
{"action":"start"}
```

Recording is independent from retrieval, remains local to the current project,
and is not promoted into durable memory automatically. Codex does not
currently provide an arbitrary custom TUI status-line item. The state is
available through `mcp__lint-ai__lint_ai_status` and the external renderer:

```bash
lint-ai --codex-statusline
```

## Replay and A/B comparison

Run the same recorded Codex prompts with Lint-AI disabled and enabled:

```bash
lint-ai --replay-session <session-id> \
  --session-provider codex \
  --replay-disable-lint-ai

lint-ai --replay-session <session-id> \
  --session-provider codex \
  --replay-enable-lint-ai
```

Each replay creates a fresh recorded `replay-*` session. Codex starts a new
conversation for the first prompt and resumes it for subsequent prompts. The
baseline archive is not modified. Use `--promote-session` to load selected
recorded events into `.lint-ai/codex-memory/`.

Generate a report from a session archive, or compare baseline and replay:

```bash
python3 metrics/generate_session_metric_report.py \
  --session .lint-ai/codex-sessions/<session-id> \
  --compare-session .lint-ai/codex-sessions/<replay-id> \
  --output metrics/reports/codex-comparison.json
```

The report covers quality, token usage, duration, time to first response,
tool calls, repeated exploration, hook/MCP overhead, memory retrieval, and
recording completeness. Baseline/replay reports also expose token, latency,
and quality deltas.

## Performance expectations

The Codex benchmark measures task success, expected-fact accuracy, parent and
all-model token usage, end-to-end latency, hook and MCP latency, context bytes,
tool activity, and subagent usage. Results are specific to the
provider/model/repository/revision under test. Missing Codex usage telemetry
is represented as unavailable, not zero. See [Codex Performance Test
Design](codex-performance-tests.md) for the controlled comparison matrix and
measured run artifacts.

## Inspect Memory

Inspect the persisted store summary:

```bash
lint-ai --inspect-index .lint-ai/codex-memory
```

Inspect the documents at each indexing stage:

```bash
lint-ai --inspect-index .lint-ai/codex-memory --inspect-view source-documents
lint-ai --inspect-index .lint-ai/codex-memory --inspect-view records
lint-ai --inspect-index .lint-ai/codex-memory --inspect-view segments
```

- `source-documents` should show the reconstructed public ingestion objects.
- `records` should show enriched `DocRecord` values used to build the query
  index.
- `segments` should show segment IDs, document membership, and profile sizes.

All views should emit JSON and can be filtered with `jq`.

## Serve

Run the Codex integration server directly:

```bash
./lint-ai --codex-serve /path/to/repo
```

The server should expose two tools:

- `search`: run a corpus query and return ranked results plus diagnostics
- `info`: return basic workspace information

## Verify Installation

After installation, verify that the configured MCP process can start and
complete both the MCP initialize and tool-list handshakes:

```bash
LINT_AI_MCP_HEALTH_PATH=/tmp/lint-ai-codex-mcp-health.json \
  ./lint-ai --codex-verify-mcp /path/to/repo --mcp-timeout-ms 30000
```

The command emits JSON with startup and handshake timings, protocol version,
tool count, and captured server diagnostics. A healthy result has
`"status": "healthy"`. Use a longer timeout for the first run on a large
repository because the persistent index may need to be built.

## Notes

- The integration uses Codex's documented lifecycle and hook/config layering
  rather than inventing a separate memory system.
- Hook failures should fail open and not block normal Codex execution.
- Captured transcript input should be bounded and should exclude tool-use
  blocks.
- The server should use the existing Rust retrieval stack and the current
  workspace path as its index root.
- Existing Codex config entries should be preserved when the installer runs.
