# Muse Code Integration

Lint-AI can run as a Muse Code MCP memory server and provide persistent,
segmented project memory with current-state retrieval, evidence links, and
supersession tracking.

Muse Code support is isolated behind a non-default Cargo feature. Build a
local integration-enabled binary with:

```bash
cargo build --release --features muse-code
```

The default core library and binary do not expose Muse Code-specific protocol
types, commands, or configuration behavior. Published standalone CLI release
assets can enable the feature explicitly.

## Install

From the repository root:

```bash
./lint-ai --muse-install /path/to/repo
```

By default this:

- merges a `mcpServers.lint-ai` entry into `~/.config/muse/settings.json`
  (or `$XDG_CONFIG_HOME/muse/settings.json` when `XDG_CONFIG_HOME` is set)
- migrates any legacy `mcp_servers` entries into `mcpServers` — Muse ignores
  the legacy key, and when both keys exist it drops the whole MCP member,
  disabling every configured server
- installs capture-only session hooks (see below)
- preserves `schema_version` and every existing key — other MCP servers,
  hooks, and settings are left untouched
- merges the Lint-AI memory policy into the project's `AGENTS.md`, which Muse
  Code prefers over `CLAUDE.md` for standing project instructions

Use `--muse-config <path>` to target a different settings file (useful for
testing without touching your real Muse Code configuration).

The MCP entry looks like this:

```json
{
  "mcpServers": {
    "lint-ai": {
      "transport": "stdio",
      "command": "/path/to/lint-ai",
      "args": ["--muse-serve", "/path/to/repo"],
      "enabled": true
    }
  }
}
```

The project root is pinned in `args` so the server indexes the right project
regardless of the client's working directory.

Muse Code reads MCP servers only from the user/global `settings.json`; there
is no project-scoped MCP location. The installer refuses to overwrite a
malformed `settings.json` rather than risk breaking your Muse Code setup —
a malformed settings file fails every `muse` command.

## Project memory location

Muse Code project memory is persisted under:

```text
<project>/.lint-ai/muse-memory/
```

Note that Muse Code also ships native project memory under
`<project>/.agents/memory/` (a Markdown index injected at session start).
Lint-AI is complementary, not a replacement: the native index is a flat set
of notes with no time-awareness, while Lint-AI tracks what is still true —
current-state retrieval, evidence links back to sources, and supersession
when a newer decision replaces an older one.

## Runtime controls and session recording

The Muse Code MCP server exposes the same provider-neutral control tools as
the other agent integrations:

| Tool | Purpose |
|---|---|
| `search` | Query project memory; returns ranked hits with source, score, `semantic_status`, and `superseded_by` |
| `info` | Return basic information about the indexed workspace |
| `list_memories` | List stored memories |
| `record_session` | Start, stop, or inspect local capture-only recording |
| `enable_lint_ai` | Enable memory retrieval/capture and recording by default |
| `disable_lint_ai` | Disable Lint-AI memory behavior without changing recording |
| `lint_ai_status` | Return `Lint-AI:ON/OFF` and `Record:ON/OFF` |

Inside Muse Code, call `record_session` with `start`, `stop`, or `status`:

```json
{"action":"start"}
```

Recording is independent from retrieval, remains local to the current
project, and is not promoted into durable memory automatically.

## Lifecycle hooks

`--muse-install` wires capture-only session hooks into the same
`settings.json`. Each event gets a matcher group that runs
`lint-ai --muse-hook <event>` as a shell command string with a 60-second
timeout — the entry schema was validated against a live `muse` 1.3.0 binary.
The hooks record the session lifecycle and tool use into
`<project>/.lint-ai/muse-sessions/`:

| Event | Captured |
|---|---|
| `SessionStart` | session id, model, permission mode |
| `UserPromptSubmit` | prompt text, turn id |
| `PreToolUse` / `PostToolUse` / `PostToolUseFailure` | tool use id, input/response, duration |
| `Stop` | last assistant message, turn id |
| `SessionEnd` | end reason |

Recording never injects memory into the Muse context and never blocks a
session: every failure path warns to stderr and exits zero. Existing user
hooks are preserved, and reinstalling is idempotent. Recording is opt-in —
enable it with the `record_session` MCP tool (`{"action":"start"}`).

Tool-event payloads (`PreToolUse`/`PostToolUse`) were validated against the
binary's embedded hook documentation rather than a live tool-calling run, so
their fields are parsed defensively.

## Serve

Run the Muse Code integration server directly:

```bash
./lint-ai --muse-serve /path/to/repo
```

## Verify installation

After installation, verify that the configured MCP process can start and
complete both the MCP initialize and tool-list handshakes:

```bash
LINT_AI_MCP_HEALTH_PATH=/tmp/lint-ai-muse-mcp-health.json \
  ./lint-ai --muse-verify-mcp /path/to/repo --mcp-timeout-ms 30000
```

The command emits JSON with startup and handshake timings, protocol version,
tool count, and captured server diagnostics. A healthy result has
`"status": "healthy"`. Use a longer timeout for the first run on a large
repository because the persistent index may need to be built.

## Inspect memory

Inspect the persisted store summary:

```bash
lint-ai --inspect-index .lint-ai/muse-memory
```

Inspect the documents at each indexing stage:

```bash
lint-ai --inspect-index .lint-ai/muse-memory --inspect-view source-documents
lint-ai --inspect-index .lint-ai/muse-memory --inspect-view records
lint-ai --inspect-index .lint-ai/muse-memory --inspect-view segments
```

All views emit JSON and can be filtered with `jq`.

## Notes

- The integration uses Muse Code's documented MCP configuration layering
  rather than inventing a separate memory system.
- Existing settings entries are preserved when the installer runs; a
  malformed `settings.json` is never overwritten.
- Session replay (`--replay-session --session-provider muse`) is not wired
  yet.
- Muse Code runs on macOS and Linux only; the integration inherits that
  platform scope.
