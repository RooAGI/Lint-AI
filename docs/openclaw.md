# OpenClaw integration

Lint-AI supports OpenClaw through an MCP adapter and lifecycle hooks,
following the OpenGraphMemory pattern: the MCP server answers explicit
memory questions, and hooks make recall and capture automatic.

```bash
cargo install --path . --features openclaw
lint-ai --openclaw-install /path/to/project
```

The installer is idempotent and preserves existing configuration. It writes:

- an MCP server entry into `~/.openclaw/openclaw.json` (`mcp.servers.lint-ai`,
  running `lint-ai --openclaw-serve <project-root>` over stdio),
- the recall hook into `<stateDir>/hooks/lint-ai/` (`~/.openclaw` by default,
  `OPENCLAW_STATE_DIR` overrides it),
- the capture plugin into `<configDir>/extensions/lint-ai/`,
- the `lint-ai-memory` skill into `~/.openclaw/skills/lint-ai-memory/SKILL.md`.

User-modified skills are preserved; `--openclaw-force-skill` replaces one
intentionally. Only the `command`/`args` keys verified against OpenClaw's MCP
docs are written — OpenClaw validates its config strictly, so no speculative
keys are added. Verify the setup with `lint-ai --openclaw-verify-mcp`, which
spawns the server and checks the MCP handshake.

## MCP tools

OpenClaw receives the same seven tools as the other stdio adapters:

`search`, `info`, `list_memories`, `record_session`, `enable_lint_ai`,
`disable_lint_ai`, `lint_ai_status`.

Board and memory writes go through the persistent shared store
(`.lint-ai/memory/`) under the cross-process write lock, so they survive the
MCP process and are visible to hooks and other providers. Reads use the
composed in-memory view. A stdio MCP server cannot be shared as a URL — the
lint-ai binary must sit next to the OpenClaw Gateway.

## Lifecycle hooks

Hook schemas were verified against live OpenClaw 2026.9.6 payloads before
shipping. The installed `handler.js` is a thin wrapper: it serializes each
event to `lint-ai --openclaw-hook <event>` (JSON on stdin) and applies the
result. All lifecycle logic lives in the Rust binary.

| Hook | Lint-AI behavior |
| --- | --- |
| `agent:bootstrap` | Recalls relevant memories and injects them as `bootstrapFiles` (`LINTAI.md` is appended, replacing any stale entry so repeated firings stay idempotent). |
| `message:received` | Remembers the latest user text per session so the bootstrap query targets the actual request. |
| `agent_end` | Captures the run's outcome. |
| `before_reset` | Captures an authoritative `SessionSummary` from the full departing transcript before OpenClaw wipes the session. |
| `session_start` / `session_end` | Records session lifecycle links. |

The hooks are stateless: there is no hook-side `state.json`. The project root
is baked into the generated `handler.js` at install time, the event's
`workspaceDir` takes precedence when present, and stable document IDs make
retries idempotent. Compaction capture is intentionally not wired: the
compaction hooks were not observed on a live host. Every hook is fail-open —
a hook failure never blocks the agent turn.

## Memory model

Lint-AI is the project's long-term memory layer and is separate from
OpenClaw's own memory (the daily logs and `MEMORY.md` in
`~/.openclaw/workspace/`): OpenClaw's memory holds conversation notes, while
Lint-AI indexes the project workspace and carries decisions, outcomes, and
supersessions across sessions.

For requests about project history, prior decisions, architectural choices,
earlier work, or why something was implemented, the agent should call the
MCP server's `search` tool before reading files or searching the repository.
Use returned results as context, distinguish retrieved facts from current
source facts, and check cited files when current source details are required.
Recorded sessions are not authoritative documentation; verify conclusions
against the current project.

To capture the current session's work so future sessions can find it, use the
`record_session` tool (`start` at the beginning of a work session, `stop` at
the end). Recording is capture-only: it never changes the workspace, and a
bad memory day never blocks normal work.
