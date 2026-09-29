# Hermes integration

Lint-AI supports Hermes Agent at two levels. The `--hermes-serve` MCP
adapter makes memory available when the agent *chooses* to call a tool; the
`hermes-plugin-lintai` hooks plugin makes capture and recall *automatic*.

## Level 1 — MCP adapter

```bash
cargo install --path . --features hermes
lint-ai --hermes-install /path/to/project
```

The installer merges the stdio MCP server entry into `~/.hermes/config.yaml`
under `mcp_servers` (as `lint-ai`, running
`lint-ai --hermes-serve <project-root>`) and installs the `lint-ai-memory`
skill. Installation is idempotent and preserves existing configuration,
including a non-lint-ai `memory.provider` setting. Verify with
`lint-ai --hermes-verify-mcp`; the adapter was validated against the real
Hermes CLI (`hermes mcp test` reports Connected, all seven tools enabled).

Hermes receives the same seven tools as the other stdio adapters —
`search`, `info`, `list_memories`, `record_session`, `enable_lint_ai`,
`disable_lint_ai`, `lint_ai_status` — surfaced to the agent as
`mcp__lint-ai__*`. Tool descriptions carry the Hermes name; no other
adapter's branding leaks through. Board and memory writes go through the
persistent shared store (`.lint-ai/memory/`) under the cross-process write
lock, so they survive the MCP process and are visible to hooks and other
providers.

## Level 2 — hooks plugin (automatic capture/recall)

`integrations/hermes-plugin-lintai/` is a Hermes directory plugin
(`plugin.yaml` + `__init__.py`, stdlib-only Python) that wires lint-ai
memory into the agent's turn loop with zero changes to Hermes and zero
changes to the lint-ai Rust core. The full design, with live-verified hook
payload schemas, is in `integrations/hermes-plugin-lintai/DESIGN.md`.

| Hook | Lint-AI behavior |
| --- | --- |
| `pre_llm_call` | Recall + inject: returns `{"context": ...}`, which Hermes stamps into the user message. Synchronous with a tight timeout, fail-open. |
| `post_tool_call` | Structured tool-event capture (name, args, result, duration, status), deduplicated by `tool_call_id`. |
| `post_llm_call` | Per-turn transcript capture, deduplicated by `(session_id, turn_id)`. |
| `on_session_start` | Session registry entry. |
| `on_session_finalize` / `on_session_reset` | Boundary markers (no transcript; per-turn accumulation is authoritative). |

The plugin deliberately does not use Hermes' native `MemoryProvider` slot:
occupying it would evict the user's existing mem0/file memory. Hooks coexist
with everything.

### Transport

Hermes hook handlers must be Python running inside the Hermes process, so the
plugin cannot call the Rust core in-process. It talks HTTP to a running
`lint-ai serve` instance (default `127.0.0.1:8080`) over a persistent
keep-alive connection, using the existing `POST /search` and
`POST /add/batch` endpoints — the same pattern mem0's own Hermes plugin uses.
Writes go through an async bounded queue (drop-oldest) with one retry on
transient errors and a cooperative flush on session boundaries. Dedupe is
stateless: request IDs are the idempotency keys, and timestamps are omitted
so hook replays are byte-identical. Every handler is fail-open: if the server
is unreachable, the agent simply gets no automatic memory — it never breaks.

## Memory model

Lint-AI is the project's long-term memory layer and is separate from Hermes'
own memory (the built-in provider's notes and skills): Hermes' memory holds
conversation notes, while Lint-AI indexes the project workspace and carries
decisions, outcomes, and supersessions across sessions.

For requests about project history, prior decisions, architectural choices,
earlier work, or why something was implemented, the agent should call the
`mcp__lint-ai__search` tool before reading files or searching the repository.
Use returned results as context, distinguish retrieved facts from current
source facts, and check cited files when current source details are required.
Say plainly when memory returns nothing. Recorded sessions are not
authoritative documentation; verify conclusions against the current project.

To capture the current session's work so future sessions can find it, use the
`mcp__lint-ai__record_session` tool (`start` at the beginning of a work
session, `stop` at the end). Recording is capture-only: it never changes the
workspace, and a bad memory day never blocks normal work.
