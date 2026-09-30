# Tutorial: add Lint-AI memory to your OpenClaw setup

This guide is for people already running OpenClaw. In about ten minutes you
will give your OpenClaw agents long-term project memory: relevant memories
injected at the start of every agent run, and automatic capture of outcomes
and session summaries.

## What you get

- **Recall on every run.** When an agent starts (`agent:bootstrap`), Lint-AI
  recalls relevant project memories and injects them, so the agent knows
  prior decisions and earlier work without being told.
- **Automatic capture.** When a run ends (`agent_end`), its outcome is
  recorded; when a session closes (`before_reset`), a session summary is
  captured from the full transcript.
- **An MCP server** with `search`, `record_session`, board, and memory tools
  the agent can call directly.

Lint-AI is separate from OpenClaw's own memory (the daily logs and
`MEMORY.md` in `~/.openclaw/workspace/`): OpenClaw's memory holds
conversation notes, while Lint-AI indexes the project workspace and carries
decisions, outcomes, and supersessions across sessions.

## Step 1 — Install the lint-ai binary

```bash
cargo install --path . --features openclaw
```

This builds the `lint-ai` binary with the OpenClaw adapter. (A release
binary works too — the rest of this guide only needs the binary on your
`PATH`, next to where the OpenClaw Gateway runs, since the MCP server runs
over stdio.)

## Step 2 — Run the installer

```bash
lint-ai --openclaw-install /path/to/project
```

This one command wires everything up, idempotently (safe to re-run; existing
configuration is preserved):

- registers the MCP server in `~/.openclaw/openclaw.json`
  (`mcp.servers.lint-ai`, running `lint-ai --openclaw-serve <project-root>`),
- installs the recall hook into `<stateDir>/hooks/lint-ai/` (`~/.openclaw`
  by default, `OPENCLAW_STATE_DIR` overrides it),
- installs the capture plugin into `<configDir>/extensions/lint-ai/`,
- installs the `lint-ai-memory` skill into
  `~/.openclaw/skills/lint-ai-memory/SKILL.md` (a user-modified skill is
  left alone unless you pass `--openclaw-force-skill`).

## Step 3 — Restart the Gateway

Restart your OpenClaw Gateway so it picks up the new MCP server entry, then
verify the handshake:

```bash
lint-ai --openclaw-verify-mcp
```

## Step 4 — Check it working

Start any agent run in your project. Lint-AI injects recalled memories at
bootstrap automatically — you don't need to change your prompts. To see the
memory tools in action, ask the agent something only a past session would
know, e.g. *"What did we decide about the API layout last week?"* The agent
calls the MCP `search` tool before reading files.

To capture the current session deliberately, the agent can use the
`record_session` tool (`start` at the beginning of a work session, `stop` at
the end). Recording is capture-only: it never changes the workspace.

## How it behaves

- **Stateless hooks.** There is no hook-side state file. The project root is
  baked into the installed hook at install time, and the event's
  `workspaceDir` takes precedence when present.
- **Idempotent.** Stable document IDs mean a retried hook never duplicates a
  memory, and repeated bootstrap firings replace rather than pile up the
  injected file.
- **Fail-open.** A hook failure never blocks the agent turn; if Lint-AI is
  unreachable, the agent simply runs without injected memory.

## Troubleshooting

- **No memories injected:** run `lint-ai --openclaw-verify-mcp` to check the
  server handshake, and confirm the Gateway was restarted after install.
- **Hook not firing:** check `<stateDir>/hooks/lint-ai/handler.js` exists
  and that `OPENCLAW_STATE_DIR` matches the state dir your Gateway uses.
- **Wrong project root:** the event's `workspaceDir` wins; otherwise the
  install-time root is used. Re-run the installer with the right path if the
  project moved.

## Reference

The seven MCP tools: `search`, `info`, `list_memories`, `record_session`,
`enable_lint_ai`, `disable_lint_ai`, `lint_ai_status`, plus board tools
(`board_open`, `board_post`, `board_read`, …). Board and memory writes go
through the persistent shared store (`.lint-ai/memory/`) under a
cross-process write lock, so they survive the MCP process and are visible to
hooks and other providers. Hook schemas were verified against live OpenClaw
2026.9.6 payloads; compaction capture is intentionally not wired, since those
hooks were not observed on a live host.
