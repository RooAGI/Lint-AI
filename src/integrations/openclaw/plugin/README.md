# lint-ai — OpenClaw plugin

Automatic Lint-AI memory for OpenClaw: captures agent outcomes and session
summaries at lifecycle points, so project decisions and results persist
across sessions.

## What it does

This is a thin observer. On OpenClaw lifecycle events it forwards a JSON
payload to the `lint-ai` binary (`lint-ai --openclaw-hook <event>`); all
capture and indexing logic lives in that binary. It does nothing else.

| Hook | Behavior |
| --- | --- |
| `agent_end` | Captures the run's outcome. |
| `before_reset` | Captures a session summary from the departing transcript. |
| `session_start` / `session_end` | Session lifecycle bookkeeping. |
| `gateway_stop` | Best-effort final flush inside OpenClaw's drain budget. |

Compaction capture is intentionally not wired: the compaction hooks were not
observed on a live OpenClaw host (verified against 2026.9.6 payloads).

## Requirements

- OpenClaw with plugin support.
- The `lint-ai` binary, version 0.2.1 or later, resolvable at runtime —
  see "Binary resolution" below. Release binaries are published at
  https://github.com/RooAGI/Lint-AI/releases.

## Install

Via ClawHub (recommended):

```bash
openclaw plugins install clawhub:lintai
openclaw plugins enable lint-ai
```

Via the lint-ai installer (same plugin, plus MCP server, hooks, and skill):

```bash
lint-ai --openclaw-install /path/to/project
```

## Configuration

Optional keys under `plugins.entries.lint-ai.config` in `openclaw.json`:

| Key | Meaning | Default |
| --- | --- | --- |
| `projectRoot` | Project root lint-ai captures for | The event's `workspaceDir` |
| `binaryPath` | Absolute path to the `lint-ai` binary | `LINT_AI_BIN`, then `lint-ai` on `PATH` |

`--openclaw-install` writes `projectRoot` for you. When installing via
ClawHub, set `projectRoot` if your Gateway runs multiple workspaces and the
events don't carry `workspaceDir`.

## Binary resolution

In order: plugin config `binaryPath` → `LINT_AI_BIN` environment variable →
`lint-ai` found on `PATH` (no shell). If none resolves, the plugin logs one
line to stderr and every event is skipped — it never throws into the host.

## Fail-open behavior

Every failure path is silent by design: a missing binary, an unreachable
project root, or a hook-process error never blocks the agent loop or the
Gateway. The worst case is simply "no automatic memory".

## Project root resolution

In order: plugin config `projectRoot` → `ctx.workspaceDir` →
`event.context.workspaceDir` → `event.workspaceDir`. If none is present the
event is skipped.
