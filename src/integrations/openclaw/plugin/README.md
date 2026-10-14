# membrick — OpenClaw plugin

Automatic Membrick memory for OpenClaw: captures agent outcomes and session
summaries at lifecycle points, so project decisions and results persist
across sessions.

## What it does

This is a thin observer. On OpenClaw lifecycle events it forwards a JSON
payload to the `membrick` binary (`membrick --openclaw-hook <event>`); all
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
- The `membrick` binary, version 0.4.0 or later, resolvable at runtime —
  see "Binary resolution" below. Release binaries are published at
  https://github.com/RooAGI/Membrick/releases.

## Install

Use the Membrick installer to configure the plugin together with its MCP
server, hooks, and skill:

```bash
membrick --openclaw-install /path/to/project
```

After the plugin is published to ClawHub, it can also be installed with
`openclaw plugins install clawhub:membrick` and enabled with
`openclaw plugins enable membrick`.

## Configuration

Optional keys under `plugins.entries.membrick.config` in `openclaw.json`:

| Key | Meaning | Default |
| --- | --- | --- |
| `projectRoot` | Project root membrick captures for | The event's `workspaceDir` |
| `binaryPath` | Absolute path to the `membrick` binary | `membrick` on `PATH` |

`--openclaw-install` writes `projectRoot` for you. If your Gateway runs
multiple workspaces and events don't carry `workspaceDir`, set `projectRoot`
in the plugin configuration.

## Binary resolution

In order: plugin config `binaryPath` → `membrick` found on `PATH` (no shell).
If none resolves, the plugin logs one
line to stderr and every event is skipped — it never throws into the host.

## Fail-open behavior

Every failure path is silent by design: a missing binary, an unreachable
project root, or a hook-process error never blocks the agent loop or the
Gateway. The worst case is simply "no automatic memory".

## Project root resolution

In order: plugin config `projectRoot` → `ctx.workspaceDir` →
`event.context.workspaceDir` → `event.workspaceDir`. If none is present the
event is skipped.
