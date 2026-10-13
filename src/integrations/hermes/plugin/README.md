# Membrick memory plugin for Hermes

Automatic membrick memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent)
via hooks — recall injected before every model call, per-turn, per-tool-call, and
subagent lifecycle capture.
The "next level" after the `--hermes-serve` MCP adapter: the MCP adapter makes memory
available when the agent *chooses* to call a tool; this plugin makes capture/recall
*automatic*.

Read [`DESIGN.md`](DESIGN.md) first — it documents the hook inventory (verified
against real Hermes payloads), the mapping, dedupe, transport, and open questions.

## Requirements

- Hermes Agent installed (`hermes` on PATH)
- A running membrick server: `membrick serve` (default `http://127.0.0.1:8080`).
  The plugin is Python running inside Hermes' process, so it talks to the server
  over HTTP keep-alive — it cannot call the Rust core in-process.

## Install

```bash
# from the membrick repo
hermes plugins install ./src/integrations/hermes/plugin
hermes plugins enable membrick
```

Or per-project (no global install):

```bash
mkdir -p .hermes/plugins
cp -r /path/to/membrick/src/integrations/hermes/plugin .hermes/plugins/membrick
HERMES_ENABLE_PROJECT_PLUGINS=1 hermes chat
```

Then start the server (separate terminal):

```bash
membrick serve   # listens on 127.0.0.1:8080
```

## Configuration

Environment variables (win over `$HERMES_HOME/membrick.json`, which wins over defaults):

| Variable | Default | Meaning |
|---|---|---|
| `MEMBRICK_SERVER_URL` | `http://127.0.0.1:8080` | Membrick server base URL |
| `MEMBRICK_SERVER_TOKEN` | empty | Optional server bearer token |
| `MEMBRICK_USER_ID` | `hermes` | tenant/user id for all reads and writes |
| `MEMBRICK_QUEUE_MAX` | `1000` | bounded async write queue; drops oldest when full |
| `MEMBRICK_CAPTURE` | `on` | `off` disables all capture hooks |
| `MEMBRICK_RECALL` | `on` | `off` disables recall injection |
| `MEMBRICK_RECALL_TOP_K` | `5` | search hits injected per turn |

The previous `LINTAI_*` environment variables and `lintai.json` file are still
accepted for existing installations.

The plugin writes through `/provider-memory/add/batch` into the workspace's
shared `.membrick/memory/` store. Recall uses `/provider-memory/search`, which
searches provider memory together with the workspace index. OpenClaw lifecycle
captures use the same server and provider store.

If the server is unreachable the plugin fails open: no injection, no capture, and
Hermes keeps working.

## What it writes

- **Turn records** (`hermes:turn:<session>:<turn>`) — user message + assistant
  response per turn, deduped by `(session_id, turn_id)`.
- **Tool-event records** (`hermes:tool:<tool_call_id>`) — function name, args,
  result, duration, status per tool call.
- **Session records** (`hermes:session:<session>`) — registry entry on start,
  rewritten with a close marker on finalize/reset.
- **Interruption markers** — session key, platform, and interruption reasons
  from Hermes `agent_loop_stopped` events (gateway and TUI/desktop only; no
  transcript content).

Sessions are namespaced as `hermes:<session_id>` so they never collide with other
integrations' data.

## Tests

```bash
cd src/integrations/hermes/plugin
python3 -m unittest discover -s tests -v
```

Stdlib only — no dependencies.
