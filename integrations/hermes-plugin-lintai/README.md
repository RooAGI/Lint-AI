# hermes-plugin-lintai

Automatic lint-ai memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent)
via hooks — recall injected before every model call, per-turn and per-tool-call capture.
The "next level" after the `--hermes-serve` MCP adapter: the MCP adapter makes memory
available when the agent *chooses* to call a tool; this plugin makes capture/recall
*automatic*.

Read [`DESIGN.md`](DESIGN.md) first — it documents the hook inventory (verified
against real Hermes payloads), the mapping, dedupe, transport, and open questions.

## Requirements

- Hermes Agent installed (`hermes` on PATH)
- A running lint-ai server: `lint-ai serve` (default `http://127.0.0.1:8080`).
  The plugin is Python running inside Hermes' process, so it talks to the server
  over HTTP keep-alive — it cannot call the Rust core in-process.

## Install

```bash
# from the lint-ai repo
hermes plugins install ./integrations/hermes-plugin-lintai
hermes plugins enable lintai
```

Or per-project (no global install):

```bash
mkdir -p .hermes/plugins
cp -r /path/to/lint-ai/integrations/hermes-plugin-lintai .hermes/plugins/lintai
HERMES_ENABLE_PROJECT_PLUGINS=1 hermes chat
```

Then start the server (separate terminal):

```bash
lint-ai serve   # listens on 127.0.0.1:8080
```

## Configuration

Environment variables (win over `$HERMES_HOME/lintai.json`, which wins over defaults):

| Variable | Default | Meaning |
|---|---|---|
| `LINTAI_SERVER_URL` | `http://127.0.0.1:8080` | lint-ai server base URL |
| `LINTAI_USER_ID` | `hermes` | tenant/user id for all reads and writes |
| `LINTAI_QUEUE_MAX` | `1000` | bounded async write queue; drops oldest when full |
| `LINTAI_CAPTURE` | `on` | `off` disables all capture hooks |
| `LINTAI_RECALL` | `on` | `off` disables recall injection |
| `LINTAI_RECALL_TOP_K` | `5` | search hits injected per turn |

If the server is unreachable the plugin fails open: no injection, no capture, and
Hermes keeps working.

## What it writes

- **Turn records** (`hermes:turn:<session>:<turn>`) — user message + assistant
  response per turn, deduped by `(session_id, turn_id)`.
- **Tool-event records** (`hermes:tool:<tool_call_id>`) — function name, args,
  result, duration, status per tool call.
- **Session records** (`hermes:session:<session>`) — registry entry on start,
  rewritten with a close marker on finalize/reset.

Sessions are namespaced as `hermes:<session_id>` so they never collide with other
integrations' data.

## Tests

```bash
cd integrations/hermes-plugin-lintai
python3 -m unittest discover -s tests -v
```

Stdlib only — no dependencies.
