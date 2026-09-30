# Tutorial: add Lint-AI memory to your Hermes Agent setup

This guide is for people already running Hermes Agent. It covers two levels:
first the MCP adapter (memory the agent calls when it chooses), then the
hooks plugin (automatic capture and recall on every turn).

## What you get

- **Level 1 — MCP adapter.** Seven memory tools (`search`, `info`,
  `list_memories`, `record_session`, `enable_lint_ai`, `disable_lint_ai`,
  `lint_ai_status`) surfaced to the agent as `mcp__lint-ai__*`.
- **Level 2 — hooks plugin.** Recall injected before every model call, plus
  automatic capture of every turn and tool call — no tool calls required.

Lint-AI is separate from Hermes' own memory (the built-in provider's notes
and skills): Hermes' memory holds conversation notes, while Lint-AI indexes
the project workspace and carries decisions, outcomes, and supersessions
across sessions.

## Level 1 — MCP adapter

### Step 1 — Install the lint-ai binary

```bash
cargo install --path . --features hermes
```

### Step 2 — Register the MCP server

```bash
lint-ai --hermes-install /path/to/project
```

This merges the stdio MCP server entry into `~/.hermes/config.yaml` under
`mcp_servers` (as `lint-ai`) and installs the `lint-ai-memory` skill.
Installation is idempotent and preserves your existing configuration,
including a non-lint-ai `memory.provider` setting.

### Step 3 — Verify

```bash
lint-ai --hermes-verify-mcp
hermes mcp test
```

`hermes mcp test` should report Connected with all seven tools enabled.

### Step 4 — Use it

Ask the agent something only a past session would know, e.g. *"Why did we
pick this retry policy?"* The agent calls `mcp__lint-ai__search` before
reading files. Use `mcp__lint-ai__record_session` (`start`/`stop`) to capture
a work session deliberately. Recording is capture-only: it never changes the
workspace.

## Level 2 — hooks plugin (automatic memory)

The plugin lives at `integrations/hermes-plugin-lintai/` in the lint-ai
repo: a Hermes directory plugin (`plugin.yaml` + `__init__.py`, stdlib-only
Python, no dependencies). It deliberately does **not** use Hermes' native
`MemoryProvider` slot — occupying it would evict your existing mem0/file
memory. Hooks coexist with everything.

### Step 1 — Install the plugin

```bash
# from the lint-ai repo
hermes plugins install ./integrations/hermes-plugin-lintai
hermes plugins enable lintai
```

Or per-project, without a global install:

```bash
mkdir -p .hermes/plugins
cp -r /path/to/lint-ai/integrations/hermes-plugin-lintai .hermes/plugins/lintai
HERMES_ENABLE_PROJECT_PLUGINS=1 hermes chat
```

### Step 2 — Start the lint-ai server

The plugin is Python running inside the Hermes process, so it cannot call
the Rust core in-process — it talks HTTP to a running server instead (the
same pattern mem0's own Hermes plugin uses). In a separate terminal:

```bash
lint-ai serve   # listens on 127.0.0.1:8080
```

### Step 3 — Check it working

Chat with the agent for a turn or two, then ask it to recall something from
an earlier turn. Recall is injected before every model call (`pre_llm_call`
returns `{"context": ...}` which Hermes stamps into the user message);
turns, tool calls, and session boundaries are captured automatically in the
background.

## How the plugin behaves

| Hook | Behavior |
| --- | --- |
| `pre_llm_call` | Recall + inject. Synchronous with a tight timeout, fail-open. |
| `post_tool_call` | Structured tool-event capture (name, args, result, duration, status), deduplicated by `tool_call_id`. |
| `post_llm_call` | Per-turn transcript capture, deduplicated by `(session_id, turn_id)`. |
| `on_session_start` | Session registry entry. |
| `on_session_finalize` / `on_session_reset` | Boundary markers (no transcript; per-turn accumulation is authoritative). |

- **Fail-open, always.** If the server is unreachable, there is no injection
  and no capture — Hermes keeps working normally.
- **Bounded and batched.** Writes go through an async bounded queue
  (drop-oldest when full); a background thread batches them to the server.
- **Tunable** via environment variables (each beats
  `$HERMES_HOME/lintai.json`, which beats the defaults):

| Variable | Default | Meaning |
|---|---|---|
| `LINTAI_SERVER_URL` | `http://127.0.0.1:8080` | lint-ai server base URL |
| `LINTAI_USER_ID` | `hermes` | tenant/user id for reads and writes |
| `LINTAI_QUEUE_MAX` | `1000` | write queue size |
| `LINTAI_CAPTURE` | `on` | `off` disables capture hooks |
| `LINTAI_RECALL` | `on` | `off` disables recall injection |
| `LINTAI_RECALL_TOP_K` | `5` | search hits injected per turn |

## Troubleshooting

- **No tools in Hermes:** re-run `lint-ai --hermes-install` and check the
  `mcp_servers` section of `~/.hermes/config.yaml`; indentation matters in
  YAML.
- **Plugin installed but no memory:** confirm `lint-ai serve` is running and
  reachable at `LINTAI_SERVER_URL`; the plugin fails open, so a dead server
  looks like "no memory" rather than an error.
- **Verify the data:** captured turns, tool events, and session records land
  in the shared store (`.lint-ai/memory/`) under the `hermes` provider, so
  they are also searchable from the MCP adapter and other providers.

## Reference

The full hook inventory with live-verified payload schemas (probed against
hermes-agent `@ e408d363`), the dedupe design, and the verification plan are
in `integrations/hermes-plugin-lintai/DESIGN.md`. The plugin's own test
suite runs with `python3 -m unittest discover -s tests` inside the plugin
directory.
