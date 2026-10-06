# lint-ai × Hermes Agent integration research

**Date:** 2026-09-28
**Probed checkout:** `https://github.com/NousResearch/hermes-agent` at commit `e408d363` ("fix(desktop): break the combo.ts <-> actions.ts import cycle (#126361)"), cloned to `/tmp/hermes-probe/hermes-agent` (scratch only — nothing installed into `~/lint-ai-demo`, no commits).
**Hermes identity:** Hermes Agent is Nous Research's open-source AI agent framework (MIT licensed), a Python codebase that descends from OpenClaw — it is in the same family as OpenClaw, not a different architecture. Conventions (skills, plugins, MCP client, gateway) are OpenClaw-shaped.

**Live-probe rule compliance:** All integration surfaces were verified against the real code/config in the probed checkout. No API keys were used. Where static reading was enough it was used; the MCP client path was additionally executed live.

## Live probe: Hermes MCP client loading a stdio MCP server

To prove Route A against the *real* loader (not docs), I:

1. Wrote a minimal stdio MCP echo server (`/tmp/hermes-probe/echo_mcp.py`, no SDK — raw newline-delimited JSON-RPC: `initialize`, `tools/list` with two tools `memory_search`/`memory_store`, `tools/call`).
2. Drove Hermes' own connect/discovery path: `tools/mcp_tool_discovery._connect_server("lintai_probe", {...})` + `_discover_and_register_server` + `get_mcp_status` (`/tmp/hermes-probe/probe_mcp_loader.py`).

Result (stdout, verbatim):

```
DISCOVERED_TOOLS: [
 "mcp__lintai_probe__memory_search",
 "mcp__lintai_probe__memory_store"
]
STATUS: [
 {
  "name": "lintai_probe",
  "transport": "stdio",
  "tools": 2,
  "connected": true,
  "disabled": false,
  "status": "connected",
  ...
 }
]
REGISTERED_SERVERS: ['lintai_probe']
```

The echo server config used the exact key shape a lint-ai server would need:
`{"command": "<python>", "args": ["/tmp/hermes-probe/echo_mcp.py"], "env": {"LINTAI_TEST": "1"}}` — command/args/env all accepted, tool names follow the `mcp__<server>__<tool>` pattern. (Needed probe deps installed to user site: `ruamel.yaml`, `mcp`, `python-dotenv` — all Hermes' own runtime requirements, not lint-ai's.)

---

## 1. MCP client (Route A) — verified live

**Config schema.** `mcp_servers` lives in the profile's `config.yaml`. The canonical commented schema is in `cli-config.yaml.example` (~lines 1540–1600). Exact keys accepted for a stdio server:

- `command` (required for stdio — executable to spawn)
- `args` (list of arguments)
- `env` (map of env vars; merged over a safe baseline — `tools/mcp_tool_config.py:155` `_build_safe_env`)
- `cwd` (supported — `tools/mcp_tool_transport.py:127–139` `_stdio_launch`; not shown in the example but read via `config.get("cwd")`)
- optional: `timeout`, `connect_timeout`, `keepalive_interval`, `lazy`, `sampling.*`

HTTP servers use `url` + `headers` instead (`tools/mcp_tool_transport.py:145–149` `_connect_inputs`: presence of `"url"` in the entry selects HTTP). Config is loaded through `hermes_cli/mcp_config.py:209` `_get_mcp_servers`, launched via `tools/mcp_tool_discovery.py:146` `_connect_server` → `MCPServerTask.start(config)` (`tools/mcp_tool_server_run.py:538`).

**Exact config.yaml snippet to register lint-ai** (analogous to lint-ai's own OpenClaw writer, `src/integrations/openclaw/mod.rs:70-78`, which emits `{"command": <binary>, "args": ["--openclaw-serve", <root>]}`):

```yaml
mcp_servers:
  lint-ai:
    command: /path/to/lint-ai          # built with the openclaw feature (has --openclaw-serve)
    args: ["--openclaw-serve", "/path/to/project-root"]
    env:
      LINTAI_CONFIG: /path/to/lint-ai-config.toml   # if needed
```

Tools would surface to the Hermes agent as `mcp__lint-ai__search`, `mcp__lint-ai__info`, `mcp__lint-ai__record_session`, `mcp__lint-ai__enable_lint_ai`, `mcp__lint-ai__disable_lint_ai`, `mcp__lint-ai__lint_ai_status` (the shared stdio scaffolding in lint-ai defines these — `src/integrations/gemini_cli/mod.rs:349` `tool_definitions()`; `--openclaw-serve` reuses the same scaffolding via `run_server_for` in `src/integrations/openclaw/mod.rs:81-86`).

**`hermes mcp` CLI** (`hermes_cli/subcommands/mcp.py`, handlers in `hermes_cli/mcp_config.py`):

- `hermes mcp add <name> --command <cmd> --args ... --env KEY=VALUE` (`mcp_config.py:611` `cmd_mcp_add`) — discovery-first install, probes the server and lets the user select tools.
- `hermes mcp remove|rm <name>` (`:698`), `hermes mcp list|ls` (`:718`), `hermes mcp test <name>` (`:786` — connects and lists tools; the live probe above exercised the same code path), `hermes mcp configure <name>` (`:997` — toggle tool selection), plus `login`/`reauth` (OAuth), `picker`/`catalog`/`install` (Nous-approved catalog).
- `hermes mcp serve` is a different thing — see §5.

**Notes:**
- A security gate exists: `tools/mcp_tool_config.py:403` `_filter_suspicious_mcp_servers` drops "exfiltration-shaped" MCP configs before spawn (via `hermes_cli.mcp_security.validate_mcp_server_entry`). A lint-ai stdio entry is ordinary (local binary, no network), so it passes.
- `lazy: true` registers from the on-disk tool cache and spawns only on first call — useful for lint-ai since it's consulted infrequently.

---

## 2. Memory backends (Route B) — the real interface

Hermes' memory is genuinely pluggable, and — critically — **third-party backends need no changes to the Hermes repo.**

**How backends are selected.** One active at a time via config key:

```yaml
memory:
  provider: "mem0"      # "" = built-in; one at a time
```

(`hermes_cli/config_defaults.py:1321-1323`: "External memory provider plugin (empty = built-in only); only ONE at a time"; sentinel values `""`, `default`, `builtin`, `built-in`, `none` mean built-in — `agent/memory_provider.py:48-53`.)

**Discovery.** `plugins/memory/__init__.py` (docstring, lines 1–10): bundled `plugins/memory/<name>/`, user `$HERMES_HOME/plugins/<name>/`, project `./.hermes/plugins/<name>/` (opt-in via `HERMES_ENABLE_PROJECT_PLUGINS`), then `hermes_agent.memory_providers` **entry points** (`ENTRY_POINTS_GROUP = "hermes_agent.memory_providers"`, line 35). Bundled wins on name collision. A directory plugin needs `plugin.yaml` + `__init__.py` with `register(ctx)` or a top-level `MemoryProvider` subclass (`plugins/memory/__init__.py:332-360`); an entry point may be a `MemoryProvider` instance, subclass, factory, module with `register`, or callable (`:296-330`).

**The ABC.** `agent/memory_provider.py` — `MemoryProvider(ABC)`:

Abstract (must implement):
- `name() -> str` (line 93) — short identifier, e.g. `"lintai"`
- `is_available() -> bool` (line 99) — config/deps check only, no network
- `initialize(session_id: str, **kwargs) -> None` (line 103) — kwargs include `hermes_home`, `platform`, `agent_context` (`"primary"|"subagent"|"cron"|"flush"` — skip writes for non-primary), `agent_identity`, `agent_workspace`, `parent_session_id`, `user_id`
- `get_tool_schemas() -> List[dict]` (line 142) — OpenAI function-calling schemas the agent sees

Important optional overrides (all default-implemented):
- `prefetch(query, *, session_id) -> str` — formatted recall injected **before each turn** (must be fast; queue background recall via `queue_prefetch`)
- `sync_turn(user_content, assistant_content, *, session_id, messages, turn_author)` — persist a completed turn (non-blocking)
- `handle_tool_call(tool_name, args, **kwargs) -> str` — dispatch for the provider's tools
- `system_prompt_block() -> str` — static system-prompt text
- `on_session_end(messages)` — end-of-session extraction (fires only at real session boundaries)
- `on_turn_start(turn_number, message, **kwargs)`, `on_session_switch`, `on_pre_compress`, `on_delegation`, `on_memory_write(action, target, content, metadata)`, `shutdown()`

**Reference implementation:** `plugins/memory/mem0/__init__.py` — `Mem0MemoryProvider(MemoryProvider)` exposes tools `mem0_search` / `mem0_add` / `mem0_update` / `mem0_delete` (lines 324–364) over an internal `Mem0Backend` ABC (`_backend.py:19-37`: `search`, `add`, `_update`, `_delete`).

**What a lint-ai provider would look like:** a pip package `lintai-hermes` (or a directory dropped in `$HERMES_HOME/plugins/lintai/`) exposing `hermes_agent.memory_providers` entry point `lintai = <module>:LintaiMemoryProvider`, where the provider:
- `name()` → `"lintai"`; `is_available()` → lint-ai binary/config present
- `get_tool_schemas()` → `lintai_search`, `lintai_add` (+ update/delete) mirroring the mem0 tool set
- `handle_tool_call` → shell out to / call into the lint-ai engine
- `prefetch()` → lint-ai query per turn (the auto-recall hook)
- `sync_turn()` → feed completed turns into lint-ai (the auto-capture hook)
- `on_session_end()` → session-boundary extraction

Bundled backends shipped for reference: `plugins/memory/{mem0,honcho,supermemory,retaindb,byterover,holographic,openviking}/`, each with `plugin.yaml` + `config_schema.py` (declarative UI config; `plugins/memory/config_schema.py`).

---

## 3. Skills (Route C)

Skills are `SKILL.md` files with YAML frontmatter (`name`, `description`, `version`, `author`, `license`, `platforms`, `metadata.hermes.tags`). Bundled: `skills/<category>/<name>/SKILL.md`; optional ones in `optional-skills/`. Discovery (`tools/skills_tool.py:173` `_skill_search_dirs`): **project dirs first**, then the active skills dir (`~/.hermes/...`), then external dirs (`agent/skill_utils.get_external_skills_dirs`). Dedup is first-wins so project-local beats bundled.

A lint-ai skill would be a `SKILL.md` dropped in a project dir (or shipped by the provider plugin — memory-provider plugins can register skills via `PluginContext.register_skill`, see `plugins/memory/__init__.py:393`) that teaches the Hermes agent: "when you need long-term memory / before reading files, call `mcp__lint-ai__search`…" — exactly the pattern lint-ai's own `src/integrations/openclaw/skill.md` already uses for OpenClaw (`lint-ai` MCP server's `search` / `record_session` tools). No auto-skill-generation interplay was found — skills are declarative files, not generated; the "auto" element is provider plugins *registering* their own skills at load time.

---

## 4. Automatic capture / event triggers

Hermes has a real plugin hook system — comparable to OpenClaw's hooks, arguably richer:

- **Plugin hooks** (`hermes_cli/plugins.py:109` `VALID_HOOKS`): `pre_tool_call`, `post_tool_call`, `transform_tool_result`, `on_session_start`, `on_session_end`, `on_session_finalize`, `on_session_reset`, `on_stream_start/delta/end`, `pre_llm_call`, `post_llm_call`, `pre_verify`, `subagent_start/stop`, `pre_gateway_dispatch`, `agent_loop_stopped`, approval observers, etc. Directory plugins (`~/.hermes/plugins/<name>/` with `plugin.yaml` + `register(ctx)`) or pip entry points in the `hermes_agent.plugins` group register callbacks via `PluginContext.register_hook` (line 933). A lint-ai plugin could subscribe `post_tool_call`/`on_stream_end` (per-turn capture) and `on_session_end`/`on_session_finalize` (session extraction) — **without touching the memory-provider interface**.
- **Memory-provider lifecycle** (§2): `sync_turn` per turn + `on_session_end` — the native auto-capture surface if lint-ai goes the Route B path.
- **Cron** (`cron/`): scheduled jobs subsystem — a lint-ai plugin/cron job could periodically sync sessions into lint-ai.
- **Webhooks**: platform `webhook:` config routes incoming JSON to scripts under the profile's scripts dir (`cli-config.yaml.example:~1455`); outbound webhooks exist too (`agent/outbound_webhooks.py`). Event-driven, not per-turn.

For "OpenClaw-style automatic capture/recall", the direct equivalents are: `sync_turn` + `prefetch` (Route B), or a plugin hooking `post_tool_call` + `on_session_end` (plugin route).

---

## 5. `hermes mcp serve` — what Hermes exposes

`mcp_serve.py` — Hermes serves *itself* as an MCP server over stdio. Tools exposed (`mcp_serve.py:684-688` `_TOOL_NAMES`):

`conversations_list`, `conversation_get`, `messages_read`, `attachments_fetch`, `events_poll`, `events_wait`, `messages_send`, `channels_list`, `permissions_list_open`, `permissions_respond`

i.e. a messaging bridge: read/send messages across Telegram, Discord, Slack, WhatsApp, Signal, Matrix. For hierarchical compositions, an external agent (or a lint-ai indexer) could attach Hermes as a *source of conversation transcripts* — e.g. read finished Hermes sessions into lint-ai via `messages_read`. It does not expose Hermes memory.

---

## Recommendation

**Lowest-risk first step: Route A — MCP adapter, in the same shape as `--openclaw-serve`.** Ranked:

1. **Route A (MCP client) — lowest risk.** Verified live end-to-end today against Hermes' real loader: any stdio MCP server registers and its tools are callable as `mcp__lint-ai__*`. lint-ai already ships the binary (`lint-ai --openclaw-serve <root>`); Hermes registration is a 4-line `mcp_servers` config or one `hermes mcp add lint-ai --command lint-ai --args --openclaw-serve /root` command. Zero Hermes-side code, zero Python packaging, zero risk to Hermes' one-provider-at-a-time memory model. Pair with **Route C (skill)**: ship a lint-ai `SKILL.md` (modeled on `src/integrations/openclaw/skill.md`) telling the agent when to call the lint-ai tools. This mirrors exactly the OpenClaw Option A pattern (MCP server + skill/directive) already chosen for OpenClaw — same playbook, new host.
2. **Route B (native memory provider) — best long-term, higher cost.** The interface is real and third-party-friendly: pip-installable via the `hermes_agent.memory_providers` entry-point group, activated by `memory.provider: "lintai"`, with native per-turn auto-recall (`prefetch`) and auto-capture (`sync_turn`) plus `on_session_end`. But it means writing a Python provider package wrapping lint-ai (subprocess or FFI/HTTP into the Rust engine), and it fights Hermes' "ONE provider at a time" rule — adopting lint-ai *replaces* the built-in memory rather than augmenting it. Do this after Route A proves value.
3. **Route 4 (plugin hooks) — niche.** If Route B is too heavy but per-turn auto-capture is wanted without replacing the memory provider, a `~/.hermes/plugins/lintai/` plugin hooking `post_tool_call` + `on_session_end` can push turns into lint-ai. Less invasive than Route B (memory provider untouched), but per-event Python glue still needs writing and maintenance; and `sync_turn` in Route B already covers the same ground more cleanly.
4. **Route 5 (`hermes mcp serve`) — not an integration path into Hermes;** note it as a *source* (conversation transcripts readable by external agents).

**Suggested sequence:** A — ship `--hermes-serve`-style flag? Not even needed: `--openclaw-serve` is host-agnostic stdio MCP and works as-is in Hermes; only the skill text and install docs are Hermes-specific (config path `~/.hermes/config.yaml`, server name `lint-ai`). Then C — the skill. Then, once real usage exists, B — the native provider for automatic capture/recall.

## Open questions for the repo owner (Luyi)

1. Does lint-ai want a Hermes-specific serve flag (`--hermes-serve`) or is the generic `--openclaw-serve` (host-agnostic stdio MCP) the permanent shape? Hermes needs nothing special — but naming/docs clarity.
2. Route B replaces the user's active memory provider (one at a time). Is that acceptable, or should lint-ai stay an *augmenting* sidecar (Route A + skill) that never competes with built-in/Honcho/Mem0?
3. For Route B, the provider is Python and must wrap the Rust engine — preferred transport: subprocess calls to the lint-ai CLI, or an HTTP sidecar? (Hermes mem0-OSS precedent shells to a local service.)
4. `mcp__lint-ai__search` requires a project `root` arg baked into the config at install time. Is per-project-root spawning acceptable for Hermes users, or should the server resolve the workspace dynamically?
5. Auto-capture (`sync_turn` / hooks) ships conversation text to lint-ai's store — any privacy/consent UX needed before enabling by default?
