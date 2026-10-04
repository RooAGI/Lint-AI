# Lint-AI × OpenClaw Integration Research
**Date:** 2026-09-28 · **Status:** research only, no code changes · **Author:** research subagent (for Luyi / Harbor)

---

## 1. TL;DR

OpenClaw (MIT-licensed TypeScript agent platform, ~steipete / OpenClaw Foundation) can consume **MCP servers** today via `mcp.servers` in `~/.openclaw/openclaw.json` (stdio, SSE, Streamable HTTP), and it ships a real **plugin SDK** with an official registry (**ClawHub**). Lint-AI's lowest-risk pattern — already established for Muse Code — is an MCP adapter: a new `--openclaw-serve <root>` flag on the `lint-ai` binary speaking standard MCP JSON-RPC over stdio, reusing the existing shared `mcp_tools`/`mcp_transport`/`mcp_index` scaffolding (the `agy` adapter is the smallest copy-template).

**Recommended first step:** ship `--openclaw-serve` + `--openclaw-install` as a lint-ai MCP adapter, and ship an agent prompt directive (plus a ClawHub *skill*) telling OpenClaw agents when to call `search`/`record_session`. This is the OpenGraphMemory-precedent pattern: MCP + prompt instructions, zero OpenClaw code changes. Hooks and a native plugin come later, only after probing hook/lifecycle entry schemas against a live OpenClaw install.

---

## 2. OpenClaw side — what it is and how it plugs in

### 2.1 Basics (verified)
- **What:** open-source, self-hosted AI assistant/agent platform — a local **Gateway** control plane that connects LLM agents (and harnesses like Claude, Codex, local models) to 20+ messaging channels, with tools, skills, plugins, scheduling, file-based memory.
- **Author:** Peter Steinberger (steipete); OpenClaw Foundation (independent 501(c)(3)); OpenAI is a donor, not owner.
- **Stack:** TypeScript, Node.js (Node 24.16+/26.1+; pnpm monorepo). **License: MIT.** Large, fast-moving project (~390k GitHub stars at check time; repo created 2025-11-24).
- **Architecture:** Gateway runs agents locally (control UI/WS on port 18789); **models and agent harnesses are plugins** (swappable); tools live in the Gateway (built-in, plugin-registered via `api.registerTool`, MCP-server tools via built-in MCP client), gated by tool policies/allowlist (`tools.alsoAllow`, tool profiles). State/memory/credentials stay on the host.
- **Sources:** https://github.com/openclaw/openclaw (README "How it fits together", Governance, License, Install, Security)

### 2.2 Memory / persistence today
- **Two-layer file-based memory** in `~/.openclaw/workspace/`: daily append-only logs `memory/YYYY-MM-DD.md` (today's + yesterday's auto-loaded at session start) + curated `MEMORY.md` (loaded only in main/private sessions, never group contexts — an explicit security design). Plus `AGENTS.md`, `SOUL.md`, `USER.md`, `TOOLS.md`, `IDENTITY.md`, `HEARTBEAT.md`, `BOOTSTRAP.md`, `BOOT.md`.
- **Semantic memory search built in:** agent tools `memory_search(query)` / `memory_get(path)`; hybrid BM25+vector index in **SQLite** (sqlite-vec, chunk embeddings cached). Providers configurable: `local` (GGUF via node-llama-cpp), `openai`, `openai-compatible`, `gemini`, `voyage`, `bedrock`, `copilot`. Two backends: `memory.backend: "builtin"` (default) and `"qmd"` (external indexer).
- **Session transcripts** persisted as `.jsonl`; sessions auto-reset daily/idle; bundled `session-memory` hook snapshots sessions on `command:new`/`command:reset`/`session:auto-reset` into `memory/`.
- **The gap for lint-ai:** OpenClaw's memory is markdown files + its own vector index; there is **no documented first-party "external memory backend" slot** — memory search is not exposed via Gateway RPC.
- **Sources:** https://github.com/it-huset/openclaw-guide/blob/HEAD/content/docs/phases/phase-2-memory.md · https://docs.openclaw.ai/reference/templates/AGENTS · https://github.com/kevinhamza/devin-4.0/blob/HEAD/repos/openclaw/docs/automation/hooks.md

### 2.3 Plugin / extension system
- **Full plugin SDK exists.** Manifest `openclaw.plugin.json` (required: `id`, `configSchema`), entry via `definePluginEntry()` with `api.registerTool`, `api.registerHook`/`api.on`, `api.registerChannel`, `api.registerGatewayMethod`, `api.registerHttpRoute`, `api.registerService`, `api.registerProvider`, `api.registerCommand`, `api.registerCli`. Plugins are trusted in-process TS/JS.
- **Install/loading:** `openclaw plugins install clawhub:<pkg> | npm:<pkg> | git:<owner>/<repo>@<ref> | --link ./dir`; enabled/gated via `plugins.allow` allowlist and `plugins.entries.<id>.enabled`.
- **Registry: YES — ClawHub** (openclaw/clawhub, clawhub.ai): official registry for **skills** and **plugins**, with semver, changelogs, downloads, security scan summaries, and a `clawhub` publish CLI.
- **Skills vs plugins:** skills are markdown prompt bundles (`SKILL.md` + frontmatter) — text only, no code; plugins are code packages. Skills load from project `skills/` → `~/.openclaw/skills/` → built-in.
- **Sources:** https://github.com/openclaw/clawhub/blob/HEAD/docs/clawhub.md · https://github.com/openclaw/clawhub/blob/HEAD/docs/quickstart.md · https://github.com/kevinhamza/devin-4.0/blob/HEAD/repos/openclaw/docs/plugins/sdk-overview.md · https://github.com/linux2010/openclaw/blob/HEAD/docs/tools/plugin.md · https://github.com/weaxs/stock-analysis-plugin/blob/HEAD/AGENTS.md (real plugin example)

### 2.4 MCP support — confirmed, first-class
- OpenClaw's built-in MCP client consumes servers over **Streamable HTTP, SSE, and stdio**, defined under **`mcp.servers`** in config. CLI (verbatim from official `docs/tools/mcp.md`):
  ```bash
  openclaw mcp add local-tools --command node --arg ./dist/mcp-server.js --cwd /srv/openclaw-tools
  openclaw mcp doctor local-tools --probe
  openclaw mcp add docs --url https://mcp.example.com/mcp --transport streamable-http --include 'search,read_*'
  ```
- Config shape (JSON5, `~/.openclaw/openclaw.json`):
  ```json5
  { mcp: { servers: {
      docs: { url: "https://mcp.example.com/mcp", transport: "streamable-http",
              enabled: true, connectionTimeoutMs: 5000, requestTimeoutMs: 20000,
              toolFilter: { include: ["search", "read_*"] } } } } }
  ```
  A stdio server needs a `command` instead of `url`; credentials go via `${ENV_VAR}` secret references, not literals. MCP tools flow through the same tool-policy/allowlist; Gateway hot-reloads changes; failed servers back off exponentially (30s → 10min). Reverse direction exists too (`openclaw mcp serve` exposes OpenClaw as an MCP server).
- **Precedent — OpenGraphMemory (real memory product, real OpenClaw integration):** adds an MCP server to config + an **agent prompt directive** telling the agent when to call `ogm_memory_search` / `ogm_record_code_fix`. I.e., *external memory via MCP + prompt instructions is the established pattern.*
- **Sources:** https://github.com/openclaw/openclaw/blob/main/docs/tools/mcp.md · https://github.com/ardiannurcahya/ogm-mcp-skills/blob/HEAD/docs/openclaw.md

### 2.5 Config files and hooks / lifecycle
- **Config:** `~/.openclaw/openclaw.json`, **JSON5** (comments, trailing commas, unquoted keys); override via `OPENCLAW_CONFIG_PATH` or `--profile`. Strict Zod validation — unknown keys prevent Gateway startup. Hot-reloads most changes. Modular via `$include` (deep-merge, 10 levels). Relevant sections: `agents.defaults.memorySearch`, `memory`, `mcp.servers`, `plugins`, `hooks.internal`, `tools`.
- **Startup:** `openclaw onboard` (wizard) → `openclaw gateway start` → loads config, channels, plugins/hooks, bootstraps workspace files.
- **Internal hooks** (file-based, `HOOK.md` + `handler.ts/js` in `<workspace>/hooks/` or `~/.openclaw/hooks/`, managed by `openclaw hooks list/enable/disable/check/info`). Verified event list: `command:new`, `command:reset`, `command:stop`, `command` (any), `session:auto-reset`, `session:compact:before/after`, `session:patch`, `agent:bootstrap` (can inject files into context), `gateway:startup`, `gateway:shutdown`, `gateway:pre-restart`, `message:received`, `message:transcribed`, `message:preprocessed`, `message:sent`.
- **Typed plugin hooks** (SDK `api.on(...)`, priority/merge/block-cancel semantics): `gateway_start`/`gateway_stop`, `before_agent_start`/`agent_end`, `before_tool_call`/`after_tool_call`/`tool_result_persist`, `message_received`/`message_sending`/`message_sent`, `before_compaction`/`after_compaction`, and a **`session_end` typed hook that fires per active session during gateway shutdown drain**, plus `before_agent_finalize`.
- **Memory-sync attach points:** `agent:bootstrap` (inject lint-ai recall into bootstrap context), `command:new`/`command:reset`/`session:auto-reset` (session boundaries — the bundled `session-memory` hook already uses these), `message:received`/`message:sent` (per-message, noisy), `gateway:startup`/`shutdown`.
- **Hook config snippet (verified):**
  ```json5
  { hooks: { internal: { enabled: true, entries: {
      "session-memory": { enabled: true, messages: 20 },
      "my-memory-sync": { enabled: true, env: { LINTAI_URL: "http://127.0.0.1:8099" } } },
    load: { extraDirs: ["/path/to/more/hooks"] } } } }
  ```
- **Sources:** https://github.com/kevinhamza/devin-4.0/blob/HEAD/repos/openclaw/docs/automation/hooks.md · https://github.com/aethonflame/memory-system/blob/HEAD/references/openclaw-hooks-research.md · https://github.com/shunkakinoki/dotfiles/blob/HEAD/generated/hooks/traces/openclaw/HOOK.md

### 2.6 Honest gaps (could not verify)
1. No `api.registerMemory(...)` or dedicated memory-plugin slot exists in any official SDK surface found — one third-party doc claims a "memory slot"; treat it as **unverified**. Safe assumption: memory backends are not a plugin slot.
2. The hooks/plugin-SDK pages were read from faithful third-party mirrors (kevinhamza/devin-4.0, linux2010/openclaw, kevincodex1/openclaw), not the canonical `openclaw/openclaw` docs copies; content is consistent across independent mirrors, but the canonical URLs were not loaded.
3. Whether `session:start`/`session:end` internal hook events have landed on latest `main` — a 2026.3.13-era research doc says "planned, not implemented"; newest mirror read (~2 weeks old) still omits them. Only the gateway-shutdown `session_end` *typed plugin* hook is confirmed.
4. Exact current default of `memory.backend` and the full `qmd` schema — sourced from a third-party guide, treat field names as indicative.
5. No behavioral details were tested against a live OpenClaw instance (hook payloads, `tool_result_persist` transform semantics, ClawHub security scans).

---

## 3. Lint-AI side — integration surfaces (local checkout `~/lint-ai-demo`)

### 3.1 Doc read: `~/workspace/lint-ai-pr/mcp-memory-server.md`
Marketing/overview page for the MCP Memory Server: claims 7 tools with identical names across adapters (`search`, `info`, `list_memories`, `record_session`, `enable_lint_ai`/`disable_lint_ai`, `lint_ai_status`), per-agent serve flags (`./lint-ai --claude-code-serve /path/to/repo`), installers that merge an MCP server entry into the host client's config, provider-local MCP server over the host's configured MCP transport, fail-open tool/hook behavior. Note: this doc **predates PR #81** — it does not mention `add_memory`/`get_memory`.

### 3.2 MCP servers — one adapter per agent (primary surface)
- **Entry points:** `src/integrations/{claude_code,codex,gemini_cli,agy}/mod.rs`; shared scaffolding in `src/integrations/mcp_tools.rs`, `mcp_index.rs`, `mcp_transport.rs`, `session_recording.rs`, `mcp_health.rs`.
- **CLI flags** (declared in `src/cli.rs`, dispatched in `src/engine.rs`):
  - `--<agent>-serve <path>` — stdio MCP server pinned to a project root (e.g. `if args.claude_code_serve { … run_server(Path::new(&args.path), …) }`).
  - `--<agent>-install <path>` — merges `{"command": <abs lint-ai path>, "args": ["--claude-code-serve", <root>]}` into `~/.claude.json` under `mcpServers.lint-ai`.
  - `--<agent>-verify-mcp` health-checks a spawned server (`mcp_health::verify`); `--<agent>-hook <event>` lifecycle hooks; `--<agent>-statusline`; `--<agent>-config/--<agent>-settings`.
- **Protocol:** standard MCP JSON-RPC over stdio. `mcp_transport.rs` reads line-delimited and `Content-Length`-framed requests; `initialize` → `protocolVersion: 2024-11-05`, `serverInfo: {name: "lint-ai", version: CARGO_PKG_VERSION}`; methods: `initialize`, `tools/list`, `tools/call`; unknown → -32601, bad args → -32602. Index builds lazily (`Mutex<Option<IndexStore>>` + `WorkspaceWatcher` rebuilds on FS changes) — the resident process stays cheap.
- **7 tools** (same names/schemas across adapters; `search` takes required `query`, `top_k` 1–20 clamped): `search`, `info`, `list_memories` (limit 1–100, default 20), `record_session` (start/stop/status, capture-only — never injects as memory), `enable_lint_ai`/`disable_lint_ai`, `lint_ai_status`. Results carry scores, sources, and a `semantic_status` verdict per hit.
- **Gap in this checkout:** no `add_memory`/`get_memory` here (grep finds nothing) — those exist only on PR #81 (`feat/snapshot-incremental`, squashed commit `dbf6496a`). Here, MCP writes happen only via session recording; explicit writes go through the HTTP server or library API.

### 3.3 Lifecycle hooks (secondary surface)
Host agent runs `<binary> --<agent>-hook <event>` as a subprocess per event (e.g. 10 events for Claude Code: SessionStart, UserPromptSubmit, …, SubagentStart/Stop); hook reads bounded JSON on stdin, prints output, fail-open. A skill file is installed (`.claude/skills/lint-ai-memory/SKILL.md`) with a user-edit marker. The `agy` adapter (`src/integrations/agy/`) is the newest, smallest — best copy-template.

### 3.4 HTTP server (`lint-ai-server` binary, axum)
- `src/bin/server.rs` (`cargo run --release --bin server`). Routes: `POST /add`, `/add/batch` (≤128), `/search`, `/delete`, `/supersede`, `/expire`; `GET /health`, `/api/status|timeseries|integrations|sessions|events|metrics`, `/metrics`, `/dashboard*`. Search reads a published immutable snapshot; mutations publish a fresh snapshot.
- **Bind policy:** `--bind` defaults `127.0.0.1:8080`; **non-loopback bind is rejected at startup** (localhost-only by design).
- Config: `--index <dir>` (else discovers `<project>/.lint-ai`), `--project-root`/`LINT_AI_PROJECT_ROOT`, `--server-token` (JWT Bearer), `--allow-unauthenticated` (dashboard only), `--tenant-id`/`SERVER_TENANT_ID` (single-tenant; `user_id` must match `MemoryService` doc filter; `group_id`/`session_id` is the core segmentation key).

### 3.5 Library API (Rust crate `lint_ai`)
`src/memory_api.rs` — `MemoryService::new(IndexStore)` → `add(AddRequest{request_id, messages, user_id, session_id})`, `add_batch`, `search(SearchRequest{query, options, user_id, top_k})`, `delete`, `supersede`, `expire`; plus `MemorySearchService`/`published_search()` snapshot for read-only paths. Production default pipeline via `default_production_pipeline_options()`. Also a `pyo3` Python-bindings cargo feature.

### 3.6 Subprocess / one-shot recall (no server)
- `--recall <query>`: one-shot chunk-level recall, JSON to stdout, no stdin, no server (`src/integrations/recall.rs` — built for remote workers that can't hold a stdio session).
- `--recall-server <root>`: keeps indexes open, newline-delimited JSON on stdio.
- `--inspect-index <path>` (+ `--inspect-view`): direct index inspection.

### 3.7 Storage & config (cross-cutting)
- **File-based, not SQLite** (no rusqlite dep). `IndexStore::at_path` persists under `<project>/.lint-ai/{workspace-memory,claude-memory,codex-memory,gemini-cli-memory,agy-memory}/` (legacy `*-mcp-index` dirs still read, not written). Each provider's MCP search composes shared `workspace-memory` with that provider's private memory at query time — memory is project- and provider-scoped. `.initialization.lock` guards init (30s timeout); in-memory fallback with stderr warning on persistence failure.
- **Config:** `--config <file>` (`src/config.rs`; `ignore_paths`, etc.), `--strict_config`, 2MB cap. Indexing limits: `--max-bytes` 5MB/file, `--max-files` 50k, `--max-depth` 20, `--max-total-bytes` 100MB.
- **Cargo features:** `claude-code`, `codex`, `gemini-cli`, `agy` (bundle `agent-integrations`); default is just `heuristic-query-semantics`. Serve/install flags are feature-gated out when their feature is off.

### 3.8 The `--muse-serve`-style pattern, made concrete
"An MCP adapter first, hooks later" (established for Muse Code per Luyi's memory). There is no literal `--muse-serve` in this checkout — the style means **copying the `--claude-code-serve`/`--codex-serve` template**:
1. **Nothing new on the wire:** the adapter is the existing `lint-ai` binary with a new `--openclaw-serve <root>` flag; the host spawns `lint-ai --openclaw-serve /project` as a stdio subprocess and speaks standard MCP JSON-RPC (`initialize`/`tools/list`/`tools/call`) with the same 7 tool names/schemas.
2. **What runs where:** one process per host connection, on the user's machine; lint-ai owns indexing (lazy `IndexStore` + `WorkspaceWatcher`) and retrieval; host owns the model. Project root pinned in the MCP entry — no cwd dependence.
3. **What code to write:** new `src/integrations/openclaw/` module (copy the `agy` adapter — smallest) reusing `mcp_tools` (tool defs + `search_results`), `mcp_index` (lazy store, memory-doc sync), `mcp_transport` (framing), `session_recording`; plus `--openclaw-install` merging the server entry into OpenClaw's MCP config and an optional skill file. Engine.rs dispatch + cli.rs flag + cargo feature follow the existing five-line pattern.
4. **Why lowest risk:** (a) standard MCP contract works with any MCP-capable host; (b) lifecycle hooks stay out of scope until entry schemas are probed against a live OpenClaw binary (Luyi's stated rule); (c) fail-open — bad memory day never breaks the host; (d) state stays project-/provider-scoped in `.lint-ai/openclaw-memory/`, composed with shared `workspace-memory`.

---

## 4. Integration options — comparison

### Option A — lint-ai as an OpenClaw-consumed MCP server (+ prompt directive)
- **How:** Add `--openclaw-serve <root>` / `--openclaw-install` to lint-ai (copy `agy` adapter). Install writes `{command: "<abs lint-ai>", args: ["--openclaw-serve", <root>]}` under `mcp.servers` in `~/.openclaw/openclaw.json` (or `openclaw mcp add lint-ai --command <abs> --arg --openclaw-serve --arg <root>` + `openclaw mcp doctor lint-ai --probe`). Ship an OpenClaw **skill** (`SKILL.md`) + a bootstrap prompt directive telling the agent to call `search` before answering and `record_session` after — exactly the OpenGraphMemory pattern.
- **Pros:** proven pattern for external memory on OpenClaw (OpenGraphMemory did this); zero OpenClaw code changes; standard MCP wire; fail-open; `mcp doctor --probe` gives a health check; Gateway hot-reloads config.
- **Cons:** no automatic capture — memory only moves when the agent decides to call the tools (needs the skill/directive); no session-boundary automation.
- **Effort:** low — new adapter module + flags + installer + skill file; roughly a `agy`-sized diff (days, not weeks). Risk is concentrated in OpenClaw config-schema quirks (unknown keys kill Gateway startup — verify key names against a live install).

### Option B — native OpenClaw plugin (`openclaw.plugin.json` + SDK)
- **How:** TypeScript plugin: `definePluginEntry` with `api.registerTool` wrapping lint-ai tools (spawning the stdio server or calling the HTTP server), typed hooks (`before_agent_start` for recall injection, `tool_result_persist` / `message_received` for capture), `api.registerService` for background index sync. Publish to **ClawHub** for one-line install.
- **Pros:** deepest integration — automatic recall/capture at real lifecycle points; distributable via the official registry; typed hooks give `session_end` (shutdown drain) and compaction hooks that internal hooks lack.
- **Cons:** must learn and track the OpenClaw plugin SDK (fast-moving project, SDK churn risk); typed-hook semantics verified only from docs mirrors, not the canonical docs; per Luyi's rule, hook entry schemas must be probed against a live OpenClaw binary before shipping.
- **Effort:** medium — a new TS package maintained against OpenClaw releases; ClawHub publish adds review/scan overhead.

### Option C — OpenClaw internal hooks only (no lint-ai code changes)
- **How:** Write a hook pack (`HOOK.md` + `handler.ts/js`) registered in `~/.openclaw/openclaw.json` under `hooks.internal.entries`: `agent:bootstrap` injects lint-ai recall output (via `--recall <query>` one-shot subprocess) into session context; `command:new`/`command:reset`/`session:auto-reset` call `record_session`/snapshot into lint-ai; `message:sent` does per-message capture. Hook `env` block carries config (e.g. `LINTAI_URL`).
- **Pros:** no lint-ai changes at all; uses already-built `--recall` subprocess mode; config-only on the OpenClaw side.
- **Cons:** **no true `session:end` event** (only gateway-shutdown drain via the typed plugin hook); per-message capture is noisy; hook payload schemas need probing against a live install; bootstrap file-injection size limits unknown.
- **Effort:** low-medium — hook scripts + config, but the probing/validation work is the same as Option B's.

### Option D — file-based memory sync (bidirectional markdown bridge)
- **How:** A small sync job (cron/hook/`api.registerService`) that writes lint-ai recall summaries into `~/.openclaw/workspace/memory/` daily files and `MEMORY.md`, and reads OpenClaw's daily logs back into lint-ai via `record_session`/`--recall` indexing.
- **Pros:** dead simple; works with OpenClaw's existing auto-loaded memory (zero agent prompting needed — it's loaded at session start); no protocol work.
- **Cons:** loses lint-ai's differentiators — supersession tracking, semantic verdicts, scores, and per-session segmentation all collapse into flat markdown; two writers can fight over `MEMORY.md` (it's curated long-term memory); drift between the two stores.
- **Effort:** low — but it's a downgrade path, not a real integration. Best used as a stopgap or companion, never the primary.

### Option E — lint-ai HTTP server + OpenClaw plugin/tool shim
- **How:** Run `lint-ai-server` (localhost-only, JWT Bearer via `--server-token`, tenant isolation); an OpenClaw plugin or internal hook calls `POST /search` / `/add` over HTTP.
- **Pros:** no stdio-session requirement (works where stdio subprocesses are awkward); explicit-write API (`/add`, `/supersede`, `/expire`) that the MCP tools lack in this checkout; tenant isolation + auth built in.
- **Cons:** a daemon to run and supervise (port, token, lifecycle); more moving parts than a spawned stdio process; the HTTP write surface is larger than the MCP read-mostly surface.
- **Effort:** medium — server ops + shim. Makes most sense as the transport *under* Option B, not as a standalone option.

---

## 5. Recommendation

**Lowest-risk first step: Option A.** Build `--openclaw-serve` / `--openclaw-install` in lint-ai (copy the `agy` adapter, reuse all shared MCP scaffolding, pin the project root, fail-open), then on the OpenClaw side add the MCP server entry and ship a ClawHub **skill** + bootstrap directive for when the agent should call `search` / `record_session` — mirroring the proven OpenGraphMemory pattern. This keeps all new code inside lint-ai, needs zero OpenClaw changes, and defers hooks until their entry schemas are probed against a live OpenClaw install (per Luyi's standing rule).

**Sequencing after that:**
1. Probe a live OpenClaw install: verify `mcp.servers` stdio entry shape, hook event payloads (`agent:bootstrap`, `command:new`), and whether `session:start`/`session:end` internal events have landed on current main.
2. If automatic capture is needed, promote to Option C (internal hooks) or Option B (native plugin) based on the probe results — Option B if ClawHub distribution matters.
3. Option D (file bridge) only as a stopgap; Option E (HTTP server) as the transport under Option B if stdio proves awkward.

## 6. Open questions for Luyi
- Should the adapter be named `--openclaw-serve` (consistent with `--claude-code-serve` etc.)?
- Does OpenClaw need the PR #81 `add_memory`/`get_memory` write tools, or is `record_session` + the 7 existing tools enough for v1?
- Is ClawHub distribution (plugin or skill) a goal, or is a local `--openclaw-install` + docs sufficient?
- Is there a live OpenClaw install available to probe hook/MCP config schemas against?

## 7. Sources
- Lint-AI repo: `~/lint-ai-demo` (`src/integrations/`, `src/cli.rs`, `src/engine.rs`, `src/memory_api.rs`, `src/bin/server.rs`, `docs/server.md`); `~/workspace/lint-ai-pr/mcp-memory-server.md`
- OpenClaw: https://github.com/openclaw/openclaw · https://github.com/openclaw/openclaw/blob/main/docs/tools/mcp.md · https://github.com/openclaw/clawhub/blob/HEAD/docs/clawhub.md · https://github.com/ardiannurcahya/ogm-mcp-skills/blob/HEAD/docs/openclaw.md · https://github.com/it-huset/openclaw-guide/blob/HEAD/content/docs/phases/phase-2-memory.md · https://github.com/kevinhamza/devin-4.0/blob/HEAD/repos/openclaw/docs/automation/hooks.md · https://github.com/aethonflame/memory-system/blob/HEAD/references/openclaw-hooks-research.md · https://github.com/linux2010/openclaw/blob/HEAD/docs/tools/plugin.md · https://github.com/weaxs/stock-analysis-plugin/blob/HEAD/AGENTS.md · https://docs.openclaw.ai/reference/templates/AGENTS · https://github.com/shunkakinoki/dotfiles/blob/HEAD/generated/hooks/traces/openclaw/HOOK.md
