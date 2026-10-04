# mem0 × Hermes Agent integration — how mem0 wired in

**Date:** 2026-09-28
**Sources probed live (no API keys used):**
- Hermes side: `https://github.com/NousResearch/hermes-agent`, local checkout `/tmp/hermes-probe/hermes-agent` (same checkout as `hermes-integration-research.md`)
- mem0 side: `https://github.com/mem0ai/mem0` at HEAD, cloned to `/tmp/mem0-repo` (scratch only)

## TL;DR

**mem0 ships the plugin itself.** It lives upstream at `mem0ai/mem0/integrations/hermes-plugin-mem0/` — a self-contained *directory plugin* (not a pip entry point, not a PR to Hermes). Hermes originally **vendored a copy** into `hermes-agent/plugins/memory/mem0/`, and the two are now migrating toward the standalone plugin (mem0's copy is newer — Hermes' bundle lags). mem0 never modified Hermes' core; it implemented Hermes' `MemoryProvider` ABC and is distributed via `hermes plugins install`. This is the exact template for lint-ai's Route B.

---

## 1. Hermes side: how mem0 is wired in

### 1.1 Distribution: bundled directory plugin → migrating to standalone

- **Bundled copy:** `plugins/memory/mem0/` in the hermes-agent repo — `__init__.py` (472 lines), `_backend.py` (215), `_oss_providers.py` (53), `_openai_llm.py`, `_setup.py` (535), `plugin.yaml`.
- **`plugin.yaml` (Hermes bundled version):** `name: mem0`, `version: 1.3.0`, `extra: mem0` — the `extra` key means "install Hermes' own pip extra named `mem0`" which the Hermes `pyproject.toml` defines as `mem0ai==2.0.10; platform_machine != 'ARM64' or sys_platform != 'win32'` (line 374–375; the ARM64/win32 gate is because `mem0ai` pulls `qdrant-client → grpcio`, which has no win_arm64 wheel — lines 371–374, 594–595).
- **Standalone copy (mem0's upstream):** `plugin.yaml` instead declares `pip_dependencies: [mem0ai>=2.0.10,<3]` plus its own `pyproject.toml` (`name = "hermes-plugin-mem0"`, `dependencies = ["mem0ai>=2.0.10,<3", "httpx>=0.27,<1"]`, optional extras `postgres`, `qdrant`). **No `[project.entry-points]` section exists** — mem0 did NOT use the `hermes_agent.memory_providers` entry-point group. It is a pure directory plugin.
- **Vendoring direction:** a `diff -rq` between the two trees shows mem0's upstream copy is newer than Hermes' bundle (mem0 upstream has lazy-install from pyproject honoring `security.allow_lazy_installs`, a `_rerank_default` recall option, and safer `sync_max_chars` parsing; Hermes' bundled copy lacks these). Per mem0's docs (`docs/integrations/hermes.mdx`, "Migration for Existing Users"): Hermes versions that still bundle mem0 prefer the bundled copy ("Bundled wins on name collision" — `plugins/memory/__init__.py`); full migration requires (1) a Hermes build with PR #114569, (2) a `mem0` catalog entry (`repo: https://github.com/mem0ai/mem0`, `subdir: integrations/hermes-plugin-mem0`, reviewed commit SHA), and (3) **removal of the bundled provider** so the standalone one loads. During `hermes update` or agent startup, Hermes auto-installs a missing configured provider (respecting `security.allow_lazy_installs`).

### 1.2 Discovery order (from `plugins/memory/__init__.py`)

Bundled `plugins/memory/<name>/` → user `$HERMES_HOME/plugins/<name>/` → project `./.hermes/plugins/<name>/` (opt-in via `HERMES_ENABLE_PROJECT_PLUGINS`) → `hermes_agent.memory_providers` entry points (line 30: `ENTRY_POINTS_GROUP = "hermes_agent.memory_providers"`). Bundled wins on name collision (line 77 `_is_bundled`). A directory plugin needs `plugin.yaml` + `__init__.py` exporting a `MemoryProvider` subclass (top-level or via `register(ctx)`), loaded by `_load_provider_from_dir` (lines 332–360).

### 1.3 The provider class — which ABC methods mem0 implements

`Mem0MemoryProvider(MemoryProvider)` (`plugins/memory/mem0/__init__.py`):

| Method | Line | What mem0 does |
|---|---|---|
| `name` | 140 | `"mem0"` |
| `is_available` | 143 | config/deps check only, no network (per ABC contract) |
| `initialize(session_id, **kwargs)` | 227 | reads `$HERMES_HOME/mem0.json` layered over `MEM0_*` env (`.env` secrets); resolves `user_id` (operator-configured > gateway-native > `"hermes-user"` placeholder); builds backend; registers atexit shutdown |
| `get_config_schema` | 154 | setup wizard fields for `hermes memory setup` |
| `save_config` | 149 | writes non-secret values to `mem0.json` |
| `system_prompt_block` | 253 | static block: active mode (platform/self-hosted/OSS), rerank note, "call mem0_search before answering…" instructions |
| `prefetch` | 291 | returns cached background recall (formatted `## Mem0 Memory` markdown); bounded 3-second wait |
| `on_turn_start` | 259 | starts the background prefetch for the *next* turn (`_start_prefetch`, line 270) |
| `sync_turn` | 302 | ships user+assistant messages (truncated to `sync_max_chars`, default 450 chars at sentence boundary) to backend in a background thread for fact extraction; best-effort, skips if previous sync still running after 5s |
| `get_tool_schemas` | 324 | `mem0_search` / `mem0_add` / `mem0_update` / `mem0_delete` (lines 324–364 in `TOOL_SCHEMAS`) |
| `handle_tool_call` | 355 | dispatches the four tools, returns JSON strings |
| `shutdown` | 384 | flushes workers, closes backend |
| circuit breaker | 188–216 | 5 consecutive failures → pause 2 min |

Notably NOT implemented: `on_memory_write` (the ABC hook that mirrors built-in memory-tool writes to the provider) — mem0 does not receive mirrors of built-in `memory add/replace/remove` operations.

### 1.4 Config — exact user-facing snippet

Selection: `hermes config set memory.provider mem0` → in the profile's `config.yaml`:

```yaml
memory:
  provider: "mem0"      # one at a time; "", default, builtin, built-in, none = built-in only
```

Provider settings in `$HERMES_HOME/mem0.json` (written by `hermes memory setup`):

```json
{ "mode": "platform", "host": "", "user_id": "my-hermes-user", "sync_max_chars": 450, "rerank": false, "oss": {...} }
```

Secrets in that profile's `.env`: `MEM0_API_KEY` (platform or self-hosted server), `OPENAI_API_KEY` (OSS mode), `MEM0_HOST` (server URL fallback).

### 1.5 Transport: three modes, no lint-ai-shaped sidecar

From `_backend.py` (`Mem0Backend` ABC, lines 19–37: `search` / `add` / `_update` / `_delete`):

- **Platform** (`PlatformBackend`): in-process `mem0.MemoryClient(api_key=...)` — mem0's cloud HTTP API client, imported lazily inside `__init__`.
- **Self-hosted server** (`SelfHostedBackend`): raw `httpx.Client` speaking the mem0 FastAPI server's real contract (`X-API-Key` header, `/memories`, `/search` routes). mem0's own `MemoryClient` is hardcoded to cloud auth, so the plugin bypasses it — a gotcha worth remembering for lint-ai's HTTP path (speak the server's contract directly).
- **OSS** (`OSSBackend`): in-process `mem0.Memory(MemoryConfig(...))` with user-chosen LLM/embedder/vector store (qdrant/pgvector). Note the `_openai_llm.py` workaround: Hermes registers a custom `hermes_openai` LLM provider into mem0's `LlmFactory` because mem0 validates `LlmConfig.provider` before its own factory lookup.

### 1.6 Gotchas (from code comments + docs)

- **Single active provider.** "ONE external provider at a time" — selecting mem0 *replaces* Honcho/supermemory/etc., never combines. Built-in file memory (`MEMORY.md`/`USER.md`) runs **alongside** it: "When Mem0 is active, it works additively with the built-in system at two points in every conversation turn" (current-turn recall + background fact extraction). The two stores never merge; mem0's provider does not implement `on_memory_write`, so built-in memory-tool writes stay in the built-in store.
- **Recall is best-effort:** prefetch waits max 3s; if the backend is slow the turn proceeds without injection (the agent can still call `mem0_search` itself).
- **Capture is best-effort:** no durable queue; forced termination can drop pending writes (Hermes' 30s exit watchdog).
- **Truncation:** `sync_turn` caps each message at 450 chars (configurable `sync_max_chars`), at a sentence boundary — long turns get partially extracted; `mem0_add` is the verbatim escape hatch.
- **User identity:** recall is scoped to `user_id` across gateways/channels (not to the current session); writes tag `metadata.channel` (`cli`, `telegram`, …).
- **Setup requires interactive terminal** on newer Hermes; on old CLI versions wizard options like `--mode` are rejected — manual `mem0.json` editing is the fallback. `hermes memory status` reports config availability, not backend health.

---

## 2. mem0 side: what mem0 actually shipped

- **Package:** `integrations/hermes-plugin-mem0/` inside the monorepo `mem0ai/mem0` — **not a PyPI package**. There is a `pyproject.toml` but it is consumed by *Hermes* (`hermes plugins install` installs the dir's pip deps into the Hermes venv), not by end users via `pip install`.
- **Entry point:** none. No `hermes_agent.memory_providers` entry point is registered; distribution is the plugin **directory** + `plugin.yaml`.
- **Install path (from mem0's docs, `docs/integrations/hermes.mdx`):**

```bash
hermes plugins install mem0ai/mem0/integrations/hermes-plugin-mem0
hermes plugins enable mem0
hermes memory setup
hermes memory status
```

`hermes plugins install` (`hermes_cli/plugins_cmd_install.py:280` `_install_plugin_core`) resolves the `owner/repo/subdir` identifier, clones the repo, extracts the subdir, copies it into the user plugins dir, installs `pip_dependencies`/`pyproject.toml` deps with user consent, and re-applies them after every `hermes update`.

- **Auth/config handling:** API keys never live in the plugin — they live in the Hermes profile's `.env` (`MEM0_API_KEY`, `OPENAI_API_KEY`) and non-secret settings in `$HERMES_HOME/mem0.json`. Env vars (`MEM0_MODE`, `MEM0_HOST`, `MEM0_USER_ID`, `MEM0_AGENT_ID`) are defaults; `mem0.json` overrides them. All profile-scoped (per-profile memory isolation falls out of `$HERMES_HOME`).
- **Transport:** in-process SDK (`mem0.MemoryClient` / `mem0.Memory`) for platform and OSS modes; raw `httpx` for the self-hosted server mode. No subprocess, no sidecar.
- **Maintenance relationship:** mem0 owns the upstream plugin (their repo is newer than Hermes' bundle); Nous Research owns the vendored bundle. The stated end-state is Hermes deleting its bundle and mem0's standalone becoming the only copy, auto-installed from the plugin catalog.

---

## 3. Direct comparison: if lint-ai followed mem0's exact path

| mem0's path | lint-ai equivalent |
|---|---|
| Directory `integrations/hermes-plugin-mem0/` in the mem0 monorepo | New directory in the lint-ai repo, e.g. `integrations/hermes-plugin-lintai/` (Rust repo shipping a Python plugin dir — fine, it's just files) |
| `plugin.yaml`: `name: mem0`, `pip_dependencies: [mem0ai>=2.0.10,<3]` | `plugin.yaml`: `name: lintai`, `pip_dependencies` — none needed if the provider talks to lint-ai over HTTP (only stdlib+`httpx`); or depend on nothing and shell to the `lint-ai` binary |
| `__init__.py`: `Mem0MemoryProvider(MemoryProvider)` | `__init__.py`: `LintaiMemoryProvider(MemoryProvider)` implementing `name` → `"lintai"`, `is_available`, `initialize`, `get_tool_schemas` → `lintai_search`/`lintai_add`(/`update`/`delete`), `handle_tool_call`, `prefetch`, `on_turn_start` (queue background recall), `sync_turn` (feed turn into lint-ai), `shutdown`, plus `get_config_schema`/`save_config` for `hermes memory setup` |
| `_backend.py`: `Mem0Backend` ABC → Platform/SelfHosted/OSS | `LintaiBackend`: HTTP client speaking lint-ai's server contract (`/search`, `/add/batch`, `/delete` — the same API the NeMo plugin uses), or subprocess calls to `lint-ai` CLI |
| User config: `hermes config set memory.provider mem0` + `$HERMES_HOME/mem0.json` | User config: `hermes config set memory.provider lintai` + `$HERMES_HOME/lintai.json` (root path, server URL, API key if any) |
| Docs page `docs/integrations/hermes.mdx` on mem0's site | Same on lint-ai's docs |
| No Hermes repo changes; no PyPI publish needed for the plugin itself | Same — the `lint-ai` binary is already the distributed artifact; the plugin dir is a thin Python wrapper |

**What changes in Hermes' config:** exactly two things — `memory.provider: "lintai"` in the profile's `config.yaml`, and a new `$HERMES_HOME/lintai.json` (+ secrets in `.env`). Nothing in the Hermes repo, no fork, no PR.

---

## 4. Implication for lint-ai's Route B decision

mem0's precedent **de-risks Route B substantially** and answers the earlier open questions:

1. **The pattern is third-party-owned, zero Hermes-side code.** mem0 never touched the Hermes repo for its *current* distribution — the plugin lives in mem0's own repo and installs via `hermes plugins install`. (The historical vendored bundle was Hermes' choice, now being unwound.) So lint-ai can ship Route B unilaterally: a `integrations/hermes-plugin-lintai/` directory + docs, no upstream negotiation needed.
2. **No pip entry point required.** The `hermes_agent.memory_providers` entry-point group exists but mem0 doesn't use it — directory plugins are the proven, simpler path. That drops the Python-packaging burden: we don't need a `lintai-hermes` PyPI package, just a plugin directory.
3. **Transport precedent favors HTTP, not subprocess.** mem0's platform and self-hosted modes are both HTTP clients against a server API — directly analogous to lint-ai's server API (`/add/batch`, `/search`, `/delete`). The provider would be a thin Python wrapper (~200 lines of backend + ~300 lines of provider), with the real memory engine staying in Rust. The mem0 self-hosted-backend gotcha is instructive: when the vendor's own client is hardcoded to cloud auth, speak the server's HTTP contract directly with `httpx`.
4. **The "one provider at a time" tradeoff is accepted by the market leader.** mem0 accepted that selecting it replaces the built-in/external provider slot, with built-in file memory continuing to run additively alongside. lint-ai would sit in exactly the same position — this normalizes Luyi's open question #2: staying an "augmenting sidecar" (Route A+MCP) vs. replacing the provider slot (Route B) is the same choice mem0 made, and mem0 chose Route B.
5. **Sequencing still holds:** mem0's docs describe the MCP-style tool path as fallback ("the model can still call `mem0_search` itself"), i.e. even the native provider ships explicit agent-callable tools — Route A (MCP adapter) and Route B (native provider) are complementary, not exclusive. Build the `--hermes-serve` MCP adapter first (already decided), then the provider plugin reuses the same server.

**Bottom line:** mem0's exact path — a `MemoryProvider` subclass in a self-owned plugin directory, HTTP transport to the memory server, config via `memory.provider` + profile JSON — is directly copyable for lint-ai. Estimated scope: one Python package directory (~500–700 lines mirroring mem0's structure: `__init__.py` provider + `_backend.py` HTTP client + `plugin.yaml` + `pyproject.toml` + README), zero Hermes-side changes, zero changes to the Rust core beyond the existing HTTP API.
