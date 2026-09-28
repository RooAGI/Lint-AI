# DESIGN: `hermes-plugin-lintai` — automatic capture/recall for Hermes Agent via hooks

**Status:** design (approved pattern; implementation follows this document)
**Date:** 2026-09-28
**Probe basis:** `~/workspace/research_notes/hermes-hooks-probe.md` — live probe against
NousResearch/hermes-agent @ `e408d363`. Every hook name, payload schema, and behavior
below was verified against the real Hermes code and a real plugin load; nothing here
is inferred from docs.

## 1. Goal

Give Hermes Agent automatic lint-ai memory — recall injected before every model call
and capture of every turn and tool call — with **zero changes to Hermes and zero
changes to the lint-ai Rust core**. This is the "next level" after the `--hermes-serve`
MCP adapter (PR #82): the MCP adapter makes memory available when the agent *chooses*
to call a tool; the hooks plugin makes capture/recall *automatic*.

Deliberately out of scope: the native `MemoryProvider` path (it would occupy Hermes'
single external-provider slot and evict the user's existing mem0/file memory —
see §10). Hooks coexist with everything.

## 2. Why HTTP, not "direct" like Claude Code / Codex

Our Claude Code / Codex / OpenClaw / `--hermes-serve` adapters are **Rust code compiled
into the lint-ai binary** — the MCP server and the memory core share one process, so
they call the core in-process.

Hermes hooks are different: hook handlers **must be Python running inside the Hermes
process** (a directory plugin: `plugin.yaml` + `__init__.py::register(ctx)` is the only
plugin format Hermes loads). Python cannot reach into our Rust binary in-process. The
bridge options are:

1. **HTTP to a running lint-ai server** (chosen) — persistent keep-alive connection,
   tiny per-turn cost, and the API already exists (`POST /add/batch`, `POST /search`,
   `POST /delete` on the `lint-ai serve` binary, default `127.0.0.1:8080`). No Rust
   changes needed. This is exactly the pattern mem0's own Hermes plugin uses
   (HTTP/SDK, no subprocess).
2. Spawning the lint-ai binary per hook event — a fresh process on every turn just to
   record a message. Rejected: wasteful and fragile.

Tradeoff stated plainly: the user must run the lint-ai server alongside Hermes (one
extra process); the `--hermes-serve` MCP adapter needs only the binary. The plugin
fails open if the server is unreachable (see §7), so a missing server degrades to
"no automatic memory", never to a broken agent.

## 3. Hook inventory (verified payload schemas)

Hooks receive exactly the kwargs from their fire sites; the dispatch layer adds
`telemetry_schema_version: "hermes.observer.v1"` to every payload.

| Hook | Fires | Verified payload (kwargs) | Role in this design |
|---|---|---|---|
| `pre_llm_call` | Before every model call | `session_id, task_id, turn_id, user_message, conversation_history, is_first_turn, model, platform, parent_session_id, sender_id` | **Recall + inject.** Return `{"context": "..."}` (or bare string) → Hermes stamps it into the user message via `_stamp_api_content_sidecar` (`agent/turn_context.py:1126+`). Verified live: the collector returned our marker verbatim and it reached the outbound request. |
| `post_tool_call` | After every tool call (terminal emission, best-effort) | `function_name, function_args, result, session_id, task_id, turn_id, tool_call_id, duration_ms, status, error_type, error_message, middleware_trace` | **Structured tool-event capture** (see §4). |
| `post_llm_call` | Once per turn, after the tool loop | `session_id, task_id, turn_id, user_message, assistant_response, conversation_history, model, platform` | **Per-turn transcript capture.** |
| `on_session_start` | Session start (skipped for persistence-disabled forks sharing the parent's session id) | `session_id, model, platform` | **Session registry entry.** |
| `on_session_finalize` | True session boundary (e.g. before `/new`) | `session_id, platform, reason="session_boundary"` | **Boundary marker.** Carries **no transcript** (see gap below). |
| `on_session_reset` | Session reset; fires for the OLD session id, the new session then fires `on_session_start` | `session_id, platform, reason="new_session"` | **Boundary marker.** No transcript, no new-session id in payload. |

### The naming trap

`on_session_end` **fires at the end of every turn, not at session close**
(`agent/turn_finalizer.py:769+`; payload includes `task_id`, `turn_id`,
`turn_exit_reason`). It also fires at shutdown/interrupt with
`completed=False, interrupted=True`. The plugin **must not** treat it as a session
boundary. (OpenClaw's analogue: `session_end.messageCount` is always 0 — both
frameworks have one unreliable-looking session hook; ours is the misleading name.)

### The gap

`on_session_finalize` / `on_session_reset` carry **no transcript** — only
`{session_id, platform, reason}`. (OpenClaw has `before_reset` with the full departing
transcript; Hermes has no equivalent — the CLI keeps the transcript for the
memory-*provider* boundary flush, which plugins never see.) Therefore **per-turn
accumulation via `post_llm_call` is the authoritative session record**; boundary hooks
only mark open/closed. This is acceptable: lint-ai retrieval is turn/segment-level,
and every turn's `conversation_history` is captured.

### Timeouts and failure mode (measured in `hermes_cli/plugins_dispatch.py`)

- Default callback timeout **30s** (`plugins.hook_callback_timeout`); all our hooks are
  bounded (non-blocking). After a timeout the same callback is suppressed 60s; max 3
  abandoned workers.
- **Fail-open everywhere**: `_invoke_hook_safely` swallows plugin exceptions;
  `_emit_terminal_post_tool_call` swallows everything. A broken plugin can never break
  the agent — but we still keep handlers exception-proof and fast.
- `pre_tool_call` is **fail-closed** (a timeout blocks the tool). We do not subscribe
  to it and never do slow work there.
- `on_session_finalize` / `on_session_reset` run on the caller thread (no timeout
  bound) — but they fire from the CLI, not the hot turn path; a short synchronous
  flush is acceptable there. All turn-path writes still go through the async queue.

## 4. The mapping

| Hermes hook | lint-ai action | Details |
|---|---|---|
| `pre_llm_call` | **Recall + inject** | Query `POST /search` with `user_message` (scoped by `user_id` from config). Format top hits as a compact context block; return `{"context": block}`. Keep it fast — the turn blocks on this hook (bounded by the 30s timeout; target <1s via the local server). Cache per `(session_id, turn_id)` so double-fires don't double-query. `parent_session_id` (present on resume/branch) is recorded for resume linking. |
| `post_tool_call` | **Structured tool-event records** | One record per tool call: `function_name`, args (truncated), `result` (truncated), `duration_ms`, `status` (+ `error_type`/`error_message` on failure). Dedupe by `tool_call_id`. Enqueued async — never blocks the agent. |
| `post_llm_call` | **Per-turn transcript records** | One record per turn: `user_message`, `assistant_response`, plus a bounded slice of `conversation_history`. Dedupe by `(session_id, turn_id)`. Enqueued async. |
| `on_session_start` | **Session registry entry** | `session_id`, `model`, `platform`, timestamp. |
| `on_session_finalize` / `on_session_reset` | **Boundary markers** | Mark the session closed (`reason` recorded). Content is already captured incrementally; on reset, link old→new temporally when the new `on_session_start` arrives. |

Not subscribed (deliberately): `pre_tool_call` (fail-closed), `llm_request` middleware
(verified working, but raw request rewriting is stronger than we need — reserved for
future use), streaming/kanban/gateway/approval hooks (out of scope for memory).

## 5. Dedupe — stateless, no state file

Following the OpenClaw purist fix (no `state.json`; store-level idempotency instead):

- **Turns:** `(session_id, turn_id)` — `turn_id` is `{session_id}:{task_id}:{uuid4[:8]}`,
  fresh per turn; re-fires/retries reuse the same `turn_id`, so they are idempotent.
  Used as the `request_id` on `POST /add/batch` (the server treats `request_id` as the
  idempotency key).
- **Tool events:** `tool_call_id` — unique per tool call; same request_id scheme.
- **Session registry / boundary markers:** `session_id` (+ reason) as the key;
  re-fires overwrite the same record.

No local state file, no seen-sets. A replayed hook with the same IDs is a no-op at the
store.

## 6. Lifecycle comparison vs the OpenClaw hooks integration

Same shape — inject → capture per turn → boundaries → flush — with the two
Hermes-specific deviations called out in §3.

| Stage | OpenClaw | Hermes (this design) |
|---|---|---|
| Recall + inject before model call | `agent:bootstrap` → append `context.bootstrapFiles` (verified verbatim) | `pre_llm_call` → return `{"context": ...}`, stamped into the user message (verified verbatim) |
| Per-turn capture | `agent_end` → incremental capture, dedupe by `runId` (retries re-fire same runId) | `post_llm_call` → per-turn transcript, dedupe by `(session_id, turn_id)` |
| Structured tool capture | (no dedicated per-tool-call typed event; tool traffic inside turn transcript) | `post_tool_call` → tool-event records, dedupe by `tool_call_id` |
| Session-close authoritative record | `before_reset` → **full departing transcript before wipe** | **No equivalent** — `on_session_finalize`/`on_session_reset` carry no transcript; per-turn accumulation is authoritative |
| Session boundaries | typed `session_start` / `session_end` (`resumedFrom`/`nextSessionId` link parent↔child); `messageCount` always 0 — unreliable | `on_session_start` (registry); `on_session_finalize`/`reset` (markers); `parent_session_id` on `pre_llm_call` for resume linking; `on_session_end` hook is a naming trap (fires per turn) |
| Shutdown drain | one shared 2s budget — bounded best-effort flush only | no reliable drain for plugins — bounded best-effort flush on interpreter exit; queue is best-effort by design |

## 7. Transport and runtime

- **HTTP client:** stdlib only (`urllib` / `http.client`) — no third-party deps, so the
  plugin installs anywhere Hermes runs. One persistent `HTTPConnection` per worker
  thread (keep-alive); short connect/read timeouts (2s/5s) so a dead server never
  stalls a hook past Hermes' own 30s bound.
- **Async bounded write queue:** hook callbacks never do network I/O inline. They
  append a work item to a bounded `queue.Queue` (default max 1000; drops oldest with a
  counter when full — memory is best-effort, the agent is not). A single background
  daemon thread drains the queue, batching into `POST /add/batch` (server limit: 128
  requests per batch).
- **`pre_llm_call` is synchronous by necessity** (its return value is the injection),
  so recall does a direct `POST /search` with a tight timeout instead of going through
  the queue. On timeout/error it returns `None` (no injection) — fail-open.
- **Fail-open everywhere:** every handler is wrapped so exceptions are swallowed and
  counted, never raised into Hermes.
- **Config** (no new files in the repo; resolved at import):
  `LINTAI_SERVER_URL` (default `http://127.0.0.1:8080`), `LINTAI_USER_ID` (default
  `"hermes"`), `LINTAI_QUEUE_MAX` (default `1000`), `LINTAI_CAPTURE` (`on`/`off`,
  default `on`), `LINTAI_RECALL` (`on`/`off`, default `on`), `LINTAI_RECALL_TOP_K`
  (default `5`). Optionally `$HERMES_HOME/lintai.json` for the same keys (env wins).
- **Never slow in `pre_tool_call`:** not subscribed (fail-closed hook).

## 8. Document types written to the lint-ai store

All writes go through `POST /add/batch` as `AddRequest { request_id, user_id,
session_id, messages: [{role, timestamp, content}] }`. `user_id` comes from config;
`session_id` is `hermes:<session_id>` (namespaced so Hermes sessions never collide
with other providers' sessions).

1. **Turn record** — `request_id = "hermes:turn:{session_id}:{turn_id}"`
   - `messages`: `[{role: "user", content: user_message}, {role: "assistant", content: assistant_response}]`
   - A bounded tail of `conversation_history` (last N messages, N=20, each truncated
     to 2000 chars) is appended as a second user-role message labeled
     `[context]` so the turn stays self-contained without unbounded growth.
   - Fields carried in content headers: `model`, `platform`, `task_id`, `is_first_turn`
     (as a small `[meta]` preamble — the store is content-addressed text; structured
     fields live inline).
2. **Tool-event record** — `request_id = "hermes:tool:{tool_call_id}"`
   - `messages`: `[{role: "system", content: "tool_call function_name=<name> status=<status> duration_ms=<n>"}]`
     followed by truncated `function_args` and `result` (each ≤2000 chars) and, on
     failure, `error_type`/`error_message`.
   - `session_id` on the request ties it to the session; the content header carries
     `turn_id` for turn-level correlation.
3. **Session record** — `request_id = "hermes:session:{session_id}"`
   - Written on `on_session_start`: `[{role: "system", content: "session_start model=<m> platform=<p> parent_session_id=<pid or none>"}}`.
   - Rewritten (same `request_id`, idempotent) on `on_session_finalize`/`reset` with
     `session_close reason=<reason>` appended. The latest write wins; history of the
     session's turns is in the turn records.

Truncation policy: no single captured field exceeds 2000 chars; a turn's total
payload is capped at ~24KB. Rationale: capture is for recall, not forensics; the
Hermes session log remains the system of record.

## 9. Open questions (carried over from the probe; need Luyi's call)

1. **Hooks (coexist) vs native provider (single slot, authoritative boundary
   transcript)?** This design recommends hooks. Confirm before we invest further.
2. **Capture default-on?** Per-turn capture of every turn is the automatic-memory
   promise, but it's the most data written. `LINTAI_CAPTURE` defaults `on`; flip to
   opt-in per project?
3. **Retention/redaction:** capture stores user messages and assistant responses
   verbatim. Any redaction rules (secrets, absolute paths) before this ships?
4. **Per-turn-only session record:** acceptable, given plugin boundary hooks never
   receive Hermes' authoritative full transcript (only the provider path does)?

## 10. Why not the native `MemoryProvider`

Considered and rejected for now (probe §"Hooks vs native MemoryProvider"): Hermes
allows exactly one external memory provider (`memory.provider`); installing lint-ai as
a provider would evict the user's mem0/file/other provider. The provider's only real
edge is the authoritative full-transcript boundary snapshot (`on_session_end(messages)`
+ `on_session_switch`), but per-turn capture already accumulates every turn and
lint-ai retrieval is turn/segment-level. A provider also needs a `MemoryProvider`
subclass (~500–700 lines, per the mem0 precedent) versus a directory plugin with zero
Hermes changes. If Luyi later wants the provider slot, mem0's plugin is the template
(`~/workspace/research_notes/hermes-mem0-integration.md`).

## 11. Verification plan

1. **Python unit tests** (`tests/test_plugin.py`, stdlib `unittest`, no deps):
   hook→record builders (turn/tool/session), dedupe key derivation, truncation caps,
   queue drop-oldest behavior, fail-open handler wrapping, config resolution
   (env > lintai.json > defaults), HTTP client batching (mock transport).
2. **Live end-to-end** against the real Hermes CLI with an isolated `HERMES_HOME`
   (never the real `~/.hermes`): install the plugin from this directory, drive
   `pre_llm_call` (assert the returned context reaches the turn), fire
   `post_tool_call` + `post_llm_call` through the real `invoke_hook` dispatch against
   a stub lint-ai HTTP server, and assert the exact JSON the plugin POSTed to
   `/search` and `/add/batch`.
3. **Rust gate untouched:** this plugin is Python-only; `cargo test --all-targets`
   is unaffected (no Rust changes in this design).
