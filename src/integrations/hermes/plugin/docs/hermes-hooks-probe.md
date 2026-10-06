# Hermes hooks probe — lint-ai automatic capture/recall

**Date:** 2026-09-28
**Hermes:** NousResearch/hermes-agent @ `e408d363` (checkout at `~/workspace/hermes-probe/hermes-agent`)
**Method:** live in-process probe against the real Hermes codebase — no LLM keys, no paid calls.
**Probe artifacts (retained):** `~/workspace/hermes-probe/`
- `home/plugins/lintai_probe/` — real directory plugin (`plugin.yaml` + `__init__.py`)
- `home/config.yaml` + `home/` as isolated `HERMES_HOME` (real `~/.hermes` untouched)
- `probe_hooks.py` — driver; `hook_log.jsonl` — redacted payload log

## What the live probe proved

1. A directory plugin at `$HERMES_HOME/plugins/lintai_probe/` with `plugin.yaml` + `__init__.py::register(ctx)` is discovered and loaded through the real `discover_plugins()` path (59 found, 54 enabled in the isolated profile; probe plugin enabled via `plugins.enabled` in config.yaml).
2. **`pre_llm_call` recall/injection is real, end to end (no key needed).** The probe's callback returned `{"context": "[LINTAI-PROBE-RECALL] ..."}`; the real collector `agent/turn_context.py::_collect_pre_llm_call_context` returned exactly that string. The caller (`turn_context.py:1126+`) stamps it into the **user message content** via `_stamp_api_content_sidecar` before the API request — not the system prompt. Multimodal (list) content gets an appended text part instead of the sidecar.
3. **`llm_request` middleware can rewrite the raw outbound request.** Through the real `apply_llm_request_middleware` chain, the probe's `return {"request": new_req}` produced `changed=True` and the last user message carried the injected marker. Middleware context: `{request, task_id, turn_id, api_request_id, session_id, platform, model, provider, base_url, api_mode, api_call_count}`.
4. All lifecycle/capture hooks fire with the payloads documented below (fired through the real `hermes_cli.plugins.invoke_hook` dispatch; payloads redacted in `hook_log.jsonl`).

## Verified hook payload schemas

Hooks receive exactly the kwargs from their fire sites. The dispatch layer adds `telemetry_schema_version: "hermes.observer.v1"` to every hook payload.

| Hook | Fire site | Payload (kwargs) | Notes |
|---|---|---|---|
| `pre_llm_call` | `agent/turn_context.py:750+` | `session_id, task_id, turn_id, user_message, conversation_history, is_first_turn, model, platform, parent_session_id, sender_id` | Return `{"context": "..."}` or a bare string → injected into user message. Oversized output spilled to disk. |
| `post_llm_call` | `agent/turn_finalizer.py:435+` | `session_id, task_id, turn_id, user_message, assistant_response, conversation_history, model, platform` | Fires once per turn, after the tool loop. |
| `post_tool_call` | `agent/inline_tool_executors.py:29+` (`emit_terminal_post_tool_call`) | `function_name, function_args, result, session_id, task_id, turn_id, tool_call_id, duration_ms, status, error_type, error_message, middleware_trace` | One terminal emission per tool call, best-effort. |
| `transform_llm_output` | (output path) | `session_id, response, model` | Can transform the final text. |
| `on_session_start` | `agent/conversation_loop.py:861` | `session_id, model, platform` | Skipped for persistence-disabled forks (they share the parent's session id). |
| `on_session_end` ⚠️ | `agent/turn_finalizer.py:769+` | `session_id, task_id, turn_id, completed, failed, interrupted, turn_exit_reason, model, platform` | **Misleading name: fires at the end of every turn**, not at session close. `turn_exit_reason` e.g. `"text_response(stop)"`. |
| `agent_loop_stopped` | gateway `_interrupt_and_clear_session` / TUI `session.interrupt` | `session_key, platform, reason, invalidation_reason` | Fires after a real running turn is interrupted; not emitted in plain CLI. Carries no message body. |
| `on_session_finalize` | `hermes_cli/cli_session_mixin.py:427` (`_notify_session_boundary`) | `session_id, platform, reason="session_boundary"` | True session-boundary signal (e.g. before `/new`). **Carries no transcript.** |
| `on_session_reset` | `hermes_cli/cli_session_mixin.py:427` | `session_id, platform, reason="new_session"` | Fires for the OLD session id; the new session then fires `on_session_start`. **No transcript, no new-session id in the payload.** |

### Key behavioral findings

- **Dedupe keys.** `turn_id` = `{session_id}:{task_id}:{uuid4[:8]}` (`agent/turn_context.py:551-555`), fresh per turn. `post_llm_call` / `on_session_end`(hook) fire exactly once per turn in the normal path; an interrupted turn fires the shutdown variant instead. **`(session_id, turn_id)` is a safe idempotent write key** — replays/double-fires with the same turn_id are idempotent, new turns get new ids. This mirrors the OpenClaw `runId` dedupe and the lint-ai store's doc-ID idempotency, so the integration can stay stateless (no `state.json` equivalent needed).
- **No authoritative full transcript reaches plugin hooks at session close.** `_notify_session_boundary` passes only `{session_id, platform, reason}`. The CLI copies `conversation_history` for the memory-provider boundary flush (`commit_session_boundary_async`), but plugins never see it. Per-turn capture via `post_llm_call` already carries each turn's `conversation_history`/`assistant_response`, so a full-session record can be reconstructed incrementally — but the plugin never gets Hermes' own authoritative boundary snapshot.
- **Resume linking works.** `pre_llm_call` receives `parent_session_id` (set on resume/branch in `agent/session_persistence.py:356-360`); a plugin can chain `(session_id → parent_session_id)` for the session registry, like OpenClaw's `resumedFrom`.
- **Fail-open everywhere.** `_invoke_hook_safely` swallows plugin exceptions; `_emit_terminal_post_tool_call` swallows everything. A broken lint-ai plugin can never break the agent.
- **Timeouts (measured in `hermes_cli/plugins_dispatch.py`).** Default callback timeout 30s (`plugins.hook_callback_timeout`). Bounded (non-blocking) hooks: `pre_llm_call`, `post_llm_call`, `post_tool_call`, `transform_llm_output`, `transform_tool_result`, `transform_terminal_output`, `pre_api_request`, `post_api_request`, `api_request_error`, `pre_auxiliary_call`, `post_auxiliary_call`, `pre_verify`, `on_session_start`, `on_session_end`. After a timeout the same callback is suppressed 60s; max 3 abandoned workers. `pre_tool_call` is **fail-closed** (timeout blocks the tool). `on_session_finalize`/`on_session_reset` run on the caller thread (no timeout bound) — but they fire from the CLI, not the hot turn path, so a short synchronous flush is acceptable there; still, capture writes should be async/fire-and-forget with a bounded local queue.
- **Plugin discovery.** User plugins: `$HERMES_HOME/plugins/<name>/`; project plugins: `./.hermes/plugins/<name>/` only with `HERMES_ENABLE_PROJECT_PLUGINS=1`; `plugin.yaml`/`plugin.yml` (portable `plugin.json` also accepted); opt-in via `plugins.enabled` in config.yaml; `register(ctx)` entry point. Project plugins are the natural per-repo install story for lint-ai (like `--hermes-install` writes the MCP config today).

## Complete `VALID_HOOKS` inventory (41, from `hermes_cli/plugins.py`)

`pre_llm_call`, `post_llm_call`, `pre_tool_call`, `post_tool_call`, `transform_tool_result`, `transform_terminal_output`, `transform_llm_output`, `pre_api_request`, `post_api_request`, `api_request_error`, `transform_api_error_classification`, `pre_verify`, `pre_auxiliary_call`, `post_auxiliary_call`, `on_stream_start`, `on_stream_delta`, `on_stream_end`, `on_interim_message`, `on_session_start`, `on_session_end`, `on_session_finalize`, `on_session_reset`, `on_skill_lifecycle`, `subagent_start`, `subagent_stop`, `pre_gateway_dispatch`, `agent_loop_stopped`, `pre_approval_request`, `post_approval_response`, `on_room_member_activity`, `pre_transcription`, `kanban_task_blocked`, `kanban_task_claimed`, `kanban_task_completed`, `on_kanban_task_updated`, `on_kanban_dispatch_tick`, `on_kanban_worker_spawned`, `on_kanban_worker_exited`, `on_kanban_worker_stale_claim`, `gateway_platform_event`, `pre_command`.

The original live probe did not exercise the subagent hooks. The plugin now also
subscribes to `subagent_start`, `subagent_stop`, and `agent_loop_stopped` using Hermes's documented
callback contract: start captures parent/child identity and the delegated goal;
stop captures the child summary, status, duration, and metadata-only tool history.
The interruption hook stores only its session key and reason metadata. Other hooks
(streaming, kanban, gateway, approvals) remain out of scope for memory.

## Hooks vs native `MemoryProvider` — verdict

The native provider lifecycle (`agent/memory_provider.py`, manager in `agent/memory_manager.py`):

| Provider method | What it gets | Hooks equivalent |
|---|---|---|
| `prefetch(query, session_id)` → str | Per-turn recall; result flows into the turn | `pre_llm_call` → `{"context": ...}` (same user-message injection point; the provider path also ultimately lands in the prompt) |
| `on_turn_start(turn_number, message, ...)` | Per-turn tick with author trio | `pre_llm_call` (richer payload) |
| `sync_turn(user_content, assistant_content, session_id, messages, turn_author)` | Per-turn capture with full message list | `post_llm_call` (has `conversation_history` + `assistant_response`) |
| `on_session_end(messages)` | **Full transcript at real session boundary**, then `on_session_switch(new_session_id, parent_session_id, ...)` — serialized on one background worker (`commit_session_boundary_async`) | `on_session_finalize` / `on_session_reset` — **no transcript, no new-session id** |
| `shutdown()` | Bounded cleanup | nothing reliable (shutdown hook path is best-effort `suppress(Exception)`) |

**Verdict: hooks win for the lint-ai design, with one acknowledged gap.**

- Hooks **preserve the user's existing memory setup**: Hermes allows exactly one external memory provider at a time (`memory.provider`); installing lint-ai as a provider would evict mem0/file/other providers. A directory plugin coexists with all of them — this is the decisive architectural point, matching how `--hermes-serve` coexists today.
- The provider's only real advantage is the authoritative full-transcript boundary snapshot. But per-turn `post_llm_call` capture already accumulates every turn's content with a stable dedupe key, so the plugin reconstructs the session incrementally; the marginal value of the boundary snapshot is small, and lint-ai's retrieval is turn/segment-level anyway.
- Hooks are fail-open, timeout-bounded, and need zero Hermes changes; a provider needs a `MemoryProvider` subclass (~500-700 lines, per the mem0 precedent) and occupies the single provider slot.
- Note mem0's own precedent: mem0's Hermes plugin uses the provider path, but mem0 *is* the memory system. lint-ai's pitch is a memory *layer under* the agent's existing memory — hooks fit that positioning better.

## Recommended minimal design (hooks-based, stateless)

1. **`pre_llm_call` → recall + inject.** Query lint-ai with `user_message` (+ `session_id` scoping); return `{"context": ...}`. Keep it fast — the 30s timeout is generous but the turn blocks on it. Cache per `(session_id, turn_id)`.
2. **`post_llm_call` → per-turn capture.** Write doc keyed by `(session_id, turn_id)` → idempotent under re-fire; no local state file needed (store-level doc-ID idempotency, per the OpenClaw purist fix).
3. **`on_session_finalize` / `on_session_reset` → boundary marker.** Mark the session closed in lint-ai; full content already captured incrementally. On reset, link old→new session ids temporally (new `on_session_start` follows); on resume, `pre_llm_call.parent_session_id` gives the true parent link.
4. **`on_session_start` → session registry entry** (session_id, model, platform).
5. **Capture path is async + bounded.** Hook callbacks enqueue to a local bounded queue flushed by a background thread; best-effort on shutdown (Hermes gives no reliable drain — same constraint as OpenClaw's 2s shared budget).
6. **Install:** ship as a directory plugin `src/integrations/hermes/plugin/` in the lint-ai repo (`plugin.yaml` + provider class over the lint-ai provider-memory HTTP API — no Hermes changes), installed via `hermes plugins install` or project plugins with `HERMES_ENABLE_PROJECT_PLUGINS=1`. This is the same shape as the mem0 precedent, but using hooks instead of the provider slot.

## Open questions for Luyi

1. **Hooks (coexist with existing memory) vs native provider (occupies the single external-provider slot, gets the authoritative boundary transcript)?** I recommend hooks; the transcript gap is covered by incremental per-turn capture. Confirm.
2. **Should capture default on?** Per-turn `post_llm_call` capture of every turn is the automatic-memory promise, but it's also the most data written. Default on, or opt-in per project?
3. **Retention/privacy expectations:** capture stores user messages and assistant responses verbatim in the lint-ai store. Any redaction rules (secrets, paths) before this ships?
4. **Is per-turn capture alone acceptable as the session record**, given plugin boundary hooks never receive Hermes' authoritative full transcript (only the provider path does)?
