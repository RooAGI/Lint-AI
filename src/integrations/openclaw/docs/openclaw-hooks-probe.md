# OpenClaw Hook System — Live Probe Report

**Date:** 2026-09-28
**Probe target:** OpenClaw **2026.9.6** (npm), installed at `~/workspace/openclaw-probe/node_modules/openclaw`
**Environment:** Node v24.20.0, npm 10.9.4, loopback-only Gateway on port 18799, auth none
**Rule honored:** every schema below was observed on a live host. Nothing is inferred from docs alone.
**Scope:** research + design only. No changes to `~/lint-ai-demo`. No API keys entered (a deliberately
fake `OPENAI_API_KEY=probe-dummy-key-not-a-real-credential` was used only to pass auth *resolution*;
all model traffic went to a local stub server on 127.0.0.1:18798). No channels connected.

**Probe artifacts (kept for re-verification):**
- Internal hook log: `hook-events.jsonl`
- Typed plugin hook log: `typed-hook-events.jsonl`
- Stub model server: `~/workspace/openclaw-probe/stub-server.mjs` (+ `stub-last-body.json`, last captured request)
- Probe hooks: `~/workspace/openclaw-probe/state/hooks/probe-all/`, `.../probe-inject/`
- Probe plugin: `~/workspace/openclaw-probe/plugins/probe-plugin/`
- Gateway consoles: `~/workspace/openclaw-probe/gateway-console*.log`

---

## 1. How the probe worked

1. Installed OpenClaw 2026.9.6 via npm (`--no-bin-links`; first attempt failed on an EPERM/chown in `node-edge-tts`).
2. Registered an **internal hook** (`probe-all`, colon-style events) logging full sanitized event objects,
   plus a second internal hook (`probe-inject`) that mutates `context.bootstrapFiles` on `agent:bootstrap`.
3. Registered a **typed plugin** (`probe-plugin`, `api.on(...)`) for
   `session_start`, `session_end`, `agent_end`, `before_reset`, `before_compaction`,
   `after_compaction`, `gateway_start`, `gateway_stop`, logging sanitized `(event, ctx)` args.
   `plugins.entries.probe-plugin.hooks.allowConversationAccess: true` was set (trusted local probe only).
4. Drove the Gateway over local RPC: `sessions.create` (with/without initial message, with
   `parentSessionKey` + `emitCommandHooks`), `sessions.send`, `sessions.reset`.
5. Ran a local OpenAI-compatible **stub** (`OPENAI_BASE_URL=http://127.0.0.1:18798/v1`) that captures
   request bodies, so the exact model input could be inspected.

---

## 2. Hooks that FIRED (live-confirmed)

### 2.1 `agent:bootstrap` → `agent` / `bootstrap` — RECALL / INJECTION POINT ✅

Fires per agent turn **before context injection**, on the turn's `sessionKey`.
Observed payload (trimmed):

```json
{
  "type": "agent",
  "action": "bootstrap",
  "sessionKey": "agent:main:dashboard:7582e0a7-84ad-48cc-97fd-11a038a29007",
  "timestamp": "2026-09-28T15:14:42.332Z",
  "context": {
    "workspaceDir": "/home/hatch/workspace/openclaw-probe/state/workspace",
    "bootstrapFiles": [
      {"name": "AGENTS.md", "path": ".../workspace/AGENTS.md", "content": "# AGENTS.md ...", "missing": false},
      {"name": "SOUL.md", "path": "...", "content": "...", "missing": false},
      {"name": "IDENTITY.md", "path": "...", "content": "...", "missing": false},
      {"name": "USER.md", "path": "...", "content": "...", "missing": false},
      {"name": "BOOTSTRAP.md", "path": "...", "content": "...", "missing": false}
    ],
    "cfg": { "...": "full resolved config" },
    "sessionId": "f917502e-181f-49dd-a262-da8adbaf791c",
    "agentId": "main"
  },
  "messages": []
}
```

**Injection VERIFIED end-to-end.** The `probe-inject` hook appended
`{name:"LINTAI.md", path:".../workspace/LINTAI.md", content:"# LINT-AI recalled context (probe)\n\nLINTAI-PROBE-MARKER-7f3a9c\n...`
to `context.bootstrapFiles`. The marker was found in the actual model request captured by the stub,
at `input[0].content[0].text`, rendered as:

```
## /home/hatch/workspace/openclaw-probe/state/workspace/LINTAI.md
# LINT-AI recalled context (probe)

LINTAI-PROBE-MARKER-7f3a9c
...
```

i.e. inline in the first (system/developer) input block, right after BOOTSTRAP.md content.
Notes:
- The array is **rebuilt per fire** (observed `before:5 → after:6` on every fire); repeated injection does not accumulate.
- Fires on **every** turn, including `sessions.send` follow-ups — not just session creation. Recall is per-turn.
- Normally fires **once per turn**. (An early 4x multi-fire was traced to my stub returning malformed SSE,
  which triggered OpenClaw's `[empty-error-retry]` 3x; with a well-formed stream it fired exactly once.
  Design defensively: make injection idempotent — it already is.)

### 2.2 `agent_end` (typed plugin hook) — PER-TURN CAPTURE POINT ✅

Fires when a run completes. Observed args:

- `event` = `{"messages": [...], "success": true, "durationMs": 143, "runId": "2f2c9d18..."}`
- `ctx` = `{"runId", "trace", "agentId": "main", "sessionKey", "sessionId",
  "workspaceDir", "modelProviderId": "openai", "modelId": "gpt-6-astra",
  "messageProvider": "webchat", "channel": "webchat", "trigger": "user",
  "channelId", "senderId": "cli", "chatId", "channelContext"}`

`messages` contains the turn's full message list: `user` (the prompt), `custom`
(`openclaw.runtime-context` carrier), `assistant` (reply content + `usage`).
On error turns, the assistant entry carries `stopReason: "error"` and `errorMessage`.
**Dedupe by `runId`** — model-error retries re-fire it (observed 4x with the broken stub; 1x normally).

### 2.3 `before_reset` (typed) — PRE-RESET CAPTURE WITH FULL TRANSCRIPT ✅

Fires **before** the transcript is wiped, on both `/reset`-style flows. This is the richest capture hook.
Observed `event`:

```json
{
  "sessionFile": "sqlite:main:613275ec-c2ca-4b37-a760-5b27f9742a38:/home/hatch/workspace/openclaw-probe/state/agents/main/sessions/sessions.json",
  "messages": [
    {"role": "user", "content": "child turn", "timestamp": 1790608..., "idempotencyKey": "5e7215d1-...:user", "__openclaw": {...}},
    {"role": "assistant", "content": [], "api": "openai-responses", "provider": "openai",
     "model": "gpt-6-astra", "usage": {...}, "stopReason": "error",
     "errorMessage": "Responses stream delivered a malformed event without a string type",
     "timestamp": 1790608...}
  ],
  "reason": "new"
}
```

- `reason` observed: `"new"` (child spawned with `emitCommandHooks`), `"reset"` (`sessions.reset`).
- `messages` is the **complete departing transcript** — no extra fetch needed.
- `sessionFile` pinpoints the session's backing store (`sqlite:<agentId>:<sessionId>:<sessions.json path>`).

### 2.4 `session_end` / `session_start` (typed) — LIFECYCLE BOUNDARIES ✅

`session_end` event: `{"sessionId", "sessionKey", "messageCount", "reason",
"nextSessionId"?, "nextSessionKey"?}`; ctx: `{"agentId", "sessionId", "sessionKey"}`.
Observed reasons:
- `"new"` — parent ended because a child was spawned with `emitCommandHooks: true`
  (carries `nextSessionId`/`nextSessionKey` of the child)
- `"reset"` — `sessions.reset`
- `"shutdown"` — Gateway shutdown (all active sessions; shares the documented 2s total drain budget)

`session_start` event: `{"sessionId", "sessionKey", "resumedFrom"?}`; ctx: `{"agentId", "sessionId", "sessionKey"}`.
- Fired for the child on `sessions.create` with `parentSessionKey` + `emitCommandHooks: true`,
  with `resumedFrom` = parent sessionId.
- **Did NOT fire for a plain `sessions.create`** (no parent). Session creation alone is only visible
  via internal hooks (`message:received`, `agent:bootstrap`).

⚠️ `messageCount` was `0` in every observed `session_end`, including sessions with messages — **do not rely on it**.

### 2.5 `command:new` / `command:reset` (internal) ✅

- `command:new` fires on the **parent** session key when `sessions.create` is called with
  `parentSessionKey` + `emitCommandHooks: true`. Context:
  `{agentId, sessionEntry, previousSessionEntry, commandSource, cfg, storePath, workspaceDir}`.
  (Code path also emits typed `before_reset` with `reason: "new"` for the parent.)
- `command:reset` fires on `sessions.reset`. Context:
  `{agentId, sessionEntry, previousSessionEntry, commandSource, cfg, storePath, workspaceDir}`.
  **`messages` is `[]` — the transcript is NOT in the payload.** The typed `before_reset` hook is the
  one that carries messages; an internal-only hook would have to read `transcript_events` in sqlite
  (old events survive the reset there — verified — but the assembled `chat.history` hides them).

### 2.6 `gateway:startup` / `gateway:shutdown` (internal) and `gateway_start` / `gateway_stop` (typed) ✅

- Internal `gateway:startup`: `context` = `{cfg, deps, workspaceDir}`, `sessionKey: "gateway:startup"`.
- Internal `gateway:shutdown`: `context` = `{"reason": "gateway stopping", "restartExpectedMs": null}`,
  `sessionKey: "gateway:shutdown"`.
- Typed `gateway_start`: args `[{port}, {port, config, workspaceDir, getCron, abortSignal}]`.
- Typed `gateway_stop`: args `[{reason: "gateway stopping"}, {port}]`.
- SIGTERM to the supervisor produced all four plus `session_end(reason=shutdown)` for the active session.
  (SIGUSR2 restart did **not** produce observable hook traffic in this setup — see §3.)

### 2.7 `message:received` / `message:preprocessed` (internal) ✅

Fire per inbound message on the session key.
- `message:received` context: `{from, content, channelId: "webchat", conversationId, messageId,
  metadata: {provider: "webchat", surface: "webchat", senderId: "cli"}}`.
- `message:preprocessed` context: `{from, body, bodyForAgent, channelId, conversationId, messageId,
  senderId: "cli", provider, surface, isGroup: false, cfg}`.
- Observation-only: no transcript access, no mutation contract. Not sufficient as capture hooks.

---

## 3. Hooks NOT observed live (documented but untriggered)

| Hook | Why not triggered | Assessment |
|---|---|---|
| `command:stop` (internal) | Fires only inside `/stop` abort of an **active** run (code-read: `commands-handlers.runtime-*.mjs` → `handleStopCommand` → `beforeKill`). `/stop` via `sessions.send` with no active run was silently consumed; no event. | Minor for lint-ai: stop doesn't wipe the transcript; the next `agent_end`/`before_reset` captures it. |
| `session:patch` (internal) | Tied to the archive lifecycle op (`sessions.patch`); not exercised. | Not design-critical. |
| `session:auto-reset` (internal) | Needs an idle-timeout auto-reset; impractical in a probe window. | Same capture path as `command:reset` presumably; treat as unverified. |
| `session:compact:before` / `:after` (internal), `before_compaction` / `after_compaction` (typed) | Need a long-context session to trigger compaction. | Unverified; would be a natural capture trigger (pre-compaction transcript). |
| `message:sent` / `message:transcribed` (internal) | `message:sent` needs real channel delivery (probe sessions had `delivery.kind: "none"`); `message:transcribed` needs voice input. | Not applicable to this setup. |
| `gateway:pre-restart` (internal) | SIGUSR2 to the gateway did not produce observable hook traffic (supervisor/worker split: the `openclaw-gateway` supervisor holds the socket; worker restart path unclear). | Not design-critical; `gateway:shutdown` + `session_end(shutdown)` cover the drain case. |

---

## 4. Behavioral findings that shape the design

1. **Injection is real and verified.** Mutating `context.bootstrapFiles` in an `agent:bootstrap` internal-hook
   handler lands verbatim in the model's input. This is the recall path.
2. **Capture has two tiers.** `agent_end` gives you each turn's messages as they complete (incremental,
   dedupe by `runId`); `before_reset` gives you the whole departing transcript in one shot **before** it's
   wiped (batch, keyed by `reason`). Use both: `agent_end` for incremental `record_session`-style capture,
   `before_reset` as the authoritative session-close record.
3. **Reset wipes assembled history but not the sqlite log.** `transcript_events` retains all events
   (verified 11 rows surviving a reset); `chat.history` hides pre-reset content. A hook that needs the
   transcript after the fact can read sqlite, but `before_reset` makes that unnecessary.
4. **The 2-second shutdown drain is real and shared.** `session_end(reason=shutdown)` fired for the active
   session on SIGTERM; docs say shutdown/restart share one 2s budget across all sessions and handlers.
   Final capture on shutdown must be bounded and crash-consistent — treat it as best-effort drain only,
   never the primary capture path.
5. **`messageCount` on `session_end` is unreliable** (always 0 in probes). Use `before_reset.messages.length`
   or count from `agent_end` payloads instead.
6. **Retry duplicates exist.** Model-error retries re-fire `agent:bootstrap` and `agent_end` with the same
   `runId`. Capture must be idempotent on `runId`.
7. **No internal `session:start`/`session:end` events exist** in 2026.9.6 — the internal catalog has
   `command:new`/`command:reset` for lifecycle; true session boundaries are the **typed** `session_start`/
   `session_end`. A lint-ai integration needs **both** systems: internal hooks for bootstrap injection
   (typed hooks have no bootstrap mutator — the closest typed hooks are `before_prompt_build`/
   `agent_turn_prepare`, which were not probed), typed hooks for session lifecycle + per-turn capture.

---

## 5. Proposed hook → lint-ai mapping (design, not yet implemented)

| OpenClaw hook | Kind | lint-ai role | Notes |
|---|---|---|---|
| internal `agent:bootstrap` | internal (mutating) | **Recall.** Query lint-ai for relevant memories for this session/turn; append as a synthetic `bootstrapFiles` entry (e.g. `{name:"LINTAI.md", ...}`). | Verified end-to-end. Idempotent content; fires per turn. |
| typed `agent_end` | Observe | **Incremental capture.** Drain the turn's `messages` (user + assistant) into lint-ai (`record_session`-equivalent). Dedupe by `runId`. | Richest per-turn payload; `ctx` carries sessionId/agentId/channel. |
| typed `before_reset` | Observe | **Authoritative session-close capture.** `messages` = full departing transcript; `reason` ∈ {new, reset, ...}. | Fires before wipe — no re-fetch needed. |
| typed `session_end` | Observe | **Lifecycle bookkeeping.** Mark session closed in lint-ai; `reason` distinguishes new/reset/shutdown. `nextSessionId` links parent→child on `reason=new`. | `messageCount` unreliable; don't use it. |
| typed `session_start` | Observe | **Session open.** Register session in lint-ai; `resumedFrom` links child→parent. | Only fires for child-with-parent creation; plain creates have no typed start. |
| internal `command:reset` / `command:new` | internal (observe) | **Fallback lifecycle signals** if only internal hooks are deployed. | No transcript in payload — pair with typed `before_reset` for capture. |
| internal `gateway:shutdown` / typed `gateway_stop` + `session_end(shutdown)` | Observe | **Bounded final drain only.** Flush pending capture queue; crash-consistent, <2s total. | Never the primary capture path. |
| internal `message:received` / `message:preprocessed` | Observe | Observation only (logging/metrics). | No transcript, no mutation — insufficient for capture or recall. |

**Recommended minimal viable integration:**
1. Internal hook on `agent:bootstrap` → recall + inject (verified working).
2. Typed plugin on `agent_end` → incremental capture, idempotent on `runId`.
3. Typed plugin on `before_reset` → full-transcript capture on session close.
4. Typed plugin on `session_start`/`session_end` → session registry bookkeeping.
5. Bounded flush on `gateway_stop` / `session_end(shutdown)`.

This keeps the two hook systems doing what each does best: internal hooks mutate the prompt pipeline;
typed plugin hooks observe lifecycle with structured payloads.

---

## 6. Open questions / follow-ups (for Luyi)

1. **Typed prompt-mutation hooks** (`before_prompt_build`, `agent_turn_prepare`) were not probed — the
   internal `agent:bootstrap` injection already works and is verified, so they may be unnecessary, but a
   future probe could compare (e.g. whether typed injection survives the same path).
2. **Compaction hooks** (`session:compact:before`, `before_compaction`) are natural capture triggers for
   long sessions; triggering one live needs a long-context run.
3. **`command:stop`** capture semantics: a stopped run's partial transcript stays in the session and is
   picked up by the next `agent_end`/`before_reset` — confirm this is acceptable vs. an explicit stop hook.
4. **Where lint-ai recall queries get their input:** `agent:bootstrap` fires per turn but the payload has no
   user message (it's pre-injection); the turn's message arrives via `message:received` just before.
   The recall hook likely needs to correlate the two (same `sessionKey`, message first, bootstrap second)
   or read the pending input from the session store.

---

## 7. Lifecycle alignment with the Claude/Codex integrations (added per Luyi 2026-09-28)

lint-ai's Claude Code and Codex integrations share one lifecycle model
(`src/integrations/claude_code/hooks/mod.rs`, `src/integrations/codex/hooks/mod.rs` in `~/lint-ai-demo`):
every hook is classified **retrieve** (memory → context) or **capture** (context → memory), and captures
are typed **Checkpoint** (pre-compaction), **Outcome** (a finished unit of work), or **SessionSummary**
(session close). The OpenClaw design below follows that same lifecycle so behavior — and the resulting
document types — stay uniform across hosts. Capture payloads from OpenClaw hooks should be written as
the same Checkpoint/Outcome/SessionSummary document types so retrieval treats all hosts identically.

| Claude/Codex lifecycle | lint-ai action | OpenClaw equivalent (probed §2) | Status |
|---|---|---|---|
| SessionStart → retrieve | inject session-scoped memories | internal `agent:bootstrap` → recall + inject synthetic bootstrap file | ✅ verified end-to-end (§2.1) |
| UserPromptSubmit → retrieve | retrieve against the user's message | `agent:bootstrap` fires per turn right after `message:received`; correlate on `sessionKey` | ⚠️ needs correlation (§6 Q4) |
| Stop → capture **Outcome** | record the finished turn | typed `agent_end` → incremental capture, dedupe by `runId` | ✅ confirmed (§2.2) |
| SessionEnd → capture **SessionSummary** | record session summary | typed `before_reset` (full transcript pre-wipe) + `session_end` bookkeeping | ✅ confirmed (§2.3–2.4) |
| SubagentStart → retrieve / SubagentStop → capture **Outcome** | scope memory to the subagent session | child sessions: `session_start` (fires only with a parent; `resumedFrom` set) → retrieve; `session_end(reason=new)` on the parent → capture | ✅ mappable (§2.4) |
| PreCompact → capture **Checkpoint** | checkpoint before context compaction | compaction hooks NOT observed live | ❌ gap — see below |

**Notes on the alignment:**
- *Checkpoint gap.* Claude/Codex capture a Checkpoint document before compaction so long sessions don't
  lose pre-compaction context. OpenClaw's `session:compact:before` / `before_compaction` hooks were not
  observed (§3). Until a long-context probe confirms them, the mitigation is the `agent_end` incremental
  capture: every turn is already durable, so a missing checkpoint loses less. `before_reset` remains the
  authoritative close.
- *Tool-level retrieve.* Claude/Codex retrieve around tool use (PreToolUse / PostToolUse /
  PermissionRequest). No observed OpenClaw hook gives tool-level granularity; the per-turn
  `agent:bootstrap` injection is the turn-granularity equivalent and is sufficient — bootstrap fires
  every turn, including follow-ups.
- *Two hook systems, one lifecycle.* The retrieve half lives on internal hooks (prompt mutation,
  §2.1); the capture half lives on typed plugin hooks (structured lifecycle payloads, §2.2–2.4).
  This mirrors the Claude/Codex split where hook *transport* differs per host but the
  retrieve/capture lifecycle — and the Checkpoint/Outcome/SessionSummary document types — are identical.
