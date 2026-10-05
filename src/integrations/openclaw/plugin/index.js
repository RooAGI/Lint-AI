// lint-ai OpenClaw typed plugin.
//
// Thin observer: forwards lifecycle events to the lint-ai server and does
// nothing else. Capture and indexing live in the Rust binary, which is
// stateless — idempotency comes from stable document IDs in the memory store.
// Every failure path is silent (fail-open) so a broken install can never
// block OpenClaw.
//
// This file is static: it ships as-is on ClawHub and is installed verbatim
// by `lint-ai --openclaw-install`. The binary path and project root both
// resolve at runtime (see below); there are no install-time placeholders.
//
// Managed by lint-ai --openclaw-install; safe to reinstall.

// Largest event payload forwarded to the binary (8 MiB, matches its stdin bound).
const MAX_PAYLOAD_BYTES = 8 * 1024 * 1024;
// Resolve the project root for a capture, in order:
//   1. the event's own workspaceDir (multi-workspace gateways: an event that
//      names its workspace must stay in that workspace, never leak into the
//      installer's project)
//   2. plugin config `projectRoot` (plugins.entries.lint-ai.config.projectRoot)
// Returns null when nothing resolves; callers skip the event (fail open).
function rootOf(config, ctx, event) {
  const root =
    ctx?.workspaceDir ?? event?.context?.workspaceDir ?? event?.workspaceDir;
  if (typeof root === "string" && root.trim()) return root.trim();
  const fromConfig =
    typeof config?.projectRoot === "string" && config.projectRoot.trim()
      ? config.projectRoot.trim()
      : null;
  return fromConfig;
}

function trimEvent(event) {
  // Drop message contents for the session_end bookkeeping path; capture hooks
  // need them, so this only trims when the caller asks for it.
  return event;
}

function payloadOf(event, ctx) {
  const payload = JSON.stringify({ event: trimEvent(event), ctx: ctx ?? {} });
  if (payload.length > MAX_PAYLOAD_BYTES) {
    // Trim message bodies before giving up: keep the envelope, drop content.
    const slim = { ...event };
    if (Array.isArray(slim.messages)) {
      slim.messages = slim.messages.map((m) => ({
        role: m?.role,
        customType: m?.customType,
        details: m?.details,
      }));
    }
    return JSON.stringify({ event: slim, ctx: ctx ?? {} });
  }
  return payload;
}

// Fire-and-forget: capture must never block the agent loop.
function fire(serverUrl, serverToken, kind, root, event, ctx) {
  if (!root) return;
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    void fetch(`${serverUrl}/integrations/openclaw/hooks/${kind}`, {
      method: "POST",
      headers: { "content-type": "application/json", ...(serverToken ? { authorization: `Bearer ${serverToken}` } : {}) },
      body: payloadOf(event, { ...ctx, workspaceDir: root }),
      signal: controller.signal,
    }).catch(() => {}).finally(() => clearTimeout(timeout));
  } catch {
    // Fail open.
  }
}

export default {
  id: "lint-ai",
  name: "Lint-AI memory",
  register(api) {
    const config = api?.pluginConfig ?? {};
    const serverUrl = (config.serverUrl ?? process.env.LINTAI_SERVER_URL ?? "http://127.0.0.1:8080").replace(/\/$/, "");
    const serverToken = config.serverToken ?? process.env.LINTAI_SERVER_TOKEN ?? "";
    const capture = (kind) => (event, ctx) =>
      fire(serverUrl, serverToken, kind, rootOf(config, ctx, event), event, ctx);
    api.on("agent_end", capture("agent-end"));
    api.on("before_reset", capture("before-reset"));
    api.on("session_start", capture("session-start"));
    api.on("session_end", capture("session-end"));
    // Compaction capture is intentionally not wired: the compaction hooks
    // were not observed on a live OpenClaw host.
    api.on("gateway_stop", (event, ctx) => {
      // Captures are sent as requests as events arrive; nothing is buffered.
      void event;
      void ctx;
    });
  },
};
