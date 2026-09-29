// lint-ai OpenClaw typed plugin.
//
// Thin observer: forwards lifecycle events to the lint-ai binary and does
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
import { execFileSync, spawn } from "node:child_process";
import { accessSync, constants, existsSync } from "node:fs";
import { delimiter, join } from "node:path";

// Largest event payload forwarded to the binary (8 MiB, matches its stdin bound).
const MAX_PAYLOAD_BYTES = 8 * 1024 * 1024;
// Shutdown drain budget for the final flush (OpenClaw allows ~2 s total).
const SHUTDOWN_TIMEOUT_MS = 1500;

// Resolve the lint-ai binary, in order:
//   1. plugin config `binaryPath` (plugins.entries.lint-ai.config.binaryPath)
//   2. LINT_AI_BIN environment variable
//   3. `lint-ai` found on PATH (no shell)
// Returns null when nothing resolves; callers fail open with one stderr line.
function resolveBin(config) {
  const fromConfig =
    typeof config?.binaryPath === "string" && config.binaryPath.trim()
      ? config.binaryPath.trim()
      : null;
  if (fromConfig) return fromConfig;
  const fromEnv =
    typeof process.env.LINT_AI_BIN === "string" && process.env.LINT_AI_BIN.trim()
      ? process.env.LINT_AI_BIN.trim()
      : null;
  if (fromEnv) return fromEnv;
  return findOnPath("lint-ai");
}

function findOnPath(name) {
  const pathEnv = process.env.PATH ?? "";
  for (const dir of pathEnv.split(delimiter)) {
    if (!dir) continue;
    if (process.platform === "win32") {
      for (const ext of ["", ".exe", ".cmd", ".bat", ".com"]) {
        const candidate = join(dir, name + ext);
        try {
          if (existsSync(candidate)) return candidate;
        } catch {
          // Ignore unreadable entries.
        }
      }
    } else {
      const candidate = join(dir, name);
      try {
        accessSync(candidate, constants.X_OK);
        return candidate;
      } catch {
        // Not present or not executable; keep looking.
      }
    }
  }
  return null;
}

// Resolve the project root for a capture, in order:
//   1. plugin config `projectRoot` (plugins.entries.lint-ai.config.projectRoot)
//   2. the event's own workspaceDir (multi-workspace gateways)
// Returns null when nothing resolves; callers skip the event (fail open).
function rootOf(config, ctx, event) {
  const fromConfig =
    typeof config?.projectRoot === "string" && config.projectRoot.trim()
      ? config.projectRoot.trim()
      : null;
  if (fromConfig) return fromConfig;
  const root =
    ctx?.workspaceDir ?? event?.context?.workspaceDir ?? event?.workspaceDir;
  if (typeof root === "string" && root.trim()) return root.trim();
  return null;
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
function fire(bin, kind, root, event, ctx) {
  if (!bin || !root) return;
  try {
    const child = spawn(bin, ["--openclaw-hook", kind, root], {
      detached: true,
      stdio: ["pipe", "ignore", "ignore"],
    });
    child.unref();
    // Both handlers fail open: a spawn failure or an early-exiting binary
    // surfaces as an async 'error' event, which would otherwise throw into
    // the host process.
    child.on("error", () => {});
    child.stdin.on("error", () => {});
    child.stdin.write(payloadOf(event, ctx));
    child.stdin.end();
  } catch {
    // Fail open.
  }
}

export default {
  id: "lint-ai",
  name: "Lint-AI memory",
  register(api) {
    const config = api?.pluginConfig ?? {};
    const bin = resolveBin(config);
    if (!bin) {
      console.error(
        "[lint-ai] plugin enabled but no lint-ai binary found: set " +
          "plugins.entries.lint-ai.config.binaryPath, LINT_AI_BIN, or put " +
          "lint-ai on PATH"
      );
    }
    const capture = (kind) => (event, ctx) =>
      fire(bin, kind, rootOf(config, ctx, event), event, ctx);
    api.on("agent_end", capture("agent-end"));
    api.on("before_reset", capture("before-reset"));
    api.on("session_start", capture("session-start"));
    api.on("session_end", capture("session-end"));
    // Compaction capture is intentionally not wired: the compaction hooks
    // were not observed on a live OpenClaw host.
    api.on("gateway_stop", (event, ctx) => {
      // Captures write through synchronously, so this is a no-op flush
      // inside the shared drain budget.
      const root = rootOf(config, ctx, event);
      if (!bin || !root) return;
      try {
        execFileSync(bin, ["--openclaw-hook", "shutdown", root], {
          input: "{}",
          timeout: SHUTDOWN_TIMEOUT_MS,
        });
      } catch {
        // Bounded best-effort flush inside the shared drain budget.
      }
      void event;
      void ctx;
    });
  },
};
