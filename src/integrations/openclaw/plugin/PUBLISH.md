# Publishing the lint-ai plugin to ClawHub

This runbook covers publishing `src/integrations/openclaw/plugin/` to
[ClawHub](https://clawhub.ai) so users can install with
`openclaw plugins install clawhub:lintai`. Our repo stays the source of
truth; ClawHub holds the published copy.

Do NOT publish without Luyi's explicit go-ahead — publishing needs his
ClawHub credentials and is a public release act.

## Preconditions

- [ ] `src/integrations/openclaw/plugin/` is static: no `__PLACEHOLDER__`
      strings remain in `index.js` (`grep -r __LINT_AI plugin/` must be empty).
- [ ] `openclaw.plugin.json` and `package.json` versions match the
      `lint-ai` binary release being published alongside (currently 0.3.0).
- [ ] `node --check src/integrations/openclaw/plugin/index.js` passes.
- [ ] `cargo test --all-targets --features agent-integrations` is green
      (covers the `--openclaw-install` path that ships these same files).
- [ ] The binary-resolution behavior was exercised (see "Smoke test" below).
- [ ] CHANGELOG-worthy behavior changes are noted in the release notes.

## Publish steps

```bash
cd src/integrations/openclaw/plugin
# 1. Sanity: the package lints clean
node --check index.js

# 2. Publish (first time: clawhub login first)
clawhub publish
```

If the `clawhub` CLI needs an explicit package path or version bump, follow
its prompts; keep the published version identical to the two manifest files.

## Post-publish verification

```bash
# In a scratch OpenClaw home (never your real one):
export OPENCLAW_STATE_DIR=/tmp/clawhub-verify
openclaw plugins install clawhub:lintai
openclaw plugins enable lint-ai
openclaw plugins inspect lint-ai --runtime --json
```

Then:

1. Confirm `api.pluginConfig` is reachable (set
   `plugins.entries.lint-ai.config.projectRoot` to a scratch project and
   check the plugin reads it — see "Smoke test").
2. Trigger `agent_end` on a scratch run and confirm a capture lands in the
   project's `.lint-ai/memory/` store.
3. Confirm fail-open: with no `lint-ai` on `PATH` and no `binaryPath`, the
   Gateway starts normally and the plugin logs exactly one stderr line.

## Smoke test (no Gateway needed)

Exercises binary resolution and the event wiring with a stub binary:

```bash
PLUGIN=src/integrations/openclaw/plugin
mkdir -p /tmp/fakebin
printf '#!/bin/sh\necho "hook args: $@" >> /tmp/fakebin/calls.log\n' > /tmp/fakebin/lint-ai
chmod +x /tmp/fakebin/lint-ai
cat > /tmp/driver.mjs <<'EOF'
import plugin from "/tmp/plugin-under-test/index.js";
const handlers = {};
const fakeApi = {
  pluginConfig: { projectRoot: "/tmp/proj" },
  on: (name, fn) => { handlers[name] = fn; },
};
plugin.register(fakeApi);
await handlers["agent_end"]({ context: {} }, {});
await new Promise((r) => setTimeout(r, 500));
EOF
mkdir -p /tmp/plugin-under-test && cp "$PLUGIN"/index.js /tmp/plugin-under-test/
PATH="/tmp/fakebin:$PATH" node /tmp/driver.mjs
cat /tmp/fakebin/calls.log   # expect: hook args: --openclaw-hook agent-end /tmp/proj
```

Also verify the fail-open path: run the driver with an empty `PATH` and no
config — it must exit 0 and print the one-line stderr diagnostic.

## Versioning discipline

- The plugin version tracks the `lint-ai` binary release it was verified
  against (both manifests carry the same version).
- Bump on any behavior change to `index.js`; manifest-only changes
  (descriptions, README) may ride the next binary release.

## Future work (out of scope for this publish)

- The file-based recall hooks (`hooks/handler.js`, `agent:bootstrap`) are
  still installed only via `lint-ai --openclaw-install`; migrating them to
  SDK `registerHook` would make the ClawHub install fully self-contained.
  Hook schemas for that migration must be probed against a live OpenClaw
  host before shipping (standing rule).
