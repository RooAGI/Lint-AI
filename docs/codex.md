# Codex Integration

Lint-AI can run as a Codex memory layer and provide persistent, segmented
project memory through Codex lifecycle hooks.

Codex support is isolated behind a non-default Cargo feature. Build a local
integration-enabled binary with:

```bash
cargo build --release --features codex
```

The default core library and binary do not expose Codex-specific protocol
types, commands, or configuration behavior. Published standalone CLI release
assets can enable the feature explicitly.

## Install

From the repository root:

```bash
./lint-ai --codex-install /path/to/repo
```

By default this should:

- merge a `mcp_servers.lint-ai` entry into `~/.codex/config.toml`
- merge Lint-AI commands into `~/.codex/hooks.json` for the supported Codex
  lifecycle events
- preserve unrelated MCP servers, hooks, and settings

Codex project memory should be persisted under:

```text
<project>/.lint-ai/codex-memory/
```

`SessionStart`, `UserPromptSubmit`, `UserPromptExpansion`, `PreToolUse`,
`PermissionRequest`, `PostToolUse`, and `SubagentStart` retrieve context.
`PreCompact`, `PostCompact`, `Stop`, `SessionEnd`, and `SubagentStop` capture
bounded session memory. A new session segment is created lazily by the first
capture hook, not by `SessionStart`.

Supported Codex hooks:

- `SessionStart`
- `UserPromptSubmit`
- `PreToolUse`
- `PermissionRequest`
- `PostToolUse`
- `UserPromptExpansion`
- `PreCompact`
- `PostCompact`
- `Stop`
- `SessionEnd`
- `SubagentStart`
- `SubagentStop`

Durable captures should be compact structured records rather than raw
conversation transcripts. Retrieved records should include capture/current Git
revisions and an exact, ancestor, diverged, or unknown revision status.

Retrieval should inject at most one preferred document per session and use
bounded query-relevant excerpts instead of complete records.

## Inspect Memory

Inspect the persisted store summary:

```bash
lint-ai --inspect-index .lint-ai/codex-memory
```

Inspect the documents at each indexing stage:

```bash
lint-ai --inspect-index .lint-ai/codex-memory --inspect-view source-documents
lint-ai --inspect-index .lint-ai/codex-memory --inspect-view records
lint-ai --inspect-index .lint-ai/codex-memory --inspect-view segments
```

- `source-documents` should show the reconstructed public ingestion objects.
- `records` should show enriched `DocRecord` values used to build the query
  index.
- `segments` should show segment IDs, document membership, and profile sizes.

All views should emit JSON and can be filtered with `jq`.

## Serve

Run the Codex integration server directly:

```bash
./lint-ai --codex-serve /path/to/repo
```

The server should expose two tools:

- `search`: run a corpus query and return ranked results plus diagnostics
- `info`: return basic workspace information

## Notes

- The integration uses Codex's documented lifecycle and hook/config layering
  rather than inventing a separate memory system.
- Hook failures should fail open and not block normal Codex execution.
- Captured transcript input should be bounded and should exclude tool-use
  blocks.
- The server should use the existing Rust retrieval stack and the current
  workspace path as its index root.
- Existing Codex config entries should be preserved when the installer runs.
