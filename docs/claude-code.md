# Claude Code Integration

Lint-AI can run as a Claude Code MCP server and provide persistent, segmented
session memory through Claude Code lifecycle hooks.

Claude Code support is isolated behind the non-default `claude-code` Cargo
feature. Build a local integration-enabled binary with:

```bash
cargo build --release --features claude-code
```

The default core library and binary do not compile or expose Claude-specific
protocol types, commands, or configuration behavior. Published standalone CLI
release assets enable the feature explicitly.

The hook architecture and API mapping are defined in
[Claude Code Hooks Design](claude-code-hooks-design.md).
The A/B methodology for accuracy, latency, token use, and repeated exploration
is defined in
[Claude Code Performance Test Design](claude-code-performance-tests.md).

## Install

From the repository root:

```bash
./lint-ai --claude-code-install /path/to/repo
```

By default this:

- merges an `mcpServers.lint-ai` entry into `~/.claude.json`
- merges Lint-AI commands into `~/.claude/settings.json` for `SessionStart`,
  `UserPromptSubmit`, `UserPromptExpansion`, `PreCompact`, `Stop`, and
  `SessionEnd`
- installs the project-scoped `lint-ai-memory` skill under
  `.claude/skills/lint-ai-memory/SKILL.md`, directing Claude to use
  `mcp__lint-ai__search` for prior project context
- preserves unrelated MCP servers, hooks, and settings

Claude session memory is persisted under:

```text
<project>/.lint-ai/claude-memory/
```

`SessionStart`, `UserPromptSubmit`, and `UserPromptExpansion` retrieve context.
`PreCompact`, `Stop`, and `SessionEnd` capture bounded session memory. A new
session segment is created lazily by the first capture hook, not by
`SessionStart`.

Durable captures are compact structured records rather than raw conversation
transcripts. Retrieved records include capture/current Git revisions and an
exact, ancestor, diverged, or unknown revision status.

Retrieval injects at most one preferred document per session and uses bounded
query-relevant excerpts instead of complete records.

## Inspect Memory

Inspect the persisted store summary:

```bash
lint-ai --inspect-index .lint-ai/claude-memory
```

Inspect the documents at each indexing stage:

```bash
lint-ai --inspect-index .lint-ai/claude-memory --inspect-view source-documents
lint-ai --inspect-index .lint-ai/claude-memory --inspect-view records
lint-ai --inspect-index .lint-ai/claude-memory --inspect-view segments
```

- `source-documents` shows the reconstructed public ingestion objects.
- `records` shows enriched `DocRecord` values used to build the query index.
- `segments` shows segment IDs, document membership, and profile sizes.

All views emit JSON and can be filtered with `jq`.

## Serve

Run the MCP server directly:

```bash
./lint-ai --claude-code-serve /path/to/repo
```

The server exposes two tools:

- `search`: run a corpus query and return ranked results plus diagnostics
- `info`: return basic workspace information

## Verify Installation

After installation, verify that the configured MCP process can start and
complete both the MCP initialize and tool-list handshakes:

```bash
LINT_AI_MCP_HEALTH_PATH=/tmp/lint-ai-claude-mcp-health.json \
  ./lint-ai --claude-code-verify-mcp /path/to/repo --mcp-timeout-ms 30000
```

The command emits JSON with startup and handshake timings, protocol version,
tool count, and captured server diagnostics. A healthy result has
`"status": "healthy"`. Use a longer timeout for the first run on a large
repository because the persistent index may need to be built.

## Notes

- The integration uses Claude Code's MCP path, which is the supported external-tool mechanism.
- Hook failures fail open and do not block normal Claude Code execution.
- Captured transcript input is bounded and excludes tool-use blocks.
- The server uses the existing Rust retrieval stack and current workspace path as its index root.
- Existing Claude Code config entries are preserved when the installer runs.
