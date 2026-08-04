# Agent Integrations

This directory contains the Claude Code and Codex integrations, plus the
shared MCP infrastructure used by both clients.

Lint-AI has two complementary integration paths:

1. **Lifecycle hooks** retrieve and capture session memory automatically.
2. **MCP** exposes the indexed workspace and memory store as callable tools.

Hooks and MCP can be enabled together. They use the same Rust retrieval stack,
but they serve different purposes: hooks provide lifecycle automation, while
MCP lets the agent explicitly request workspace or memory search.

## Hooks Versus MCP

Hooks and MCP are separate integration mechanisms. Enabling the MCP server
does not automatically make the client call `search`, and enabling hooks does
not require MCP to be running.

| Concern | Lifecycle hooks | MCP server |
| --- | --- | --- |
| Invocation | Client lifecycle event starts Lint-AI automatically | Client or model explicitly calls an MCP tool |
| Retrieval | Runs at prompt/session lifecycle events | Runs when `search` is selected |
| Capture | Stop, compaction, and session-end hooks persist memory | MCP does not automatically capture conversations |
| Primary role | Automatic durable session memory | Explicit workspace and memory search |
| Process model | Short-lived hook processes | Long-running JSON-RPC server |
| Model control | Retrieval does not depend on tool selection | Tool selection is model/client controlled |
| Failure behavior | Fails open so normal client work continues | Reports startup or tool errors to the MCP client |

Use hooks when memory should be retrieved and captured automatically as part
of the agent lifecycle. Use MCP when the agent needs an explicit search over
the current workspace and persisted memory. Use both when automatic session
memory and on-demand repository search are required.

The two paths share the retrieval engine and provider-specific persistent
stores, but they have different cost centers. Hooks add process launch,
bounded retrieval, capture, and context-injection work. MCP adds server startup,
index initialization or loading, JSON-RPC transport, and model-selected tool
calls. Keep their latency and reliability measurements separate.

## Memory Layer and Hooks

The Lint-AI memory layer is the durable project-local store used to carry
relevant context between agent sessions. It is not the client provider's
built-in memory layer. The provider's native memory and Lint-AI memory can be
enabled independently, or both can be enabled for comparison and fallback.

Each provider has an isolated store because its lifecycle payloads and
transcript formats differ:

```text
<project>/.lint-ai/claude-memory/
<project>/.lint-ai/codex-memory/
```

The stores share the Rust indexing and retrieval implementation, but records
are tagged with their provider and session provenance. This prevents Claude
and Codex hook payloads from being silently treated as the same lifecycle
format while still allowing common search and index behavior.

### Retrieval hooks

Retrieval hooks run before or around client work and inject bounded context into
the next model operation. They:

1. Read the provider-specific hook payload.
2. Build a query from the user prompt, session metadata, and relevant tool
   fields.
3. Open or reuse the provider's persistent index.
4. Rank relevant memories using the shared retrieval stack.
5. Select bounded, query-relevant excerpts rather than complete transcripts.
6. Return additional context to the client.

Retrieval is bounded by result count and context bytes. The adapters prefer the
most relevant document per session and include capture/current Git revision
provenance when available. Revision status can be exact, ancestor, diverged,
or unknown, so stale memories can be identified rather than presented as
current facts without qualification.

### Capture hooks

Capture hooks run at lifecycle boundaries and persist compact memory records.
They:

1. Locate the provider's session or transcript input.
2. Extract bounded conversational messages.
3. Exclude tool-use blocks and oversized tool-result content.
4. Redact sensitive material according to the provider adapter rules.
5. Convert the result into a structured source document with session and
   revision provenance.
6. Upsert the document into the provider's persistent store.

Capture is intentionally deferred. A session-start event does not create an
empty memory record or eagerly write a session segment. The first meaningful
capture event creates the segment, which avoids polluting the store with empty
sessions.

### Provider lifecycle mapping

Claude and Codex expose different hook sets, but the memory-layer roles are
the same:

| Role | Claude Code | Codex |
| --- | --- | --- |
| Prompt retrieval | `SessionStart`, `UserPromptSubmit`, `UserPromptExpansion` | `SessionStart`, `UserPromptSubmit`, `UserPromptExpansion` |
| Tool-aware retrieval | Provider MCP or skill guidance | `PreToolUse`, `PermissionRequest`, `PostToolUse` when enabled |
| Compaction capture | `PreCompact` | `PreCompact`, `PostCompact` |
| Session capture | `Stop`, `SessionEnd` | `Stop`, `SessionEnd`, `SubagentStop` |
| Subagent lifecycle | Provider-dependent | `SubagentStart`, `SubagentStop` |

The common comparison profile uses the overlapping Claude-equivalent lifecycle
events. Codex-specific tool and subagent hooks remain available in production
but are excluded from the fair hooks-only comparison unless explicitly being
tested.

### Failure and isolation rules

Hook processes must fail open: a retrieval or capture error must not block the
agent from continuing its task. Errors should be observable through logs or
diagnostics while the client receives a safe empty or partial hook response.

Memory stores are project-scoped rather than global so one repository cannot
accidentally retrieve another repository's decisions. MCP and hooks for the
same provider use the same project store, allowing MCP search to find records
captured by hooks after synchronization.

## Layout

```text
src/integrations/
  claude_code/       Claude Code installer, hooks, MCP server, and skill
  codex/             Codex installer, hooks, and MCP server
  mcp_index.rs       Shared persistent-index lifecycle and synchronization
  mcp_health.rs      Shared MCP startup and initialize/tools-list verification
  mcp_transport.rs   Shared JSON-RPC framing and response transport
```

The provider modules own client-specific configuration, hook payloads,
transcript parsing, and MCP tool adapters. Shared MCP modules must not depend
on Claude- or Codex-specific protocol details.

## MCP Server

Both providers expose the same logical tools:

- `search`: query the indexed workspace and persisted agent memory; returns
  ranked results and diagnostics.
- `info`: report basic workspace/index information; it is a status tool, not a
  memory search.

The server accepts both newline-delimited JSON-RPC and MCP
`Content-Length`-framed messages. The normal startup sequence is:

1. Start the provider-specific MCP process.
2. Complete `initialize`.
3. Complete `tools/list`.
4. Initialize or open the persistent store when the first store-dependent
   request arrives.
5. Serve `search` and `info` requests.

Index state is persisted below the project and synchronized before searches:

```text
<project>/.lint-ai/claude-memory/
<project>/.lint-ai/codex-memory/
```

The stores are separate because Claude and Codex capture different lifecycle
payloads and document schemas. They use the same indexing and retrieval
implementation and can be queried through the same MCP tool contract.

## Claude Code

Build with the Claude feature:

```bash
cargo build --release --features claude-code
```

Install into a project:

```bash
./lint-ai --claude-code-install /path/to/repo
```

Installation configures the project MCP entry, lifecycle hooks, and the
project-scoped skill:

```text
/path/to/repo/.claude/skills/lint-ai-memory/SKILL.md
```

The skill guides Claude to call `mcp__lint-ai__search` first for questions
about project history, prior decisions, or earlier work. Skills are
model-invoked guidance, not enforcement; Claude may still choose repository
tools instead. A direct MCP client can verify server capability independently
of model tool selection.

Run the server directly:

```bash
./lint-ai --claude-code-serve /path/to/repo
```

Verify startup and tool discovery:

```bash
./lint-ai --claude-code-verify-mcp /path/to/repo --mcp-timeout-ms 30000
```

## Codex

Build with the Codex feature:

```bash
cargo build --release --features codex
```

Install into a project:

```bash
./lint-ai --codex-install /path/to/repo
```

Installation updates the Codex MCP configuration and hook settings while
preserving unrelated entries. Codex uses its documented hook events for
prompt retrieval, tool-use context, compaction, and session capture.

Run the server directly:

```bash
./lint-ai --codex-serve /path/to/repo
```

Verify startup and tool discovery:

```bash
./lint-ai --codex-verify-mcp /path/to/repo --mcp-timeout-ms 30000
```

## Hook and MCP Responsibilities

Hooks should remain small and fail open. They are responsible for:

- extracting bounded, non-tool transcript context;
- retrieving relevant durable memory at lifecycle points;
- capturing compact structured session records; and
- injecting bounded context into the client lifecycle response.

MCP is responsible for:

- serving explicit `search` and `info` requests;
- indexing the current workspace and persisted memory;
- synchronizing newly captured memory before a query; and
- returning ranked results with diagnostics and timing data.

Do not duplicate indexing or transport behavior in the provider modules. Add
provider-specific behavior only when the client protocol or lifecycle payload
requires it.

## Testing

Run provider-focused tests:

```bash
cargo test --features claude-code claude_code
cargo test --features codex codex
```

The MCP health commands validate process startup, `initialize`, and
`tools/list`. To validate the actual search path, use an MCP client to send a
`tools/call` request with:

```json
{
  "name": "search",
  "arguments": {
    "query": "project memory and prior decisions",
    "top_k": 3
  }
}
```

Keep direct MCP-client results separate from Claude or Codex benchmark results:
the client test measures server behavior, while an agent benchmark also
includes model tool selection, prompt interpretation, and client latency.

## Design Rules

- Preserve unrelated client configuration during installation.
- Keep hooks fail-open and bound transcript/context sizes.
- Keep MCP startup observable through health output and optional tracing.
- Reuse the shared transport, index lifecycle, and health modules.
- Treat MCP availability and model invocation as separate measurements.
- Keep provider-specific stores isolated even though their retrieval engine is
  shared.
