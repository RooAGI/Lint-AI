# `rooagi_runtime`

Lint-AI integrates with Roo Runtime through Roo's external process hook protocol. The adapter is part of the `lint-ai` binary and persists through `MemoryService` into the workspace's shared `.lint-ai/memory` store.

## Install

Build Lint-AI with the `roo-runtime` feature (included by `agent-integrations`), then run:

```sh
lint-ai --roo-runtime-install
```

This installs user-level hooks in `~/.rooagi/hooks.json`. Set `ROO_CONFIG_HOME` or pass `--roo-runtime-config /path/to/hooks.json` to use another location. Existing hooks are preserved; reinstalling replaces only Lint-AI's own registrations.

The installer registers `run_start`, `agent_turn_start`, `agent_turn_end`, `run_end`, `pre_tool_use`, and `post_tool_use`. Turn start retrieves up to five relevant memories and returns bounded `additionalContext`. Turn end stores the bounded user prompt and assistant response. Pre-tool and post-tool hooks store bounded, redacted tool inputs and results in the shared memory store, scoped by project, run, and tool call. Run start does no memory work. Run end stores its status only if Roo includes `workspaceRoot` in the payload; the current Roo Runtime projection omits it, so run-end capture is skipped. Failed hook calls fail open and return `{}`.

Roo Runtime's process hook executes the configured absolute `lint-ai` binary directly. The hook reads the workspace path and turn data from Roo's JSON invocation. Capture is attributed to the `rooagi_runtime` provider and deduplicated by stable source IDs. Disable memory for a project by writing `{"enabled": false}` to `.lint-ai/rooagi_runtime-state/integration.json`; the integration follows Lint-AI's shared provider enablement convention.

Inline hook input is capped by Lint-AI's standard 8 MiB limit. Roo keeps its stdin envelope within 64 KiB and transports larger trusted tool fields through temporary sanitized JSON files under `.roo/hook-payloads`. Lint-AI verifies their workspace path, byte size and SHA-256 hash before reading them. Each referenced field may contain up to 128 MiB; larger fields fail capture explicitly. Roo removes the files after the hook completes. Stored prompt and response fields and returned recall context are bounded. Hook output contains only Roo's JSON response; diagnostics go to stderr.

If Roo Runtime has a global `sandboxPolicy` configured in `hooks.json`, its data allowlist must include `run_identity`, `workspace_identity`, `conversation_bounded`, and `tool_metadata` for event identity, workspace path, turn text, and tool inputs/results to reach Lint-AI. The policy's filesystem grants must also allow the configured process to read and write the workspace's `.lint-ai` directory. The installer leaves an existing policy unchanged.

## Build without default provider integrations

```sh
cargo build --no-default-features --features heuristic-query-semantics,roo-runtime
```

## Tool evidence and capture coverage

Roo includes the conversation's `sessionId` in lifecycle and tool hooks. Lint-AI groups captures by that session within the workspace and records `session_id` metadata. Run and tool-call identities still distinguish executions and protect against duplicates. If the caller omits a session ID, capture retains the run-based grouping fallback. Workspace recall remains available across sessions.

The post-tool handler uses the `context_provider` sandbox profile and returns a bounded, sanitized result preview in `additionalContext` after successful ingestion. Reinstall hooks after upgrading an existing observer-only installation. Tool-result documents use stable source IDs so replaying the same run/tool-call event replaces the existing document. Arrays retain all entries within the capture byte limit. Sanitized results are split into UTF-8-safe 32 KiB source documents; replacing a capture removes obsolete chunks. Each document records tool, run, workspace and chunk provenance. The `capture_completeness` filter records `complete_sanitized` or `truncated`; redaction still applies in both cases.

Recall searches the workspace's shared provider pool across runs and extracts a query-oriented passage rather than always taking the document prefix. Memory search returns selected evidence, not exhaustive dataset coverage. Original results remain necessary for exact processing.

## On-demand retrieval through Roo's MCP transport

Run `lint-ai --roo-runtime-serve /absolute/workspace/root` to expose the existing shared-memory MCP tools over stdio. A Roo MCP server entry uses `id: "lint-ai"` and `transport: { "type": "stdio", "command": "/absolute/path/to/lint-ai", "args": ["--roo-runtime-serve", "/absolute/workspace/root"] }`. The search capability is `lint-ai__search`. The host must select the root and enforce workspace permissions; tool arguments must not select a different workspace. Configure the graph's search tool with this capability through Roo's existing MCP mechanism. Hosted Builder supplies the reserved `workspace_memory_search` tool using server ID `roo-memory` and capability `roo-memory__search` automatically. Local providers coexist with HTTP MCP providers. Searches return bounded query excerpts and provenance, with coverage marked as a ranked subset; retrieved memory is excluded from tool capture. Node and Python bindings load standard hook configuration from the host-selected project root.

Roo no longer embeds Lint-AI's index or offers `artifact_search` as an internal tool. Original artifact chunk access, execution records and checkpoints remain available. Hosted builder runs provision persistent per-workspace roots under `~/.rooagi/workspaces`; mount that directory in containers. Hook configuration and the external executable must be available in the runtime container.
