# MCP Interface

Each agent adapter exposes a provider-local MCP server over the host client's
configured MCP transport. The server gives the agent direct access to Membrick
search, inspection, memory controls, and session controls.

Bundled Rust provider MCP adapters and hook subcommands reach one shared
resident workspace owner. They try the user-private local IPC endpoint first
(Unix socket on macOS/Linux, named pipe on Windows), then an optional loopback
HTTP listener configured in `.membrick/server.toml`, then an atomic filesystem
request queue under `.membrick/runtime/requests`. The first request starts the
resident automatically. Every transport feeds the same router and
`MemoryService`; a project ownership lock prevents a second resident from
opening the same workspace. The bundled adapters do not read environment
variables to select a server or transport.
Resident sockets, startup coordination files, and diagnostics are kept in the
project's `.membrick/runtime/` directory. At resident startup, logs over 1 MiB
are rotated to `.membrick/runtime/resident.log.1`.
See [Local IPC server architecture](local-ipc-architecture.md) for the request
flow, endpoint permissions, and its relationship to the standalone HTTP
server.

## Where each mode applies

MCP and server mode are general Membrick interfaces; they are not restricted to
agent clients:

| Mode | Available to | What it provides |
|---|---|---|
| Core/file-based CLI | Any workflow | Index and query local files through the `MemoryService` backed CLI. |
| HTTP server | Any application or client | `/add`, `/search`, `/delete`, `/supersede`, and `/expire` over HTTP. |
| MCP server | Any MCP-capable client | Search, memory inspection, and integration controls through JSON-RPC tools. |
| Lifecycle hooks | Supported agents | Automatic retrieval, context injection, and turn/session capture. |

Only the last row is agent-specific. An application does not need Claude Code,
Codex, Gemini CLI, or AGY to use the core index, HTTP API, or MCP interface.

## Tools

| Tool | Purpose |
|---|---|
| `search` | Search the indexed workspace and return ranked results. Requires `query`; accepts `top_k` (1–20). Each hit carries `created_at` and `session_id`, and the hit text is prefixed with `[session date: ...]` when the memory has a timestamp, so relative date expressions ("yesterday", "last month") resolve against an absolute date. |
| `info` | Return the workspace root, indexed document count, and the transport and endpoint that served the current request. |
| `list_memories` | List bounded previews of indexed agent memories. Accepts `limit` (1–100, default 20). |
| `add_memory` | Add an explicitly chosen memory. Accepts `content`, `request_id`, `session_id`, optional `role`, and optional `supersedes_id` to correct an older memory. |
| `get_memory` | Read one memory by ID. Accepts `memory_id` and optional `include_inactive`. |
| `delete_memory` | Permanently delete one MCP-owned memory by ID. Returns `deleted: false` when it is missing or belongs to another owner. |
| `record_session` | Start, stop, or inspect capture-only session recording. Use `action: start`, `stop`, or `status`. |
| `enable_membrick` | Enable memory retrieval and capture for future hook events; recording is enabled by default. |
| `disable_membrick` | Disable Membrick retrieval/injection and automatic capture; recording state is unchanged. |
| `membrick_status` | Report independent memory and recording state. |

The same tool names are available to Claude Code, Codex, Gemini CLI, and
Antigravity CLI through their configured Membrick MCP integration.
The MCP memory-write tools are scoped to the stable `mcp` owner within the
workspace. Lifecycle-captured memories use a separate owner and are not deleted
by `delete_memory`.
The `info` transport status describes the connection that served that request:
`ipc`, `http`, or `file-queue`. `connected` means the resident owner responded
successfully through that transport.

## Search versus hooks

MCP calls are explicit agent actions. Lifecycle hooks are automatic client
callbacks. A hook can retrieve memory and inject it into the host agent without
the model calling `search`; the model can also call `search` directly when it
needs workspace evidence.

`memory_add`/`memory_search` are the direct HTTP memory API equivalents used by
the standalone server contract. A successful add refreshes the index before it
returns, so a subsequent search can see the new memory.

## State and safety

Memory and recording state are project- and provider-scoped. Recording is
capture-only: it writes session events but does not inject them as memory.
Recorded content is bounded and redacted according to the session-recording
policy. Tool and hook failures are fail-open so the host agent can continue.

For HTTP deployment and authentication, see [Server and MCP](server.md).
