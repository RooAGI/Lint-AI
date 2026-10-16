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

Search responses include a stable `search_id` that connects the recorded query
and result to the review tools. Use `review_search` to inspect a recorded
search, then use `search_feedback` to submit feedback about that one search.
For an ordinary search, the agent can send feedback directly with the returned
ID; it does not need to call `review_search` first. `review_search` is for
looking up an exact past search or working through the paginated unreviewed
queue. The conditional guidance in a search response prompts the agent to send
feedback when results are inadequate. It does not call the feedback tool or
judge the result automatically.

| Tool | Purpose |
|---|---|
| `search` | Search the indexed workspace and return ranked results. Requires `query`; accepts `top_k` (1–20). Each hit carries `doc_id`, its original 1-based `rank`, `source_hash`, `created_at`, and `session_id`; hit text is prefixed with `[session date: ...]` when available. Use `doc_id` and `source_hash` as the target and expected revision when calling `enrich_doc`. The response includes a stable `search_id`. If results miss the question or important context, call `search_feedback` with an overall label and one document judgment for every returned hit; adequate results need no feedback call. |
| `review_search` | With no `search_id`, return a page of unreviewed searches. Accepts optional `cursor` and `page_size` (default 10, maximum 100). With `search_id`, return that exact recorded search, its result, and available source documents, whether or not it was reviewed. Reading does not mark a search reviewed. |
| `search_feedback` | Submit an overall usefulness label, explanation, and a relevance judgment (`relevant`, `partially_relevant`, or `not_relevant`) for every document returned by one `search_id`. The service validates complete coverage and records the original rank and source hash from the immutable search snapshot. It may include grounded additions to supported query-only rules. Judgments support evaluation; they do not immediately change ranking or source documents. |
| `enrich_doc` | Enrich one existing source document by `doc_id`, using its expected source hash and hash-cited evidence documents. Routes the validated update through `MemoryService`; it does not label a search. |
| `info` | Return the workspace root, indexed document count, and the transport and endpoint that served the current request. |
| `list_memories` | List bounded previews of indexed agent memories. Accepts `limit` (1–100, default 20). |
| `add_memory` | Add an explicitly chosen memory. Accepts `content`, `request_id`, `session_id`, optional `role`, and optional `supersedes_id` to correct an older memory. |
| `get_memory` | Read one memory by ID. Accepts `memory_id` and optional `include_inactive`. |
| `delete_memory` | Permanently delete one MCP-owned memory by ID. Returns `deleted: false` when it is missing or belongs to another owner. |
| `record_session` | Start, stop, or inspect capture-only session recording. Use `action: start`, `stop`, or `status`. |
| `enable_membrick` | Enable memory retrieval and capture for future hook events; recording is enabled by default. |
| `disable_membrick` | Disable Membrick retrieval/injection and automatic capture; recording state is unchanged. |
| `membrick_status` | Report independent memory and recording state. |

The shared tools are exposed through the configured Membrick MCP integrations
for Claude Code, Codex, Gemini CLI, Antigravity CLI, and Muse Code. Antigravity
uses the Gemini MCP implementation with AGY-specific provider routing.
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

Search review reads the local `.membrick/mcp-call-history.jsonl` ledger. With
no `search_id`, `review_search` scans from the supplied cursor and returns up
to `page_size` unreviewed search records. The cursor is opaque: pass it back
unchanged. It includes a history-prefix fingerprint, so replacing or editing
the already-read part of the ledger invalidates it. A page can contain fewer
than `page_size` results if reviewed entries were skipped. Supplying
`search_id` returns that exact search even if it has already received
feedback. Reading a page or exact search does not mark anything reviewed.

For feedback on the search just performed, pass its returned `search_id`
directly to `search_feedback`. Use `review_search` when the agent needs to
inspect the recorded query and result again, or when it is reviewing older
searches from the unreviewed queue.

The review flow uses `review_search` for both queue browsing and exact lookup:

```json
{"name":"review_search","arguments":{"page_size":10}}
```

Continue with the returned `next_cursor` when `has_more` is true. To inspect a
particular search, pass its ID:

```json
{"name":"review_search","arguments":{"search_id":"<search_id>"}}
```

Then submit the overall label, explanation, and one relevance judgment for
every returned document with `search_feedback`. Each judgment identifies a
`doc_id` and uses `relevant`, `partially_relevant`, or `not_relevant`; an
optional per-document explanation can clarify the judgment. The service takes
the rank and source hash from the immutable recorded result, rejects missing,
duplicate, or foreign document IDs, and stores the judgments with the search
review. Optional `query_rule_additions` can propose grounded terms for the
supported query-only rule lists. These judgments are evaluation data; they do
not by themselves train or alter online ranking.

To improve a particular document, call `enrich_doc` with its exact `doc_id`,
`expected_source_hash`, and one or more hash-cited evidence documents. It can
add key phrases, update a session-context conclusion/headings/links, or record
an explicit same-scope whole-document supersession. The selected document and
evidence must still match the supplied source hashes. The mutation is applied
through `MemoryService` and published before the tool returns. Retrying an
identical enrichment is idempotent. Search labeling and document enrichment
are independent operations; `enrich_doc` does not require a `search_id`.

Feedback does not train ranking. It can add terms only to the supported
query-only lists documented by the tool schema, and each proposed term must
appear in the reviewed query or its returned source documents. `MemoryService`
validates and adds those terms to the configured `.membrick/rules.json` overlay;
the existing watcher activates valid changes for later searches without a
restart. Extraction-impacting rules are rejected because they require
re-extraction and compatible index publication. Future retrieval can also
benefit from accepted phrases, document links, and session conclusions through
the existing indexed source-document paths.

After a search, provide feedback only when the result missed the question,
omitted useful context, or included misleading material. Adequate results do
not need a feedback call. Labels support evaluation; they do not train ranking
directly. Future search behavior changes only when validated feedback changes
indexed document content or an implemented query rule.
