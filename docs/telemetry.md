# Telemetry

Membrick telemetry is local operational data used to understand query behavior
and provider activity. It is separate from indexed memory and session archives.
Aggregate telemetry is bounded. Detailed MCP tool history retains complete
arguments and responses.

## Telemetry streams

Membrick currently has four related streams:

| Stream | What it records | Storage |
| --- | --- | --- |
| HTTP query telemetry | Request count, errors, empty results, and latency distribution | In-memory in the running HTTP server |
| MCP query telemetry | The same project-level query aggregates for provider MCP processes | `.membrick/query-telemetry.json` |
| MCP tool history | Detailed MCP tool inputs and the response returned to the provider | `.membrick/mcp-call-history.jsonl` |
| Provider lifecycle telemetry | Provider hook events and available token usage | `.membrick/provider-telemetry/{provider}.json` |

HTTP query telemetry is process-local. Provider MCP processes persist their
query aggregates so the dashboard can observe queries handled outside the HTTP
server process. The query aggregates are project-wide and do not attribute a
request to a provider. Detailed history is recorded after every provider
`tools/call`, including search, memory reads and writes, memory controls, and
update actions. The Claude, Codex, Gemini CLI, AGY, Muse, OpenClaw, Hermes, and
Roo runtime adapters all use the same recorder.

## Query aggregates

Queries are grouped into five-second buckets. The most recent 120 buckets are
kept, giving a rolling ten-minute window. Each bucket tracks request count,
error count, empty-result count, and a latency histogram. The dashboard derives
request rate and latency summaries from these aggregates. Query strings,
session IDs, and document contents are not stored in this aggregate stream.

The detailed MCP tool history is an append-only JSON Lines file with no
retention, entry-count, or payload-size limit. Each entry includes provider,
tool name, complete arguments, timestamp, elapsed time, success or error
status, and the complete response returned by the resident handler. Session
IDs and content are retained as supplied. On Unix, the file is created with
owner-only permissions. Entries can contain private project information and
will grow until manually removed, so treat this file as sensitive and account
for its disk usage. It is an operational archive, not a semantic memory
document, and is not included in normal memory search results.

## Provider lifecycle events

Provider events are kept in a separate JSON ledger for each provider. Each
ledger retains at most the latest 500 events. The event schema can include:

- event name, category, and timestamp
- a one-way session key rather than the provider's raw session ID
- bounded agent, turn, and tool metadata when the provider supplies it
- redacted, bounded previews for selected prompt or tool activity
- retrieval timing and result counts when available
- normalized token counts when available

Lifecycle telemetry is written independently of session recording. Turning off
session recording prevents full event archives from being written; it does not
by itself disable operational lifecycle telemetry. Full session content is not
copied wholesale into the telemetry ledger.

## Provider coverage

Coverage depends on what each provider exposes and which integration features
are compiled:

| Provider | Query aggregates | Lifecycle events | Token usage |
| --- | --- | --- | --- |
| Claude Code | Yes | Yes | Yes, when hook or transcript data includes usage |
| Codex | Yes | Yes | Yes, when hook or transcript data includes usage |
| Gemini CLI | Yes | Yes | Supported when usage data is available |
| AGY | Yes | Yes | Supported when usage data is available |
| Muse Code | Yes | Yes | Not currently wired to the shared transcript usage path |
| OpenClaw | Yes | Yes | Not currently wired to the shared transcript usage path |
| Hermes | Yes | No Hermes lifecycle events are currently wired into the provider telemetry ledger | Not currently wired to the shared transcript usage path |

“Yes” means the code path is instrumented; it does not mean telemetry exists
for a provider in a particular workspace. A provider remains `not_observed`
until it emits an event. Token usage may be unavailable when the host does not
include usage data in its hook payload or transcript.

The repository also contains a separate optional Python Hermes hooks plugin
for memory capture. That plugin is not connected to the provider telemetry
ledger described here, so it does not make Hermes lifecycle telemetry appear
on the dashboard.

## Dashboard and API

The local dashboard reads query and provider telemetry through these endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/status` | Index state, query summary, and integration cards |
| `GET /api/timeseries` | Five-second query aggregates over the rolling window |
| `GET /api/integrations` | Provider readiness |
| `GET /api/sessions` | Recent sessions and sanitized lifecycle counts |
| `GET /api/events` | Latest sanitized provider events |
| `GET /api/sessions/:session_key/events` | Events for one opaque session key |
| `GET /api/metrics` and `GET /metrics` | Aggregate metrics, including Prometheus text format |

See [HTTP server](server.md) for starting the dashboard and configuring its
project root and authentication.

## File watcher changes

Workspace file diffs are not operational telemetry. The watcher captures
bounded before-and-after snapshots and persists eligible diffs as memory
`SourceDocument`s. It does not send file contents or diffs to the telemetry
ledgers. This keeps source content in the memory store and operational metrics
in the telemetry path.

Content capture skips `.env` files, private-key files, and standard credential
locations such as `.ssh` and `.aws`. Known credential values and sensitive
assignments are redacted from captured diff lines. This is best-effort filtering;
custom secret formats and arbitrary secret filenames may need additional ignore
rules. Existing stored documents are not retroactively sanitized.

On Unix, snapshot reads reject symlinks in every path component and use held
directory descriptors to avoid check/open races. Reads are bounded to 256 KiB.
Automatic snapshots and `rooagi_runtime` file payload references are unavailable
on non-Unix targets until an equivalent safe reader is implemented. Watcher
invalidation and inline hook payloads continue to work there.

## Implementation references

- `src/telemetry.rs` implements query aggregates, provider event ledgers, and
  MCP call history.
- `src/integrations/session_recording.rs` normalizes provider events and token
  usage, and separates operational telemetry from full session recording.
- `src/pipeline/watcher.rs` captures bounded workspace file snapshots for
  diffs.
