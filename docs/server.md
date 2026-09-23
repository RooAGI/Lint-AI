# Lint-AI server

`server` exposes a versioned memory API backed by `MemoryService`. The storage
engine is an implementation detail of that service; clients should use the
memory endpoints rather than depend on `IndexStore`. It is a standalone HTTP interface for any application;
`POST /add/batch` accepts up to 128 normal add requests and publishes one
snapshot after the batch, while each request still enforces the 1,024-message
limit.

Identity is an API concern: `MemoryService` translates `user_id` into a generic
document filter, while `IndexStore` and segmented indexing remain unaware of
users. `group_id`/`session_id` is the core segmentation key and must not be
treated as a user identifier.
it does not require Claude Code, Codex, Gemini CLI, or Antigravity.

## Run locally

Use an explicit index directory for a persistent evaluation instance:

```bash
cargo run --release --bin server -- \
  --bind 127.0.0.1:8080 \
  --index /var/lib/lint-ai/memory-index \
  --server-token "$SERVER_TOKEN"
```

`--bind` defaults to `127.0.0.1:8080`. If `--index` is omitted, the server
discovers indexes under `<project-root>/.lint-ai` and uses the shared
`workspace-memory` store when present, falling back to a legacy Codex MCP index,
the first discovered index, or an in-memory store. Use `--index` to select one
store explicitly.
Provider lifecycle telemetry is read from `<project-root>/.lint-ai/provider-telemetry`;
the project root defaults to the server's current directory and can be set with
`--project-root` or `LINT_AI_PROJECT_ROOT`.
With only `--project-root`, one server can inspect the shared workspace store
and the shared memory store in the dashboard. A project now uses this
layout:

```text
.lint-ai/
  workspace-memory/   # code and documentation, indexed once for the project
  memory/             # session memories, shared by all agents
```

MCP search composes `workspace-memory` with the shared `memory/` store at query
time; the provider travels on each document (`integration`, `author_agent`,
`{provider}-session:{id}` group ids) instead of in the directory layout.
Legacy per-provider stores (`claude-memory/`, `codex-memory/`,
`gemini-cli-memory/`, `agy-memory/`, `muse-memory/`) are migrated into
`memory/` on first run and removed once their migration succeeds.
Legacy `*-mcp-index` directories are recognized for compatibility but are no
longer created.

The server publishes a segmented memory index grouped by `session_id` and routes
each search to the three most locally distinctive candidate segments.
Fixed top-3 routing remains the default. To opt into adaptive routing, pass
`--adaptive-segment-max-n N` or set `ADAPTIVE_SEGMENT_MAX_N=N`, where `N` is
greater than 3. Adaptive routing starts with the same three segments and may
expand up to `N` when the initial routes do not cover enough query evidence.
Use `--single-index` for the non-segmented layout or `--global-index` to query
every segmented shard. These modes are intended primarily for controlled
comparisons instead of the default routed segmented layout.
The server is intentionally localhost-only. `--bind` may select a loopback
address and port, such as `127.0.0.1:8080` or `[::1]:8080`, but non-loopback
addresses are rejected at startup. Authentication remains available for
defense in depth; `--allow-unauthenticated` is intended for the local dashboard.

For a single-tenant deployment, also set `--tenant-id TENANT` (or
`SERVER_TENANT_ID`). Requests whose `user_id` does not match this configured
tenant are rejected; this prevents a bearer token from being used to select
another tenant by changing the request body.

### Run the local dashboard

For local Claude Code and Codex development, build the corresponding provider
features and let the server discover all provider indexes from the project root:

```bash
cargo run --release --bin server --features claude-code,codex -- \
  --project-root /Users/louis/sources/Lint-AI \
  --bind 127.0.0.1:8080 \
  --allow-unauthenticated
```

Then open <http://127.0.0.1:8080/dashboard>. The server discovers indexes under
`/Users/louis/sources/Lint-AI/.lint-ai/`, while the dashboard separates provider
stores into tabs. The HTTP API uses the selected primary index; the dashboard
can inspect the other provider stores without starting additional servers.

The server intentionally speaks plain HTTP because it is restricted to
localhost. Do not forward the port through a public or network-facing proxy.

Mutations are admitted through a single-writer gate. If a refresh is already in
progress, additional mutation requests receive `429` rather than queueing
without bound; searches continue using the last published snapshot.

For per-user authentication, set `JWT_SECRET`. The server accepts HS256 JWTs
with a non-empty `sub` claim and a valid `exp` claim; that subject is treated
as the authenticated `user_id` and must match the request scope. `JWT_SECRET`
takes precedence over the legacy shared `SERVER_TOKEN` mode.

The server exposes `GET /health`, the legacy mutation/search routes, and the
versioned memory routes:

* `GET /v1/memories` lists memories with `user_id`, optional `session_id`,
  `limit`, and cursor parameters.
* `POST /v1/memories` adds memories.
* `GET /v1/memories/:memory_id` retrieves one memory.
* `PATCH /v1/memories/:memory_id` updates one memory.
* `DELETE /v1/memories/:memory_id` deletes one memory.
* `POST /v1/memories/search` searches memories.
* `POST /v1/memories/refresh` publishes pending changes.

The legacy routes are `POST /add`, `POST /add/batch`, `POST /search`,
`POST /delete`, `POST /supersede`, and `POST /expire`; they remain for
backward compatibility.

## Dashboard

Open `http://127.0.0.1:8080/dashboard` for a read-only operational view of
the running IndexStore. The page shows index freshness, revisions, segment
counts, rolling query rate, p50/p95 latency, error and empty-result rates, and
the compiled/provider integration state. It polls every five seconds and keeps
only bounded aggregate telemetry; query text, identifiers, and memory content
are never recorded.

The dashboard page and static assets are public so they can load before an API
token is entered. The data endpoints use the same authentication as the rest
of the server. The provider workspace is organized into provider tabs; each tab
shows session history, the latest live session, tool-call frequency cards, and
filterable tool-call history with bounded argument previews:

* `GET /api/status` returns index state, query summary, and integration cards.
* `GET /api/timeseries` returns five-second IndexStore query aggregates over
  the last ten minutes, shared by HTTP and provider MCP processes.
* `GET /api/integrations` returns provider readiness details.
* `GET /api/sessions` returns recent sessions and their sanitized lifecycle
  event counts.
* `GET /api/events` returns the latest sanitized provider lifecycle events.
* `GET /api/sessions/:session_key/events` filters that event history by the
  one-way session key returned by `/api/sessions`.
* `GET /api/metrics` returns machine-readable query and provider aggregates.
* `GET /metrics` exposes the same core counters in Prometheus text format.

Provider hooks write a bounded history of the last 500 lifecycle events per
provider. Events contain only the provider, event name, category, timestamp,
and a one-way session key. Claude Code and Codex subagent events additionally
include bounded `agent_id`, `agent_type`, and `turn_id` fields when supplied by
the provider. Tool events additionally include the tool name and a small
redacted argument preview. Prompt, tool-response, and final-response events use
the same bounded preview format; raw provider payloads are not copied into the
telemetry ledger. Provider cards report `not_observed` until a
provider hook or MCP process sends explicit lifecycle telemetry. IndexStore
The HTTP server keeps query aggregates in memory for a low-overhead request
path. Provider MCP processes persist their query aggregates to
`.lint-ai/query-telemetry.json`, allowing the dashboard to observe queries
executed by separate MCP processes. This server does not infer agent
connectivity from IndexStore health.

An observed provider is `active` when its latest event arrived within the last
minute and `idle` otherwise. Lifecycle telemetry is local-first: each provider
uses a bounded JSON ledger under `.lint-ai/provider-telemetry`, so an operator
can inspect recent sessions and event ordering without retaining provider
content. Claude Code and Codex token usage is captured from lifecycle payloads
or provider transcripts when available; cost and productivity accounting still
require provider-specific contracts.

Search requests use the latest immutable `MemorySearchService` snapshot, so
multiple searches can run concurrently. A dedicated writer owns mutable
`MemoryService` state and refreshes it before publishing the replacement
snapshot with a short swap. A search that overlaps refresh sees either the
previous complete snapshot or the newly refreshed one, never a partially
updated index. Blocking index work runs on Tokio's blocking pool rather than
the Axum async executor. Reproduce the measured concurrency behavior in
[`Comparison`](comparison.md).

## Performance

The historical 0.1.9 HTTP baseline used a 23,366-record corpus, 100
`POST /search` requests per cell, `top_k: 20`, and the same keyword query on a
local machine. The server was run in release mode with the file-backed single index.
Latency is reported in milliseconds; throughput is completed requests per
second.

| Concurrent requests | p50 | p90 | p99 | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 6.96 ms | 7.65 ms | 28.17 ms | 139.52 req/s |
| 10 | 10.23 ms | 11.87 ms | 12.52 ms | 952.07 req/s |

The first post-refactor v0.2.0 routed-segment run on the same corpus recorded:

| Concurrent requests | p50 | p90 | p99 | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 10.39 ms | 10.90 ms | 12.21 ms | 95.00 req/s |
| 10 | 23.71 ms | 38.02 ms | 45.89 ms | 386.55 req/s |

The 0.1.9 table is retained as a historical baseline; it is not a like-for-like
comparison with the segmented v0.2.0 server.

A fresh uncached v0.2.0 layout comparison was run on 2026-09-17 on a MacBook
Pro (Mac17,9) with an Apple M5 Pro chip (15 cores), 24 GB RAM, macOS 26.6.2,
Rust 1.96.0, Cargo 1.96.0, and uv 0.12.7. It used the same 23,366 records, 100
requests per cell, `top_k: 20`, and query, repeated five times with no warm-up
requests. The table reports medians and distinguishes the server's three index
layouts:

| Layout | C=1 throughput | C=10 throughput | C=10 p50 | C=10 p99 |
|---|---:|---:|---:|---:|
| Single index | 148.42 req/s | 997.62 req/s | 8.87 ms | 18.39 ms |
| Global segmented | 237.82 req/s | 1,306.82 req/s | 6.30 ms | 12.39 ms |
| Routed segment | 237.84 req/s | 1,512.31 req/s | 5.91 ms | 11.97 ms |

The standalone HTTP benchmark does not initialize provider MCP watchers. Query
telemetry is recorded in memory on this request path; provider MCP telemetry
continues to use the persisted project ledger.

At the smaller 5,000-record scale, the same test measured 2.06 ms p50 / 2.38
ms p90 at concurrency 1 and 3.79 ms p50 / 5.21 ms p90 at concurrency 10.
The current read path uses independently published immutable snapshots;
mutation work is serialized by the writer gate and only the final snapshot swap
briefly needs the reader-facing write lock.

These are local service-load measurements, not an internet-facing SLA. They
exclude network distance, TLS termination, and client-side processing. The
full scripts, corpus sizes, commands, and JSON results are in
[`comparison/README.md`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/README.md).

Memory lifecycle fields are optional on each `/add` message. Set
`expires_at_ms` to hide a memory after a Unix-millisecond deadline. Set
`supersedes_id` to mark an older memory as replaced. Lifecycle operations are
scoped by `user_id`:

```json
{"user_id":"user-0","doc_id":"memory-id"}
{"user_id":"user-0","replacement_id":"new-id","old_id":"old-id"}
{"user_id":"user-0"}
```

These are the request bodies for `/delete`, `/supersede`, and `/expire`,
respectively. Delete is idempotent; expire removes all expired memories for
the user. Search omits expired and superseded memories.

`/search` accepts `query`, `user_id`, and `top_k` (capped at 100); its optional
`options` field is retained for client compatibility. `/add` requires a
non-empty `messages` array, and each message must have a `role` of `user` or
`assistant` plus non-empty `content`. Identifiers are scoped and validated by
the server, so callers should use the same `user_id` for adding and searching
that user's memories. An add request accepts at most 1,024 messages; each
message is capped at 1 MiB, and identifiers are capped at 256 bytes.

`/add` is idempotent by `(user_id, request_id)`. The stored fingerprint is a
SHA-256 digest over the canonical `session_id` and messages; message content is
not copied into filter metadata. Reusing a request ID with a different session
or message payload is rejected.

The server token accepts `X-Api-Key`, `Authorization: Bearer <token>`, or the
raw token in `Authorization`. It can also be supplied through
`SERVER_TOKEN`.

If no token is configured, the server refuses to start on any non-loopback
bind address (anything other than `127.0.0.1`/`::1`). Pass
`--allow-unauthenticated` is available for the single-user localhost mode;
non-loopback binds are rejected regardless of authentication settings.

The server limits request bodies to 16 MiB and concurrent in-flight requests to
128. Requests time out after 30 seconds.
Malformed JSON or invalid request fields return `422`; missing or invalid
credentials return `401`; an authenticated subject or configured tenant that
does not match `user_id` returns `403`; unknown routes return `404`; an
oversized body returns `413`; a busy mutation writer returns `429`; timed-out
requests return `408`; and internal search, persistence, or publication
failures return `500`.

## Local contract smoke test

```bash
curl -sS http://127.0.0.1:8080/health

curl -sS -X POST http://127.0.0.1:8080/add \
  -H "X-Api-Key: $SERVER_TOKEN" \
  -H 'Content-Type: application/json' \
  --data '{
    "request_id": "local-run:session-0:chunk-0",
    "messages": [{"role": "user", "content": "I prefer dark mode."}],
    "user_id": "local-run:user-0",
    "session_id": "local-run:session-0"
  }'

curl -sS -X POST http://127.0.0.1:8080/search \
  -H "X-Api-Key: $SERVER_TOKEN" \
  -H 'Content-Type: application/json' \
  --data '{
    "query": "What interface preference does the user have?",
    "user_id": "local-run:user-0",
    "top_k": 100
  }'
```
