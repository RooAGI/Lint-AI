# Lint-AI server

`server` exposes an HTTP Add/Search contract with Lint-AI's `IndexStore`
as the memory backend. It is a standalone HTTP interface for any application;
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
  --bind 0.0.0.0:8080 \
  --index /var/lib/lint-ai/memory-index \
  --server-token "$SERVER_TOKEN"
```

`--bind` defaults to `127.0.0.1:8080`. If `--index` is omitted, the server
uses an in-memory store; provide `--index` when data must survive a restart.
The server publishes a segmented memory index grouped by `session_id` and routes
each search to the three most locally distinctive candidate segments.
Fixed top-3 routing remains the default. To opt into adaptive routing, pass
`--adaptive-segment-max-n N` or set `ADAPTIVE_SEGMENT_MAX_N=N`, where `N` is
greater than 3. Adaptive routing starts with the same three segments and may
expand up to `N` when the initial routes do not cover enough query evidence.
Use `--single-index` for the non-segmented layout or `--global-index` to query
every segmented shard. These modes are intended primarily for controlled
comparisons instead of the default routed segmented layout.
For non-loopback binds, configure `--server-token` (or `SERVER_TOKEN`) unless
you explicitly use `--allow-unauthenticated` on a closed network.

For a single-tenant deployment, also set `--tenant-id TENANT` (or
`SERVER_TENANT_ID`). Requests whose `user_id` does not match this configured
tenant are rejected; this prevents a bearer token from being used to select
another tenant by changing the request body.

The server intentionally speaks plain HTTP. When exposed beyond localhost,
place it behind a TLS-terminating reverse proxy (or a private encrypted
network); do not send bearer tokens over an unencrypted network.

Mutations are admitted through a single-writer gate. If a refresh is already in
progress, additional mutation requests receive `429` rather than queueing
without bound; searches continue using the last published snapshot.

For per-user authentication, set `JWT_SECRET`. The server accepts HS256 JWTs
with a non-empty `sub` claim and a valid `exp` claim; that subject is treated
as the authenticated `user_id` and must match the request scope. `JWT_SECRET`
takes precedence over the legacy shared `SERVER_TOKEN` mode.

The server exposes `GET /health`, `POST /add`, `POST /add/batch`,
`POST /search`, `POST /delete`, `POST /supersede`, and `POST /expire`.

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

A fresh uncached v0.2.0 layout comparison was run on 2026-09-16 on a MacBook
Pro (Mac17,9) with an Apple M5 Pro chip (15 cores), 24 GB RAM, macOS 26.6.2,
Rust 1.96.0, and Cargo 1.96.0. It used the same 23,366 records, 100 requests per
cell, `top_k: 20`, and query. The results distinguish the server's three index
layouts:

| Layout | C=1 throughput | C=10 throughput | C=10 p50 | C=10 p99 |
|---|---:|---:|---:|---:|
| Single index | 168.77 req/s | 1,085.93 req/s | 8.16 ms | 15.96 ms |
| Global segmented | 274.45 req/s | 1,662.27 req/s | 5.65 ms | 10.07 ms |
| Routed segment | 270.01 req/s | 1,383.66 req/s | 6.32 ms | 11.36 ms |

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
`--allow-unauthenticated` to override this for closed networks; every request
is then accepted without a token.

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
