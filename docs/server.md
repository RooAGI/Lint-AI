# Lint-AI server

`server` exposes a synchronous Add/Search contract with Lint-AI's `IndexStore`
as the memory backend. It is a standalone HTTP interface for any application;
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
For non-loopback binds, configure `--server-token` (or `SERVER_TOKEN`) unless
you explicitly use `--allow-unauthenticated` on a closed network.

The server exposes `GET /health`, `POST /add`, `POST /search`, `POST /delete`,
`POST /supersede`, and `POST /expire`.

Search requests use a read lock over the latest immutable index snapshot, so
multiple searches can run concurrently. Writes take the exclusive lock and
refresh the snapshot before returning; a search that overlaps a write sees
either the previous complete snapshot or the newly refreshed one, never a
partially updated index. Reproduce the measured concurrency behavior in
[`Comparison`](comparison.md).

## Performance

The recorded HTTP benchmark used a 23,366-record corpus, 100 `POST /search`
requests per cell, `top_k: 20`, and the same keyword query on a local machine.
The server was run in release mode with the file-backed index. Latency is
reported in milliseconds; throughput is completed requests per second.

| Concurrent requests | p50 | p90 | p99 | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 6.96 ms | 7.65 ms | 28.17 ms | 139.52 req/s |
| 10 | 10.23 ms | 11.87 ms | 12.52 ms | 952.07 req/s |

At the smaller 5,000-record scale, the same test measured 2.06 ms p50 / 2.38
ms p90 at concurrency 1 and 3.79 ms p50 / 5.21 ms p90 at concurrency 10.
The read path uses shared access to an immutable snapshot, allowing concurrent
searches; writes briefly take the exclusive lock to refresh that snapshot.

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
that user's memories.

The server token accepts `X-Api-Key`, `Authorization: Bearer <token>`, or
`Authorization: Token <token>`. It can also be supplied through
`SERVER_TOKEN`.

If no token is configured, the server refuses to start on any non-loopback
bind address (anything other than `127.0.0.1`/`::1`). Pass
`--allow-unauthenticated` to override this for closed networks; every request
is then accepted without a token.

The server limits request bodies to 16 MiB, header data to 64 KiB, and active
connections to 128. Stalled socket reads and writes time out after 30 seconds.
Malformed JSON or invalid request fields return `422`; missing or invalid
credentials return `401`; unknown routes return `404`; an oversized body
returns `413`; and a saturated connection limit returns `503`.

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
