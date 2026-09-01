# Lint-AI server

`server` exposes a synchronous Add/Search contract with Lint-AI's `IndexStore`
as the memory backend. The API remains compatible with the Agent Memory
Leaderboard.

## Run locally

Use an explicit index directory for a persistent evaluation instance:

```bash
cargo run --release --bin server -- \
  --bind 0.0.0.0:8080 \
  --index /var/lib/lint-ai/memory-index \
  --server-token "$SERVER_TOKEN"
```

The server exposes `GET /health`, `POST /add`, `POST /search`, `POST /delete`,
`POST /supersede`, and `POST /expire`.

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

The server token accepts `X-Api-Key`, `Authorization: Bearer <token>`, or
`Authorization: Token <token>`. It can also be supplied through
`SERVER_TOKEN`.

If no token is configured, the server refuses to start on any non-loopback
bind address (anything other than `127.0.0.1`/`::1`). Pass
`--allow-unauthenticated` to override this for closed networks; every request
is then accepted without a token.

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

## AML submission flow

1. Deploy the server at a public HTTPS address.
2. Keep the three endpoints stable for the evaluation period.
3. Submit the endpoint, API-key scheme, fixed product version, capacity,
   timeout, and rate-limit details through the AML evaluation request.
4. Bind the returned AML key to the deployed version and run the AML smoke
   test.
5. Start the formal Full evaluation only after the smoke test passes.

AML supplies benchmark memories and questions. Search returns evidence only;
it must not generate the final answer. Evaluation data must be isolated by the
exact `user_id` supplied in Add and Search and deleted according to AML's
data-retention requirements.
