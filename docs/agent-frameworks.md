# Other agent frameworks

Lint-AI can serve as the memory layer for an agent framework that is not one
of the built-in Claude Code, Codex, Gemini CLI, or Antigravity adapters. The
framework owns the agent loop; Lint-AI owns durable memory, indexing, search,
and lifecycle operations.

## Choose an integration surface

| Surface | Use it when | Framework responsibility |
|---|---|---|
| HTTP server | The framework can make HTTP requests or you want a service boundary. | Call `/add` after a turn and `/search` before the next model decision. |
| MCP server | The framework supports MCP tools and you want the model to request memory explicitly. | Connect Lint-AI as an MCP server and expose its tools to the agent. |
| Rust library | The framework is Rust-based and can embed the index directly. | Manage an `IndexStore` and call `upsert`/`query` in the agent loop. |

The HTTP and MCP paths work with any framework and do not require Lint-AI's
provider-specific lifecycle hooks.

## Recommended turn pattern

At the start of a turn, search using a stable user and session identity. Add
the returned excerpts to the framework's context or memory field before the
model is called. After the turn completes, persist the useful exchange or
structured outcome with the same identity.

```text
prompt arrives
    ↓
Lint-AI /search (user_id + query)
    ↓
framework adds bounded results to model context
    ↓
agent runs tools and produces a response
    ↓
framework extracts durable facts/outcome
    ↓
Lint-AI /add (user_id + session_id + messages)
    ↓
index refresh completes; the memory is available to later searches
```

The current turn is not rewritten after `/add`; the newly captured memory is
retrieved on a later turn. Keep `user_id` stable for one user's memory and use
`session_id` to group records from a conversation.

## HTTP example

Start a local server as described in the [HTTP server guide](server.md), then
call it from the framework's memory middleware:

```python
import requests

BASE = "http://127.0.0.1:8080"

def retrieve(user_id, prompt):
    response = requests.post(f"{BASE}/search", json={
        "query": prompt,
        "user_id": user_id,
        "top_k": 20,
    })
    response.raise_for_status()
    return response.json()["data"]

def record(user_id, session_id, request_id, role, content):
    response = requests.post(f"{BASE}/add", json={
        "request_id": request_id,
        "user_id": user_id,
        "session_id": session_id,
        "messages": [{"role": role, "content": content}],
    })
    response.raise_for_status()
```

For production, send one of the supported authentication headers and use a
persistent `--index` directory. See the server page for lifecycle fields,
limits, concurrency behavior, and measured performance.

## MCP example

When the framework supports MCP, register Lint-AI as an MCP server and make
the `search` tool available to the agent. Use `record_session` only when you
want capture-only event recording; it is independent from searchable memory.
The [MCP interface guide](mcp.md) documents the available tools and state
controls.

## Framework adapter checklist

- Assign a stable `user_id`; do not use a process-wide default for all users.
- Use a unique `request_id` for each add operation so retries are safe.
- Bound injected context by tokens or characters before passing it to the model.
- Add only durable facts, decisions, and outcomes, not every transient tool log.
- Treat retrieval failure as non-fatal so the agent can continue without memory.
- Keep memory controls and deletion flows visible to the application owner.
