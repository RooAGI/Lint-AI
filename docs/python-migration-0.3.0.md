# Python binding migration guide: 0.2.x → 0.3.0

`lint-ai` 0.3.0 cleans up the Python binding around the one supported
surface: `MemoryService`. Most existing `Memory` calls keep working, but
there are two intentional breaks. This guide covers what changed, what
stayed compatible, and what the new knobs actually do.

## Breaking changes

### 1. `RemoteMemory` is gone

```python
# 0.2.x
remote = lint_ai.RemoteMemory("http://127.0.0.1:8080", api_key="token")

# 0.3.0 — same behavior, one class
remote = lint_ai.Memory(base_url="http://127.0.0.1:8080", api_key="token")
```

**Pitfall:** `base_url` must be passed by keyword. The first positional
argument of `Memory(...)` is `path`, so
`lint_ai.Memory("http://127.0.0.1:8080")` does **not** connect to a server —
it silently creates a *local* index in a directory literally named
`http://127.0.0.1:8080`. This fails silently, not loudly; always use the
keyword form.

### 2. Python `IndexStore` binding removed

The `lint_ai.IndexStore` class no longer exists. Python applications must
go through `lint_ai.Memory`, which is the only supported external API.

This is a **Python-side** change only. The Rust `IndexStore` remains public
for Rust integrators (`use lint_ai::{IndexStore, PipelineOptions, ...}`).
If you were using `IndexStore` from Python, migrate to `Memory` or stay on
the 0.2.x line.

## What stayed compatible

- `Memory(path=...)` / `Memory()` — same as before; new options are
  keyword-only additions at the end.
- `search(query, user_id, top_k)` — the original three positional arguments
  still work in the same order. `top_k` now defaults to `10`, so
  `search(query, user_id)` is valid.
- `add`, `get`, `list`, `update`, `delete`, `refresh` — argument order
  unchanged, including `delete(user_id, memory_id)`.
- `lint_ai.version()` returns the package version (`"0.3.0"`).

## New in 0.3.0

### Constructor options (local backend only)

```python
memory = lint_ai.Memory(
    path="./memory-index",
    ner_provider="heuristic",  # 'heuristic' (default, Python-free) | 'spacy'
    spacy_model=None,       # explicit override; default follows detected language
)
```

**Language is auto-detected by the engine.** There is no language knob:
the engine detects the content language per text (script statistics plus
language signals) and selects per-language defaults — notably the spaCy
model — automatically. An explicit `spacy_model` always wins. (An earlier
draft of this binding exposed a `language` option; it was removed when the
engine moved to auto-detection.)

**Remote backends ignore these knobs.** The options are validated
client-side (bad values still raise `ValueError`), but the server owns its
configuration — there is no way to configure a remote server from the
Python client.

### Session / scope / filters on search

```python
memory.search(
    "what did we decide about the API?",
    "user-1",
    top_k=10,
    session_id="session-1",   # enables follow-up resolution + rerank
    scope="team-a",           # conversation-state scope (not a doc filter)
    filters={"request_id": "req-42"},  # document filters, see below
)
```

- `session_id` is a **conversation-state key**, not a document-session
  filter: it loads bounded prior state so follow-up phrasing ("what about
  the Python one?") resolves against the conversation. It never becomes an
  implicit ownership filter — `user_id` still scopes document visibility.
- `scope` lets the conversation state be shared under a different key than
  the document-owning `user_id`. Ownership is still enforced on `user_id`.
- `filters` are forwarded to the engine's document filter path. Their exact
  contract is pinned by the test suite (`tests/python/test_memory.py`,
  `TestFilterContract`):
  - Filters are ANDed, but a filter whose key has no index postings — or
    whose key exists but value has no postings — is **silently dropped**,
    not exclusionary. `{"request_id": "no-such"}` returns all owned docs.
  - Ownership (`memory_user_id`) is always enforced and cannot be
    overridden by caller filters (a caller-supplied `memory_user_id` is
    stripped; a literal `"user_id"` key is unknown and dropped).
  - **Structured-fact bypass:** the structured-fact retrieval arm is built
    with `filters=None` — it gates only on ownership, supersession, and
    expiry. A fact question (e.g. "Which city have both Gina and Jon
    visited?") can therefore return documents the caller's filters would
    have excluded on the lexical path, ranked at the top (score > 1000).
    This is current engine behavior, not a binding bug; changing it is a
    retrieval change that needs an explicit decision.
  - In short: filters are best-effort narrowing, not a security boundary.
    Ownership and expiry are always enforced regardless of filters.

### `add_batch`

```python
responses = memory.add_batch([
    {"request_id": "r1", "user_id": "u", "session_id": "s",
     "messages": [{"role": "user", "content": "..."}]},
    # ... up to 128 entries
])
```

At most 128 entries per call (larger batches raise `ValueError`; split
them client-side). Local batching applies entries sequentially with a
single refresh at the end; there is no rollback — if entry N fails,
entries before N are already stored. Retrying with the same `request_id`s
is idempotent for the entries that succeeded. Against a remote server the
client falls back to sequential `/v1` adds (the server exposes no batch
endpoint) and stops at the first error.

## Thread safety

`Memory` releases the GIL during engine/server calls, but the instance is
**not thread-shareable** (`unsendable` in pyo3 terms): do not use one
`Memory` from multiple Python threads. Create one instance per thread, or
serialize access with a lock.
