# Agent Integrations

Membrick connects to agent clients through lifecycle hooks and a shared
file-based memory index. Each adapter preserves the host client protocol while
using the same capture, indexing, and retrieval model.

## Supported agents

| Agent | Integration guide | Memory store |
|---|---|---|
| Claude Code | [Claude Code](claude-code.md) | `.membrick/memory` |
| Codex | [Codex](codex.md) | `.membrick/memory` |
| Gemini CLI | [Gemini CLI](gemini-cli.md) | `.membrick/memory` |
| Antigravity CLI | [Antigravity CLI](agy.md) | `.membrick/memory` |
| Muse Code | [Muse Code](muse-code.md) | `.membrick/memory` |
| OpenClaw | [OpenClaw](openclaw.md) | `.membrick/memory` |
| Hermes | [Hermes](hermes.md) | `.membrick/memory` |
| RooAGI AgentFlow | [RooAGI AgentFlow](roo-runtime.md) | `.membrick/memory` |

All agents share the single project store at `.membrick/memory/` — a decision
recorded by Claude is visible to Codex, and vice versa. Provider attribution
travels on each document (`integration`, `author_agent`,
`{provider}-session:{id}` group ids), not in the directory layout.

Upgrading from an older version? Legacy per-provider stores
(`.membrick/claude-memory/`, `.membrick/codex-memory/`,
`.membrick/gemini-cli-memory/`, `.membrick/agy-memory/`,
`.membrick/muse-memory/`) are migrated into `.membrick/memory/` automatically on
first run and removed once their migration succeeds. A failed migration leaves
the legacy directory untouched, so no memory is lost.

## How an agent uses memory

1. A lifecycle hook receives the agent event and transcript context.
2. Retrieval hooks query the existing `MemoryService` and return relevant context
   to the agent.
3. Claude Code and Codex capture hooks preserve every parsed transcript
   observation with its speaker and source order, including structured tool
   and image blocks serialized from the transcript. Other adapters capture the
   observations available in their provider payloads.
4. The adapter upserts the document and refreshes the index snapshot.
5. The next prompt can query the newly refreshed snapshot.

See [Server and MCP](server.md) for direct API access.

## Memory lifecycle

All supported adapters retrieve memory during an active turn and capture new
memory at lifecycle boundaries. The exact hook names vary by client, but the
contract is the same: query the current snapshot, capture durable context, then
refresh the snapshot.

### One turn inside a session

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent<br/>(Codex · Claude Code · Gemini CLI · AGY)
    participant H as Lifecycle hook
    participant I as Membrick

    U->>A: Submit prompt
    A->>H: UserPromptSubmit
    H->>I: Query existing memories
    I-->>H: Ranked excerpts
    H-->>A: Inject context in hook response
    A->>H: PreToolUse / PostToolUse (optional)
    H->>I: Query existing memories
    I-->>H: Ranked excerpts
    H-->>A: Inject context for next decision
    A-->>U: Complete response
    A->>H: Stop / AfterAgent
    H->>H: Preserve available session observations
    H->>I: Upsert outcome + refresh snapshot
```

```text
1. User submits a prompt
2. Retrieval hook queries the existing MemoryService
3. Agent may call tools (pre-tool and post-tool hooks can retrieve context)
4. Agent finishes the turn
5. Stop/AfterAgent captures the turn outcome and refreshes MemoryService
6. The next prompt can retrieve that captured outcome
```

`Stop` (or Gemini CLI's `AfterAgent`) is a per-turn boundary and can occur
multiple times in one session. It is different from `SessionEnd`, which runs
when the entire session closes and captures the final session summary:

```text
Turn 1 → Stop       capture outcome
Turn 2 → Stop       capture outcome
Turn 3 → Stop       capture outcome
Session exits → SessionEnd   capture session summary
```

The current turn is not retroactively re-prompted with its newly captured
memory. That memory is injected on a later retrieval event.

### Retrieval events

Adapters query their provider-specific `MemoryService` on events such as:

- `SessionStart`
- `UserPromptSubmit`
- `PreToolUse` and `PostToolUse`
- `SubagentStart`

Selected records are returned as additional context to the client. Queries are
scoped to the active project and provider memory store.

### Retrieval and injection path

```text
Agent hook receives prompt/tool context
        ↓
Adapter builds a provider-scoped query
        ↓
MemoryService ranks matching records with project/user filters
        ↓
Adapter bounds and formats the selected excerpts
        ↓
Hook response returns the context to the host agent
        ↓
Host agent includes it in the next model/tool decision
```

The adapter does not rewrite the original user prompt. It uses each client’s
hook response protocol:

| Provider | Injection field |
|---|---|
| Claude Code | `hookSpecificOutput.additionalContext` |
| Codex | `hookSpecificOutput.additionalContext` |
| Gemini CLI | `hookSpecificOutput.additionalContext` |
| Antigravity CLI | `injectSteps[].ephemeralMessage` |

If the query is empty, the store is absent, or no relevant records survive
filtering, the hook returns no injected context and the agent continues normally.

### Turning the integration on and off

Memory behavior and session recording are independent controls exposed through
the provider’s Membrick MCP server:

| Control | Effect |
|---|---|
| `enable_membrick` | Turns retrieval and capture on; recording is enabled by default. |
| `disable_membrick` | Turns Membrick retrieval/injection and automatic capture off; existing recordings and memories remain. |
| `record_session` with `start` | Records hook events in capture-only mode without injecting memory. |
| `record_session` with `stop` | Stops recording without disabling memory integration. |
| `membrick_status` | Reports independent `Membrick:ON/OFF` and `Record:ON/OFF` state. |

The controls are project- and provider-scoped. Disabling the integration does
not uninstall hooks, delete the index, or delete session archives; it only
prevents Membrick from participating in subsequent turns.

### Hook events recorded for replay (no index update)

When session recording is enabled, the adapters record every supported hook
invocation for replay and diagnostics. These records are telemetry; they do
not update the searchable `MemoryService`:

| Provider | Recorded hook events |
|---|---|
| Claude Code | `SessionStart`, `UserPromptSubmit`, `UserPromptExpansion`, `PreToolUse`, `PostToolUse`, `PreCompact`, `Stop`, `SessionEnd`, `SubagentStart`, `SubagentStop` |
| Codex | `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `UserPromptExpansion`, `PreCompact`, `PostCompact`, `Stop`, `SessionEnd`, `SubagentStart`, `SubagentStop` |
| Gemini CLI | `SessionStart`, `BeforeAgent`, `AfterAgent`, `BeforeModel`, `BeforeToolSelection`, `BeforeTool`, `AfterTool`, `PreCompress`, `SessionEnd` |
| Antigravity CLI | `PreToolUse`, `PostToolUse`, `PreInvocation`, `PostInvocation`, `Stop` |

### Events that update searchable memory

| Provider | Capture-attempt events | Searchable store |
|---|---|---|
| Claude Code | `PreCompact`, `Stop`, `SessionEnd`, `SubagentStop` | `.membrick/memory` |
| Codex | `PreCompact`, `PostCompact`, `Stop`, `SessionEnd`, `SubagentStop` | `.membrick/memory` |
| Gemini CLI | `AfterAgent`, `PreCompress`, `SessionEnd` | `.membrick/memory` |
| Antigravity CLI | `Stop` (transcript capture) | `.membrick/memory` |

At a capture event, Claude Code and Codex read only complete transcript lines
appended since the previous durable capture. They preserve every parsed message
and structured block in source order, after credential redaction. Each message
becomes an observation document; the captured slice also becomes a turn
document, and a stable session-context document links the observations. The
cursor advances only after the batch is accepted by the resident `MemoryService`.
Other adapters capture the session data exposed in their hook payloads. Claude
and Codex send each batch as one upsert. The hook returns once the batch is
journaled, and the documents become searchable on the next background
publication. They are not injected back into the conversation that just ended.
They can be returned on a later retrieval event, typically the next prompt or
next session. If the transcript is missing or has no parsed messages, the event
completes without adding a document.

### Two kinds of persistence

- `MemoryService` contains searchable memory documents.
- Session recording archives raw provider hook events for replay and debugging.

Recording a hook event does not by itself make a document searchable; it must
also pass through the provider's capture and index-refresh path.

The HTTP/MCP API follows the same guarantee: a successful `memory_add` performs
the upsert and refresh before returning, so a subsequent `memory_search` can
query the new memory immediately.

## Provider-specific benchmarks

The [benchmark overview](benchmark.md),
[cross-client integration benchmark](benchmark-integration.md), and provider
benchmark pages document standalone retrieval, corpus-scale, and integration
measurements for these adapters.

Session recording is documented separately in
[Session recording](session-recording-design.md). It records lifecycle events
for replay and inspection and is intentionally independent from searchable
memory and context injection.

The provider MCP tools and their request semantics are documented in the
[MCP interface guide](mcp.md). For onboarding, see the Get started entries for
the [HTTP server](server.md) and [MCP interface](mcp.md).

## Durable Claude and Codex transcript capture

Claude and Codex hooks submit a durable capture request to the existing project
resident and return after acceptance. Acceptance means queued, not fully
captured or searchable. The resident's MemoryService capture task drains the
transcript independently of the hook process and its timeout. It reuses the
provider parsers, source documents, stable document IDs and single writer.

Pending requests are saved under `.membrick/runtime/transcript-captures/`.
Repeated hooks for the same provider and session extend the pending target
instead of starting competing replay loops. The resident processes one batch
per pending session per pass, and saves the existing provider cursor only after
MemoryService durably accepts that batch. A failed write leaves the cursor
unchanged and retries with backoff. A crash after a write but before the cursor
update safely repeats the same document IDs. Startup resumes saved jobs without
another hook; graceful shutdown finishes the in-flight batch and retains the
remaining work. Partial JSONL lines wait for completion. Requests use the same
Unix socket, loopback HTTP, then filesystem queue fallback as other resident
operations.

The MCP `info` response and dashboard Overview show captured bytes, target bytes,
remaining bytes, last progress time and the last error. `GET /api/captures`
returns the same per-session progress. A pending capture can be accepted even
while indexing is busy; it becomes searchable after the normal background
publication.
