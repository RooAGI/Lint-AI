# Integration glossary

This glossary defines the provider-neutral terms used by the Lint-AI
integrations. Provider documentation may use different names for equivalent
lifecycle events, but these meanings remain the same.

## Core concepts

**Memory** is durable, project-scoped information retained for use by later
agent work. In Lint-AI, memory is an indexed source document derived from a
session, note, decision, trace, or other supported corpus item. Memory is not
the same as a provider's built-in memory and is not automatically equivalent
to a complete transcript.

**Context** is information supplied to an agent for its current turn or task.
Retrieved memory is one possible source of context. Context is temporary from
Lint-AI's perspective unless it is explicitly captured as memory.

**Session** is one provider interaction identified by a provider session ID.
It may contain multiple turns, messages, tool calls, lifecycle events, and
subagent activity. A replay is a new session that references an earlier
session; it is not a continuation of the original archive.

**Message** is a conversational input or output within a session, normally
associated with a role such as `user` or `assistant`. A message is content,
not the whole session and not necessarily a complete lifecycle event.

**Event** is an observed provider lifecycle or activity record, such as a
prompt submission, tool call, tool result, compaction, or stop notification.
Events are the units captured by session recording. A recorded event does not
become durable memory automatically.

## Retrieval and capture

**Recall** is the act of finding relevant indexed memory or source material
for a query. It is a retrieval operation, not a storage operation. The CLI
`--recall` mode and the integration hook/MCP search paths perform recall.

**Retrieval** is the broader process of querying, ranking, and selecting
relevant records. Recall names the user-visible result; retrieval describes
the implementation flow that produces it.

**Injection** is returning selected retrieved material to the provider as
additional context for the current turn. Injection is bounded and does not
write the returned text back into memory by itself.

**Capture** is observing provider input or lifecycle data, normalizing it,
redacting sensitive content, and persisting a bounded record. Capture may
produce a session archive or a durable memory record depending on the path.

**Session recording** is capture into an append-only, provider-specific event
archive. Recording is independent from memory retrieval and injection; it can
run in `capture-only` mode.

**Memory promotion** is the explicit conversion of selected recorded session
material into durable, searchable memory. It is distinct from recording so
that every event does not become long-term memory.

## Integration terms

**Provider** is the agent client being integrated, such as Claude Code, Codex,
Gemini CLI, or AGY. Provider-specific adapters translate that client's hook
payloads and transcript format into the shared Lint-AI model.

**Lifecycle hook** is a provider callback at a session or activity boundary.
Hooks can trigger recall, injection, capture, or status handling. A hook is
not itself a memory record.

**MCP** is the explicit tool interface exposed by Lint-AI to the provider.
MCP tools let the client or model request operations such as search, status,
and session recording control. MCP does not capture conversations
automatically.

**Memory store** is the project-local persistent index containing searchable
memory records. Provider stores remain isolated because their payloads and
provenance differ, even though they share the same indexing implementation.

**Session archive** is the provider-specific append-only record of captured
events. It is an audit or evaluation artifact, not automatically a memory
store.

**Native memory** is memory maintained by the provider itself. It is separate
from Lint-AI memory and may be enabled, disabled, or compared independently.

The usual flow is:

```text
provider event -> hook -> capture -> session archive
                         |
                         +-> optional memory promotion -> memory store

current prompt -> recall/retrieval -> bounded context injection -> provider
```
