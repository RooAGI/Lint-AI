# Claude Code Hooks Design

## Status

This document defines the Claude Code hook integration. The current
implementation supports ten hooks: the original six (`SessionStart`,
`UserPromptSubmit`, `UserPromptExpansion`, `PreCompact`, `Stop`, `SessionEnd`)
plus four added in the second release (`PreToolUse`, `PostToolUse`,
`SubagentStart`, `SubagentStop`). It also provides persistent segmented session
memory, complete transcript text capture, context injection, and settings
installation.

Structured semantic summarization, configurable budgets, stronger secret
classification, cross-process MCP memory sharing, `PostToolUseFailure` capture,
and a subagent parent/child segment split remain future work.

The design deliberately keeps Claude-specific protocol types at the integration
boundary. Membrick's public ingestion and query APIs remain `SourceDocument`,
`MemoryService`, `MemoryIndexSnapshot`, and `SearchResult`.

The entire adapter is compiled only with the non-default `claude-code` Cargo
feature. Core library and Python builds do not include Claude protocol or CLI
surfaces unless they explicitly enable that feature.

## Goals

The hook integration should let Membrick act as a persistent memory layer for
Claude Code:

- retrieve relevant project and session memory before Claude starts work
- preserve useful state before context compaction or session termination
- route retrieval to a small number of relevant memory segments
- reduce repeated repository exploration and repeated user explanations
- bound hook latency and the number of tokens injected into Claude's context

The integration stores every parsed user and assistant transcript observation
in source order as session memory, including tool calls, tool results, empty
messages, and image blocks. Text blocks and serialized structured blocks are
kept in full. Credentials are redacted before persistence.

Each capture hook stores only the transcript written since the previous capture
of the same session. The byte position of the last capture is kept per session
under `.membrick/claude-state/capture-cursors/`, and it advances only after the
capture is stored, so a failed capture is retried by the next hook. A trailing
line that is still being written is left for the next capture.

Each parsed transcript observation in that slice is stored as its own stable
`SourceDocument` with its speaker, order, provider message ID when available,
timestamp when available, and complete text or serialized blocks. The slice's
text is also stored once as a turn document of the hook's type (`outcome`,
`checkpoint`, or `session-summary`) with project semantic scope, so its claims
are compared with other sessions. Transcript discovery uses the hook path or
project `.membrick/settings.toml`; it does not read environment variables.

One `session-context` document per session holds a short header and links the
observations of the latest capture. Every capture is a single batch sent to
`POST /provider-memory/mcp/documents/upsert`. The resident owner journals the
batch and replies once it is durable; the background publisher makes it
searchable, usually within the refresh interval. The server does not rebuild
the session from the store, so a capture never waits for a full index refresh.
If the transcript becomes shorter than the stored position, the capture starts
from the beginning of the file and upserts the observations again. Observations
from the old transcript that no longer exist remain as history. Session
documents participate in normal hook retrieval. Hooks
inject only query-matched excerpts, and the shared policy excludes conflicted
results. Explicit search remains available for inspecting conflicts and full
source documents.

## Ownership Boundary

Claude Code owns hook names, hook payload JSON, matcher behavior, and hook
response JSON. The Claude adapter translates those values into existing Membrick
operations.

```text
Claude hook JSON
        |
        v
Claude protocol and adapter
        |
        +---- retrieval ----> MemoryService::query(...)
        |                           |
        |                           v
        |                    Vec<SearchResult>
        |                           |
        |                           v
        |                  Claude additionalContext
        |
        +---- capture ------> ClaudeCodeDocument
                                    |
                                    v
                              SourceDocument
                                    |
                                    v
                            MemoryService::upsert(...)
```

The Claude integration does not construct `MemoryIndex` or
`SegmentedMemoryIndex` directly. `MemoryService` owns the published
`MemoryIndexSnapshot`, which can be either a single index or a segmented index.

## Current Directory Structure

```text
src/integrations/claude_code/
  mod.rs
  document.rs
  hooks/
    mod.rs
    protocol.rs
```

- `mod.rs` contains MCP serving and configuration installation.
- `document.rs` defines `ClaudeCodeDocument` and converts it to
  `SourceDocument`.
- `hooks/protocol.rs` contains only Claude hook input and output schemas.
- `hooks/mod.rs` dispatches hooks, queries `MemoryService`, extracts full
  transcript content, and captures durable memory.

MCP and hook behavior can be split into smaller files later when those modules
grow independently. That refactor does not change the API boundary described
here.

## ClaudeCodeDocument

Only hook events selected for durable capture become a `ClaudeCodeDocument`.
Retrieval-only hooks use their Claude protocol payload directly.

```rust
pub struct ClaudeCodeDocument {
    pub event_id: String,
    pub session_id: String,
    pub document_type: ClaudeCodeDocumentType,
    pub content: String,
    pub cwd: PathBuf,
    pub timestamp: Option<String>,
    pub command_name: Option<String>,
    pub command_args: Option<String>,
    pub affected_paths: Vec<String>,
    pub branch: Option<String>,
    pub revision: Option<String>,
}

impl ClaudeCodeDocument {
    pub fn into_source_document(self) -> anyhow::Result<SourceDocument>;
}
```

The object is Claude-specific, but its output is the existing generic ingestion
type. A future Codex integration can define `CodexDocument` with its own mapping
to `SourceDocument`.

## SourceDocument Mapping

`ClaudeCodeDocument::into_source_document()` uses the following mapping:

| `SourceDocument` field | Claude mapping |
| --- | --- |
| `doc_id` | deterministic event or checkpoint ID |
| `source` | `claude-code://<project>/<session>/<type>` |
| `content` | filtered checkpoint, outcome, or session summary |
| `concept` | document type, such as `checkpoint` or `outcome` |
| `group_id` | stable project and session segment ID |
| `headings` | structured capture sections, when present |
| `links` | affected repository paths |
| `timestamp` | event timestamp |
| `author_agent` | `claude-code` |
| `filters` | project, session, type, and command metadata |

The initial segment ID is:

```text
claude-session:<project-id>:<session-id>
```

`project-id` must be stable for the repository and must not expose the complete
absolute path. Session segmentation gives the router a natural local-memory
boundary. Profiles from every session remain available to the router, which
allows relevant older sessions to be selected.

## MemoryService Configuration

Claude interaction memory uses a dedicated persistent store:

```text
<project>/.membrick/memory/
  lexical/
  semantic/
  metadata.json
```

The adapter opens it with `MemoryService::at_path()` and segmented layout:

```rust
let options = PipelineOptions {
    memory_index_layout: MemoryIndexLayout::Segmented {
        query_top_n: 3,
        routing_strategy: SegmentRoutingStrategy::LocalDistinctiveness,
    },
    ..PipelineOptions::default()
};

let store = MemoryService::at_path(memory_path, options)?;
```

Capture uses the current mutable API:

```rust
let source_document = claude_document.into_source_document()?;
store.upsert(source_document);
store.refresh()?;
```

Retrieval uses the current query API:

```rust
let results = store.query(&retrieval_query, top_k)?;
```

The same store can be inspected without reading internal persistence files:

```text
membrick inspect-index <project>/.membrick/memory
membrick inspect-index <project>/.membrick/memory --view source-documents
membrick inspect-index <project>/.membrick/memory --view records
membrick inspect-index <project>/.membrick/memory --view segments
```

With segmented layout, `MemoryService::search_with_filters()` routes over the configured top-N
segments. Its global index scores documents from the selected segments and is
also the persistence surface. The current router returns no results when it
finds no segment with useful signal; automatic all-segment fallback is not part
of the current `MemoryService` API.

The Claude memory store is separate from the existing workspace MCP index in
the first release. Command hooks are short-lived processes, while the MCP server
is long-lived; sharing one mutable on-disk store across those processes would
require explicit locking and snapshot reload semantics that `MemoryService` does
not currently provide.

## Supported Hooks

### SessionStart

Purpose: restore useful memory when a session starts or resumes.

Operation:

1. Read `session_id`, `cwd`, and session source from the Claude payload.
2. Resolve the project memory store.
3. Build a retrieval query for recent decisions, unresolved work, and failures.
4. Call `MemoryService::search_with_filters()`.
5. Format results within the session-start context budget.
6. Return the text as Claude `additionalContext`.

This hook does not create a `SourceDocument`.

### UserPromptSubmit

Purpose: retrieve memory relevant to a normal user prompt before Claude handles
it.

Operation:

1. Preserve the submitted prompt unchanged.
2. Use the prompt as the base retrieval query.
3. Apply Membrick's existing query analysis and expansion through the normal
   `MemoryIndex` query path.
4. Route through the segmented `MemoryService` snapshot.
5. Drop low-confidence, duplicate, or oversized results.
6. Return bounded `additionalContext` alongside the original prompt.

The hook does not rewrite the user's prompt. Prompt capture is disabled in the
initial release.

### UserPromptExpansion

Purpose: retrieve command-specific memory when a user-entered slash command,
skill, or MCP prompt expands before reaching Claude.

Claude provides `expansion_type`, `command_name`, `command_args`,
`command_source`, and the original `prompt`. The retrieval query combines the
command name, arguments, and original prompt:

```text
<command_name> <command_args> <original_prompt>
```

`command_name` is a strong routing signal and should be included in captured
memory content or concepts so the current segment profiles can index it. It is
also retained in `filters` for provenance. The current segmented query API does
not apply filters during routing, so filter-based preference is a possible
follow-up rather than an initial dependency. The expanded skill body must not
be stored or repeatedly indexed because common skill instructions would
dominate segment profiles.

This hook returns `additionalContext` and does not create a `SourceDocument` in
the initial release.

### PreCompact

Purpose: preserve resumable state before Claude compacts its context.

Operation:

1. Read the complete transcript identified by the hook.
2. Preserve each parsed message's speaker and full text in source order.
3. Redact credentials before persistence.
4. Create a deterministic `ClaudeCodeDocument` of type `Checkpoint`.
5. Convert it to `SourceDocument` and upsert it through `MemoryService`.

Repeated compaction for the same session uses deterministic checkpoint IDs or
content hashes so retries do not create duplicate memory.

### Stop

Purpose: capture the durable outcome after Claude finishes a response.

The adapter stores all parsed transcript text observations, including earlier
decisions, corrections, tool text, and results, in source order. The complete
snapshot becomes a `ClaudeCodeDocument` of type `Outcome` and is upserted into
the current session segment. The stable per-session document identity replaces
the prior snapshot on each capture, avoiding a duplicate copy for every turn.

The implementation must respect Claude's stop-hook recursion indicator and must
not cause another stop cycle.

### SessionEnd

Purpose: record final resumable session state and termination metadata.

The adapter creates a compact `SessionSummary` document containing the latest
known outcome, unresolved work, and affected paths. It does
not inject context. Session end does not make the segment immutable; resumed
sessions may update it.

## Retrieval Provenance

Each injected memory item includes its source, document type, capture
timestamp, memory status, and relation evidence. Hooks do not run `git` or read
the repository's `.git` directory, so captures carry no branch or revision
metadata. Staleness comes from the capture timestamp and Membrick's semantic
status (current, superseded, or conflicted), not from repository ancestry.

Retrieval emits at most one document per Claude session. When a session has
multiple durable captures, the adapter prefers `session-summary`, then
`outcome`, then `checkpoint`; the newest record breaks ties within a type.
Session ranking still comes from `MemoryService::search_with_filters`.

The adapter does not inject complete stored records. It scores content lines
using the query and the index's matched terms, keeps up to three matching lines,
and limits each excerpt to 800 UTF-8 bytes. Records with no matching line use a
small bounded leading excerpt.

### SubagentStart

Purpose: retrieve memory relevant to a delegated subagent's task before it
begins.

Operation:

1. Use the subagent's prompt as the retrieval query.
2. Route through the segmented store using the parent session's context.
3. Return bounded `additionalContext`.

The subagent runs in the same session segment as its parent. A parent/child
segment split is deferred pending evidence that it improves retrieval precision.

This hook does not create a `SourceDocument`.

### SubagentStop

Purpose: capture the durable outcome after a subagent finishes.

The adapter captures the subagent's completed work as an `Outcome` document in
the current session segment. This follows the same pattern as the `Stop` hook.
The subagent does not check the `stop_hook_active` recursion guard because
`SubagentStop` is only delivered to the parent hook process, not re-entered.

### PostToolUse

Purpose: retrieve memory relevant to a tool result, so Claude has past context
about this tool or related paths before acting on the output.

Operation:

1. Build a retrieval query from `tool_name`, `tool_input`, `tool_response`,
   `turn_id`, `agent_id`, and `agent_type` fields.
2. Truncate large `tool_input` and `tool_response` values to `MAX_EXCERPT_BYTES`
   before forming the query string.
3. Route through the segmented store.
4. Return bounded `additionalContext`.

This hook performs retrieval only. Capture on tool events is deferred because
raw tool output is high-volume and requires stronger filtering before
committing to the memory store.

### PreToolUse

Purpose: retrieve memory relevant to a tool call before Claude executes it.

The adapter builds a query from the available tool name, input, turn, agent,
and response metadata and returns bounded `additionalContext`. Unlike
`PostToolUse`, this hook does not require a minimum query-term threshold because
the pre-tool event is the earliest opportunity to provide context for the
planned action.

This hook performs retrieval only. It does not approve, deny, or modify the
tool call; tool policy remains Claude Code's responsibility.

## Hooks Deferred From The First Release

- `PostToolUseFailure`: useful after failure deduplication is defined. The
  failure case is currently not distinguishable without additional filtering.
- `SubagentStart`/`SubagentStop` parent/child segment split: subagents
  currently write into the parent session segment. A dedicated child segment
  and cross-segment retrieval policy is deferred pending benchmark evidence.
- `PermissionRequest`: a policy hook rather than a memory hook.
- notification and configuration hooks: do not add durable memory value yet.

## Hook Response

Retrieval hooks return structured Claude hook output:

```json
{
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": "Relevant Membrick memory..."
  }
}
```

`UserPromptExpansion` uses the same shape with its own hook event name. Capture
hooks normally emit no context after a successful write.

The hook executable writes protocol JSON only to stdout. Diagnostics and errors
go to stderr so they cannot corrupt Claude's hook response.

## Context Selection

Retrieved memories are formatted under a configured byte or token budget. The
formatter should:

- include source, timestamp, and affected paths when available
- preserve the highest-scoring distinct memories
- avoid repeating the same content from checkpoints and outcomes
- exclude results below a relevance threshold
- return no context when routing has no useful signal

Initial budgets and thresholds must be benchmarked rather than treated as API
constants.

## Failure Behavior

Retrieval hooks fail open by default. If the store is unavailable or a query
fails, Claude receives no added context and continues processing the prompt.

Capture failures are logged to stderr and return success unless strict capture
mode is explicitly enabled. A memory-layer failure must not prevent a user from
ending or compacting a Claude session.

## Security And Retention

Before creating `SourceDocument`, capture redacts credentials, tokens, and
private keys. It retains the transcript's tool input, output, and image blocks
so session claims can be checked against the recorded evidence.

Retention and deletion operate through stable `doc_id` values and
`MemoryService::remove()`. A later retention policy can remove old session outcomes
without changing the Claude protocol adapter.

## Installation Shape

The installer will continue to preserve existing user configuration. It will:

1. Install or update the `membrick` MCP entry.
2. Merge the selected hook entries into Claude settings.
3. Avoid replacing unrelated hooks or settings.
4. Produce the same configuration on repeated installation.

Installed hook command shape:

```text
membrick hook claude session-start /path/to/project
membrick hook claude user-prompt-submit /path/to/project
membrick hook claude user-prompt-expansion /path/to/project
membrick hook claude pre-tool-use /path/to/project
membrick hook claude post-tool-use /path/to/project
membrick hook claude pre-compact /path/to/project
membrick hook claude stop /path/to/project
membrick hook claude session-end /path/to/project
membrick hook claude subagent-start /path/to/project
membrick hook claude subagent-stop /path/to/project
```

Each command reads one Claude hook payload from stdin and writes one hook
response to stdout.

## Remaining Work

1. Add benchmark coverage for retrieval quality, hook latency, and injected
   token usage, including the three new hooks added in this release.
2. Replace deterministic transcript extraction with optional structured
   summarization after its quality and cost are measured.
3. Add configurable context budgets, routing thresholds, and retention.
4. Define cross-process locking and snapshot reload semantics before sharing the
   hook memory store with the long-running MCP server.
5. Evaluate `PostToolUseFailure` capture after failure deduplication is defined.
6. Evaluate a parent/child segment split for subagents using benchmark evidence.

## Acceptance Criteria

- existing MCP `search` and `info` behavior remains compatible
- the default non-Claude `MemoryService` behavior remains a single index
- Claude memory documents are assigned stable session `group_id` values
- hook retrieval uses segmented `MemoryService::search_with_filters()`
- capture survives process restart
- repeated hook delivery does not duplicate memory
- irrelevant retrieval injects no context
- hook failures do not interrupt normal Claude use by default
- installation preserves unrelated Claude configuration
- tests cover every supported hook payload and response shape
- `PostToolUse` retrieval uses tool name and truncated input/response as query
- `PreToolUse` retrieval uses the planned tool call as query
- `SubagentStart` retrieval uses the subagent prompt
- `SubagentStop` capture is idempotent and stores an `outcome` document type
