---
name: membrick-memory
description: Use the Membrick MCP server as the first source for project memory, prior decisions, and earlier work before inspecting the repository.
---

<!-- membrick-managed-skill -->

# Membrick Memory

## MCP-first memory policy

For requests about project history, prior decisions, architectural choices,
earlier work, or "why" something was implemented, the first tool call must be
`mcp__membrick__search`. Do not begin with `Bash`, `Read`, repository search, or
another memory system for those requests.

Use this sequence:

1. Call `mcp__membrick__search` with a concise query describing the missing
   context.
2. Use the returned results as context and distinguish retrieved facts from
   current source facts.
3. Check cited files only after the MCP search, and only when the answer also
   requires current source details.
4. Prefer exact-revision or clearly relevant results and state uncertainty when
   the results are incomplete.

Use `mcp__membrick__info` only to check MCP workspace status or diagnose the
server. It is not a memory search and should not replace `search`.

## Explicit session recording

Session recording is off by default until the user enables Membrick. Enabling
Membrick also starts recording by default. Users can explicitly override this
with `mcp__membrick__record_session` and `{"action":"stop"}`. Recording is
capture-only: it records redacted lifecycle events locally and does not inject
memory into the session.

When the user asks to stop recording, call the same tool with
`{"action":"stop"}`. Use `{"action":"status"}` only when the user asks
whether recording is active. Never start recording implicitly for ordinary
memory searches or project work.

## Membrick memory controls

When the user explicitly asks to turn Membrick memory off, call
`mcp__membrick__disable_membrick`. When the user asks to turn it back on, call
`mcp__membrick__enable_membrick`. These controls affect future Membrick retrieval
and memory capture. Enabling memory also starts recording by default, while
disabling memory leaves the recording choice unchanged. Use
`mcp__membrick__membrick_status` when the user asks whether memory behavior is
enabled. Do not toggle memory implicitly during a task.

When installed, Claude Code may show `Membrick:ON | Record:OFF` in its native
persistent status line. This is informational only; use the MCP control tools
to change either state. Preserve any user-defined Claude status line.

Replay is a separate CLI operation, not an automatic MCP action. Use
`membrick session replay <session-id> --provider claude
--enable-membrick` when the user explicitly asks to rerun a recorded
prompt with Claude. Replay always records its output and creates a new session
ID; it does not modify the baseline archive.

For requests that are exclusively about the current contents of a file or
current command output, repository tools may be used directly. Do not claim
that MCP was searched when the call was unavailable or returned no results.
