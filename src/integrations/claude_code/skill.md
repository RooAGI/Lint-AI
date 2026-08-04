---
name: lint-ai-memory
description: Use the Lint-AI MCP server as the first source for project memory, prior decisions, and earlier work before inspecting the repository.
---

# Lint-AI Memory

## MCP-first memory policy

For requests about project history, prior decisions, architectural choices,
earlier work, or "why" something was implemented, the first tool call must be
`mcp__lint-ai__search`. Do not begin with `Bash`, `Read`, repository search, or
another memory system for those requests.

Use this sequence:

1. Call `mcp__lint-ai__search` with a concise query describing the missing
   context.
2. Use the returned results as context and distinguish retrieved facts from
   current source facts.
3. Check cited files only after the MCP search, and only when the answer also
   requires current source details.
4. Prefer exact-revision or clearly relevant results and state uncertainty when
   the results are incomplete.

Use `mcp__lint-ai__info` only to check MCP workspace status or diagnose the
server. It is not a memory search and should not replace `search`.

For requests that are exclusively about the current contents of a file or
current command output, repository tools may be used directly. Do not claim
that MCP was searched when the call was unavailable or returned no results.
