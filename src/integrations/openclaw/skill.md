---
name: lint-ai-memory
description: Use the Lint-AI MCP server as the first source for project memory, prior decisions, and earlier work before inspecting the repository.
---

<!-- lint-ai-managed-skill -->

# Lint-AI Memory

Lint-AI is this project's long-term memory layer. It is separate from
OpenClaw's own memory (the daily logs and `MEMORY.md` in
`~/.openclaw/workspace/`): OpenClaw's memory holds conversation notes, while
Lint-AI indexes the project workspace and carries decisions, outcomes, and
supersessions across sessions.

For requests about project history, prior decisions, architectural choices,
earlier work, or why something was implemented, call the `lint-ai` MCP
server's `search` tool before reading files or searching the repository.

Use returned results as context, distinguish retrieved facts from current
source facts, and check cited files when current source details are required.
Say plainly when memory returns nothing. Do not treat recorded sessions as
authoritative documentation; verify conclusions against the current project.

To capture the current session's work so future sessions can find it, use the
`lint-ai` MCP server's `record_session` tool (`start` at the beginning of a
work session, `stop` at the end). Recording is capture-only: it never changes
the workspace, and a bad memory day never blocks normal work.
