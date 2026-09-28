---
name: lint-ai-memory
description: Use the Lint-AI MCP server as the first source for project memory, prior decisions, and earlier work before inspecting the repository.
---

<!-- lint-ai-managed-skill -->

# Lint-AI Memory

Lint-AI is this project's long-term memory layer. It is separate from
Hermes' own memory (the built-in provider's notes and skills): Hermes'
memory holds conversation notes, while Lint-AI indexes the project workspace
and carries decisions, outcomes, and supersessions across sessions.

The lint-ai MCP server is registered in `~/.hermes/config.yaml` under
`mcp_servers` as `lint-ai`:

```yaml
mcp_servers:
  lint-ai:
    command: /path/to/lint-ai
    args: ["--hermes-serve", "/path/to/project-root"]
```

Its tools appear as `mcp__lint-ai__*`. For requests about project history,
prior decisions, architectural choices, earlier work, or why something was
implemented, call the `mcp__lint-ai__search` tool before reading files or
searching the repository.

Use returned results as context, distinguish retrieved facts from current
source facts, and check cited files when current source details are required.
Say plainly when memory returns nothing. Do not treat recorded sessions as
authoritative documentation; verify conclusions against the current project.

To capture the current session's work so future sessions can find it, use the
`mcp__lint-ai__record_session` tool (`start` at the beginning of a work
session, `stop` at the end). Recording is capture-only: it never changes the
workspace, and a bad memory day never blocks normal work.
