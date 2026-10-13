---
name: membrick-memory
description: Use the Membrick MCP server as the first source for project memory, prior decisions, and earlier work before inspecting the repository.
---

<!-- membrick-managed-skill -->

# Membrick Memory

Membrick is this project's long-term memory layer. It is separate from
Hermes' own memory (the built-in provider's notes and skills): Hermes'
memory holds conversation notes, while Membrick indexes the project workspace
and carries decisions, outcomes, and supersessions across sessions.

The membrick MCP server is registered in `~/.hermes/config.yaml` under
`mcp_servers` as `membrick`:

```yaml
mcp_servers:
  membrick:
    command: /path/to/membrick
    args: ["--hermes-serve", "/path/to/project-root"]
```

Its tools appear as `mcp__membrick__*`. For requests about project history,
prior decisions, architectural choices, earlier work, or why something was
implemented, call the `mcp__membrick__search` tool before reading files or
searching the repository.

Use returned results as context, distinguish retrieved facts from current
source facts, and check cited files when current source details are required.
Say plainly when memory returns nothing. Do not treat recorded sessions as
authoritative documentation; verify conclusions against the current project.

To capture the current session's work so future sessions can find it, use the
`mcp__membrick__record_session` tool (`start` at the beginning of a work
session, `stop` at the end). Recording is capture-only: it never changes the
workspace, and a bad memory day never blocks normal work.
