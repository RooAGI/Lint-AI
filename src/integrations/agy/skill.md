---
name: membrick-memory
description: Use the Membrick MCP server as the first source for project memory, prior decisions, and earlier work before inspecting the repository.
---

<!-- membrick-managed-skill -->

# Membrick Memory

For requests about project history, prior decisions, architectural choices,
earlier work, or why something was implemented, call the `membrick` MCP
server's `search` tool before reading files or searching the repository.

Use returned results as context, distinguish retrieved facts from current
source facts, and check cited files when current source details are required.
Say plainly when memory returns nothing. Do not treat recorded sessions as
authoritative documentation; verify conclusions against the current project.
