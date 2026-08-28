---
name: lint-ai-memory
description: Use the Lint-AI MCP server as the first source for project memory, prior decisions, and earlier work before inspecting the repository.
---

<!-- lint-ai-managed-skill -->

# Lint-AI Memory

For requests about project history, prior decisions, architectural choices,
earlier work, or why something was implemented, call the `lint-ai` MCP
server's `search` tool before reading files or searching the repository.

Use returned results as context, distinguish retrieved facts from current
source facts, and check cited files when current source details are required.
Say plainly when memory returns nothing. Do not treat recorded sessions as
authoritative documentation; verify conclusions against the current project.
