<!-- lint-ai:memory-policy:start -->
## Lint-AI memory policy

For requests about project history, prior decisions, architectural choices,
earlier work, or "why" something was implemented, call the `lint-ai` MCP
server's `search` tool before reading files or searching the repository.

1. Call `search` with a concise query describing the missing context.
2. Read the returned sources before opening files yourself.
3. Say plainly when memory returned nothing, and only then fall back to
   reading the repository.

Recorded sessions are memory of this project, not documentation of it: a
finding can be out of date. Check a conclusion against the code before relying
on it, and prefer the most recent one when two disagree.
<!-- lint-ai:memory-policy:end -->
