---
meta:
  - property: og:type
    content: website
  - property: og:title
    content: "MCP Memory Server: Agent Memory Over the Model Context Protocol"
  - property: og:description
    content: "Lint-AI exposes persistent agent memory as an MCP server: search, session recording, and memory controls as tools any MCP-capable client can call."
  - property: og:image
    content: https://rooagi.github.io/Lint-AI/assets/images/social/mcp-memory-server.png
  - property: og:image:width
    content: "1200"
  - property: og:image:height
    content: "630"
  - property: og:url
    content: https://rooagi.github.io/Lint-AI/mcp-memory-server/
  - name: twitter:card
    content: summary_large_image
---

# MCP Memory Server: Agent Memory Over the Model Context Protocol

Plenty of projects ship a "memory MCP server." Most of them do the same thing: store text blobs, return them by keyword or vector similarity, and call it memory. Your agent gets back whatever is topically related — including the decisions you already reversed.

What you need is an MCP memory server that knows what is *still true*.

## The problem with naive memory servers

A typical memory MCP server gives your agent two operations: save this, find things like this. That breaks down in practice:

- **No sense of time.** A decision from January and its reversal from June are equally retrievable. The server can't tell your agent which one is current. (This is the [stale-memory problem](stale-memory.md).)
- **No evidence.** Results come back as bare text with no link to where they came from — no session, no revision, no provenance.
- **Locked to one client.** Memory you build inside one agent's proprietary format doesn't travel.

Lint-AI's MCP interface is built on the same retrieval core as everything else in the project: current-state ranking, evidence links, and supersession tracking — exposed as standard MCP tools any MCP-capable client can call.

## What the server gives your agent

Seven tools, same names across every supported agent:

| Tool | What it does |
| --- | --- |
| `search` | Search the indexed workspace, ranked by what's still true. Takes `query`, optional `top_k` (1–20). |
| `info` | Basic information about the indexed workspace. |
| `list_memories` | Bounded previews of indexed agent memories (`limit` 1–100, default 20). |
| `record_session` | Start, stop, or inspect capture-only session recording. |
| `enable_lint_ai` / `disable_lint_ai` | Toggle memory retrieval and capture without touching config. |
| `lint_ai_status` | Report independent memory and recording state. |

MCP calls are explicit agent actions — the model calls `search` when it needs workspace evidence. (For automatic injection without a tool call, see the [lifecycle hooks](claude-code.md).)

The memory behind these tools is the real thing, not a demo: the project's replay harnesses measure it end to end — [Claude Code](claude-code.md) sessions continued in 7.05 s vs 22.47 s on native memory, [Codex](codex.md) in 18.53 s vs 38.13 s, both with a fraction of the input tokens. Same index, same ranking, now callable over MCP.

## How it connects

Each agent adapter exposes a provider-local MCP server over the host client's configured MCP transport. The installers wire it up automatically:

```bash
# Claude Code: merges the MCP server entry into ~/.claude.json
./lint-ai --claude-code-install /path/to/project

# Codex: merges mcp_servers.lint-ai into ~/.codex/config.toml
./lint-ai --codex-install /path/to/project
```

Or run a server directly — for example, to point any MCP-capable client at it:

```bash
./lint-ai --claude-code-serve /path/to/repo
./lint-ai --codex-serve /path/to/repo
```

Memory and recording state are project- and provider-scoped. Recording is capture-only: it writes session events but never injects them as memory. Tool and hook failures are fail-open, so the host agent keeps working if the memory layer has a bad day.

## Try it — no building required

Download the release binary for your platform (macOS, Linux, or Windows) from the [releases page](https://github.com/RooAGI/Lint-AI/releases), then start a server against any project:

```bash
./lint-ai --claude-code-serve /path/to/project
```

We did exactly that against the real Lint-AI repo and sent a standard MCP `tools/call`. Verbatim exchange:

```json
// client sends
{"jsonrpc": "2.0", "id": 2, "method": "tools/call",
 "params": {"name": "search",
            "arguments": {"query": "why did we drop embeddings", "top_k": 3}}}
```

```json
// server responds — content[0].text parsed for readability,
// excerpts trimmed; scores and statuses verbatim
{"results": [
  {"source": "docs/benchmark.md",  "score": 14.17,
   "semantic_status": "current", "superseded_by": null, ...},
  {"source": "docs/comparison.md", "score": 12.54,
   "semantic_status": "current", "superseded_by": null, ...},
  {"source": "docs/agent-memory.md", "score": 10.88,
   "semantic_status": "current", "superseded_by": null, ...}
]}
```

Ranked results with scores, sources, and a `semantic_status` verdict on every hit — over plain JSON-RPC, callable from any MCP client. To see what's in the index underneath, inspect it directly:

```bash
lint-ai --inspect-index .lint-ai/claude-memory
```

For the full agent experience (automatic retrieval at session start, capture at session end), use the installer for your agent instead of running the server by hand.

## Frequently asked questions

**Which clients can use the MCP server?**
Any MCP-capable client. The same tool names are exposed through the Claude Code, Codex, Gemini CLI, and Antigravity CLI integrations, and the server speaks standard MCP JSON-RPC over the client's configured transport.

**Do I need the agent CLIs installed to use it?**
No. MCP and server mode are general Lint-AI interfaces — an application doesn't need Claude Code, Codex, Gemini CLI, or Antigravity to use the core index, the HTTP API, or the MCP tools. Only the lifecycle hooks are agent-specific.

**How is this different from other memory MCP servers?**
Current-state retrieval. Most memory servers rank by topical or vector similarity; Lint-AI ranks by what is still true, tracks supersession (new decisions replace old ones without deleting history), and returns evidence with every result. See [stale agent memory](stale-memory.md) for why that matters.

**Is the memory shared between agents?**
Memory state is project- and provider-scoped: the same project indexed once serves every connected client through the same tools.

## Go deeper

- [MCP interface reference](mcp.md) — the complete tool contract and server modes
- [Stale agent memory](stale-memory.md) — why current-state ranking beats similarity ranking
- [Claude Code memory](claude-code.md) / [Codex memory](codex.md) — the full agent integrations
- [Agent memory guide](agent-memory.md) — the bigger picture on persistent agent memory
- [Quickstart](quickstart.md) — the five-minute path from zero to indexed
