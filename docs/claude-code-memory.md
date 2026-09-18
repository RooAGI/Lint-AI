---
meta:
  - property: og:type
    content: website
  - property: og:title
    content: "Claude Code Memory: Persistent Project Memory for Claude Code"
  - property: og:description
    content: "Stop re-explaining your project every session. Lint-AI gives Claude Code persistent memory that knows what is still true."
  - property: og:image
    content: https://rooagi.github.io/Lint-AI/assets/images/social/claude-code-memory.png
  - property: og:image:width
    content: "1200"
  - property: og:image:height
    content: "630"
  - property: og:url
    content: https://rooagi.github.io/Lint-AI/claude-code-memory/
  - name: twitter:card
    content: summary_large_image
---

# Claude Code Memory: Persistent Project Memory for Claude Code

Every new Claude Code session starts from zero. It doesn't remember the architecture decision you made last Tuesday, the API contract you changed last week, or the runbook you rewrote yesterday. You re-explain, it re-discovers, and you pay for the same context in tokens every single time.

If you run AI coding agents against a living codebase, this amnesia is expensive — and it gets worse as the project grows.

What you need is real Claude Code memory — persistent, project-scoped, and current.

## The problem with static memory files

Claude Code's native memory story is files: `CLAUDE.md`, project memory, things you write by hand. They work, until they don't:

- **They go stale.** A decision changes; the file doesn't. Or it does, but the old wording lingows in some other note. Your agent reads outdated guidance as if it were current. (This is the [stale-memory problem](stale-memory.md), and it deserves its own page.)
- **They cost tokens forever.** Everything in memory files gets injected into context on every session, relevant or not.
- **They don't capture what actually happened.** The useful memory of a project lives in past sessions — the debugging, the decisions, the dead ends — not just in the curated notes.

## What the replay measurements show

The project keeps a controlled A/B harness that replays recorded Claude Code sessions with memory disabled vs. enabled, measuring continuation time, token use, and scenario recall. One measured run:

| | Claude native memory | Claude + Lint-AI |
| --- | --- | --- |
| Session continuation | 22.47 s | 7.05 s |
| Input tokens | 123,536 | 15,914 |
| Tool calls to recover context | 4 | 0 |
| Scenario recall | 2/3 | 3/3 |
| Hook overhead | 0 ms | 1.40 s |

Fewer tokens because the agent gets the *right* context injected instead of re-discovering it through tool calls. Better recall because past sessions are searchable, not just the curated files.

The honest note, straight from the repo: these are **diagnostic one-run measurements, not universal performance guarantees**. They are specific to the provider, model, repository, and revision under test. The methodology and measured run artifacts are published in [Claude Code performance tests](claude-code-performance-tests.md) so you can evaluate the claim instead of taking it on faith.

## How it works

Lint-AI plugs into Claude Code through its supported extension points — lifecycle hooks and MCP — not hacks:

- **Lifecycle hooks.** `SessionStart`, `UserPromptSubmit`, and `UserPromptExpansion` retrieve relevant project context into the session. `PreCompact`, `Stop`, and `SessionEnd` capture bounded session memory. Hooks are **fail-open** with a 2-second budget: if anything goes wrong, Claude Code just continues without the memory layer.
- **Project-scoped storage.** Memory lives under `<project>/.lint-ai/claude-memory/` — isolated per project, inspectable as JSON, yours to keep.
- **Current-state retrieval.** Retrieved context isn't just topically relevant; it's ranked by what's *still true*. Old decisions are preserved as history, not presented as current guidance. See [stale agent memory](stale-memory.md).
- **Runtime controls.** MCP tools (`record_session`, `enable_lint_ai`, `disable_lint_ai`, `lint_ai_status`, `list_memories`) let you or the agent start/stop recording and toggle the memory layer without touching config files.

### What the agent actually sees

We pointed the release binary at the real Lint-AI repository and asked a real question. This is the verbatim report — real code, real docs, nothing simulated:

```json
{
  "query": "why did we drop embeddings",
  "elapsed_ms": 1601,
  "results": [
    {
      "source": "docs/benchmark.md",
      "score": 13.84,
      "semantic_status": "current",
      "text": "correct answer session appears in the top *K* results. ..."
    },
    {
      "source": "docs/comparison.md",
      "score": 12.21,
      "semantic_status": "current",
      "text": "We select comparison systems using four requirements: ..."
    },
    {
      "source": "docs/agent-memory.md",
      "score": 11.01,
      "semantic_status": "current",
      "text": "A basic agent-memory system answers questions such as: ..."
    }
  ]
}
```

Every hit carries its source, its rank score, and its `semantic_status`. The agent doesn't get a pile of similar text — it gets ranked evidence with a verdict on what's still true. (Produced with `lint-ai --recall` against the public Lint-AI repo; you can reproduce it from the release binary. Excerpts trimmed for display.)

At session start, the `SessionStart` hook injects the same kind of record — source, revision provenance, and bounded excerpts — which is what keeps the token cost down (15,914 vs 123,536 input tokens in the measured run).

## Try it — no building required

Claude Code support ships enabled in the published release binaries. Download the binary for your platform (macOS, Linux, or Windows) from the [releases page](https://github.com/RooAGI/Lint-AI/releases), then:

```bash
# install the hooks, MCP server entry, and memory skill into your project
./lint-ai --claude-code-install /path/to/project
```

The installer merges an MCP server entry into `~/.claude.json`, adds Lint-AI hooks to `~/.claude/settings.json`, and installs a project skill that directs Claude to search prior project context. Your existing config entries are preserved.

Verify the installation:

```bash
./lint-ai --claude-code-verify-mcp /path/to/project
# healthy result: {"status": "healthy", ...}
```

Then just use Claude Code. New sessions automatically pull in relevant, current project context at `SessionStart`. When a session ends, the useful parts are captured for next time.

Want to see what's actually stored? It's all inspectable:

```bash
lint-ai --inspect-index .lint-ai/claude-memory
```

And if you ever want it off, the MCP controls or hooks settings disable it without uninstalling anything.

## Frequently asked questions

**Does Claude Code have memory built in?**
Only static memory: `CLAUDE.md` and project memory files that you write and maintain by hand. They work, but they go stale, they cost tokens on every session, and they don't capture what actually happened in past sessions. Lint-AI adds persistent, automatic session memory on top — capture at session end, retrieval at session start, with nothing to curate.

**Will it slow down my sessions?**
The hooks are fail-open with a 2-second budget: if retrieval or capture ever fails or times out, Claude Code just continues without the memory layer. In the measured replay run, hook overhead was 1.40 s — and overall session continuation got *faster* (22.47 s → 7.05 s), because the agent receives the right context instead of burning tool calls re-discovering it.

**Where is the memory stored?**
Under `<project>/.lint-ai/claude-memory/`, scoped to the project. It's inspectable JSON (`lint-ai --inspect-index .lint-ai/claude-memory`), and it stays on your machine.

**Can I turn it off?**
Yes. Use the `disable_lint_ai` MCP tool, or remove the hooks from your Claude settings. Disabling stops retrieval and capture but doesn't delete stored memory or uninstall anything.

## Go deeper

- [Stale agent memory](stale-memory.md) — why agents resurface outdated decisions, with a real benchmark case study
- [Agent memory guide](agent-memory.md) — the bigger picture on persistent agent memory
- [Claude Code integration reference](claude-code.md) — hooks, MCP tools, replay, and verification in full detail
- [Quickstart](quickstart.md) — the five-minute path from zero to indexed
