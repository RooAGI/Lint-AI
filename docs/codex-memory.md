---
meta:
  - property: og:type
    content: website
  - property: og:title
    content: "Codex Memory: Persistent Project Memory for Codex"
  - property: og:description
    content: "Stop re-explaining your project every session. Lint-AI gives Codex persistent memory that knows what is still true."
  - property: og:image
    content: https://rooagi.github.io/Lint-AI/assets/images/social/codex-memory.png
  - property: og:image:width
    content: "1200"
  - property: og:image:height
    content: "630"
  - property: og:url
    content: https://rooagi.github.io/Lint-AI/codex-memory/
  - name: twitter:card
    content: summary_large_image
---

# Codex Memory: Persistent Project Memory for Codex

Every new Codex session starts from zero. It doesn't remember the architecture decision you made last Tuesday, the API contract you changed last week, or the runbook you rewrote yesterday. You re-explain, it re-discovers, and you pay for the same context in tokens every single time.

If you run AI coding agents against a living codebase, this amnesia is expensive — and it gets worse as the project grows.

What you need is real Codex memory — persistent, project-scoped, and current.

## The problem with static memory files

Codex reads standing project instructions from `AGENTS.md`. It's a good mechanism, and it has the same three failure modes as every static memory file:

- **It goes stale.** A decision changes; the file doesn't — or the old wording lingers somewhere else. Your agent reads outdated guidance as if it were current. (This is the [stale-memory problem](stale-memory.md), and it deserves its own page.)
- **It costs tokens forever.** Everything in the instructions gets injected into context on every session, relevant or not.
- **It doesn't capture what actually happened.** The useful memory of a project lives in past sessions — the debugging, the decisions, the dead ends — not just in the curated notes.

## What the replay measurements show

The project keeps a controlled A/B harness that replays recorded Codex sessions with memory disabled vs. enabled, measuring continuation time, token use, and scenario recall. One measured run:

| | Codex native memory | Codex + Lint-AI |
| --- | --- | --- |
| Session continuation | 38.13 s | 18.53 s |
| Input tokens | 151,000 | 62,478 |
| Tool calls to recover context | 5 | 2 |
| Scenario recall | 2/3 | 2/3 |
| Hook overhead | 0 ms | 1.33 s |

Fewer tokens and faster continuation, because the agent gets the *right* context injected instead of re-discovering it through tool calls. The honest detail: scenario recall was 2/3 in both runs — this particular measurement shows an efficiency win, not a recall win.

And the standing caveat, straight from the repo: these are **diagnostic one-run measurements, not universal performance guarantees**. They are specific to the provider, model, repository, and revision under test. The methodology and measured run artifacts are published in [Codex performance tests](codex-performance-tests.md) so you can evaluate the claim instead of taking it on faith.

## How it works

Lint-AI plugs into Codex through its supported extension points — lifecycle hooks and MCP — not hacks:

- **Lifecycle hooks.** `SessionStart`, `UserPromptSubmit`, `UserPromptExpansion`, `PreToolUse`, `PermissionRequest`, `PostToolUse`, and `SubagentStart` retrieve relevant project context. `PreCompact`, `PostCompact`, `Stop`, `SessionEnd`, and `SubagentStop` capture bounded session memory. Hooks are **fail-open** with a 2-second budget: if anything goes wrong, Codex just continues without the memory layer.
- **AGENTS.md policy merge.** Installation merges the Lint-AI memory policy into your project's `AGENTS.md` — the file Codex already uses for standing instructions — so the agent knows to consult prior project context.
- **Project-scoped storage.** Memory lives under `<project>/.lint-ai/memory/` — isolated per project, inspectable as JSON, yours to keep.
- **Current-state retrieval.** Retrieved context isn't just topically relevant; it's ranked by what's *still true*. Old decisions are preserved as history, not presented as current guidance. See [stale agent memory](stale-memory.md).
- **Runtime controls.** MCP tools (`record_session`, `enable_lint_ai`, `disable_lint_ai`, `lint_ai_status`) let you or the agent start/stop recording and toggle the memory layer without touching config files.

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

At session start, the `SessionStart` hook injects the same kind of record — source, revision provenance, and bounded excerpts — which is what keeps the token cost down (62,478 vs 151,000 input tokens in the measured run).

## Try it — no building required

Codex support ships in the published release binaries. Download the binary for your platform (macOS, Linux, or Windows) from the [releases page](https://github.com/RooAGI/Lint-AI/releases), then:

```bash
# install hooks, MCP server entry, and the AGENTS.md memory policy
./lint-ai --codex-install /path/to/project
```

The installer merges an MCP server entry into `~/.codex/config.toml`, enables Codex's hooks feature flag, adds Lint-AI hooks to `~/.codex/hooks.json`, and merges the memory policy into your project's `AGENTS.md`. Your existing config entries are preserved.

Restart Codex Desktop so it picks up the new config, then verify the installation:

```bash
./lint-ai --codex-verify-mcp /path/to/project
# healthy result: {"status": "healthy", ...}
```

Then just use Codex. New sessions automatically pull in relevant, current project context. When a session ends, the useful parts are captured for next time.

Want to see what's actually stored? It's all inspectable:

```bash
lint-ai --inspect-index .lint-ai/memory
```

And if you ever want it off, the MCP controls or hooks settings disable it without uninstalling anything.

## Frequently asked questions

**Does Codex have memory built in?**
Only static memory: `AGENTS.md` and whatever instruction files you maintain by hand. They work, but they go stale, they cost tokens on every session, and they don't capture what actually happened in past sessions. Lint-AI adds persistent, automatic session memory on top — capture at session end, retrieval at session start, with nothing to curate.

**Will it slow down my sessions?**
The hooks are fail-open with a 2-second budget: if retrieval or capture ever fails or times out, Codex just continues without the memory layer. In the measured replay run, hook overhead was 1.33 s — and overall session continuation got *faster* (38.13 s → 18.53 s), because the agent receives the right context instead of burning tool calls re-discovering it.

**Where is the memory stored?**
Under `<project>/.lint-ai/memory/`, scoped to the project. It's inspectable JSON (`lint-ai --inspect-index .lint-ai/memory`), and it stays on your machine.

**Can I turn it off?**
Yes. Use the `disable_lint_ai` MCP tool, or remove the hooks from your Codex config. Disabling stops retrieval and capture but doesn't delete stored memory or uninstall anything.

## Go deeper

- [Stale agent memory](stale-memory.md) — why agents resurface outdated decisions, with a real benchmark case study
- [Claude Code memory](claude-code-memory.md) — the same memory layer for Claude Code sessions
- [Agent memory guide](agent-memory.md) — the bigger picture on persistent agent memory
- [Codex integration reference](codex.md) — hooks, MCP tools, replay, and verification in full detail
- [Quickstart](quickstart.md) — the five-minute path from zero to indexed
