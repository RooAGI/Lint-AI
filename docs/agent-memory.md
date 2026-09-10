# Agent Memory for AI Coding Agents

**Lint-AI — AI memory that knows what is still true.**

**Agent memory** lets an AI agent carry useful project knowledge across sessions instead of rediscovering the same decisions, files, failures, and conventions every time it starts work.

Lint-AI is an open-source **current-state agent memory** layer for AI coding agents. It works with Claude Code, Codex, Gemini CLI, and Antigravity CLI (AGY), and exposes project memory through lifecycle hooks, MCP tools, CLI commands, HTTP, Python, and Rust interfaces.

The problem Lint-AI focuses on is simple: persistent memory can remember something correctly and still give an agent the wrong answer today.

## Persistent memory is only the first step

A basic agent-memory system answers questions such as:

- What did we decide about retries?
- Which module owns authentication?
- What failed the last time we tried this migration?
- Where did we leave the unfinished work?

That is valuable because it reduces repeated repository exploration and user restatement. But long-running projects introduce a second problem: **project truth changes**.

A decision that was correct last month may have been replaced yesterday. An API contract may have changed. Ownership may have moved to another team. A runbook may still be topically relevant even though a newer runbook superseded it.

If an agent-memory layer retrieves both versions without distinguishing their state, the model must spend context tokens reading conflicting evidence and then guess which version is current.

Lint-AI treats agent memory as a **state problem as well as a retrieval problem**.

## Relevant does not always mean current

Imagine a project contains two decisions:

```text
January: gateway timeout retries = 5
February: gateway timeout retries = 2
```

Both memories are relevant to the query:

```text
How many retry attempts should we use for gateway timeouts?
```

Semantic similarity alone does not tell the agent which value is active now.

Lint-AI combines relevance with time and semantic relationships so current-state retrieval can rank the newer decision as current while preserving the older decision as historical evidence. Explicit supersession metadata is supported, and simple chronological replacement can also be inferred inside an established semantic domain.

That distinction is useful for architecture decisions, configuration values, API contracts, runbooks, ownership, terminology, implementation plans, and other knowledge that changes over time.

## Agent memory and token budget

Every memory injected into an LLM consumes context. More context is not automatically better context.

An agent-memory system should therefore answer two questions:

1. **What information is relevant enough to retrieve?**
2. **Which of that information is current enough to act on?**

Filtering stale or superseded evidence before it reaches the model can reduce unnecessary context and reduce ambiguity in the prompt. Lint-AI does not currently claim a universal percentage reduction in tokens versus other third-party memory systems; token use depends on the agent, retrieval settings, task, and integration path.

The repository includes reproducible Claude Code and Codex integration measurements that report model tokens, retrieved context, tool calls, latency, and recall. These are diagnostic workload measurements rather than universal product guarantees. See the [Claude Code performance tests](claude-code-performance-tests.md) and [Codex performance tests](codex-performance-tests.md).

## Agent memory and hallucination risk

Memory cannot eliminate hallucinations. A model can still reason incorrectly or produce unsupported details.

What a memory layer can control is the **evidence supplied to the model**. Stale, contradictory, or weakly sourced context can contribute to confident wrong answers. Lint-AI is designed to make those conditions visible by tracking current versus superseded evidence, temporal intent, provenance, contradictions, and semantic drift.

For that reason, the defensible goal is not “zero hallucinations.” It is **less stale and conflicting context, with evidence the agent can inspect**.

## Retrieval quality and speed

Lint-AI publishes reproducible retrieval and service-load benchmarks in the repository.

On the current 500-question LongMemEval-S retrieval track, the heuristic release backend reports:

| Metric | Result |
|---|---:|
| Any-hit Recall@5 | 92.4% |
| Any-hit Recall@10 | 95.6% |
| Any-hit Recall@20 | 97.0% |
| MRR | 84.0% |
| NDCG@10 | 81.8% |
| Average in-process query latency | 1.88 ms |

The normalized HTTP load test uses 23,366 records and 100 requests per cell. At concurrency 10, Lint-AI recorded 952 requests per second in that test. Benchmark conditions, scripts, comparison data, and caveats are published in the [benchmark overview](benchmark.md) and [comparison](comparison.md).

## Claude Code memory, Codex memory, and Gemini CLI memory

Lint-AI provides project-scoped memory integrations for major coding-agent clients:

| Agent | Project memory | Lifecycle capture | MCP tools |
|---|---:|---:|---:|
| Claude Code | Yes | Yes | Yes |
| Codex | Yes | Yes | Yes |
| Gemini CLI | Yes | Yes | Yes |
| Antigravity CLI / AGY | Yes | Yes | Yes |

Each provider can keep isolated project memory while using Lint-AI's shared retrieval and current-state semantics. See [agent integrations](agents.md) for setup details.

## When current-state agent memory is useful

Lint-AI is a strong fit when an AI coding agent works on a project long enough for history to accumulate and change. Typical cases include long-running codebases, changing architecture decisions, evolving documentation, multi-session implementation work, operational runbooks, and projects where provenance matters.

If the only requirement is simple keyword search over a static set of documents, a current-state memory layer may be more machinery than needed.

## Get started

Install Lint-AI directly from GitHub:

```bash
cargo install --git https://github.com/RooAGI/Lint-AI
```

Then index or lint a local project corpus:

```bash
lint-ai /path/to/repo
```

Query current project memory:

```bash
lint-ai --query "what is the current retry policy?" /path/to/repo/docs
```

For complete installation and integration instructions, see the [quickstart](quickstart.md).

## Related terms

Developers may describe this category as **agent memory**, **AI agent memory**, **persistent agent memory**, **coding agent memory**, **LLM memory**, **Claude Code memory**, **Codex memory**, or **MCP memory**. Lint-AI's specific focus within that category is **current-state memory**: retrieving evidence that is relevant while distinguishing what is still true from what has become historical.
