# Lint-AI

**Current-state memory for AI agents.**

> Relevant context is not always true.

Lint-AI turns scattered project history — sessions, documents, decisions, traces, notes, and code — into **current, evidence-backed context** at the moment an agent needs it.

Search can find the right topic. Lint-AI helps an agent distinguish **what is relevant** from **what is still true**.

It is built for long-running AI workflows where old decisions remain semantically relevant even after newer evidence has superseded them.

**Works with Claude Code, Codex, Gemini CLI, and Antigravity CLI (AGY)** through project-scoped memory, lifecycle hooks, and MCP tools.

[Documentation](https://rooagi.github.io/Lint-AI/) · [Quickstart](docs/quickstart.md) · [Agent integrations](docs/agents.md) · [Benchmark methodology](docs/comparison.md)

---

## Why Lint-AI?

Imagine your project history contains both of these notes:

| | Evidence |
|---|---|
| **Older** | “Increase retries to recover from the intermittent timeout.” |
| **Current** | “Retries amplify load. Cap attempts and fix the token clock skew.” |

A normal retrieval system can still rank the first passage highly because it is topically relevant.

Lint-AI adds the signals needed to recover the **current state** while preserving the older guidance as history:

1. **Relevance** — is this about the question?
2. **Recency** — when was this evidence true?
3. **Supersession** — did a newer decision replace it?
4. **Evidence** — where did the answer come from?

That makes Lint-AI useful for agent memory, project knowledge, semantic review, and any corpus where **“what changed?”** and **“what is current?”** matter as much as keyword search.

### Ordinary retrieval vs. Lint-AI

| Ordinary retrieval | Lint-AI |
|---|---|
| Finds topically similar text | Ranks relevant **and current** evidence |
| Can surface stale decisions as if they are current | Preserves supersession and time signals |
| Returns passages | Returns context with source and relationship evidence |
| Treats documents mostly in isolation | Connects facts, entities, links, symbols, ownership, and time |
| Leaves corpus drift hidden | Surfaces contradictions, stale claims, terminology drift, orphan pages, and missing links |

---

## Quickstart

Install directly from GitHub:

```bash
cargo install --git https://github.com/RooAGI/Lint-AI
```

Index or lint a local corpus:

```bash
lint-ai /path/to/repo
```

Query it:

```bash
lint-ai --query "docker install linux" /path/to/repo/docs
```

Ask for LLM-ready retrieval context:

```bash
lint-ai --llm-context "docker install linux" /path/to/repo/docs
```

For Docker, HTTP, Python, Rust, and MCP setup, see the [full quickstart](docs/quickstart.md).

---

## Agent integrations

Lint-AI provides opt-in integrations for the major coding-agent clients.

| Integration | Project memory | Lifecycle capture | MCP tools | Replay / comparison |
|---|---:|---:|---:|---:|
| [Claude Code](docs/claude-code.md) | Yes | Yes | Yes | Yes |
| [Codex](docs/codex.md) | Yes | Yes | Yes | Yes |
| [Gemini CLI](docs/gemini-cli.md) | Yes | Yes | Yes | — |
| [Antigravity CLI / AGY](docs/agy.md) | Yes | Yes | Yes | — |

Build the Claude Code and Codex integrations:

```bash
cargo build --release --features claude-code,codex
./target/release/lint-ai --claude-code-install /path/to/project
./target/release/lint-ai --codex-install /path/to/project
```

Provider memory remains isolated by project and provider:

```text
/path/to/project/.lint-ai/claude-memory/
/path/to/project/.lint-ai/codex-memory/
```

The shared MCP controls include session recording, memory listing, and runtime enable/disable controls.

```text
mcp__lint-ai__record_session       {"action":"start|stop|status"}
mcp__lint-ai__list_memories        {"limit":20}
mcp__lint-ai__enable_lint_ai       {}
mcp__lint-ai__disable_lint_ai      {}
mcp__lint-ai__lint_ai_status       {}
```

See [agent integrations](docs/agents.md) and the [MCP guide](docs/mcp.md) for the complete setup.

---

## Benchmark highlights

Lint-AI is evaluated on **LongMemEval-S**, a public benchmark for long-context agent-memory retrieval over multi-session conversation corpora.

The release backend uses no embedding vectors in this benchmark.

### Heuristic release backend

**500 scoped questions · single CPU core · no GPU**

| Metric | Result |
|---|---:|
| recall@5 | **83.5%** |
| recall@10 | **89.5%** |
| recall_any@10 | **95.6%** |
| MRR | **84.0%** |
| NDCG@10 | **81.8%** |
| Average query latency | **1.9 ms** |

`recall@k` is fractional recall over all gold sessions. `recall_any@k` counts a query as successful when any gold session appears in the top *k*.

Results: `benchmark/data/lintai_longmemeval_scoped_results.json`

<details>
<summary><strong>Experimental rust-bert POS/NER branch results</strong></summary>

<br>

| Metric | Result |
|---|---:|
| recall@5 | **86.9%** |
| recall@10 | **93.8%** |
| recall_any@10 | **98.2%** |
| MRR | **87.1%** |
| NDCG@10 | **86.0%** |
| Average query latency | **5.1 ms** |

Results: `benchmark/data/lintai_longmemeval_scoped_results_0512.json`

The model-backed POS/NER path is experimental and is kept separate from the audited release dependency graph.

</details>

### Integration smoke measurements

The repository also includes reproducible Claude Code and Codex replay/performance tests. These are diagnostic one-run measurements, **not universal performance guarantees**.

| Provider / arm | Continuation | Input tokens | Tool calls | Recall | Hook time |
|---|---:|---:|---:|---:|---:|
| Claude native memory | 22.47 s | 123,536 | 4 | 2/3 | 0 ms |
| Claude Lint-AI only | 7.05 s | 15,914 | 0 | 3/3 | 1.40 s |
| Codex native memory | 38.13 s | 151,000 | 5 | 2/3 | 0 ms |
| Codex Lint-AI only | 18.53 s | 62,478 | 2 | 2/3 | 1.33 s |

Read the full conditions and limitations:

- [Claude Code performance tests](docs/claude-code-performance-tests.md)
- [Codex performance tests](docs/codex-performance-tests.md)
- [Comparison methodology](docs/comparison.md)
- [Session metrics](metrics/README.md)

---

## What Lint-AI can do

- Recover the right past session or note when an agent needs context
- Rank current evidence ahead of superseded guidance
- Preserve historical answers when a question depends on the past
- Catch contradictions before they spread
- Detect terminology drift across documents
- Find orphaned or weakly linked pages
- Query facts, entities, symbols, ownership, and time
- Build review packets from changed files and related semantic context
- Feed grounded context into an LLM instead of raw text blobs
- Keep agent memory inspectable and local to the project

---

## How it works

Lint-AI treats project memory as more than a bag of text.

```text
Sessions · docs · code · traces
            │
            ▼
        1. Ingest
            │
            ▼
 Facts · entities · symbols · time
            │
            ▼
      2. Understand
            │
            ▼
 Links · ownership · co-occurrence
            │
            ▼
       3. Connect
            │
            ▼
 Ranked · sourced · current context
            │
            ▼
       4. Retrieve
```

Under the hood, Lint-AI builds a lexical index plus sparse entity/term tables and overlays graph structure for links, symbols, ownership, and co-occurrence. Queries are analyzed for intent, entities, and temporal hints, then scored with lexical, claim, topic, timestamp, and graph signals.

The same corpus model powers review-oriented checks such as missing cross-references, orphan pages, low-confidence claims, stale knowledge, and semantic drift.

For implementation details, see:

- [Chunk strategy](docs/chunk-strategy.md)
- [Lexical data](docs/lexical-data.md)
- [Artifact indexing](docs/artifact-indexing.md)
- [Shared integration architecture](src/integrations/README.md)

---

## Use Lint-AI from your stack

### CLI

```bash
lint-ai --query "what is the current retry policy?" /path/to/repo
```

### Docker / HTTP

```bash
export SERVER_TOKEN=local-dev-token
docker compose up --build -d
curl http://127.0.0.1:8080/health
```

See the [HTTP server guide](docs/server.md).

### Python

Build the optional Python extension with `maturin`, then use the same in-memory index from Python:

```python
import lint_ai

store = lint_ai.IndexStore()
store.upsert("doc-1", "Docker install guide for Ubuntu hosts")
print(store.query("docker ubuntu", 5))
```

### Rust

```rust
use lint_ai::{IndexStore, PipelineOptions, SourceDocument};

fn main() -> anyhow::Result<()> {
    let mut index = IndexStore::in_memory(PipelineOptions::default());

    index.upsert(SourceDocument {
        doc_id: "artifact-1".to_string(),
        source: "artifact://artifact-1".to_string(),
        content: "docker install guide for linux hosts".to_string(),
        concept: "docker install".to_string(),
        group_id: None,
        headings: vec!["Overview".to_string()],
        links: vec![],
        timestamp: None,
        doc_length: 36,
        author_agent: None,
    });

    let results = index.query("docker install", 5)?;
    println!("{}", serde_json::to_string_pretty(&results)?);
    Ok(())
}
```

See [docs/quickstart.md](docs/quickstart.md) for complete examples.

---

## When is Lint-AI a good fit?

Lint-AI is especially useful when you maintain:

- Long-running agent memory over conversations, notes, and decisions
- Markdown knowledge bases with many cross-links
- Internal documentation where terminology and decisions drift over time
- Codebases where symbols, ownership, usage, and review context matter
- Fast-changing corpora where freshness matters as much as relevance

If your only requirement is simple keyword search over a static set of documents, Lint-AI is probably more machinery than you need.

---

## Project status

Lint-AI is under active development. The release query path uses the heuristic backend; experimental model-backed query semantics remain separate from the audited release dependency graph.

Issues, benchmark reproductions, integration feedback, and contributions are welcome.

- [Open an issue](https://github.com/RooAGI/Lint-AI/issues)
- [Join a discussion](https://github.com/RooAGI/Lint-AI/discussions)
- [Read the documentation](https://rooagi.github.io/Lint-AI/)

## License

Apache-2.0
