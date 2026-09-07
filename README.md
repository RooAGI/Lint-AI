# Lint-AI

**Current-state memory for AI agents.**

> Relevant context is not always true.

Lint-AI turns project history — sessions, documents, decisions, traces, notes, and code — into **current, evidence-backed context** when an agent needs it.

Search can find the right topic. Lint-AI helps an agent distinguish **what is relevant** from **what is still true**.

It is built for long-running AI workflows where old decisions remain semantically relevant even after newer evidence has replaced them.

**Works with Claude Code, Codex, Gemini CLI, and Antigravity CLI (AGY)** through project-scoped memory, lifecycle hooks, and MCP tools.

[Documentation](https://rooagi.github.io/Lint-AI/) · [Quickstart](docs/quickstart.md) · [Agent integrations](docs/agents.md) · [Benchmark methodology](docs/benchmark.md)

---

## The problem: your agent retrieved the right document — and still got the wrong answer

Imagine your project contains two versions of the same decision:

**Older — `decision-a.md`**

```md
# Gateway Retry Policy

Gateway timeout retry attempts: 5.
```

**Newer — `decision-b.md`**

```md
# Gateway Retry Policy

Gateway timeout retry attempts: 2.
```

Both documents are relevant to:

```text
How many retry attempts should we use for gateway timeouts?
```

A normal retriever can return either one — or both — and leave the model to guess which value is current.

Lint-AI uses **relevance + time + semantic relationships** to keep the newer value in current-state retrieval while preserving the older document as history.

There is no required `supersedes:` frontmatter in this example. For simple configuration/value claims, Lint-AI can infer chronological replacement when the documents establish the same semantic domain. Inferred supersession is domain-scoped so unrelated settings such as two different services' `timeout` values do not suppress one another. Explicit supersession metadata remains authoritative when you provide it.

```bash
lint-ai --query \
  "How many retry attempts should we use for gateway timeouts?" \
  examples/demo-real-terminal
```

Current-state retrieval returns the newer decision with `semantic_status: current`, and `--llm-context` excludes the stale value.

### Real terminal demo

The repository includes a reproducible **asciinema PTY recording** that builds the real `lint-ai` binary, sets deterministic file mtimes, runs the neutral query above, verifies the result, and renders the captured terminal session.

![Lint-AI real terminal demo](docs/assets/lint-ai-real-terminal.gif)

```bash
bash scripts/run_real_terminal_demo.sh
```

See [`examples/demo-real-terminal/`](examples/demo-real-terminal/) and [`.github/workflows/demo-real-terminal.yml`](.github/workflows/demo-real-terminal.yml).

---

## Why Lint-AI?

Agent memory is not only a retrieval problem. It is a **state problem**.

A useful memory layer has to answer:

1. **Relevance** — is this about the question?
2. **Recency** — when was this evidence true?
3. **Supersession** — did something newer replace it?
4. **Temporal intent** — is the user asking about now, last Tuesday, or a historical state?
5. **Evidence** — where did the answer come from?

| Ordinary retrieval | Lint-AI |
|---|---|
| Finds topically similar text | Ranks relevant **and current** evidence |
| Can surface stale decisions as if they are current | Tracks time and supersession |
| Treats relative time as mostly query text | Resolves temporal intent against a reference clock |
| Returns passages | Returns context with source and relationship evidence |
| Treats documents mostly in isolation | Connects facts, entities, links, symbols, ownership, and time |
| Leaves corpus drift hidden | Surfaces contradictions, stale claims, terminology drift, orphan pages, and missing links |

Historical evidence is not deleted. A historical query can still retrieve superseded material as history rather than presenting it as current state.

---

## Quickstart

Install directly from GitHub:

```bash
cargo install --git https://github.com/RooAGI/Lint-AI
```

Lint or index a local corpus:

```bash
lint-ai /path/to/repo
```

Query it:

```bash
lint-ai --query "what is the current retry policy?" /path/to/repo/docs
```

Get LLM-ready context:

```bash
lint-ai --llm-context "what is the current retry policy?" /path/to/repo/docs
```

For Docker, HTTP, Python, Rust, and MCP setup, see the [full quickstart](docs/quickstart.md).

---

## Agent integrations

Lint-AI provides opt-in integrations for major coding-agent clients.

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

The shared MCP controls include session recording, memory listing, and runtime enable/disable controls:

```text
mcp__lint-ai__record_session       {"action":"start|stop|status"}
mcp__lint-ai__list_memories        {"limit":20}
mcp__lint-ai__enable_lint_ai       {}
mcp__lint-ai__disable_lint_ai      {}
mcp__lint-ai__lint_ai_status        {}
```

See [agent integrations](docs/agents.md) and the [MCP guide](docs/mcp.md) for the complete setup.

---

## Benchmark highlights

Lint-AI is evaluated on **LongMemEval-S**, a public benchmark for long-context agent-memory retrieval over multi-session conversations.

The current release benchmark uses the **official cleaned LongMemEval-S dataset**, runs queries through the normal production-style `IndexStore` path, and uses no embedding vectors.

**500 questions · heuristic release backend · single CPU · no GPU**

| Metric | Result |
|---|---:|
| Recall@5 | **83.6%** |
| Recall@10 | **89.7%** |
| Recall@20 | **91.2%** |
| Recall-any@10 | **95.8%** |
| MRR | **84.1%** |
| NDCG@10 | **81.8%** |
| Average query latency | **~4.7 ms** |

`Recall@k` is fractional recall over all gold sessions. `Recall-any@k` counts a query as successful when any gold session appears in the top *k*.

The repository keeps the cleaned-dataset downloader, production-style benchmark binary, main-vs-branch comparison workflow, and temporal-reasoning experiment harnesses so results can be reproduced and changes can be evaluated without silently changing the query path.

See [benchmark methodology](docs/benchmark.md), [benchmark results](docs/benchmark-results.md), and [comparison methodology](docs/comparison.md).

<details>
<summary><strong>Experimental rust-bert POS/NER branch</strong></summary>

<br>

An experimental model-backed POS/NER branch has also reached higher retrieval metrics, but it is intentionally kept separate from the audited heuristic release dependency graph. See the benchmark documentation for its results and conditions.

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

- Recover the right past session or project note when an agent needs context
- Rank current evidence ahead of superseded guidance
- Infer simple chronological configuration updates inside an established semantic domain
- Preserve historical evidence for historical questions
- Resolve relative-time queries against an explicit reference date when one is available
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
 Links · ownership · supersession
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

Under the hood, Lint-AI builds a lexical index plus sparse entity/term tables and overlays graph structure for links, symbols, ownership, semantic relations, and time.

A single prepared-query path owns query analysis, augmented search text, temporal context, routing intent, reference-clock handling, semantic suppression, and provenance annotation. The CLI and `IndexStore` use the same semantic/temporal query policy so current-state behavior does not depend on which high-level entry point you call.

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

    index.upsert(SourceDocument::with_stable_doc_id_from_source(
        "docs/install.md".to_string(),
        "Docker install guide for Linux hosts".to_string(),
        "docker install".to_string(),
        None,
        vec!["Overview".to_string()],
        vec![],
        None,
        None,
    ));

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

The security audit workflow runs `cargo audit`. The current release has no newly introduced blocking RustSec vulnerability from the current-state query changes; known dependency warnings are documented in the repository's audit workflow and dependency history.

Issues, benchmark reproductions, integration feedback, and contributions are welcome.

- [Open an issue](https://github.com/RooAGI/Lint-AI/issues)
- [Join a discussion](https://github.com/RooAGI/Lint-AI/discussions)
- [Read the documentation](https://rooagi.github.io/Lint-AI/)

## License

Apache-2.0
