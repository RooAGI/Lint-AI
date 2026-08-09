# Lint-AI

Lint-AI is for teams building agent memory and semantic review over large, fast-changing corpora: assistant sessions, task notes, traces, reports, decisions, code, and documentation that accumulate faster than any team can review manually.

It is a good fit for people who maintain:
- long-running agent memory over conversations, notes, and decisions
- markdown knowledge bases with lots of cross-links
- internal docs where terminology drifts over time
- codebases where symbols, ownership, usage, and review context matter
- corpora where “what changed?” and “what is current?” matter as much as keyword search

Use Lint-AI when plain search is not enough and memory recall needs evidence. It treats stored context as a network of facts, concepts, links, symbols, ownership facts, and timestamps, then surfaces misalignment, missing context, and retrieval results suited for downstream review or LLM grounding.

The problem it solves is **system-level consistency**: reading individual documents is not enough when terminology drifts, definitions conflict, ownership changes, or outdated claims persist across a growing corpus. Lint-AI analyzes corpora collectively, not in isolation, to catch these issues before they spread.

Why people use it:
- to recover the right past session or note when an agent needs context
- to catch contradictions before they spread
- to detect terminology drift across documents
- to find orphaned or weakly linked pages
- to ask corpus-level questions over facts, entities, symbols, and time
- to build review packets from changed files and related semantic context
- to feed grounded context into an LLM instead of raw text blobs

## Benchmark

Evaluated on **LongMemEval-S** (500 questions), a public benchmark for long-context agent memory retrieval over multi-session conversation corpora. The scoped variant is used: each query searches only the sessions attached to that question, matching the realistic setting where a system knows which sessions are candidates for a given user. No embedding vectors are used anywhere in the pipeline.

The section below shows both the rust-bert POS/NER branch result and the default heuristic release result.

### Rust-BERT POS/NER Branch

**Aggregate (n=500):**

| metric | value |
|---|---|
| recall@5 | 86.9% |
| recall@10 | 93.8% |
| recall@20 | 94.4% |
| recall_any@5 | 94.8% |
| recall_any@10 | 98.2% |
| MRR | 87.1% |
| NDCG@10 | 86.0% |
| avg query latency | 5.1 ms |

`recall@k` is fractional recall over all gold sessions. `recall_any@k` counts 1.0 if any gold session appears in the top k. Latency is measured on a single CPU core with no GPU.

**By question type:**

| question type | n | recall@5 | recall@10 | recall_any@5 | MRR | NDCG@10 |
|---|---|---|---|---|---|---|
| single-session-assistant | 56 | 100.0% | 100.0% | 100.0% | 99.1% | 99.3% |
| single-session-user | 70 | 97.1% | 98.6% | 97.1% | 86.8% | 89.8% |
| knowledge-update | 78 | 96.8% | 98.7% | 100.0% | 95.2% | 94.4% |
| single-session-preference | 30 | 80.0% | 96.7% | 80.0% | 64.7% | 72.3% |
| temporal-reasoning | 133 | 81.1% | 92.0% | 91.7% | 84.1% | 82.3% |
| multi-session | 133 | 77.7% | 87.1% | 94.7% | 85.3% | 80.2% |

Results file: `benchmark/data/lintai_longmemeval_scoped_results_0512.json`

### Heuristic Release Backend

**Aggregate (n=500):**

| metric | value |
|---|---|
| recall@5 | 83.7% |
| recall@10 | 89.6% |
| recall@20 | 91.1% |
| recall_any@5 | 92.4% |
| recall_any@10 | 95.6% |
| recall_any@20 | 97.0% |
| MRR | 84.3% |
| NDCG@10 | 81.9% |
| avg query latency | 6.1 ms |

`recall@k` is fractional recall over all gold sessions. `recall_any@k` counts 1.0 if any gold session appears in the top k. Latency is measured on a single CPU core with no GPU.

**By question type:**

| question type | n | recall@5 | recall@10 | recall_any@5 | MRR | NDCG@10 |
|---|---|---|---|---|---|---|
| single-session-assistant | 56 | 100.0% | 100.0% | 100.0% | 98.2% | 98.7% |
| single-session-user | 70 | 94.3% | 98.6% | 94.3% | 77.4% | 82.6% |
| knowledge-update | 78 | 94.2% | 96.8% | 98.7% | 94.6% | 92.4% |
| single-session-preference | 30 | 80.0% | 93.3% | 80.0% | 65.8% | 72.2% |
| temporal-reasoning | 133 | 77.0% | 85.7% | 86.5% | 81.8% | 78.2% |
| multi-session | 133 | 72.5% | 79.3% | 93.2% | 82.7% | 74.3% |

Results file: `benchmark/data/lintai_longmemeval_scoped_results.json`

## Quickstart

See [docs/quickstart.md](docs/quickstart.md) for the shortest path to build, run, query, and use the crate from Rust.

## Claude Code and Codex integrations

Lint-AI provides provider-specific integrations for Claude Code and Codex. Each
integration is opt-in at build time and combines lifecycle hooks with a
project-scoped MCP server:

| Feature | Claude Code | Codex |
|---|---:|---:|
| Project-scoped MCP server | Yes | Yes |
| Lifecycle memory retrieval and capture | Yes | Yes |
| Segmented persistent memory | Yes | Yes |
| `search` and `info` MCP tools | Yes | Yes |
| `record_session` control | Yes | Yes |
| Enable/disable Lint-AI at runtime | Yes | Yes |
| Session status reporting | Native status line plus MCP | MCP plus terminal status line |
| Replay with Lint-AI enabled or disabled | Yes | Yes |
| Session import into provider memory | Yes | Yes |

Build and install both integrations:

```bash
cargo build --release --features claude-code,codex
./lint-ai --claude-code-install /path/to/project
./lint-ai --codex-install /path/to/project
```

Provider memory remains isolated:

```text
/path/to/project/.lint-ai/claude-memory/
/path/to/project/.lint-ai/codex-memory/
```

The MCP controls are available inside either client:

```text
mcp__lint-ai__record_session       {"action":"start|stop|status"}
mcp__lint-ai__enable_lint_ai       {}
mcp__lint-ai__disable_lint_ai      {}
mcp__lint-ai__lint_ai_status        {}
```

Enabling Lint-AI enables recording by default. Recording can then be stopped
independently, so memory retrieval and full session capture remain separate
choices. Recording is local, bounded, redacted, and capture-only; it does not
inject memory by itself.

For a baseline/replay A/B comparison:

```bash
./lint-ai --replay-session <session-id> \
  --session-provider codex \
  --replay-disable-lint-ai
./lint-ai --replay-session <session-id> \
  --session-provider codex \
  --replay-enable-lint-ai
```

Each replay receives a new `replay-*` session ID and a separate archive. The
comparison workflow measures task quality, token usage, latency, tool
activity, memory retrieval, and recording completeness.

### Integration benchmark

These are the latest validated smoke-run measurements for the segmented-routing
scenario: one repetition, continuation phase only. They are diagnostic results
for the stated run, not universal performance guarantees.

| Provider / arm | Continuation | Input tokens | Cached input | Output tokens | Tool calls | Recall | Hook time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Claude native memory | 22.47 s | 123,536 | 123,526 | 1,239 | 4 | 2/3 | 0 ms |
| Claude Lint-AI only | 7.05 s | 15,914 | 15,912 | 376 | 0 | 3/3 | 1.40 s |
| Claude both layers | 7.17 s | 15,914 | 15,912 | 271 | 0 | 3/3 | 1.36 s |
| Codex native memory | 38.13 s | 151,000 | 99,840 | 1,505 | 5 | 2/3 | 0 ms |
| Codex Lint-AI only | 18.53 s | 62,478 | 43,264 | 604 | 2 | 2/3 | 1.33 s |
| Codex both layers | 13.98 s | 31,684 | 25,088 | 454 | 1 | 2/3 | 1.39 s |

The corresponding reports and run conditions are documented in the [Claude
Code performance tests](docs/claude-code-performance-tests.md) and [Codex
performance tests](docs/codex-performance-tests.md). Both documents describe
the one-run limitation and the provider/model/repository versions used.

MCP-path smoke validation measured 47.09 s for Claude MCP-only and 9.38 s for
Codex with Lint-AI MCP. The Codex run recorded 13,246 input tokens, 77 output
tokens, 2/3 recall, and 1.35 s hook time; the Claude run recorded 189,022
input tokens, 3,678 output tokens, and one Lint-AI MCP call. A separate direct
Claude MCP search check returned results in approximately 220 ms; it did not
measure agent search-selection behavior.

More detail: [Claude Code integration](docs/claude-code.md), [Codex
integration](docs/codex.md), [shared integration architecture](src/integrations/README.md),
and [session metrics](metrics/README.md).

## How It Works

Lint-AI ingests sessions, notes, traces, documents, and code-oriented artifacts, builds a lexical index plus sparse entity/term tables, and overlays graph structure for links, symbols, ownership, and co-occurrence. Queries are analyzed for intent, entities, and temporal hints, then scored with a blend of lexical, semantic, claim, topic, timestamp, and graph signals. The same corpus analysis also powers misalignment checks such as missing cross-refs, orphan pages, low-confidence claims, and review-oriented packet generation.

If you want the deeper implementation view, see:
- `docs/chunk-strategy.md`
- `docs/lexical-data.md`
- `docs/artifact-indexing.md`

## How To Use

Query semantics currently use the heuristic backend in the release build.
The model-backed POS/NER path was used in the experimental rust-bert branch and is kept separate from the audited release graph.

### CLI

Lint a corpus:

```bash
./lint-ai /path/to/repo
```

If you prefer `cargo run`:

```bash
cargo run --bin lint-ai -- /path/to/repo
```

Query the corpus:

```bash
./lint-ai --query "docker install linux" /path/to/repo/docs
```

Get LLM-ready retrieval context:

```bash
./lint-ai --llm-context "docker install linux" /path/to/repo/docs
```

Inspect the derived inventory:

```bash
./lint-ai /path/to/repo/docs --show-concepts
./lint-ai /path/to/repo/docs --show-headings
```

Show the main query-oriented modes:

```bash
./lint-ai --index /path/to/repo/docs
./lint-ai --show-tier0 /path/to/repo
./lint-ai --show-tier1-entities /path/to/repo
./lint-ai --show-tier1-terms /path/to/repo --tier1-term-ranker yake
```

Use spaCy for Tier 1 entities if available:

```bash
./lint-ai /path/to/repo --show-tier1-entities --tier1-ner-provider spacy --spacy-model en_core_web_sm
```

For optional Claude Code support, build with `--features claude-code` and see
[docs/claude-code.md](docs/claude-code.md) for the MCP, hooks, and install flow.

Common graph exports, chunking knobs, lexical subset regeneration, and artifact indexing details are documented in `docs/`.

### Rust Library

Use `IndexStore` when you want mutable ingestion, `MemoryIndex` when you want an immutable query snapshot, and `SourceDocument` to add content. For the broader semantic graph, use `CorpusGraph`, `SymbolStore`, `UsageGraph`, and the review model types that are re-exported from the crate root.

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

For corpus-local persistence under `.lint-ai/`, use:

```rust
use std::path::Path;
use lint_ai::{IndexStore, PipelineOptions};

let index = IndexStore::for_corpus(Path::new("/path/to/corpus"), PipelineOptions::default())?;
```

If you already have prepared `DocRecord` values and want the built search structure directly, use `lint_ai::index::MemoryIndex`.

For symbol and review workflows, the crate root also re-exports:

- `CorpusGraph` for querying documents, symbols, and usage together
- `SymbolRecord` / `SymbolStore` for symbol indexing
- `OwnershipRecord` / `OwnershipSummary` for ownership facts and summaries
- `UsageGraph` / `UsageEdge` / `UsageNode` for symbol and ownership relationship tracing
- `ReviewPacket` / `ReviewFinding` / `ReviewDiff` for structured review output

## Advanced

By default the linter skips files larger than 5MB and stops after 50k files. Override these limits:

```bash
./lint-ai /path/to/repo --max-bytes 10000000 --max-files 100000
```

Limit directory traversal depth:

```bash
./lint-ai /path/to/repo --max-depth 10
```

Limit total bytes read across the corpus:

```bash
./lint-ai /path/to/repo --max-total-bytes 100000000
```

The tool automatically scopes to `/path/to/repo/docs/**` when that folder exists.

## Configuration

Analyze a corpus and emit a suggested `lint-ai.json` config:

```bash
./lint-ai /path/to/repo/docs --analyze
```

Example output:

```
Suggested config:
{
  "stopwords": ["group messages", "pairing", "channel routing"],
  "ignore_sections": ["unscoped", "related"],
  "ignore_crossref_sections": ["unscoped", "related"],
  "ignore_paths": [],
  "allowlist_concepts": []
}

Stats:
pages: 31
top concepts:
- group messages (25)
- pairing (25)
- channel routing (22)
- slack (11)
- telegram (11)
top sections:
- configuration (41)
- setup (35)
- unscoped (31)
- security (28)
- related (22)
```

You can place a `lint-ai.json` file in the target root, or pass `--config /path/to/lint-ai.json`, to control filters.

Use `--strict-config` to fail fast if the config is invalid.

Limit config size:

```bash
./lint-ai /path/to/repo --max-config-bytes 2000000
```

Example used for Openclaw channels:

```json
{
  "stopwords": ["channel", "message", "messages", "bot", "client", "config"],
  "ignore_sections": ["related", "unscoped"],
  "ignore_crossref_sections": ["related", "unscoped"],
  "ignore_paths": [],
  "allowlist_concepts": ["discord", "slack", "telegram", "whatsapp", "signal", "matrix"],
  "scope_prefix": "docs/channels/"
}
```

Run it with `./lint-ai /path/to/openclaw/docs/channels --config /path/to/openclaw/lint-ai.json`.

## Contributing

If you want to contribute, the most useful workflow is:

```bash
cargo build
cargo test
cargo fmt --all
```

Please keep changes focused and include tests when behavior changes.

Useful contribution rules:

- run `cargo test` before opening a PR
- run `cargo fmt --all` for Rust code changes
- update docs when CLI flags, benchmarks, or query behavior change
- include benchmark notes when ranking or retrieval logic changes
- prefer small PRs with a clear scope

If the change affects query quality, mention the benchmark result it moves.

## Output Examples

Running the linter emits findings tagged with severity and link-debt signals:

```text
Missing cross-ref in docs/channels/discord.md -> [[signal]] (high)
Low link density in docs/channels/location.md (outgoing 1, avg 4.2)
Unreachable page: docs/channels/legacy.md
Orphan page: docs/channels/unused.md
```

Use `--show-concepts` when you need the derived concept inventory for tuning stopwords or allowlists in `lint-ai.json`.
