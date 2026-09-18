# Comparison

This comparison asks a specific question: how does Lint-AI's retrieval layer
perform against another reproducible, retrieval-oriented memory system? It does
not claim that every memory product or agent runtime can be reduced to one
Recall@K score.

The complete scripts and machine-readable artifacts live in the repository's
[`comparison/`](https://github.com/RooAGI/Lint-AI/tree/main/comparison)
directory.

## System selection and scope

We select comparison systems using four requirements:

1. The project must be available for local, reproducible evaluation.
2. It must expose a retrieval layer that returns ranked memory records.
3. It must run against the same dataset, query set, and scoring formulas.
4. The result must not depend on a hosted-only service or undisclosed model
   configuration.

This scope compares retrieval layers. Full agent runtimes should instead be
compared with end-to-end tasks that measure answer accuracy, memory decisions,
token use, and latency.

| System | Role | Direct retrieval without a model? | Model required? | Embeddings required? | In this comparison? |
|---|---|---:|---:|---:|---:|
| Lint-AI | File-based retrieval and memory layer | Yes | No | No | Yes |
| AgentMemory | Retrieval-oriented memory layer | Yes, in BM25 mode | No | No, in BM25 mode | Yes |
| Mem0 | Model-assisted memory layer | No, for its standard semantic pipeline | Yes | Yes | No |
| LangMem | Memory toolkit for LangGraph | No | Yes | Yes | No |
| Letta | Stateful agent runtime with self-managed memory | No; evaluate end to end | Yes | Yes | No |

“Yes” and “No” describe the standard configuration used for a fair semantic
memory comparison. A project may provide additional backends, but those must
be documented and tested separately rather than mixed into this result.

## Selected comparator: AgentMemory

AgentMemory is the selected comparator because it meets the retrieval-layer
requirements without requiring a model or embedding provider in its BM25 mode.
Both systems can therefore be evaluated on the same 500-question
LongMemEval-S set using the same any-hit Recall@K, MRR, and NDCG@10 scorer.

The comparison uses AgentMemory's local `POST /agentmemory/smart-search`
endpoint and its BM25 result artifact. Lint-AI uses local `POST /search` with
the same query set and top-k cutoffs (5, 10, and 20). Hosted Mem0 results and
agent-runtime results are intentionally excluded because they answer a
different evaluation question.

## Detailed comparison

### Retrieval quality

The shared retrieval track contains 500 LongMemEval-S questions. The table
reports **any-hit recall**: the percentage of questions where at least one
correct answer session appears in the top K results.

| System | Any-hit Recall@5 | Any-hit Recall@10 | Any-hit Recall@20 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Lint-AI | 92.4% | 95.6% | 97.0% | 84.0% | 81.8% |
| AgentMemory | 87.0% | 94.8% | 98.4% | 71.6% | 73.0% |

The full scorer and per-question outputs are available in
`comparison/results/retrieval-longmemeval-500.json`.

### HTTP search load

The normalized service run used 23,366 records, 100 requests per cell,
`top_k`/`limit` 20, and a keyword query on the same local machine. These are
0.1.9 baseline numbers; they predate the 0.2.0 Axum/snapshot refactor.

| System | Concurrency | p50 | p90 | p99 | Throughput |
|---|---:|---:|---:|---:|---:|
| Lint-AI | 1 | 6.96 ms | 7.65 ms | 28.17 ms | 140 req/s |
| AgentMemory | 1 | 7.04 ms | 10.53 ms | 16.94 ms | 130 req/s |
| Lint-AI | 10 | 10.23 ms | 11.87 ms | 12.52 ms | 952 req/s |
| AgentMemory | 10 | 58.81 ms | 60.35 ms | 61.33 ms | 171 req/s |

At concurrency 10, Lint-AI delivered about 5.6× the throughput. This measures
service behavior, not retrieval quality; the endpoint implementations and
corpus contents are not identical.

### v0.2.0 layout throughput

The current uncached layout run is recorded in
[`throughput-layout-v0.2.0.json`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-layout-v0.2.0.json).
It used release mode, 23,366 records, 100 requests per cell, `top_k: 20`, and
the query `deployment configuration system decision`. The machine was an Intel
Core i7-7700K with 8 logical CPUs running Ubuntu 22.04.5 LTS (kernel
5.15.0-177-generic) with Rust 1.94.1.

| Layout | C=1 req/s | C=10 req/s | C=10 p50 |
|---|---:|---:|---:|
| Single index | 50.02 | 261.72 | 38.36 ms |
| Global segmented | 83.19 | 385.81 | 25.54 ms |
| Routed segment | 88.10 | 433.32 | 22.28 ms |

These are local uncached measurements. The `single`, `global`, and `segment`
runner commands are documented in `comparison/README.md`.

#### Latest macOS rerun

A fresh rerun was performed on 2026-09-17 against the current server build. It
used the same 23,366-record corpus, 100 requests per cell, `top_k: 20`, keyword
query, and disabled query cache. The corpus was seeded through `POST /add/batch`
and the search measurements used the standard `comparison/http_latency.py`
client through `uv`.

| Layout | C=1 req/s | C=10 req/s | C=10 p50 | C=10 p99 |
|---|---:|---:|---:|---:|
| Single index | 148.42 | 997.62 | 8.87 ms | 18.39 ms |
| Global segmented | 237.82 | 1,306.82 | 6.30 ms | 12.39 ms |
| Routed segment | 237.84 | 1,512.31 | 5.91 ms | 11.97 ms |

The machine was a MacBook Pro (Mac17,9) with an Apple M5 Pro chip (15 cores)
and 24 GB RAM, running macOS 26.6.2. The toolchain was Rust 1.96.0, Cargo
1.96.0, and uv 0.12.7. These results are local uncached measurements and
should not be compared directly with the Ubuntu results above without matching
the hardware and software environment. The provider MCP watcher is not
initialized by this standalone HTTP benchmark; the run validates the current
server build but is not an isolated watcher-overhead measurement.

### Reproduce the comparison

See [`comparison/README.md`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/README.md)
for setup, corpus seeding, AgentMemory instructions, and all caveats. To run
the Lint-AI HTTP client after starting the server:

```bash
python3 comparison/http_latency.py \
  --url http://127.0.0.1:8080/search \
  --payload '{"query":"deployment configuration system decision","user_id":"bench-user","top_k":20}'
```

For the layout throughput run, use `uv` with a repository-local temporary
directory on macOS. This avoids `/tmp` resolving through a symlink, which the
server rejects for safety:

```bash
mkdir -p .benchmark-tmp
TMPDIR="$PWD/.benchmark-tmp" uv run python comparison/throughput.py \
  --mode single --no-cache
```

Repeat with `--mode global` and `--mode segment` for the complete comparison.

The 2026-09-17 rerun is stored in
[`throughput-layout-watcher-rerun-2026-09-17.json`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-layout-watcher-rerun-2026-09-17.json).
It used the 23,366-record uncached workload through `uv`. The standalone HTTP
server does not initialize provider MCP watchers, so this confirms the current
server benchmark but is not an isolated watcher-overhead measurement.
