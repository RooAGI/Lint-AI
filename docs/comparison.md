# Comparison

This comparison asks a specific question: how does Lint-AI's retrieval layer
perform against another reproducible, retrieval-oriented memory system? It does
not claim that every memory product or agent runtime can be reduced to one
Recall@K score.

The complete scripts and machine-readable artifacts live in the repository's
[`comparison/`](https://github.com/RooAGI/Lint-AI/tree/main/comparison)
directory.

For a dedicated breakdown of the current `POST /search`, `POST /add`,
`POST /add/batch`, and mixed read/write server measurements, see
[HTTP server benchmarks](http-server-benchmarks.md).

## Current HTTP throughput at a glance (2026-10-05)

These are three different workloads and their rates are not interchangeable.
Read throughput counts completed `POST /search` requests against a seeded
corpus. Write acknowledgement throughput counts records accepted by the durable
staging journal. Completed write throughput includes the final flush, so it
accounts for accepted work that became searchable before the run ended.

| Workload | Endpoint and load | Observed median rate | What the rate means |
|---|---|---:|---|
| Read only | `POST /search`, C=10, 23,366 seeded records | 2,246.69 searches/s | Five-run Tantivy 0.25.0 single-index measurement; no concurrent writes. |
| Single-record write | `POST /add`, one writer | 222.52 accepted records/s; 212.36 completed records/s | Latest direct-endpoint run: three 65-second repetitions, no 429 responses. Ack p50/p95/p99: 3.996/4.336/6.964 ms. |
| Batched write | `POST /add/batch`, 128 records per call | 2,528.46 accepted records/s; 1,977.12 completed records/s | Latest default-schedule batch run: three 65-second repetitions; 19.75 successful calls/s. It rejected about 25,500–26,700 requests per run, so the accepted rate is a burst result, not sustainable capacity. |
| Sparse mixed load | Ten search workers and one `/add` per second | 1,248.69 searches/s during writes; 1,209.01 without writes | Latest concentrated-write run; three paired repetitions, 65 writes per run, no 429s, and final writes searchable. |

The read-only, single-write, batch-write, and mixed-load figures are different
workloads and should not be compared as if they ran together. The batch rate is
not sustainable because admission rejected excess requests. Raw artifacts:
[read layout/version run](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-tantivy-upgrade-2026-10-04.json),
[direct `/add` run](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-add-endpoint-2026-10-05.json),
[128-record default-schedule run](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-publication-temporal-incremental-2026-10-05.json),
and [concentrated sparse mixed-load run](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-publication-mixed-concentrated-2026-10-05.json).

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
| Lint-AI | 94.2% | 96.8% | 97.6% | 87.0% | 85.0% |
| AgentMemory | 87.0% | 94.8% | 98.4% | 71.6% | 73.0% |

The full scorer and per-question outputs are available in
`comparison/results/retrieval-longmemeval-500.json`.

### HTTP search load

The normalized service run used 23,366 records, 100 requests per cell,
`top_k`/`limit` 20, and a keyword query on the same local machine. The Lint-AI
rows are the historical v0.1.9 single-index baseline. They predate the v0.2.0
Axum/snapshot refactor and do not describe the current release.

| System | Concurrency | p50 | p90 | p99 | Throughput |
|---|---:|---:|---:|---:|---:|
| Lint-AI | 1 | 6.96 ms | 7.65 ms | 28.17 ms | 140 req/s |
| AgentMemory | 1 | 7.04 ms | 10.53 ms | 16.94 ms | 130 req/s |
| Lint-AI | 10 | 10.23 ms | 11.87 ms | 12.52 ms | 952 req/s |
| AgentMemory | 10 | 58.81 ms | 60.35 ms | 61.33 ms | 171 req/s |

In this historical run at concurrency 10, Lint-AI delivered about 5.6× the
throughput. This measures service behavior, not retrieval quality; the endpoint
implementations and corpus contents are not identical. The full run is in
[`latency-23366.json`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/latency-23366.json).
There is no separate throughput run against the exact v0.2.1 release artifact.

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

#### Historical v0.2.0 macOS rerun (2026-09-17)

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

#### Tantivy 0.25.0 throughput verification (2026-10-04)

The current dependency remains Tantivy 0.25.0. This matched five-run release
comparison used 23,366 records across five sessions, 1,000 measured requests
and 100 warm-ups per concurrency cell, `top_k: 20`, and the same keyword query.
The table reports median throughput across runs:

| Layout | Tantivy | C=1 req/s | C=10 req/s |
|---|---:|---:|---:|
| Single index | 0.25.0 | 376.70 | 2,246.69 |
| Routed segments | 0.25.0 | 332.79 | 888.86 |
| Single index | 0.26.2 | 601.71 | 2,769.53 |
| Routed segments | 0.26.2 | 354.91 | 901.29 |

The 0.26.2 upgrade had higher measured throughput, but the 500-question
LongMemEval rerun showed lower retrieval quality, so this project stays on
0.25.0. These rates describe this specific machine and workload; they are not
directly comparable with the shorter 100-request v0.2.0 runs above. Full
measurements, latency percentiles, binary hashes, and protocol are in
[`throughput-tantivy-upgrade-2026-10-04.json`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-tantivy-upgrade-2026-10-04.json)
and the [Tantivy verification notes](releases/tantivy-0.26.2-verification.md).

#### Current single-index optimization diagnostic

On 2026-10-03, a single-index build with the current query-path optimizations
measured 322.59 req/s at C=1 and 1,723.67 req/s at C=10. It used 23,366
records, 100 measured requests and 10 warm-ups per concurrency level, disabled
query caching, and used the same keyword query. This was one run from a dirty
working tree on Darwin arm64; the exact hardware model was not recorded. Treat
it as a diagnostic for that source state, not as a release result or a direct
comparison with the five-run, cold-start v0.2.0 table. Details and raw values
are in
[`throughput-single-index-2026-10-03.json`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-single-index-2026-10-03.json).

#### Routed segment throughput and Rayon investigation (2026-10-04)

To isolate the C=10 gap, I reran one segment and 23 session segments under a
matched workload: 23,366 records, the same keyword query and `top_k: 20`, 1,000
measured requests and 250 warm-ups per concurrency cell, and five repetitions.
The multi-segment route selected five segments. The table reports median
throughput and median p50 latency across runs.

| Layout | C=1 req/s | C=1 p50 | C=10 req/s | C=10 p50 |
|---|---:|---:|---:|---:|
| One segment | 392.81 | 2.481 ms | 2,477.54 | 3.726 ms |
| 23 segments, route 5 | 484.43 | 2.035 ms | 1,274.95 | 7.686 ms |

The route scorer is not the main cost. In a separate instrumented profile at
C=10, query-term preparation had a 0.012 ms median, route scoring 0.066 ms, and
selected-segment execution 6.003 ms. The profile has 750 samples per stage,
including 250 warm-ups and 500 measured requests, and must not be used as a
throughput measurement because the timers add overhead. It also corrects earlier
instrumentation descriptions: `filter_ms` times document filtering, while the
old `default_segment_execution_ms` included coordinator setup as well as query
execution.

We tested whether limiting each request's Rayon fan-out would improve C=10
throughput. Under the same five-segment route and five-run workload, smaller
caps made throughput worse:

| Per-request fan-out cap | C=1 req/s | C=10 req/s |
|---:|---:|---:|
| 1 | 205.43 | 1,011.25 |
| 2 | 297.45 | 1,060.68 |
| 8 (current cap) | 484.43 | 1,274.95 |

Rayon uses a shared, bounded worker pool. Ten HTTP requests can enqueue segment
tasks concurrently, but do not create 50 Rayon workers. The evidence does not
support changing pool size based on HTTP concurrency or shrinking the
per-request fan-out. The expensive part is doing multiple segment searches per
request.

We then measured route widths 1, 2, 3, and 5. Narrower routes raised C=10
throughput, but lowered Recall@5 on the 100-question LongMemEval-S diagnostic
subset:

| Segments searched | C=10 req/s | Recall@5 | Any-hit Recall@5 |
|---:|---:|---:|---:|
| 1 | 2,336.67 | 0.534 | 0.68 |
| 2 | 1,565.83 | 0.723 | 0.85 |
| 3 | 1,424.67 | 0.811 | 0.89 |
| 5 (current default) | 1,274.95 | 0.886 | 0.94 |

This is a 100-question diagnostic, not a basis for changing the default. Keep
the current route width of five and Rayon cap of eight for now. A promising
follow-up is to search two segments first and expand toward five when the
results look weak; validate any change on the complete 500-question set.
Compile-time sweep values were restored afterward. These measurements were
made on a modified working tree, so they are diagnostics rather than release
claims. Earlier trials fixed redundant corpus-wide ID allow-list construction
in the filter path, then used Instruments to inspect candidate scoring and
allocation. Partial ranking reduced a small ranking timer but yielded only a
modest throughput change; removing candidate-ID materialization changed an
internal timer but did not show a repeatable end-to-end gain. The detailed
trial record and related artifacts are in the
[server performance notes](server.md#multi-segment-routed-filter-cost).

Artifacts: [matched one-vs-five run](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-routed-one-vs-five-2026-10-04.json),
[fan-out and route-width sweep](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-routed-concurrency-breadth-sweep-2026-10-04.json),
[query-stage profile](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-routed-query-stage-profile-2026-10-04.json),
and [LongMemEval width 1](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/retrieval-longmemeval-route-width-1-100-2026-10-04.json.gz),
[2](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/retrieval-longmemeval-route-width-2-100-2026-10-04.json.gz),
[3](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/retrieval-longmemeval-route-width-3-100-2026-10-04.json.gz),
and [5](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/retrieval-longmemeval-route-width-5-100-2026-10-04.json.gz).

#### Mixed HTTP read/write load

The layout throughput run above measures reads after seeding the corpus. For
read performance during writes, use the mixed-load runner. It seeds an
isolated temporary corpus, measures a read-only control, then runs ten search
workers concurrently with batch writers. Search terms vary by request to avoid
reusing the same prepared-query cache entry. The report includes read/write
throughput and latency percentiles, rejected writes, and a check that a record
accepted during the mixed phase can be retrieved after that phase.

```bash
python3 comparison/mixed_load.py \
  --seconds 15 --readers 10 --writers 1 --write-batch 8 \
  --repetitions 3 \
  --output comparison/results/throughput-mixed-read-write-latest.json
```

The paired control and mixed workload run against the same server and corpus;
each repetition starts with a fresh temporary index. See the latest mixed-load
artifact at
[`throughput-mixed-read-write-2026-10-04.json`](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-mixed-read-write-2026-10-04.json).

The October 4 artifact is retained as historical diagnostic evidence. Its
readers used `mixed-bench-user` while the seed belonged to `bench-user`, so its
6,053 req/s control measured empty-scope queries. It cannot be compared with
populated-corpus throughput. The corrected runner searches `bench-user`,
records empty responses and measures final flush time. See the staged-write
results below for the corrected workload.

#### Write-only diagnostic (2026-10-05)

One sequential writer ran with no concurrent searches against a release server
using Tantivy 0.25.0 on an Apple M5 Pro. Each cell used a fresh index seeded
with 23,366 records, five warm-up calls, an eight-second measured phase, and
three repetitions. Batch size is the number of add requests in each
`/add/batch` HTTP call.

| Add requests per batch | HTTP req/s | Records/s | p50 latency | p95 latency |
|---:|---:|---:|---:|---:|
| 1 | 3.42 | 3.42 | 293.614 ms | 306.294 ms |
| 8 | 1.72 | 13.76 | 579.729 ms | 591.153 ms |
| 32 | 0.83 | 26.58 | 1,203.081 ms | 1,217.001 ms |
| 128 | 0.74 | 94.20 | 1,340.838 ms | 1,387.500 ms |

Batching increased record throughput as batch size grew; 128 adds per request
delivered about 27.5 times the single-add record rate, with higher request
latency. No writes failed, and a final search found the last written record in
every repetition. This is a working-tree diagnostic, not a release capacity
claim. The [raw results](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-only-2026-10-05.json)
include environment and per-repetition details; reproduction instructions are
in the [comparison README](https://github.com/RooAGI/Lint-AI/blob/main/comparison/README.md#write-only-http-throughput).

#### Write path optimization profile (2026-10-05)

A macOS `sample` profile during the workload attributed about 44% of sampled
write-path CPU time to rebuilding and persisting the full semantic record
snapshot, and about 35% to refreshing segmented indexes and routing summaries.
We changed both semantic JSON files to compact serialization while preserving
the documented `chunk_lifecycle.json` sidecar. Receipt persistence now appends
only newly created receipts to `receipts.jsonl`; existing `receipts.json` files
remain readable, and retries can reconstruct missing receipts from stored
documents.

A full three-repetition rerun on the same Apple M5 Pro setup measured 3.91
records/s for one add per request, versus 3.42 in the earlier baseline. The
median p50 fell from 293.614 ms to 255.507 ms. Batches of 128 measured 94.03
records/s versus 94.20, with p50 latency of 1,346.843 ms versus 1,340.838 ms.
The single-add improvement is directional; the largest batch shows no
meaningful gain. All 12 cells had zero errors and the last record was visible
to search. These are working-tree diagnostics, not release capacity claims.
The earlier exploratory run that skipped the lifecycle sidecar is not used as
the comparison because it changed a documented persisted artifact. See the
[compatibility-preserving rerun](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-only-compact-records-2026-10-05-rerun.json),
[exploratory compact run](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-only-compact-records-2026-10-05.json),
and [receipt-journal comparison](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-only-receipt-journal-2026-10-05.json).

#### Staged write flow (2026-10-05)

The primary HTTP server now persists adds to a synced journal, accumulates them
in the mutable store and publishes an independent read view. Defaults are a
250 ms publication trigger, 512 pending documents and a 30-second checkpoint
cadence. [The concurrency document](index-concurrency.md#staged-http-writes)
explains visibility, flush, recovery and ownership.

The write-only runner separately reports durable acknowledgement rate and
published record rate. Published rate includes the final flush and checkpoint;
it avoids treating an unfinished backlog as completed work. The previous
synchronous diagnostic acknowledged already published writes, so compare it
with the new published rate when assessing completed-work throughput.

The final write-only rerun used the same 23,366-record seed, 23 session groups,
five warm-up requests, eight seconds per cell and three repetitions. No build
or test process ran alongside this rerun. All writes succeeded and the last
record was searchable in all six cells.

| Add requests per HTTP batch | Previous published records/s | Staged ack records/s | Published records/s including flush | Ack p50 | Ack p99 |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.91 | 44.22 | 42.53 | 3.990 ms | 1,115.332 ms |
| 128 | 94.03 | 307.50 | 259.98 | 4.959 ms | 1,851.218 ms |

Completed-work throughput increased about 10.9 times for single adds and
2.8 times for batches of 128. Final flush took roughly 0.34 seconds for single
adds and 1.65 seconds for batches of 128. These short runs show the effect of
accumulation and deferred checkpoints; sustained capacity needs a longer run
that includes multiple periodic checkpoints.

The corrected mixed run used 23,366 seeded records, 23 session groups, ten
readers, one writer, batches of eight and three paired 10-second repetitions.
Its median populated-corpus read-only rate was 1,145.63 req/s, with 8.602 ms
p50. During writes, median search rate was 788.59 req/s, with 11.033 ms p50
and 22.606 ms p95. Writes acknowledged at 188.00 records/s (23.50 HTTP req/s)
and completed at 181.75 records/s including final flush. Median write p50 was
4.102 ms; p99 was 1,665.212 ms. Every search returned results, all operations
succeeded, and the final written record was searchable in every repetition.

Readers now retain the previous generation during refresh, so refresh no longer
holds their service lock. CPU contention remains: rebuilds consume CPU and the
single writer pauses ingestion during publication. The high write p99 is an
expected limitation of this simple sequential design, and the 250 ms trigger
is not a 250 ms visibility guarantee. These are working-tree diagnostics on
an Apple M5 Pro with Tantivy 0.25.0, not release capacity claims.

Raw reports: [write-only](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-only-staged-2026-10-05-final.json)
and [corrected mixed workload](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-mixed-read-write-staged-2026-10-05.json).

Reproduce the staged measurements after building the release server:

```bash
cargo build --release --bin server
python3 comparison/write_only.py --batch-sizes 1 128 --seconds 8 --repetitions 3 \
  --output comparison/results/throughput-write-only-staged-latest.json
python3 comparison/mixed_load.py --seconds 10 --readers 10 --writers 1 \
  --write-batch 8 --repetitions 3 \
  --output comparison/results/throughput-mixed-read-write-staged-latest.json
```

#### Sustained staged-write profile and rerun (2026-10-05)

We profiled the single mutable writer during a 65-second, one-writer/no-reader
HTTP run on the same 23,366-record, 23-session corpus. The first profile showed
`SemanticAggregate::remove_doc` consuming 5,119 of 8,367 writer-thread samples.
Every changed document called it, including new documents absent from the
aggregate; it scanned all posting maps even when there was nothing to remove.
An early return when `doc_to_chunks` has no entry reduced this to one sample in
the matched follow-up profile. The check also handles existing documents with
zero chunks because insertion stores an empty `doc_to_chunks` entry.

The next profile showed repeated per-document sorting while appending semantic
postings. Inserts now append and mark changed keys; the immutable index builder
sorts those posting lists once immediately before publication. The profile
after this change no longer showed `insert_doc_state` sorting in the refresh
stack. Segmented catalog and routing-summary refresh remain the largest sampled
costs, so publication can still stall the single writer.

The release-mode, three-repetition default-schedule run used Tantivy 0.25.0,
five warm-up requests, 65 measured seconds per cell, a 250 ms refresh trigger,
512 pending documents, and a 30-second checkpoint interval. Each run flushed
after measurement and found its last record through search. The earlier
baseline artifact used the same schedule and corpus but only one run per cell.

| Add requests per HTTP batch | Earlier ack records/s | Earlier published records/s | Current ack records/s | Current published records/s | Current p50 | Current p95 | Current p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 44.99 | 44.71 | 49.12 | 48.77 | 3.955 ms | 4.181 ms | 1,066.625 ms |
| 128 | 211.20 | 204.06 | 304.13 | 293.56 | 4.951 ms | 1,921.721 ms | 2,063.493 ms |

At batch size 128, acknowledgement throughput improved 44% and published
throughput improved 44%; p99 fell from 2,845.844 ms to 2,063.493 ms. Single
adds improved by about 9%. The comparison is directional because the baseline
has one repetition while the updated run has three. These remain local
working-tree diagnostics, not release-capacity claims.

A separate three-repetition diagnostic set the refresh threshold to 32,768
documents to isolate amortized write cost while retaining the 30-second
checkpoint. It measured 1,454.98 acknowledged and 1,299.84 published
records/s for batches of 128, with 4,114.138 ms p99. This threshold is an
experimental setting, not the production default; it trades publication
frequency for longer visibility delays and larger refresh stalls.

Raw reports: [default schedule](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-only-staged-long-default-3rep-2026-10-05.json),
[large-threshold diagnostic](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-only-staged-long-batch32768-batched-sort-3rep-2026-10-05.json),
and [earlier baseline](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-only-staged-long-baseline-2026-10-05.json).
Reproduce with the benchmark schedule flags:

```bash
python3 comparison/write_only.py --batch-sizes 1 128 --seconds 65 --repetitions 3 \
  --output comparison/results/throughput-write-only-staged-long-latest.json
```

#### Background publication diagnostic (2026-10-05)

After moving index preparation and checkpoint serialization to the background
publisher, we ran three 65-second repetitions per HTTP batch size against the
same seeded corpus. This release binary was built from working-tree changes on
revision `a1907ffe66ae8fb26e082845986a0b29b53b4096`, with Tantivy 0.25.0, one
writer, no readers, a 250 ms refresh trigger, 512-document refresh batches,
and a 30-second checkpoint interval.

| Add requests per batch | Accepted records/s (median) | Completed records/s including final flush (median) | p50 | p95 | p99 | Final flush |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 221.05 | 212.30 | 3.996 ms | 4.643 ms | 5.945 ms | 2.68 s |
| 128 | 2,528.43 | 1,983.57 | 1.480 ms | 3.855 ms | 6.168 ms | 17.86 s |

The batch-128 load exceeded publisher capacity: each run accepted 164,352
records, then returned 25,478–25,766 HTTP 429 responses as the pending-write
limit applied. Its acknowledgement rate is therefore burst throughput, not a
sustainable no-rejection rate. Publication lagged acceptance, and the final
flush took about 18 seconds. Single-add runs had no 429 responses. Every run
returned a successful flush and the benchmark found its last accepted record
through search, but it does not yet verify every accepted ID. Treat these as
local diagnostics, not a capacity guarantee or a like-for-like historical
comparison.

The complete raw report records the binary SHA-256, seed size, latency values,
errors, and each repetition: [background publication write-only report](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-publication-long-2026-10-05.json).

The runner's completed-work rate is accepted records divided by measurement
duration plus final-flush duration. It is useful for comparing the full run and
drain, but it is not a direct measurement of the background publisher's steady
service rate. The reports do not yet expose per-generation build throughput or
the pending-document high-water mark.

#### Copy-on-write capture rerun (2026-10-05)

After sharing the source-document, record, and chunk-lifecycle maps across
publication capture, we repeated the same three 65-second cells with a rebuilt
release binary (SHA-256 `969f7ad50da09656cb7a8b29caf3ec4a41f737d81fff398396e89cd8ffac0d02`).
Single-add throughput measured 225.21 accepted and 215.46 completed records/s
at the median, versus 221.05 and 212.30 before the map-sharing change. Median
p50/p95/p99 latency was 3.995/4.257/6.024 ms. All three single-add runs had no
429s, successful flushes, and searchable final writes. The observed throughput
gain is small and should be treated as within local run variation until paired
repetitions confirm it.

With 128 records per request, the median remained 2,528 accepted records/s,
while median completed-work throughput was 1,927.65 records/s. This saturated the
pending-write limit, with 24,992–27,110 rejected requests per run and final
flushes of 17.0–23.8 seconds. It remains a burst measurement, not a sustainable
write rate. [Copy-on-write rerun report](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-publication-cow-long-2026-10-05.json)
contains all repetitions and per-run outcomes.

The completed-work rate is accepted records divided by measurement time plus
final-flush time. It does not directly measure publisher service capacity.

#### Temporal append and coalescing diagnostics (2026-10-05)

An append-only temporal-fact path avoids sorting and rebuilding existing facts
when new records follow the corpus date/document order. Replacements, deletes,
and out-of-order writes keep the full-rebuild fallback. Three 65-second release
runs measured 224.37 accepted / 217.54 completed single records/s, effectively
unchanged from the copy-on-write run. Batch-128 median was 2,528.46 accepted /
1,977.12 completed records/s, with 25,501–26,673 rejected requests. The temporal
rebuild was not the dominant write bottleneck for this benchmark. [Temporal
append report](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-publication-temporal-incremental-2026-10-05.json)
contains the binary hash and per-run results.

We then raised the publication trigger from 250 ms / 512 documents to 1 second
/ 4,096 documents for three 65-second batch-128 runs. Median accepted rate was
2,583.54 records/s; completed-work rate was 2,077.75 records/s. Final flushes
took 14.5–16.3 seconds, and each run still rejected 26,760–27,128 requests.
Coalescing reduced rebuild overhead modestly but did not eliminate saturation.
All flushes succeeded and the final accepted write was searchable. This is a
diagnostic schedule, not a new default. [Coalescing report](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-publication-coalesced-2026-10-05.json)
records the schedule and binary hash.

The write-only harness uses the production segmented layout and seeds 23
session segments. Its default writer round-robins records across those same 23
sessions, so a 128-request publication touches every segment. A paired
three-run, 15-second probe compared that pattern with writes pinned to the
existing `bench-session-0` segment. Round-robin measured 6,058 accepted / 3,531
completed records/s; concentrated writes measured 5,683 / 3,945. These short
cells still saturated admission (about 9,000 and 6,700–6,800 rejections per
run respectively), and corpus growth makes them directional only. The completed
rate includes final flush. Both reports contain raw run data:
[round-robin](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-distribution-probe-2026-10-05.json)
and [concentrated](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-concentrated-probe-2026-10-05.json).
The write-only runner now accepts `--writer-session-group` for this comparison.

#### Direct `/add` endpoint (2026-10-05)

The earlier 225 records/s single-write measurement used `/add/batch` with one
request per HTTP call. We repeated the same 65-second, three-repetition load
against the actual `/add` endpoint using the same release binary. `/add`
measured 222.52 accepted records/s and 212.36 completed-work records/s at the
median, with 3.996 ms p50, 4.336 ms p95, and 6.964 ms p99 acknowledgement
latency. There were no 429s; every flush succeeded and the last accepted record
was searchable. Both routes use the same durable staged writer, so this
confirms the single-add API itself accepts about 220 writes/s under this local
configuration. Acknowledgement means the journal was synced; search visibility
is provided by later publication/flush. [Direct `/add` report](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-add-endpoint-2026-10-05.json)
includes the binary hash and per-run results.

#### Read-heavy mixed-load checks (2026-10-05)

To model an infrequently changing corpus, we paired three 65-second read-only
phases with three phases using ten search workers and one writer submitting one
single-document add per second. Each pair used the same seeded 23,366-record
corpus and fresh server process. Read throughput medians were 1,171 searches/s
without writes and 1,118 searches/s with sparse writes; the median paired
change was about -2%. Median search p99 changed from 16.953 ms to 17.965 ms.
All 195 writes were accepted, there were no 429s, each flush completed in
0.261–0.298 seconds, and the final accepted record was searchable in every run.
The per-run read throughput varied, so treat this as a directional local
measurement.

We also ran one intentionally saturated probe with ten readers and an
unrestricted batch-8 writer. Search throughput fell from 1,277 to 305
searches/s, 13,803 write requests were rejected with 429, and the explicit
flush hit the server's 30-second HTTP timeout (408); the last accepted record
was not yet searchable when checked. That overload result is not representative
of one write per second, but it shows the bounded publisher correctly applies
backpressure while heavy writers compete with searches.

Raw reports: [sparse mixed load](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-publication-mixed-sparse-2026-10-05.json)
and [saturated probe](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-publication-mixed-probe-2026-10-05.json).
The mixed-load runner supports `--write-interval-ms` to reproduce sparse rates.

After adding a fixed writer-session option, we repeated the same paired
three-repetition run on the current release binary with one single-document
write per second pinned to the existing `bench-session-0` segment. Median
read throughput was 1,209 searches/s in control and 1,249 searches/s with
writes; median search p99 changed from 17.445 to 18.352 ms. All 195 writes were
accepted with no 429s, flushes took 0.212–0.215 seconds, and every final write
was searchable. This measured no meaningful read-throughput loss for sparse,
concentrated writes. [Concentrated mixed-load report](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/throughput-write-publication-mixed-concentrated-2026-10-05.json)
records the paired data, writer-segment distribution, and binary hash.

The write-path changes also passed a retrieval control: a release-mode,
single-index rerun on the same 500 LongMemEval questions with `k=1,3,5,10,20`
matched the Tantivy 0.25.0 control's top-10 session ranking for all 500
questions. Recall@5/10/20 was 94.2%/96.8%/97.6%, MRR 0.86965, and NDCG@10
0.85115 in both reports. The rerun and control are available as [write-path
rerun](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/retrieval-longmemeval-current-2026-10-05-write-publication-k20.json.gz)
and [Tantivy 0.25.0 control](https://github.com/RooAGI/Lint-AI/blob/main/comparison/results/retrieval-longmemeval-current-2026-10-05-tantivy-0.25.0-control.json.gz).

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
