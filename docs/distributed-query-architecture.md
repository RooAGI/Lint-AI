# Distributed Query Architecture

Status: single-process architecture under hardening; remote transport is a later
deployment option

This document is the architectural contract for segmented querying in `lint-ai`.
It describes the system when one `lint-ai` process owns the complete corpus and
how that coordinator can participate in a larger deployment. No network service
is required for the local architecture below.

## Scope and terminology

The word *distributed* describes the query shape: independent segments are
queried and reduced by a coordinator. In the current deployment all components
run in one process and share one immutable snapshot.

- **Record**: the persisted unit of content and metadata.
- **Segment**: an independently searchable group of records with its own
  `MemoryIndex` and compact `SegmentProfile`.
- **Snapshot**: the immutable set of segments published for one query view.
- **Coordinator**: the query layer that routes, executes, and reduces a
  snapshot's segments.
- **Compatibility core**: a temporary single `MemoryIndex` reconstructed from
  records only for legacy dump/inspection formats. It is never part of a
  segmented snapshot and is never used for routed search.

## Invariants

The following are correctness requirements, not implementation preferences:

1. A segmented snapshot owns segments, not a redundant corpus-wide index.
2. Every segment query in one coordinator operation uses the same snapshot
   view and the same corpus-wide BM25 statistics.
3. Routing may reduce work, but it must be observable when it excludes
   segments. Explicit fallback-to-all behavior is preferable to hidden global
   index access.
4. Segment-local aggregation and coordinator aggregation are separate phases.
   Local aggregation makes each segment's candidate set useful; coordinator
   aggregation combines evidence across segments and sessions.
5. A query can return partial results only with explicit completeness
   diagnostics identifying expected, successful, and failed segments.
6. Result ordering is deterministic for equal scores.
7. Persistence format compatibility must not reintroduce retained global-index
   ownership into the runtime snapshot.

## Runtime topology

```text
                         one lint-ai process
                                  |
                         immutable query snapshot
                                  |
                    +-------------+-------------+
                    |                           |
             segment catalog              corpus statistics
          profiles and IDs             global BM25 provider
                    |                           |
                    +-------------+-------------+
                                  |
                         query coordinator
                   route -> execute -> reduce
                       /          |          \
                      /           |           \
                 Segment A     Segment B     Segment C
                records/index records/index records/index
                      \           |           /
                       +----------+----------+
                                  |
                  session/group aggregation -> top K
```

`SegmentedMemoryIndex` owns the segment collection, `SegmentCorpusStats`, and
the cached BM25 provider for one generation. Its generation is the publishing
`IndexStore` revision, exposed in query diagnostics so callers can correlate
results with the snapshot that produced them.
`MemoryIndexSnapshot` publishes either one ordinary index or one segmented
index; the two layouts do not share a hidden global core.

## Query lifecycle

### 1. Snapshot selection

`IndexStore` obtains the currently published snapshot. A query must use that
same snapshot for segment profiles, indexes, and statistics. Refresh publishes
a replacement snapshot with the current store revision as its generation; it
does not mutate the one already being queried.

### 2. Query preparation

The query is normalized and its routing terms are extracted. Temporal context,
reference date, allowed document IDs, and filters are prepared before routing so
the coordinator can report what was actually searched.

### 3. Routing

The coordinator scores segment profiles using `SegmentRoutingStrategy`. It
selects up to the configured segment limit and records both selected and
fallback routes. An empty or low-signal query may use an explicit fallback
policy; it must not silently consult a second corpus-wide index.

Routing is an optimization, not a scoring authority. A selected segment still
performs the real lexical/semantic query against its own index.

### 4. Statistics and execution

`GlobalBm25Statistics` combines the live Tantivy searchers from the snapshot's
segments. Each selected segment receives the same provider, so BM25 document
frequency and document-length statistics are corpus-wide even though postings
are segment-local. For a published `SegmentedMemoryIndex`, that provider is
constructed once with the snapshot and reused across queries; standalone
segment-slice helper functions construct an equivalent provider for their
short-lived call.

The execution boundary is fallible. Local execution currently implements it
directly in bounded batches of up to eight segments; a future transport may
implement the same logical operation remotely. Failures are recorded per
segment instead of being converted into an all-or-nothing query failure.
Batching bounds local thread creation while preserving deterministic
coordinator reduction.

### 5. Reduction

The coordinator merges successful segment candidates, applies the
coordinator-tier session/group aggregation, sorts deterministically, and only
then truncates to `top_k`. Segment-local aggregation remains distinct because
it serves a different purpose and operates before cross-segment evidence is
available. Each segment receives an overflow-safe candidate budget of
`2 * top_k` (with saturation at `usize::MAX`) so group collapse during reduce
does not unnecessarily starve the final result set.

The output contains results and diagnostics. Diagnostics include routing,
coverage, per-segment result counts, and completeness. Callers can therefore
distinguish “no relevant result” from “some segments did not answer.” All
segmented query variants, including enrichment and temporal-path variants,
carry the snapshot generation in their diagnostics.

## Scoring contract

Scores are comparable only within a consistent statistics view. Per-segment
normalization is not a substitute for global BM25 statistics: it can promote a
weak result from a weak segment to the same normalized score as a strong result
from another segment.

The current provider is intentionally live and local. Before remote transport,
the statistics abstraction should gain an explicit serializable form containing
query-scoped field totals, document frequencies, and a snapshot/statistics
generation. That is a transport requirement, not a reason to restore
`global_index` ownership.

## Failure and consistency model

For the single-process deployment:

- A query is evaluated against one immutable snapshot and reports its
  snapshot generation.
- Segment execution is best-effort and produces completeness diagnostics.
- A missing or failed selected segment reduces completeness; it does not imply
  that the successful results are invalid.
- Refresh creates a new snapshot and statistics view for subsequent queries.
- Normal segmented persistence stores records plus a logical `segments.json`
  manifest and rebuilds segment-local indexes without a global compatibility
  core. Older stores without a manifest fall back to deterministic group-based
  reconstruction; explicit legacy dumps may still reconstruct a compatibility
  core, but that reconstruction is outside the query path.

The architecture intentionally does not claim remote guarantees yet. Network
deadlines, worker discovery, retries, replication, and cross-process generation
negotiation belong to the transport layer.

## Maturity gates

The local architecture is ready for further hardening when these properties are
measured and enforced:

- **Snapshot identity:** expose and test a generation shared by routing,
  indexes, and statistics.
- **Bounded execution:** implemented locally with a maximum batch concurrency
  of eight; benchmark the setting against corpus size and query latency.
- **Candidate budget:** implemented locally as a `2 * top_k` oversampling policy;
  benchmark the multiplier against ranking recall and query cost.
- **Statistics lifecycle:** snapshot-owned provider caching is implemented;
  benchmark construction and query costs before adding invalidation complexity.
- **Ranking parity:** compare all-segment routed results with the single-index
  reference on representative corpora.
- **Routing recall:** measure relevant-result loss as segment limits change.
- **Operational evidence:** benchmark latency, memory, statistics construction,
  completeness, and degraded-segment behavior.
- **Persistence evolution:** introduce a physical segmented manifest and
  per-segment cores before record-based rebuild becomes a scalability
  bottleneck. The logical manifest is implemented; per-segment binary cores
  remain the next persistence optimization.

These gates are deliberately local. Passing them gives us a stable query
contract to transport; it does not by itself make the system a remote cluster.

## Current evidence

A previous 500-question run of `segment_scoped_benchmark` on the checked-in
LongMemEval-S data used an average of 47.7 segments per question and
`top_k=5`. It remains diagnostic evidence, not a release result:

| Query scope | Recall@5 | MRR | Mean latency |
| --- | ---: | ---: | ---: |
| Top 1 routed segment | 41.3% | 65.4% | 2.83 ms |
| Top 3 routed segments | 70.9% | 79.9% | 4.33 ms |
| Top 5 routed segments | 79.5% | 82.8% | 6.07 ms |
| All segmented partitions | 83.5% | 84.6% | 24.36 ms |

The result supports keeping `query_top_n` configurable and exposing routing
misses. It does not support silently treating three segments as equivalent to
full-corpus retrieval. The specialized enrichment variants were slower at
roughly 9–10 ms without improving top-3 recall, while connected enrichment
was roughly 67 ms and reduced recall; they should remain opt-in until a
larger quality case justifies their cost. The candidate multiplier should also
remain a measured tuning parameter rather than a permanent promise.

The current implementation was also compared with a clean HEAD checkout using
the same five-query slice. After removing repeated BM25-statistics scans, the
mean routed latency was:

| Query scope | Clean HEAD | Current | Current recall@5 |
| --- | ---: | ---: | ---: |
| Top 1 routed segment | 0.79 ms | 1.14 ms | 80% |
| Top 5 routed segments | 0.74 ms | 0.90 ms | 80% |

This confirms a remaining structural fan-out cost for one selected segment. It
is tracked as a maturity gap; no compatibility global index is being restored
to hide it. The benchmark must be rerun on a representative corpus before
declaring a performance target or enabling remote fan-out by default.

## Remote extension

### Two-tier topology

The intended deployment has two distinct levels of distribution:

```text
                    lint-service coordinator
                 discovery / deadline / reduce
                       /         |         \
                      /          |          \
             lint-ai worker A  lint-ai worker B  lint-ai worker C
                 local                local              local
             coordinator            coordinator          coordinator
             route -> reduce        route -> reduce      route -> reduce
              / | \                  / | \                / | \
           seg seg seg             seg seg seg           seg seg seg
```

Each worker owns one corpus and may use the local segmented layout. The remote
coordinator owns worker selection and cross-corpus reduction; it must not assume
that a worker's local segment IDs or local index generation are globally unique.

### Current `lint-service` compatibility

The existing service is useful transport infrastructure, but its current
`RunLint` protocol is a generic command-dispatch protocol:

- a request contains CLI `args`, `working_dir`, and `timeout_ms`;
- a worker launches `lint-ai` and returns stdout/stderr and an exit code;
- the dispatcher extracts JSON `results` from stdout and merges local `score`
  values;
- worker failures and deadlines are represented as unavailable worker results.

This supports operational fan-out and partial results, but it is not yet a
correct implementation of the typed query contract above. In particular, local
BM25 scores from different worker corpora are not guaranteed comparable, and
the response does not carry query-scoped statistics, snapshot generations,
segment completeness, or a stable result identity contract. Those are correctness
gaps, not merely optimization opportunities.

### Required remote query contract

The next transport revision should carry a versioned structured request and
response rather than scrape CLI output. At minimum:

```text
Request:  query, top_k, filters, temporal context, reference date,
          coordinator snapshot/generation, statistics contract version,
          deadline and request ID
Response: worker ID, worker snapshot generation, statistics generation,
          results with stable document IDs and score components,
          expected/successful/failed segment or worker IDs, timings
```

The remote reducer must reject or explicitly mark results whose statistics
generation is incompatible, preserve partial-result completeness, and apply
deterministic tie-breaking after merging. The worker-facing endpoint can be
gRPC or HTTP; the semantic contract is independent of that choice.

Until that protocol exists, `lint-service` should be described as generic CLI
federation rather than as a fully consistent distributed `MemoryIndex`.

The concrete wire contract and acceptance criteria are in
[`remote-query-contract.md`](remote-query-contract.md).
