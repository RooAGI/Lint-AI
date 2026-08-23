# Distributed Segment Index

Status: design proposal

`SegmentedMemoryIndex` today is a single-process structure, but its shape is
already that of a sharded index: each segment is a self-contained
`MemoryIndex` with its own tantivy index, and routing selects which segments a
query touches. This document describes what has to change for segments to live
on different machines, and — just as importantly — which of the current
"duplication" is load-bearing and must not be refactored away.

It does not describe an implemented change.

## What the current design already gets right

Three properties matter for distribution and are already true.

**Segments are self-contained.** `MemoryIndexSegment` (`src/segments.rs:40`)
owns a full `MemoryIndex`, which owns its own tantivy index. A segment can be
built, queried, and scored without reference to any other segment. That is what
makes a segment relocatable at all.

**Routing needs metadata, not documents.** `SegmentCorpusStats`
(`src/segments.rs:3460`) is built from segment *profiles* — term, entity, topic,
and local-memory keys — not from document bodies. `term_segment_counts` answers
"how many segments contain this term", and `idf()` derives a segment-level IDF
from it. A coordinator can therefore route a query while holding only compact
per-segment summaries. This is the single most important property for a
distributed design and it would have been expensive to retrofit.

**Aggregation is already two-tier.** Every segment query runs a full
`MemoryIndex` query (`src/segments.rs:1904`), which internally groups candidates
and applies `aggregate_group_score` (`src/index.rs:2235`). The results merged
across segments are then aggregated again by
`aggregate_segment_results_by_session` (`src/segments.rs:2028`). These are not
duplicate implementations of one policy: they are the shard-local phase and the
coordinator reduce phase of a query-then-fetch pipeline.

> **Do not consolidate the two aggregators into one shared policy.** An earlier
> proposal (`docs/query-context-and-ranking-consolidation.md`) reads them as
> accidental duplication. Under a distributed design they are distinct tiers
> with distinct jobs, and collapsing them would erase the split that
> distribution requires. What should be shared is a defined two-phase
> *contract*, not a single function.

## Tiers

```text
                    ┌──────────────────────────────┐
                    │        coordinator           │
                    │  - segment profiles          │
                    │  - GlobalCorpusStats         │
                    │  - routing                   │
                    │  - reduce (session aggregate)│
                    └──────────────────────────────┘
                       │            │            │
              ┌────────┘            │            └────────┐
              ▼                     ▼                     ▼
      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
      │  shard A     │      │  shard B     │      │  shard C     │
      │  segments…   │      │  segments…   │      │  segments…   │
      │  local BM25  │      │  local BM25  │      │  local BM25  │
      │  + postings  │      │  + postings  │      │  + postings  │
      │  + group agg │      │  + group agg │      │  + group agg │
      └──────────────┘      └──────────────┘      └──────────────┘
```

The coordinator holds only metadata. Documents live on shards and stay there;
a query returns scored results, not corpora.

## Problem 1: cross-shard score comparability

Each segment's tantivy index computes BM25 against **its own** corpus
statistics: document frequency and average document length over that segment
alone. A term that is rare across the corpus but common within one segment is
weighted very differently by different shards, so raw scores from two shards are
not on the same scale.

Today this is partly masked. `normalize_positive_score`
(`src/segments.rs:2251`) divides by the maximum score within a result set, which
makes values comparable in magnitude but does not make them *calibrated* — a
weak result from a shard whose best hit was also weak is scaled up to 1.0
exactly like a strong result from a strong shard.

This is the classic distributed-search problem, and it is solved by giving every
shard the same global term statistics rather than by post-hoc normalization.

### Tantivy already supports this

`tantivy::query::Bm25StatisticsProvider` exists in the pinned version (0.25) and
is exactly the required hook:

```rust
pub trait Bm25StatisticsProvider {
    fn total_num_tokens(&self, field: Field) -> tantivy::Result<u64>;
    fn total_num_docs(&self) -> tantivy::Result<u64>;
    fn doc_freq(&self, term: &Term) -> tantivy::Result<u64>;
}
```

`Searcher::search_with_statistics_provider(query, collector, provider)` runs a
search scoring against supplied statistics instead of the local index's own.
`Searcher` implements the trait for itself, which is what the ordinary
`search()` path uses.

No tantivy upgrade is required to adopt this. (Note that tantivy 0.26 is
currently *not* recommended for unrelated reasons — see
`docs/query-context-and-ranking-consolidation.md` and the 0.26 tie-breaking
change.)

### Proposed: `GlobalCorpusStats`

Introduce a corpus-statistics object that is authoritative across all shards:

```rust
pub struct GlobalCorpusStats {
    total_num_docs: u64,
    total_num_tokens: HashMap<Field, u64>,
    doc_freq: HashMap<Term, u64>,
}

impl Bm25StatisticsProvider for GlobalCorpusStats { /* … */ }
```

Shard-local scoring then calls `search_with_statistics_provider` with this
object rather than the local searcher, making BM25 scores from every shard
directly comparable without normalization tricks.

Two things to settle during implementation:

- **Size.** `doc_freq` is keyed by term and grows with corpus vocabulary, not
  with document count, but it is still the largest object the coordinator
  distributes. It is a candidate for a sketch (count-min) if exactness proves
  unnecessary — that trade should be measured, not assumed.
- **Freshness.** Statistics drift as shards ingest. A stale snapshot degrades
  ranking gradually rather than breaking it, so periodic refresh is likely
  sufficient; strict consistency is probably not worth its cost. Whatever is
  chosen, the staleness window should be explicit and observable.

## Problem 2: `global_index` does not survive distribution

`SegmentedMemoryIndex.global_index: Option<MemoryIndex>` (`src/segments.rs:47`)
holds **every document in the corpus**, in addition to each segment holding its
own copy. On one machine that is redundant work; across machines it is
incoherent, because it would require replicating the entire corpus to the
coordinator — precisely what sharding exists to avoid.

It is load-bearing today: `global_index()` has 11 call sites across
`src/pipeline.rs` and `src/segments.rs`, serving two distinct purposes that
should be separated.

| Current use | Distributed replacement |
| --- | --- |
| Whole-corpus and filtered queries that bypass routing (`pipeline.rs`) | Route to all shards and reduce, i.e. a fan-out query rather than a local one |
| Fallback when routing selects nothing (`segments.rs:1008`, `1018`) | An explicit "route to all shards" policy, not a second index |
| *(implicitly)* a source of corpus-wide statistics | `GlobalCorpusStats` (Problem 1) |

The third row is the important one, and it is why Problems 1 and 2 are the same
problem: the reason a full global index is tempting is that it is the only thing
that currently knows corpus-wide statistics. Extracting `GlobalCorpusStats`
removes that reason. What remains of `global_index` is a query convenience,
which fan-out replaces.

**Proposed end state:** `global_index` is removed. The coordinator holds
`GlobalCorpusStats` plus segment profiles; any query that would have hit the
global index becomes a fan-out to all shards followed by the normal reduce.

**Migration note.** This is a larger change than it looks, because
`IndexStore` currently treats `global_index()` as its primary query surface
even in the single-segment case. A staged path is to introduce
`GlobalCorpusStats` first (a pure addition that improves single-process
scoring consistency on its own merits), then convert the fallback paths to
fan-out, then remove the field.

## Problem 3: shard access is assumed to be local, synchronous, and infallible

This one is independent of the first two, and it is the one that changes API
signatures.

Every segment query today is a direct method call:

```rust
segment.index.query_with_temporal_context(&enrichment.enriched_query, top_k, temporal)
```

It cannot fail, cannot time out, and returns no indication of completeness.
Over a network all three change. The design must state, and the types must
express, answers to:

- **Partial results.** If one shard of five is unreachable, is the query an
  error, or a success over four shards? Memory recall usually prefers the
  latter, but silently returning fewer results makes ranking quality
  unreproducible and hides outages. Recommendation: return results *plus* an
  explicit completeness marker (which segments answered, which were skipped),
  and let callers decide. Benchmarks must record it, or a degraded cluster will
  look like a ranking regression.
- **Timeouts and stragglers.** A per-shard deadline with a coordinator-side
  budget, so one slow shard cannot stall a query. Interacts with the previous
  point: a timed-out shard is a missing shard.
- **Fallibility in the signature.** Shard-local query calls become
  `Result`-returning and likely `async`. This ripples through `segments.rs`
  broadly, so it is best done deliberately rather than as a side effect of the
  ranking work.
- **Retries and idempotency.** Queries are read-only, so retry is safe; the
  budget for it belongs with the timeout policy.

Ingestion raises its own questions (which shard owns a new document, how
segments split when they grow, whether documents ever move) that are out of
scope here and deserve a separate document.

## Consequences for work currently in flight

1. **The ranking consolidation in
   `docs/query-context-and-ranking-consolidation.md` should not proceed as
   written.** Its core proposal — one shared `ranking::aggregate_groups`
   replacing both aggregators — conflicts with the two-tier split this design
   depends on. The query-*preparation* half of that document (shared
   `PreparedQuery` across CLI and memory server) is unaffected and remains
   valid.

2. **The two aggregation formulas should be documented as tiers, not
   reconciled.** A measured comparison on the multi-session slice (133 queries,
   `coverage-local` router, top-N 5) found that swapping the coordinator-tier
   formula for the shard-tier one *lost* 0.34pp MRR and 0.09pp NDCG@10, with
   recall unchanged. The tiers are doing different jobs and are tuned
   differently; that is the intended state, not drift.

3. **`GlobalCorpusStats` is worth building before distribution.** It improves
   score consistency in the current single-process segmented mode too, where
   per-segment IDF already makes cross-segment scores incomparable. It can be
   benchmarked immediately with the existing `segment_scoped_benchmark`.

## Open questions

- Does the coordinator hold segment profiles authoritatively, or fetch them
  from shards on a schedule?
- How are segments assigned to shards, and can a segment move without a
  reindex?
- Is there a replication story, or is a lost shard a lost partition?
- Does `SegmentRoutingStrategy` stay coordinator-side only, or do shards
  participate in routing decisions about their own segments?
- What is the consistency contract between a write being acknowledged and it
  becoming visible to a routed query?
