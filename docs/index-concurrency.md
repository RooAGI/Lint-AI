# IndexStore and MemoryIndex Concurrency

This note captures the recommended concurrency model for large Lint-AI
deployments.

## Current Model

`IndexStore` is the mutable owner of the corpus state:

- source documents
- derived `DocRecord`s
- chunk lifecycle metadata
- temporal facts
- Tantivy lexical state
- the current semantic `MemoryIndex` snapshot

`MemoryIndex` is the immutable semantic query snapshot. It is optimized for
batch construction and fast reads over compact global structures. It is not the
right object to mutate incrementally in place.

Today, public queries call `refresh()` before searching. That gives strongest
freshness, but it also means the first query after a write can pay the full
semantic rebuild cost.

There is also a private background refresh scaffold. It preserves the old
snapshot while building a newer one and publishes the result only if no newer
write has advanced the store revision.

## Recommended Rule

There are two separate concurrency questions:

- snapshot publication: how many replacements for the currently queryable
  state should be built at the same time
- segmentation: whether one logical corpus should be represented by many
  smaller queryable `MemoryIndex` segments

For snapshot publication, build at most one replacement generation at a time.

In steady state, a store should have:

- one published query generation used by readers
- zero or one pending query generation for the next revision

If the store is not segmented, a query generation is one `MemoryIndex`. If the
store is segmented, a query generation is a set of `MemoryIndex` segments plus a
router/merge layer.

Avoid building many complete replacement generations for the same store. They
duplicate most of the corpus in memory, compete for CPU, and usually finish with
obsolete revisions when writes continue arriving.

## Segmented MemoryIndex Sets

For large corpora, the next useful architecture is likely an `IndexStore` that
can publish many smaller `MemoryIndex` segments as one logical search surface.

Examples:

- docs corpus: one segment per folder, product area, or bounded chunk range
- one giant document: split into chunks, then pack those chunks into bounded
  segments
- agent memory: one segment per session, or per session window, when session is
  the natural routing unit

This is different from creating many unrelated `IndexStore`s. The `IndexStore`
can still be the persistent owner of records, lifecycle metadata, and lexical
state, while the query layer owns a set of immutable segment snapshots.

A reasonable segment policy is:

- route-based segment when a stable key exists, such as folder, repo, session,
  tenant, language, or time window
- size-based segment when no stable key is small enough
- configurable default chunk/document limit per segment
- rebuild only the changed segment when possible
- publish the new segment set atomically as one generation

For a giant document split into chunks, the rough upper bound is:

```text
segment_count = ceil(total_chunks / memory_index_chunk_limit)
```

That segment count is not automatically the desired CPU concurrency. The build
concurrency should be capped separately by available cores, memory headroom, and
I/O pressure:

```text
build_parallelism = min(segment_count, configured_parallelism, resource_limit)
```

In other words, `memory_index_chunk_limit` controls segment size. A separate
parallelism limit controls how many segment builds can run at once.

The chunking/build pipeline is another level of parallelism. Its worker count
should normally be a small CPU-bound limit, for example four workers by default,
rather than the number of segments. That keeps the pipeline predictable:

```text
chunk_pipeline_workers = min(configured_workers, cpu_limit)
```

## Minimal Segmented Query Step

The smallest useful change is not a full sharded indexing system. It is the
ability to query one selected segment `MemoryIndex`, then compare its result
against the existing global query path.

Small units:

1. Define `MemoryIndexSegment`.
   - `segment_id`
   - `boundary`, such as folder, session, language, time window, or chunk range
   - `doc_ids`
   - one inner `MemoryIndex`
   - one compact `SegmentProfile`
2. Build profiles for existing records without changing query behavior.
3. Add a router that ranks segments for a query.
4. Query only the top segment.
5. Query the top N segments and merge top-k results.
6. Rebuild only the segment touched by an upsert/remove.
7. Publish the replacement segment set as one generation.

This gives a small benchmarkable ladder. Each step can be compared against the
current one-global-`MemoryIndex` baseline.

## Segment Routing

Routing should start with cheap sparse statistics that match the current
implementation.

`MemoryIndex` already derives:

- important terms
- key entities
- topics
- document type guesses
- temporal metadata
- group/session metadata
- sparse postings

A `SegmentProfile` can summarize these as normalized distributions:

- term distribution
- entity distribution
- topic distribution
- doc type distribution
- optional time range
- optional route keys, such as folder/session/language

The first router can use sparse similarity:

```text
score(segment, query) =
  lexical_overlap(query_terms, segment_terms)
  + entity_overlap(query_entities, segment_entities)
  + topic_match(query_topic, segment_topics)
  + route_key_bonus
  + temporal_window_bonus
```

For a more ML-style router, treat the query as a sparse distribution and compare
it with each segment profile. KL divergence can work if smoothing is handled
carefully:

```text
route_score = -KL(query_distribution || segment_distribution)
```

Jensen-Shannon divergence may be safer for early implementation because it is
symmetric and bounded:

```text
route_score = 1 - JS(query_distribution, segment_distribution)
```

K-means is useful at segment construction time when there is no obvious boundary
like folder or session. In that case, cluster chunks/documents by sparse
term/entity vectors, then build one `MemoryIndex` per cluster. For docs and
agent-memory corpora, explicit boundaries should be tried first because they are
easier to explain and update incrementally.

The router should keep recall safeguards:

- query at least one fallback segment when routing confidence is low
- query more segments for broad or ambiguous questions
- preserve a `query_all_segments` debug mode
- record segment routing diagnostics in benchmarks

## Freshness Modes

The useful public API shape is likely three query modes:

- `Fresh`: synchronously refresh before query; best for correctness-sensitive
  reads and small stores.
- `LatestAvailable`: query the current snapshot immediately and start or poll a
  background refresh when the store is dirty; best default for large stores.
- `SnapshotOnly`: query the current snapshot without starting refresh work; best
  for high-QPS serving paths where refresh is scheduled separately.

The existing `query()` method behaves like `Fresh`.

## When To Use Multiple IndexStores

Use multiple `IndexStore`s only when there is a stable routing boundary that
queries can exploit before search:

- tenant or organization
- product area
- repository
- project or workspace
- time partition, if most queries target a known window
- language, once multi-language support has separate analyzers or ranking rules

Do not shard only because the corpus is large. Sharding helps when it reduces
the candidate set for most queries. If every query must fan out to every shard,
the system pays extra orchestration and merge cost while losing global ranking
context.

When the desired behavior is "one corpus, many searchable pieces," prefer
segmented `MemoryIndex` sets inside one logical `IndexStore` before splitting
into many independent stores.

## Large Corpus Strategy

For a huge corpus, prefer this order:

1. Keep one `IndexStore` per routable corpus scope.
2. Add `MemoryIndex` segmentation inside that store when the corpus is too large
   for one snapshot.
3. Publish one query generation at a time, where a generation may contain many
   segments.
4. Rebuild only changed segments when possible.
5. Cap segment build parallelism separately from segment count.
6. Debounce rebuilds when many writes arrive together.
7. Add a scheduled refresh path for serving workloads.
8. Split into multiple `IndexStore`s only after query routing data shows most
   queries hit a subset.

This model keeps memory bounded and avoids rebuilding obsolete snapshots.

## Current Implementation Gap

The current code does not yet implement segmented `MemoryIndex` sets.

Current behavior:

- `SourceDocument` is the canonical ingestion unit.
- `build_doc_records()` processes source documents serially.
- `assemble_doc_record()` chunks one source document into `section_chunks`.
- `build_query_snapshot()` builds one `MemoryIndex` from all records passed to
  that call.
- `IndexStore::refresh()` rebuilds one semantic `MemoryIndex` snapshot from all
  current records in the store.
- group/session/folder identity is metadata, not a separate index boundary.

For the scoped haystack benchmark in this checkout, each question builds a
snapshot over that question's candidate sessions; each turn is a
`SourceDocument`, and session identity is carried in `group_id`.

## Implementation Direction

The next implementation step should be routing over segmented query generations.
Keep it smaller than a low-level incremental rewrite of `MemoryIndex`.

Useful pieces:

1. Add segment metadata/profile types without changing public query behavior.
2. Build one `MemoryIndex` per explicit boundary, such as folder or session.
3. Add a `SegmentRouter` that returns ranked segment ids plus diagnostics.
4. Query the top one segment and benchmark recall/latency.
5. Query top N segments and merge top-k results.
6. Add low-confidence fallback to the global index or all segments.
7. Add CPU-capped parallel segment builds.
8. Add stale-while-refresh publication for segment generations.

Useful controls:

- `memory_index_chunk_limit`: maximum chunk/doc payload per size-based segment
- `segment_build_workers`: CPU-capped worker count, usually defaulting near four
- `segment_query_top_n`: number of routed segments to query
- `segment_router_min_confidence`: threshold for broad fallback
- `segment_strategy`: folder, session, language, time, chunk range, or cluster

Useful metrics:

- segment count
- selected segment ids
- router confidence
- routed segment recall
- top-N merge recall
- segment build duration
- skipped obsolete builds
- snapshot age

True incremental semantic postings updates are still a larger redesign. They
should be driven by benchmark evidence that segmented rebuilds are still not
enough.
