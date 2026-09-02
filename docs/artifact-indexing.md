# Artifact Indexing and Update Model

This document explains how `lint-ai` can be used from an artifact-oriented
system, where each fetch creates or updates an artifact record that should later
be queried through the retrieval layer.

## What Is Implemented Now

The library now supports a practical artifact flow with:

- `IndexStore` as the public mutable artifact-facing facade
- `MemoryIndex` as the frozen built query structure

Implemented pieces:

- `SourceDocument`
  - generic input document type for non-Markdown sources
- `PipelineOptions`
  - library-facing configuration for chunking and enrichment
- `build_index_store(...)`
  - builder for the public mutable `IndexStore`
- `build_query_snapshot(...)`
  - compatibility-named builder for the frozen search snapshot
- `IndexStore`
  - public mutable store that owns source docs, cached records, tombstones,
    an internal Tantivy lexical index, and the current built semantic snapshot

Relevant API surface:

- `src/source.rs`
  - `SourceDocument`
- `src/pipeline.rs`
  - `PipelineOptions`
  - `ChunkStrategy`
  - `Tier1NerProvider`
  - `Tier1TermRankerKind`
  - `build_index_store(...)`
  - `build_query_snapshot(...)`
  - `build_query_snapshot_from_source_documents(...)`
  - `IndexStore`
- `src/index.rs`
  - `MemoryIndex`
  - `SearchResult`

Top-level re-exports are also available from `lint_ai` through `src/lib.rs`.

## Current Library Flow

The intended artifact flow is:

1. Fetch or load an artifact in the host application.
2. Extract normalized text and metadata from that artifact.
3. Convert it into `SourceDocument`.
4. Internally derive or update a cached `DocRecord`.
5. Insert or update it inside `IndexStore`.
6. Query through the current snapshot.

Example:

```rust
use lint_ai::{IndexStore, PipelineOptions, SourceDocument};

let mut index = IndexStore::new(PipelineOptions::default());

index.upsert(SourceDocument {
    doc_id: "artifact-123".to_string(),
    source: "s3://bucket/report.pdf".to_string(),
    content: "normalized extracted text".to_string(),
    concept: "quarterly report".to_string(),
    headings: vec!["Summary".to_string(), "Financials".to_string()],
    links: vec![],
    timestamp: None,
    doc_length: 24,
    author_agent: None,
});

let results = index.query("financial summary", 5)?;
# Ok::<(), anyhow::Error>(())
```

## Artifact to SourceDocument Mapping

Each artifact should be normalized into one `SourceDocument`.

Recommended mapping:

- `doc_id`
  - stable artifact id
- `source`
  - source URI, file path, or artifact key
- `content`
  - extracted plain text
- `concept`
  - canonical label, title, or filename-derived subject
- `headings`
  - optional structure extracted from the artifact
- `links`
  - optional references to related documents or artifacts
- `timestamp`
  - optional creation or fetch time
- `doc_length`
  - extracted text length
- `author_agent`
  - optional provenance

Example:

```rust
use lint_ai::SourceDocument;

let doc = SourceDocument {
    doc_id: "artifact-123".to_string(),
    source: "s3://bucket/report.pdf".to_string(),
    content: "normalized extracted text".to_string(),
    concept: "quarterly report".to_string(),
    headings: vec!["Summary".to_string(), "Financials".to_string()],
    links: vec![],
    timestamp: None,
    doc_length: 24,
    author_agent: None,
};
```

## Storage Model

The preferred storage model is:

- one corpus root
- one local `.lint-ai/` index root under that corpus

If the corpus root is:

```text
/path/to/corpus
```

then the default index root should be:

```text
/path/to/corpus/.lint-ai/
```

Suggested layout:

```text
.lint-ai/
  lexical/
  semantic/
  metadata.json
```

Where:

- `lexical/`
  - Tantivy lexical index
- `semantic/`
  - persisted semantic records and binary core
- `metadata.json`
  - index schema/layout metadata for validation and migration checks

This keeps index ownership local to the corpus and avoids guessing unrelated
global paths.

## Index Location Modes

Conceptually, the library should support these modes:

- in-memory
  - no filesystem state
- under corpus root
  - default persistent mode using `.lint-ai/`
- explicit path
  - caller provides an explicit index root

Current API surface:

- `IndexStore::in_memory(options)`
- `IndexStore::for_corpus(corpus_root, options)`
- `IndexStore::at_path(index_root, options)`
- `resolve_store_paths(corpus_root, options)`

The intended rule is:

- default to `InMemory` when no persistent root is available
- default to `UnderCorpusRoot` when indexing a filesystem corpus
- only use a shared or external index root when explicitly configured

Index creation and opening should be centralized in one place. The code should
not create Tantivy indexes independently in multiple files or subsystems.

## Multi-Root Operation

If there are multiple corpus roots or multiple index locations, they should not
implicitly mutate one shared physical index directory.

Instead, the clean model is:

- one corpus root -> one `IndexStore`
- many corpus roots -> many `IndexStore`s
- query all stores and merge results

This is effectively a sharded query model.

Example:

```text
IndexStore A -> results A
IndexStore B -> results B
IndexStore C -> results C
merge/rerank -> final results
```

This is safer and simpler than trying to make several unrelated filesystem
locations behave as one writable Tantivy index.

## Explicit Global Index Root

If one logical global index is desired, it should be explicit.

For example:

```text
/var/lib/lint-ai/indexes/my-index/
```

with:

```text
/var/lib/lint-ai/indexes/my-index/lexical/
/var/lib/lint-ai/indexes/my-index/semantic/
/var/lib/lint-ai/indexes/my-index/metadata.json
```

This should be an explicit configuration choice, not something inferred from
multiple input locations.

## Versioning and Migration Policy

Persistent `IndexStore` roots now use `metadata.json` as a minimal formal
versioning contract.

Current metadata includes:

- `schema_version`
- `layout_version`
- `crate_version`
- `index_location`

Behavior:

- when a persistent store is opened, existing metadata is loaded and validated
- schema or layout mismatches cause store initialization to fail explicitly
- when a persistent store is created or refreshed, metadata is written back to
  disk

This is a minimal migration policy, not a full migration framework. It does,
however, make schema/layout incompatibilities explicit instead of relying only
on path conventions.

## Internal Layering

The current internal layering is:

- `SourceDocument`
  - public normalized input document
- `DocRecord`
  - internal per-document indexed artifact
- `MemoryIndex`
  - built whole-corpus semantic query structure
- `IndexStore`
  - mutable orchestration layer over source documents, cached `DocRecord`s,
    tombstones, an internal Tantivy lexical backend, and the current built
    `MemoryIndex`

This means the “indexed document” layer already exists in practice today:
`DocRecord` is that layer.

From the user-facing API perspective:

- users provide `SourceDocument`
- the library internally derives and caches `DocRecord`
- the library builds or refreshes `MemoryIndex` from cached `DocRecord`s

`DocRecord` is therefore an internal document-level indexing artifact, not the
primary user-facing ingestion contract.

## How IndexStore Works Today

`IndexStore` is the public mutable artifact-facing index.

In the current design, `refresh()` is the boundary where mutable document state
is turned into the immutable queryable snapshot.

It stores:

- `PipelineOptions`
- `HashMap<String, SourceDocument>`
- `HashMap<String, DocRecord>`
- tombstone set for removed document ids
- internal Tantivy BM25 state for lexical upsert/delete/search
- optional `MemoryIndex`
- dirty flag

Current supported operations:

- `new(options)`
- `with_documents(options, docs)`
- `upsert(doc)`
- `remove(doc_id)`
- `refresh()`
- `query(query, top_k)`
- `len()`
- `is_empty()`
- `is_dirty()`
- `tombstones()`
- `source_documents()`
- `records()`

Behavior:

- `upsert` replaces or inserts a `SourceDocument`
- `remove` deletes a document from the mutable source set and records a tombstone
- `refresh` converts the current mutable state into a rebuilt immutable
  semantic `MemoryIndex` if the index is dirty
- `query` is the single public query path and uses a fresh semantic snapshot
- unchanged documents reuse cached `DocRecord`s during refresh
- `IndexLocation` determines whether Tantivy is in-memory, corpus-local, or
  under an explicit index root

This means the write model is now mutable at the API layer, even though the
underlying query index is still rebuilt in batch.

Another way to think about it:

- `upsert/remove` edit the working set
- `refresh()` updates Tantivy incrementally for lexical state and compiles the
  semantic working set into the current searchable snapshot
- `query()` combines Tantivy lexical hits with a fresh semantic snapshot

## Chunk Lifecycle Metadata

Chunk lifecycle is now tracked separately from raw chunk content.

Design intent:

- keep `SectionChunk` as content/span identity
- keep lifecycle/versioning in a separate metadata layer
- allow history and lineage without polluting raw chunk payloads

Current metadata shape:

- `chunk_id`
- `doc_id`
- `lineage_key`
- `version`
- `is_latest`
- `supersedes_chunk_id`
- `updated_at_ms`
- `change_reason`

Persistence:

- lifecycle is written with semantic state under `.lint-ai/semantic/`
- currently persisted both:
  - inline in `records.json` as `chunk_lifecycle`
  - in a dedicated `chunk_lifecycle.json`

Transition behavior:

- on `upsert` + `refresh`, each rebuilt chunk is matched by deterministic
  lineage key (doc + heading + line span)
- if same `chunk_id` already exists:
  - lifecycle entry remains current (`is_latest = true`)
- if lineage exists but `chunk_id` changed:
  - prior latest is marked `is_latest = false`
  - new lifecycle entry is created with `version = old + 1`
  - `supersedes_chunk_id` points to prior latest chunk id
- if lineage is new:
  - new lifecycle entry starts at `version = 1`, `is_latest = true`
- on `remove(doc_id)`:
  - lifecycle entries for that doc are removed from the active store

Current query semantics:

- snapshots query current chunks only
- lifecycle metadata is used for evolution tracking and auditability, not for
  historical chunk retrieval by default
- historical chunk metadata remains available, but old chunk content is not
  part of active `DocRecord.section_chunks`

## Temporal Fact / Assertion Layer

Lifecycle metadata is enough to track how documents and chunks change over
time. It is not enough to answer "what was true when" questions on its own.

The next layer should therefore be a separate temporal fact store that is
derived from chunks, but not mixed into raw chunk payloads.

Recommended role separation:

- `SectionChunk.timestamp`
  - source/event time for the chunk
- `ChunkLifecycleMeta.updated_at_ms`
  - when the chunk version changed in the index
- `DocumentLifecycleMeta.updated_at_ms`
  - when the document version changed in the index
- `TemporalFact`
  - semantic assertion layer with validity intervals

Suggested `TemporalFact` shape:

- `fact_id`
- `subject`
- `predicate`
- `object` or `value`
- `unit`
- `scope`
- `valid_from`
- `valid_to`
- `source_doc_id`
- `source_chunk_id`
- `source_chunk_version`
- `chunk_timestamp`
- `confidence`
- `is_latest`

How it should be used:

1. Derive facts from chunks during indexing.
2. Keep provenance back to the source chunk and document.
3. Version facts when later evidence changes the asserted state.
4. Query the fact layer for:
   - `as_of(date)`
   - `between(start, end)`
   - `timeline(subject)`
   - `diff(date1, date2)`

What this is not:

- it is not another chunking layer
- it is not a replacement for `DocRecord`
- it is not a reason to store `tcommit`-style graph jargon in the public API

## Markdown supersession

When a Markdown document replaces an earlier document whose wording or
timestamp is not enough to establish the relationship, declare the relationship
in frontmatter:

```yaml
---
supersedes: decisions/legacy-routing.md
---
```

The referenced document remains in the indexed corpus for historical inspection,
but default project searches omit it and return the replacement instead. This
explicit relationship is intentionally preferred over guessing semantic
replacement between unrelated passages.

Implementation direction:

- store facts in the existing memory store as a dedicated collection or
  namespace
- keep the document/chunk lifecycle model intact
- use the fact layer only for temporal state reasoning, not for ordinary
  retrieval

## Current Drawback of the Implemented Approach

The current design is intentionally conservative.

Main tradeoff:

- update semantics are incremental
- lexical updates are incremental, but semantic index construction is still
  batch-oriented

In practice:

- one changed artifact updates Tantivy incrementally, but can still trigger a
  full semantic `MemoryIndex` rebuild on next refresh or query
- the first query after updates may pay the rebuild cost
- the library keeps both source documents and an immutable snapshot in memory

This is acceptable for small and medium corpora and is much simpler than true
low-level incremental mutation of the frozen query structures.

## Why We Kept MemoryIndex Separate Internally

The internal `MemoryIndex` is optimized for fast batch construction and fast
queries. It uses compact global structures such as:

- flattened chunk metadata
- postings vectors
- tries
- interval trees
- lexical BM25 index

Those structures are efficient to build and query, but awkward to mutate
incrementally. Supporting in-place updates would require a deeper redesign of
query internals.

## What Is Still Missing

The current implementation is useful, but it is not yet a full incremental
indexing system.

Remaining gaps:

- true incremental semantic postings update
  - semantic `MemoryIndex` is still rebuilt from cached `DocRecord`s
- richer concurrent publication model
  - current background refresh is internal-only and simple rather than a full
  multi-reader / multi-writer snapshot system
  - see [IndexStore and MemoryIndex Concurrency](index-concurrency.md) for the
    recommended large-corpus publication model
- stronger freshness policy controls
  - current public API intentionally avoids exposing multiple refresh/query
    modes, and does not yet implement advanced scheduling or debouncing policies

## Semantic Persistence

Persistent stores now use `.lint-ai/semantic/` for semantic state.

Current files:

- `records.json`
  - persisted semantic document records
- `core.bin`
  - binary semantic core used to rebuild the semantic `MemoryIndex`

Behavior:

- on `refresh()`, semantic state is written into the resolved semantic path
- on persistent store initialization, semantic state is loaded from disk when
  compatible metadata and semantic files are present
- restored semantic records are also used to repopulate the lexical index

This means the documented `semantic/` path is now active, not just reserved.

## Snapshot as an Internal Concept

The library still uses the concept of a snapshot internally:

- mutable state exists inside public `IndexStore`
- Tantivy lexical state is updated incrementally inside `IndexStore`
- the queryable `MemoryIndex` is the frozen semantic search snapshot
- `refresh()` materializes that semantic snapshot from current mutable state

What is intentionally *not* introduced right now is a separate public snapshot
manager subsystem.

That is deliberate for two reasons:

- the current synchronous `refresh()` path is sufficient for the current scale
  and workflow
- introducing a dedicated snapshot manager now would add concurrency and
  lifecycle complexity without solving the harder low-level incremental indexing
  problem

So at this stage:

- snapshot is a real internal concept
- snapshot manager is not a required standalone abstraction yet

## Recommended Next Steps

### Phase 3

If scale demands it, redesign internals for true incremental updates:

- stable doc and chunk ids
- mutable postings maps
- incremental lexical indexing
- efficient remove and replace operations

This is a larger project and should only be pursued if performance data shows
the current snapshot model is too expensive.

Before that larger redesign, prefer the lighter concurrency path documented in
[IndexStore and MemoryIndex Concurrency](index-concurrency.md): one published
query generation plus at most one in-flight replacement generation per logical
`IndexStore`. A generation may be one `MemoryIndex` today or a segmented set of
`MemoryIndex` values in a future large-corpus design.

## Recommended Host Architecture

For an artifact system, the recommended architecture remains:

1. Artifact store
   - source of truth for fetched artifacts
2. Normalization layer
   - converts artifacts into `SourceDocument`
3. Mutable artifact index state
   - public `IndexStore`
4. Immutable query snapshot
   - internal `MemoryIndex`, produced by `refresh()`

This keeps responsibilities clear:

- artifact system owns fetching, storage, OCR, file parsing, and metadata
- `lint-ai` owns chunking, enrichment, indexing, and retrieval

## Summary

Current state:

- artifact-friendly document type is implemented
- library-facing pipeline config is implemented
- public mutable `IndexStore` with `upsert/remove/refresh/query` is implemented
- per-document `DocRecord` cache is implemented
- `DocRecord` already serves as the internal per-document indexed artifact layer
- tombstone tracking is implemented
- configurable lexical snapshot persistence is implemented
- `refresh()` is the mutable-to-immutable boundary
- internal `MemoryIndex` is still rebuilt in batch from cached records

So `lint-ai` now supports artifact ingestion in a practical way, but it does
not yet provide true low-level incremental indexing.
