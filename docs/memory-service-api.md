# MemoryService API boundary

`MemoryService` is the sole public memory access API. Applications, MCP
adapters, hooks, and HTTP handlers use its methods. The public Rust surface
also includes its request, response, document, configuration, and inspection
types. These data types do not provide independent storage access.

## Public operations

- Construct with `MemoryService::in_memory(options)` or `MemoryService::at_path(path, options)`.
- Use `add`, `add_batch`, `search`, `get`, `list`, `update`, `delete`, `supersede`, and `expire` for memories.
- Use the service's `board_*` methods for boards.
- For source documents, use `upsert` and `refresh`, then `search_with_filters`.
  Use `enrich_source_document(s)` for validated, evidence-backed additions to
  existing documents. Enrichment checks the reviewed source hash and stages
  the complete batch through the service before publishing the updated search
  view.
- Use `inspection` for structural diagnostics without obtaining an index handle.

Request and response types are available from the crate root and from
`membrick::memory_api`. Configuration and document types are available from
the crate root, for example `membrick::{PipelineOptions, Lang, SourceDocument}`.

## Source-document updates and index work

`MemoryService::upsert` inserts or replaces one `SourceDocument` by `doc_id`.
It applies the service's common key-phrase enrichment scheduling and then
updates the service-owned store. A caller that needs an immediately visible
search view must publish through the service (`refresh` for an in-memory
caller, or the resident staged writer for durable workspace operations).

For reviewed provider changes, use `enrich_source_document(s)` instead of
constructing a replacement directly. It checks the expected source hash and
the full batch before applying any update, merges supported additions into the
current source, and stages the changed documents through `MemoryService`.
`enrich_doc` uses this same service method through the resident writer.
`search_feedback` persists an overall search label and explanation together
with a relevance judgment for every hit in the immutable recorded result. The
service resolves IDs, original ranks, and source hashes from that snapshot;
judgments are evaluation data and do not mutate documents or train ranking.
Supported grounded query-rule additions remain a separate optional part of
the feedback request.
Provider adapters do not open or mutate `IndexStore` themselves.

Enriching one source document does not rescan or reingest the project, and it
does not replace the original observation. The changed source is run through
the existing document-to-record pipeline, then the service publishes the
affected search structures. Segmented indexes can reuse unaffected segments;
shared statistics or other index layouts can require broader index work. The
published generation remains consistent while that work runs. Query-only rule
changes use the workspace service's reloadable rules snapshot and do not
rebuild document records.

## Memory identity and compatibility

Memory IDs are opaque. Obtain them through `get`, `list`, or `search` instead
of constructing them from user and request identifiers. New `add` memories use
SHA-256 over a versioned, length-delimited identity. Case and whitespace in
identifiers are significant. Conflicting existing documents are rejected before
the request writes any messages or supersessions.

Existing stored IDs remain valid. Reopening a store reconstructs request
fingerprints from its documents, so retrying an existing request preserves its
original IDs without creating duplicates. This change does not recover memories
already overwritten by the previous ID scheme.

## Internal implementation

`IndexStore`, `MemoryIndex`, snapshots, segment indexes, builders, persistence
paths, and writer locks are private to the crate. The service owns the mutable
store and snapshot lifecycle. Its internal workspace implementation owns shared
store selection, cross-process write locking, composition, and synchronization.
Adapters use service methods to invoke those operations. The MCP
`review_search`, `search_feedback`, and `enrich_doc` tools use the same boundary:
the service selects recorded searches, persists overall labels and per-document
relevance judgments, validates
evidence-backed document updates, and applies supported query-rule changes. Adapter code does
not write index records or rule state directly. Workspace query-rule overlays
are owned by the workspace `MemoryService`; validated query-only edits reload
into that service's immutable rules snapshot without restarting it.

This visibility change does not merge the workspace and agent-memory stores,
change the on-disk format, or make the separate persistence steps atomic.

## Migrating Rust callers

This is a breaking Rust API change. Direct imports from `index`, `pipeline`,
`segments`, and other implementation modules no longer compile. Imports of
`IndexStore`, `MemoryIndex`, `MemoryIndexSnapshot`, and the `build_*` index
functions from the crate root are also unavailable.

Replace direct index construction and queries with `MemoryService`. Import
supporting data types from the crate root. Tests that require raw indexes now
live inside the crate; external integration tests exercise the service API.
Compile-fail documentation tests enforce the visibility boundary.

Bundled executables retain their existing names and command-line arguments.
Their implementations compile as private library modules. Thin binary launchers
call a hidden process entry point that returns no service, store, or index
handles. Development benchmarks can inspect internals without making those
internals part of the public memory API.
