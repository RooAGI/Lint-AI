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
- Use `inspection` for structural diagnostics without obtaining an index handle.

Request and response types are available from the crate root and from
`lint_ai::memory_api`. Configuration and document types are available from
the crate root, for example `lint_ai::{PipelineOptions, Lang, SourceDocument}`.

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
Adapters use service methods to invoke those operations.

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
