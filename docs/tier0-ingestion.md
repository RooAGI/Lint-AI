# Tier 0: Ingestion Only

## Goal
Collect a stable, low-cost record for every processed document.

Tier 0 does not perform semantic reasoning. It only captures ingestion metadata that later tiers can build on.

## Required Fields Per Document
- `id`
- `source`
- `timestamp`
- `author/agent`
- `doc_length`
- `basic metadata`

## Current Record Shape
`Tier0Record` includes:
- `id`: stable document identifier (currently relative path)
- `source`: source document path (relative to scan root)
- `timestamp`: last modified time (unix seconds string, when available)
- `author_agent`: extracted from frontmatter fields such as `author`, `agent`, `author_agent`, `created_by`
- `doc_length`: document size in bytes
- `metadata`: lightweight document metadata map

### Basic Metadata (Current)
- `concept`
- `raw_concept`
- `file_ext`
- `heading_count`
- `outbound_link_count`
- `path`
- `file_size_bytes`
- frontmatter presence flags (when present)

## Linking Metadata to Documents
Tier 0 links metadata to documents through `id` and `source`.
Both map back to the processed document path.

## Tier 0 Index File
Use the CLI to write a persistent index:

```bash
./target/debug/lint-ai /path/to/repo --tier0-index-out
```

Default output file:
- `tier0-index.json`

Custom output path:

```bash
./target/debug/lint-ai /path/to/repo --tier0-index-out out/my-index
```

If the provided output path has no extension, `.json` is appended automatically.

## Index Structure
The index file includes:
- `tier`
- `generated_at_unix`
- `path`
- `document_count`
- `documents_by_id` (`id -> Tier0Record`)

This provides a stable lookup table for Tier 1+ pipelines.
