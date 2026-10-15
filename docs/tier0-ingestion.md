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

## Persistent Index
Build or refresh the project's persistent index with the current CLI:

```bash
membrick index /path/to/repo
```

The current CLI stores index data under the project's `.membrick` directory.
The older `--tier0-index-out` JSON export switch is hidden for compatibility
and is not part of the supported command-line workflow.

## Legacy Tier 0 export format
Older standalone Tier 0 JSON exports include:
- `tier`
- `generated_at_unix`
- `path`
- `document_count`
- `documents_by_id` (`id -> Tier0Record`)

This provides a stable lookup table for Tier 1+ pipelines.
