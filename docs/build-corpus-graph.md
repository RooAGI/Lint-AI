# Build Corpus Graph

The corpus graph stage converts a set of Markdown files into a structured representation that later checks can reason over.

## Inputs
- Root path (repo or docs directory)
- Limits (`max_bytes`, `max_files`, `max_depth`, `max_total_bytes`)
- Optional single-file mode

## File discovery
- Walk the target directory recursively (respecting depth/file limits)
- Keep only `.md` files
- Skip oversized files and stop on global byte/file caps

## Per-document extraction
For each Markdown file:
- `rel_path`: path relative to scan root
- `concept`: normalized concept derived from filename stem
- `raw_concept`: original filename stem
- `content`: full file text
- `links`: normalized outbound links from:
  - wiki links `[[...]]`
  - markdown links `[...](...)`
- `headings`: extracted Markdown headings

## Tier 0 side output
For each processed document, also create a Tier 0 record:
- `id`, `source`, `timestamp`, `author_agent`, `doc_length`, `metadata`

## Graph construction
- Create one node per document concept
- Add directed edges from document A -> document B when A links to B's concept
- Store:
  - `pages` (all parsed docs)
  - `index` (concept -> node index)
  - `graph` (directed link graph)
  - `tier0_records` (ingestion records)

## Output
A `Graph` object that is the canonical in-memory corpus model used by:
- orphan/unreachable checks
- cross-reference checks
- analysis/debug modes
- Tier 0 export paths
