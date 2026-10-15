# Show Headings

This page describes the legacy heading diagnostic. Its `--show-headings`
switch is hidden from normal CLI help and retained only for compatibility; it
is not part of the supported command-line workflow. Use `membrick index` to
build the current project index.

## Goal
Provide a quick structural view of the corpus so you can inspect document organization before semantic checks.

## How it works
1. Discover and parse Markdown files using the normal corpus build path.
2. For each document, extract headings in document order.
- Primary path: regex-based heading extraction from Markdown (`#` to `######`).
- Fallback path: Markdown AST parsing when no headings are found via regex.
3. Emit per-document output:
- document relative path
- list of extracted headings

## Output format
Printed to stdout as blocks per file:

```text
docs/channels/discord.md
- Overview
- Setup
- Configuration
- Troubleshooting
```

If a document has no explicit heading text, fallback placeholders may appear for structural continuity.

## Inputs that affect output
- corpus path and scan limits (`max_depth`, `max_files`, `max_bytes`, `max_total_bytes`)
- ignore-path filtering from config
- Markdown content quality/format

## Typical use cases
- Audit documentation structure consistency.
- Find pages missing expected sections.
- Validate section naming during documentation maintenance.
