# Show Concepts

`--show-concepts` prints a section-aware concept inventory from the corpus.

## Goal
Provide a fast view of which concepts are most prominent in which document sections.

## How it works
1. Build concept matcher from corpus pages
- Collect concepts from document filenames.
- Generate surface forms (case/spacing/plural variants).
- Remove ambiguous forms that map to multiple concepts.

2. Parse each document into sections
- Split content by Markdown headings.
- Track an implicit `unscoped` section for text outside headings.

3. Match concepts in section text
- Run phrase matching over normalized text.
- Ignore code blocks and code spans.
- Apply noise filters and config constraints.

4. Normalize section names
- Map heading variants to stable buckets (examples: `setup`, `configuration`, `related`, `troubleshooting`, `security`, `routing`, `overview`, `unscoped`).

5. Aggregate counts
- Count concept occurrences per normalized section across all documents.
- Sort by frequency, then name.

## Output format
Printed to stdout as grouped sections:

```text
Section: setup
- pairing (4)
- signal (3)

Section: configuration
- pairing (7)
- signal (6)
```

Each line means: concept appears in that section in N documents/section-matches (aggregated corpus-wide).

## Inputs that affect output
- corpus path
- config filters (`stopwords`, `ignore_sections`, `scope_prefix`, etc.)
- document content/headings

## Typical use cases
- Identify dominant concepts by section type.
- Detect noisy/global terms that should become stopwords.
- Validate section taxonomy quality before deeper alignment passes.
