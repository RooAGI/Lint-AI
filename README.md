# Lint AI

Lint-AI is a system for analyzing and aligning large corpora of AI-generated documentation.

As AI systems produce increasing amounts of documentation--task records, traces, logs, decisions, and reports--these artifacts often become inconsistent, outdated, or misaligned with each other. Lint-AI addresses this by treating documentation as a network of facts, rather than isolated text.

## How it works

Lint-AI processes documentation in several stages:

### 1. Fact Extraction
Extracts entities, concepts, and claims from each document.

### 2. Concept & Entity Resolution
Identifies when different documents refer to the same concept using different terms.

### 3. Fact Graph Construction
Builds a network of normalized facts with context such as:
- source document
- time
- confidence
- status (current, deprecated, proposed)

### 4. Misalignment Detection
Identifies potential issues such as:
- contradictions
- terminology drift
- scope conflicts
- unsupported claims
- missing required context

### 5. AI Review
Routes suspicious cases to an AI reviewer that verifies and explains the issue with context.

Instead of enforcing rigid templates, Lint-AI focuses on understanding and comparing what documents actually say, enabling systematic detection of misalignment at scale.

The result is a continuous alignment layer that helps ensure AI-generated work remains consistent, interpretable, and trustworthy over time.

## Why Lint-AI?

AI systems don't just produce outputs--they produce documentation about their work.

Over time, this creates a growing body of:
- task summaries
- decision notes
- traces and logs
- generated reports

Without alignment:
- terminology drifts
- definitions conflict
- outdated concepts persist
- claims become unsupported or inconsistent

Reading individual documents is not enough. The problem is **system-level consistency**.

Lint-AI addresses this by analyzing documentation collectively, not in isolation.

## Vision

As AI systems perform more work, they will continuously generate documentation describing their actions, decisions, and outputs.

Lint-AI aims to ensure that this growing body of AI-generated knowledge remains:
- consistent
- traceable
- interpretable
- aligned over time

## Under the hood

Lint-AI builds on techniques such as:
- concept extraction
- corpus-wide matching
- terminology analysis

These are used as part of a larger system for fact extraction and alignment reasoning.

## What Lint-AI is NOT

Lint-AI is not:
- a Markdown style checker
- a grammar tool
- a fixed template enforcer

It does not require documents to follow a rigid schema.

Instead, it focuses on understanding what documents *mean* and how they relate to each other.

## Usage

### How To Use

Run the linter against a docs directory:

```bash
./lint-ai /path/to/repo
```

### Tier 0 and Tier 1 Outputs

Show Tier 0 ingestion records:

```bash
./lint-ai /path/to/repo --show-tier0
```

Write a Tier 0 index JSON:

```bash
./lint-ai /path/to/repo --tier0-index-out
```

Show Tier 1 key entities:

```bash
./lint-ai /path/to/repo --show-tier1-entities
```

Use spaCy for Tier 1 entities (falls back to heuristic if unavailable):

```bash
./lint-ai /path/to/repo --show-tier1-entities --tier1-ner-provider spacy --spacy-model en_core_web_sm
```

Show Tier 1 important terms:

```bash
./lint-ai /path/to/repo --show-tier1-terms --tier1-term-ranker yake
```

Available term rankers:
- `yake`
- `rake`
- `cvalue`
- `textrank`

### Index and Query

Build and print the in-memory hybrid index:

```bash
./lint-ai --index /path/to/repo/docs
```

Query the corpus (index is built automatically behind the scenes):

```bash
./lint-ai --query "docker install linux" /path/to/repo/docs
```

The query pipeline uses hybrid scoring with:
- BM25 lexical scoring
- key-entity overlap
- important-term overlap
- topic/doc-type boosts when available
- score breakdown output for transparency

### Download Release Binaries

Download the latest release binary from the GitHub Releases page for this repo, then verify the checksum.

Release artifacts (v0.1.2):

```
lint-ai-linux-x86_64
sha256:7ec06e0ed69a2fa1c2acd55c5ef1ee2c951ed57a35a3d7a64481e61fa35c18eb

lint-ai-macos-x86_64
sha256:9bc2879e90434f470782ec9630fe1eab26fcb399da6574ae20bfcf3b37794d46

lint-ai-windows-x86_64.exe
sha256:a5be16b5543b49d5a7a931612f117d112fe63f7f5cd2d791d0808ab3be5a5fc0
```

Verify checksums:

```bash
sha256sum lint-ai-linux-x86_64
shasum -a 256 lint-ai-macos-x86_64
```

## Advanced

By default the linter skips files larger than 5MB and stops after 50k files. Override these limits:

```bash
./lint-ai /path/to/repo --max-bytes 10000000 --max-files 100000
```

Limit directory traversal depth:

```bash
./lint-ai /path/to/repo --max-depth 10
```

Limit total bytes read across the corpus:

```bash
./lint-ai /path/to/repo --max-total-bytes 100000000
```

The tool will automatically scope to `/path/to/repo/docs/**` when that folder exists.

Example with a local repo:

```bash
./lint-ai /path/to/openclaw
```

Show the inferred concept inventory:

```bash
./lint-ai /path/to/openclaw/docs/channels --show-concepts
```

Show Markdown headings per file (structure/architecture hints):

```bash
./lint-ai /path/to/openclaw/docs/channels --show-headings
```

Debug phrase matches (prints matched text fragments and concepts):

```bash
./lint-ai /path/to/openclaw/docs/channels --debug-matches
```

Analyze a corpus and emit a suggested `lint-ai.json`:

```bash
./lint-ai /path/to/openclaw/docs/channels --analyze
```

Example analysis output (Openclaw channels):

```
Suggested config:
{
  "stopwords": ["group messages", "pairing", "channel routing"],
  "ignore_sections": ["unscoped", "related"],
  "ignore_crossref_sections": ["unscoped", "related"],
  "ignore_paths": [],
  "allowlist_concepts": []
}

Stats:
pages: 31
top concepts:
- group messages (25)
- pairing (25)
- channel routing (22)
- slack (11)
- telegram (11)
- signal (10)
- whatsapp (10)
- discord (9)
- troubleshooting (9)
- line (8)
- imessage (7)
- matrix (6)
- zalo (4)
- irc (3)
- location (3)
top sections:
- configuration (41)
- setup (35)
- unscoped (31)
- security (28)
- related (22)
- troubleshooting (22)
- bundled plugin (14)
- routing (14)
- overview (10)
- notes (4)
```

## Configuration

You can place a `lint-ai.json` file in the target root (or pass `--config /path/to/lint-ai.json`)
to control filters.

Use `--strict-config` to fail fast if the config is invalid.

Limit config size:

```bash
./lint-ai /path/to/repo --max-config-bytes 2000000
```

```json
{
  "stopwords": ["workflow", "example"],
  "ignore_sections": ["related", "unscoped"],
  "ignore_crossref_sections": ["related", "unscoped"],
  "ignore_paths": ["docs/reference/"]
}
```

Example used for Openclaw channels (reduce false positives by skipping "Related" sections and
ignoring generic terms):

```json
{
  "stopwords": ["channel", "message", "messages", "bot", "client", "config"],
  "ignore_sections": ["related", "unscoped"],
  "ignore_crossref_sections": ["related", "unscoped"],
  "ignore_paths": [],
  "allowlist_concepts": ["discord", "slack", "telegram", "whatsapp", "signal", "matrix"],
  "scope_prefix": "docs/channels/"
}
```

Run it:

```bash
./lint-ai /path/to/openclaw/docs/channels --config /path/to/openclaw/lint-ai.json
```

## Development

### Build

```bash
cargo build
```

### Test

```bash
cargo test
```

### Contributing

1. Fork the repo and create a feature branch.
2. Make changes with tests where appropriate.
3. Run `cargo test`.
4. Open a PR.

## Concept Examples

Concept inventory (derived from filenames in `docs/channels/**`):

```
discord
slack
telegram
whatsapp
group messages
channel routing
```

Concepts grouped by section (aggregated across the corpus):

```
Section: setup
- pairing (4)
- signal (3)
- feishu (2)
- zalo (2)

Section: configuration
- pairing (7)
- signal (6)
- feishu (3)
- groups (3)

Section: related
- channel routing (21)
- groups (21)
- pairing (21)
```

Surface forms (for matching text to a concept):

```
group messages
group-messages
group_messages
groupmessages
group message
group messages
```

## Output Examples

Sample findings now include severity tags and link‑debt signals:

```
Missing cross-ref in docs/channels/discord.md -> [[signal]] (high)
Low link density in docs/channels/location.md (outgoing 1, avg 4.2)
Unreachable page: docs/channels/legacy.md
Orphan page: docs/channels/unused.md
```

Orphan detection example command:

```bash
./lint-ai /path/to/openclaw/docs/channels
```
