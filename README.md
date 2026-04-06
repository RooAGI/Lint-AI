# Lint AI

Lint AI is a semantic documentation linter for Markdown knowledge bases. It builds a concept
inventory from your docs, generates surface forms for each concept, matches phrases across
the corpus, and applies false-positive filters to surface missing cross-references and
orphan pages.

The linter is designed to be general-purpose. You point it at any docs folder (for example
`docs/`), and it produces a report you can review to tighten navigation, eliminate stale
islands, and keep a large doc set coherent.

## Usage

Run the linter against a docs directory:

```bash
cargo run -- /path/to/repo
```

By default the linter skips files larger than 5MB and stops after 50k files. Override these limits:

```bash
cargo run -- /path/to/repo --max-bytes 10000000 --max-files 100000
```

Limit directory traversal depth:

```bash
cargo run -- /path/to/repo --max-depth 10
```

Limit total bytes read across the corpus:

```bash
cargo run -- /path/to/repo --max-total-bytes 100000000
```

The tool will automatically scope to `/path/to/repo/docs/**` when that folder exists.

Example with a local repo:

```bash
cargo run -- /path/to/openclaw
```

Show the inferred concept inventory:

```bash
cargo run -- /path/to/openclaw/docs/channels --show-concepts
```

Show Markdown headings per file (structure/architecture hints):

```bash
cargo run -- /path/to/openclaw/docs/channels --show-headings
```

Debug phrase matches (prints matched text fragments and concepts):

```bash
cargo run -- /path/to/openclaw/docs/channels --debug-matches
```

Analyze a corpus and emit a suggested `lint-ai.json`:

```bash
cargo run -- /path/to/openclaw/docs/channels --analyze
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
cargo run -- /path/to/repo --max-config-bytes 2000000
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
cargo run -- /path/to/openclaw/docs/channels --config /path/to/openclaw/lint-ai.json
```

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
