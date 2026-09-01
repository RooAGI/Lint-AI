# Quickstart

This guide gives you the shortest path to try Lint-AI on a local repository,
notes, or an agent-memory corpus. Choose the interface that fits your use case:

- **CLI** for a one-off local scan or query.
- **Python** for scripting and notebook workflows.
- **Docker** for a containerized HTTP server.
- **HTTP server** for an application or service integration.
- **MCP** for an MCP-capable agent client. See the [MCP interface guide](mcp.md).

## Fastest path

From the repository root, these three commands build Lint-AI, index a corpus,
and run a query:

```bash
cargo build --release
cargo run --release -- /path/to/repo
cargo run --release -- --query "your question" /path/to/repo
```

For an application integration, jump directly to the [Docker](#run-with-docker),
[HTTP server](server.md),
[MCP interface](mcp.md), or [other agent frameworks](agent-frameworks.md)
guide below.

## 1. Build or install

```bash
cargo build --release
```

Run the compiled binary directly:

```bash
target/release/lint-ai --help
```

Or install it on your `PATH`:

```bash
cargo install --path .
lint-ai --help
```

Query semantics use the heuristic backend in this release.
The rust-bert POS/NER path is experimental and not part of the audited release dependency graph.

## 2. Lint or index a corpus

Point Lint-AI at a repository or memory corpus directory:

```bash
cargo run --release -- /path/to/repo
```

If the repository has a `docs/` folder, the tool will usually scope itself there automatically.

## 3. Inspect the corpus

Show the derived inventory:

```bash
cargo run --release -- /path/to/repo/docs --show-concepts
cargo run --release -- /path/to/repo/docs --show-headings
```

Show the entity and term views:

```bash
cargo run --release -- /path/to/repo --show-tier0
cargo run --release -- /path/to/repo --show-tier1-entities
cargo run --release -- /path/to/repo --show-tier1-terms --tier1-term-ranker yake
```

If you want spaCy-based entity extraction:

```bash
cargo run --release -- /path/to/repo --show-tier1-entities \
  --tier1-ner-provider spacy --spacy-model en_core_web_sm
```

## 4. Query the corpus

Ask a simple memory retrieval question:

```bash
cargo run --release -- --query "docker install linux" /path/to/repo/docs
```

Ask for LLM-ready retrieval context:

```bash
cargo run --release -- --llm-context "docker install linux" /path/to/repo/docs
```

## Run with Docker

The repository includes a Compose configuration. Set a token, then build and
start the server from the repository root:

```bash
export SERVER_TOKEN=local-dev-token
docker compose up --build -d
```

The service uses a named Docker volume for the persistent index. Verify that it
is ready:

```bash
curl http://127.0.0.1:8080/health
```

To stop it:

```bash
docker compose down
```

The image runs the release HTTP server on `0.0.0.0:8080` and stores its
file-backed index under `/data/index`. For a one-off container without Compose,
see the [HTTP server guide](server.md).

## 6. Run the HTTP server

Use the standalone server when another application will add and search
memories over HTTP:

```bash
cargo run --release --bin server -- \
  --bind 127.0.0.1:8080 \
  --index .lint-ai/memory-index
```

Check that it is ready:

```bash
curl http://127.0.0.1:8080/health
```

See the [HTTP server guide](server.md) for the request contract,
authentication, lifecycle operations, and performance measurements.

## 7. Connect an agent with MCP

MCP is for agent clients that support the Model Context Protocol. Install and
configure the provider-specific adapter, then restart the client so it loads
Lint-AI's MCP server and hooks. Start with the [agent integrations guide](agents.md)
or the [MCP interface guide](mcp.md). HTTP and MCP are optional; the CLI and
Rust library work without an agent client.

## 8. Use it from Python

The Python extension exposes an in-memory `IndexStore` with `upsert`, `query`,
`remove`, and inspection methods. Build it locally with [maturin](https://www.maturin.rs/):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install maturin
maturin develop --release
```

Then use it from Python:

```python
import lint_ai

store = lint_ai.IndexStore()
store.upsert("doc-1", "Docker install guide for Ubuntu hosts")
print(store.query("docker ubuntu", 5))
```

The binding is enabled by the Rust `python` feature and does not require an
agent client or the HTTP server.

## 9. Use it as a Rust library

If you are integrating Lint-AI into a Rust app, start with `IndexStore` and `SourceDocument`.

```rust
use lint_ai::{IndexStore, PipelineOptions, SourceDocument};

fn main() -> anyhow::Result<()> {
    let mut index = IndexStore::in_memory(PipelineOptions::default());

    index.upsert(SourceDocument {
        doc_id: "artifact-1".to_string(),
        source: "artifact://artifact-1".to_string(),
        content: "docker install guide for linux hosts".to_string(),
        concept: "docker install".to_string(),
        group_id: None,
        headings: vec!["Overview".to_string()],
        links: vec![],
        timestamp: None,
        doc_length: 36,
        author_agent: None,
    });

    let results = index.query("docker install", 5)?;
    println!("{}", serde_json::to_string_pretty(&results)?);
    Ok(())
}
```

For corpus-local persistence under `.lint-ai/`, use:

```rust
use std::path::Path;
use lint_ai::{IndexStore, PipelineOptions};

let index = IndexStore::for_corpus(Path::new("/path/to/corpus"), PipelineOptions::default())?;
```

If you already have `DocRecord` values, use `lint_ai::index::MemoryIndex` for the built search structure.
