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

Tier-1 entity extraction uses spaCy (`en_core_web_sm`) by default when it is
available, and falls back to the built-in heuristic ranker otherwise (for
example, when the spaCy model is not installed). Pass
`--tier1-ner-provider heuristic` to use the heuristic ranker explicitly.
The rust-bert POS/NER path is experimental and not part of the audited release dependency graph.

## 2. Lint or index a corpus

Point Lint-AI at a repository or memory corpus directory:

```bash
cargo run --release -- /path/to/repo
```

If the repository has a `docs/` folder, the tool will usually scope itself there automatically.

Lint-AI discovers `lint-ai.json` beside the target corpus, or accepts an
explicit file with `--config PATH`. A malformed configuration prints a warning
and falls back to defaults. Use `--strict-config` in CI or production when a
missing, oversized, or malformed configuration must fail the command instead.

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

If you want heuristic entity extraction instead of the spaCy default:

```bash
cargo run --release -- /path/to/repo --show-tier1-entities \
  --tier1-ner-provider heuristic
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

The Python extension exposes the application-facing `Memory` service with
`add`, `search`, `get`, `list`, `update`, `delete`, and `refresh` methods. Build
it locally with [uv](https://docs.astral.sh/uv/) and
[maturin](https://www.maturin.rs/):

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install "maturin>=1.9.4,<2"
maturin develop --release --uv
```

The active uv environment selects the Python interpreter, and Maturin handles
the extension-module linker configuration. To build a wheel for distribution,
run `maturin build --release`; it writes the wheel under `target/wheels/`.

Then use it from Python:

```python
import lint_ai

memory = lint_ai.Memory(path="./memory-index")
memory.add(
    "request-1",
    "user-1",
    "session-1",
    [{"role": "user", "content": "Docker runs on Ubuntu hosts"}],
)
print(memory.search("docker ubuntu", "user-1", 5))
```

For a server-backed client, use `lint_ai.RemoteMemory(base_url, api_key=None)`
with the same lifecycle methods. Python clients use `Memory` or `RemoteMemory`;
both route memory operations through the service.

## 9. Use it as a Rust library

Use `MemoryService` for all memory access. Its supporting data types are
available at the crate root; stores, indexes, snapshots, and builders are internal.

```rust
use lint_ai::{MemoryService, PipelineOptions, SourceDocument};
use std::collections::BTreeMap;

fn main() -> anyhow::Result<()> {
    let mut memory = MemoryService::in_memory(PipelineOptions::default());
    memory.upsert(SourceDocument::with_stable_doc_id_from_source(
        "artifact://artifact-1".into(),
        "Docker install guide for Linux hosts".into(),
        "docker install".into(),
        None, vec!["Overview".into()], vec![], None, None,
    ));
    memory.refresh()?;
    let results = memory.search_with_filters(
        "docker install", "artifacts", None, 5, &BTreeMap::new(),
    )?;
    println!("{}", serde_json::to_string_pretty(&results)?);
    Ok(())
}
```

For persistence, construct the service with an explicit store root:

```rust
use lint_ai::{MemoryService, PipelineOptions};

let memory = MemoryService::at_path(
    "/path/to/corpus/.lint-ai/memory", PipelineOptions::default(),
)?;
# Ok::<(), anyhow::Error>(())
```

Use `add` and `add_batch` for user/session memories. See the
[API migration guide](memory-service-api.md) for changes from direct index access.
