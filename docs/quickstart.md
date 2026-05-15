# Quickstart

This guide gives you the shortest path to try Lint-AI on an agent memory corpus, local repository, or note base.

## 1. Build the binary

```bash
cargo build
```

If you want to run it without installing a binary, use:

```bash
cargo run --bin lint-ai -- --help
```

Query semantics use the heuristic backend in this release.
The rust-bert POS/NER path is experimental and not part of the audited release dependency graph.

## 2. Lint a corpus

Point Lint-AI at a repository or memory corpus directory:

```bash
./lint-ai /path/to/repo
```

If the repository has a `docs/` folder, the tool will usually scope itself there automatically.

## 3. Inspect the corpus

Show the derived inventory:

```bash
./lint-ai /path/to/repo/docs --show-concepts
./lint-ai /path/to/repo/docs --show-headings
```

Show the entity and term views:

```bash
./lint-ai /path/to/repo --show-tier0
./lint-ai /path/to/repo --show-tier1-entities
./lint-ai /path/to/repo --show-tier1-terms --tier1-term-ranker yake
```

If you want spaCy-based entity extraction:

```bash
./lint-ai /path/to/repo --show-tier1-entities --tier1-ner-provider spacy --spacy-model en_core_web_sm
```

## 4. Query the corpus

Ask a simple memory retrieval question:

```bash
./lint-ai --query "docker install linux" /path/to/repo/docs
```

Ask for LLM-ready retrieval context:

```bash
./lint-ai --llm-context "docker install linux" /path/to/repo/docs
```

## 5. Use it as a library

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
