use crate::pipeline::{IndexStore, MemoryIndexLayout, PipelineOptions};
use crate::segments::SegmentRoutingStrategy;
use crate::source::SourceDocument;
use anyhow::Result;
use serde_json::{json, Value};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;

pub fn segmented_store_options() -> PipelineOptions {
    PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 3,
            routing_strategy: SegmentRoutingStrategy::LocalDistinctiveness,
        },
        ..PipelineOptions::default()
    }
}

pub fn trace_event(event: &str) {
    let Some(path) = std::env::var_os("LINT_AI_MCP_TRACE_PATH") else {
        return;
    };
    if let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{event}");
    }
}

/// Open the project index, rebuilding it only when the workspace it describes
/// has moved on. `ignore_paths` is part of that description: the documents the
/// caller hands over depend on it, so an index built under different ignores is
/// as stale as one built at a different revision.
pub fn open_persistent_store(
    root: &Path,
    index_name: &str,
    memory_name: &str,
    ignore_paths: &[String],
    source_documents: impl FnOnce() -> Result<Vec<SourceDocument>>,
) -> Result<IndexStore> {
    let index_root = root.join(".lint-ai").join(index_name);
    let mut store = IndexStore::at_path(&index_root, segmented_store_options())?;
    let state = workspace_state(root, ignore_paths);
    if store.is_empty() || !index_is_current(&index_root, state.as_ref()) {
        for doc_id in store
            .source_documents()
            .into_iter()
            .map(|document| document.doc_id.clone())
            .collect::<Vec<_>>()
        {
            store.remove(&doc_id);
        }
        for document in source_documents()? {
            store.upsert(document);
        }
        store.refresh()?;
        write_index_state(&index_root, state.as_ref())?;
    }
    sync_memory_documents(&root.join(".lint-ai").join(memory_name), &mut store)?;
    store.refresh()?;
    Ok(store)
}

pub fn sync_memory_documents(memory_root: &Path, target: &mut IndexStore) -> Result<()> {
    if !memory_root.exists() {
        return Ok(());
    }
    let memory = IndexStore::at_path(memory_root, segmented_store_options())?;
    for document in memory.source_documents() {
        let unchanged = target
            .source_document_by_id(&document.doc_id)
            .map(|current| {
                current.source == document.source
                    && current.content == document.content
                    && current.group_id == document.group_id
                    && current.timestamp == document.timestamp
                    && current.filters == document.filters
                    && current.headings == document.headings
                    && current.links == document.links
            })
            .unwrap_or(false);
        if !unchanged {
            target.upsert(document.clone());
        }
    }
    Ok(())
}

fn workspace_state(root: &Path, ignore_paths: &[String]) -> Option<Value> {
    let revision = git_output(root, &["rev-parse", "HEAD"])?;
    let status = git_output(root, &["status", "--porcelain=v1", "--untracked-files=all"])?;
    let status = status
        .lines()
        .filter(|line| !line.contains(".lint-ai/"))
        .collect::<Vec<_>>()
        .join("\n");
    let mut ignore_paths = ignore_paths.to_vec();
    ignore_paths.sort();
    Some(json!({ "revision": revision, "status": status, "ignore_paths": ignore_paths }))
}

fn git_output(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn index_is_current(index_root: &Path, state: Option<&Value>) -> bool {
    let Some(state) = state else {
        return false;
    };
    fs::read_to_string(index_root.join("workspace-state.json"))
        .ok()
        .and_then(|content| serde_json::from_str::<Value>(&content).ok())
        .as_ref()
        == Some(state)
}

fn write_index_state(index_root: &Path, state: Option<&Value>) -> Result<()> {
    let Some(state) = state else {
        return Ok(());
    };
    fs::create_dir_all(index_root)?;
    fs::write(
        index_root.join("workspace-state.json"),
        serde_json::to_string_pretty(state)?,
    )?;
    Ok(())
}
