use crate::integrations::session_recording::{provider_state_dir, RecordingProvider};
pub use crate::pipeline::WorkspaceWatcher;
use crate::pipeline::{IndexStore, MemoryIndexLayout, PipelineOptions};
use crate::segments::SegmentRoutingStrategy;
use crate::source::SourceDocument;
use anyhow::Result;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::thread;
use std::time::Duration;
use walkdir::WalkDir;

const STORE_INIT_LOCK_WAIT: Duration = Duration::from_secs(30);
const STORE_INIT_LOCK_RETRY: Duration = Duration::from_millis(100);

struct StoreInitLock {
    path: std::path::PathBuf,
    _file: File,
}

impl StoreInitLock {
    fn acquire(index_root: &Path) -> Result<Self> {
        fs::create_dir_all(index_root)?;
        let path = index_root.join(".initialization.lock");
        let started = std::time::Instant::now();
        loop {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => return Ok(Self { path, _file: file }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    if started.elapsed() >= STORE_INIT_LOCK_WAIT {
                        return Err(anyhow::anyhow!(
                            "timed out waiting for persistent store initialization lock at {}",
                            path.display()
                        ));
                    }
                    thread::sleep(STORE_INIT_LOCK_RETRY);
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
}

impl Drop for StoreInitLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Directory name (under `.lint-ai/`) for the shared cross-provider memory
/// store. All agents read and write the same memory; the provider is kept as
/// per-document attribution (`filters.provider`, `author_agent`, and the
/// `{provider}-session:{id}` group id) rather than as a storage silo.
pub const SHARED_MEMORY_DIR: &str = "memory";

/// Legacy per-provider memory directories, migrated into [`SHARED_MEMORY_DIR`]
/// on first use. Kept in sync with the providers that used to own a silo.
const LEGACY_PROVIDERS: &[RecordingProvider] = &[
    RecordingProvider::Claude,
    RecordingProvider::Codex,
    RecordingProvider::Gemini,
    RecordingProvider::Agy,
    RecordingProvider::Muse,
];

/// Directory name (under `.lint-ai/`) of the legacy per-provider memory silo
/// for `provider`, e.g. `codex-memory`.
fn legacy_provider_memory_dir(provider: RecordingProvider) -> String {
    format!("{}-memory", provider.as_str())
}

/// Path to the shared memory store for a workspace root.
pub fn shared_memory_root(root: &Path) -> std::path::PathBuf {
    root.join(".lint-ai").join(SHARED_MEMORY_DIR)
}

/// One-time, idempotent migration of legacy per-provider memory stores into
/// the shared store. Each legacy store's documents are upserted (provider
/// attribution travels with the documents, so nothing is lost or duplicated),
/// and the legacy directory is removed only after the shared store refreshes
/// successfully. Failures leave the legacy directory untouched.
pub fn migrate_legacy_provider_memory_dirs(root: &Path) -> Result<()> {
    let lint_ai = root.join(".lint-ai");
    let shared_root = lint_ai.join(SHARED_MEMORY_DIR);
    let mut migrated_any = false;
    for provider in LEGACY_PROVIDERS {
        let provider = *provider;
        let legacy_root = lint_ai.join(legacy_provider_memory_dir(provider));
        if !legacy_root.exists() {
            continue;
        }
        // The per-provider on/off state (`integration.json`) now lives outside
        // the legacy directory; carry it over before the directory is removed
        // so a disabled provider is not silently re-enabled by migration.
        // A state file already at the new location wins (it is fresher).
        let legacy_state = legacy_root.join("integration.json");
        if legacy_state.is_file() {
            let state_dir = provider_state_dir(provider, root);
            let new_state = state_dir.join("integration.json");
            if !new_state.exists() {
                fs::create_dir_all(&state_dir)?;
                fs::rename(&legacy_state, &new_state)?;
            }
        }
        let documents: Vec<SourceDocument> =
            match IndexStore::at_path(&legacy_root, segmented_store_options()) {
                Ok(legacy_store) => legacy_store
                    .source_documents()
                    .into_iter()
                    .cloned()
                    .collect(),
                // A concurrent server migrated and removed the directory first.
                Err(error) if is_not_found(&error) => continue,
                Err(error) => return Err(error),
            };
        if documents.is_empty() {
            let _ = fs::remove_dir_all(&legacy_root);
            continue;
        }
        let mut shared = IndexStore::at_path(&shared_root, segmented_store_options())?;
        for mut document in documents {
            normalize_migrated_document(&mut document, provider);
            shared.upsert(document);
        }
        shared.refresh()?;
        match fs::remove_dir_all(&legacy_root) {
            Ok(()) => {}
            // A concurrent server removed it first; the documents are already
            // in the shared store.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        migrated_any = true;
    }
    if migrated_any {
        trace_event("migrated legacy provider memory stores into shared memory");
    }
    Ok(())
}

/// Normalize a document copied out of a legacy per-provider silo.
/// Older Lint-AI versions stamped documents with `filters.integration`
/// (values like "claude-code", and the key itself is pre-canonicalization);
/// the shared store and the MCP `provider` search argument use the
/// canonical `filters.provider` values. The legacy directory is
/// authoritative for which provider captured a document, so migration
/// stamps the canonical value and drops the old key. Without this, a
/// provider-filtered search would miss every migrated memory.
fn normalize_migrated_document(document: &mut SourceDocument, provider: RecordingProvider) {
    document.filters.remove("integration");
    document
        .filters
        .insert("provider".to_string(), provider.as_str().to_string());
}

fn is_not_found(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound)
    })
}

pub fn segmented_store_options() -> PipelineOptions {
    PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 3,
            routing_strategy: SegmentRoutingStrategy::LocalDistinctiveness,
        },
        ..PipelineOptions::default()
    }
}

/// The shared project corpus. It contains workspace files once; provider
/// memories stay in their own stores and are composed into an in-memory query
/// view for the requesting provider.
pub const WORKSPACE_MEMORY_NAME: &str = "workspace-memory";
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
    let _init_lock = StoreInitLock::acquire(&index_root)?;
    let mut store = IndexStore::at_path(&index_root, segmented_store_options())?;
    let state = workspace_state(root, ignore_paths)?;
    if store.is_empty() || !index_is_current(&index_root, &state) {
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
        write_index_state(&index_root, &state)?;
    }
    if sync_memory_documents(&root.join(".lint-ai").join(memory_name), &mut store)? {
        store.refresh()?;
    }
    Ok(store)
}

/// Open the persistent workspace memory and compose it with one provider's
/// private memory store. The composed store is intentionally in-memory: it
/// transfers the already-published workspace and provider segments rather than
/// reprocessing source documents, then recomputes only cross-store routing and
/// ranking statistics.
pub fn open_workspace_memory_store(
    root: &Path,
    memory_name: &str,
    ignore_paths: &[String],
    source_documents: impl FnOnce() -> Result<Vec<SourceDocument>>,
) -> Result<IndexStore> {
    // One-time migration: legacy per-provider silos (e.g. `claude-memory/`)
    // merge into the shared `memory/` store. New callers pass
    // `SHARED_MEMORY_DIR`; the parameter is kept for the transition.
    if memory_name == SHARED_MEMORY_DIR {
        migrate_legacy_provider_memory_dirs(root)?;
    }
    let workspace_root = root.join(".lint-ai").join(WORKSPACE_MEMORY_NAME);
    let _init_lock = StoreInitLock::acquire(&workspace_root)?;
    let mut workspace = IndexStore::at_path(&workspace_root, segmented_store_options())?;
    let state = workspace_state(root, ignore_paths)?;
    if workspace.is_empty() || !index_is_current(&workspace_root, &state) {
        for doc_id in workspace
            .source_documents()
            .into_iter()
            .map(|document| document.doc_id.clone())
            .collect::<Vec<_>>()
        {
            workspace.remove(&doc_id);
        }
        for document in source_documents()? {
            workspace.upsert(document);
        }
        workspace.refresh()?;
        write_index_state(&workspace_root, &state)?;
    }

    let memory_root = root.join(".lint-ai").join(memory_name);
    let provider_memory = memory_root
        .exists()
        .then(|| IndexStore::at_path(&memory_root, segmented_store_options()))
        .transpose()?;
    IndexStore::compose_segmented(workspace, provider_memory)
}

pub fn sync_memory_documents(memory_root: &Path, target: &mut IndexStore) -> Result<bool> {
    if !memory_root.exists() {
        return Ok(false);
    }
    let memory = IndexStore::at_path(memory_root, segmented_store_options())?;
    let mut changed = false;
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
            changed = true;
        }
    }
    Ok(changed)
}

/// Produce a deterministic state from the actual project filesystem. Git is
/// deliberately not consulted: a workspace can be outside a repository, and
/// Git's two-state dirty marker cannot distinguish successive saves.
fn workspace_state(root: &Path, ignore_paths: &[String]) -> Result<Value> {
    let mut ignore_paths = ignore_paths.to_vec();
    ignore_paths
        .iter_mut()
        .for_each(|path| *path = path.to_lowercase());
    ignore_paths.sort();
    let mut entries = WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|entry| {
            let relative = entry.path().strip_prefix(root).unwrap_or(entry.path());
            !workspace_path_is_ignored(relative, &ignore_paths)
        })
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .filter_map(|entry| {
            let relative = entry.path().strip_prefix(root).ok()?.to_path_buf();
            let metadata = entry.metadata().ok()?;
            Some((relative, metadata.len(), metadata.modified().ok()))
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(&right.0));

    let mut fingerprint = Sha256::new();
    for (relative, length, modified) in entries {
        fingerprint.update(relative.to_string_lossy().as_bytes());
        fingerprint.update([0]);
        fingerprint.update(length.to_le_bytes());
        let modified = modified
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .unwrap_or_default();
        fingerprint.update(modified.as_secs().to_le_bytes());
        fingerprint.update(modified.subsec_nanos().to_le_bytes());
    }
    Ok(json!({
        "version": 2,
        "filesystem_fingerprint": format!("{:x}", fingerprint.finalize()),
        "ignore_paths": ignore_paths,
    }))
}

fn workspace_path_is_ignored(relative: &Path, ignore_paths: &[String]) -> bool {
    let path = relative.to_string_lossy().replace('\\', "/").to_lowercase();
    path.split('/')
        .any(|component| component == ".git" || component == ".lint-ai")
        || ignore_paths.iter().any(|fragment| path.contains(fragment))
}

fn index_is_current(index_root: &Path, state: &Value) -> bool {
    fs::read_to_string(index_root.join("workspace-state.json"))
        .ok()
        .and_then(|content| serde_json::from_str::<Value>(&content).ok())
        .as_ref()
        == Some(state)
}

fn write_index_state(index_root: &Path, state: &Value) -> Result<()> {
    fs::create_dir_all(index_root)?;
    fs::write(
        index_root.join("workspace-state.json"),
        serde_json::to_string_pretty(state)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn document(doc_id: &str, source: &str, content: &str) -> SourceDocument {
        SourceDocument {
            doc_id: doc_id.to_string(),
            source: source.to_string(),
            content: content.to_string(),
            concept: "test".to_string(),
            group_id: None,
            filters: BTreeMap::new(),
            headings: vec![],
            links: vec![],
            timestamp: None,
            doc_length: content.len(),
            author_agent: None,
        }
    }

    #[test]
    fn workspace_documents_are_shared_without_copying_provider_indexes() {
        let temp_base = std::env::temp_dir()
            .canonicalize()
            .unwrap_or_else(|_| std::env::temp_dir());
        let root = temp_base.join(format!(
            "lint-ai-workspace-memory-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();

        // Legacy per-provider silos, as written by older Lint-AI versions.
        for (memory_name, id) in [
            ("codex-memory", "codex-session"),
            ("claude-memory", "claude-session"),
        ] {
            let memory_root = root.join(".lint-ai").join(memory_name);
            let mut memory = IndexStore::at_path(&memory_root, segmented_store_options()).unwrap();
            memory.upsert(document(id, &format!("{memory_name}://session"), id));
            memory.refresh().unwrap();
        }

        // Opening with the shared dir migrates the silos and composes one
        // store where every provider's memories are visible.
        let shared = open_workspace_memory_store(&root, SHARED_MEMORY_DIR, &[], || {
            Ok(vec![document(
                "workspace-guide",
                "docs/guide.md",
                "shared workspace architecture",
            )])
        })
        .unwrap();

        assert!(root
            .join(".lint-ai/workspace-memory/metadata.json")
            .is_file());
        assert!(!root.join(".lint-ai/codex-memory").exists());
        assert!(!root.join(".lint-ai/claude-memory").exists());
        assert!(shared.source_document_by_id("workspace-guide").is_some());
        assert!(shared.source_document_by_id("codex-session").is_some());
        assert!(shared.source_document_by_id("claude-session").is_some());

        // Second open reuses the migrated state without re-running migration.
        let reopened = open_workspace_memory_store(&root, SHARED_MEMORY_DIR, &[], || {
            panic!("the current workspace store must be reused")
        })
        .unwrap();
        assert!(reopened.source_document_by_id("codex-session").is_some());
        assert!(reopened.source_document_by_id("claude-session").is_some());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn migration_preserves_provider_integration_state() {
        use crate::integrations::session_recording::{lint_ai_enabled, set_lint_ai_state};

        let temp_base = std::env::temp_dir()
            .canonicalize()
            .unwrap_or_else(|_| std::env::temp_dir());
        let root = temp_base.join(format!(
            "lint-ai-migration-state-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();

        // A provider disabled under the old layout: state file lives in the
        // legacy directory, which holds no memory documents.
        let legacy = root.join(".lint-ai").join("codex-memory");
        fs::create_dir_all(&legacy).unwrap();
        fs::write(
            legacy.join("integration.json"),
            r#"{"enabled": false, "provider": "codex"}"#,
        )
        .unwrap();

        migrate_legacy_provider_memory_dirs(&root).unwrap();

        // The legacy directory is gone, but the disabled state survived at the
        // new location instead of being silently reset to enabled.
        assert!(!legacy.exists());
        assert!(root.join(".lint-ai/codex-state/integration.json").is_file());
        assert!(!lint_ai_enabled(RecordingProvider::Codex, &root).unwrap());

        // The new location round-trips through the public state API.
        set_lint_ai_state(RecordingProvider::Codex, &root, true).unwrap();
        assert!(lint_ai_enabled(RecordingProvider::Codex, &root).unwrap());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn migration_normalizes_legacy_provider_filters() {
        use crate::integrations::mcp_tools::search_provider_filters;
        use crate::query_plan::PreparedQuery;
        use serde_json::json;

        let temp_base = std::env::temp_dir()
            .canonicalize()
            .unwrap_or_else(|_| std::env::temp_dir());
        let root = temp_base.join(format!(
            "lint-ai-migration-provider-filter-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();

        // Legacy silos, as written by older Lint-AI versions: documents carry
        // the old `filters.integration` key (values like "claude-code", not
        // the canonical "claude") instead of `filters.provider`.
        fn legacy_document(doc_id: &str, integration: &str) -> SourceDocument {
            let mut doc = document(
                doc_id,
                &format!("{integration}://session"),
                "The deployment pipeline codename is cobalt",
            );
            doc.filters
                .insert("integration".to_string(), integration.to_string());
            doc
        }

        for (memory_name, doc_id, integration) in [
            ("codex-memory", "legacy-codex-memory", "codex"),
            ("claude-memory", "legacy-claude-memory", "claude-code"),
        ] {
            let memory_root = root.join(".lint-ai").join(memory_name);
            let mut memory = IndexStore::at_path(&memory_root, segmented_store_options()).unwrap();
            memory.upsert(legacy_document(doc_id, integration));
            memory.refresh().unwrap();
        }

        migrate_legacy_provider_memory_dirs(&root).unwrap();

        let shared_root = root.join(".lint-ai").join(SHARED_MEMORY_DIR);
        let mut shared = IndexStore::at_path(&shared_root, segmented_store_options()).unwrap();

        // Metadata was normalized to the canonical provider values.
        for (doc_id, provider) in [
            ("legacy-codex-memory", "codex"),
            ("legacy-claude-memory", "claude"),
        ] {
            let doc = shared.source_document_by_id(doc_id).unwrap();
            assert_eq!(
                doc.filters.get("provider").map(String::as_str),
                Some(provider),
                "{doc_id} was not stamped with the canonical provider"
            );
            assert!(
                !doc.filters.contains_key("integration"),
                "{doc_id} still carries the legacy integration filter"
            );
        }

        // The MCP provider filter finds the migrated memories.
        for (provider, doc_id) in [
            ("codex", "legacy-codex-memory"),
            ("claude", "legacy-claude-memory"),
        ] {
            let filters =
                search_provider_filters(&json!({"query": "x", "provider": provider})).unwrap();
            let hits = shared
                .query_prepared(
                    &PreparedQuery::new("deployment pipeline codename"),
                    10,
                    &filters,
                )
                .unwrap();
            assert_eq!(
                hits.iter()
                    .map(|hit| hit.doc_id.as_str())
                    .collect::<Vec<_>>(),
                vec![doc_id],
                "provider filter {provider:?} missed its migrated memory"
            );
        }

        // Unfiltered search still sees the whole shared pool.
        let unfiltered = shared
            .query_prepared(
                &PreparedQuery::new("deployment pipeline codename"),
                10,
                &BTreeMap::new(),
            )
            .unwrap();
        assert_eq!(unfiltered.len(), 2);

        let _ = fs::remove_dir_all(root);
    }
}
