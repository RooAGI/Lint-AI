use crate::integrations::session_recording::{provider_state_dir, RecordingProvider};
use crate::memory_api::MemoryService;
pub use crate::pipeline::WorkspaceWatcher;
use crate::pipeline::{MemoryIndexLayout, PipelineOptions};
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

/// Kernel-released file-lock primitive shared by both guards below.
///
/// On Unix this is an advisory exclusive lock via `File::try_lock` (std):
/// the kernel releases the lock when the holding process dies for any
/// reason — including SIGKILL — so a crashed writer can never wedge the
/// store with an orphaned lock file. The lock file itself is never deleted;
/// only the lock state matters. Holding the returned `File` open holds the
/// lock; dropping it releases.
#[cfg(unix)]
fn acquire_file_lock(path: &Path, wait: Duration, retry: Duration, what: &str) -> Result<File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(path)?;
    let started = std::time::Instant::now();
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(std::fs::TryLockError::WouldBlock) => {
                if started.elapsed() >= wait {
                    return Err(anyhow::anyhow!(
                        "timed out waiting for {what} at {}",
                        path.display()
                    ));
                }
                thread::sleep(retry);
            }
            Err(std::fs::TryLockError::Error(error)) => return Err(error.into()),
        }
    }
}

/// Non-Unix fallback: the create_new + remove-on-Drop scheme. Keeps the
/// crate building on targets without `flock`; carries the old crash caveat
/// (an orphaned file wedges later writers until manually removed).
#[cfg(not(unix))]
fn acquire_file_lock(path: &Path, wait: Duration, retry: Duration, what: &str) -> Result<File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let started = std::time::Instant::now();
    loop {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(file) => return Ok(file),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if started.elapsed() >= wait {
                    return Err(anyhow::anyhow!(
                        "timed out waiting for {what} at {}",
                        path.display()
                    ));
                }
                thread::sleep(retry);
            }
            Err(error) => return Err(error.into()),
        }
    }
}

struct StoreInitLock {
    _file: File,
    // Only the non-Unix fallback deletes the lock file on release.
    #[cfg(not(unix))]
    path: std::path::PathBuf,
}

impl StoreInitLock {
    fn acquire(index_root: &Path) -> Result<Self> {
        fs::create_dir_all(index_root)?;
        let path = index_root.join(".initialization.lock");
        let file = acquire_file_lock(
            &path,
            STORE_INIT_LOCK_WAIT,
            STORE_INIT_LOCK_RETRY,
            "persistent store initialization lock",
        )?;
        Ok(Self {
            _file: file,
            #[cfg(not(unix))]
            path,
        })
    }
}

#[cfg(not(unix))]
impl Drop for StoreInitLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Bounded wait for the shared-store write lock: acquisition retries, then
/// errors after ~10s — hooks must never hang indefinitely on contention.
const SHARED_STORE_WRITE_LOCK_WAIT: Duration = Duration::from_secs(10);
const SHARED_STORE_WRITE_LOCK_RETRY: Duration = Duration::from_millis(50);

/// Open the persistent shared-memory service under the cross-process write
/// lock and run `operation` against it. This is the write gate for ALL
/// shared-store mutations — board operations, memory upserts, hook captures,
/// migration: everything written here survives the process and is visible to
/// hooks and other providers. Persistence happens inside `operation` via the
/// usual `upsert` / `add` / `board_open` / `board_post` calls (they refresh
/// internally).
///
/// MCP adapters keep a separate, composed in-memory service for workspace
/// search. Board/memory writes must use this persistent service so another
/// provider MCP process and later invocations see the same documents. The
/// advisory file lock serializes mutations across those processes, so
/// sequence assignment and request-id recovery run against the latest
/// persisted state; the kernel releases the lock on holder death (even
/// SIGKILL), so a crashed writer can never wedge the store. Acquisition
/// waits ~10s with retries, then errors — hooks must never hang
/// indefinitely on contention.
///
/// Luyi's architectural rule: external paths go through `MemoryService`;
/// this helper opens one on the shared store — it never touches
/// `IndexStore` directly.
pub(super) fn with_shared_memory_service<T>(
    root: &Path,
    operation: impl FnOnce(&mut MemoryService) -> Result<T>,
) -> Result<T> {
    let memory_root = shared_memory_root(root);
    fs::create_dir_all(&memory_root)?;
    let lock_path = memory_root.join(".board-operation.lock");
    let _lock = acquire_file_lock(
        &lock_path,
        SHARED_STORE_WRITE_LOCK_WAIT,
        SHARED_STORE_WRITE_LOCK_RETRY,
        "shared store write lock",
    )?;

    let mut service = MemoryService::at_path(&memory_root, segmented_store_options())?;
    operation(&mut service)
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
    RecordingProvider::OpenClaw,
    RecordingProvider::RooRuntime,
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
pub(super) fn migrate_legacy_provider_memory_dirs(root: &Path) -> Result<()> {
    let lint_ai = root.join(".lint-ai");
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
            match MemoryService::at_path(&legacy_root, segmented_store_options()) {
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
        with_shared_memory_service(root, |shared| {
            for mut document in documents {
                normalize_migrated_document(&mut document, provider);
                shared.upsert(document);
            }
            shared.refresh_index()
        })?;
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
pub(super) fn open_persistent_store(
    root: &Path,
    index_name: &str,
    memory_name: &str,
    ignore_paths: &[String],
    source_documents: impl FnOnce() -> Result<Vec<SourceDocument>>,
) -> Result<MemoryService> {
    let index_root = root.join(".lint-ai").join(index_name);
    let _init_lock = StoreInitLock::acquire(&index_root)?;
    let mut store = MemoryService::at_path(&index_root, segmented_store_options())?;
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
        store.refresh_index()?;
        write_index_state(&index_root, &state)?;
    }
    if sync_memory_documents(&root.join(".lint-ai").join(memory_name), &mut store)? {
        store.refresh_index()?;
    }
    Ok(store)
}

/// Open the persistent workspace memory and compose it with one provider's
/// private memory store. The composed store is intentionally in-memory: it
/// transfers the already-published workspace and provider segments rather than
/// reprocessing source documents, then recomputes only cross-store routing and
/// ranking statistics.
pub(super) fn open_workspace_memory_store(
    root: &Path,
    memory_name: &str,
    ignore_paths: &[String],
    source_documents: impl FnOnce() -> Result<Vec<SourceDocument>>,
) -> Result<MemoryService> {
    // One-time migration: legacy per-provider silos (e.g. `claude-memory/`)
    // merge into the shared `memory/` store. New callers pass
    // `SHARED_MEMORY_DIR`; the parameter is kept for the transition.
    if memory_name == SHARED_MEMORY_DIR {
        migrate_legacy_provider_memory_dirs(root)?;
    }
    let workspace_root = root.join(".lint-ai").join(WORKSPACE_MEMORY_NAME);
    let _init_lock = StoreInitLock::acquire(&workspace_root)?;
    let mut workspace = MemoryService::at_path(&workspace_root, segmented_store_options())?;
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
        workspace.refresh_index()?;
        write_index_state(&workspace_root, &state)?;
    }

    let memory_root = root.join(".lint-ai").join(memory_name);
    let provider_memory = memory_root
        .exists()
        .then(|| MemoryService::at_path(&memory_root, segmented_store_options()))
        .transpose()?;
    MemoryService::compose_segmented(workspace, provider_memory)
}

pub(super) fn sync_memory_documents(
    memory_root: &Path,
    target: &mut MemoryService,
) -> Result<bool> {
    if !memory_root.exists() {
        return Ok(false);
    }
    let memory = MemoryService::at_path(memory_root, segmented_store_options())?;
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
    use crate::pipeline::IndexStore;
    use std::collections::BTreeMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn board_test_root(name: &str) -> std::path::PathBuf {
        let base = std::env::temp_dir()
            .canonicalize()
            .unwrap_or_else(|_| std::env::temp_dir());
        let root = base.join(format!(
            "lint-ai-shared-board-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&root);
        root
    }

    #[test]
    fn board_post_is_readable_after_reopening_shared_service() {
        let root = board_test_root("reopen");
        let workspace = root.to_string_lossy().into_owned();
        let board = with_shared_memory_service(&root, |service| {
            service.board_open("mcp", &workspace, "research", "Research")
        })
        .unwrap();

        let posted = with_shared_memory_service(&root, |service| {
            service.board_post(
                Some(&board.board_id),
                "mcp",
                &workspace,
                None,
                "agent-a",
                "codex",
                "The parser failure is in the empty-input path.",
                "research-post-1",
            )
        })
        .unwrap();
        assert_eq!(posted.sequence, 1);

        // A fresh MemoryService instance must read the durable post from the
        // same explicit board ID, as a later MCP invocation would.
        let posts = with_shared_memory_service(&root, |service| {
            service.board_read(Some(&board.board_id), "mcp", &workspace, None, None, 20)
        })
        .unwrap();
        assert_eq!(posts.len(), 1);
        assert_eq!(posts[0].post_id, posted.post_id);
        assert_eq!(posts[0].board_id, board.board_id);
        assert_eq!(posts[0].content, posted.content);
        assert_eq!(posts[0].sequence, 1);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn shared_board_lock_serializes_writers_across_service_instances() {
        const WRITERS: usize = 6;
        let root = board_test_root("concurrent");
        let workspace = root.to_string_lossy().into_owned();
        let board = with_shared_memory_service(&root, |service| {
            service.board_open("mcp", &workspace, "parallel", "Parallel")
        })
        .unwrap();

        let mut writers = Vec::new();
        for writer in 0..WRITERS {
            let root = root.clone();
            let workspace = workspace.clone();
            let board_id = board.board_id.clone();
            writers.push(std::thread::spawn(move || {
                with_shared_memory_service(&root, |service| {
                    service.board_post(
                        Some(&board_id),
                        "mcp",
                        &workspace,
                        None,
                        &format!("agent-{writer}"),
                        "codex",
                        &format!("post from writer {writer}"),
                        &format!("parallel-{writer}"),
                    )
                })
                .unwrap()
                .sequence
            }));
        }

        let mut sequences: Vec<u64> = writers
            .into_iter()
            .map(|writer| writer.join().unwrap())
            .collect();
        sequences.sort_unstable();
        assert_eq!(sequences, (1..=WRITERS as u64).collect::<Vec<_>>());

        let posts = with_shared_memory_service(&root, |service| {
            service.board_read(Some(&board.board_id), "mcp", &workspace, None, None, 100)
        })
        .unwrap();
        assert_eq!(posts.len(), WRITERS);
        assert_eq!(
            posts.iter().map(|post| post.sequence).collect::<Vec<_>>(),
            (1..=WRITERS as u64).collect::<Vec<_>>()
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn shared_memory_lock_preserves_concurrent_provider_writes() {
        const WRITERS: usize = 6;
        let root = board_test_root("provider-memory-race");
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(WRITERS));

        let writers: Vec<_> = (0..WRITERS)
            .map(|writer| {
                let root = root.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    with_shared_memory_service(&root, |service| {
                        let id = format!("provider-memory-{writer}");
                        let content = format!("concurrent provider capture {writer}");
                        service.upsert(document(
                            &id,
                            &format!("provider://session/{writer}"),
                            &content,
                        ));
                        service.refresh_index()
                    })
                    .unwrap();
                })
            })
            .collect();

        for writer in writers {
            writer.join().unwrap();
        }

        with_shared_memory_service(&root, |service| {
            for writer in 0..WRITERS {
                let id = format!("provider-memory-{writer}");
                let document = service
                    .source_document_by_id(&id)
                    .unwrap_or_else(|| panic!("concurrent write {id} was lost"));
                assert_eq!(
                    document.content,
                    format!("concurrent provider capture {writer}")
                );
            }
            Ok(())
        })
        .unwrap();

        let _ = fs::remove_dir_all(root);
    }

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
            key_phrases: Vec::new(),
            key_phrase_extraction_hash: String::new(),
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

    fn write_lock_test_root(name: &str) -> std::path::PathBuf {
        std::env::temp_dir()
            .canonicalize()
            .unwrap_or_else(|_| std::env::temp_dir())
            .join(format!(
                "lint-ai-{name}-{}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ))
    }

    /// A second writer must time out instead of hanging forever when the
    /// write lock is held by a genuinely concurrent writer. (A crashed
    /// holder is covered by
    /// `shared_store_write_lock_survives_crashed_holder`: the kernel
    /// releases its lock on death.)
    #[test]
    fn shared_store_write_lock_times_out_when_held() {
        let root = write_lock_test_root("write-lock-held");
        let memory_root = shared_memory_root(&root);
        fs::create_dir_all(&memory_root).expect("lock dir");
        let lock_path = memory_root.join(".board-operation.lock");
        let _held = acquire_file_lock(
            &lock_path,
            Duration::from_secs(30),
            Duration::from_millis(50),
            "test-held lock",
        )
        .expect("first acquire");
        let error = with_shared_memory_service(&root, |_| Ok(())).unwrap_err();
        assert!(
            error.to_string().contains("timed out"),
            "expected a timeout error, got: {error:#}"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// Sequential writes through the helper both succeed (the lock is
    /// released between calls — no self-deadlock) and land on disk,
    /// visible to a fresh opener.
    #[test]
    fn shared_store_write_persists_and_releases_lock() {
        let root = write_lock_test_root("write-lock-persist");
        with_shared_memory_service(&root, |store| {
            store.upsert(document("doc-1", "test", "persistent content"));
            store.refresh_index()
        })
        .expect("first write");
        with_shared_memory_service(&root, |store| {
            store.upsert(document("doc-2", "test", "more content"));
            store.refresh_index()
        })
        .expect("second write");
        // The lock file itself is never deleted — only the lock state
        // matters — so assert prompt re-acquisition instead of absence.
        let start = std::time::Instant::now();
        with_shared_memory_service(&root, |_| Ok(())).expect("re-acquire after release");
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "lock was not released promptly"
        );
        let store = MemoryService::at_path(shared_memory_root(&root), segmented_store_options())
            .expect("reopen");
        assert!(store.source_document_by_id("doc-1").is_some());
        assert!(store.source_document_by_id("doc-2").is_some());
        let _ = fs::remove_dir_all(&root);
    }

    /// A SIGKILLed lock holder must not wedge the store: the kernel
    /// releases the lock on process death, so a fresh writer re-acquires
    /// promptly. (With a create_new + Drop-remove scheme this test fails:
    /// the orphaned lock file makes every later acquisition time out.)
    #[cfg(unix)]
    #[test]
    fn shared_store_write_lock_survives_crashed_holder() {
        use std::process::Command;
        // Child mode: the re-executed test binary holds the lock, signals
        // readiness, then sleeps until killed.
        if std::env::var_os("LINT_AI_LOCK_CRASH_CHILD").is_some() {
            let root = std::path::PathBuf::from(
                std::env::var("LINT_AI_LOCK_CRASH_ROOT").expect("crash child root"),
            );
            let ready = root.join("holder-ready");
            let _ = with_shared_memory_service(&root, |_| {
                fs::write(&ready, b"ready").expect("signal readiness");
                std::thread::sleep(Duration::from_secs(120));
                Ok(())
            });
            return;
        }
        let root = write_lock_test_root("write-lock-crash");
        fs::create_dir_all(&root).expect("crash test root");
        let exe = std::env::current_exe().expect("test binary path");
        let mut child = Command::new(exe)
            .arg("--exact")
            .arg(
                "memory_api::workspace::tests::\
                 shared_store_write_lock_survives_crashed_holder",
            )
            .arg("--nocapture")
            .env("LINT_AI_LOCK_CRASH_CHILD", "1")
            .env("LINT_AI_LOCK_CRASH_ROOT", &root)
            .spawn()
            .expect("spawn crash child");
        // Wait until the child actually holds the lock.
        let ready = root.join("holder-ready");
        let start = std::time::Instant::now();
        while !ready.exists() {
            if start.elapsed() > Duration::from_secs(30) {
                let _ = child.kill();
                panic!("crash child never acquired the lock");
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        // SIGKILL: no Drop runs, no userspace cleanup is possible.
        unsafe {
            assert_eq!(libc::kill(child.id() as libc::pid_t, libc::SIGKILL), 0);
        }
        let _ = child.wait();
        // Re-acquisition must succeed promptly — well under the ~10s
        // contention timeout.
        let start = std::time::Instant::now();
        with_shared_memory_service(&root, |_| Ok(())).expect("re-acquire after crash");
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "lock was not released by the kernel after holder death"
        );
        let _ = fs::remove_dir_all(&root);
    }
}
