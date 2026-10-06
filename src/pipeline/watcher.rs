use anyhow::Result;
use notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_full::{new_debouncer, DebounceEventResult, Debouncer, RecommendedCache};
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Mutex;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};
use tantivy::doc;
const MAX_TRACKED_FILE_BYTES: usize = 256 * 1024;
const MAX_TRACKED_TOTAL_BYTES: usize = 16 * 1024 * 1024;
const MAX_PENDING_CHANGES: usize = 64;
/// A debounced project-file change with bounded before/after UTF-8 snapshots.
/// Snapshots are available only for indexed files or paths already observed by
/// the watcher, and are capped to keep memory use bounded.
#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceChangeEvent {
    pub event: String,
    pub file_path: String,
    pub timestamp_ms: u64,
    /// Previous UTF-8 file contents when a baseline was available and within
    /// the tracking limits. `None` means the file was new, too large, or not
    /// present in the initial indexed snapshot.
    pub before: Option<String>,
    /// Current UTF-8 file contents after the filesystem event. `None` means
    /// the file was deleted or could not be captured within the limits.
    pub after: Option<String>,
}

/// Debounced project-file changes shared by every long-lived index consumer.
pub struct WorkspaceWatcher {
    _watcher: Debouncer<RecommendedWatcher, RecommendedCache>,
    events: Mutex<Receiver<DebounceEventResult>>,
    root: PathBuf,
    ignore_paths: Vec<String>,
    snapshots: Mutex<HashMap<String, String>>,
    snapshot_bytes: Mutex<usize>,
    pending_changes: Mutex<Vec<WorkspaceChangeEvent>>,
}

impl WorkspaceWatcher {
    pub fn new(root: &Path, ignore_paths: &[String]) -> Result<Self> {
        let root = root.canonicalize()?;
        let (sender, receiver) = mpsc::channel();
        let mut watcher = new_debouncer(Duration::from_millis(350), None, sender)?;
        watcher.watch(&root, RecursiveMode::Recursive)?;
        Ok(Self {
            _watcher: watcher,
            events: Mutex::new(receiver),
            root,
            ignore_paths: ignore_paths
                .iter()
                .map(|path| path.to_lowercase())
                .collect(),
            snapshots: Mutex::new(HashMap::new()),
            snapshot_bytes: Mutex::new(0),
            pending_changes: Mutex::new(Vec::new()),
        })
    }

    /// Seed the comparison baseline from the same source documents used to
    /// build the workspace index. This avoids a second workspace-wide read.
    pub fn seed_baseline<'a>(&self, files: impl IntoIterator<Item = (&'a str, &'a str)>) {
        let (Ok(mut snapshots), Ok(mut total)) =
            (self.snapshots.lock(), self.snapshot_bytes.lock())
        else {
            return;
        };
        for (path, content) in files {
            let relative = Path::new(path);
            if pipeline_workspace_path_is_ignored(relative, &self.ignore_paths)
                || content.len() > MAX_TRACKED_FILE_BYTES
            {
                continue;
            }
            let normalized = relative.to_string_lossy().replace('\\', "/");
            if snapshots.contains_key(&normalized) {
                continue;
            }
            if total.saturating_add(content.len()) > MAX_TRACKED_TOTAL_BYTES {
                break;
            }
            *total += content.len();
            snapshots.insert(normalized, content.to_string());
        }
    }

    pub(crate) fn take_events(&self) -> Vec<WorkspaceChangeEvent> {
        let Ok(events) = self.events.lock() else {
            return vec![WorkspaceChangeEvent {
                event: "watcher_error".to_string(),
                file_path: "<watcher-error>".to_string(),
                timestamp_ms: workspace_now_ms(),
                before: None,
                after: None,
            }];
        };
        let mut changes = Vec::new();
        loop {
            match events.try_recv() {
                Ok(Ok(events)) => {
                    for event in events {
                        // Access events (opens/reads) are not content changes.
                        // The watcher exists to invalidate cached state when
                        // workspace files actually change; reacting to access
                        // is both incorrect and self-perpetuating: rebuilding
                        // the cached service itself does read_dir(root), which
                        // emits Access(Open) on the root, and that event would
                        // otherwise discard the fresh service (along with its
                        // in-memory key-phrase stamps) on the very next call.
                        if matches!(event.event.kind, notify::EventKind::Access(_)) {
                            continue;
                        }
                        for path in event.event.paths {
                            let Ok(relative) = path.strip_prefix(&self.root) else {
                                continue;
                            };
                            if pipeline_workspace_path_is_ignored(relative, &self.ignore_paths) {
                                continue;
                            }
                            let file_path = relative.to_string_lossy().replace('\\', "/");
                            if !changes
                                .iter()
                                .any(|change: &WorkspaceChangeEvent| change.file_path == file_path)
                            {
                                let before = self
                                    .snapshots
                                    .lock()
                                    .ok()
                                    .and_then(|snapshots| snapshots.get(&file_path).cloned());
                                let after = read_tracked_file(&self.root.join(relative));
                                self.update_snapshot(&file_path, after.as_deref());
                                changes.push(WorkspaceChangeEvent {
                                    event: format!("file_{:?}", event.event.kind).to_lowercase(),
                                    file_path,
                                    timestamp_ms: workspace_now_ms(),
                                    before,
                                    after,
                                });
                            }
                        }
                    }
                }
                Ok(Err(_)) | Err(TryRecvError::Disconnected) => {
                    changes.push(WorkspaceChangeEvent {
                        event: "watcher_error".to_string(),
                        file_path: "<watcher-error>".to_string(),
                        timestamp_ms: workspace_now_ms(),
                        before: None,
                        after: None,
                    });
                    return changes;
                }
                Err(TryRecvError::Empty) => return changes,
            }
        }
    }

    fn update_snapshot(&self, path: &str, content: Option<&str>) {
        let (Ok(mut snapshots), Ok(mut total)) =
            (self.snapshots.lock(), self.snapshot_bytes.lock())
        else {
            return;
        };
        if let Some(previous) = snapshots.remove(path) {
            *total = total.saturating_sub(previous.len());
        }
        let Some(content) = content.filter(|value| value.len() <= MAX_TRACKED_FILE_BYTES) else {
            return;
        };
        if total.saturating_add(content.len()) > MAX_TRACKED_TOTAL_BYTES {
            return;
        }
        *total += content.len();
        snapshots.insert(path.to_string(), content.to_string());
    }

    /// Drain OS file events with before/after content. The baseline advances
    /// on each event, so repeated edits produce successive deltas.
    pub fn take_file_changes(&self) -> Vec<WorkspaceChangeEvent> {
        let mut changes = self
            .pending_changes
            .lock()
            .map(|mut pending| std::mem::take(&mut *pending))
            .unwrap_or_default();
        changes.extend(self.take_events());
        changes
    }

    pub fn take_change(&self) -> bool {
        let changes = self.take_events();
        let changed = !changes.is_empty();
        if let Ok(mut pending) = self.pending_changes.lock() {
            pending.extend(changes);
            if pending.len() > MAX_PENDING_CHANGES {
                let discard = pending.len() - MAX_PENDING_CHANGES;
                pending.drain(..discard);
            }
        }
        changed
    }
}

fn read_tracked_file(path: &Path) -> Option<String> {
    use std::io::Read;
    let file = super::file_access::open_regular_file(path).ok()?;
    if file.metadata().ok()?.len() > MAX_TRACKED_FILE_BYTES as u64 {
        return None;
    }
    let mut bytes = Vec::new();
    file.take(MAX_TRACKED_FILE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() > MAX_TRACKED_FILE_BYTES {
        return None;
    }
    String::from_utf8(bytes).ok()
}

pub(crate) fn workspace_now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub(crate) fn workspace_path_is_sensitive(relative: &Path) -> bool {
    relative.components().any(|component| {
        let std::path::Component::Normal(name) = component else {
            return true;
        };
        let name = name.to_string_lossy().to_ascii_lowercase();
        name == ".env"
            || name.starts_with(".env.")
            || matches!(
                name.as_str(),
                ".ssh"
                    | ".aws"
                    | ".gnupg"
                    | "credentials"
                    | "credentials.json"
                    | "id_rsa"
                    | "id_ed25519"
                    | "id_ecdsa"
            )
            || name.ends_with(".pem")
            || name.ends_with(".key")
            || name.ends_with(".p12")
            || name.ends_with(".pfx")
    })
}

fn pipeline_workspace_path_is_ignored(relative: &Path, ignore_paths: &[String]) -> bool {
    let path = relative.to_string_lossy().replace('\\', "/").to_lowercase();
    workspace_path_is_sensitive(relative)
        || path
            .split('/')
            .any(|component| component == ".git" || component == ".lint-ai")
        || ignore_paths.iter().any(|fragment| path.contains(fragment))
}

#[cfg(all(test, unix))]
mod security_tests {
    use super::*;

    #[test]
    fn tracked_file_reads_reject_symlinks_and_linked_parents() {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "lint-ai-watcher-links-{}-{}",
            std::process::id(),
            workspace_now_ms()
        ));
        std::fs::create_dir_all(root.join("outside")).unwrap();
        std::fs::write(root.join("outside/secret.txt"), "external secret").unwrap();
        std::os::unix::fs::symlink(root.join("outside/secret.txt"), root.join("linked.txt"))
            .unwrap();
        std::os::unix::fs::symlink(root.join("outside"), root.join("linked-dir")).unwrap();
        assert!(
            read_tracked_file(&root.join("linked.txt")).is_none(),
            "watcher copied content through a file symlink"
        );
        assert!(
            read_tracked_file(&root.join("linked-dir/secret.txt")).is_none(),
            "watcher copied content through a directory symlink"
        );
        assert_eq!(
            read_tracked_file(&root.join("outside/secret.txt")).as_deref(),
            Some("external secret")
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
