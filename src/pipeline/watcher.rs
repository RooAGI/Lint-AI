use anyhow::Result;
use notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_full::{new_debouncer, DebounceEventResult, Debouncer, RecommendedCache};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Mutex;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};
use tantivy::doc;
/// A debounced project-file change published by the core index layer.
/// Contents are intentionally excluded; consumers can request bounded,
/// authorized inspection separately when needed.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct WorkspaceChangeEvent {
    pub event: String,
    pub file_path: String,
    pub timestamp_ms: u64,
}

/// Debounced project-file changes shared by every long-lived index consumer.
pub struct WorkspaceWatcher {
    _watcher: Debouncer<RecommendedWatcher, RecommendedCache>,
    events: Mutex<Receiver<DebounceEventResult>>,
    root: PathBuf,
    ignore_paths: Vec<String>,
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
        })
    }

    pub(crate) fn take_events(&self) -> Vec<WorkspaceChangeEvent> {
        let Ok(events) = self.events.lock() else {
            return vec![WorkspaceChangeEvent {
                event: "watcher_error".to_string(),
                file_path: "<watcher-error>".to_string(),
                timestamp_ms: workspace_now_ms(),
            }];
        };
        let mut changes = Vec::new();
        loop {
            match events.try_recv() {
                Ok(Ok(events)) => {
                    for event in events {
                        for path in event.event.paths {
                            let relative = path.strip_prefix(&self.root).unwrap_or(&path);
                            if pipeline_workspace_path_is_ignored(relative, &self.ignore_paths) {
                                continue;
                            }
                            let file_path = relative.to_string_lossy().replace('\\', "/");
                            if !changes
                                .iter()
                                .any(|change: &WorkspaceChangeEvent| change.file_path == file_path)
                            {
                                changes.push(WorkspaceChangeEvent {
                                    event: format!("file_{:?}", event.event.kind).to_lowercase(),
                                    file_path,
                                    timestamp_ms: workspace_now_ms(),
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
                    });
                    return changes;
                }
                Err(TryRecvError::Empty) => return changes,
            }
        }
    }

    pub fn take_change(&self) -> bool {
        !self.take_events().is_empty()
    }
}

pub(crate) fn workspace_now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn pipeline_workspace_path_is_ignored(relative: &Path, ignore_paths: &[String]) -> bool {
    let path = relative.to_string_lossy().replace('\\', "/").to_lowercase();
    path.split('/')
        .any(|component| component == ".git" || component == ".lint-ai")
        || ignore_paths.iter().any(|fragment| path.contains(fragment))
}
