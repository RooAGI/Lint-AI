use anyhow::Result;
use std::fs::{self, File, OpenOptions};
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};
use tantivy::directory::error::OpenWriteError;
use tantivy::schema::Schema;
use tantivy::{Index, IndexWriter, TantivyError};

const WRITE_LOCK_WAIT: Duration = Duration::from_secs(10);
const WRITE_LOCK_RETRY: Duration = Duration::from_millis(100);

/// Cross-process guard shared by every persistent Tantivy writer path.
///
/// The lock file is a stable sibling of the index directory, so rebuilding an
/// index directory cannot unlink the inode that other processes use to lock.
pub(crate) struct PersistentIndexWriteLock {
    _file: File,
}

impl PersistentIndexWriteLock {
    pub(crate) fn acquire(index_dir: &Path) -> Result<Self> {
        let parent = index_dir
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let name = index_dir
            .file_name()
            .map(|name| name.to_string_lossy())
            .unwrap_or_else(|| "index".into());
        let path = parent.join(format!(".{name}.lint-ai-writer.lock"));

        {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)?;
            let started = Instant::now();
            loop {
                match file.try_lock() {
                    Ok(()) => return Ok(Self { _file: file }),
                    Err(std::fs::TryLockError::WouldBlock) => {
                        if started.elapsed() >= WRITE_LOCK_WAIT {
                            anyhow::bail!(
                                "timed out waiting for Tantivy writer lock at {}",
                                path.display()
                            );
                        }
                        thread::sleep(WRITE_LOCK_RETRY);
                    }
                    Err(std::fs::TryLockError::Error(error)) => return Err(error.into()),
                }
            }
        }
    }
}

/// The single Tantivy writer entry point. Persistent callers provide a
/// prepare callback that runs under the per-index process lock and returns
/// the latest index plus whether this operation must write it. In-memory
/// indexes use the same writer/commit path without a filesystem lock.
pub(crate) fn with_guarded_index_writer<T>(
    index: &mut Index,
    index_dir: Option<&Path>,
    writer_heap: usize,
    mut prepare: impl FnMut(&Path, &Schema) -> Result<(Index, bool)>,
    mut operation: impl FnMut(&mut IndexWriter) -> Result<T>,
) -> Result<Option<T>> {
    // A collision means Tantivy's managed-file view may be stale after an
    // interrupted commit. Drop the writer and process guard, reopen from
    // meta.json, collect orphan files, and retry once. Never wipe the index
    // or its authoritative records for this recoverable case.
    for attempt in 0..=1 {
        let result = (|| -> Result<Option<T>> {
            let _lock = index_dir
                .map(PersistentIndexWriteLock::acquire)
                .transpose()?;
            if let Some(index_dir) = index_dir {
                let schema = index.schema();
                let (latest_index, should_write) = prepare(index_dir, &schema)?;
                *index = latest_index;
                if !should_write && attempt == 0 {
                    return Ok(None);
                }
            }
            let mut writer = index.writer(writer_heap)?;
            if index_dir.is_some() {
                writer.garbage_collect_files().wait()?;
            }
            let output = operation(&mut writer)?;
            writer.commit().map(|_| Some(output)).map_err(Into::into)
        })();
        match result {
            Ok(output) => return Ok(output),
            Err(error) if attempt == 0 && is_file_collision(&error) && index_dir.is_some() => {
                // The next pass reopens and garbage-collects while holding
                // the same cross-process guard.
                continue;
            }
            Err(error) if is_file_collision(&error) => {
                return Err(anyhow::anyhow!(
                    "Tantivy hit a duplicate component file after recovery; index write was not completed: {error:#}"
                ));
            }
            Err(error) => return Err(error),
        }
    }
    unreachable!("bounded Tantivy recovery loop always returns")
}

fn is_file_collision(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<TantivyError>()
            .is_some_and(|tantivy_error| {
                matches!(
                    tantivy_error,
                    TantivyError::OpenWriteError(OpenWriteError::FileAlreadyExists(_))
                )
            })
    })
}

#[cfg(test)]
mod tests {
    use super::{with_guarded_index_writer, PersistentIndexWriteLock};
    use std::cell::Cell;
    use std::fs;
    use std::sync::mpsc;
    use std::time::Duration;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tantivy::directory::error::OpenWriteError;
    use tantivy::schema::Schema;
    use tantivy::{Index, TantivyError};

    #[test]
    fn persistent_tantivy_writer_lock_serializes_callers() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "lint-ai-tantivy-lock-{}-{suffix}",
            std::process::id()
        ));
        let index_dir = root.join("lexical");
        let held = PersistentIndexWriteLock::acquire(&index_dir).unwrap();

        let (send, receive) = mpsc::channel();
        let waiting_dir = index_dir.clone();
        let waiting = std::thread::spawn(move || {
            let lock = PersistentIndexWriteLock::acquire(&waiting_dir).unwrap();
            send.send(()).unwrap();
            drop(lock);
        });

        assert!(receive.recv_timeout(Duration::from_millis(150)).is_err());
        drop(held);
        receive
            .recv_timeout(Duration::from_secs(2))
            .expect("second writer should proceed after the guard is released");
        waiting.join().unwrap();
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn duplicate_component_collision_reopens_and_retries_once() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "lint-ai-tantivy-recovery-{}-{suffix}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let schema = Schema::builder().build();
        let mut index = Index::create_in_dir(&root, schema.clone()).unwrap();
        let operation_count = Cell::new(0);
        let prepare_count = Cell::new(0);
        let result =
            with_guarded_index_writer(
                &mut index,
                Some(&root),
                15_000_000,
                |dir, _schema| {
                    let count = prepare_count.get();
                    prepare_count.set(count + 1);
                    // A snapshot builder sees an existing index on retry.
                    // That must not skip replaying its failed population.
                    Ok((Index::open_in_dir(dir)?, count == 0))
                },
                |_| {
                    let attempt = operation_count.get() + 1;
                    operation_count.set(attempt);
                    if attempt == 1 {
                        return Err(TantivyError::OpenWriteError(
                            OpenWriteError::FileAlreadyExists(root.join("orphan.del")),
                        )
                        .into());
                    }
                    Ok(())
                },
            );

        assert!(result.is_ok());
        assert_eq!(operation_count.get(), 2);
        assert_eq!(prepare_count.get(), 2);
        let _ = fs::remove_dir_all(root);
    }
}
