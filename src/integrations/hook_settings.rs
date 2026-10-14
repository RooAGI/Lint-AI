use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

const DEFAULT_HOOK_TIMEOUT_MS: u64 = 12_000;
const MIN_HOOK_TIMEOUT_MS: u64 = 12_000;
const MAX_HOOK_TIMEOUT_MS: u64 = 30_000;

/// Read the shared Claude/Codex hook budget from project settings.
/// The floor stays above the shared-store lock wait.
pub(crate) fn hook_timeout_ms(project_root: &Path) -> u64 {
    let settings_path = project_root.join(".membrick").join("settings.toml");
    let configured = std::fs::read_to_string(settings_path)
        .ok()
        .and_then(|text| text.parse::<toml::Value>().ok())
        .and_then(|value| value.get("hooks")?.get("timeout_ms")?.as_integer())
        .and_then(|value| u64::try_from(value).ok())
        .unwrap_or(DEFAULT_HOOK_TIMEOUT_MS);
    configured.clamp(MIN_HOOK_TIMEOUT_MS, MAX_HOOK_TIMEOUT_MS)
}

/// Optional hook timing log configured by the project. Relative paths resolve
/// under the project root.
pub(crate) fn hook_timings_path(project_root: &Path) -> Option<std::path::PathBuf> {
    let settings_path = project_root.join(".membrick").join("settings.toml");
    let settings = std::fs::read_to_string(settings_path).ok()?;
    let settings = settings.parse::<toml::Value>().ok()?;
    let configured = settings.get("hooks")?.get("timings_path")?.as_str()?;
    let path = std::path::PathBuf::from(configured);
    Some(if path.is_absolute() {
        path
    } else {
        project_root.join(path)
    })
}

pub(crate) fn run_with_budget<T, F>(
    timeout_ms: u64,
    complete_after_timeout: bool,
    operation: F,
) -> anyhow::Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> anyhow::Result<T> + Send + 'static,
{
    let (sender, receiver) = mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let _ = sender.send(operation());
    });
    match receiver.recv_timeout(Duration::from_millis(timeout_ms)) {
        Ok(result) => result,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            if complete_after_timeout {
                return receiver
                    .recv()
                    .map_err(|_| anyhow::anyhow!("hook operation worker disconnected"))?;
            }
            anyhow::bail!("hook operation exceeded {timeout_ms}ms budget")
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            anyhow::bail!("hook operation worker disconnected")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TempRoot(std::path::PathBuf);

    impl TempRoot {
        fn new() -> Self {
            static NEXT_ID: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "membrick-hook-settings-{}-{}",
                std::process::id(),
                NEXT_ID.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn hook_budget_defaults_above_shared_lock_wait() {
        let root = TempRoot::new();
        assert_eq!(hook_timeout_ms(root.path()), 12_000);
    }

    #[test]
    fn hook_budget_reads_and_clamps_project_setting() {
        let root = TempRoot::new();
        let settings = root.path().join(".membrick");
        std::fs::create_dir_all(&settings).unwrap();
        let path = settings.join("settings.toml");
        std::fs::write(&path, "[hooks]\ntimeout_ms = 20000\n").unwrap();
        assert_eq!(hook_timeout_ms(root.path()), 20_000);
        std::fs::write(&path, "[hooks]\ntimeout_ms = 50000\n").unwrap();
        assert_eq!(hook_timeout_ms(root.path()), 30_000);
        std::fs::write(&path, "[hooks]\ntimeout_ms = 100\n").unwrap();
        assert_eq!(hook_timeout_ms(root.path()), 12_000);
    }

    #[test]
    fn capture_finishes_before_hook_returns_when_budget_expires() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::time::Instant;

        let finished = Arc::new(AtomicBool::new(false));
        let worker_finished = Arc::clone(&finished);
        let started = Instant::now();
        let result = run_with_budget(5, true, move || {
            std::thread::sleep(Duration::from_millis(40));
            worker_finished.store(true, Ordering::SeqCst);
            Ok(())
        });

        assert!(result.is_ok(), "capture result should be returned");
        assert!(
            finished.load(Ordering::SeqCst),
            "capture must finish before hook returns"
        );
        assert!(started.elapsed() >= Duration::from_millis(35));
    }
}
