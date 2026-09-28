//! Shared long-lived Python subprocess daemon for `--serve` scripts.
//!
//! Spawning a fresh Python interpreter and loading the spaCy model costs
//! seconds; every per-request spawn paid that cost. This daemon keeps one
//! `--serve` child alive and speaks a line-delimited JSON protocol over its
//! stdin/stdout, so the model load is paid once per process instead of once
//! per request.
//!
//! Fail-open by construction: every failure mode (missing script, spawn
//! failure, dead child, timeout, bad output, lock contention) yields `None`,
//! and callers fall back to a one-shot subprocess exactly as before. The
//! daemon is a latency optimization only; it never changes judgment or
//! extraction semantics.
//!
//! Typed wrappers own their protocol: `crate::segments::extractor_daemon`
//! (index-time extraction) and `crate::behood_query` (query-time entities)
//! build the request line and parse the response line; this module only
//! moves lines.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{mpsc, Mutex};
use std::time::Duration;

/// Handle to a process-wide `--serve` child: one JSON request line in on
/// stdin, one JSON response line out on stdout.
///
/// Cheap to clone; all clones share the single child process.
#[derive(Clone)]
pub struct JsonLinesDaemon {
    inner: std::sync::Arc<DaemonState>,
}

struct DaemonState {
    /// Short name for log lines and the reader thread, e.g. "extractor".
    name: &'static str,
    script: PathBuf,
    python: String,
    mutable: Mutex<DaemonMutable>,
}

/// The child process and its I/O. `None` fields mean "not running"; the
/// next `query` respawns.
struct DaemonMutable {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    responses: Option<mpsc::Receiver<String>>,
}

impl JsonLinesDaemon {
    /// A daemon over an explicit script. Each daemon owns exactly one
    /// child; wrappers keep one process-wide instance per script.
    pub fn new(name: &'static str, script: PathBuf, python: String) -> Self {
        JsonLinesDaemon {
            inner: std::sync::Arc::new(DaemonState {
                name,
                script,
                python,
                mutable: Mutex::new(DaemonMutable {
                    child: None,
                    stdin: None,
                    responses: None,
                }),
            }),
        }
    }

    /// Start the child now so the first real request does not pay the spawn
    /// cost. Best-effort: failures are silent; requests fall back to the
    /// one-shot subprocess.
    pub fn prewarm(&self) {
        if let Ok(mut mutable) = self.inner.mutable.lock() {
            let _ = mutable.ensure_running(
                self.inner.name,
                &self.inner.script,
                &self.inner.python,
            );
        }
    }

    /// Send one request line, return the raw response line. Returns `None`
    /// on any failure (including lock contention — the daemon is a fast
    /// path, never a queue); the caller falls back to a one-shot
    /// subprocess.
    pub fn query(&self, request_line: &str, timeout: Duration) -> Option<String> {
        // Fast path only: never block behind another in-flight request.
        let mut mutable = self.inner.mutable.try_lock().ok()?;
        mutable.ensure_running(
            self.inner.name,
            &self.inner.script,
            &self.inner.python,
        )?;
        if mutable.write_line(request_line).is_err() {
            mutable.kill();
            return None;
        }
        match mutable
            .responses
            .as_ref()
            .and_then(|rx| rx.recv_timeout(timeout).ok())
        {
            Some(line) => Some(line),
            None => {
                // Timeout or dead child: abandon the in-flight request and
                // kill the child so a stale late response can never be
                // misattributed to a later request. The next call respawns.
                mutable.kill();
                None
            }
        }
    }

    /// Test hook: simulate child death so tests can verify respawn behavior.
    #[cfg(test)]
    pub fn kill_child_for_test(&self) {
        if let Ok(mut mutable) = self.inner.mutable.lock() {
            mutable.kill();
        }
    }
}

impl DaemonMutable {
    /// Spawn the `--serve` child unless one is already alive. Returns
    /// `None` when the child cannot be started.
    fn ensure_running(
        &mut self,
        name: &str,
        script: &Path,
        python: &str,
    ) -> Option<()> {
        if let Some(child) = self.child.as_mut() {
            match child.try_wait() {
                Ok(None) => return Some(()), // alive
                _ => self.kill(),            // exited or unwaitable: respawn
            }
        }
        if !script.is_file() {
            eprintln!("{name} daemon: script missing: {}", script.display());
            return None;
        }
        let mut child = Command::new(python)
            .arg(script)
            .arg("--serve")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Diagnostics surface in-protocol as {"error": ...}; the
            // one-shot fallback captures stderr when the daemon cannot run.
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| {
                eprintln!("{name} daemon: failed to spawn: {e}");
            })
            .ok()?;
        let stdin: ChildStdin = child.stdin.take()?;
        let stdout = child.stdout.take()?;
        let (tx, rx) = mpsc::channel();
        let thread_name = format!("{name}-daemon-reader");
        std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    match line {
                        Ok(text) => {
                            if tx.send(text).is_err() {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
            })
            .ok()?;
        self.child = Some(child);
        self.stdin = Some(stdin);
        self.responses = Some(rx);
        Some(())
    }

    fn write_line(&mut self, line: &str) -> std::io::Result<()> {
        let stdin = self.stdin.as_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "daemon not running",
            )
        })?;
        stdin.write_all(line.as_bytes())?;
        stdin.write_all(b"\n")?;
        stdin.flush()
    }

    /// Kill the child and drop its I/O. The reader thread observes EOF and
    /// exits on its own; the next `ensure_running` respawns.
    fn kill(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            // Reap the zombie. SIGKILL cannot be caught or ignored, so this
            // does not block indefinitely.
            let _ = child.wait();
        }
        self.stdin = None;
        self.responses = None;
    }
}

impl Drop for DaemonMutable {
    fn drop(&mut self) {
        self.kill();
    }
}
