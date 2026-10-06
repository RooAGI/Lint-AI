//! Shared long-lived `--serve` subprocess daemon.
//!
//! Spawning a fresh Python interpreter and loading the spaCy model costs
//! seconds; every per-request spawn paid that cost. This daemon keeps one
//! `--serve` child alive and speaks a line-delimited JSON protocol over its
//! stdin/stdout, so the model load is paid once per process instead of once
//! per request.
//!
//! The child can be a Python `--serve` script or a native binary with a
//! `--serve` mode (e.g. bekind): the daemon only moves lines, argv[0] is
//! the executable.
//!
//! Fail-open by construction: missing executables, spawn failures, dead
//! children, timeouts, and bad output are reported to the typed caller.
//! Contending requests wait for the in-flight request up to their timeout;
//! callers decide how to handle a timeout. The daemon never changes
//! judgment or extraction semantics.
//!
//! Typed wrappers own their protocol: `crate::segments::extractor_daemon`
//! (index-time extraction) and `crate::behood_query` (query-time behood)
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
    /// Full child command line: argv[0] is the executable, the rest are
    /// args. Python daemons: [python, extra_args..., script, "--serve"];
    /// native daemons: [binary, "--serve"].
    argv: Vec<String>,
    mutable: Mutex<DaemonMutable>,
}

/// The child process and its I/O. `None` fields mean "not running"; the
/// next `query` respawns.
struct DaemonMutable {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    responses: Option<mpsc::Receiver<String>>,
}

/// Query failure mode. `Busy` means the request's deadline expired while
/// waiting for another request to finish. Callers that track backend health
/// should only penalize `Failed`, not `Busy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryStatus {
    /// The request deadline expired while another request held the daemon.
    Busy,
    /// The daemon failed: write error, timeout, or dead child.
    Failed,
}

impl JsonLinesDaemon {
    /// A daemon over an explicit script. Each daemon owns exactly one
    /// child; wrappers keep one process-wide instance per script.
    pub fn new(name: &'static str, script: PathBuf, python: String) -> Self {
        Self::new_with_args(name, script, python, Vec::new())
    }

    /// A daemon over an explicit script plus extra interpreter args placed
    /// before the script (e.g. `["-I"]` for `python -I script --serve`).
    pub fn new_with_args(
        name: &'static str,
        script: PathBuf,
        python: String,
        extra_args: Vec<String>,
    ) -> Self {
        let mut argv = Vec::with_capacity(extra_args.len() + 3);
        argv.push(python);
        argv.extend(extra_args);
        argv.push(script.to_string_lossy().into_owned());
        argv.push("--serve".to_string());
        Self::new_command(name, argv)
    }

    /// A daemon over an explicit command line: argv[0] is the executable.
    /// Used for native `--serve` binaries (bekind) that are not driven
    /// through a Python interpreter.
    pub fn new_command(name: &'static str, argv: Vec<String>) -> Self {
        JsonLinesDaemon {
            inner: std::sync::Arc::new(DaemonState {
                name,
                argv,
                mutable: Mutex::new(DaemonMutable {
                    child: None,
                    stdin: None,
                    responses: None,
                }),
            }),
        }
    }

    /// Start the child now so the first real request does not pay the spawn
    /// cost. Best-effort: failures are silent; the typed caller decides how
    /// to handle a later request failure.
    pub fn prewarm(&self) {
        if let Ok(mut mutable) = self.inner.mutable.lock() {
            let _ = mutable.ensure_running(self.inner.name, &self.inner.argv);
        }
    }

    /// Send one request line and return the raw response line. Returns
    /// `None` when the daemon request fails or its deadline expires.
    pub fn query(&self, request_line: &str, timeout: Duration) -> Option<String> {
        self.query_with_status(request_line, timeout).ok()
    }

    /// Like [`query`](Self::query), but distinguishes lock contention
    /// ([`QueryStatus::Busy`]) from actual daemon failure
    /// ([`QueryStatus::Failed`]) so callers can avoid penalizing the
    /// backend for ordinary contention.
    pub fn query_with_status(
        &self,
        request_line: &str,
        timeout: Duration,
    ) -> Result<String, QueryStatus> {
        // The child speaks one request/response at a time. Give the current
        // request a chance to finish instead of dropping semantic work on
        // the first sign of contention. Charge queue time against the same
        // end-to-end timeout so contention cannot extend the caller's
        // deadline. Sleep briefly between attempts to avoid a busy loop.
        let deadline = std::time::Instant::now() + timeout;
        let mut mutable = loop {
            match self.inner.mutable.try_lock() {
                Ok(guard) => break guard,
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    return Err(QueryStatus::Failed);
                }
                Err(std::sync::TryLockError::WouldBlock) => {
                    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                    if remaining.is_zero() {
                        return Err(QueryStatus::Busy);
                    }
                    std::thread::sleep(remaining.min(Duration::from_millis(1)));
                }
            }
        };
        mutable
            .ensure_running(self.inner.name, &self.inner.argv)
            .ok_or(QueryStatus::Failed)?;
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            return Err(QueryStatus::Busy);
        }
        if mutable.write_line(request_line).is_err() {
            mutable.kill();
            return Err(QueryStatus::Failed);
        }
        match mutable
            .responses
            .as_ref()
            .and_then(|rx| rx.recv_timeout(remaining).ok())
        {
            Some(line) => Ok(line),
            None => {
                // Timeout or dead child: abandon the in-flight request and
                // kill the child so a stale late response can never be
                // misattributed to a later request. The next call respawns.
                mutable.kill();
                Err(QueryStatus::Failed)
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
    fn ensure_running(&mut self, name: &str, argv: &[String]) -> Option<()> {
        if let Some(child) = self.child.as_mut() {
            match child.try_wait() {
                Ok(None) => return Some(()), // alive
                _ => self.kill(),            // exited or unwaitable: respawn
            }
        }
        let exe = argv.first().map(String::as_str).unwrap_or("");
        if exe.is_empty() {
            eprintln!("{name} daemon: empty command line");
            return None;
        }
        // A bare executable name resolves via PATH at spawn; only check
        // existence when argv[0] names a path (as the old script check did).
        if exe.contains(std::path::MAIN_SEPARATOR) && !Path::new(exe).is_file() {
            eprintln!("{name} daemon: executable missing: {exe}");
            return None;
        }
        let mut child = Command::new(exe)
            .args(&argv[1..])
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
            std::io::Error::new(std::io::ErrorKind::BrokenPipe, "daemon not running")
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
