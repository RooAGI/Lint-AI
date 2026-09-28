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
use std::time::{Duration, Instant};

/// Outcome of a daemon request, distinguishing "another request is in
/// flight" from "the daemon failed". Callers use this to decide whether a
/// failure cooldown is warranted: lock contention says nothing about the
/// daemon's health, so it must not trigger one.
#[derive(Debug, PartialEq, Eq)]
pub enum QueryOutcome {
    /// The daemon answered; the line may still need protocol validation.
    Answered(String),
    /// The daemon could not serve (missing script, spawn failure, dead
    /// child, timeout, write failure).
    Failed,
    /// Another request holds the daemon lock. The daemon is a fast path,
    /// never a queue: the caller should fall back immediately.
    Contended,
}

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
        match self.query_detailed(request_line, timeout) {
            QueryOutcome::Answered(line) => Some(line),
            QueryOutcome::Failed | QueryOutcome::Contended => None,
        }
    }

    /// Like [`JsonLinesDaemon::query`], but distinguishes lock contention
    /// from daemon failure so wrappers can decide whether a failure
    /// cooldown is warranted.
    pub fn query_detailed(
        &self,
        request_line: &str,
        timeout: Duration,
    ) -> QueryOutcome {
        // Fast path only: never block behind another in-flight request.
        let mut mutable = match self.inner.mutable.try_lock() {
            Ok(m) => m,
            Err(_) => return QueryOutcome::Contended,
        };
        if mutable
            .ensure_running(self.inner.name, &self.inner.script, &self.inner.python)
            .is_none()
        {
            return QueryOutcome::Failed;
        }
        if mutable.write_line(request_line).is_err() {
            mutable.kill();
            return QueryOutcome::Failed;
        }
        match mutable
            .responses
            .as_ref()
            .and_then(|rx| rx.recv_timeout(timeout).ok())
        {
            Some(line) => QueryOutcome::Answered(line),
            None => {
                // Timeout or dead child: abandon the in-flight request and
                // kill the child so a stale late response can never be
                // misattributed to a later request. The next call respawns.
                mutable.kill();
                QueryOutcome::Failed
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

// ---------------------------------------------------------------------------
// Shared singleton daemon (unix): `--serve-socket` on a well-known path.
//
// A per-process `--serve` child dies with its parent, so short-lived
// processes (hooks, CLIs) pay the full interpreter + model load on every
// invocation. The singleton tier keeps ONE daemon per user on a unix
// socket: the first process to need it spawns it (double-forked, so it
// outlives the spawner), and every later process — even a fresh one — gets
// warm answers over the socket. The daemon exits after a few idle minutes;
// stale socket files are reclaimed via the bind itself as the singleflight
// election (racing starters: exactly one wins).
//
// Fail-open: every failure yields `None` and the tiers above fall back.
// Clients NEVER kill the shared daemon — other processes may be using it.
// ---------------------------------------------------------------------------

/// Wire protocol version; must match the scripts' PROTOCOL_VERSION.
/// Wrappers put this in the socket filename so a newer client never
/// mistakes an older daemon's socket for its own.
#[cfg(unix)]
pub(crate) const SOCKET_PROTOCOL_VERSION: u32 = 1;
/// Consecutive tier failures before the tier cools down to the next fallback.
#[cfg(unix)]
const SOCKET_MAX_FAILURES: u32 = 3;
/// How long a broken tier is skipped before being retried.
#[cfg(unix)]
const SOCKET_COOLDOWN: Duration = Duration::from_secs(60);
/// Fast-fail for a singleton connect: a missing daemon falls back immediately.
#[cfg(unix)]
const SOCKET_CONNECT_TIMEOUT: Duration = Duration::from_millis(500);
/// Bound for establishing the singleton daemon (covers its model load).
#[cfg(unix)]
const SOCKET_READY_TIMEOUT: Duration = Duration::from_secs(60);
/// Bound for the ready handshake on an already-listening daemon.
#[cfg(unix)]
const SOCKET_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// Consecutive-failure circuit breaker for the singleton tier.
#[cfg(unix)]
struct CircuitBreaker {
    failures: u32,
    cooldown_until: Option<Instant>,
}

#[cfg(unix)]
impl CircuitBreaker {
    fn new() -> Self {
        CircuitBreaker {
            failures: 0,
            cooldown_until: None,
        }
    }

    fn available(&mut self) -> bool {
        match self.cooldown_until {
            Some(until) if Instant::now() < until => false,
            _ => {
                self.cooldown_until = None;
                true
            }
        }
    }

    fn note_success(&mut self) {
        self.failures = 0;
        self.cooldown_until = None;
    }

    fn note_failure(&mut self) {
        self.failures += 1;
        if self.failures >= SOCKET_MAX_FAILURES {
            self.cooldown_until = Some(Instant::now() + SOCKET_COOLDOWN);
            self.failures = 0;
        }
    }
}

/// Handle to the per-user shared `--serve-socket` daemon.
///
/// Cheap to clone; all clones share the tier state (but never the daemon
/// itself — each query opens a fresh connection, so a dead daemon is
/// noticed on the next call).
#[cfg(unix)]
#[derive(Clone)]
pub struct SocketDaemon {
    inner: std::sync::Arc<SocketDaemonState>,
}

#[cfg(unix)]
struct SocketDaemonState {
    /// Short name for log lines, e.g. "behood-query".
    name: &'static str,
    script: PathBuf,
    python: String,
    socket_path: PathBuf,
    /// Idle timeout passed to the spawned daemon; `None` takes the script's
    /// default. Tests use a short timeout so daemons don't linger.
    spawn_idle_secs: Option<u64>,
    breaker: Mutex<CircuitBreaker>,
}

#[cfg(unix)]
impl SocketDaemon {
    /// A singleton tier over an explicit socket path. Each daemon serves
    /// exactly one protocol; wrappers keep one process-wide instance.
    pub fn new(
        name: &'static str,
        script: PathBuf,
        python: String,
        socket_path: PathBuf,
    ) -> Self {
        SocketDaemon {
            inner: std::sync::Arc::new(SocketDaemonState {
                name,
                script,
                python,
                socket_path,
                spawn_idle_secs: None,
                breaker: Mutex::new(CircuitBreaker::new()),
            }),
        }
    }

    /// Test hook: how long a spawned daemon idles before exiting.
    #[cfg(test)]
    pub fn with_spawn_idle_secs(mut self, secs: u64) -> Self {
        let state = std::sync::Arc::get_mut(&mut self.inner)
            .expect("with_spawn_idle_secs before any clone");
        state.spawn_idle_secs = Some(secs);
        self
    }

    /// Spawn the shared daemon now (if needed) so the first real query
    /// does not pay the spawn cost. Best-effort: failures are silent;
    /// queries fall back to the per-process tier.
    pub fn prewarm(&self) {
        let _ = connect_ready(
            self.inner.name,
            &self.inner.script,
            &self.inner.python,
            &self.inner.socket_path,
            self.inner.spawn_idle_secs,
        );
    }

    /// One lock-step request against the shared daemon: connect (spawning
    /// the daemon first if needed), read the handshake, send one request
    /// line, read one response line. Returns `None` on any failure; the
    /// caller falls back to the per-process tier.
    pub fn query(&self, request_line: &str, timeout: Duration) -> Option<String> {
        if !self
            .inner
            .breaker
            .lock()
            .map(|mut b| b.available())
            .unwrap_or(false)
        {
            return None;
        }
        let outcome = self.query_inner(request_line, timeout);
        if let Ok(mut breaker) = self.inner.breaker.lock() {
            match outcome {
                Some(_) => breaker.note_success(),
                None => breaker.note_failure(),
            }
        }
        outcome
    }

    fn query_inner(&self, request_line: &str, timeout: Duration) -> Option<String> {
        let stream = connect_ready(
            self.inner.name,
            &self.inner.script,
            &self.inner.python,
            &self.inner.socket_path,
            self.inner.spawn_idle_secs,
        )?;
        stream.set_read_timeout(Some(timeout)).ok()?;
        (&stream).write_all(request_line.as_bytes()).ok()?;
        (&stream).write_all(b"\n").ok()?;
        let mut reader = BufReader::new(&stream);
        let mut line = String::new();
        reader.read_line(&mut line).ok()?;
        if line.trim().is_empty() {
            return None;
        }
        Some(line)
    }
}

/// Connect to a live singleton daemon, spawning one first if needed.
///
/// The daemon's socket `bind` is the singleflight election: racing starters
/// each spawn a daemon, exactly one wins the bind, the losers exit quietly.
/// A stale socket file (dead daemon) fails connect and the newly spawned
/// daemon reclaims it.
#[cfg(unix)]
fn connect_ready(
    name: &str,
    script: &Path,
    python: &str,
    path: &Path,
    spawn_idle_secs: Option<u64>,
) -> Option<std::os::unix::net::UnixStream> {
    use std::os::unix::net::UnixStream;
    // Fast path: a warm daemon is already listening.
    if let Ok(stream) = connect_with_timeout(path, SOCKET_CONNECT_TIMEOUT) {
        match read_handshake(&stream, SOCKET_HANDSHAKE_TIMEOUT) {
            Some(v) if v == SOCKET_PROTOCOL_VERSION => return Some(stream),
            // Wrong version or wedged daemon: do not disturb it; fall back.
            _ => return None,
        }
    }
    // No live daemon: spawn one, then wait for its handshake. Connects made
    // while it loads the model queue in its listen backlog instead of failing.
    spawn_socket_daemon(name, script, python, path, spawn_idle_secs)?;
    let start = Instant::now();
    while start.elapsed() < SOCKET_READY_TIMEOUT {
        if let Ok(stream) = connect_with_timeout(path, SOCKET_CONNECT_TIMEOUT) {
            match read_handshake(&stream, SOCKET_HANDSHAKE_TIMEOUT) {
                Some(v) if v == SOCKET_PROTOCOL_VERSION => return Some(stream),
                _ => {}
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    None
}

/// Connect with a timeout, via a throwaway thread (std has no
/// `UnixStream::connect_timeout`). Connects to a unix socket fail fast when
/// nobody listens; the timeout only guards a pathological full backlog.
#[cfg(unix)]
fn connect_with_timeout(
    path: &Path,
    timeout: Duration,
) -> std::io::Result<std::os::unix::net::UnixStream> {
    use std::os::unix::net::UnixStream;
    let (tx, rx) = mpsc::channel();
    let owned = path.to_path_buf();
    std::thread::spawn(move || {
        let _ = tx.send(UnixStream::connect(&owned));
    });
    match rx.recv_timeout(timeout) {
        Ok(result) => result,
        Err(mpsc::RecvTimeoutError::Timeout) => Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "unix socket connect timed out",
        )),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            "connect thread failed",
        )),
    }
}

/// Read the per-connection ready handshake; `Some(version)` on success.
#[cfg(unix)]
fn read_handshake(
    stream: &std::os::unix::net::UnixStream,
    timeout: Duration,
) -> Option<u32> {
    stream.set_read_timeout(Some(timeout)).ok()?;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    let value: serde_json::Value = serde_json::from_str(&line).ok()?;
    if value.get("ready") == Some(&serde_json::Value::Bool(true)) {
        value
            .get("protocol")
            .and_then(|p| p.as_u64())
            .map(|p| p as u32)
    } else {
        None
    }
}

/// Spawn the singleton daemon. It double-forks, so the intermediate child
/// exits at once: reap it (never a zombie) and forget the grandchild, which
/// is reparented to init and manages its own lifetime via idle timeout.
///
/// A non-zero intermediate exit means the daemon never started; fail fast
/// instead of burning the whole ready timeout polling for it.
#[cfg(unix)]
fn spawn_socket_daemon(
    name: &str,
    script: &Path,
    python: &str,
    path: &Path,
    idle_secs: Option<u64>,
) -> Option<()> {
    use std::os::unix::ffi::OsStrExt;
    if !script.is_file() {
        eprintln!("{name} socket daemon: script missing: {}", script.display());
        return None;
    }
    // Unix socket paths are limited to ~108 bytes; a longer path can never
    // bind, so fail fast instead of burning the ready timeout polling for it.
    if path.as_os_str().as_bytes().len() >= 108 {
        return None;
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok()?;
        }
    }
    let mut cmd = Command::new(python);
    cmd.arg(script)
        .arg("--serve-socket")
        .arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(secs) = idle_secs {
        cmd.arg("--idle-timeout").arg(secs.to_string());
    }
    let mut child = cmd.spawn().ok()?;
    match child.wait().ok()?.success() {
        true => Some(()),
        false => None,
    }
}
