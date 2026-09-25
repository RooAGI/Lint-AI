//! Long-lived spaCy extractor subprocess (`scripts/spacy_relations.py --serve`).
//!
//! Spawning a fresh Python interpreter and loading the spaCy model costs
//! ~2-3s per extraction. The daemon keeps one `--serve` child alive and
//! speaks the line-delimited JSON protocol over its stdin/stdout, so the
//! model load is paid once per process instead of once per extraction.
//!
//! Fail-open by construction: every failure mode (missing script, spawn
//! failure, dead child, timeout, bad output, lock contention, poisoned
//! lock) yields `None`, and callers fall back to a one-shot subprocess
//! exactly as before. The daemon is a latency optimization only; it never
//! changes extraction semantics.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{mpsc, Mutex, OnceLock};
use std::time::Duration;

use super::relations::{parse_extractor_output, python_executable, ExtractorOutput, RelationTurn};

/// Handle to the process-wide extractor daemon.
///
/// Cheap to clone; all clones share the single child process.
#[derive(Clone)]
pub struct ExtractorDaemon {
    inner: std::sync::Arc<DaemonState>,
}

struct DaemonState {
    script: PathBuf,
    python: String,
    mutable: Mutex<DaemonMutable>,
}

/// The child process and its I/O. `None` fields mean "not running"; the
/// next `extract` respawns.
struct DaemonMutable {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    responses: Option<mpsc::Receiver<String>>,
}

impl ExtractorDaemon {
    /// The process-wide daemon over the default extractor script. Used by
    /// the production extraction paths; tests construct their own via
    /// [`ExtractorDaemon::new`] for isolation.
    pub fn global() -> &'static ExtractorDaemon {
        static DAEMON: OnceLock<ExtractorDaemon> = OnceLock::new();
        DAEMON.get_or_init(|| {
            let script =
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/spacy_relations.py");
            ExtractorDaemon::new(script, python_executable())
        })
    }

    /// A daemon over an explicit script (tests, benchmarks).
    pub fn new(script: PathBuf, python: String) -> Self {
        ExtractorDaemon {
            inner: std::sync::Arc::new(DaemonState {
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

    /// Start the child now so the first real extraction does not pay the
    /// spawn cost. Best-effort: failures are silent; extraction falls back
    /// to one-shot subprocesses.
    pub fn prewarm(&self) {
        if let Ok(mut mutable) = self.inner.mutable.lock() {
            let _ = mutable.ensure_running(&self.inner.script, &self.inner.python);
        }
    }

    /// Extract over `turns` via the daemon. Returns `None` on any failure
    /// (including lock contention — the daemon is a fast path, never a
    /// queue); the caller falls back to a one-shot subprocess.
    pub fn extract(
        &self,
        turns: &[RelationTurn],
        key_phrases_only: bool,
        timeout: Duration,
    ) -> Option<ExtractorOutput> {
        // Fast path only: never block behind another in-flight extraction.
        let mut mutable = self.inner.mutable.try_lock().ok()?;
        mutable.ensure_running(&self.inner.script, &self.inner.python)?;
        let payload = serde_json::json!({
            "model": "en_core_web_sm",
            "turns": turns,
            "key_phrases_only": key_phrases_only,
        });
        let line = serde_json::to_string(&payload).ok()?;
        if mutable.write_line(&line).is_err() {
            mutable.kill();
            return None;
        }
        let response = match mutable
            .responses
            .as_ref()
            .and_then(|rx| rx.recv_timeout(timeout).ok())
        {
            Some(line) => line,
            None => {
                // Timeout or dead child: abandon the in-flight request and
                // kill the child so a stale late response can never be
                // misattributed to a later request. The next call respawns.
                mutable.kill();
                return None;
            }
        };
        parse_extractor_output(&response)
    }
}

impl DaemonMutable {
    /// Spawn the `--serve` child unless one is already alive. Returns
    /// `None` when the child cannot be started.
    fn ensure_running(&mut self, script: &Path, python: &str) -> Option<()> {
        if let Some(child) = self.child.as_mut() {
            match child.try_wait() {
                Ok(None) => return Some(()), // alive
                _ => self.kill(),            // exited or unwaitable: respawn
            }
        }
        if !script.exists() {
            eprintln!("extractor daemon: script missing: {}", script.display());
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
                eprintln!("extractor daemon: failed to spawn: {e}");
            })
            .ok()?;
        let stdin: ChildStdin = child.stdin.take()?;
        let stdout = child.stdout.take()?;
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("extractor-daemon-reader".to_string())
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

#[cfg(test)]
impl ExtractorDaemon {
    /// Simulate child death so tests can verify respawn behavior.
    pub fn kill_child_for_test(&self) {
        if let Ok(mut mutable) = self.inner.mutable.lock() {
            mutable.kill();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_script() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/spacy_relations.py")
    }

    fn daemon_turn(doc_id: &str, text: &str) -> RelationTurn {
        RelationTurn {
            speaker: String::new(),
            text: text.to_string(),
            session_id: "test-session".to_string(),
            turn_idx: 0,
            doc_id: doc_id.to_string(),
            session_date: None,
        }
    }

    /// A fake `--serve` script: one JSON payload per stdin line, one JSON
    /// result per stdout line. A turn whose text contains "SLEEP-<n>"
    /// sleeps n seconds before answering, simulating a hung extractor.
    fn write_fake_serve_script(dir: &std::path::Path) -> PathBuf {
        let path = dir.join("fake_serve.py");
        std::fs::write(
            &path,
            r#"import json, sys, time
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    payload = json.loads(line)
    turn = (payload.get("turns") or [{}])[0]
    text = turn.get("text", "")
    if "SLEEP-" in text:
        try:
            time.sleep(int(text.split("SLEEP-")[1].split()[0]))
        except Exception:
            time.sleep(30)
    sys.stdout.write(json.dumps({
        "relations": [],
        "key_phrases": [{
            "text": "canned phrase",
            "kind": "test",
            "session_id": turn.get("session_id", ""),
            "doc_id": turn.get("doc_id", ""),
            "turn_idx": 0,
        }],
    }) + "\n")
    sys.stdout.flush()
"#,
        )
        .expect("write fake serve script");
        path
    }

    #[test]
    fn daemon_extracts_key_phrases_end_to_end() {
        let daemon = ExtractorDaemon::new(default_script(), python_executable());
        let turns = vec![daemon_turn(
            "doc-1",
            "the Paris jazz festival was wonderful",
        )];
        let output = daemon
            .extract(&turns, true, Duration::from_secs(60))
            .expect("daemon extraction should succeed");
        let texts: Vec<&str> = output.key_phrases.iter().map(|p| p.text.as_str()).collect();
        assert!(
            texts.contains(&"Paris jazz festival"),
            "expected the real extractor's phrase, got {texts:?}"
        );
    }

    #[test]
    fn daemon_second_call_is_warm() {
        let daemon = ExtractorDaemon::new(default_script(), python_executable());
        let turns = vec![daemon_turn(
            "doc-1",
            "the Paris jazz festival was wonderful",
        )];
        daemon
            .extract(&turns, true, Duration::from_secs(120))
            .expect("first call warms the model");
        let start = std::time::Instant::now();
        let output = daemon
            .extract(&turns, true, Duration::from_secs(60))
            .expect("second call should succeed");
        let elapsed = start.elapsed();
        assert!(!output.key_phrases.is_empty());
        // Warm model inference on one short turn is far below the ~2-3s
        // cold spawn+load cost. Generous bound for loaded CI machines.
        assert!(
            elapsed < Duration::from_secs(30),
            "warm daemon call took too long: {elapsed:?}"
        );
    }

    #[test]
    fn daemon_respawns_dead_child() {
        let daemon = ExtractorDaemon::new(default_script(), python_executable());
        let turns = vec![daemon_turn(
            "doc-1",
            "the Paris jazz festival was wonderful",
        )];
        daemon
            .extract(&turns, true, Duration::from_secs(120))
            .expect("first call starts the child");
        daemon.kill_child_for_test();
        let output = daemon
            .extract(&turns, true, Duration::from_secs(120))
            .expect("daemon should respawn the child and succeed");
        assert!(!output.key_phrases.is_empty());
    }

    #[test]
    fn daemon_returns_none_when_script_missing() {
        let daemon = ExtractorDaemon::new(
            PathBuf::from("/nonexistent/spacy_relations.py"),
            python_executable(),
        );
        let turns = vec![daemon_turn("doc-1", "anything")];
        assert!(
            daemon
                .extract(&turns, true, Duration::from_secs(5))
                .is_none(),
            "missing script must fail open"
        );
    }

    #[test]
    fn daemon_timeout_kills_and_respawns() {
        let dir = std::env::temp_dir().join(format!(
            "daemon-timeout-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let script = write_fake_serve_script(&dir);
        let daemon = ExtractorDaemon::new(script, python_executable());

        // Fast path works against the fake script.
        let fast_turns = vec![daemon_turn("doc-1", "anything")];
        let fast = daemon
            .extract(&fast_turns, true, Duration::from_secs(10))
            .expect("fake serve script should answer fast");
        assert_eq!(fast.key_phrases[0].text, "canned phrase");
        assert_eq!(fast.key_phrases[0].doc_id, "doc-1");

        // A hung extractor: the daemon must give up within the timeout and
        // kill the child rather than hanging the caller.
        let hung_turns = vec![daemon_turn("doc-2", "SLEEP-30 please")];
        let start = std::time::Instant::now();
        assert!(
            daemon
                .extract(&hung_turns, true, Duration::from_secs(2))
                .is_none(),
            "hung child must time out"
        );
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "timed-out extract returned too late: {:?}",
            start.elapsed()
        );

        // The hung child was killed and replaced: the next request is
        // served promptly by a fresh child instead of hanging behind (or
        // misattributing) the abandoned in-flight response.
        let start = std::time::Instant::now();
        let recovered = daemon
            .extract(&fast_turns, true, Duration::from_secs(10))
            .expect("daemon should serve after killing the hung child");
        assert_eq!(recovered.key_phrases[0].text, "canned phrase");
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "post-timeout extract hung: {:?}",
            start.elapsed()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
