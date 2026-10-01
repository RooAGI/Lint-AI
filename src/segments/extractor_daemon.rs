//! Long-lived spaCy extractor subprocess (`scripts/spacy_relations.py --serve`).
//!
//! Thin typed wrapper over [`crate::daemon::JsonLinesDaemon`]: it builds the
//! extraction request line and parses the response. See that module for the
//! spawn/IO/timeout/fail-open mechanics.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Duration;

use crate::daemon::JsonLinesDaemon;

use super::relations::{
    extractor_script_path, parse_extractor_output, python_executable, ExtractorOutput, RelationTurn,
};

/// Handle to the process-wide extractor daemon.
///
/// Cheap to clone; all clones share the single child process.
#[derive(Clone)]
pub struct ExtractorDaemon {
    daemon: JsonLinesDaemon,
}

impl ExtractorDaemon {
    /// The process-wide daemon over the default extractor script. Used by
    /// the production extraction paths; tests construct their own via
    /// [`ExtractorDaemon::new`] for isolation.
    pub fn global() -> &'static ExtractorDaemon {
        static DAEMON: OnceLock<ExtractorDaemon> = OnceLock::new();
        DAEMON.get_or_init(|| {
            let script = extractor_script_path();
            ExtractorDaemon::new(script, python_executable())
        })
    }

    /// A daemon over an explicit script (tests, benchmarks).
    pub fn new(script: PathBuf, python: String) -> Self {
        ExtractorDaemon {
            daemon: JsonLinesDaemon::new("extractor", script, python),
        }
    }

    /// Start the child now so the first real extraction does not pay the
    /// spawn cost. Best-effort: failures are silent; extraction falls back
    /// to one-shot subprocesses.
    pub fn prewarm(&self) {
        self.daemon.prewarm();
    }

    /// Extract over `turns` via the daemon. Returns `None` on any failure
    /// (including lock contention — the daemon is a fast path, never a
    /// queue); the caller falls back to a one-shot subprocess.
    ///
    /// `model` is the spaCy model name sent to the script (e.g.
    /// `ko_core_news_sm` for Korean turns, `zh_core_web_sm` for Chinese
    /// turns); the script caches models by
    /// name, so mixed-language processes are fine.
    pub fn extract(
        &self,
        turns: &[RelationTurn],
        key_phrases_only: bool,
        timeout: Duration,
        model: &str,
    ) -> Option<ExtractorOutput> {
        let payload = serde_json::json!({
            "model": model,
            "turns": turns,
            "key_phrases_only": key_phrases_only,
        });
        let line = serde_json::to_string(&payload).ok()?;
        let response = self.daemon.query(&line, timeout)?;
        parse_extractor_output(&response)
    }

    /// Test hook: simulate child death so tests can verify respawn behavior.
    #[cfg(test)]
    pub fn kill_child_for_test(&self) {
        self.daemon.kill_child_for_test();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_script() -> PathBuf {
        extractor_script_path()
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
        let dir = std::env::temp_dir().join(format!(
            "daemon-e2e-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let script = write_fake_serve_script(&dir);
        let daemon = ExtractorDaemon::new(script, python_executable());
        let turns = vec![daemon_turn(
            "doc-1",
            "the Paris jazz festival was wonderful",
        )];
        let output = daemon
            .extract(
                &turns,
                true,
                Duration::from_secs(60),
                crate::tier1::DEFAULT_SPACY_MODEL,
            )
            .expect("daemon extraction should succeed");
        let texts: Vec<&str> = output.key_phrases.iter().map(|p| p.text.as_str()).collect();
        assert!(
            texts.contains(&"canned phrase"),
            "expected the fake extractor's phrase, got {texts:?}"
        );
    }

    #[test]
    fn daemon_second_call_is_warm() {
        let dir = std::env::temp_dir().join(format!(
            "daemon-warm-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let script = write_fake_serve_script(&dir);
        let daemon = ExtractorDaemon::new(script, python_executable());
        let turns = vec![daemon_turn(
            "doc-1",
            "the Paris jazz festival was wonderful",
        )];
        daemon
            .extract(
                &turns,
                true,
                Duration::from_secs(120),
                crate::tier1::DEFAULT_SPACY_MODEL,
            )
            .expect("first call warms the daemon");
        let start = std::time::Instant::now();
        let output = daemon
            .extract(
                &turns,
                true,
                Duration::from_secs(60),
                crate::tier1::DEFAULT_SPACY_MODEL,
            )
            .expect("second call should succeed");
        let elapsed = start.elapsed();
        assert!(!output.key_phrases.is_empty());
        // Warm daemon call reuses the child process, far below the spawn cost.
        // Generous bound for loaded CI machines.
        assert!(
            elapsed < Duration::from_secs(30),
            "warm daemon call took too long: {elapsed:?}"
        );
    }

    #[test]
    fn daemon_respawns_dead_child() {
        let dir = std::env::temp_dir().join(format!(
            "daemon-respawn-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let script = write_fake_serve_script(&dir);
        let daemon = ExtractorDaemon::new(script, python_executable());
        let turns = vec![daemon_turn(
            "doc-1",
            "the Paris jazz festival was wonderful",
        )];
        daemon
            .extract(
                &turns,
                true,
                Duration::from_secs(120),
                crate::tier1::DEFAULT_SPACY_MODEL,
            )
            .expect("first call starts the child");
        daemon.kill_child_for_test();
        let output = daemon
            .extract(
                &turns,
                true,
                Duration::from_secs(120),
                crate::tier1::DEFAULT_SPACY_MODEL,
            )
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
                .extract(
                    &turns,
                    true,
                    Duration::from_secs(5),
                    crate::tier1::DEFAULT_SPACY_MODEL
                )
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
            .extract(
                &fast_turns,
                true,
                Duration::from_secs(10),
                crate::tier1::DEFAULT_SPACY_MODEL,
            )
            .expect("fake serve script should answer fast");
        assert_eq!(fast.key_phrases[0].text, "canned phrase");
        assert_eq!(fast.key_phrases[0].doc_id, "doc-1");

        // A hung extractor: the daemon must give up within the timeout and
        // kill the child rather than hanging the caller.
        let hung_turns = vec![daemon_turn("doc-2", "SLEEP-30 please")];
        let start = std::time::Instant::now();
        assert!(
            daemon
                .extract(
                    &hung_turns,
                    true,
                    Duration::from_secs(2),
                    crate::tier1::DEFAULT_SPACY_MODEL
                )
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
            .extract(
                &fast_turns,
                true,
                Duration::from_secs(10),
                crate::tier1::DEFAULT_SPACY_MODEL,
            )
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
