//! Long-lived spaCy NER subprocess (`scripts/spacy_ner.py --serve`).
//!
//! Thin typed wrapper over [`crate::daemon::JsonLinesDaemon`]: it builds the
//! NER request line and parses the response. See that module for the
//! spawn/IO/timeout/fail-open mechanics. Mirrors
//! [`crate::segments::extractor_daemon`].
//!
//! Spawning a fresh Python interpreter and loading the spaCy model costs
//! seconds; the one-shot `SpacyKeyEntityRanker` paid that per call (once per
//! document on the `/add` path). This daemon keeps one `--serve` child alive
//! so the model load is paid once per process instead of once per request.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Duration;

use crate::daemon::JsonLinesDaemon;
use crate::tier1::{
    default_spacy_script_path, detect_python_executable, SpacyBatchInput,
    SpacyBatchOutput, SpacyDocInput, Tier1DocInput, Tier1Entity,
};

/// Handle to the process-wide NER daemon.
///
/// Cheap to clone; all clones share the single child process.
#[derive(Clone)]
pub struct NerDaemon {
    daemon: JsonLinesDaemon,
}

impl NerDaemon {
    /// The process-wide daemon over the default NER script. Used by the
    /// production NER paths; tests construct their own via
    /// [`NerDaemon::new`] for isolation.
    pub fn global() -> &'static NerDaemon {
        static DAEMON: OnceLock<NerDaemon> = OnceLock::new();
        DAEMON.get_or_init(|| {
            NerDaemon::new(
                default_spacy_script_path(),
                detect_python_executable(),
            )
        })
    }

    /// A daemon over an explicit script (tests, benchmarks).
    pub fn new(script: PathBuf, python: String) -> Self {
        NerDaemon {
            // `-I`: match the one-shot `python -I script` invocation so the
            // daemon and its fallback use identical interpreter semantics
            // (notably: user site-packages stay excluded in both).
            daemon: JsonLinesDaemon::new_with_args(
                "spacy-ner",
                script,
                python,
                vec!["-I".to_string()],
            ),
        }
    }

    /// Start the child now so the first real NER call does not pay the
    /// spawn + model-load cost. Best-effort: failures are silent; ranking
    /// falls back to the one-shot subprocess.
    pub fn prewarm(&self) {
        self.daemon.prewarm();
    }

    /// Rank key entities for `docs` via the daemon. Returns `None` on any
    /// failure (including lock contention — the daemon is a fast path,
    /// never a queue); the caller falls back to the one-shot subprocess.
    pub fn rank(
        &self,
        model: &str,
        docs: &[Tier1DocInput],
        timeout: Duration,
    ) -> Option<HashMap<String, Vec<Tier1Entity>>> {
        let payload = SpacyBatchInput {
            model,
            documents: docs
                .iter()
                .map(|d| SpacyDocInput {
                    id: &d.id,
                    text: &d.content,
                })
                .collect(),
        };
        let line = serde_json::to_string(&payload).ok()?;
        let response = self.daemon.query(&line, timeout)?;
        let value: serde_json::Value =
            serde_json::from_str(response.trim()).ok()?;
        // In-protocol errors (bad payload, model refused) fail open like
        // any other daemon failure.
        if value.get("error").is_some() {
            return None;
        }
        let parsed: SpacyBatchOutput = serde_json::from_value(value).ok()?;
        let mut out = HashMap::new();
        for doc in parsed.documents {
            let entities = doc
                .entities
                .into_iter()
                .map(|e| Tier1Entity {
                    text: e.text,
                    label: e.label,
                    start: e.start,
                    end: e.end,
                    score: e.score,
                    source: "spacy".to_string(),
                })
                .collect();
            out.insert(doc.id, entities);
        }
        Some(out)
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

    /// A fake `--serve` NER script: one JSON payload per stdin line, one
    /// JSON result per stdout line. A document whose text contains
    /// "SLEEP-<n>" sleeps n seconds before answering, simulating a hung
    /// child.
    fn write_fake_ner_serve_script(dir: &std::path::Path) -> PathBuf {
        let path = dir.join("fake_ner_serve.py");
        std::fs::write(
            &path,
            r#"import json, sys, time
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    payload = json.loads(line)
    docs = payload.get("documents", [])
    if docs and "SLEEP-" in docs[0].get("text", ""):
        try:
            time.sleep(int(docs[0]["text"].split("SLEEP-")[1].split()[0]))
        except Exception:
            time.sleep(30)
    out_docs = []
    for doc in docs:
        out_docs.append({
            "id": doc.get("id", ""),
            "entities": [{
                "text": "Canned Entity",
                "label": "TEST",
                "start": 0,
                "end": 13,
                "score": None,
            }],
        })
    sys.stdout.write(json.dumps({"documents": out_docs}) + "\n")
    sys.stdout.flush()
"#,
        )
        .expect("write fake serve script");
        path
    }

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ner-daemon-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn daemon_doc(id: &str, text: &str) -> Tier1DocInput {
        Tier1DocInput {
            id: id.to_string(),
            source: String::new(),
            content: text.to_string(),
            concept: String::new(),
            headings: Vec::new(),
        }
    }

    /// The daemon tests need a python that can run the fake script. The
    /// fake script is pure stdlib, so plain `python3` suffices — but the
    /// daemon always passes `-I`, which is harmless here.
    fn test_python() -> String {
        "python3".to_string()
    }

    #[test]
    fn daemon_ranks_entities_end_to_end() {
        let dir = test_dir("e2e");
        let script = write_fake_ner_serve_script(&dir);
        let daemon = NerDaemon::new(script, test_python());
        let docs = vec![daemon_doc("doc-1", "the Paris jazz festival")];
        let out = daemon
            .rank("en_core_web_sm", &docs, Duration::from_secs(60))
            .expect("daemon NER should succeed");
        let entities = out.get("doc-1").expect("doc-1 present");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "Canned Entity");
        assert_eq!(entities[0].source, "spacy");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn daemon_second_call_is_warm() {
        let dir = test_dir("warm");
        let script = write_fake_ner_serve_script(&dir);
        let daemon = NerDaemon::new(script, test_python());
        let docs = vec![daemon_doc("doc-1", "the Paris jazz festival")];
        daemon
            .rank("en_core_web_sm", &docs, Duration::from_secs(120))
            .expect("first call warms the daemon");
        let start = std::time::Instant::now();
        let out = daemon
            .rank("en_core_web_sm", &docs, Duration::from_secs(60))
            .expect("second call should succeed");
        assert!(!out.get("doc-1").expect("doc-1 present").is_empty());
        // Warm daemon call reuses the child process, far below the spawn cost.
        // Generous bound for loaded CI machines.
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "warm daemon call took too long: {:?}",
            start.elapsed()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn daemon_respawns_dead_child() {
        let dir = test_dir("respawn");
        let script = write_fake_ner_serve_script(&dir);
        let daemon = NerDaemon::new(script, test_python());
        let docs = vec![daemon_doc("doc-1", "the Paris jazz festival")];
        daemon
            .rank("en_core_web_sm", &docs, Duration::from_secs(120))
            .expect("first call starts the child");
        daemon.kill_child_for_test();
        let out = daemon
            .rank("en_core_web_sm", &docs, Duration::from_secs(120))
            .expect("daemon should respawn the child and succeed");
        assert!(!out.get("doc-1").expect("doc-1 present").is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn daemon_returns_none_when_script_missing() {
        let daemon = NerDaemon::new(
            PathBuf::from("/nonexistent/spacy_ner.py"),
            test_python(),
        );
        let docs = vec![daemon_doc("doc-1", "anything")];
        assert!(
            daemon
                .rank("en_core_web_sm", &docs, Duration::from_secs(5))
                .is_none(),
            "missing script must fail open"
        );
    }

    #[test]
    fn daemon_returns_none_on_in_protocol_error() {
        // A serve script that answers every request with {"error": ...}
        // must fail open, exactly like a dead child.
        let dir = test_dir("inprotoerr");
        let path = dir.join("err_serve.py");
        std::fs::write(
            &path,
            "import json, sys\nfor line in sys.stdin:\n    sys.stdout.write(json.dumps({\"error\": \"nope\"}) + \"\\n\")\n    sys.stdout.flush()\n",
        )
        .expect("write err script");
        let daemon = NerDaemon::new(path, test_python());
        let docs = vec![daemon_doc("doc-1", "anything")];
        assert!(
            daemon
                .rank("en_core_web_sm", &docs, Duration::from_secs(10))
                .is_none(),
            "in-protocol error must fail open"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn daemon_timeout_kills_and_respawns() {
        let dir = test_dir("timeout");
        let script = write_fake_ner_serve_script(&dir);
        let daemon = NerDaemon::new(script, test_python());

        let fast_docs = vec![daemon_doc("doc-1", "anything")];
        let fast = daemon
            .rank("en_core_web_sm", &fast_docs, Duration::from_secs(10))
            .expect("fake serve script should answer fast");
        assert_eq!(fast["doc-1"][0].text, "Canned Entity");

        // A hung child: the daemon must give up within the timeout and kill
        // the child rather than hanging the caller.
        let hung_docs = vec![daemon_doc("doc-2", "SLEEP-30 please")];
        let start = std::time::Instant::now();
        assert!(
            daemon
                .rank("en_core_web_sm", &hung_docs, Duration::from_secs(2))
                .is_none(),
            "hung child must time out"
        );
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "timed-out rank returned too late: {:?}",
            start.elapsed()
        );

        // The hung child was killed and replaced: the next request is
        // served promptly by a fresh child.
        let start = std::time::Instant::now();
        let recovered = daemon
            .rank("en_core_web_sm", &fast_docs, Duration::from_secs(10))
            .expect("daemon should serve after killing the hung child");
        assert_eq!(recovered["doc-1"][0].text, "Canned Entity");
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "post-timeout rank hung: {:?}",
            start.elapsed()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
