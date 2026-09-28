//! Query-time behood entity analysis.
//!
//! Luyi's design: "the behood provide people as the source, then we have
//! place and thing." At query time, behood judges the question's entities
//! and returns (text, kind) pairs. Lint-ai uses the text for matching and
//! the kind for filtering in structured question analysis.
//!
//! This runs `scripts/behood_query.py --serve` as a long-lived daemon: a
//! fresh Python interpreter plus the spaCy model load costs seconds, so
//! spawning one per search made every query pay that cost. The daemon keeps
//! one `--serve` child alive and speaks the line-delimited JSON protocol
//! over its stdin/stdout, so the model load is paid once per process.
//! Fail-open by construction: every daemon failure yields `None` and the
//! caller falls back to a one-shot subprocess exactly as before. Behood
//! owns judgment; the caller owns knowledge.

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::daemon::JsonLinesDaemon;

/// A (text, kind) pair judged by behood at query time.
#[derive(Debug, Clone)]
pub struct QueryEntity {
    /// The entity text as it appears in the question.
    pub text: String,
    /// Behood's ontological kind: person, place, org, event, work, food, thing.
    pub kind: String,
}

/// Analyze a question with behood, returning (text, kind) pairs.
///
/// Daemon first: the process-wide `--serve` child answers in milliseconds.
/// `None` from the daemon means the daemon itself failed (not "no entities"),
/// and only then do we fall back to a one-shot subprocess. An empty `Some`
/// is authoritative — callers fall back to heuristics, never to another
/// subprocess. Fail-open throughout: callers fall back to heuristics.
pub fn analyze_query_entities(question: &str) -> Vec<QueryEntity> {
    if let Some(entities) =
        BehoodQueryDaemon::global().analyze(question, Duration::from_secs(30))
    {
        return entities;
    }
    oneshot_analyze_query_entities(question)
}

/// One-shot `python3 scripts/behood_query.py <question>`, exactly as before
/// the daemon existed. Used only when the daemon cannot serve.
fn oneshot_analyze_query_entities(question: &str) -> Vec<QueryEntity> {
    // Locate the script relative to the crate root.
    let script = match script_path() {
        Some(p) => p,
        None => return Vec::new(),
    };

    let output = Command::new(crate::segments::relations::python_executable())
        .arg(&script)
        .arg(question)
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_entities(&stdout)
}

fn script_path() -> Option<std::path::PathBuf> {
    // CARGO_MANIFEST_DIR is set at compile time to the crate root.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(manifest_dir).join("scripts/behood_query.py");
    if path.is_file() {
        Some(path)
    } else {
        None
    }
}

/// Long-lived `scripts/behood_query.py --serve` child.
///
/// Thin typed wrapper over [`crate::daemon::JsonLinesDaemon`]: it builds the
/// `{"question": ...}` request line and parses the `{"entities": [...]`}
/// response. See that module for the spawn/IO/timeout/fail-open mechanics.
///
/// Fail-open by construction: every daemon failure yields `None` and the
/// caller falls back to a one-shot subprocess exactly as before. The daemon
/// is a latency optimization only; it never changes judgment semantics.
#[derive(Clone)]
pub struct BehoodQueryDaemon {
    daemon: JsonLinesDaemon,
    /// When the child cannot serve (e.g. no bekind binary: it exits 3 at
    /// startup), don't pay a fresh interpreter spawn on every query.
    last_serve_failure: std::sync::Arc<Mutex<Option<Instant>>>,
}

/// How long a serve failure suppresses respawn attempts. The backend does
/// not heal in milliseconds; the one-shot fallback covers the gap.
const SERVE_FAILURE_COOLDOWN: Duration = Duration::from_secs(30);

impl BehoodQueryDaemon {
    /// The process-wide daemon over the default query script. Used by the
    /// production query path; tests construct their own via
    /// [`BehoodQueryDaemon::new`] for isolation.
    pub fn global() -> &'static BehoodQueryDaemon {
        static DAEMON: OnceLock<BehoodQueryDaemon> = OnceLock::new();
        DAEMON.get_or_init(|| {
            let script = script_path().unwrap_or_default();
            BehoodQueryDaemon::new(
                script,
                crate::segments::relations::python_executable(),
            )
        })
    }

    /// A daemon over an explicit script (tests, benchmarks).
    pub fn new(script: PathBuf, python: String) -> Self {
        BehoodQueryDaemon {
            daemon: JsonLinesDaemon::new("behood-query", script, python),
            last_serve_failure: std::sync::Arc::new(Mutex::new(None)),
        }
    }

    /// Start the child now so the first real query does not pay the spawn
    /// cost. Best-effort: failures are silent; queries fall back to the
    /// one-shot subprocess.
    pub fn prewarm(&self) {
        self.daemon.prewarm();
    }

    /// Analyze `question` via the daemon. Returns `None` on any failure
    /// (including lock contention — the daemon is a fast path, never a
    /// queue); the caller falls back to a one-shot subprocess. `Some(vec)`
    /// is authoritative even when empty.
    pub fn analyze(
        &self,
        question: &str,
        timeout: Duration,
    ) -> Option<Vec<QueryEntity>> {
        // Cooldown: after a serve failure, don't pay a fresh interpreter
        // spawn on every query; the one-shot fallback covers the gap.
        // The mutex is never held across `query`, so lock ordering is safe.
        if let Ok(last) = self.last_serve_failure.lock() {
            if let Some(failed_at) = *last {
                if failed_at.elapsed() < SERVE_FAILURE_COOLDOWN {
                    return None;
                }
            }
        }
        let payload = serde_json::json!({ "question": question });
        let line = serde_json::to_string(&payload).ok()?;
        let response = match self.daemon.query(&line, timeout) {
            Some(response) => response,
            None => {
                if let Ok(mut last) = self.last_serve_failure.lock() {
                    *last = Some(Instant::now());
                }
                return None;
            }
        };
        if let Ok(mut last) = self.last_serve_failure.lock() {
            *last = None;
        }
        Some(parse_entities(&response))
    }

    /// Test hook: simulate child death so tests can verify respawn behavior.
    #[cfg(test)]
    pub fn kill_child_for_test(&self) {
        self.daemon.kill_child_for_test();
    }
}

fn parse_entities(json_str: &str) -> Vec<QueryEntity> {
    let parsed: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let mut entities = Vec::new();
    if let Some(arr) = parsed.get("entities").and_then(|e| e.as_array()) {
        for item in arr {
            let text = item.get("text").and_then(|t| t.as_str()).unwrap_or("");
            let kind = item.get("kind").and_then(|k| k.as_str()).unwrap_or("thing");
            if !text.is_empty() {
                entities.push(QueryEntity {
                    text: text.to_string(),
                    kind: kind.to_string(),
                });
            }
        }
    }
    // Deduplicate by text, keeping first occurrence.
    let mut seen = std::collections::HashSet::new();
    entities.retain(|e| seen.insert(e.text.clone()));
    entities
}

/// Extract person names from behood's query-time entities.
///
/// Returns the person-kind entity texts, cleaned of determiners and
/// quantifiers that spaCy's noun chunks may include (e.g. "both Jean").
/// Deduplicated after cleaning (behood may return both "both Jean" from
/// the noun chunk and "Jean" from the PROPN mention).
pub fn query_persons(entities: &[QueryEntity]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    entities
        .iter()
        .filter(|e| e.kind == "person")
        .map(|e| clean_person_text(&e.text))
        .filter(|t| !t.is_empty())
        .filter(|t| seen.insert(t.clone()))
        .collect()
}

/// Strip leading determiners/quantifiers from a person entity text.
fn clean_person_text(text: &str) -> String {
    let lower = text.to_lowercase();
    // Strip leading quantifiers/determiners that noun chunks include.
    for prefix in &["both ", "the ", "a ", "an "] {
        if lower.starts_with(prefix) {
            return text[prefix.len()..].trim().to_string();
        }
    }
    text.trim().to_string()
}

/// Whether behood judged any entity in the question as the given kind.
pub fn query_has_kind(entities: &[QueryEntity], kind: &str) -> bool {
    entities.iter().any(|e| e.kind == kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_entities_dedups() {
        let json = r#"{"entities": [{"text": "John", "kind": "person"}, {"text": "John", "kind": "person"}]}"#;
        let entities = parse_entities(json);
        assert_eq!(entities.len(), 1);
    }

    #[test]
    fn clean_person_text_strips_both() {
        assert_eq!(clean_person_text("both Jean"), "Jean");
        assert_eq!(clean_person_text("John"), "John");
    }

    #[test]
    fn parse_entities_fail_open() {
        assert!(parse_entities("not json").is_empty());
        assert!(parse_entities("{}").is_empty());
    }

    /// A fake `--serve` script: one {"question": ...} per stdin line, one
    /// {"entities": [...]} per stdout line. A question containing "SLEEP-<n>"
    /// sleeps n seconds before answering, simulating a hung child.
    fn write_fake_serve_script(dir: &std::path::Path) -> PathBuf {
        let path = dir.join("fake_behood_serve.py");
        std::fs::write(
            &path,
            r#"import json, sys, time
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    payload = json.loads(line)
    question = payload.get("question", "")
    if "SLEEP-" in question:
        try:
            time.sleep(int(question.split("SLEEP-")[1].split()[0]))
        except Exception:
            time.sleep(30)
    sys.stdout.write(json.dumps({
        "entities": [{"text": "canned", "kind": "person"}],
    }) + "\n")
    sys.stdout.flush()
"#,
        )
        .expect("write fake serve script");
        path
    }

    fn test_daemon(script: PathBuf) -> BehoodQueryDaemon {
        BehoodQueryDaemon::new(
            script,
            crate::segments::relations::python_executable(),
        )
    }

    fn unique_temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "behood-daemon-{}-test-{}",
            tag,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    #[test]
    fn daemon_answers_end_to_end() {
        let dir = unique_temp_dir("e2e");
        let daemon = test_daemon(write_fake_serve_script(&dir));
        let entities = daemon
            .analyze("Who visited Paris?", Duration::from_secs(60))
            .expect("daemon should answer");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "canned");
        assert_eq!(entities[0].kind, "person");
    }

    #[test]
    fn daemon_second_call_is_warm() {
        let dir = unique_temp_dir("warm");
        let daemon = test_daemon(write_fake_serve_script(&dir));
        daemon
            .analyze("first question", Duration::from_secs(120))
            .expect("first call warms the daemon");
        let start = std::time::Instant::now();
        let entities = daemon
            .analyze("second question", Duration::from_secs(60))
            .expect("second call should succeed");
        let elapsed = start.elapsed();
        assert_eq!(entities[0].text, "canned");
        // Warm daemon call reuses the child process, far below the spawn cost.
        // Generous bound for loaded CI machines.
        assert!(
            elapsed < Duration::from_secs(30),
            "warm daemon call took too long: {elapsed:?}"
        );
    }

    #[test]
    fn daemon_respawns_dead_child() {
        let dir = unique_temp_dir("respawn");
        let daemon = test_daemon(write_fake_serve_script(&dir));
        daemon
            .analyze("first question", Duration::from_secs(120))
            .expect("first call starts the child");
        daemon.kill_child_for_test();
        let entities = daemon
            .analyze("after kill", Duration::from_secs(120))
            .expect("daemon should respawn the child and succeed");
        assert_eq!(entities[0].text, "canned");
    }

    #[test]
    fn daemon_returns_none_when_script_missing() {
        let daemon = test_daemon(PathBuf::from("/nonexistent/behood_query.py"));
        assert!(
            daemon
                .analyze("anything", Duration::from_secs(5))
                .is_none(),
            "missing script must fail open"
        );
    }

    #[test]
    fn daemon_timeout_kills_and_respawns() {
        let dir = unique_temp_dir("timeout");
        let daemon = test_daemon(write_fake_serve_script(&dir));

        // Fast path works against the fake script.
        let fast = daemon
            .analyze("anything", Duration::from_secs(10))
            .expect("fake serve script should answer fast");
        assert_eq!(fast[0].text, "canned");

        // A hung child: the daemon must give up within the timeout and kill
        // the child rather than hanging the caller.
        let start = std::time::Instant::now();
        assert!(
            daemon
                .analyze("SLEEP-30 please", Duration::from_secs(2))
                .is_none(),
            "hung child must time out"
        );
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "timeout was not respected: {:?}",
            start.elapsed()
        );

        // After a timeout the child is dead; the cooldown suppresses an
        // immediate respawn, so this returns None fast rather than hanging.
        let start = std::time::Instant::now();
        assert!(
            daemon
                .analyze("anything", Duration::from_secs(10))
                .is_none(),
            "cooldown must suppress respawn right after a serve failure"
        );
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "cooldown was not respected: {:?}",
            start.elapsed()
        );
    }
}
