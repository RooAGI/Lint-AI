//! Query-time behood: thin client over the bekind `--serve` daemon.
//!
//! bekind owns the full tag→chunk→judge pipeline (Luyi 2026-09-30):
//! lint-ai sends raw texts and gets back per-text verdicts. No descriptors
//! cross the process boundary; no Python process is involved.
//!
//! Luyi's design: "the behood provide people as the source, then we have
//! place and thing." At query time, behood judges the question's entities
//! and returns (text, kind) pairs. Lint-ai uses the text for matching and
//! the kind for filtering in structured question analysis.
//!
//! One long-lived child (`bekind --serve`) does all the work. Spawning the
//! bekind binary per request cost ~13ms; the daemon pays it once per
//! process. Fail-open by construction: every daemon failure yields empty
//! verdicts and the caller falls back to heuristics. Behood owns judgment;
//! the caller owns knowledge.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

use serde_json::{json, Value};

use crate::daemon::JsonLinesDaemon;

/// How long one daemon round-trip may take before the caller fails open.
const DAEMON_TIMEOUT: Duration = Duration::from_secs(30);

/// How long a serve failure suppresses respawn attempts. The backend does
/// not heal in milliseconds; fallbacks cover the gap.
const SERVE_FAILURE_COOLDOWN: Duration = Duration::from_secs(30);

// Bekind is an optional enrichment. Keep it disabled unless the application
// explicitly opts in (the server exposes this as --bekind).
static BEKIND_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_enabled(enabled: bool) {
    BEKIND_ENABLED.store(enabled, Ordering::Relaxed);
}

fn is_enabled() -> bool {
    BEKIND_ENABLED.load(Ordering::Relaxed)
}

/// A (text, kind) pair judged by behood at query time.
#[derive(Debug, Clone)]
pub struct QueryEntity {
    /// The entity text as it appears in the question.
    pub text: String,
    /// Behood's ontological kind: person, place, org, event, work, food,
    /// herb (admitted closed set), thing.
    pub kind: String,
}

/// Locate the bekind binary: `BEHOOD_BIN` first, then `PATH`
/// (`bekind`, falling back to the old `behood` name).
fn bekind_bin() -> Option<PathBuf> {
    if let Ok(env) = std::env::var("BEHOOD_BIN") {
        let p = PathBuf::from(&env);
        if p.is_file() {
            return Some(p);
        }
    }
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        for name in ["bekind", "behood"] {
            let p = dir.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// How long a serve failure suppresses respawn attempts.
#[derive(Clone)]
struct ServeCooldown {
    until: std::sync::Arc<std::sync::Mutex<std::time::Instant>>,
}

impl ServeCooldown {
    fn new() -> Self {
        ServeCooldown {
            until: std::sync::Arc::new(std::sync::Mutex::new(std::time::Instant::now())),
        }
    }

    /// True when a request may proceed; false during the cooldown.
    fn gate(&self) -> bool {
        std::time::Instant::now() >= *self.until.lock().unwrap()
    }

    fn note_failure(&self) {
        *self.until.lock().unwrap() = std::time::Instant::now() + SERVE_FAILURE_COOLDOWN;
    }

    fn note_success(&self) {
        // Success clears any cooldown immediately.
        *self.until.lock().unwrap() = std::time::Instant::now();
    }
}

/// Long-lived judge daemon: `bekind --serve`.
///
/// Pure judgment over raw texts — bekind tags, chunks, and judges
/// internally (fused protocol). This module never spawns a subprocess and
/// never touches Python.
#[derive(Clone)]
pub struct BekindDaemon {
    daemon: JsonLinesDaemon,
    cooldown: ServeCooldown,
}

impl BekindDaemon {
    /// The process-wide judge daemon.
    pub fn global() -> &'static BekindDaemon {
        static DAEMON: OnceLock<BekindDaemon> = OnceLock::new();
        DAEMON.get_or_init(|| {
            let binary = bekind_bin().unwrap_or_default();
            BekindDaemon::new(binary)
        })
    }

    /// A daemon over an explicit binary (tests).
    pub fn new(binary: PathBuf) -> Self {
        BekindDaemon {
            daemon: JsonLinesDaemon::new_command(
                "bekind",
                vec![binary.to_string_lossy().into_owned(), "--serve".to_string()],
            ),
            cooldown: ServeCooldown::new(),
        }
    }

    /// Start the child now so the first real query does not pay the spawn
    /// cost. Best-effort: failures are silent; queries fall back.
    pub fn prewarm(&self) {
        if is_enabled() {
            self.daemon.prewarm();
        }
    }

    /// Judge raw texts: one JSON line through the daemon, one Response JSON
    /// back. Returns the per-text results (`text_results`) on success,
    /// `None` on any failure (including lock contention — the daemon is a
    /// fast path, never a queue); the caller fails open.
    fn judge_texts(&self, texts: &[(String, &str, bool)]) -> Option<Vec<FusedTextResult>> {
        if !is_enabled() {
            return None;
        }
        if !self.cooldown.gate() {
            return None;
        }
        let request = json!({
            "texts": texts.iter().map(|(id, text, with_scope)| {
                json!({"id": id, "text": text, "with_scope": with_scope, "with_activity": true})
            }).collect::<Vec<_>>(),
        });
        let line = serde_json::to_string(&request).ok()?;
        // Distinguish a saturated daemon deadline from an actual backend
        // failure. Both fail open at the search layer, but only backend
        // failures trigger the cooldown.
        let response = match self.daemon.query_with_status(&line, DAEMON_TIMEOUT) {
            Ok(response) => response,
            Err(crate::daemon::QueryStatus::Busy) => return None,
            Err(crate::daemon::QueryStatus::Failed) => {
                self.cooldown.note_failure();
                return None;
            }
        };
        self.cooldown.note_success();
        let value: Value = serde_json::from_str(&response).ok()?;
        parse_text_results(&value)
    }

    #[cfg(test)]
    pub fn kill_child_for_test(&self) {
        self.daemon.kill_child_for_test();
    }
}

/// One text's fused verdicts, as returned by bekind.
#[derive(Debug, Clone)]
struct FusedTextResult {
    id: String,
    scope: Option<FusedScope>,
    /// Activity verdicts (Luyi 2026-10-07): is_activity judgments.
    activity: Vec<FusedActivity>,
    entities: Vec<QueryEntity>,
}

#[derive(Debug, Clone)]
struct FusedActivity {
    is_activity: bool,
}

#[derive(Debug, Clone)]
struct FusedScope {
    activity_phrase: String,
    temporal_words: Vec<String>,
    habitual: bool,
    where_phrase: String,
}

/// Parse bekind's `text_results` array. Pure: unit-testable, no I/O.
fn parse_text_results(response: &Value) -> Option<Vec<FusedTextResult>> {
    let arr = response.get("text_results")?.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let id = item
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let scope = item.get("scope").and_then(|s| {
            if s.is_null() {
                None
            } else {
                Some(FusedScope {
                    activity_phrase: s
                        .get("activity_phrase")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    temporal_words: s
                        .get("temporal_words")
                        .and_then(|v| v.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect()
                        })
                        .unwrap_or_default(),
                    habitual: s.get("habitual").and_then(|v| v.as_bool()).unwrap_or(false),
                    where_phrase: s
                        .get("where_phrase")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                })
            }
        });
        let entities = item
            .get("entities")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|e| {
                        Some(QueryEntity {
                            text: e.get("text")?.as_str()?.to_string(),
                            kind: e
                                .get("kind")
                                .and_then(|v| v.as_str())
                                .unwrap_or("thing")
                                .to_string(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        let activity = item
            .get("activity")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| {
                        Some(FusedActivity {
                            is_activity: v.get("is_activity").and_then(|b| b.as_bool()).unwrap_or(false),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        out.push(FusedTextResult {
            id,
            scope,
            activity,
            entities,
        });
    }
    Some(out)
}

/// Temporal question words ("when", "what time", ...) ask for a time.
/// Purely local match on the lowered question; behood judges these as
/// time-seeking and lint-ai uses the kind for answer-kind filtering.
/// (Moved from `scripts/behood_query.py` when the Python layer became
/// parse-only; the text stays lowercase exactly as before.)
fn temporal_question_entity(question: &str) -> Option<QueryEntity> {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        regex::Regex::new(r"\b(when|what time|how long|what date|which date|what day|which day)\b")
            .expect("temporal question-word regex")
    });
    let lowered = question.to_lowercase();
    re.find(&lowered).map(|m| QueryEntity {
        text: m.as_str().to_string(),
        kind: "time".to_string(),
    })
}

/// Analyze a question with behood, returning (text, kind) pairs.
///
/// One fused bekind call (tag→chunk→judge internally). Fail-open: `None`
/// from the daemon yields an empty vec — callers fall back to heuristics.
pub fn analyze_query_entities(question: &str) -> Vec<QueryEntity> {
    let (_, entities) = analyze_query_semantics(question);
    entities
}

/// Scope verdicts + query entities for one question, via one fused bekind
/// call.
///
/// Fail-open: ([], []) on any daemon failure; callers fall back to
/// heuristics. The temporal question-word entity ("when"/"time") is
/// prepended locally, winning text ties, as the old bridge did.
pub fn analyze_query_semantics(question: &str) -> (Vec<ScopeVerdict>, Vec<QueryEntity>) {
    let results = match BekindDaemon::global().judge_texts(&[("q".to_string(), question, true)]) {
        Some(results) => results,
        None => return (Vec::new(), Vec::new()),
    };
    let result = match results.into_iter().next() {
        Some(r) => r,
        None => return (Vec::new(), Vec::new()),
    };
    let scope = result
        .scope
        .map(|s| ScopeVerdict {
            id: "s:0".to_string(),
            activity_phrase: s.activity_phrase,
            temporal_words: s.temporal_words,
            habitual: s.habitual,
            where_phrase: s.where_phrase,
        })
        .into_iter()
        .collect();
    let mut entities = result.entities;
    if let Some(temporal) = temporal_question_entity(question) {
        entities.insert(0, temporal);
        let mut seen = HashSet::new();
        entities.retain(|e| seen.insert(e.text.clone()));
    }
    (scope, entities)
}

/// Extract person names from behood's query-time entities.
///
/// Returns the person-kind entity texts, cleaned of determiners and
/// quantifiers that the noun chunker may include (e.g. "both Jean").
/// Deduplicated after cleaning.
pub fn query_persons(entities: &[QueryEntity]) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for e in entities {
        if e.kind != "person" {
            continue;
        }
        let cleaned = clean_person_text(&e.text);
        if !cleaned.is_empty() && seen.insert(cleaned.clone()) {
            out.push(cleaned);
        }
    }
    out
}

fn clean_person_text(text: &str) -> String {
    // Strip leading determiners/quantifiers ("both Jean" -> "Jean").
    let lower = text.to_lowercase();
    for prefix in ["both ", "the ", "a ", "an "] {
        if lower.starts_with(prefix) {
            return text[prefix.len()..].trim().to_string();
        }
    }
    text.trim().to_string()
}

/// Whether any entity has the given kind.
pub fn query_has_kind(entities: &[QueryEntity], kind: &str) -> bool {
    entities.iter().any(|e| e.kind == kind)
}

/// A temporal-scope verdict judged by bekind for one text span.
#[derive(Debug, Clone)]
pub struct ScopeVerdict {
    /// Caller-assigned id, echoed back ("s:{i}").
    pub id: String,
    /// The activity verb phrase extracted from the text ("eat out"),
    /// empty when the text names no activity verb. Deterministic and
    /// rule-based; never a guess.
    pub activity_phrase: String,
    /// Canonicalized temporal words (closed 7-day set: "weekend"/"weekday").
    pub temporal_words: Vec<String>,
    /// Whether the text describes a habitual/recurring activity.
    pub habitual: bool,
    /// The where of the activity (venue/location phrase). Luyi 2026-10-07:
    /// where should be a kind.
    pub where_phrase: String,
}

/// Scope verdicts for raw text spans, via one fused bekind call.
///
/// Fail-open: any daemon failure yields an empty vec, and the caller
/// emits no tags. Search never breaks because of scope verdicts.
pub fn analyze_scope_verdicts(texts: &[&str]) -> Vec<ScopeVerdict> {
    if texts.is_empty() || !is_enabled() {
        return Vec::new();
    }
    let inputs: Vec<(String, &str, bool)> = texts
        .iter()
        .enumerate()
        .map(|(i, t)| (format!("s:{i}"), *t, true))
        .collect();
    let results = match BekindDaemon::global().judge_texts(&inputs) {
        Some(results) => results,
        None => return Vec::new(),
    };
    results
        .into_iter()
        .filter_map(|r| {
            r.scope.map(|s| ScopeVerdict {
                id: r.id,
                activity_phrase: s.activity_phrase,
                temporal_words: s.temporal_words,
                habitual: s.habitual,
                where_phrase: s.where_phrase,
            })
        })
        .collect()
}

/// One descriptor's kind judgment, as it appears in the judged span.
#[derive(Debug, Clone)]
pub struct KindHit {
    /// Entity text as it appears in the judged span.
    pub text: String,
    /// bekind's ontological kind ("herb", "food", "thing", ...).
    pub kind: String,
}

/// bekind kind verdicts for one text span: every entity's kind.
#[derive(Debug, Clone)]
pub struct KindVerdict {
    /// Caller-assigned id, echoed back ("k:{i}").
    pub id: String,
    /// Per-entity (text, kind) judgments.
    pub kinds: Vec<KindHit>,
}

/// Kind verdicts for raw text spans: one fused bekind call for all texts.
///
/// Fail-open: any daemon failure yields an empty vec, and the caller
/// emits no kind tags. Search never breaks because of kind verdicts.
pub fn analyze_kind_verdicts(texts: &[&str]) -> Vec<KindVerdict> {
    if texts.is_empty() || !is_enabled() {
        return Vec::new();
    }
    let inputs: Vec<(String, &str, bool)> = texts
        .iter()
        .enumerate()
        .map(|(i, t)| (format!("k:{i}"), *t, false))
        .collect();
    let results = match BekindDaemon::global().judge_texts(&inputs) {
        Some(results) => results,
        None => return Vec::new(),
    };
    results
        .into_iter()
        .map(|r| KindVerdict {
            id: r.id,
            kinds: r
                .entities
                .into_iter()
                .map(|e| KindHit {
                    text: e.text,
                    kind: e.kind,
                })
                .collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- pure-function tests (no I/O) ----

    #[test]
    fn temporal_question_entity_matches_variants() {
        let e = temporal_question_entity("When did Jean visit Paris?").expect("when");
        assert_eq!(e.text, "when");
        assert_eq!(e.kind, "time");
        let e = temporal_question_entity("how long does it take?").expect("how long");
        assert_eq!(e.text, "how long");
        assert!(temporal_question_entity("What color is it?").is_none());
    }

    #[test]
    fn query_persons_cleans_determiners_and_dedupes() {
        let entities = vec![
            QueryEntity {
                text: "both Jean".to_string(),
                kind: "person".to_string(),
            },
            QueryEntity {
                text: "Jean".to_string(),
                kind: "person".to_string(),
            },
            QueryEntity {
                text: "Paris".to_string(),
                kind: "place".to_string(),
            },
        ];
        assert_eq!(query_persons(&entities), vec!["Jean".to_string()]);
    }

    #[test]
    fn query_has_kind_matches() {
        let entities = vec![QueryEntity {
            text: "Paris".to_string(),
            kind: "place".to_string(),
        }];
        assert!(query_has_kind(&entities, "place"));
        assert!(!query_has_kind(&entities, "person"));
    }

    #[test]
    fn parse_text_results_parses_fused_response() {
        let response = json!({
            "text_results": [
                {
                    "id": "q",
                    "scope": {
                        "id": "q:s:0",
                        "activity_phrase": "eat out",
                        "temporal_words": ["weekend"],
                        "habitual": true,
                    },
                    "entities": [
                        {"text": "Jean", "kind": "person"},
                        {"text": "cilantro", "kind": "herb"},
                    ],
                },
                {
                    "id": "k:0",
                    "entities": [{"text": "Paris", "kind": "place"}],
                },
            ],
        });
        let results = parse_text_results(&response).expect("parse");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "q");
        let scope = results[0].scope.as_ref().expect("scope");
        assert_eq!(scope.activity_phrase, "eat out");
        assert_eq!(scope.temporal_words, vec!["weekend".to_string()]);
        assert!(scope.habitual);
        assert_eq!(results[0].entities.len(), 2);
        assert_eq!(results[0].entities[0].text, "Jean");
        assert_eq!(results[0].entities[0].kind, "person");
        // No scope key → None scope, entities still parsed.
        assert!(results[1].scope.is_none());
        assert_eq!(results[1].entities[0].kind, "place");
    }

    #[test]
    fn parse_text_results_rejects_missing_array() {
        assert!(parse_text_results(&json!({})).is_none());
        assert!(parse_text_results(&json!({"text_results": []}))
            .unwrap()
            .is_empty());
    }

    // ---- daemon tests (fake bekind --serve speaking the fused protocol) ----

    fn unique_temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "bekind_fused_test_{}_{}_{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// Fake `bekind --serve`: reads fused `texts` requests, returns canned
    /// `text_results` (weekend scope + one herb entity per text).
    fn write_fake_fused_bekind(dir: &std::path::Path) -> std::path::PathBuf {
        let path = dir.join("fake_bekind_fused.py");
        let count_path = dir.join("bekind_count.txt");
        let script = format!(
            r#"import json, sys
count_path = {count_path:?}
n = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    n += 1
    open(count_path, "w").write(str(n))
    payload = json.loads(line)
    results = []
    for t in payload.get("texts", []):
        scope = None
        if t.get("with_scope"):
            scope = {{"id": t["id"] + ":s:0", "activity_phrase": "",
                      "temporal_words": ["weekend"], "habitual": False}}
        results.append({{
            "id": t["id"],
            "scope": scope,
            "entities": [{{"text": "cilantro", "kind": "herb"}}],
        }})
    sys.stdout.write(json.dumps({{"text_results": results}}) + "\n")
    sys.stdout.flush()
"#
        );
        std::fs::write(&path, script).expect("write fake bekind script");
        path
    }

    fn test_judge_daemon(script: std::path::PathBuf) -> BekindDaemon {
        // The daemon spawns the script directly; use python3 as the binary
        // with the script as argv (mirrors JsonLinesDaemon::new_command).
        BekindDaemon {
            daemon: JsonLinesDaemon::new_command(
                "fake-bekind",
                vec!["python3".to_string(), script.to_string_lossy().into_owned()],
            ),
            cooldown: ServeCooldown::new(),
        }
    }

    #[test]
    fn fused_daemon_returns_per_text_verdicts() {
        let dir = unique_temp_dir("fused");
        let daemon = test_judge_daemon(write_fake_fused_bekind(&dir));
        set_enabled(true);
        let results = daemon.judge_texts(&[
            ("q".to_string(), "weekend cilantro?", true),
            ("k:0".to_string(), "cilantro", false),
        ]);
        set_enabled(false);
        let results = results.expect("fused judge should answer");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "q");
        assert_eq!(
            results[0].scope.as_ref().unwrap().temporal_words,
            vec!["weekend".to_string()]
        );
        assert_eq!(results[0].entities[0].text, "cilantro");
        assert_eq!(results[0].entities[0].kind, "herb");
        assert!(results[1].scope.is_none());
        // One fused call for both texts.
        let count = std::fs::read_to_string(dir.join("bekind_count.txt")).expect("count");
        assert_eq!(count, "1");
    }

    #[test]
    fn fused_daemon_fails_open_on_bad_binary() {
        let daemon = BekindDaemon::new(std::path::PathBuf::from("/nonexistent/bekind"));
        assert!(
            daemon
                .judge_texts(&[("q".to_string(), "anything", true)])
                .is_none(),
            "bad binary must fail open"
        );
    }
}
