//! Query-time behood: parse + judge (bekind daemon).
//!
//! Luyi's design: "the behood provide people as the source, then we have
//! place and thing." At query time, behood judges the question's entities
//! and returns (text, kind) pairs. Lint-ai uses the text for matching and
//! the kind for filtering in structured question analysis.
//!
//! Two long-lived children, each doing one job:
//!
//! - Parse backend (selectable via [`BehoodParseProvider`]): the default
//!   `scripts/behood_query.py --serve` (parse daemon) is pure spaCy parsing,
//!   one pass per text over the already-loaded model; the
//!   [`BehoodParseProvider::Heuristic`] backend is the pure-Rust
//!   [`crate::heuristic_parse`] chunker — no Python, no spaCy. Both emit
//!   bekind-ready descriptors (`mentions`, `np_mentions`) and never spawn
//!   a subprocess.
//! - `bekind --serve` (judge daemon): pure judgment over descriptors, no
//!   parsing, no spaCy. Owned directly by this module — Rust passes the
//!   payload straight to the binary instead of routing through Python.
//!
//! Spawning a fresh Python interpreter plus the spaCy model load costs
//! seconds, and spawning the bekind binary per request cost ~13ms; the
//! daemons pay each cost once per process. Fail-open by construction:
//! every daemon failure yields `None` and the caller falls back to a
//! one-shot chain (or empty) exactly as before. Behood owns judgment; the
//! caller owns knowledge.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::daemon::JsonLinesDaemon;
use crate::segments::relations::python_executable;

/// How long one daemon round-trip may take before the caller fails open.
const DAEMON_TIMEOUT: Duration = Duration::from_secs(30);

/// How long a serve failure suppresses respawn attempts. The backend does
/// not heal in milliseconds; fallbacks cover the gap.
const SERVE_FAILURE_COOLDOWN: Duration = Duration::from_secs(30);

/// A (text, kind) pair judged by behood at query time.
#[derive(Debug, Clone)]
pub struct QueryEntity {
    /// The entity text as it appears in the question.
    pub text: String,
    /// Behood's ontological kind: person, place, org, event, work, food,
    /// herb (admitted closed set), thing.
    pub kind: String,
}

/// One text's parse result: bekind-ready descriptors from the parse daemon.
#[derive(Debug, Clone)]
pub struct ParsedText {
    /// Caller-assigned id, echoed back ("p:{index}").
    pub id: String,
    /// PROPN personhood mention descriptors (bekind Mention pieces).
    pub mentions: Value,
    /// Noun-phrase descriptors (bekind NpMention pieces).
    pub np_mentions: Value,
}

/// Suppresses respawn attempts for a while after a serve failure.
#[derive(Clone)]
struct ServeCooldown {
    last_failure: Arc<Mutex<Option<Instant>>>,
}

impl ServeCooldown {
    fn new() -> Self {
        ServeCooldown {
            last_failure: Arc::new(Mutex::new(None)),
        }
    }

    /// `false` while cooling down (the caller fails open); `true` when the
    /// daemon may be asked to serve.
    fn gate(&self) -> bool {
        // The mutex is never held across `query`, so lock ordering is safe.
        if let Ok(last) = self.last_failure.lock() {
            if let Some(failed_at) = *last {
                if failed_at.elapsed() < SERVE_FAILURE_COOLDOWN {
                    return false;
                }
            }
        }
        true
    }

    fn note_failure(&self) {
        if let Ok(mut last) = self.last_failure.lock() {
            *last = Some(Instant::now());
        }
    }

    fn note_success(&self) {
        if let Ok(mut last) = self.last_failure.lock() {
            *last = None;
        }
    }
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

/// Locate `scripts/behood_query.py` relative to the crate root.
fn script_path() -> Option<PathBuf> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = Path::new(manifest_dir).join("scripts/behood_query.py");
    if path.is_file() {
        Some(path)
    } else {
        None
    }
}

/// Build the bekind discourse Request for one question's descriptors,
/// optionally carrying the question itself as a scope text so scope
/// verdicts come back in the same judge call.
fn discourse_request(mentions: &Value, np_mentions: &Value, scope_text: Option<&str>) -> Value {
    let mut request = json!({
        "strategy": "discourse",
        "mentions": mentions,
        "chunks": [],
        "np_mentions": np_mentions,
        "context": {"speaker_names": []},
    });
    if let Some(text) = scope_text {
        request["scope_texts"] = json!([{"id": "s:0", "text": text}]);
    }
    request
}

/// Whether both descriptor arrays are empty (or absent).
fn descriptors_empty(mentions: &Value, np_mentions: &Value) -> bool {
    mentions
        .as_array()
        .map(|a| a.is_empty())
        .unwrap_or(true)
        && np_mentions
            .as_array()
            .map(|a| a.is_empty())
            .unwrap_or(true)
}

/// Map bekind phrase/person verdicts back to (text, kind) entities using
/// the descriptors the parse daemon produced (ids echoed by bekind).
/// Deduplicated by text, keeping first occurrence.
fn map_bekind_entities(
    response: &Value,
    mentions: &Value,
    np_mentions: &Value,
) -> Vec<QueryEntity> {
    let mut id_to_text: HashMap<&str, &str> = HashMap::new();
    for arr in [np_mentions, mentions] {
        if let Some(items) = arr.as_array() {
            for d in items {
                if let (Some(id), Some(text)) = (
                    d.get("id").and_then(|v| v.as_str()),
                    d.get("text").and_then(|v| v.as_str()),
                ) {
                    id_to_text.insert(id, text);
                }
            }
        }
    }
    let mut entities = Vec::new();
    if let Some(arr) = response
        .get("phrase_verdicts")
        .and_then(|v| v.as_array())
    {
        for v in arr {
            if !v
                .get("is_entity_mention")
                .and_then(|b| b.as_bool())
                .unwrap_or(false)
            {
                continue;
            }
            let text = v
                .get("id")
                .and_then(|id| id.as_str())
                .and_then(|id| id_to_text.get(id))
                .copied()
                .unwrap_or("");
            let kind = v
                .get("kind")
                .and_then(|k| k.as_str())
                .unwrap_or("thing");
            if !text.is_empty() {
                entities.push(QueryEntity {
                    text: text.to_string(),
                    kind: kind.to_string(),
                });
            }
        }
    }
    if let Some(arr) = response.get("verdicts").and_then(|v| v.as_array()) {
        for v in arr {
            if !v
                .get("is_person")
                .and_then(|b| b.as_bool())
                .unwrap_or(false)
            {
                continue;
            }
            let text = v
                .get("id")
                .and_then(|id| id.as_str())
                .and_then(|id| id_to_text.get(id))
                .copied()
                .unwrap_or("");
            if !text.is_empty() && !entities.iter().any(|e| e.text == text) {
                entities.push(QueryEntity {
                    text: text.to_string(),
                    kind: "person".to_string(),
                });
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    entities.retain(|e| seen.insert(e.text.clone()));
    entities
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

/// Scope verdicts + entities from one judge Response: scope verdicts when
/// requested, entities mapped from the verdicts, and the local temporal
/// question-word entity prepended (winning text ties, as the old bridge
/// did — it led the entity list there too).
fn scope_and_entities(
    response: &Value,
    mentions: &Value,
    np_mentions: &Value,
    question: &str,
    with_scope: bool,
) -> (Vec<ScopeVerdict>, Vec<QueryEntity>) {
    let scope = if with_scope {
        parse_scope_verdicts(response)
    } else {
        Vec::new()
    };
    let mut entities = map_bekind_entities(response, mentions, np_mentions);
    if let Some(temporal) = temporal_question_entity(question) {
        entities.insert(0, temporal);
        let mut seen = std::collections::HashSet::new();
        entities.retain(|e| seen.insert(e.text.clone()));
    }
    (scope, entities)
}

// ---------------------------------------------------------------------------
// Daemons.
// ---------------------------------------------------------------------------

/// Long-lived parse daemon: `scripts/behood_query.py --serve`.
///
/// Pure spaCy parsing, one pass per text over the already-loaded model.
/// It never spawns a subprocess and never touches the bekind binary;
/// judgment happens in [`BekindDaemon`], owned directly by this module.
#[derive(Clone)]
pub struct BehoodQueryDaemon {
    daemon: JsonLinesDaemon,
    cooldown: ServeCooldown,
}

impl BehoodQueryDaemon {
    /// The process-wide parse daemon. Used by the production query path;
    /// tests construct their own via [`BehoodQueryDaemon::new`] for
    /// isolation.
    pub fn global() -> &'static BehoodQueryDaemon {
        static DAEMON: OnceLock<BehoodQueryDaemon> = OnceLock::new();
        DAEMON.get_or_init(|| {
            let script = script_path().unwrap_or_default();
            BehoodQueryDaemon::new(script, python_executable())
        })
    }

    /// A daemon over an explicit script (tests, benchmarks).
    pub fn new(script: PathBuf, python: String) -> Self {
        BehoodQueryDaemon {
            daemon: JsonLinesDaemon::new("behood-query", script, python),
            cooldown: ServeCooldown::new(),
        }
    }

    /// Start the child now so the first real query does not pay the spawn
    /// cost. Best-effort: failures are silent; queries fall back.
    pub fn prewarm(&self) {
        self.daemon.prewarm();
    }

    /// One raw JSON line through the daemon with serve-failure cooldown.
    /// Returns `None` on cooldown, serialization failure, or any daemon
    /// failure (including lock contention — the daemon is a fast path,
    /// never a queue). Callers fail open on `None`.
    fn raw_request(&self, payload: Value, timeout: Duration) -> Option<String> {
        if !self.cooldown.gate() {
            return None;
        }
        let line = serde_json::to_string(&payload).ok()?;
        let response = match self.daemon.query(&line, timeout) {
            Some(response) => response,
            None => {
                self.cooldown.note_failure();
                return None;
            }
        };
        self.cooldown.note_success();
        Some(response)
    }

    /// Parse raw texts into bekind-ready descriptors via the daemon.
    ///
    /// Returns `None` on any failure (including lock contention); the
    /// caller fails open. `Some(vec)` is authoritative: a text that failed
    /// to parse is simply absent, each entry carrying its "p:{index}" id.
    pub fn parse_texts(&self, texts: &[&str], timeout: Duration) -> Option<Vec<ParsedText>> {
        if texts.is_empty() {
            return Some(Vec::new());
        }
        let parse_texts: Vec<Value> = texts
            .iter()
            .enumerate()
            .map(|(i, text)| json!({"id": format!("p:{i}"), "text": text}))
            .collect();
        let payload = json!({ "parse_texts": parse_texts });
        let response = self.raw_request(payload, timeout)?;
        let parsed: Value = serde_json::from_str(&response).ok()?;
        let mut out = Vec::new();
        for item in parsed
            .get("parsed")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
        {
            out.push(ParsedText {
                id: item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                mentions: item.get("mentions").cloned().unwrap_or(json!([])),
                np_mentions: item.get("np_mentions").cloned().unwrap_or(json!([])),
            });
        }
        Some(out)
    }

    #[cfg(test)]
    pub fn kill_child_for_test(&self) {
        self.daemon.kill_child_for_test();
    }
}

/// Long-lived judge daemon: `bekind --serve`, owned directly by lint-ai.
///
/// Pure judgment over descriptors: no parsing, no spaCy. One Request JSON
/// line in, one Response JSON line out. This is what lets Rust pass the
/// payload straight to the binary instead of routing through Python.
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
                vec![
                    binary.to_string_lossy().into_owned(),
                    "--serve".to_string(),
                ],
            ),
            cooldown: ServeCooldown::new(),
        }
    }

    /// Start the child now so the first real query does not pay the spawn
    /// cost. Best-effort: failures are silent; queries fall back.
    pub fn prewarm(&self) {
        self.daemon.prewarm();
    }

    /// Judge one bekind Request: one JSON line through the daemon, one
    /// Response JSON back. Returns `None` on any failure (including lock
    /// contention — the daemon is a fast path, never a queue); the caller
    /// fails open.
    pub fn judge(&self, request: &Value, timeout: Duration) -> Option<Value> {
        if !self.cooldown.gate() {
            return None;
        }
        let line = serde_json::to_string(request).ok()?;
        let response = match self.daemon.query(&line, timeout) {
            Some(response) => response,
            None => {
                self.cooldown.note_failure();
                return None;
            }
        };
        self.cooldown.note_success();
        serde_json::from_str(&response).ok()
    }

    #[cfg(test)]
    pub fn kill_child_for_test(&self) {
        self.daemon.kill_child_for_test();
    }
}

// ---------------------------------------------------------------------------
// Query composition: parse, then judge.
// ---------------------------------------------------------------------------

/// Which backend parses raw text into bekind descriptors.
///
/// - [`BehoodParseProvider::Spacy`]: `scripts/behood_query.py --serve`
///   (spaCy). Default; matches the published benchmark numbers.
/// - [`BehoodParseProvider::Heuristic`]: the pure-Rust
///   [`crate::heuristic_parse::heuristic_parse_texts`] chunker. No Python
///   process, no spaCy model — the query path is fully spaCy-free.
///
/// The judge backend is always the bekind daemon; only the parse step is
/// selectable. Set once per process at startup via
/// [`set_behood_parse_provider`]; later calls are ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum, serde::Serialize)]
pub enum BehoodParseProvider {
    #[default]
    Spacy,
    Heuristic,
}

static PARSE_PROVIDER: OnceLock<BehoodParseProvider> = OnceLock::new();

/// Select the behood parse backend for this process. Call once at startup
/// (server, benchmark); subsequent calls are ignored. Defaults to
/// [`BehoodParseProvider::Spacy`].
pub fn set_behood_parse_provider(provider: BehoodParseProvider) {
    let _ = PARSE_PROVIDER.set(provider);
}

/// Resolve the effective parse backend: an explicit `--parse-provider`
/// wins; otherwise the backend follows the NER provider (heuristic NER ⇒
/// heuristic parse, fully spaCy-free on the query path).
pub fn resolve_parse_provider(
    explicit: Option<BehoodParseProvider>,
    heuristic_ner: bool,
) -> BehoodParseProvider {
    explicit.unwrap_or(if heuristic_ner {
        BehoodParseProvider::Heuristic
    } else {
        BehoodParseProvider::Spacy
    })
}

fn behood_parse_provider() -> BehoodParseProvider {
    PARSE_PROVIDER.get().copied().unwrap_or_default()
}

/// Parse texts into descriptors via the selected backend: the spaCy parse
/// daemon, or the pure-Rust heuristic builder (no subprocess, no model).
fn parse_texts_via_provider(
    provider: BehoodParseProvider,
    parse: &BehoodQueryDaemon,
    texts: &[&str],
    timeout: Duration,
) -> Option<Vec<ParsedText>> {
    match provider {
        BehoodParseProvider::Heuristic => {
            Some(crate::heuristic_parse::heuristic_parse_texts(texts))
        }
        BehoodParseProvider::Spacy => parse.parse_texts(texts, timeout),
    }
}

/// Parse + judge one question through the daemon pair: the parse daemon
/// turns the text into descriptors (one spaCy pass), the judge daemon
/// judges them (one bekind call, no subprocess spawn anywhere).
///
/// `with_scope` carries the question as a scope text in the same judge
/// call, so scope verdicts and entities come back together. Returns `None`
/// when the pair cannot serve; the caller fails open or one-shots.
fn query_judge(
    parse: &BehoodQueryDaemon,
    judge: &BekindDaemon,
    question: &str,
    with_scope: bool,
) -> Option<(Vec<ScopeVerdict>, Vec<QueryEntity>)> {
    let parsed =
        parse_texts_via_provider(behood_parse_provider(), parse, &[question], DAEMON_TIMEOUT)?;
    let first = parsed.iter().find(|p| p.id == "p:0");
    let (mentions, np_mentions) = match first {
        Some(p) => (p.mentions.clone(), p.np_mentions.clone()),
        None => (json!([]), json!([])),
    };
    // No descriptors and no scope to judge: nothing for bekind to do.
    // (Preserves the old bridge's early return; the temporal entity is
    // dropped with it, exactly as before.)
    if !with_scope && descriptors_empty(&mentions, &np_mentions) {
        return Some((Vec::new(), Vec::new()));
    }
    let request = discourse_request(&mentions, &np_mentions, with_scope.then_some(question));
    let response = judge.judge(&request, DAEMON_TIMEOUT)?;
    Some(scope_and_entities(
        &response,
        &mentions,
        &np_mentions,
        question,
        with_scope,
    ))
}

/// One-shot fallback: descriptors (spaCy one-shot, or the pure-Rust
/// heuristic builder when the heuristic parse backend is selected) piped
/// into one-shot `bekind`. Used only when the daemon pair cannot serve.
fn oneshot_query_judge(
    question: &str,
    with_scope: bool,
) -> Option<(Vec<ScopeVerdict>, Vec<QueryEntity>)> {
    // Heuristic parse backend: no Python at all — descriptors come from
    // the Rust builder, only the judge is a one-shot subprocess.
    if behood_parse_provider() == BehoodParseProvider::Heuristic {
        let binary = bekind_bin()?;
        let parsed = crate::heuristic_parse::heuristic_parse_texts(&[question]);
        let first = parsed.iter().find(|p| p.id == "p:0");
        let (mentions, np_mentions) = match first {
            Some(p) => (p.mentions.clone(), p.np_mentions.clone()),
            None => (json!([]), json!([])),
        };
        if !with_scope && descriptors_empty(&mentions, &np_mentions) {
            return Some((Vec::new(), Vec::new()));
        }
        let request = discourse_request(&mentions, &np_mentions, with_scope.then_some(question));
        let response = oneshot_judge(&binary, &request)?;
        return Some(scope_and_entities(
            &response,
            &mentions,
            &np_mentions,
            question,
            with_scope,
        ));
    }
    let script = script_path()?;
    let binary = bekind_bin()?;
    let parse_out = Command::new(python_executable())
        .arg(&script)
        .arg("--parse")
        .arg(question)
        .output()
        .ok()?;
    if !parse_out.status.success() {
        return None;
    }
    let parsed: Value = serde_json::from_str(&String::from_utf8_lossy(&parse_out.stdout)).ok()?;
    let first = parsed
        .get("parsed")
        .and_then(|v| v.as_array())
        .and_then(|a| {
            a.iter()
                .find(|p| p.get("id").and_then(|v| v.as_str()) == Some("p:0"))
        });
    let (mentions, np_mentions) = match first {
        Some(p) => (
            p.get("mentions").cloned().unwrap_or(json!([])),
            p.get("np_mentions").cloned().unwrap_or(json!([])),
        ),
        None => (json!([]), json!([])),
    };
    if !with_scope && descriptors_empty(&mentions, &np_mentions) {
        return Some((Vec::new(), Vec::new()));
    }
    let request = discourse_request(&mentions, &np_mentions, with_scope.then_some(question));
    let response = oneshot_judge(&binary, &request)?;
    Some(scope_and_entities(
        &response,
        &mentions,
        &np_mentions,
        question,
        with_scope,
    ))
}

/// One bekind invocation: Request JSON on stdin, Response JSON on stdout.
/// Returns `None` on any failure.
fn oneshot_judge(binary: &PathBuf, request: &Value) -> Option<Value> {
    let mut child = Command::new(binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let payload = serde_json::to_vec(request).ok()?;
    child.stdin.as_mut()?.write_all(&payload).ok()?;
    // Close stdin so the one-shot child sees EOF.
    let _ = child.stdin.take();
    let out = child.wait_with_output().ok()?;
    if !out.status.success() {
        return None;
    }
    serde_json::from_str(&String::from_utf8_lossy(&out.stdout)).ok()
}

/// Analyze a question with behood, returning (text, kind) pairs.
///
/// Daemon pair first (parse, then judge). `None` from the pair means the
/// daemons themselves failed (not "no entities"), and only then do we
/// fall back to a one-shot chain. An empty vec is authoritative — callers
/// fall back to heuristics, never to another subprocess. Fail-open
/// throughout: callers fall back to heuristics.
pub fn analyze_query_entities(question: &str) -> Vec<QueryEntity> {
    if let Some((_, entities)) = query_judge(
        BehoodQueryDaemon::global(),
        BekindDaemon::global(),
        question,
        false,
    ) {
        return entities;
    }
    oneshot_query_judge(question, false)
        .map(|(_, entities)| entities)
        .unwrap_or_default()
}

/// Scope verdicts + query entities for one question: one spaCy parse plus
/// one bekind judge call, through the daemon pair.
///
/// Daemon pair first, then a one-shot chain — mirroring
/// [`analyze_query_entities`]' resilience. Fail-open: ([], []) on any
/// failure; callers emit no tags.
pub fn analyze_query_semantics(question: &str) -> (Vec<ScopeVerdict>, Vec<QueryEntity>) {
    if let Some(pair) = query_judge(
        BehoodQueryDaemon::global(),
        BekindDaemon::global(),
        question,
        true,
    ) {
        return pair;
    }
    oneshot_query_judge(question, true).unwrap_or_default()
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

// ---------------------------------------------------------------------------
// Temporal-scope verdicts (bekind's scope layer).
//
// A scope verdict judges one raw text span: the activity it describes, its
// canonicalized temporal words (closed 7-day set only: "weekend"/"weekday"),
// and whether it is habitual. Lint-ai turns these verdicts into
// definitional semantic tags (Luyi 2026-09-28): index-time and query-time
// SHOULD matches inside tantivy BM25 — never a filter, never a bonus.
// Activity compatibility (running ⊂ exercise) is deliberately out of
// scope here; that stays caller-side knowledge work.
// ---------------------------------------------------------------------------

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
}

/// Scope verdicts for raw text spans, straight from the judge daemon.
///
/// Scope needs no parsing — raw texts go to bekind directly, so the parse
/// daemon is not involved at all.
///
/// Fail-open: any daemon failure yields an empty vec, and the caller
/// emits no scope tags. Search never breaks because of scope. There is
/// deliberately no one-shot subprocess fallback (unchanged contract):
/// per-batch one-shots would pay a process spawn per candidate batch; the
/// daemon is the scope path.
pub fn analyze_scope_verdicts(texts: &[&str]) -> Vec<ScopeVerdict> {
    if texts.is_empty() {
        return Vec::new();
    }
    let scope_texts: Vec<Value> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| json!({"id": format!("s:{i}"), "text": text}))
        .collect();
    BekindDaemon::global()
        .judge(&json!({ "scope_texts": scope_texts }), DAEMON_TIMEOUT)
        .map(|response| parse_scope_verdicts(&response))
        .unwrap_or_default()
}

fn parse_scope_verdicts(response: &Value) -> Vec<ScopeVerdict> {
    let mut verdicts = Vec::new();
    if let Some(arr) = response.get("scope_verdicts").and_then(|v| v.as_array()) {
        for item in arr {
            let id = item
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let activity_phrase = item
                .get("activity_phrase")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let temporal_words = item
                .get("temporal_words")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str())
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            let habitual = item
                .get("habitual")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            verdicts.push(ScopeVerdict {
                id,
                activity_phrase,
                temporal_words,
                habitual,
            });
        }
    }
    verdicts
}

// ---------------------------------------------------------------------------
// Closed-set kind verdicts (bekind's kind layer).
//
// A kind verdict judges one raw text span's noun-phrase descriptors,
// reporting every descriptor's ontological kind. Lint-ai turns admitted
// closed-set kinds into definitional semantic tags (Luyi 2026-09-28):
// index-time and query-time SHOULD matches inside tantivy BM25 — never a
// filter, never a bonus. Currently only "culinary_herb" -> kind "herb" is
// admitted. Per-set admission: each new category needs its own explicit
// admission; this machinery does not generalize them automatically.
// ---------------------------------------------------------------------------

/// One descriptor's kind judgment within a [`KindVerdict`].
#[derive(Debug, Clone)]
pub struct KindHit {
    /// Descriptor text as it appears in the judged span.
    pub text: String,
    /// bekind's ontological kind ("herb", "food", "thing", ...).
    pub kind: String,
}

/// bekind kind verdicts for one text span: every descriptor's kind.
///
/// Unlike [`QueryEntity`] (entity mentions only), this includes
/// descriptors that are NOT entity mentions: bekind reports `kind_of`
/// for rejected mentions too (e.g. a POS-mistagged "cilantro" still
/// judges herb-kind), and the tag only needs the kind signal.
#[derive(Debug, Clone)]
pub struct KindVerdict {
    /// Caller-assigned id, echoed back ("k:{i}").
    pub id: String,
    /// Per-descriptor (text, kind) judgments.
    pub kinds: Vec<KindHit>,
}

/// Kind verdicts for raw text spans: parse once per text (batched), judge
/// once for all descriptors, split per text.
///
/// Fail-open: any daemon failure yields an empty vec, and the caller
/// emits no kind tags. Search never breaks because of kind verdicts.
/// Like scope, there is deliberately no one-shot subprocess fallback
/// (unchanged contract): the daemon pair is the path.
pub fn analyze_kind_verdicts(texts: &[&str]) -> Vec<KindVerdict> {
    if texts.is_empty() {
        return Vec::new();
    }
    let parsed = match parse_texts_via_provider(
        behood_parse_provider(),
        BehoodQueryDaemon::global(),
        texts,
        DAEMON_TIMEOUT,
    ) {
        Some(parsed) => parsed,
        None => return Vec::new(),
    };
    // Merge every text's noun-phrase descriptors into one judge call,
    // prefixing descriptor ids as "k:{text_index}:{descriptor_id}" so
    // verdicts map back per text. Only np_mentions: kind judges
    // descriptors, not personhood mentions — same as the old bridge.
    let mut all_np: Vec<Value> = Vec::new();
    let mut id_to_text: HashMap<String, String> = HashMap::new();
    for p in &parsed {
        let text_index: usize = match p.id.strip_prefix("p:").and_then(|s| s.parse().ok()) {
            Some(i) => i,
            None => continue,
        };
        if text_index >= texts.len() {
            continue;
        }
        if let Some(arr) = p.np_mentions.as_array() {
            for d in arr {
                let did = d.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let new_id = format!("k:{text_index}:{did}");
                let text = d
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                id_to_text.insert(new_id.clone(), text);
                let mut renamed = d.clone();
                renamed["id"] = json!(new_id);
                all_np.push(renamed);
            }
        }
    }
    let response = match BekindDaemon::global().judge(
        &json!({
            "strategy": "discourse",
            "mentions": [],
            "chunks": [],
            "np_mentions": all_np,
            "context": {"speaker_names": []},
        }),
        DAEMON_TIMEOUT,
    ) {
        Some(response) => response,
        None => return Vec::new(),
    };
    // One KindVerdict per input text, in input order (empty kinds when a
    // text had no descriptors) — the same shape the old bridge produced.
    let per_text = split_kind_hits(&response, &id_to_text, texts.len());
    per_text
        .into_iter()
        .enumerate()
        .map(|(i, kinds)| KindVerdict {
            id: format!("k:{i}"),
            kinds,
        })
        .collect()
}

/// Split phrase verdicts with "k:{index}:..." ids back into per-text kind
/// hits, in input order. Pure: unit-testable, no I/O.
fn split_kind_hits(
    response: &Value,
    id_to_text: &HashMap<String, String>,
    text_count: usize,
) -> Vec<Vec<KindHit>> {
    let mut per_text: Vec<Vec<KindHit>> = vec![Vec::new(); text_count];
    let arr = match response.get("phrase_verdicts").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return per_text,
    };
    for v in arr {
        let vid = v.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let mut parts = vid.splitn(3, ':');
        if parts.next() != Some("k") {
            continue;
        }
        let index: usize = match parts.next().and_then(|s| s.parse().ok()) {
            Some(i) if i < text_count => i,
            _ => continue,
        };
        let text = match id_to_text.get(vid) {
            Some(text) => text.clone(),
            None => continue,
        };
        let kind = v
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("thing")
            .to_string();
        per_text[index].push(KindHit { text, kind });
    }
    per_text
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
        let e = temporal_question_entity("What time is dinner?").expect("what time");
        assert_eq!(e.text, "what time");
        assert_eq!(e.kind, "time");
        let e = temporal_question_entity("HOW LONG is the drive?").expect("how long");
        assert_eq!(e.text, "how long");
        assert!(temporal_question_entity("Who visited Paris?").is_none());
        assert!(temporal_question_entity("Which city is biggest?").is_none());
    }

    #[test]
    fn map_bekind_entities_maps_verdicts_to_text() {
        let mentions = json!([{"id": "m:0", "text": "Jean"}]);
        let np_mentions = json!([
            {"id": "q:0", "text": "cilantro"},
            {"id": "q:1", "text": "Paris"},
        ]);
        let response = json!({
            "phrase_verdicts": [
                {"id": "q:0", "is_entity_mention": true, "kind": "herb"},
                {"id": "q:1", "is_entity_mention": false, "kind": "thing"},
            ],
            "verdicts": [{"id": "m:0", "is_person": true}],
        });
        let entities = map_bekind_entities(&response, &mentions, &np_mentions);
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].text, "cilantro");
        assert_eq!(entities[0].kind, "herb");
        assert_eq!(entities[1].text, "Jean");
        assert_eq!(entities[1].kind, "person");
    }

    #[test]
    fn map_bekind_entities_dedups_person_double_count() {
        // "Jean" judged both as an entity mention and as a person: one entry.
        let mentions = json!([{"id": "m:0", "text": "Jean"}]);
        let np_mentions = json!([{"id": "q:0", "text": "Jean"}]);
        let response = json!({
            "phrase_verdicts": [{"id": "q:0", "is_entity_mention": true, "kind": "person"}],
            "verdicts": [{"id": "m:0", "is_person": true}],
        });
        let entities = map_bekind_entities(&response, &mentions, &np_mentions);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "Jean");
    }

    #[test]
    fn parse_scope_verdicts_reads_fields() {
        let response = json!({
            "scope_verdicts": [{
                "id": "s:0",
                "activity_phrase": "eat out",
                "temporal_words": ["weekend"],
                "habitual": true,
            }],
        });
        let verdicts = parse_scope_verdicts(&response);
        assert_eq!(verdicts.len(), 1);
        assert_eq!(verdicts[0].id, "s:0");
        assert_eq!(verdicts[0].activity_phrase, "eat out");
        assert_eq!(verdicts[0].temporal_words, vec!["weekend".to_string()]);
        assert!(verdicts[0].habitual);
    }

    #[test]
    fn parse_scope_verdicts_fail_open() {
        assert!(parse_scope_verdicts(&json!({})).is_empty());
        assert!(parse_scope_verdicts(&json!({"error": "bad"})).is_empty());
    }

    #[test]
    fn split_kind_hits_groups_per_text() {
        let mut id_to_text = HashMap::new();
        id_to_text.insert("k:0:q:0".to_string(), "cilantro".to_string());
        id_to_text.insert("k:1:q:0".to_string(), "basil".to_string());
        let response = json!({
            "phrase_verdicts": [
                {"id": "k:1:q:0", "kind": "herb"},
                {"id": "k:0:q:0", "kind": "herb"},
                {"id": "other", "kind": "thing"},
            ],
        });
        let per_text = split_kind_hits(&response, &id_to_text, 2);
        assert_eq!(per_text.len(), 2);
        assert_eq!(per_text[0].len(), 1);
        assert_eq!(per_text[0][0].text, "cilantro");
        assert_eq!(per_text[1][0].text, "basil");
    }

    #[test]
    fn clean_person_text_strips_both() {
        assert_eq!(clean_person_text("both Jean"), "Jean");
        assert_eq!(clean_person_text("John"), "John");
    }

    // ---- daemon integration tests (fake children) ----

    /// A fake parse daemon: one {"parse_texts": [...]} per stdin line, one
    /// {"parsed": [...]} per stdout line, with a canned descriptor. Counts
    /// request lines in a side file so tests can assert round-trip counts.
    fn write_fake_parse_script(dir: &std::path::Path) -> std::path::PathBuf {
        let path = dir.join("fake_behood_parse.py");
        let count_path = dir.join("parse_count.txt");
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
    assert "parse_texts" in payload, "expected the parse request"
    parsed = [
        {{"id": t["id"], "mentions": [],
          "np_mentions": [{{"id": "q:0", "text": "cilantro"}}]}}
        for t in payload["parse_texts"]
    ]
    sys.stdout.write(json.dumps({{"parsed": parsed}}) + "\n")
    sys.stdout.flush()
"#
        );
        std::fs::write(&path, script).expect("write fake parse script");
        path
    }

    /// A fake `bekind --serve`: one Request JSON per stdin line, one
    /// Response JSON per stdout line. Echoes scope verdicts for
    /// `scope_texts` and herb-kind phrase verdicts for `np_mentions`.
    /// Counts request lines in a side file.
    fn write_fake_bekind_script(dir: &std::path::Path) -> std::path::PathBuf {
        let path = dir.join("fake_bekind_serve.py");
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
    scope_verdicts = [
        {{"id": t["id"], "activity_phrase": "",
          "temporal_words": ["weekend"], "habitual": False}}
        for t in payload.get("scope_texts", [])
    ]
    phrase_verdicts = [
        {{"id": d["id"], "is_entity_mention": True, "kind": "herb"}}
        for d in payload.get("np_mentions", [])
    ]
    sys.stdout.write(json.dumps({{
        "verdicts": [], "entity_verdicts": [], "phrase_verdicts": phrase_verdicts,
        "activity_verdicts": [], "scope_verdicts": scope_verdicts,
    }}) + "\n")
    sys.stdout.flush()
"#
        );
        std::fs::write(&path, script).expect("write fake bekind script");
        path
    }

    fn test_parse_daemon(script: std::path::PathBuf) -> BehoodQueryDaemon {
        BehoodQueryDaemon::new(script, python_executable())
    }

    fn test_judge_daemon(script: std::path::PathBuf) -> BekindDaemon {
        BekindDaemon {
            daemon: JsonLinesDaemon::new_command(
                "bekind",
                vec![
                    python_executable(),
                    script.to_string_lossy().into_owned(),
                    "--serve".to_string(),
                ],
            ),
            cooldown: ServeCooldown::new(),
        }
    }

    fn unique_temp_dir(tag: &str) -> std::path::PathBuf {
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

    fn request_count(dir: &std::path::Path, name: &str) -> String {
        std::fs::read_to_string(dir.join(name))
            .expect("count file")
            .trim()
            .to_string()
    }

    #[test]
    fn parse_daemon_returns_descriptors() {
        let dir = unique_temp_dir("parse");
        let daemon = test_parse_daemon(write_fake_parse_script(&dir));
        let parsed = daemon
            .parse_texts(&["cilantro question"], Duration::from_secs(60))
            .expect("daemon should answer");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].id, "p:0");
        assert_eq!(parsed[0].np_mentions[0]["text"], "cilantro");
        assert_eq!(request_count(&dir, "parse_count.txt"), "1");
    }

    #[test]
    fn bekind_daemon_judges_one_line() {
        let dir = unique_temp_dir("judge");
        let daemon = test_judge_daemon(write_fake_bekind_script(&dir));
        let response = daemon
            .judge(
                &json!({"scope_texts": [{"id": "s:0", "text": "weekend run"}]}),
                Duration::from_secs(60),
            )
            .expect("daemon should answer");
        let verdicts = parse_scope_verdicts(&response);
        assert_eq!(verdicts.len(), 1);
        assert_eq!(verdicts[0].temporal_words, vec!["weekend".to_string()]);
        assert_eq!(request_count(&dir, "bekind_count.txt"), "1");
    }

    #[test]
    fn query_judge_combines_parse_and_judge() {
        let dir = unique_temp_dir("combined");
        let parse = test_parse_daemon(write_fake_parse_script(&dir));
        let judge = test_judge_daemon(write_fake_bekind_script(&dir));
        let (scope, entities) = query_judge(&parse, &judge, "weekend cilantro?", true)
            .expect("daemon pair should answer");
        assert_eq!(scope.len(), 1);
        assert_eq!(scope[0].temporal_words, vec!["weekend".to_string()]);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "cilantro");
        assert_eq!(entities[0].kind, "herb");
        assert_eq!(request_count(&dir, "parse_count.txt"), "1");
        assert_eq!(request_count(&dir, "bekind_count.txt"), "1");
    }

    #[test]
    fn query_judge_fails_open_when_parse_daemon_missing() {
        let parse = test_parse_daemon(std::path::PathBuf::from("/nonexistent/behood_query.py"));
        let judge = test_judge_daemon(write_fake_bekind_script(&unique_temp_dir("x")));
        assert!(
            query_judge(&parse, &judge, "anything", true).is_none(),
            "missing parse daemon must fail open"
        );
    }

    #[test]
    fn heuristic_provider_bypasses_parse_daemon() {
        // The heuristic backend must not touch the parse daemon at all:
        // point it at a nonexistent script and the descriptors still come
        // from the Rust builder.
        let parse = test_parse_daemon(std::path::PathBuf::from("/nonexistent/behood_query.py"));
        let parsed = parse_texts_via_provider(
            BehoodParseProvider::Heuristic,
            &parse,
            &["When did Jean visit Paris"],
            Duration::from_secs(30),
        )
        .expect("heuristic parse needs no daemon");
        assert_eq!(parsed.len(), 1);
        let heads: Vec<&str> = parsed[0]
            .np_mentions
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["head_lemma"].as_str().unwrap())
            .collect();
        assert!(heads.contains(&"jean"), "heads: {heads:?}");
        assert!(heads.contains(&"paris"), "heads: {heads:?}");
        assert!(
            !heads.contains(&"visit"),
            "verb must not be a chunk head: {heads:?}"
        );
    }

    #[test]
    fn spacy_provider_uses_parse_daemon() {
        let dir = unique_temp_dir("spacy_provider");
        let parse = test_parse_daemon(write_fake_parse_script(&dir));
        let parsed = parse_texts_via_provider(
            BehoodParseProvider::Spacy,
            &parse,
            &["hello"],
            Duration::from_secs(30),
        )
        .expect("spacy provider delegates to the daemon");
        assert_eq!(parsed.len(), 1);
        assert_eq!(request_count(&dir, "parse_count.txt"), "1");
    }

    #[test]
    fn daemon_respawns_dead_child() {
        let dir = unique_temp_dir("respawn");
        let daemon = test_parse_daemon(write_fake_parse_script(&dir));
        daemon
            .parse_texts(&["first"], Duration::from_secs(120))
            .expect("first call starts the child");
        daemon.kill_child_for_test();
        let parsed = daemon
            .parse_texts(&["after kill"], Duration::from_secs(120))
            .expect("daemon should respawn the child and succeed");
        assert_eq!(parsed.len(), 1);
    }
}
