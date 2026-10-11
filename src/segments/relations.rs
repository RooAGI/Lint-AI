//! Entity-relation (SPO) fact store and structured query.
//!
//! Extraction is dependency-parse based (`scripts/spacy_relations.py`):
//! general grammatical rules over dependency labels produce
//! `(subject, predicate, object)` triples, where the predicate is
//! `verb_lemma[_prt][_prep]` (e.g. `go_to`, `be_in`, `volunteer_at`).
//! No verb phrase is ever enumerated in code. The Rust side owns:
//!
//! - the predicate -> family lexicon ([`predicate_family`], declarative data);
//! - the per-conversation fact index ([`RelationIndex`]);
//! - subject resolution (exact, fuzzy);
//! - structured queries: [`RelationIndex::query_shared`] (multi-hop
//!   "both X and Y" intersection) and [`RelationIndex::query_temporal_span`]
//!   ("where was X between <dates>").
//!
//! Every relation carries the session date as a temporal anchor, a
//! spaCy-NER place flag (`is_place`), and behood's ontological kind for the
//! object noun phrase (`object_kind`: "event", "place", "org", "person",
//! "work", "food", "thing"). The place-expectation filter accepts NER
//! places and behood-kind place/event objects alike.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// One dialogue turn to extract relations from.
#[derive(Debug, Clone, Serialize)]
pub struct RelationTurn {
    /// Display name of the speaker, e.g. "Gina".
    pub speaker: String,
    /// Raw turn text (without the "Speaker: " prefix).
    pub text: String,
    /// Session id, e.g. "conv-30::session_2".
    pub session_id: String,
    /// Turn index within the session.
    pub turn_idx: usize,
    /// Document id this turn was indexed under.
    pub doc_id: String,
    /// Raw session date string, e.g. "1:08 pm on 11 August, 2023".
    pub session_date: Option<String>,
}

/// Calendar date used as a temporal anchor. Field order gives chronological
/// ordering via the derived `Ord`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Ymd {
    pub year: i32,
    pub month: u32,
    pub day: u32,
}

impl Ymd {
    /// "July 2022" — for date keywords matching the "[session date: ...]"
    /// prefix lexically.
    pub fn month_year_string(&self) -> String {
        let name = match self.month {
            1 => "january",
            2 => "february",
            3 => "march",
            4 => "april",
            5 => "may",
            6 => "june",
            7 => "july",
            8 => "august",
            9 => "september",
            10 => "october",
            11 => "november",
            12 => "december",
            _ => "unknown",
        };
        format!("{} {}", name, self.year)
    }
}

/// Month name (full or 3-letter) to month number.
pub fn month_number(lower: &str) -> Option<u32> {
    Some(match lower {
        "january" | "jan" => 1,
        "february" | "feb" => 2,
        "march" | "mar" => 3,
        "april" | "apr" => 4,
        "may" => 5,
        "june" | "jun" => 6,
        "july" | "jul" => 7,
        "august" | "aug" => 8,
        "september" | "sep" | "sept" => 9,
        "october" | "oct" => 10,
        "november" | "nov" => 11,
        "december" | "dec" => 12,
        _ => return None,
    })
}

/// Strip an English ordinal suffix ("11th" -> "11").
fn strip_ordinal(w: &str) -> &str {
    w.strip_suffix("st")
        .or_else(|| w.strip_suffix("nd"))
        .or_else(|| w.strip_suffix("rd"))
        .or_else(|| w.strip_suffix("th"))
        .unwrap_or(w)
}

/// Parse a LoCoMo session date like "1:08 pm on 11 August, 2023".
/// Needs a month, an adjacent 1-2 digit day, and a 4-digit year.
pub fn parse_session_date(s: &str) -> Option<Ymd> {
    let words: Vec<&str> = s
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let mi = words
        .iter()
        .position(|w| month_number(&w.to_lowercase()).is_some())?;
    let month = month_number(&words[mi].to_lowercase())?;
    // Day: closest 1-2 digit number next to the month ("11 August",
    // "August 11", "August 11th").
    let mut day: Option<u32> = None;
    for off in [1, 2] {
        for j in [mi.checked_sub(off), mi.checked_add(off)]
            .into_iter()
            .flatten()
        {
            if let Some(w) = words.get(j) {
                let lowered = w.to_lowercase();
                let digits = strip_ordinal(&lowered);
                if digits.len() <= 2 && digits.chars().all(|c| c.is_ascii_digit()) {
                    if let Ok(d) = digits.parse::<u32>() {
                        if (1..=31).contains(&d) {
                            day = Some(d);
                            break;
                        }
                    }
                }
            }
        }
        if day.is_some() {
            break;
        }
    }
    let year = words.iter().find_map(|w| {
        if w.len() == 4 && w.chars().all(|c| c.is_ascii_digit()) {
            w.parse::<i32>().ok()
        } else {
            None
        }
    })?;
    Some(Ymd {
        year,
        month,
        day: day?,
    })
}

/// One raw key phrase from `scripts/spacy_relations.py` (JSON field-for-field):
/// a grammar-accepted entity mention with behood's ontological kind.
/// `doc_id`/`turn_idx` are echoed from the input turn so phrases join back
/// to their exact source document; older script output omits them (serde
/// defaults) and falls back to per-session matching.
#[derive(Debug, Clone, Deserialize)]
pub struct RawKeyPhrase {
    pub text: String,
    pub kind: String,
    pub session_id: String,
    #[serde(default)]
    pub doc_id: String,
    #[serde(default)]
    pub turn_idx: usize,
}

/// Full output of the dependency-parse extractor: triples plus key phrases.
#[derive(Debug, Clone, Default)]
pub struct ExtractorOutput {
    pub relations: Vec<RawRelation>,
    pub key_phrases: Vec<RawKeyPhrase>,
}

/// One raw triple from `scripts/spacy_relations.py` (JSON field-for-field).
#[derive(Debug, Clone, Deserialize)]
pub struct RawRelation {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub is_place: bool,
    /// Behood's ontological kind for the object noun phrase ("event",
    /// "place", "org", "person", "work", "food", "thing"); "thing" when
    /// the object is not a judged mention. Older script output omits it.
    #[serde(default = "default_object_kind")]
    pub object_kind: String,
    /// Behood's activity verdict: is the verb an activity (vs copula,
    /// auxiliary, light verb, stative)? Older script output omits it;
    /// defaults to true (fail-open).
    #[serde(default = "default_is_activity")]
    pub is_activity: bool,
    pub session_id: String,
    pub turn_idx: usize,
    pub doc_id: String,
    pub session_date: Option<String>,
    pub evidence: String,
    pub confidence: f32,
    /// Pronoun-resolution provenance, e.g. "she->Maria"; None when the
    /// subject and object were explicit mentions.
    #[serde(default)]
    pub coref: Option<String>,
}

/// Fail-open default for [`RawRelation::object_kind`].
fn default_object_kind() -> String {
    "thing".to_string()
}

/// Fail-open default for [`RawRelation::is_activity`].
fn default_is_activity() -> bool {
    true
}

/// One extracted (subject, predicate, object) triple with evidence.
#[derive(Debug, Clone, Serialize)]
pub struct EntityRelation {
    /// Display subject, e.g. "Gina".
    pub subject: String,
    /// Normalized subject for matching.
    pub subject_norm: String,
    /// Predicate string, e.g. "go_to" (`verb_lemma[_prt][_prep]`).
    pub predicate: String,
    /// Display object, e.g. "Rome".
    pub object: String,
    /// Normalized object for intersection.
    pub object_norm: String,
    /// spaCy-NER place flag (GPE/LOC/FAC overlap).
    pub is_place: bool,
    pub session_id: String,
    pub turn_idx: usize,
    pub doc_id: String,
    /// "Speaker: text" evidence line.
    pub evidence: String,
    pub confidence: f32,
    /// Temporal anchor from the session date; None when unparseable.
    pub date: Option<Ymd>,
}

/// A family of predicates that count as "being somewhere" for intersection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PredicateFamily {
    /// go_to / be_in / volunteer_at / ...: physical presence at a place.
    PlacePresence,
    /// develop / create / make / build / work_on / design: making things.
    Creation,
    /// Family-agnostic: any predicate family. Used for 1-person + time-window
    /// questions whose answer type is not a known family (e.g. "What setback
    /// did Melanie face in October 2023?"), served by
    /// [`RelationIndex::query_temporal_span_general`].
    Any,
}

impl PredicateFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            PredicateFamily::PlacePresence => "place-presence",
            PredicateFamily::Creation => "creation",
            PredicateFamily::Any => "any",
        }
    }
}

/// Declarative predicate -> family lexicon.
///
/// Predicates are `verb_lemma[_prt][_prep]` strings from the dependency
/// extractor. This table is *data*: adding a new verb never changes the
/// extraction or query mechanism.
pub fn predicate_family(predicate: &str) -> Option<PredicateFamily> {
    Some(match predicate {
        "be_to" | "go_to" | "come_to" | "travel_to" | "move_to" | "fly_to" | "drive_to"
        | "walk_to" | "take_to" | "get_to" | "head_to" | "return_to" | "visit" | "be_in"
        | "stay_in" | "live_in" | "arrive_in" | "stay_at" | "arrive_at" | "volunteer_at"
        | "work_at" => PredicateFamily::PlacePresence,
        "develop" | "create" | "make" | "build" | "work_on" | "design" => PredicateFamily::Creation,
        _ => return None,
    })
}

/// Normalize a name or object for matching: lowercase, trim, collapse space.
pub fn normalize_relation_token(s: &str) -> String {
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Small Levenshtein distance over chars (names are short).
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j] + cost).min((curr[j] + 1).min(prev[j + 1] + 1));
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

// ---------------------------------------------------------------------------
// spaCy subprocess extraction.
// ---------------------------------------------------------------------------

/// Python executable for the extractor scripts. Mirrors the project's
/// `detect_python_executable` convention (`PYTHON_EXECUTABLE` / `PYTHON` /
/// `VIRTUAL_ENV`, else `python3`).
pub(crate) fn python_executable() -> String {
    if let Ok(value) = std::env::var("PYTHON_EXECUTABLE") {
        let value = value.trim();
        if !value.is_empty() {
            return value.to_string();
        }
    }
    if let Ok(value) = std::env::var("PYTHON") {
        let value = value.trim();
        if !value.is_empty() {
            return value.to_string();
        }
    }
    "python3".to_string()
}

/// Path to the Behood relation extractor script.
///
/// Behood owns the relation-extraction judgment; lint-ai consumes it.
/// Resolution order:
/// 1. `BEHOOD_RELATIONS` env var (explicit override)
/// 2. Bundled `scripts/spacy_relations.py` (backward compatibility)
pub(crate) fn extractor_script_path() -> std::path::PathBuf {
    if let Ok(value) = std::env::var("BEHOOD_RELATIONS") {
        let value = value.trim();
        if !value.is_empty() {
            return std::path::PathBuf::from(value);
        }
    }
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/spacy_relations.py")
}

/// Run the dependency-parse extractor (`scripts/spacy_relations.py`) over the
/// turns and return raw triples plus grammar-accepted key phrases.
///
/// Never panics: any failure (missing Python/spaCy, bad output) yields an
/// empty output and the caller declines to the adaptive retrieval path.
///
/// The production path (no script override) goes through the long-lived
/// extractor daemon so the spaCy model load is paid once per process; every
/// daemon failure falls back to a one-shot subprocess, which is also what
/// script overrides always use.
pub fn extract_relations_via_spacy(
    turns: &[RelationTurn],
    timeout: Duration,
    model: &str,
) -> ExtractorOutput {
    run_extractor(turns, false, None, timeout, model).unwrap_or_default()
}

/// Key-phrase half of the extractor: grammar-accepted entity mentions for
/// the segment entity channel. Runs the script in `key_phrases_only` mode
/// (skips triple/frame extraction) and returns phrases keyed by `doc_id`.
/// Fail-open like the relations path: any failure yields no phrases.
pub fn extract_key_phrases_via_spacy(
    turns: &[RelationTurn],
    script_override: Option<&std::path::Path>,
    timeout: Duration,
    model: &str,
) -> Vec<RawKeyPhrase> {
    try_extract_key_phrases_via_spacy(turns, script_override, timeout, model).unwrap_or_default()
}

/// Fallible variant: `None` when the extractor subprocess failed to run or
/// produced unusable output; `Some` (possibly empty) when it ran to
/// completion. Lets callers distinguish a failed extraction — which must
/// stay retryable — from a genuinely empty one.
pub fn try_extract_key_phrases_via_spacy(
    turns: &[RelationTurn],
    script_override: Option<&std::path::Path>,
    timeout: Duration,
    model: &str,
) -> Option<Vec<RawKeyPhrase>> {
    run_extractor(turns, true, script_override, timeout, model).map(|output| output.key_phrases)
}

/// Parse one extractor response object (one-shot stdout or one daemon
/// protocol line). Returns `None` when the response carries an `error` —
/// the extractor declined the payload — or is not valid extractor JSON;
/// `Some` (possibly empty) on a completed run. An error response must never
/// become a successful-empty result: callers stamp completed-empty runs as
/// done, and stamping a failed run would retire it permanently.
pub(crate) fn parse_extractor_output(response: &str) -> Option<ExtractorOutput> {
    let value: serde_json::Value = serde_json::from_str(response.trim()).ok()?;
    if value.get("error").is_some() {
        return None;
    }
    #[derive(Deserialize)]
    struct Output {
        #[serde(default)]
        relations: Vec<RawRelation>,
        #[serde(default)]
        key_phrases: Vec<RawKeyPhrase>,
    }
    let parsed: Output = serde_json::from_value(value).ok()?;
    Some(ExtractorOutput {
        relations: parsed.relations,
        key_phrases: parsed.key_phrases,
    })
}

/// Pick the spaCy model for a batch of extractor turns from the turns'
/// detected script: Korean turns get `ko_core_news_sm`, Chinese turns get
/// `zh_core_web_sm`, everything else the English default.
pub(crate) fn extractor_model_for_turns(turns: &[RelationTurn]) -> &'static str {
    let mut text = String::new();
    for t in turns {
        text.push_str(&t.text);
        text.push('\n');
    }
    crate::lang::default_spacy_model_for_lang(crate::lang::detect_lang(&text))
}

/// Runs the spaCy extractor script. Returns `None` when the subprocess
/// could not run or its output was unusable; `Some` on a completed run even
/// when it produced no relations or phrases.
fn run_extractor(
    turns: &[RelationTurn],
    key_phrases_only: bool,
    script_override: Option<&std::path::Path>,
    timeout: Duration,
    model: &str,
) -> Option<ExtractorOutput> {
    if script_override.is_none() {
        if let Some(output) = super::extractor_daemon::ExtractorDaemon::global().extract(
            turns,
            key_phrases_only,
            timeout,
            model,
        ) {
            return Some(output);
        }
        // Daemon unavailable (first-start failure, dead child, timeout,
        // contention): fall through to a one-shot subprocess.
    }
    run_extractor_oneshot(turns, key_phrases_only, script_override, model)
}

/// One-shot extractor subprocess: spawn Python, feed the payload on stdin,
/// parse stdout. Used for script overrides and as the daemon fallback.
fn run_extractor_oneshot(
    turns: &[RelationTurn],
    key_phrases_only: bool,
    script_override: Option<&std::path::Path>,
    model: &str,
) -> Option<ExtractorOutput> {
    let script = match script_override {
        Some(path) => path.to_path_buf(),
        None => extractor_script_path(),
    };
    if !script.exists() {
        eprintln!("relations: extractor script missing: {}", script.display());
        return None;
    }
    let payload = serde_json::json!({
        "model": model,
        "turns": turns,
        "key_phrases_only": key_phrases_only,
    });
    let mut child = match Command::new(python_executable())
        .arg(&script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            eprintln!("relations: failed to spawn extractor: {e}");
            return None;
        }
    };
    let write_result = child
        .stdin
        .as_mut()
        .map(|stdin| {
            let bytes = serde_json::to_vec(&payload).unwrap_or_default();
            stdin.write_all(&bytes).and_then(|_| stdin.flush())
        })
        .unwrap_or(Ok(()));
    if let Err(e) = write_result {
        eprintln!("relations: failed to write extractor input: {e}");
        return None;
    }
    // Take stdout/stderr handles before the wait loop; we'll read them
    // after the child exits.
    let mut stdout_handle = child.stdout.take();
    let mut stderr_handle = child.stderr.take();
    // Close stdin so the child sees EOF and won't block on input.
    drop(child.stdin.take());
    // wait_with_output() has no timeout and would block forever on a hung
    // extractor. Poll try_wait() with a bound instead; on timeout, kill
    // the child so repeated timed-out queries can't leak Python processes.
    const ONESHOT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
    const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);
    let start = std::time::Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if start.elapsed() >= ONESHOT_TIMEOUT {
                    eprintln!("relations: one-shot extractor timed out after 120s; killing child");
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(POLL_INTERVAL);
            }
            Err(e) => {
                eprintln!("relations: extractor wait failed: {e}");
                return None;
            }
        }
    };
    // Child exited; collect its output.
    let mut stdout_buf = Vec::new();
    let mut stderr_buf = Vec::new();
    if let Some(mut out) = stdout_handle.take() {
        use std::io::Read;
        let _ = out.read_to_end(&mut stdout_buf);
    }
    if let Some(mut err) = stderr_handle.take() {
        use std::io::Read;
        let _ = err.read_to_end(&mut stderr_buf);
    }
    let output = std::process::Output {
        status,
        stdout: stdout_buf,
        stderr: stderr_buf,
    };
    if !output.status.success() {
        eprintln!(
            "relations: extractor failed: {}",
            String::from_utf8_lossy(&output.stderr)
                .chars()
                .take(300)
                .collect::<String>()
        );
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    match parse_extractor_output(&stdout) {
        Some(parsed) => Some(parsed),
        None => {
            eprintln!(
                "relations: bad extractor output: {}",
                stdout.chars().take(300).collect::<String>()
            );
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Relation index + structured query.
// ---------------------------------------------------------------------------

/// One indexed hit for (subject, predicate).
#[derive(Debug, Clone)]
pub struct RelationHit {
    pub object: String,
    pub object_norm: String,
    pub is_place: bool,
    /// Behood's ontological kind for the object noun phrase, carried from
    /// [`RawRelation::object_kind`].
    pub object_kind: String,
    /// Behood's activity verdict for the verb, carried from
    /// [`RawRelation::is_activity`].
    pub is_activity: bool,
    pub session_id: String,
    pub turn_idx: usize,
    pub doc_id: String,
    pub evidence: String,
    pub confidence: f32,
    /// Temporal anchor from the session date; None when unparseable.
    pub date: Option<Ymd>,
    /// Display name of the turn's speaker, e.g. "Dave". Enables directed
    /// communication queries ("what did X tell Y") to filter relations by
    /// who said them, not just who the subject is.
    pub speaker: String,
}

impl RelationHit {
    /// Whether this object answers a place-seeking question ("what places",
    /// "where"). Physical places come from spaCy NER (`is_place`); event
    /// venues come from behood's kind judgment: a tournament or a game
    /// convention is a valid "place" answer even though no NER label fires.
    pub fn place_like(&self) -> bool {
        self.is_place || matches!(self.object_kind.as_str(), "place" | "event")
    }
}

/// One object shared by all queried subjects, with per-subject evidence.
#[derive(Debug, Clone, Serialize)]
pub struct SharedObject {
    pub object: String,
    pub object_norm: String,
    pub evidence: Vec<SharedEvidence>,
    pub score: f32,
}

/// Evidence for one subject's relation to the shared object.
#[derive(Debug, Clone, Serialize)]
pub struct SharedEvidence {
    pub subject: String,
    pub session_id: String,
    pub turn_idx: usize,
    pub doc_id: String,
    pub text: String,
}

/// Per-conversation relation index: (subject_norm, predicate) -> hits.
#[derive(Debug, Default)]
pub struct RelationIndex {
    by_sp: HashMap<(String, String), Vec<RelationHit>>,
    /// Display names of known persons (speakers first).
    pub persons: Vec<String>,
    person_norms: Vec<String>,
}

/// What kind of answer a question wants, for keyword extraction.
/// Places were the first case; the same (person, activity) ->
/// typed-objects machinery generalizes to objects and dates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnswerKind {
    /// "what places", "where", "which city" -> place/event objects.
    Place,
    /// "what kind of X", "what does P do" -> all objects of the activity.
    Object,
    /// "when ..." -> session dates of the activity relations.
    Date,
}

impl RelationIndex {
    /// Build from dialogue turns (persons) and raw extractor triples.
    pub fn build(turns: &[RelationTurn], raw: &[RawRelation]) -> Self {
        let mut idx = RelationIndex::default();
        let mut seen_person: HashSet<String> = HashSet::new();
        // Map (session_id, turn_idx) -> speaker for RelationHit.speaker.
        let mut turn_speaker: HashMap<(String, usize), String> = HashMap::new();
        for turn in turns {
            let norm = normalize_relation_token(&turn.speaker);
            if seen_person.insert(norm.clone()) {
                idx.persons.push(turn.speaker.clone());
                idx.person_norms.push(norm);
            }
            turn_speaker.insert(
                (turn.session_id.clone(), turn.turn_idx),
                turn.speaker.clone(),
            );
        }
        for rel in raw {
            let subject_norm = normalize_relation_token(&rel.subject);
            let object_norm = normalize_relation_token(&rel.object);
            if subject_norm.is_empty() || object_norm.is_empty() {
                continue;
            }
            // Fold named subjects into the person list.
            if !idx.person_norms.contains(&subject_norm) {
                idx.person_norms.push(subject_norm.clone());
                idx.persons.push(rel.subject.clone());
            }
            let date = rel.session_date.as_deref().and_then(parse_session_date);
            let speaker = turn_speaker
                .get(&(rel.session_id.clone(), rel.turn_idx))
                .cloned()
                .unwrap_or_default();
            idx.by_sp
                .entry((subject_norm, rel.predicate.clone()))
                .or_default()
                .push(RelationHit {
                    object: rel.object.clone(),
                    object_norm,
                    is_place: rel.is_place,
                    object_kind: rel.object_kind.clone(),
                    is_activity: rel.is_activity,
                    session_id: rel.session_id.clone(),
                    turn_idx: rel.turn_idx,
                    doc_id: rel.doc_id.clone(),
                    evidence: rel.evidence.clone(),
                    confidence: rel.confidence,
                    date,
                    speaker,
                });
        }
        idx
    }

    /// Number of indexed (subject, predicate) keys.
    pub fn len(&self) -> usize {
        self.by_sp.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_sp.is_empty()
    }

    /// Total indexed triples.
    pub fn triple_count(&self) -> usize {
        self.by_sp.values().map(Vec::len).sum()
    }

    /// Hits for one subject under any predicate in the family.
    fn hits_for<'a>(
        &'a self,
        snorm: &'a str,
        family: PredicateFamily,
    ) -> impl Iterator<Item = &'a RelationHit> {
        self.by_sp
            .iter()
            .filter(move |((s, p), _)| s == snorm && predicate_family(p) == Some(family))
            .flat_map(|(_, hits)| hits.iter())
    }

    /// All hits for one subject across every predicate family.
    fn hits_for_any<'a>(&'a self, snorm: &'a str) -> impl Iterator<Item = &'a RelationHit> {
        self.by_sp
            .iter()
            .filter(move |((s, _), _)| s == snorm)
            .flat_map(|(_, hits)| hits.iter())
    }

    /// Relationship-derived keywords: distinct place-like object texts for
    /// `person`, restricted to documents where the person also has a
    /// relation whose predicate contains `activity_verb` (lemmatized, e.g.
    /// "meet"). For "what places has Nate met new people": the places in
    /// Nate's "meet" documents (the tournament, the game convention) become
    /// routing keywords. The index already carries behood's `object_kind`,
    /// so event venues count via `place_like()`.

    pub fn place_keywords_for_activity(&self, person: &str, activity_verb: &str) -> Vec<String> {
        self.keywords_for_question(person, activity_verb, AnswerKind::Place)
    }

    /// Doc IDs where `person` performs `activity_verb` (lemmatized predicate
    /// match, or same predicate family). Used by the structured place-seeking
    /// path: the caller expands these docs' sessions for the reader.
    pub fn docs_for_activity(&self, person: &str, activity_verb: &str) -> Vec<String> {
        let snorm = normalize_relation_token(person);
        let verb_family = predicate_family(activity_verb);
        let mut seen = std::collections::HashSet::new();
        let mut docs = Vec::new();
        for ((s, p), hits) in self.by_sp.iter() {
            if s != &snorm {
                continue;
            }
            if !(p.contains(activity_verb)
                || verb_family.is_some() && predicate_family(p) == verb_family)
            {
                continue;
            }
            for hit in hits {
                if seen.insert(hit.doc_id.clone()) {
                    docs.push(hit.doc_id.clone());
                }
            }
        }
        docs
    }

    /// Doc IDs where `person` is the subject of ANY activity (any verb).
    /// Used by the activity-seeking path ("what does X do"): the question
    /// asks FOR the activity, so we return all of the person's activity
    /// documents and let the lexical score rank by context terms.
    pub fn docs_for_person_activities(&self, person: &str) -> Vec<String> {
        let snorm = normalize_relation_token(person);
        let mut seen = std::collections::HashSet::new();
        let mut docs = Vec::new();
        for ((s, _), hits) in self.by_sp.iter() {
            if s != &snorm {
                continue;
            }
            for hit in hits {
                // Behood's activity verdict: only real activities, not
                // copulas, auxiliaries, light verbs, or statives.
                if !hit.is_activity {
                    continue;
                }
                if seen.insert(hit.doc_id.clone()) {
                    docs.push(hit.doc_id.clone());
                }
            }
        }
        docs
    }

    /// Generalized relationship-derived keywords: distinct object texts
    /// (or session dates) for `person`, restricted to documents where the
    /// person also has a relation whose predicate contains `activity_verb`
    /// (lemmatized). The `AnswerKind` selects which facet becomes keywords:
    /// - `Place`: place-like objects (existing behavior for "what places").
    /// - `Object`: all objects (for "what kind of games has James developed",
    ///   "what does Melanie do on hikes").
    /// - `Date`: session dates as "month year" strings (for "when will John
    ///   start his job": the date prefix "[session date: ... July, 2022]"
    ///   matches lexically).
    pub fn keywords_for_question(
        &self,
        person: &str,
        activity_verb: &str,
        kind: AnswerKind,
    ) -> Vec<String> {
        let snorm = normalize_relation_token(person);
        // Documents where the person performs the activity: predicate contains
        // the verb, or shares its predicate family (e.g. "develop" matches
        // "work_on", "create" via Creation).
        let verb_family = predicate_family(activity_verb);
        let activity_docs: std::collections::HashSet<&str> = self
            .by_sp
            .iter()
            .filter(|((s, p), _)| {
                s == &snorm
                    && (p.contains(activity_verb)
                        || verb_family.is_some() && predicate_family(p) == verb_family)
            })
            .flat_map(|(_, hits)| hits.iter())
            .map(|h| h.doc_id.as_str())
            .collect();
        if activity_docs.is_empty() {
            return Vec::new();
        }
        let mut seen = std::collections::HashSet::new();
        let mut keywords = Vec::new();
        for hit in self.hits_for_any(&snorm) {
            if !activity_docs.contains(hit.doc_id.as_str()) {
                continue;
            }
            match kind {
                AnswerKind::Place => {
                    if !hit.place_like() {
                        continue;
                    }
                    if seen.insert(hit.object_norm.clone()) {
                        keywords.push(hit.object.clone());
                    }
                }
                AnswerKind::Object => {
                    if seen.insert(hit.object_norm.clone()) {
                        keywords.push(hit.object.clone());
                    }
                }
                AnswerKind::Date => {
                    if let Some(date) = hit.date {
                        let key = format!("{:04}-{:02}", date.year, date.month);
                        if seen.insert(key) {
                            keywords.push(date.month_year_string());
                        }
                    }
                }
            }
        }
        keywords
    }

    /// Resolve display names to normalized person ids.
    ///
    /// Exact match, then fuzzy (edit distance <= 1 for short names, <= 2
    /// otherwise). Returns None if any name cannot be resolved: a name with
    /// no evidence in the data is never mapped by process of elimination
    /// (e.g. "Jean" is a real, distinct person elsewhere, not a typo for
    /// whoever is left over).
    pub fn resolve_subjects(&self, names: &[String]) -> Option<Vec<String>> {
        let mut resolved: Vec<Option<String>> = Vec::with_capacity(names.len());
        for name in names {
            let norm = normalize_relation_token(name);
            if let Some(i) = self.person_norms.iter().position(|p| p == &norm) {
                resolved.push(Some(self.person_norms[i].clone()));
                continue;
            }
            let mut best: Option<(usize, usize)> = None;
            for (i, p) in self.person_norms.iter().enumerate() {
                // Short names get a strict threshold: "jean" must not match
                // "jon" (distance 2); "john" -> "jon" (distance 1) is fine.
                let max_dist = if norm.len() <= 4 || p.len() <= 4 {
                    1
                } else {
                    2
                };
                let d = levenshtein(&norm, p);
                if d <= max_dist && best.is_none_or(|(_, bd)| d < bd) {
                    best = Some((i, d));
                }
            }
            resolved.push(best.map(|(i, _)| self.person_norms[i].clone()));
        }
        resolved.into_iter().collect()
    }

    /// Objects related to every subject under any predicate in the family.
    ///
    /// Returns None when a subject cannot be resolved (caller declines to the
    /// adaptive path); Some(vec![]) when resolution worked but nothing is
    /// shared.
    pub fn query_shared(
        &self,
        names: &[String],
        family: PredicateFamily,
        expect_place: bool,
    ) -> Option<Vec<SharedObject>> {
        let subjects = self.resolve_subjects(names)?;
        // Per subject: object_norm -> (display, hits).
        let mut per_subject: Vec<HashMap<String, (String, Vec<RelationHit>)>> = Vec::new();
        for snorm in &subjects {
            let mut objects: HashMap<String, (String, Vec<RelationHit>)> = HashMap::new();
            for hit in self.hits_for(snorm, family) {
                if expect_place && !hit.place_like() {
                    continue;
                }
                objects
                    .entry(hit.object_norm.clone())
                    .or_insert_with(|| (hit.object.clone(), Vec::new()))
                    .1
                    .push(hit.clone());
            }
            per_subject.push(objects);
        }
        let mut shared: Vec<SharedObject> = Vec::new();
        if let Some((first, rest)) = per_subject.split_first() {
            for (onorm, (display, _)) in first {
                if rest.iter().all(|m| m.contains_key(onorm)) {
                    let mut evidence = Vec::new();
                    let mut score = 0.0f32;
                    for (si, m) in per_subject.iter().enumerate() {
                        if let Some((_, hits)) = m.get(onorm) {
                            for hit in hits {
                                score += hit.confidence;
                                evidence.push(SharedEvidence {
                                    subject: subjects[si].clone(),
                                    session_id: hit.session_id.clone(),
                                    turn_idx: hit.turn_idx,
                                    doc_id: hit.doc_id.clone(),
                                    text: hit.evidence.clone(),
                                });
                            }
                        }
                    }
                    shared.push(SharedObject {
                        object: display.clone(),
                        object_norm: onorm.clone(),
                        evidence,
                        score,
                    });
                }
            }
        }
        shared.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.object.cmp(&b.object))
        });
        Some(shared)
    }

    /// Directed communication: relations where `recipient` is the subject
    /// and `speaker` said the turn, filtered by topic keywords matching the
    /// object. E.g. "What is Dave's advice to Calvin regarding his dreams?"
    /// -> recipient=Calvin, speaker=Dave, topic=["dreams"] finds
    /// `Calvin -> forget -> dreams` from Dave's turn.
    ///
    /// Returns None when either name cannot be resolved or no relations match.
    pub fn query_directed(
        &self,
        speaker: &str,
        recipient: &str,
        topic_keywords: &[String],
    ) -> Option<Vec<StructuredHit>> {
        let recipient_norm = normalize_relation_token(recipient);
        let speaker_norm = normalize_relation_token(speaker);
        if recipient_norm.is_empty() || speaker_norm.is_empty() {
            return None;
        }
        // Resolve recipient against known persons (fuzzy).
        let resolved_recipient = self.resolve_subjects(&[recipient.to_string()])?;
        let rnorm = normalize_relation_token(&resolved_recipient[0]);

        let mut hits = Vec::new();
        for ((snorm, _pred), rel_hits) in &self.by_sp {
            if *snorm != rnorm {
                continue;
            }
            for hit in rel_hits {
                // Speaker must match (the one who communicated).
                if normalize_relation_token(&hit.speaker) != speaker_norm {
                    continue;
                }
                // Topic keywords must appear in the object (if any given).
                if !topic_keywords.is_empty() {
                    let obj_lower = hit.object.to_lowercase();
                    let matches_topic = topic_keywords
                        .iter()
                        .any(|kw| obj_lower.contains(&kw.to_lowercase()));
                    if !matches_topic {
                        continue;
                    }
                }
                hits.push(StructuredHit {
                    doc_id: hit.doc_id.clone(),
                    score: 1000.0 + hit.confidence,
                    confidence: hit.confidence,
                    evidence_label: format!(
                        "directed: {} -> {} -> {}",
                        hit.speaker, recipient, hit.object
                    ),
                });
            }
        }
        if hits.is_empty() {
            return None;
        }
        // Deduplicate by doc_id, keep highest score.
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut seen = std::collections::HashSet::new();
        hits.retain(|h| seen.insert(h.doc_id.clone()));
        Some(hits)
    }

    /// Objects related to one subject under the family whose session date
    /// falls inside `[start, end]` (inclusive), e.g. "Where was John between
    /// August 11 and August 15 2023?".
    ///
    /// Returns None when the subject cannot be resolved (caller declines to
    /// the adaptive path); relations without a parseable date are excluded.
    /// Reuses [`SharedObject`] with single-subject evidence.
    pub fn query_temporal_span(
        &self,
        name: &str,
        family: PredicateFamily,
        start: Ymd,
        end: Ymd,
        expect_place: bool,
    ) -> Option<Vec<SharedObject>> {
        let subjects = self.resolve_subjects(std::slice::from_ref(&name.to_string()))?;
        let snorm = &subjects[0];
        let mut objects: HashMap<String, (String, Vec<SharedEvidence>, f32)> = HashMap::new();
        for hit in self.hits_for(snorm, family) {
            let date = match hit.date {
                Some(d) => d,
                None => continue,
            };
            if date < start || date > end {
                continue;
            }
            if expect_place && !hit.place_like() {
                continue;
            }
            let entry = objects
                .entry(hit.object_norm.clone())
                .or_insert_with(|| (hit.object.clone(), Vec::new(), 0.0));
            entry.2 += hit.confidence;
            entry.1.push(SharedEvidence {
                subject: snorm.clone(),
                session_id: hit.session_id.clone(),
                turn_idx: hit.turn_idx,
                doc_id: hit.doc_id.clone(),
                text: hit.evidence.clone(),
            });
        }
        let mut out: Vec<SharedObject> = objects
            .into_iter()
            .map(|(object_norm, (object, evidence, score))| SharedObject {
                object,
                object_norm,
                evidence,
                score,
            })
            .collect();
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.object.cmp(&b.object))
        });
        Some(out)
    }

    /// Family-agnostic temporal span: objects related to one subject under ANY
    /// predicate family whose session date falls inside `[start, end]`
    /// (inclusive), ranked by content overlap with the question.
    ///
    /// Serves 1-person + time-window questions whose answer type is not a
    /// known predicate family (e.g. "What setback did Melanie face in October
    /// 2023?"). Content terms are the stopword-filtered stemmed question
    /// tokens ([`super::catalog::query_tokens`]); each hit scores its overlap
    /// against the stemmed tokens of its object + evidence text, so hits
    /// naming the question's distinctive terms ("hurt", "pottery") outrank
    /// unrelated in-window hits. Returns None when the subject cannot be
    /// resolved (caller declines to the adaptive path); relations without a
    /// parseable date are excluded.
    pub fn query_temporal_span_general(
        &self,
        name: &str,
        start: Ymd,
        end: Ymd,
        question: &str,
    ) -> Option<Vec<SharedObject>> {
        let subjects = self.resolve_subjects(std::slice::from_ref(&name.to_string()))?;
        let snorm = &subjects[0];
        // Luyi: use the focus word to filter on the activity. The focus is
        // what the question asks about ("health incidents"), not the person
        // ("evan") or the verb ("face"). This avoids the "evan matches every
        // hit" poisoning.
        let focus = crate::question_focus::identify_focus(question);
        let content: HashSet<String> = focus
            .focus_terms
            .iter()
            .flat_map(|t| super::catalog::query_tokens(t))
            .collect();
        if content.is_empty() {
            return None;
        }
        let mut objects: HashMap<String, (String, Vec<SharedEvidence>, f32)> = HashMap::new();
        for hit in self.hits_for_any(snorm) {
            let date = match hit.date {
                Some(d) => d,
                None => continue,
            };
            if date < start || date > end {
                continue;
            }
            let hit_terms: HashSet<String> =
                super::catalog::query_tokens(&format!("{} {}", hit.object, hit.evidence));
            let overlap = content.intersection(&hit_terms).count() as f32;
            if overlap == 0.0 {
                continue;
            }
            let entry = objects
                .entry(hit.object_norm.clone())
                .or_insert_with(|| (hit.object.clone(), Vec::new(), 0.0));
            entry.2 += overlap * hit.confidence;
            entry.1.push(SharedEvidence {
                subject: snorm.clone(),
                session_id: hit.session_id.clone(),
                turn_idx: hit.turn_idx,
                doc_id: hit.doc_id.clone(),
                text: hit.evidence.clone(),
            });
        }
        let mut out: Vec<SharedObject> = objects
            .into_iter()
            .map(|(object_norm, (object, evidence, score))| SharedObject {
                object,
                object_norm,
                evidence,
                score,
            })
            .collect();
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.object.cmp(&b.object))
        });
        Some(out)
    }

    /// Activity-seeking temporal span: like `query_temporal_span_general`
    /// but restricted to the person's ACTIVITY relations (Behood's
    /// is_activity verdict). Serves "what ... does X face/experience in
    /// <time>?" — the question asks FOR the activity objects, so we take
    /// all of the person's activities in the window and let the question's
    /// content terms ("health", "incidents") rank them.
    pub fn query_temporal_span_activity(
        &self,
        name: &str,
        start: Ymd,
        end: Ymd,
        question: &str,
    ) -> Option<Vec<SharedObject>> {
        let subjects = self.resolve_subjects(std::slice::from_ref(&name.to_string()))?;
        let snorm = &subjects[0];
        // Luyi: use the focus word to filter on the activity. The focus is
        // what the question asks about ("health incidents"), not the person
        // ("evan") or the verb ("face"). This avoids the "evan matches every
        // hit" poisoning.
        let focus = crate::question_focus::identify_focus(question);
        let content: HashSet<String> = focus
            .focus_terms
            .iter()
            .flat_map(|t| super::catalog::query_tokens(t))
            .collect();
        if content.is_empty() {
            return None;
        }
        let mut objects: HashMap<String, (String, Vec<SharedEvidence>, f32)> = HashMap::new();
        for hit in self.hits_for_any(snorm) {
            // Only real activities, not copulas/statives (Behood's verdict).
            if !hit.is_activity {
                continue;
            }
            let date = match hit.date {
                Some(d) => d,
                None => continue,
            };
            if date < start || date > end {
                continue;
            }
            let hit_terms: HashSet<String> =
                super::catalog::query_tokens(&format!("{} {}", hit.object, hit.evidence));
            let overlap = content.intersection(&hit_terms).count() as f32;
            if overlap == 0.0 {
                continue;
            }
            let entry = objects
                .entry(hit.object_norm.clone())
                .or_insert_with(|| (hit.object.clone(), Vec::new(), 0.0));
            entry.2 += overlap * hit.confidence;
            entry.1.push(SharedEvidence {
                subject: snorm.clone(),
                session_id: hit.session_id.clone(),
                turn_idx: hit.turn_idx,
                doc_id: hit.doc_id.clone(),
                text: hit.evidence.clone(),
            });
        }
        let mut out: Vec<SharedObject> = objects
            .into_iter()
            .map(|(object_norm, (object, evidence, score))| SharedObject {
                object,
                object_norm,
                evidence,
                score,
            })
            .collect();
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.object.cmp(&b.object))
        });
        Some(out)
    }
}

// ---------------------------------------------------------------------------
// Question analysis: (persons, time window, answer type) -> structured query.
// ---------------------------------------------------------------------------

/// A structured fact query over the relation index.
///
/// Three independent analyzers extract person candidates (capitalized words
/// that are not question openers or month names), an optional time window
/// (general date grammar), and the answer type (wh-word / answer noun ->
/// family, declarative). Composition then picks the query: 2+ persons ->
/// shared-relation intersection; exactly one person plus a time window ->
/// temporal-span filter. Anything else declines (None) to the adaptive
/// retrieval path. Adding a question shape never adds a branch here; it
/// adds an entry to the answer-type table or the date grammar.
#[derive(Debug, Clone)]
pub struct StructuredFactQuery {
    /// Person display names as written in the question.
    pub persons: Vec<String>,
    pub family: PredicateFamily,
    /// Inclusive date window, when the question carries one.
    pub time_window: Option<(Ymd, Ymd)>,
    pub expect_place: bool,
    /// Directed communication ("what did X tell Y"): Some when the question
    /// asks for content X communicated to Y. None for all other shapes.
    pub directed: Option<DirectedComm>,
}

/// Directed communication query: X (speaker) communicated something to Y
/// (recipient) about the topic keywords. E.g. "What is Dave's advice to
/// Calvin regarding his dreams?" -> speaker=Dave, recipient=Calvin,
/// topic=["dreams"].
#[derive(Debug, Clone)]
pub struct DirectedComm {
    /// Who communicated (the turn speaker to filter by).
    pub speaker: String,
    /// Who received it (the relation subject to match).
    pub recipient: String,
    /// Topic keywords from the question (matched against object text).
    pub topic_keywords: Vec<String>,
}

/// Question openers / auxiliaries that are never person names.
const NON_NAMES: &[&str] = &[
    "what", "which", "where", "when", "who", "whom", "whose", "how", "did", "does", "do", "is",
    "are", "was", "were", "has", "have", "had", "can", "could", "would", "will",
];

/// Person-name candidates: capitalized words that are not question openers
/// or month names. Exact/fuzzy resolution happens downstream in
/// [`RelationIndex::resolve_subjects`]; names with no evidence in the data
/// (e.g. "Jean", a distinct person from another conversation) are kept as
/// candidates so the structured path can decline honestly.
fn extract_person_candidates(question: &str) -> Vec<String> {
    let mut out = Vec::new();
    for word in question.split_whitespace() {
        let clean: String = word
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_string();
        let lower = clean.to_lowercase();
        if clean.len() >= 3
            && clean.chars().next().is_some_and(|c| c.is_uppercase())
            && !NON_NAMES.contains(&lower.as_str())
            && month_number(&lower).is_none()
        {
            out.push(clean);
        }
    }
    out
}

/// Find a 4-digit year in text.
fn find_year(text: &str) -> Option<i32> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() == 4 && w.chars().all(|c| c.is_ascii_digit()))
        .find_map(|w| w.parse::<i32>().ok())
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        _ => 30,
    }
}

/// Inclusive date window from general date grammar: "between <date> and
/// <date>", "from <date> to <date>", "in <year>", "in <month> <year>".
/// A date fragment without a year borrows the year found elsewhere in the
/// question ("between August 11 and August 15 2023").
fn extract_time_window(question: &str) -> Option<(Ymd, Ymd)> {
    let ql = question.to_lowercase();
    for (open, sep) in [("between", "and"), ("from", "to")] {
        if let Some(rest) = ql.split(open).nth(1) {
            let mut parts = rest.split(sep);
            let a = parts.next()?.trim();
            let b = parts.next().unwrap_or("").trim();
            if a.is_empty() || b.is_empty() {
                continue;
            }
            let year = find_year(&format!("{a} {b}"))?;
            let d1 = parse_session_date(&format!("{a} {year}"))?;
            let d2 = parse_session_date(&format!("{b} {year}"))?;
            let (start, end) = if d1 <= d2 { (d1, d2) } else { (d2, d1) };
            return Some((start, end));
        }
    }
    // "in <year>" -> whole year; "in <month> <year>" -> whole month.
    if let Some(rest) = ql.split("in ").nth(1) {
        let head: String = rest.chars().take(32).collect();
        if let Some(year) = find_year(&head) {
            let first_word = head
                .split(|c: char| !c.is_alphabetic())
                .find(|w| !w.is_empty())
                .unwrap_or("");
            if let Some(month) = month_number(first_word) {
                return Some((
                    Ymd {
                        year,
                        month,
                        day: 1,
                    },
                    Ymd {
                        year,
                        month,
                        day: days_in_month(year, month),
                    },
                ));
            }
            return Some((
                Ymd {
                    year,
                    month: 1,
                    day: 1,
                },
                Ymd {
                    year,
                    month: 12,
                    day: 31,
                },
            ));
        }
    }
    None
}

/// Communication nouns that signal directed communication ("X's advice to Y").
/// Systematic: covers the communication-noun family, not per-question words.
const COMM_NOUNS: &[&str] = &[
    "advice",
    "suggestion",
    "suggestions",
    "warning",
    "warnings",
    "tip",
    "tips",
    "guidance",
    "counsel",
    "recommendation",
    "recommendations",
];

/// Communication verbs that signal directed communication ("what did X tell Y").
/// Systematic: covers the communication-verb family.
const COMM_VERBS: &[&str] = &[
    "tell",
    "told",
    "say",
    "said",
    "advise",
    "advised",
    "suggest",
    "suggested",
    "warn",
    "warned",
    "recommend",
    "recommended",
];

/// Detect directed communication: "What is X's [advice] to Y [regarding Z]?"
/// or "What did X [tell] Y [about Z]?". Returns (speaker, recipient, topic
/// keywords) when the pattern matches and both persons are in the list.
///
/// The speaker is X (possessive or verb subject), the recipient is Y (after
/// "to"). Topic keywords are the remaining content words after removing
/// question words, the comm noun/verb, and the person names.
fn extract_directed_comm(question: &str, persons: &[String]) -> Option<DirectedComm> {
    if persons.len() != 2 {
        return None;
    }
    let ql = normalized_question(question);
    let words: Vec<&str> = ql.split_whitespace().collect();

    // Pattern 1: "X's <comm_noun> to Y" (e.g. "Dave's advice to Calvin")
    // Pattern 2: "did X <comm_verb> Y" (e.g. "did Dave tell Calvin")
    let mut speaker: Option<&str> = None;
    let mut recipient: Option<&str> = None;
    let mut comm_idx: Option<usize> = None;

    // Pattern 1: look for "<name>'s <comm_noun> to <name>"
    // normalized_question turns "Dave's" into "dave s", so check for "s" after name
    for (i, w) in words.iter().enumerate() {
        if COMM_NOUNS.contains(w) {
            // Look backwards for "<name> s" (possessive)
            if i >= 2 && words[i - 1] == "s" {
                let name_word = words[i - 2];
                // Look forwards for "to <name>"
                if i + 2 < words.len() && words[i + 1] == "to" {
                    let recip_word = words[i + 2];
                    // Match against persons (case-insensitive)
                    let speaker_match = persons.iter().find(|p| p.to_lowercase() == name_word);
                    let recip_match = persons.iter().find(|p| p.to_lowercase() == recip_word);
                    if let (Some(s), Some(r)) = (speaker_match, recip_match) {
                        speaker = Some(s);
                        recipient = Some(r);
                        comm_idx = Some(i);
                        break;
                    }
                }
            }
        }
    }

    // Pattern 2: "[aux] <name> <comm_verb> [to] <name>"
    // e.g. "did Dave tell Calvin", "has Sam recommended to Evan"
    if speaker.is_none() {
        for (i, w) in words.iter().enumerate() {
            if COMM_VERBS.contains(w) && i >= 1 && i + 1 < words.len() {
                let prev_word = words[i - 1];
                // Next word may be "to" (then the name after), or the name directly.
                let (recip_word, recip_idx) = if words[i + 1] == "to" && i + 2 < words.len() {
                    (words[i + 2], i + 2)
                } else {
                    (words[i + 1], i + 1)
                };
                let _ = recip_idx;
                let speaker_match = persons.iter().find(|p| p.to_lowercase() == prev_word);
                let recip_match = persons.iter().find(|p| p.to_lowercase() == recip_word);
                if let (Some(s), Some(r)) = (speaker_match, recip_match) {
                    speaker = Some(s);
                    recipient = Some(r);
                    comm_idx = Some(i);
                    break;
                }
            }
        }
    }

    let (speaker, recipient, comm_idx) = match (speaker, recipient, comm_idx) {
        (Some(s), Some(r), Some(i)) => (s, r, i),
        _ => return None,
    };

    // Topic keywords: content words excluding question words, comm words,
    // person names, and common stopwords.
    let stopwords: std::collections::HashSet<&str> = [
        "what",
        "which",
        "where",
        "when",
        "who",
        "whom",
        "whose",
        "how",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "can",
        "could",
        "would",
        "will",
        "the",
        "a",
        "an",
        "to",
        "of",
        "in",
        "on",
        "for",
        "regarding",
        "about",
        "s",
        "his",
        "her",
        "their",
        "its",
        "my",
        "your",
        "our",
        "kind",
        "or",
        "and",
        "some",
    ]
    .into_iter()
    .collect();
    let person_lowers: std::collections::HashSet<String> =
        persons.iter().map(|p| p.to_lowercase()).collect();

    let mut topic_keywords = Vec::new();
    for (i, w) in words.iter().enumerate() {
        if i == comm_idx {
            continue;
        }
        if stopwords.contains(w) {
            continue;
        }
        if person_lowers.contains(&w.to_string()) {
            continue;
        }
        if COMM_NOUNS.contains(w) || COMM_VERBS.contains(w) {
            continue;
        }
        topic_keywords.push(w.to_string());
    }

    Some(DirectedComm {
        speaker: speaker.to_string(),
        recipient: recipient.to_string(),
        topic_keywords,
    })
}

/// Answer type: (family, expect_place) from the question's wh-word / answer
/// noun. Declarative table; adding an answer shape never changes the
/// analyzers or the composition below.
fn extract_answer_type(question: &str) -> Option<(PredicateFamily, bool)> {
    let ql = normalized_question(question);
    if ql.starts_with("where ") {
        return Some((PredicateFamily::PlacePresence, true));
    }
    if ql.contains("which city") || ql.contains("what city") {
        return Some((PredicateFamily::PlacePresence, true));
    }
    if ql.contains("what places") || ql.contains("what place") {
        return Some((PredicateFamily::PlacePresence, true));
    }
    if ql.contains("volunteering") {
        return Some((PredicateFamily::PlacePresence, false));
    }
    None
}

/// Normalize a question for answer-shape matching: lowercase, non-
/// alphanumeric to spaces, single-spaced.
fn normalized_question(question: &str) -> String {
    question
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whether the question asks FOR a person's activities/experiences
/// (not for a specific fact). Broadened from just "what does X do" to
/// include "what ... does X face/experience/deal with" etc.
/// The question asks for the activity objects; the verb is the means,
/// not the answer.
fn is_activity_seeking(question: &str) -> bool {
    let qlow = question.to_lowercase();
    // "what does X do", "what do X do", "what did X do"
    if qlow.starts_with("what does ")
        || qlow.starts_with("what do ")
        || qlow.starts_with("what did ")
    {
        return true;
    }
    // "what ... does X face", "what ... did X experience", etc.
    // The question asks for things the person faced/experienced.
    let activity_verbs = [
        " face ",
        " faces ",
        " faced ",
        " experience ",
        " experiences ",
        " experienced ",
        " deal with ",
        " deals with ",
        " dealt with ",
        " suffer ",
        " suffers ",
        " suffered ",
        " undergo ",
        " undergoes ",
        " underwent ",
        " encounter ",
        " encounters ",
        " encountered ",
    ];
    if qlow.starts_with("what ") {
        for verb in activity_verbs {
            if qlow.contains(verb) {
                return true;
            }
        }
    }
    false
}

/// What facet of the relation index answers this question: place objects,
/// all objects, or session dates. Returns None when the question is not a
/// one-person activity question the keyword path handles.
///
/// Luyi's design: "the behood provide people as the source, then we have
/// place and thing." Behood judges the question's entities at query time;
/// the kind judgment supplements the keyword matching below. Behood owns
/// judgment; the caller owns knowledge.
pub fn answer_kind_from_behood(
    entities: &[crate::behood_query::QueryEntity],
) -> Option<AnswerKind> {
    // Behood judges "what places" / "which city" as place-kind at query time.
    // This is the query-time half of the kind filtering: the index-time half
    // (object_kind on relations) is already stored.
    if crate::behood_query::query_has_kind(entities, "place") {
        return Some(AnswerKind::Place);
    }
    // Behood judges "when" / "what time" / "how long" as time-kind at query
    // time. Lint-ai uses this to apply the temporal filter.
    if crate::behood_query::query_has_kind(entities, "time") {
        return Some(AnswerKind::Date);
    }
    None
}

pub fn extract_answer_kind(question: &str) -> Option<AnswerKind> {
    let ql = normalized_question(question);
    if ql.starts_with("where ")
        || ql.contains("which city")
        || ql.contains("what city")
        || ql.contains("what places")
        || ql.contains("what place")
    {
        return Some(AnswerKind::Place);
    }
    if ql.starts_with("when ") {
        return Some(AnswerKind::Date);
    }
    if ql.contains("what kind of") || ql.contains("what does") || ql.contains("what do") {
        return Some(AnswerKind::Object);
    }
    None
}

/// Auxiliaries that are never the activity verb.
const AUXILIARIES: &[&str] = &[
    "has", "have", "had", "is", "are", "was", "were", "do", "does", "did", "can", "could", "would",
    "will", "shall", "should",
];

/// Irregular past-tense -> base form for activity matching.
fn lemmatize_verb(verb: &str) -> String {
    match verb {
        "met" => "meet".to_string(),
        "went" => "go".to_string(),
        "won" => "win".to_string(),
        "did" => "do".to_string(),
        "had" => "have".to_string(),
        "was" | "were" => "be".to_string(),
        _ => {
            let v = verb.strip_suffix("ing").unwrap_or(verb);
            let v = v.strip_suffix("ed").unwrap_or(v);
            v.strip_suffix('s').unwrap_or(v).to_string()
        }
    }
}

/// The activity verb in a "what places has <person> <verb>..." question:
/// the first non-auxiliary word after the person name. For relationship-
/// derived keywords ("the places in the relationship"): the verb selects
/// which documents' place objects become keywords.
pub fn extract_activity_verb(question: &str, person: &str) -> Option<String> {
    let ql = question.to_lowercase();
    let person_ql = person.to_lowercase();
    let words: Vec<&str> = ql
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let person_words: Vec<&str> = person_ql
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    // Find the person name, take the first non-auxiliary word after it.
    // Skips "try/tried to <verb>" (takes the infinitive: "tried to develop"
    // -> "develop") and prefers prepositional activity phrases ("on hikes"
    // -> "hike") over earlier prepositions like "with".
    let mut i = 0;
    while i + person_words.len() <= words.len() {
        if words[i..i + person_words.len()] == person_words[..] {
            let rest: Vec<&&str> = words.iter().skip(i + person_words.len()).collect();
            // Prefer "on <activity>" anywhere after the person: "what does
            // Melanie do with her family on hikes" -> "hike".
            for (j, w) in rest.iter().enumerate() {
                if **w == "on" && j + 1 < rest.len() {
                    return Some(lemmatize_verb(rest[j + 1]));
                }
            }
            let mut j = 0;
            while j < rest.len() {
                let w = rest[j];
                if NON_NAMES.contains(w) || AUXILIARIES.contains(w) {
                    j += 1;
                    continue;
                }
                // "tried to develop" -> skip to the infinitive.
                if (*w == "try" || *w == "tried" || *w == "tries")
                    && j + 2 < rest.len()
                    && *rest[j + 1] == "to"
                {
                    return Some(lemmatize_verb(rest[j + 2]));
                }
                return Some(lemmatize_verb(w));
            }
            return None;
        }
        i += 1;
    }
    None
}

/// Analyze a question into a structured fact query, or decline (None) to
/// the adaptive retrieval path.
///
/// Composition: 2+ persons -> shared-relation intersection; exactly one
/// person plus a time window -> temporal-span filter; exactly one person
/// with a place-seeking answer type and no time window -> the
/// relationship-keyword path (place-like objects from the person's
/// activity documents become routing keywords; see
/// [`RelationIndex::place_keywords_for_activity`]). When the answer type
/// is not a known predicate family, exactly one person + a time window
/// still gets structured handling via the family-agnostic temporal-span
/// path ([`PredicateFamily::Any`]: person + window + content overlap)
/// instead of declining. Anything else declines (None).
pub fn analyze_fact_question(question: &str) -> Option<StructuredFactQuery> {
    // Query-time behood: behood judges the question's entities and returns
    // (text, kind) pairs. We use the text for matching and the kind for
    // filtering. Falls back to the capitalized-word heuristic when behood
    // is unavailable (fail-open).
    //
    // Luyi's design: "the behood provide people as the source, then we have
    // place and thing." Behood owns judgment; the caller owns knowledge.
    let behood_entities = crate::behood_query::analyze_query_entities(question);
    let persons = if behood_entities.is_empty() {
        extract_person_candidates(question)
    } else {
        let behood_persons = crate::behood_query::query_persons(&behood_entities);
        if behood_persons.is_empty() {
            extract_person_candidates(question)
        } else {
            behood_persons
        }
    };
    if persons.is_empty() {
        return None;
    }
    let time_window = extract_time_window(question);
    // Answer kind: behood's query-time kind judgment first (it judges
    // "what places" as place-kind), falling back to keyword matching.
    // Luyi's design: behood provides the kinds; lint-ai uses them.
    let answer_kind =
        answer_kind_from_behood(&behood_entities).or_else(|| extract_answer_kind(question));
    // Object/date-seeking questions with one person and no time window take
    // the generalized keyword path directly, bypassing the answer-type
    // fallback (which would decline them).
    if persons.len() == 1
        && time_window.is_none()
        && matches!(
            answer_kind,
            Some(AnswerKind::Object) | Some(AnswerKind::Date)
        )
    {
        return Some(StructuredFactQuery {
            persons,
            family: PredicateFamily::Any,
            time_window: None,
            expect_place: false,
            directed: None,
        });
    }
    let (family, expect_place) = match extract_answer_type(question) {
        Some(typed) => typed,
        None => match (persons.len(), time_window) {
            // No known answer type, but the person + window anchor is enough
            // for the general temporal-span path.
            (1, Some(_)) => (PredicateFamily::Any, false),
            // Directed communication ("what is X's advice to Y"): 2 persons,
            // no time window, no other answer type. The directed path queries
            // relations by speaker/recipient, not by predicate family.
            _ => {
                if let Some(directed) = extract_directed_comm(question, &persons) {
                    return Some(StructuredFactQuery {
                        persons,
                        family: PredicateFamily::Any,
                        time_window: None,
                        expect_place: false,
                        directed: Some(directed),
                    });
                }
                return None;
            }
        },
    };
    match (persons.len(), time_window) {
        (1, Some(window)) => Some(StructuredFactQuery {
            persons,
            family,
            time_window: Some(window),
            expect_place,
            directed: None,
        }),
        // The shared-intersection path is family-typed; Any never reaches it
        // (the fallback above declines for 2+ persons).
        (2.., _) if family != PredicateFamily::Any => Some(StructuredFactQuery {
            persons,
            family,
            time_window: None,
            expect_place,
            directed: None,
        }),
        // One person, place-seeking, no time window: the relationship
        // keyword path. `query_structured` declines this shape (no
        // document hits); the caller uses `place_keywords_for_activity`
        // to expand the routing query instead.
        (1, None) if expect_place => Some(StructuredFactQuery {
            persons,
            family,
            time_window: None,
            expect_place,
            directed: None,
        }),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Production path: build the relation index from indexed source documents
// and run the structured-fact query inside the serving search path.
//
// The benchmark shim used to own this wiring per-conversation; it now lives
// here so every serving path (HTTP server, MCP, Python bindings) gets the
// structured-fact behavior from the same code.
// ---------------------------------------------------------------------------

/// One structured-fact hit: the evidence document id, its score, and the
/// relation evidence label (e.g. "shared relation: Rome") for consumers.
#[derive(Debug, Clone)]
pub struct StructuredHit {
    pub doc_id: String,
    pub score: f32,
    pub confidence: f32,
    pub evidence_label: String,
}

/// Split a leading "Speaker: " prefix off document content.
/// Benchmark corpora store turns as "Name: text" with no author_agent;
/// production documents may carry the speaker in author_agent instead.
fn split_speaker_prefix(content: &str) -> (Option<String>, String) {
    let mut parts = content.splitn(2, ':');
    let head = parts.next().unwrap_or("").trim();
    if let Some(rest) = parts.next() {
        let looks_like_name = !head.is_empty()
            && head.len() <= 40
            && head
                .chars()
                .all(|c| c.is_alphabetic() || c == ' ' || c == '-' || c == '\'')
            && rest.starts_with(' ');
        if looks_like_name {
            return (Some(head.to_string()), rest.trim_start().to_string());
        }
    }
    (None, content.to_string())
}

/// Build relation turns from indexed source documents.
///
/// This is the production counterpart to the benchmark's per-conversation
/// turn list: speaker from the "Name: " content prefix (falling back to
/// author_agent), session from the document's group, date from its
/// timestamp. Turn index parses from a trailing "/turn/{n}" source marker
/// when present, else 0 (it is carried as evidence metadata, not ranked on).
pub fn relation_turns_from_docs(docs: &[&crate::SourceDocument]) -> Vec<RelationTurn> {
    let mut turns: Vec<RelationTurn> = Vec::with_capacity(docs.len());
    for doc in docs {
        let (prefix_speaker, text) = split_speaker_prefix(&doc.content);
        let speaker = prefix_speaker
            .or_else(|| doc.author_agent.clone())
            .unwrap_or_default();
        let turn_idx = doc
            .source
            .rsplit("/turn/")
            .next()
            .and_then(|t| t.parse::<usize>().ok())
            .unwrap_or(0);
        turns.push(RelationTurn {
            speaker,
            text,
            session_id: doc.group_id.clone().unwrap_or_else(|| doc.doc_id.clone()),
            turn_idx,
            doc_id: doc.doc_id.clone(),
            session_date: doc.timestamp.clone(),
        });
    }
    turns.sort_by(|a, b| {
        a.session_id
            .cmp(&b.session_id)
            .then(a.turn_idx.cmp(&b.turn_idx))
    });
    turns
}

/// Run the structured-fact path over a question against a relation index.
///
/// Returns `None` when the question is not a structured fact question or
/// the index holds no evidence for it, so callers fall through to the
/// lexical path. Evidence flattens to document ids (deduped, best score
/// wins), ready to blend ahead of lexical `SearchResult`s.
pub fn query_structured(index: &RelationIndex, question: &str) -> Option<Vec<StructuredHit>> {
    let fq = analyze_fact_question(question)?;
    // Directed communication ("what did X tell Y") takes precedence: it has
    // its own speaker/recipient/topic shape that doesn't fit the
    // person-count/family arms below.
    if let Some(directed) = &fq.directed {
        return index.query_directed(
            &directed.speaker,
            &directed.recipient,
            &directed.topic_keywords,
        );
    }
    let (label, objects): (&str, Vec<SharedObject>) = match (fq.persons.len(), fq.time_window) {
        (2.., _) => (
            "shared relation",
            index.query_shared(&fq.persons, fq.family, fq.expect_place)?,
        ),
        (1, Some((start, end))) => (
            "temporal relation",
            // Activity-seeking with time window: intersect activity docs with
            // temporal filter. E.g. "What health incidents does Evan face in
            // 2023?" -> Evan's activity docs from 2023, ranked by "health".
            if is_activity_seeking(question) {
                index.query_temporal_span_activity(&fq.persons[0], start, end, question)?
            } else if fq.family == PredicateFamily::Any {
                index.query_temporal_span_general(&fq.persons[0], start, end, question)?
            } else {
                index.query_temporal_span(&fq.persons[0], fq.family, start, end, fq.expect_place)?
            },
        ),
        // One person, place-seeking, no time window: the subject's activity
        // documents. Returns doc_ids directly (not via SharedObject).
        // Luyi's design: "(person, verb) identifies evidence documents;
        // behood kinds are metadata on the answers." The kind-filtered
        // answer objects (place/event via place_like) go into the evidence
        // label so the caller sees which objects are the answers.
        (1, None) if fq.expect_place => {
            let activity = extract_activity_verb(question, &fq.persons[0])?;
            let doc_ids = index.docs_for_activity(&fq.persons[0], &activity);
            if doc_ids.is_empty() {
                return None;
            }
            let answers = index.keywords_for_question(&fq.persons[0], &activity, AnswerKind::Place);
            let answer_str = if answers.is_empty() {
                String::new()
            } else {
                format!(" answers: {}", answers.join(", "))
            };
            let hits: Vec<StructuredHit> = doc_ids
                .into_iter()
                .map(|doc_id| StructuredHit {
                    doc_id,
                    score: 1000.0,
                    confidence: 1.0,
                    evidence_label: format!(
                        "place-seeking: {} {}{}",
                        fq.persons[0], activity, answer_str
                    ),
                })
                .collect();
            return Some(hits);
        }
        // One person, activity-seeking ("what does X do"), no time window:
        // Luyi's design: "(person, verb) identifies evidence documents."
        // The question asks FOR the activity, so we can't extract it from
        // the question. Return the person's activity documents (filtered by
        // Behood's is_activity verdict); the lexical score ranks by the
        // question's context terms ("hikes", "family").
        // Only for "what does/do/did" — "when" questions fall through to
        // lexical (they're temporal, not activity-seeking).
        (1, None) => {
            if !is_activity_seeking(question) {
                return None;
            }
            let person = &fq.persons[0];
            // "what does X do" asks FOR the activity — don't extract from
            // the question (the "hike" in "on hikes" is context, not the
            // answer). Get all of the person's real activities.
            let doc_ids = index.docs_for_person_activities(person);
            if doc_ids.is_empty() {
                return None;
            }
            let hits: Vec<StructuredHit> = doc_ids
                .into_iter()
                .map(|doc_id| StructuredHit {
                    doc_id,
                    score: 1000.0,
                    confidence: 1.0,
                    evidence_label: format!("activity-seeking: {}", person),
                })
                .collect();
            return Some(hits);
        }
        _ => return None,
    };
    let mut seen = HashSet::new();
    let mut hits = Vec::new();
    for obj in &objects {
        for ev in &obj.evidence {
            if seen.insert(ev.doc_id.clone()) {
                hits.push(StructuredHit {
                    doc_id: ev.doc_id.clone(),
                    score: 1000.0 + obj.score,
                    confidence: obj.score,
                    evidence_label: format!("{label}: {}", obj.object),
                });
            }
        }
    }
    if hits.is_empty() {
        None
    } else {
        Some(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Raw triples exactly as `scripts/spacy_relations.py` emits them for the
    /// gold turns (verified by running the script; see /tmp/gen_fix.py).
    fn fixture_raw() -> Vec<RawRelation> {
        serde_json::from_str(
            r#"[
            {"subject":"Gina","predicate":"be_to","object":"Rome","is_place":true,"session_id":"conv-30::session_2","turn_idx":5,"doc_id":"d1","session_date":null,"evidence":"Gina: Been only to Rome once.","confidence":0.9},
            {"subject":"Jon","predicate":"take_to","object":"Rome","is_place":true,"session_id":"conv-30::session_15","turn_idx":1,"doc_id":"d2","session_date":null,"evidence":"Jon: Took a short trip last week to Rome.","confidence":0.9},
            {"subject":"John","predicate":"go_to","object":"homeless shelter","is_place":false,"session_id":"conv-41::session_3","turn_idx":5,"doc_id":"d3","session_date":null,"evidence":"John: We went to a homeless shelter.","confidence":1.0},
            {"subject":"Maria","predicate":"volunteer_at","object":"homeless shelter","is_place":false,"session_id":"conv-41::session_2","turn_idx":1,"doc_id":"d4","session_date":null,"evidence":"Maria: I volunteer at a homeless shelter.","confidence":0.85},
            {"subject":"Maria","predicate":"volunteer_at","object":"yesterday","is_place":false,"session_id":"conv-41::session_2","turn_idx":1,"doc_id":"d4","session_date":null,"evidence":"Maria: I volunteer at a homeless shelter.","confidence":1.0},
            {"subject":"John","predicate":"take_to","object":"new place","is_place":false,"session_id":"conv-43::session_6","turn_idx":0,"doc_id":"d5","session_date":"1:08 pm on 11 August, 2023","evidence":"John: Took a trip to a new place.","confidence":0.9},
            {"subject":"John","predicate":"be_in","object":"Chicago","is_place":true,"session_id":"conv-43::session_6","turn_idx":2,"doc_id":"d6","session_date":"1:08 pm on 11 August, 2023","evidence":"John: I was in Chicago.","confidence":1.0},
            {"subject":"John","predicate":"meet_up_with","object":"teammates","is_place":false,"session_id":"conv-43::session_7","turn_idx":0,"doc_id":"d7","session_date":"7:54 pm on 17 August, 2023","evidence":"John: I met back up with my teammates.","confidence":1.0}
        ]"#,
        )
        .expect("valid fixture json")
    }

    fn fixture_turns() -> Vec<RelationTurn> {
        ["Gina", "Jon", "John", "Maria"]
            .into_iter()
            .map(|s| RelationTurn {
                speaker: s.to_string(),
                text: String::new(),
                session_id: "s".to_string(),
                turn_idx: 0,
                doc_id: "d".to_string(),
                session_date: None,
            })
            .collect()
    }

    fn fixture_index() -> RelationIndex {
        RelationIndex::build(&fixture_turns(), &fixture_raw())
    }

    /// Two-person index (Jon + Gina) for the Rome shared-object case.
    fn fixture_index_rome() -> RelationIndex {
        let turns = ["Jon", "Gina"]
            .into_iter()
            .map(|s| RelationTurn {
                speaker: s.to_string(),
                text: String::new(),
                session_id: "s".to_string(),
                turn_idx: 0,
                doc_id: "d".to_string(),
                session_date: None,
            })
            .collect::<Vec<_>>();
        let raw: Vec<RawRelation> = fixture_raw()
            .into_iter()
            .filter(|r| r.session_id.starts_with("conv-30"))
            .collect();
        RelationIndex::build(&turns, &raw)
    }

    #[test]
    fn raw_relation_deserializes_script_output() {
        let raw = fixture_raw();
        assert_eq!(raw.len(), 8);
        assert_eq!(raw[1].predicate, "take_to");
        assert!(raw[1].is_place);
        assert_eq!(
            raw[6].session_date.as_deref(),
            Some("1:08 pm on 11 August, 2023")
        );
    }

    #[test]
    fn raw_relation_object_kind_defaults_to_thing() {
        // Script output predating object_kind still deserializes: the
        // kind is fail-open, never a hard error.
        let raw: Vec<RawRelation> = serde_json::from_str(
            r#"[{"subject":"Nate","predicate":"go_to","object":"game convention","is_place":false,"session_id":"s","turn_idx":0,"doc_id":"d","session_date":null,"evidence":"e","confidence":1.0}]"#,
        )
        .unwrap();
        assert_eq!(raw[0].object_kind, "thing");
        // New output carries behood's kind for the object NP.
        let raw: Vec<RawRelation> = serde_json::from_str(
            r#"[{"subject":"Nate","predicate":"win","object":"tournament","is_place":false,"object_kind":"event","session_id":"s","turn_idx":0,"doc_id":"d","session_date":null,"evidence":"e","confidence":1.0}]"#,
        )
        .unwrap();
        assert_eq!(raw[0].object_kind, "event");
    }

    #[test]
    fn place_like_honors_behood_kind() {
        let hit = RelationHit {
            object: "tournament".to_string(),
            object_norm: "tournament".to_string(),
            is_place: false,
            object_kind: "event".to_string(),
            is_activity: true,
            session_id: "s".to_string(),
            turn_idx: 0,
            doc_id: "d".to_string(),
            evidence: "e".to_string(),
            confidence: 1.0,
            date: None,
            speaker: "Nate".to_string(),
        };
        assert!(
            hit.place_like(),
            "behood event-kind counts as a place answer"
        );
        let ner = RelationHit {
            is_place: true,
            object_kind: "thing".to_string(),
            ..hit.clone()
        };
        assert!(ner.place_like(), "spaCy-NER place still counts");
        let other = RelationHit {
            object_kind: "thing".to_string(),
            ..hit.clone()
        };
        assert!(!other.place_like(), "non-place objects stay filtered out");
    }

    #[test]
    fn query_shared_place_filter_accepts_behood_places() {
        // conv-42_q1 shape: neither decisive object is spaCy-NER-tagged
        // (is_place=false), but behood kinds them place/event, so a
        // place-seeking query keeps both and drops the non-place object.
        let turns = ["Nate"]
            .into_iter()
            .map(|s| RelationTurn {
                speaker: s.to_string(),
                text: String::new(),
                session_id: "s".to_string(),
                turn_idx: 0,
                doc_id: "d".to_string(),
                session_date: None,
            })
            .collect::<Vec<_>>();
        let mk = |predicate: &str, object: &str, object_kind: &str| RawRelation {
            subject: "Nate".to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            is_place: false,
            object_kind: object_kind.to_string(),
            is_activity: true,
            session_id: "conv-42::session_23".to_string(),
            turn_idx: 0,
            doc_id: "d".to_string(),
            session_date: None,
            evidence: "Nate: ...".to_string(),
            confidence: 1.0,
            coref: None,
        };
        let raw = vec![
            mk("go_to", "game convention", "place"),
            mk("be_in", "video game tournament", "event"),
            // Real noise shape from the fixture: a time adverbial under a
            // place-presence predicate must not leak into place answers.
            mk("volunteer_at", "yesterday", "thing"),
        ];
        let idx = RelationIndex::build(&turns, &raw);
        let shared = idx
            .query_shared(&["Nate".to_string()], PredicateFamily::PlacePresence, true)
            .expect("resolves");
        let objects: HashSet<&str> = shared.iter().map(|s| s.object.as_str()).collect();
        assert!(
            objects.contains("game convention"),
            "place-kind kept: {objects:?}"
        );
        assert!(
            objects.contains("video game tournament"),
            "event-kind kept: {objects:?}"
        );
        assert_eq!(shared.len(), 2, "non-place filtered: {shared:?}");
    }

    #[test]
    fn raw_relation_coref_field_optional() {
        // Old script output without "coref" still deserializes.
        let raw: Vec<RawRelation> = serde_json::from_str(
            r#"[{"subject":"Jon","predicate":"take","object":"trip","is_place":false,"session_id":"s","turn_idx":0,"doc_id":"d","session_date":null,"evidence":"e","confidence":1.0}]"#,
        )
        .unwrap();
        assert_eq!(raw[0].coref, None);
        // New output carries pronoun-resolution provenance.
        let raw: Vec<RawRelation> = serde_json::from_str(
            r#"[{"subject":"Maria","predicate":"invite_to","object":"Paris","is_place":true,"session_id":"s","turn_idx":0,"doc_id":"d","session_date":null,"evidence":"e","confidence":0.8,"coref":"She->Maria"}]"#,
        )
        .unwrap();
        assert_eq!(raw[0].coref.as_deref(), Some("She->Maria"));
        // A coref-resolved non-speaker subject folds into the person list
        // and resolves for shared-relation queries.
        let turns = ["Jon"]
            .into_iter()
            .map(|s| RelationTurn {
                speaker: s.to_string(),
                text: String::new(),
                session_id: "s".to_string(),
                turn_idx: 0,
                doc_id: "d".to_string(),
                session_date: None,
            })
            .collect::<Vec<_>>();
        let idx = RelationIndex::build(&turns, &raw);
        assert_eq!(
            idx.resolve_subjects(&["Maria".to_string()]),
            Some(vec!["maria".to_string()])
        );
    }

    #[test]
    fn predicate_family_lexicon() {
        for p in [
            "be_to",
            "go_to",
            "come_to",
            "travel_to",
            "take_to",
            "visit",
            "be_in",
            "stay_in",
            "volunteer_at",
            "work_at",
        ] {
            assert_eq!(
                predicate_family(p),
                Some(PredicateFamily::PlacePresence),
                "predicate {p}"
            );
        }
        for p in ["donate", "take", "have", "meet_up_with", "twist", "love"] {
            assert_eq!(predicate_family(p), None, "predicate {p}");
        }
    }

    #[test]
    fn build_indexes_persons_and_triples() {
        let idx = fixture_index();
        assert_eq!(idx.triple_count(), 8);
        for p in ["gina", "jon", "john", "maria"] {
            assert!(idx.person_norms.contains(&p.to_string()), "missing {p}");
        }
    }

    #[test]
    fn resolve_subjects_exact_and_fuzzy() {
        // Two-person conversation.
        let turns = ["Jon", "Gina"]
            .into_iter()
            .map(|s| RelationTurn {
                speaker: s.to_string(),
                text: String::new(),
                session_id: "s".to_string(),
                turn_idx: 0,
                doc_id: "d".to_string(),
                session_date: None,
            })
            .collect::<Vec<_>>();
        let raw: Vec<RawRelation> = serde_json::from_str(
            r#"[
            {"subject":"Jon","predicate":"take_to","object":"Rome","is_place":true,"session_id":"s","turn_idx":0,"doc_id":"d","session_date":null,"evidence":"Jon: Took a trip to Rome.","confidence":0.9}
        ]"#,
        )
        .unwrap();
        let idx = RelationIndex::build(&turns, &raw);
        assert_eq!(
            idx.resolve_subjects(&["Jon".to_string(), "Gina".to_string()]),
            Some(vec!["jon".to_string(), "gina".to_string()])
        );
        // Fuzzy: John -> jon (distance 1).
        assert_eq!(
            idx.resolve_subjects(&["John".to_string(), "Gina".to_string()]),
            Some(vec!["jon".to_string(), "gina".to_string()])
        );
        // "Jean" is a distinct person with no evidence in this index: no
        // elimination mapping to whoever is left over; resolution declines.
        assert_eq!(
            idx.resolve_subjects(&["Jean".to_string(), "John".to_string()]),
            None
        );
    }

    #[test]
    fn query_shared_intersects_across_predicates() {
        let idx = fixture_index_rome();
        // Gina (exact) + John -> jon (fuzzy distance 1).
        let shared = idx
            .query_shared(
                &["Gina".to_string(), "John".to_string()],
                PredicateFamily::PlacePresence,
                true,
            )
            .expect("resolves via exact+fuzzy");
        assert_eq!(shared.len(), 1, "shared: {shared:?}");
        assert_eq!(shared[0].object, "Rome");
        assert_eq!(shared[0].evidence.len(), 2);
        let sessions: HashSet<&str> = shared[0]
            .evidence
            .iter()
            .map(|e| e.session_id.as_str())
            .collect();
        assert!(sessions.contains("conv-30::session_2"));
        assert!(sessions.contains("conv-30::session_15"));
    }

    #[test]
    fn query_shared_volunteering_family_bridge() {
        let idx = fixture_index();
        let shared = idx
            .query_shared(
                &["John".to_string(), "Maria".to_string()],
                PredicateFamily::PlacePresence,
                false,
            )
            .expect("resolves");
        // "homeless shelter" intersects; Maria's noisy "yesterday" does not.
        assert_eq!(shared.len(), 1, "shared: {shared:?}");
        assert_eq!(shared[0].object, "homeless shelter");
    }

    #[test]
    fn query_shared_unresolvable_declines() {
        let idx = fixture_index();
        // "Zelda" appears nowhere in the index, so the shared query declines.
        assert!(idx
            .query_shared(
                &["Zelda".to_string(), "Jon".to_string()],
                PredicateFamily::PlacePresence,
                true
            )
            .is_none());
    }

    #[test]
    fn parse_session_date_formats() {
        assert_eq!(
            parse_session_date("1:08 pm on 11 August, 2023"),
            Some(Ymd {
                year: 2023,
                month: 8,
                day: 11
            })
        );
        assert_eq!(
            parse_session_date("7:54 pm on 17 August, 2023"),
            Some(Ymd {
                year: 2023,
                month: 8,
                day: 17
            })
        );
        assert_eq!(
            parse_session_date("August 11th 2023"),
            Some(Ymd {
                year: 2023,
                month: 8,
                day: 11
            })
        );
        assert_eq!(parse_session_date("no date here"), None);
        assert_eq!(parse_session_date("August 2023"), None);
    }

    #[test]
    fn query_temporal_span_chicago_window() {
        let idx = fixture_index();
        let start = Ymd {
            year: 2023,
            month: 8,
            day: 11,
        };
        let end = Ymd {
            year: 2023,
            month: 8,
            day: 15,
        };
        let hits = idx
            .query_temporal_span("John", PredicateFamily::PlacePresence, start, end, true)
            .expect("resolves");
        // Only Chicago: "new place" is not place-like (NER), session_7 is
        // outside the window, and meet_up_with is not a place predicate.
        assert_eq!(hits.len(), 1, "hits: {hits:?}");
        assert_eq!(hits[0].object, "Chicago");
        let sessions: HashSet<&str> = hits[0]
            .evidence
            .iter()
            .map(|e| e.session_id.as_str())
            .collect();
        assert_eq!(sessions, HashSet::from(["conv-43::session_6"]));
    }

    #[test]
    fn query_temporal_span_outside_window_empty() {
        let idx = fixture_index();
        let hits = idx
            .query_temporal_span(
                "John",
                PredicateFamily::PlacePresence,
                Ymd {
                    year: 2023,
                    month: 8,
                    day: 16,
                },
                Ymd {
                    year: 2023,
                    month: 8,
                    day: 20,
                },
                true,
            )
            .expect("resolves");
        assert!(hits.is_empty(), "hits: {hits:?}");
    }

    #[test]
    fn query_temporal_span_unresolvable_declines() {
        let idx = fixture_index();
        assert!(idx
            .query_temporal_span(
                "Zelda",
                PredicateFamily::PlacePresence,
                Ymd {
                    year: 2023,
                    month: 8,
                    day: 11
                },
                Ymd {
                    year: 2023,
                    month: 8,
                    day: 15
                },
                true,
            )
            .is_none());
    }

    #[test]
    fn analyze_temporal_span_chicago_question() {
        let q = analyze_fact_question("Where was John between August 11 and August 15 2023?")
            .expect("should analyze");
        assert_eq!(q.persons, vec!["John".to_string()]);
        assert_eq!(q.family, PredicateFamily::PlacePresence);
        assert!(q.expect_place);
        let (start, end) = q.time_window.expect("should have window");
        assert_eq!((start.year, start.month, start.day), (2023, 8, 11));
        assert_eq!((end.year, end.month, end.day), (2023, 8, 15));
    }

    #[test]
    fn analyze_general_temporal_span_question() {
        // No known answer type ("what setback" is not a place question), but
        // 1 person + a time window -> the family-agnostic temporal-span path.
        let q = analyze_fact_question("What setback did Melanie face in October 2023?")
            .expect("should analyze");
        assert_eq!(q.persons, vec!["Melanie".to_string()]);
        assert_eq!(q.family, PredicateFamily::Any);
        assert!(!q.expect_place);
        let (start, end) = q.time_window.expect("should have window");
        assert_eq!((start.year, start.month, start.day), (2023, 10, 1));
        assert_eq!((end.year, end.month, end.day), (2023, 10, 31));
    }

    #[test]
    fn analyze_general_path_declines_without_window() {
        // 1 person, no time window, no known answer type and no answer kind
        // -> decline.
        assert!(analyze_fact_question("What did Melanie discuss with her family?").is_none());
        // 2 persons, no known answer type -> decline (shared path stays typed).
        assert!(analyze_fact_question("What did John and Melanie discuss?").is_none());
    }

    /// Melanie index for the family-agnostic temporal-span path. Kept
    /// separate from fixture_raw so the existing len() assertions hold.
    fn fixture_index_general() -> RelationIndex {
        let turns = ["Melanie"]
            .into_iter()
            .map(|s| RelationTurn {
                speaker: s.to_string(),
                text: String::new(),
                session_id: "s".to_string(),
                turn_idx: 0,
                doc_id: "d".to_string(),
                session_date: None,
            })
            .collect::<Vec<_>>();
        let raw: Vec<RawRelation> = serde_json::from_str(
            r#"[
            {"subject":"Melanie","predicate":"get_hurt","object":"ankle","is_place":false,"session_id":"conv-26::session_17","turn_idx":2,"doc_id":"d8","session_date":"2:00 pm on 12 October, 2023","evidence":"Melanie: I got hurt and had to take a break from pottery.","confidence":0.9},
            {"subject":"Melanie","predicate":"go_to","object":"farmers market","is_place":true,"session_id":"conv-26::session_18","turn_idx":0,"doc_id":"d9","session_date":"10:00 am on 20 October, 2023","evidence":"Melanie: We went to the farmers market.","confidence":0.9},
            {"subject":"Melanie","predicate":"visit","object":"sister","is_place":false,"session_id":"conv-26::session_19","turn_idx":1,"doc_id":"d10","session_date":"5:00 pm on 3 November, 2023","evidence":"Melanie: I visited my sister.","confidence":0.9}
            ]"#,
        )
        .expect("valid fixture json");
        RelationIndex::build(&turns, &raw)
    }

    #[test]
    fn query_temporal_span_general_ranks_content_overlap() {
        let idx = fixture_index_general();
        let start = Ymd {
            year: 2023,
            month: 10,
            day: 1,
        };
        let end = Ymd {
            year: 2023,
            month: 10,
            day: 31,
        };
        let hits = idx
            .query_temporal_span_general(
                "Melanie",
                start,
                end,
                "What injury forced Melanie to take a break from pottery in October 2023?",
            )
            .expect("resolves");
        // The hurt/pottery hit shares distinctive content terms ("break",
        // "pottery"). The farmers-market hit is in-window but content-poor;
        // the focus-word filter (6561402) intentionally drops it since it
        // shares no focus terms with the question. The November visit is
        // outside the window.
        assert_eq!(hits.len(), 1, "hits: {hits:?}");
        assert_eq!(hits[0].object, "ankle");
        let sessions: HashSet<&str> = hits[0]
            .evidence
            .iter()
            .map(|e| e.session_id.as_str())
            .collect();
        assert_eq!(sessions, HashSet::from(["conv-26::session_17"]));
    }

    #[test]
    fn query_temporal_span_general_unresolvable_declines() {
        let idx = fixture_index_general();
        assert!(idx
            .query_temporal_span_general(
                "Zelda",
                Ymd {
                    year: 2023,
                    month: 10,
                    day: 1
                },
                Ymd {
                    year: 2023,
                    month: 10,
                    day: 31
                },
                "What did Zelda do in October 2023?",
            )
            .is_none());
    }

    #[test]
    fn analyze_shared_questions() {
        let q = analyze_fact_question("Which city have both Jean and John visited?")
            .expect("should analyze");
        assert_eq!(q.persons, vec!["Jean".to_string(), "John".to_string()]);
        assert_eq!(q.family, PredicateFamily::PlacePresence);
        assert!(q.expect_place);
        assert!(q.time_window.is_none());

        let q = analyze_fact_question("What type of volunteering have John and Maria both done?")
            .expect("should analyze");
        assert_eq!(q.persons, vec!["John".to_string(), "Maria".to_string()]);
        assert!(!q.expect_place);
    }

    #[test]
    fn analyze_time_window_variants() {
        let q = analyze_fact_question("Where was John from August 11 to August 15 2023?")
            .expect("from/to");
        let (s, e) = q.time_window.unwrap();
        assert_eq!((s.month, s.day), (8, 11));
        assert_eq!((e.month, e.day), (8, 15));

        let q = analyze_fact_question("Where was John in 2023?").expect("in year");
        let (s, e) = q.time_window.unwrap();
        assert_eq!((s.month, s.day), (1, 1));
        assert_eq!((e.month, e.day), (12, 31));

        let q = analyze_fact_question("Where was John in August 2023?").expect("in month");
        let (s, e) = q.time_window.unwrap();
        assert_eq!((s.month, s.day, e.day), (8, 1, 31));
    }

    #[test]
    fn what_places_question_takes_keyword_path() {
        // "What places has Nate met new people?" -> 1 person, place-seeking,
        // no time window: the relationship-keyword composition.
        let q = analyze_fact_question("What places has Nate met new people?")
            .expect("what places analyzes");
        assert_eq!(q.persons, vec!["Nate".to_string()]);
        assert!(q.expect_place);
        assert!(q.time_window.is_none());
        // The activity verb after the person name.
        assert_eq!(
            extract_activity_verb("What places has Nate met new people?", "Nate"),
            Some("meet".to_string())
        );
    }

    #[test]
    fn place_keywords_come_from_activity_documents() {
        // Nate wins a tournament (event) and meets people in the same turn;
        // goes to a convention (place) and meets people in the same turn;
        // meets people in a third turn with no place. Keywords are the
        // place-like objects from the "meet" documents only.
        let raw: Vec<RawRelation> = serde_json::from_str(
            r#"[
            {"subject":"Nate","predicate":"win","object":"another regional video game tournament","is_place":false,"object_kind":"event","session_id":"s14","turn_idx":0,"doc_id":"d14","session_date":null,"evidence":"e1","confidence":1.0},
            {"subject":"Nate","predicate":"meet","object":"new people","is_place":false,"object_kind":"thing","session_id":"s14","turn_idx":0,"doc_id":"d14","session_date":null,"evidence":"e1","confidence":1.0},
            {"subject":"Nate","predicate":"go_to","object":"game convention","is_place":false,"object_kind":"place","session_id":"s23","turn_idx":0,"doc_id":"d23","session_date":null,"evidence":"e2","confidence":1.0},
            {"subject":"Nate","predicate":"meet","object":"new people","is_place":false,"object_kind":"thing","session_id":"s23","turn_idx":0,"doc_id":"d23","session_date":null,"evidence":"e2","confidence":1.0},
            {"subject":"Nate","predicate":"meet","object":"old friends","is_place":false,"object_kind":"thing","session_id":"s30","turn_idx":0,"doc_id":"d30","session_date":null,"evidence":"e3","confidence":1.0},
            {"subject":"Nate","predicate":"live_in","object":"Portland","is_place":true,"object_kind":"place","session_id":"s31","turn_idx":0,"doc_id":"d31","session_date":null,"evidence":"e4","confidence":1.0}
        ]"#,
        )
        .expect("valid fixture json");
        let turns = vec![RelationTurn {
            speaker: "Nate".to_string(),
            text: String::new(),
            session_id: "s".to_string(),
            turn_idx: 0,
            doc_id: "d".to_string(),
            session_date: None,
        }];
        let index = RelationIndex::build(&turns, &raw);
        let mut kw = index.place_keywords_for_activity("Nate", "meet");
        kw.sort();
        // Portland is place-like but Nate never "meets" in its document.
        assert_eq!(
            kw,
            vec![
                "another regional video game tournament".to_string(),
                "game convention".to_string()
            ]
        );
        // Unknown activity -> no keywords, not a panic.
        assert!(index
            .place_keywords_for_activity("Nate", "marry")
            .is_empty());
    }

    #[test]
    fn place_seeking_evidence_carries_kind_filtered_answers() {
        // Luyi's design: "(person, verb) identifies evidence documents;
        // behood kinds are metadata on the answers." The structured hit's
        // evidence label carries the kind-filtered answer objects.
        let raw: Vec<RawRelation> = serde_json::from_str(
            r#"[
            {"subject":"Nate","predicate":"win","object":"another regional video game tournament","is_place":false,"object_kind":"event","session_id":"s14","turn_idx":0,"doc_id":"d14","session_date":null,"evidence":"e1","confidence":1.0},
            {"subject":"Nate","predicate":"meet","object":"new people","is_place":false,"object_kind":"thing","session_id":"s14","turn_idx":0,"doc_id":"d14","session_date":null,"evidence":"e1","confidence":1.0},
            {"subject":"Nate","predicate":"go_to","object":"game convention","is_place":false,"object_kind":"place","session_id":"s23","turn_idx":0,"doc_id":"d23","session_date":null,"evidence":"e2","confidence":1.0},
            {"subject":"Nate","predicate":"meet","object":"new people","is_place":false,"object_kind":"thing","session_id":"s23","turn_idx":0,"doc_id":"d23","session_date":null,"evidence":"e2","confidence":1.0}
        ]"#,
        )
        .expect("valid fixture json");
        let turns = vec![RelationTurn {
            speaker: "Nate".to_string(),
            text: String::new(),
            session_id: "s".to_string(),
            turn_idx: 0,
            doc_id: "d".to_string(),
            session_date: None,
        }];
        let index = RelationIndex::build(&turns, &raw);
        let hits = query_structured(&index, "What places has Nate met new people?")
            .expect("place-seeking question should hit");
        // Both the tournament (event) and the convention (place) survive:
        // place OR event, not strict place-only.
        for h in &hits {
            assert!(
                h.evidence_label
                    .contains("another regional video game tournament"),
                "tournament (event) is a valid place answer: {}",
                h.evidence_label
            );
            assert!(
                h.evidence_label.contains("game convention"),
                "convention (place) is a valid place answer: {}",
                h.evidence_label
            );
            // "new people" (thing) is not a place answer.
            assert!(
                !h.evidence_label.contains("new people"),
                "thing-kind objects are filtered out: {}",
                h.evidence_label
            );
        }
    }

    #[test]
    fn analyze_declines_without_answer_type_or_person() {
        assert!(analyze_fact_question("Did John visit Rome?").is_none());
        // 1 person + a time window now takes the family-agnostic temporal-span
        // path instead of declining (systematic generalization, not a revert
        // of the decline policy: the person + window anchor is sufficient).
        let q = analyze_fact_question("What personal health incidents does Evan face in 2023?")
            .expect("1 person + window analyzes");
        assert_eq!(q.family, PredicateFamily::Any);
        // Single person, place-seeking, no time window takes the
        // relationship-keyword path (not a decline): the person's
        // place-like objects become routing keywords.
        let q = analyze_fact_question("Where was John?").expect("place keyword path");
        assert!(q.expect_place);
        assert!(q.time_window.is_none());
        // Non-place single-person questions without a window take the
        // generalized keyword path when they have an answer kind.
        let q = analyze_fact_question("What does Melanie do with her family on hikes?")
            .expect("object keyword path");
        assert!(!q.expect_place);
        assert!(q.time_window.is_none());
        // Questions with no answer kind still decline.
        assert!(analyze_fact_question("What did Melanie discuss with her family?").is_none());
    }

    #[test]
    fn analyze_temporal_span_end_to_end() {
        // Chicago question: analyzer -> index -> temporal-span hit.
        let turns = ["John"]
            .iter()
            .map(|s| RelationTurn {
                speaker: s.to_string(),
                text: String::new(),
                session_id: "s".to_string(),
                turn_idx: 0,
                doc_id: "d".to_string(),
                session_date: None,
            })
            .collect::<Vec<_>>();
        let raw: Vec<RawRelation> = serde_json::from_str(
            r#"[
            {"subject":"John","predicate":"be_in","object":"Chicago","is_place":true,"session_id":"D6:1","turn_idx":0,"doc_id":"d","session_date":"1:08 pm on 11 August, 2023","evidence":"John: I was in Chicago.","confidence":0.9},
            {"subject":"John","predicate":"be_in","object":"Chicago","is_place":true,"session_id":"D7:1","turn_idx":0,"doc_id":"d","session_date":"7:54 pm on 17 August, 2023","evidence":"John: Chicago again.","confidence":0.9}
        ]"#,
        )
        .unwrap();
        let idx = RelationIndex::build(&turns, &raw);
        let q = analyze_fact_question("Where was John between August 11 and August 15 2023?")
            .expect("should analyze");
        let (start, end) = q.time_window.unwrap();
        let objs = idx
            .query_temporal_span(&q.persons[0], q.family, start, end, q.expect_place)
            .expect("should resolve");
        assert_eq!(objs.len(), 1);
        assert_eq!(objs[0].object, "Chicago");
        assert_eq!(objs[0].evidence[0].session_id, "D6:1");
    }

    // --- Production path: docs -> turns -> structured hits ---

    fn prod_doc(
        doc_id: &str,
        content: &str,
        group_id: Option<&str>,
        timestamp: Option<&str>,
        author_agent: Option<&str>,
        user: &str,
    ) -> crate::SourceDocument {
        let mut filters = std::collections::BTreeMap::new();
        filters.insert("memory_user_id".to_string(), user.to_string());
        crate::SourceDocument {
            doc_id: doc_id.to_string(),
            source: format!("test/{doc_id}"),
            content: content.to_string(),
            concept: "test".to_string(),
            group_id: group_id.map(|s| s.to_string()),
            headings: Vec::new(),
            links: Vec::new(),
            timestamp: timestamp.map(|s| s.to_string()),
            doc_length: content.len(),
            author_agent: author_agent.map(|s| s.to_string()),
            filters,
            key_phrases: Vec::new(),
            key_phrase_extraction_hash: String::new(),
        }
    }

    #[test]
    fn relation_turns_from_docs_maps_speaker_session_date() {
        let docs = vec![
            prod_doc(
                "d1",
                "Gina: Been only to Rome once.",
                Some("sess-a"),
                Some("2020-05-01"),
                None,
                "u1",
            ),
            // No "Name: " prefix and no group: speaker falls back to
            // author_agent, session falls back to the doc id.
            prod_doc("d2", "no prefix here", None, None, Some("agent-x"), "u1"),
        ];
        let refs: Vec<&crate::SourceDocument> = docs.iter().collect();
        let turns = relation_turns_from_docs(&refs);
        assert_eq!(turns.len(), 2);
        // Sorted by (session_id, turn_idx): d2's session falls back to "d2",
        // which sorts before "sess-a".
        assert_eq!(turns[0].doc_id, "d2");
        assert_eq!(turns[0].speaker, "agent-x");
        assert_eq!(turns[0].text, "no prefix here");
        assert_eq!(turns[0].session_id, "d2");
        assert_eq!(turns[1].doc_id, "d1");
        assert_eq!(turns[1].speaker, "Gina");
        assert_eq!(turns[1].text, "Been only to Rome once.");
        assert_eq!(turns[1].session_id, "sess-a");
        assert_eq!(turns[1].session_date.as_deref(), Some("2020-05-01"));
    }

    fn shared_place_index() -> RelationIndex {
        let turns = vec![
            RelationTurn {
                speaker: "Gina".to_string(),
                text: "Been only to Rome once.".to_string(),
                session_id: "sess-a".to_string(),
                turn_idx: 0,
                doc_id: "d1".to_string(),
                session_date: None,
            },
            RelationTurn {
                speaker: "Jon".to_string(),
                text: "Took a short trip last week to Rome.".to_string(),
                session_id: "sess-b".to_string(),
                turn_idx: 0,
                doc_id: "d2".to_string(),
                session_date: None,
            },
        ];
        let raw = vec![
            RawRelation {
                subject: "Gina".to_string(),
                predicate: "go_to".to_string(),
                object: "Rome".to_string(),
                is_place: true,
                object_kind: "place".to_string(),
                is_activity: true,
                session_id: "sess-a".to_string(),
                turn_idx: 0,
                doc_id: "d1".to_string(),
                session_date: None,
                evidence: "Gina: Been only to Rome once.".to_string(),
                confidence: 0.9,
                coref: None,
            },
            RawRelation {
                subject: "Jon".to_string(),
                predicate: "take_to".to_string(),
                object: "Rome".to_string(),
                is_place: true,
                object_kind: "place".to_string(),
                is_activity: true,
                session_id: "sess-b".to_string(),
                turn_idx: 0,
                doc_id: "d2".to_string(),
                session_date: None,
                evidence: "Jon: Took a short trip last week to Rome.".to_string(),
                confidence: 0.9,
                coref: None,
            },
        ];
        RelationIndex::build(&turns, &raw)
    }

    #[test]
    fn query_structured_shared_place_returns_evidence_hits() {
        let index = shared_place_index();
        let hits = query_structured(&index, "Which city have both Gina and Jon visited?")
            .expect("structured fact question should hit");
        let mut ids: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec!["d1", "d2"]);
        for h in &hits {
            assert!(
                h.score > 1000.0,
                "structured evidence blends ahead of lexical"
            );
            assert_eq!(h.evidence_label, "shared relation: Rome");
        }
    }

    #[test]
    fn query_structured_declines_non_fact_questions() {
        let index = shared_place_index();
        assert!(query_structured(&index, "What did Gina and Jon discuss?").is_none());
        assert!(query_structured(&index, "Tell me about Rome.").is_none());
    }

    #[test]
    fn answer_kind_detection() {
        assert_eq!(
            extract_answer_kind("What places has Nate met new people?"),
            Some(AnswerKind::Place)
        );
        assert_eq!(
            extract_answer_kind("Where did Nate go?"),
            Some(AnswerKind::Place)
        );
        assert_eq!(
            extract_answer_kind("What kind of games has James tried to develop?"),
            Some(AnswerKind::Object)
        );
        assert_eq!(
            extract_answer_kind("What does Melanie do with her family on hikes?"),
            Some(AnswerKind::Object)
        );
        assert_eq!(
            extract_answer_kind("When will John start his new job?"),
            Some(AnswerKind::Date)
        );
        assert_eq!(extract_answer_kind("What did Gina and Jon discuss?"), None);
    }

    #[test]
    fn answer_kind_from_behood_place() {
        // Luyi's design: behood judges the question's kinds at query time.
        // A place-kind entity in the question -> AnswerKind::Place.
        use crate::behood_query::QueryEntity;
        let entities = vec![
            QueryEntity {
                text: "What places".to_string(),
                kind: "place".to_string(),
                closed_sets: Vec::new(),
            },
            QueryEntity {
                text: "Nate".to_string(),
                kind: "person".to_string(),
                closed_sets: Vec::new(),
            },
        ];
        assert_eq!(answer_kind_from_behood(&entities), Some(AnswerKind::Place));
        // No place-kind entity -> None (falls back to keyword matching).
        let entities = vec![QueryEntity {
            text: "Nate".to_string(),
            kind: "person".to_string(),
            closed_sets: Vec::new(),
        }];
        assert_eq!(answer_kind_from_behood(&entities), None);
    }

    #[test]
    fn answer_kind_from_behood_time() {
        // Luyi's design: behood judges "when" as time-kind at query time.
        // A time-kind entity in the question -> AnswerKind::Date.
        use crate::behood_query::QueryEntity;
        let entities = vec![
            QueryEntity {
                text: "when".to_string(),
                kind: "time".to_string(),
                closed_sets: Vec::new(),
            },
            QueryEntity {
                text: "John".to_string(),
                kind: "person".to_string(),
                closed_sets: Vec::new(),
            },
        ];
        assert_eq!(answer_kind_from_behood(&entities), Some(AnswerKind::Date));
    }

    #[test]
    fn activity_verb_skips_try_to_infinitive() {
        // "tried to develop" -> the infinitive "develop", not "tried".
        assert_eq!(
            extract_activity_verb("What kind of games has James tried to develop?", "James"),
            Some("develop".to_string())
        );
    }

    #[test]
    fn activity_verb_from_prepositional_phrase() {
        // "on hikes" -> the activity noun "hike".
        assert_eq!(
            extract_activity_verb("What does Melanie do with her family on hikes?", "Melanie"),
            Some("hike".to_string())
        );
    }

    #[test]
    fn object_keywords_come_from_activity_documents() {
        // James develops a football simulator and a virtual world; also
        // plays chess (different activity). Object keywords for "develop"
        // are the developed things only.
        let raw: Vec<RawRelation> = serde_json::from_str(
            r#"[
            {"subject":"James","predicate":"develop","object":"football simulator","is_place":false,"object_kind":"thing","session_id":"s1","turn_idx":0,"doc_id":"d1","session_date":null,"evidence":"e1","confidence":1.0},
            {"subject":"James","predicate":"develop","object":"virtual world inspired by Witcher 3","is_place":false,"object_kind":"thing","session_id":"s1","turn_idx":1,"doc_id":"d1","session_date":null,"evidence":"e2","confidence":1.0},
            {"subject":"James","predicate":"play","object":"chess","is_place":false,"object_kind":"thing","session_id":"s2","turn_idx":0,"doc_id":"d2","session_date":null,"evidence":"e3","confidence":1.0}
        ]"#,
        )
        .unwrap();
        let turns = vec![];
        let index = RelationIndex::build(&turns, &raw);
        let mut kw = index.keywords_for_question("James", "develop", AnswerKind::Object);
        kw.sort_unstable();
        assert_eq!(
            kw,
            vec![
                "football simulator".to_string(),
                "virtual world inspired by Witcher 3".to_string()
            ]
        );
        // Place kind finds nothing (no place-like objects).
        assert!(index
            .keywords_for_question("James", "develop", AnswerKind::Place)
            .is_empty());
    }

    #[test]
    fn date_keywords_come_from_activity_sessions() {
        // John starts his job in a July 2022 session; also visits Rome in
        // March 2022 (different activity). Date keywords for "start" are
        // the job-start months only.
        let raw: Vec<RawRelation> = serde_json::from_str(
            r#"[
            {"subject":"John","predicate":"start","object":"new job","is_place":false,"object_kind":"thing","session_id":"s1","turn_idx":0,"doc_id":"d1","session_date":"10:00 am on 15 July, 2022","evidence":"e1","confidence":1.0},
            {"subject":"John","predicate":"visit","object":"Rome","is_place":true,"object_kind":"place","session_id":"s2","turn_idx":0,"doc_id":"d2","session_date":"10:00 am on 3 March, 2022","evidence":"e2","confidence":1.0}
        ]"#,
        )
        .unwrap();
        let turns = vec![];
        let index = RelationIndex::build(&turns, &raw);
        let kw = index.keywords_for_question("John", "start", AnswerKind::Date);
        assert_eq!(kw, vec!["july 2022".to_string()]);
    }
}
