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
//! Every relation carries the session date as a temporal anchor and a
//! spaCy-NER place flag (`is_place`) backing the place-expectation filter.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::process::{Command, Stdio};

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
#[derive(Debug, Clone, Deserialize)]
pub struct RawKeyPhrase {
    pub text: String,
    pub kind: String,
    pub session_id: String,
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
}

impl PredicateFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            PredicateFamily::PlacePresence => "place-presence",
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
fn python_executable() -> String {
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

/// Run the dependency-parse extractor (`scripts/spacy_relations.py`) over the
/// turns and return raw triples plus grammar-accepted key phrases.
///
/// Never panics: any failure (missing Python/spaCy, bad output) yields an
/// empty output and the caller declines to the adaptive retrieval path.
pub fn extract_relations_via_spacy(turns: &[RelationTurn]) -> ExtractorOutput {
    let script =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/spacy_relations.py");
    if !script.exists() {
        eprintln!("relations: extractor script missing: {}", script.display());
        return ExtractorOutput::default();
    }
    let payload = serde_json::json!({"model": "en_core_web_sm", "turns": turns});
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
            return ExtractorOutput::default();
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
        return ExtractorOutput::default();
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(e) => {
            eprintln!("relations: extractor wait failed: {e}");
            return ExtractorOutput::default();
        }
    };
    if !output.status.success() {
        eprintln!(
            "relations: extractor failed: {}",
            String::from_utf8_lossy(&output.stderr)
                .chars()
                .take(300)
                .collect::<String>()
        );
        return ExtractorOutput::default();
    }
    #[derive(Deserialize)]
    struct Output {
        #[serde(default)]
        relations: Vec<RawRelation>,
        #[serde(default)]
        key_phrases: Vec<RawKeyPhrase>,
    }
    match serde_json::from_slice::<Output>(&output.stdout) {
        Ok(parsed) => ExtractorOutput {
            relations: parsed.relations,
            key_phrases: parsed.key_phrases,
        },
        Err(e) => {
            eprintln!("relations: bad extractor output: {e}");
            ExtractorOutput::default()
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
    pub session_id: String,
    pub turn_idx: usize,
    pub doc_id: String,
    pub evidence: String,
    pub confidence: f32,
    /// Temporal anchor from the session date; None when unparseable.
    pub date: Option<Ymd>,
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

impl RelationIndex {
    /// Build from dialogue turns (persons) and raw extractor triples.
    pub fn build(turns: &[RelationTurn], raw: &[RawRelation]) -> Self {
        let mut idx = RelationIndex::default();
        let mut seen_person: HashSet<String> = HashSet::new();
        for turn in turns {
            let norm = normalize_relation_token(&turn.speaker);
            if seen_person.insert(norm.clone()) {
                idx.persons.push(turn.speaker.clone());
                idx.person_norms.push(norm);
            }
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
            idx.by_sp
                .entry((subject_norm, rel.predicate.clone()))
                .or_default()
                .push(RelationHit {
                    object: rel.object.clone(),
                    object_norm,
                    is_place: rel.is_place,
                    session_id: rel.session_id.clone(),
                    turn_idx: rel.turn_idx,
                    doc_id: rel.doc_id.clone(),
                    evidence: rel.evidence.clone(),
                    confidence: rel.confidence,
                    date,
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
                if expect_place && !hit.is_place {
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
            if expect_place && !hit.is_place {
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

/// Answer type: (family, expect_place) from the question's wh-word / answer
/// noun. Declarative table; adding an answer shape never changes the
/// analyzers or the composition below.
fn extract_answer_type(question: &str) -> Option<(PredicateFamily, bool)> {
    let ql: String = question
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    let ql = ql.split_whitespace().collect::<Vec<_>>().join(" ");
    if ql.starts_with("where ") {
        return Some((PredicateFamily::PlacePresence, true));
    }
    if ql.contains("which city") || ql.contains("what city") {
        return Some((PredicateFamily::PlacePresence, true));
    }
    if ql.contains("volunteering") {
        return Some((PredicateFamily::PlacePresence, false));
    }
    None
}

/// Analyze a question into a structured fact query, or decline (None) to
/// the adaptive retrieval path.
pub fn analyze_fact_question(question: &str) -> Option<StructuredFactQuery> {
    let (family, expect_place) = extract_answer_type(question)?;
    let persons = extract_person_candidates(question);
    if persons.is_empty() {
        return None;
    }
    let time_window = extract_time_window(question);
    match (persons.len(), time_window) {
        (1, Some(window)) => Some(StructuredFactQuery {
            persons,
            family,
            time_window: Some(window),
            expect_place,
        }),
        (2.., _) => Some(StructuredFactQuery {
            persons,
            family,
            time_window: None,
            expect_place,
        }),
        _ => None,
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
    fn analyze_declines_without_answer_type_or_person() {
        assert!(analyze_fact_question("Did John visit Rome?").is_none());
        assert!(
            analyze_fact_question("What personal health incidents does Evan face in 2023?")
                .is_none()
        );
        // Single person without a time window declines to the adaptive path.
        assert!(analyze_fact_question("Where was John?").is_none());
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
}
