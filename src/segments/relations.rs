//! Entity-relation (SPO) extraction and shared-relation query.
//!
//! Rule-based and dependency-free: verb-phrase patterns over dialogue turns
//! with speaker attribution. Covers place-presence predicates (`visited`,
//! `volunteered_at`) used by multi-hop "both X and Y" questions, e.g.
//! "Which city have both Jean and John visited?".
//!
//! Design notes:
//! - Subjects resolve to the turn speaker for first-person / "we" phrasing;
//!   an explicit capitalized name immediately before the verb wins.
//! - Objects are noun phrases after the verb phrase; pronouns and articles
//!   are stripped. No NER dependency: the patterns carry the semantics.
//! - The query side ([`RelationIndex::query_shared`]) intersects objects
//!   across subjects within a predicate family, so "went to X" (visited)
//!   and "volunteer at X" (volunteered_at) can meet on the same object.
//! - If spaCy ever becomes available here, this module is the seam to
//!   upgrade extraction to dependency-parse triples; the store and query
//!   API stay the same.

use std::collections::{HashMap, HashSet};

use regex::Regex;
use serde::Serialize;
use std::sync::OnceLock;

/// One dialogue turn to extract relations from.
#[derive(Debug, Clone)]
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
}

/// One extracted (subject, predicate, object) triple with evidence.
#[derive(Debug, Clone, Serialize)]
pub struct EntityRelation {
    /// Display subject, e.g. "Gina".
    pub subject: String,
    /// Normalized subject for matching.
    pub subject_norm: String,
    /// Canonical predicate: "visited" or "volunteered_at".
    pub predicate: &'static str,
    /// Display object, e.g. "Rome".
    pub object: String,
    /// Normalized object for intersection.
    pub object_norm: String,
    pub session_id: String,
    pub turn_idx: usize,
    pub doc_id: String,
    /// "Speaker: text" evidence line.
    pub evidence: String,
    pub confidence: f32,
}

/// A family of predicates that count as "being somewhere" for intersection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PredicateFamily {
    /// visited, volunteered_at: physical presence at a place.
    PlacePresence,
}

impl PredicateFamily {
    pub fn predicates(&self) -> &'static [&'static str] {
        match self {
            PredicateFamily::PlacePresence => &["visited", "volunteered_at"],
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            PredicateFamily::PlacePresence => "place-presence",
        }
    }
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
// Extraction patterns.
// ---------------------------------------------------------------------------

/// Object capture: a noun phrase, terminated by punctuation, a clause
/// boundary, or end of text. The boundary is a *consuming* group (the
/// `regex` crate has no look-around); group 1 is the object.
const OBJ: &str = r"([A-Za-z][A-Za-z\s\-']{0,38}?)";
const BOUND: &str = r"(?:[.,;:!?]|\s+(?:to|and|but|or|because|when|where|which|that|who|with|for|from|at|in|on)\b|\s*$)";

/// (regex source, predicate, confidence). Order matters: most specific first.
fn patterns() -> &'static [(String, &'static str, f32)] {
    static CELL: OnceLock<Vec<(String, &'static str, f32)>> = OnceLock::new();
    CELL.get_or_init(|| {
        vec![
            // "been only to Rome once" / "have been to Paris"
            (
                format!(r"(?i)\bbeen\s+(?:only\s+)?to\s+{OBJ}{BOUND}"),
                "visited",
                1.0,
            ),
            // "took a short trip last week to Rome"
            (
                format!(
                    r"(?i)\bt(?:ook|ake)\s+(?:a\s+)?(?:\w+\s+){{0,2}}trip\b[^.?!]{{0,30}}?\bto\s+{OBJ}{BOUND}"
                ),
                "visited",
                0.9,
            ),
            // "went to a homeless shelter"
            (
                format!(r"(?i)\bwent\s+to\s+{OBJ}{BOUND}"),
                "visited",
                1.0,
            ),
            // "visited Rome" / "visiting Paris" / "visits Berlin"
            (
                format!(r"(?i)\bvisit(?:s|ed|ing)?\s+{OBJ}{BOUND}"),
                "visited",
                1.0,
            ),
            // "travelled to Tokyo"
            (
                format!(r"(?i)\btravel(?:led|ling|s)?\s+to\s+{OBJ}{BOUND}"),
                "visited",
                1.0,
            ),
            // "volunteer at a shelter" / "volunteered with Habitat"
            (
                format!(
                    r"(?i)\bvolunteer(?:s|ed|ing)?\s+(?:at|with|for)\s+{OBJ}{BOUND}"
                ),
                "volunteered_at",
                1.0,
            ),
            // relative clause: "a homeless shelter I volunteer at".
            // Anchored on the preceding "to " so the object starts at the
            // noun phrase (no look-around: the regex crate forbids it).
            (
                r"(?i)\bto\s+([A-Za-z][A-Za-z\-']*(?:\s+[A-Za-z][A-Za-z\-']*){0,3})\s+(?:that\s+)?i\s+volunteer\s+at\b"
                    .to_string(),
                "volunteered_at",
                0.85,
            ),
        ]
    })
}

/// Compiled patterns, cached (extraction runs per turn at index time).
fn compiled_patterns() -> &'static [(Regex, &'static str, f32)] {
    static CELL: OnceLock<Vec<(Regex, &'static str, f32)>> = OnceLock::new();
    CELL.get_or_init(|| {
        patterns()
            .iter()
            .map(|(src, pred, conf)| {
                (
                    Regex::new(src).expect("valid relation pattern"),
                    *pred,
                    *conf,
                )
            })
            .collect()
    })
}

/// Words stripped from the front of a captured object.
const LEADING_STRIP: &[&str] = &[
    "a", "an", "the", "to", "at", "in", "of", "for", "with", "on",
];

/// Trailing adverbs stripped from a captured object ("Rome once" -> "Rome").
const TRAILING_STRIP: &[&str] = &[
    "once",
    "twice",
    "yesterday",
    "today",
    "tomorrow",
    "again",
    "before",
    "ago",
    "recently",
    "lately",
    "already",
    "just",
    "still",
    "there",
];

/// Pronouns / deictics that are never valid objects.
const BAD_OBJECTS: &[&str] = &[
    "it", "them", "him", "her", "us", "me", "this", "that", "there", "here",
];

/// Pronouns that never name an explicit subject.
const PRONOUNS: &[&str] = &["i", "we", "you", "he", "she", "it", "they", "this", "that"];

/// Greetings / interjections that look like names but aren't.
const NOT_NAMES: &[&str] = &[
    "hey", "hi", "hello", "wow", "oh", "well", "yeah", "yes", "no", "ok", "okay", "thanks",
    "thank", "please", "sorry",
];

fn clean_object(raw: &str) -> Option<String> {
    let mut words: Vec<&str> = raw.split_whitespace().collect();
    while let Some(first) = words.first() {
        if LEADING_STRIP.contains(&first.to_lowercase().as_str()) {
            words.remove(0);
        } else {
            break;
        }
    }
    while let Some(last) = words.last() {
        if TRAILING_STRIP.contains(&last.to_lowercase().as_str()) {
            words.pop();
        } else {
            break;
        }
    }
    if words.is_empty() {
        return None;
    }
    let object = words.join(" ");
    let norm = normalize_relation_token(&object);
    if norm.is_empty() || BAD_OBJECTS.contains(&norm.as_str()) {
        return None;
    }
    // Single generic nouns ("place", "spot") carry no signal.
    if words.len() == 1 && matches!(norm.as_str(), "place" | "spot" | "trip") {
        return None;
    }
    Some(object)
}

/// Find an explicit capitalized name just before the match; else the speaker.
/// Leading vocatives ("Hey Gina! ...") address someone and are skipped.
fn resolve_match_subject(text: &str, match_start: usize, speaker: &str) -> String {
    let before = &text[..match_start];
    // Last 1-3 words before the verb phrase.
    let tail: Vec<&str> = before.split_whitespace().rev().take(3).collect();
    for word in tail.iter().rev() {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
        let lower = clean.to_lowercase();
        if clean.len() >= 3
            && clean.chars().next().is_some_and(|c| c.is_uppercase())
            && !PRONOUNS.contains(&lower.as_str())
            && !NOT_NAMES.contains(&lower.as_str())
            && !is_vocative(before, clean)
        {
            return clean.to_string();
        }
    }
    speaker.to_string()
}

/// True if `name` appears as a leading vocative ("Hey Gina! ...", "John, ...").
fn is_vocative(before: &str, name: &str) -> bool {
    let trimmed = before.trim_start();
    // Vocative = within the first few words and followed by ! or , .
    let prefix: String = trimmed
        .split_whitespace()
        .take(4)
        .collect::<Vec<_>>()
        .join(" ");
    // Find the name in the prefix; check the char after it.
    if let Some(pos) = prefix.find(name) {
        let after = prefix[pos + name.len()..].chars().next();
        if matches!(after, Some('!') | Some(',')) {
            return true;
        }
    }
    // Also: "Hey <Name>!" greeting form (case-insensitive).
    let lower_prefix = prefix.to_lowercase();
    let lower_name = name.to_lowercase();
    for greet in ["hey", "hi", "hello"] {
        let gp = format!("{greet} {lower_name}");
        if let Some(pos) = lower_prefix.find(&gp) {
            let after = lower_prefix[pos + gp.len()..].chars().next();
            if matches!(after, Some('!') | Some(',')) {
                return true;
            }
        }
    }
    false
}

/// Extract (subject, predicate, object) triples from dialogue turns.
pub fn extract_relations(turns: &[RelationTurn]) -> Vec<EntityRelation> {
    let mut out = Vec::new();
    for turn in turns {
        let mut seen: HashSet<(String, &'static str, String)> = HashSet::new();
        for (re, predicate, confidence) in compiled_patterns() {
            for caps in re.captures_iter(&turn.text) {
                let raw_obj = match caps.get(1) {
                    Some(m) => m.as_str(),
                    None => continue,
                };
                let object = match clean_object(raw_obj) {
                    Some(o) => o,
                    None => continue,
                };
                let m0 = caps.get(0).expect("full match");
                let subject = resolve_match_subject(&turn.text, m0.start(), &turn.speaker);
                let key = (
                    normalize_relation_token(&subject),
                    *predicate,
                    normalize_relation_token(&object),
                );
                if !seen.insert(key.clone()) {
                    continue;
                }
                out.push(EntityRelation {
                    subject: subject.clone(),
                    subject_norm: key.0,
                    predicate,
                    object: object.clone(),
                    object_norm: key.2,
                    session_id: turn.session_id.clone(),
                    turn_idx: turn.turn_idx,
                    doc_id: turn.doc_id.clone(),
                    evidence: format!("{}: {}", turn.speaker, turn.text),
                    confidence: *confidence,
                });
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Relation index + shared-relation query.
// ---------------------------------------------------------------------------

/// One indexed hit for (subject, predicate).
#[derive(Debug, Clone)]
pub struct RelationHit {
    pub object: String,
    pub object_norm: String,
    pub session_id: String,
    pub turn_idx: usize,
    pub doc_id: String,
    pub evidence: String,
    pub confidence: f32,
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
    by_sp: HashMap<(String, &'static str), Vec<RelationHit>>,
    /// Display names of known persons (speakers first).
    pub persons: Vec<String>,
    person_norms: Vec<String>,
}

impl RelationIndex {
    /// Build from dialogue turns (extraction + indexing).
    pub fn build(turns: &[RelationTurn]) -> Self {
        let mut idx = RelationIndex::default();
        let mut seen_person: HashSet<String> = HashSet::new();
        for turn in turns {
            let norm = normalize_relation_token(&turn.speaker);
            if seen_person.insert(norm.clone()) {
                idx.persons.push(turn.speaker.clone());
                idx.person_norms.push(norm);
            }
        }
        for rel in extract_relations(turns) {
            // Fold named subjects into the person list.
            let snorm = rel.subject_norm.clone();
            if !idx.person_norms.contains(&snorm) {
                idx.person_norms.push(snorm.clone());
                idx.persons.push(rel.subject.clone());
            }
            idx.by_sp
                .entry((snorm, rel.predicate))
                .or_default()
                .push(RelationHit {
                    object: rel.object,
                    object_norm: rel.object_norm,
                    session_id: rel.session_id,
                    turn_idx: rel.turn_idx,
                    doc_id: rel.doc_id,
                    evidence: rel.evidence,
                    confidence: rel.confidence,
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

    /// Resolve display names to normalized person ids.
    ///
    /// Exact match, then fuzzy (edit distance <= 2), then elimination when
    /// the conversation has exactly two persons and one name is unresolved.
    /// Returns None if any name cannot be resolved.
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
        // Elimination: two persons, two names, exactly one unresolved.
        if self.person_norms.len() == 2 && names.len() == 2 {
            let unresolved: Vec<usize> = resolved
                .iter()
                .enumerate()
                .filter_map(|(i, r)| r.is_none().then_some(i))
                .collect();
            if unresolved.len() == 1 {
                let taken: HashSet<&str> = resolved.iter().filter_map(|r| r.as_deref()).collect();
                if let Some(other) = self
                    .person_norms
                    .iter()
                    .find(|p| !taken.contains(p.as_str()))
                {
                    resolved[unresolved[0]] = Some(other.clone());
                }
            }
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
            for pred in family.predicates() {
                if let Some(hits) = self.by_sp.get(&(snorm.clone(), *pred)) {
                    for hit in hits {
                        if expect_place && !looks_like_place(&hit.object) {
                            continue;
                        }
                        objects
                            .entry(hit.object_norm.clone())
                            .or_insert_with(|| (hit.object.clone(), Vec::new()))
                            .1
                            .push(hit.clone());
                    }
                }
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
}

/// Heuristic place check for the expect_place filter: proper-noun object.
fn looks_like_place(object: &str) -> bool {
    let first = object.split_whitespace().next().unwrap_or("");
    first.chars().next().is_some_and(|c| c.is_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(speaker: &str, text: &str, session: &str, idx: usize) -> RelationTurn {
        RelationTurn {
            speaker: speaker.to_string(),
            text: text.to_string(),
            session_id: session.to_string(),
            turn_idx: idx,
            doc_id: format!("{session}::turn{idx}"),
        }
    }

    #[test]
    fn extract_been_to_with_speaker_subject() {
        let turns = vec![turn(
            "Gina",
            "Paris?! That is really great Jon! Never had a chance to visit it. Been only to Rome once.",
            "conv-30::session_2",
            5,
        )];
        let rels = extract_relations(&turns);
        assert_eq!(rels.len(), 1, "rels: {rels:?}");
        let r = &rels[0];
        assert_eq!(r.subject, "Gina");
        assert_eq!(r.predicate, "visited");
        assert_eq!(r.object, "Rome");
        // "to visit it" must not produce a triple (pronoun object).
        assert!(!rels.iter().any(|x| x.object_norm == "it"));
    }

    #[test]
    fn extract_trip_to_light_verb() {
        let turns = vec![turn(
            "Jon",
            "Hey Gina! Took a short trip last week to Rome to clear my mind a little.",
            "conv-30::session_15",
            1,
        )];
        let rels = extract_relations(&turns);
        assert_eq!(rels.len(), 1, "rels: {rels:?}");
        assert_eq!(rels[0].subject, "Jon");
        assert_eq!(rels[0].predicate, "visited");
        assert_eq!(rels[0].object, "Rome");
    }

    #[test]
    fn extract_went_to_with_we_subject() {
        let turns = vec![turn(
            "John",
            "We went to a homeless shelter to give out food and supplies.",
            "conv-41::session_3",
            5,
        )];
        let rels = extract_relations(&turns);
        assert_eq!(rels.len(), 1, "rels: {rels:?}");
        assert_eq!(rels[0].subject, "John");
        assert_eq!(rels[0].predicate, "visited");
        assert_eq!(rels[0].object, "homeless shelter");
    }

    #[test]
    fn extract_volunteer_at_relative_clause() {
        let turns = vec![turn(
            "Maria",
            "I donated my old car to a homeless shelter I volunteer at yesterday.",
            "conv-41::session_2",
            1,
        )];
        let rels = extract_relations(&turns);
        assert_eq!(rels.len(), 1, "rels: {rels:?}");
        assert_eq!(rels[0].subject, "Maria");
        assert_eq!(rels[0].predicate, "volunteered_at");
        assert_eq!(rels[0].object, "homeless shelter");
    }

    #[test]
    fn no_triple_without_verb_phrase() {
        let turns = vec![turn("Jon", "Paris?! That is really great!", "s", 0)];
        assert!(extract_relations(&turns).is_empty());
    }

    #[test]
    fn resolve_subjects_exact_fuzzy_elimination() {
        let turns = vec![
            turn("Jon", "Took a trip to Rome.", "s1", 0),
            turn("Gina", "Been to Rome.", "s2", 0),
        ];
        let idx = RelationIndex::build(&turns);
        // Exact.
        assert_eq!(
            idx.resolve_subjects(&["Jon".to_string(), "Gina".to_string()]),
            Some(vec!["jon".to_string(), "gina".to_string()])
        );
        // Fuzzy: John -> jon (distance 1).
        assert_eq!(
            idx.resolve_subjects(&["John".to_string(), "Gina".to_string()]),
            Some(vec!["jon".to_string(), "gina".to_string()])
        );
        // Elimination: Jean unresolved, two persons, John -> jon, so Jean -> gina.
        assert_eq!(
            idx.resolve_subjects(&["Jean".to_string(), "John".to_string()]),
            Some(vec!["gina".to_string(), "jon".to_string()])
        );
    }

    #[test]
    fn query_shared_intersects_across_predicates() {
        let turns = vec![
            turn("Gina", "Been only to Rome once.", "conv-30::session_2", 5),
            turn(
                "Jon",
                "Took a short trip last week to Rome.",
                "conv-30::session_15",
                1,
            ),
            turn("Gina", "I love Paris.", "conv-30::session_3", 0),
        ];
        let idx = RelationIndex::build(&turns);
        let shared = idx
            .query_shared(
                &["Jean".to_string(), "John".to_string()],
                PredicateFamily::PlacePresence,
                true,
            )
            .expect("resolves via fuzzy+elimination");
        assert_eq!(shared.len(), 1, "shared: {shared:?}");
        assert_eq!(shared[0].object, "Rome");
        assert_eq!(shared[0].evidence.len(), 2);
        // Sessions are the gold ones.
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
        let turns = vec![
            turn(
                "John",
                "We went to a homeless shelter to give out food.",
                "conv-41::session_3",
                5,
            ),
            turn(
                "Maria",
                "I donated my car to a homeless shelter I volunteer at.",
                "conv-41::session_2",
                1,
            ),
        ];
        let idx = RelationIndex::build(&turns);
        let shared = idx
            .query_shared(
                &["John".to_string(), "Maria".to_string()],
                PredicateFamily::PlacePresence,
                false,
            )
            .expect("resolves");
        assert_eq!(shared.len(), 1, "shared: {shared:?}");
        assert_eq!(shared[0].object, "homeless shelter");
    }

    #[test]
    fn query_shared_unresolvable_declines() {
        let turns = vec![turn("Jon", "Took a trip to Rome.", "s1", 0)];
        let idx = RelationIndex::build(&turns);
        // Only one person known; "Zelda" cannot resolve and elimination
        // needs exactly two persons.
        assert!(idx
            .query_shared(
                &["Zelda".to_string(), "Jon".to_string()],
                PredicateFamily::PlacePresence,
                true
            )
            .is_none());
    }
}
