//! Intent-revealing question decomposition + per-operand retrieval.
//!
//! The old entity-join grabbed every capitalized span as a flat anchor list and
//! demanded single-document co-occurrence of all anchors. That never matched
//! the benchmark's multi-entity questions, which are aggregation / comparison
//! *across* sessions ("days in Hawaii and New York City", "days between the
//! MoMA visit and the Met exhibit", "which device first, A or B").
//!
//! This module instead *reveals the real question*: it parses each question
//! into an [`IntentOperation`] plus [`OperandScope`]s (what role each entity
//! plays), then retrieves per operand and unions the evidence with raw,
//! globally-comparable BM25 scores. No per-anchor normalization, no
//! doc-level intersection, no candidate discovery: for these operations the
//! answer is computed from per-operand evidence, not found as a third entity.
//!
//! Pure reveal ([`reveal_question`]) is fully unit-testable without an index.
//! [`SegmentedMemoryIndex::query_with_intent`] validates operand entities
//! against the corpus entity index and returns `None` (caller falls back to
//! the adaptive path) unless the operation is multi-operand and at least two
//! operands carry corpus-validated entities.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::index::{SearchResult, TemporalQueryContext};
use crate::tokenizer::{is_stopword, tokenize, TokenizerMode};

use super::catalog::query_connection_profile;
use super::diagnostics::{SegmentQueryDiagnostics, SegmentQueryOutput};
use super::model::MemoryIndexSegment;
use super::relations::PredicateFamily;
use super::routing::SegmentRoutingStrategy;
use super::segmented::SegmentedMemoryIndex;

/// What the question is actually asking for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum IntentOperation {
    /// Default: single-fact lookup. Never fires the intent path.
    Lookup,
    /// Sum/aggregate a value across operands ("in total", "how many ... and").
    Aggregate,
    /// Difference between two operand timestamps ("how many days between X and Y").
    TemporalDiff,
    /// Order operands/options by time ("which first", "most recent").
    TemporalOrder,
    /// Choose among explicit options ("A or B").
    Choose,
    /// Filter a class of events by a pivot event ("events before X").
    TemporalFilter,
    /// "both X and Y <verb>": intersect per-subject relation objects.
    SharedRelation,
}

impl IntentOperation {
    fn as_str(self) -> &'static str {
        match self {
            IntentOperation::Lookup => "lookup",
            IntentOperation::Aggregate => "aggregate",
            IntentOperation::TemporalDiff => "temporal-diff",
            IntentOperation::TemporalOrder => "temporal-order",
            IntentOperation::Choose => "choose",
            IntentOperation::TemporalFilter => "temporal-filter",
            IntentOperation::SharedRelation => "shared-relation",
        }
    }

    /// True for operations whose retrieval needs per-operand fan-out + union,
    /// or relation intersection. Lookup never fires a special path.
    fn is_multi_operand(self) -> bool {
        !matches!(self, IntentOperation::Lookup)
    }
}

/// One operand scope: a slice of the question playing one role.
#[derive(Debug, Clone)]
pub struct OperandScope {
    /// Raw scope text, e.g. "in Hawaii" or "the Samsung Galaxy S22".
    pub text: String,
    /// Entity mentions extracted *within* this scope (pre-validation).
    pub entities: Vec<ScopeEntity>,
    /// For TemporalFilter: keyword text used when the scope has no entities.
    pub keywords: Vec<String>,
}

/// One entity mention inside an operand scope.
#[derive(Debug, Clone, Serialize)]
pub struct ScopeEntity {
    /// Normalized (stemmed, space-joined) mention used for corpus matching.
    pub mention: String,
    /// Original-case text used in the sub-query.
    pub display: String,
}

/// A "both X and Y <verb>" question: intersect relation objects across subjects.
#[derive(Debug, Clone, Serialize)]
pub struct SharedRelationQuery {
    /// Subject display names as written in the question, e.g. ["Jean", "John"].
    pub subjects: Vec<String>,
    /// Predicate family to look up per subject.
    pub family: PredicateFamily,
    /// True for "which city/where": prefer proper-noun (place-like) objects.
    pub expect_place: bool,
}

/// The revealed question: operation + operand roles + answer shape.
#[derive(Debug, Clone)]
pub struct RevealedQuestion {
    pub operation: IntentOperation,
    pub operands: Vec<OperandScope>,
    /// Standalone time window constraining the question, if any.
    pub time_constraint: Option<String>,
    /// What answer-time composition would compute (not executed here).
    pub answer_shape: String,
    /// Set when the intent path should not run (caller falls back).
    pub fallback_reason: Option<String>,
    /// Set for SharedRelation: the relation query to run.
    pub shared_relation: Option<SharedRelationQuery>,
}

impl RevealedQuestion {
    /// Human-readable rendering of the revealed question, e.g.
    /// `aggregate(days | Hawaii, New York City)`.
    pub fn rendered(&self) -> String {
        let ops: Vec<String> = self
            .operands
            .iter()
            .map(|o| {
                if o.entities.is_empty() {
                    o.keywords.join(" ")
                } else {
                    o.entities
                        .iter()
                        .map(|e| e.display.clone())
                        .collect::<Vec<_>>()
                        .join(" + ")
                }
            })
            .collect();
        format!("{}({})", self.operation.as_str(), ops.join(" | "))
    }
}

/// Diagnostics for one intent query.
#[derive(Debug, Clone, Default, Serialize)]
pub struct IntentDiagnostics {
    pub operation: String,
    pub revealed: String,
    pub operands: Vec<String>,
    pub time_constraint: Option<String>,
    pub validated_operand_count: usize,
    pub fallback: Option<String>,
}

// ---------------------------------------------------------------------------
// Pure text helpers (no index).
// ---------------------------------------------------------------------------

/// Parenthesized acronyms like "(MoMA)": 2-6 uppercase letters in parens.
fn paren_acronyms(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '(' {
            let mut j = i + 1;
            while j < chars.len() && chars[j] != ')' && j - i < 10 {
                j += 1;
            }
            if j < chars.len() && chars[j] == ')' {
                let inner: String = chars[i + 1..j].iter().collect();
                if (2..=6).contains(&inner.len()) && inner.chars().all(|c| c.is_ascii_uppercase()) {
                    out.push(inner);
                }
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }
    out.sort();
    out.dedup();
    out
}

fn is_connector(lower: &str) -> bool {
    matches!(lower, "of" | "the" | "a" | "an" | "de" | "for")
}

fn is_cap_word(w: &str) -> bool {
    w.chars().next().is_some_and(|c| c.is_ascii_uppercase())
        && w.chars().any(|c| c.is_ascii_lowercase())
}

/// Capitalized spans allowing lowercase connectors inside names
/// ("Museum of Modern Art", "Metropolitan Museum of Art"), up to 5 words.
fn name_spans(text: &str) -> Vec<String> {
    let words: Vec<&str> = text
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < words.len() {
        if !is_cap_word(words[i]) {
            i += 1;
            continue;
        }
        let mut span = vec![words[i]];
        let mut j = i + 1;
        while j < words.len() && span.len() < 5 {
            let nw = words[j];
            if is_cap_word(nw) {
                span.push(nw);
                j += 1;
            } else if is_connector(&nw.to_ascii_lowercase())
                && j + 1 < words.len()
                && is_cap_word(words[j + 1])
            {
                span.push(nw);
                span.push(words[j + 1]);
                j += 2;
            } else {
                break;
            }
        }
        spans.push(span.join(" "));
        i = j;
    }
    spans.sort();
    spans.dedup();
    spans
}

/// Normalize a mention the way the catalog does: stemmed tokens, space-joined.
fn normalize_mention(text: &str) -> String {
    tokenize(text, TokenizerMode::Stemmed).join(" ")
}

fn is_month_token(lower: &str) -> bool {
    matches!(
        lower,
        "january"
            | "february"
            | "march"
            | "april"
            | "may"
            | "june"
            | "july"
            | "august"
            | "september"
            | "october"
            | "november"
            | "december"
            | "jan"
            | "feb"
            | "mar"
            | "apr"
            | "jun"
            | "jul"
            | "aug"
            | "sep"
            | "sept"
            | "oct"
            | "nov"
            | "dec"
    )
}

/// Case-insensitive byte search for a literal needle.
fn find_ci(hay: &str, needle: &str) -> Option<usize> {
    let h = hay.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() || n.len() > h.len() {
        return None;
    }
    (0..=h.len() - n.len()).find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
}

fn rfind_ci(hay: &str, needle: &str) -> Option<usize> {
    let h = hay.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() || n.len() > h.len() {
        return None;
    }
    (0..=h.len() - n.len())
        .rev()
        .find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric()
}

/// Case-insensitive whole-word search.
fn find_word_ci(hay: &str, word: &str) -> Option<usize> {
    let mut start = 0;
    while let Some(pos) = find_ci(&hay[start..], word) {
        let abs = start + pos;
        let before_ok = abs == 0 || !hay[..abs].chars().last().is_some_and(is_word_char);
        let after = abs + word.len();
        let after_ok = after >= hay.len() || !hay[after..].chars().next().is_some_and(is_word_char);
        if before_ok && after_ok {
            return Some(abs);
        }
        start = abs + 1;
    }
    None
}

/// Split on the last occurrence of " and " (case-insensitive).
fn rsplit_and(text: &str) -> Option<(String, String)> {
    let pos = rfind_ci(text, " and ")?;
    Some((
        text[..pos].trim().to_string(),
        text[pos + 5..].trim().to_string(),
    ))
}

/// Quoted spans ('...' or "...") as atomic units, in encounter order.
fn quoted_spans(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\'' || c == '"' {
            let mut j = i + 1;
            while j < chars.len() && chars[j] != c {
                j += 1;
            }
            if j < chars.len() && j > i + 1 {
                let inner: String = chars[i + 1..j].iter().collect();
                let inner = inner.trim().to_string();
                if !inner.is_empty() {
                    out.push(inner);
                }
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }
    out.sort();
    out.dedup();
    out
}

/// Extract entities within one operand scope, in priority order:
/// parenthesized acronyms ("(MoMA)"), quoted spans (atomic), then
/// capitalized name spans not covered by the first two and not pure time.
fn scope_entities(scope: &str) -> Vec<ScopeEntity> {
    let mut entities = Vec::new();
    let mut seen = HashSet::new();
    let push = |display: String, entities: &mut Vec<ScopeEntity>, seen: &mut HashSet<String>| {
        let mention = normalize_mention(&display);
        if mention.is_empty() || !seen.insert(mention.clone()) {
            return;
        }
        let tokens: Vec<&str> = mention.split_whitespace().collect();
        if tokens
            .iter()
            .all(|t| is_stopword(t, TokenizerMode::Stemmed))
        {
            return;
        }
        entities.push(ScopeEntity { mention, display });
    };
    for acro in paren_acronyms(scope) {
        push(acro, &mut entities, &mut seen);
    }
    // Strip the acronyms so they don't merge into name spans.
    let mut text = scope.to_string();
    for acro in paren_acronyms(scope) {
        text = text.replace(&format!("({acro})"), " ");
    }
    for q in quoted_spans(&text) {
        push(q, &mut entities, &mut seen);
    }
    let quoted_displays: Vec<String> = entities.iter().map(|e| e.display.clone()).collect();
    for span in name_spans(&text) {
        // Covered by a quoted span or acronym already extracted.
        if quoted_displays
            .iter()
            .any(|qd| qd.to_lowercase().contains(&span.to_lowercase()))
        {
            continue;
        }
        let mention = normalize_mention(&span);
        // Pure time spans ("August", "2023") belong to the time constraint.
        if !mention.is_empty()
            && mention
                .split_whitespace()
                .all(|t| is_month_token(t) || t.chars().any(|c| c.is_ascii_digit()))
        {
            continue;
        }
        push(span, &mut entities, &mut seen);
    }
    entities
}

// ---------------------------------------------------------------------------
// Scope splitting: parse the question into operand roles.
// ---------------------------------------------------------------------------

enum SplitKind {
    /// "between X and Y": exactly two operand scopes.
    Between,
    /// "A or B": option scopes.
    Options,
    /// "X and Y [and Z]": operand scopes.
    And,
    /// Temporal filter: direction word; scopes are [class, pivot].
    Filter { direction: &'static str },
    /// No structure found: the whole question is one scope.
    Single,
}

/// Split a question into operand scope texts plus the kind of split.
fn split_scopes(question: &str) -> (SplitKind, Vec<String>) {
    let q = question.trim();

    // Temporal filter first: "how many ... before|after|since ... 'Pivot'".
    // The pivot is quoted (or a trailing capitalized span); the class side
    // keeps keyword text for its sub-query.
    let q_lower = q.to_lowercase();
    let how_many = q_lower.starts_with("how many")
        || q_lower.starts_with("how much")
        || q_lower.starts_with("how long")
        || q_lower.contains("how many");
    if how_many {
        for (word, dir) in [("before", "before"), ("after", "after"), ("since", "since")] {
            if let Some(pos) = find_word_ci(q, word) {
                let quotes = quoted_spans(q);
                let pivot_inner = quotes
                    .first()
                    .cloned()
                    .or_else(|| name_spans(&q[pos + word.len()..]).first().cloned());
                if let Some(inner) = pivot_inner {
                    // Re-attach quotes so scope_entities treats the pivot
                    // atomically instead of splitting it into name spans.
                    let pivot = format!("'{inner}'");
                    // Class scope: question with the pivot span removed.
                    // Leftover quote marks are harmless to keyword extraction.
                    let class_text = q.replacen(&inner, "", 1);
                    return (
                        SplitKind::Filter { direction: dir },
                        vec![class_text.trim().to_string(), pivot],
                    );
                }
            }
        }
    }

    // "between X and Y": split on the LAST " and " so names containing
    // "and" inside X stay intact.
    if let Some(bpos) = find_word_ci(q, "between") {
        let rest = q[bpos + "between".len()..].trim_start().to_string();
        if let Some((a, b)) = rsplit_and(&rest) {
            if !a.is_empty() && !b.is_empty() {
                return (SplitKind::Between, vec![a, b]);
            }
        }
    }

    // "A or B" options: split on the last " or ", left side trimmed to the
    // last comma (options are usually enumerated after one).
    if let Some(opos) = rfind_ci(q, " or ") {
        let left_full = q[..opos].trim();
        let right = q[opos + 4..].trim().trim_end_matches('?').to_string();
        let left = match left_full.rfind(',') {
            Some(cpos) => left_full[cpos + 1..].trim().to_string(),
            None => left_full.to_string(),
        };
        if !left.is_empty() && !right.is_empty() {
            let left_has = !scope_entities(&left).is_empty() || !quoted_spans(&left).is_empty();
            let right_has = !scope_entities(&right).is_empty() || !quoted_spans(&right).is_empty();
            if left_has || right_has {
                return (SplitKind::Options, vec![left, right]);
            }
        }
    }

    // "X and Y": split on every " and ".
    if find_ci(q, " and ").is_some() {
        let mut parts: Vec<String> = Vec::new();
        let mut rest = q.to_string();
        loop {
            match rsplit_and(&rest) {
                Some((a, b)) => {
                    parts.push(b);
                    rest = a;
                }
                None => break,
            }
            if find_ci(&rest, " and ").is_none() {
                break;
            }
        }
        parts.push(rest);
        parts.reverse();
        let parts: Vec<String> = parts
            .into_iter()
            .map(|p| p.trim().trim_end_matches('?').to_string())
            .filter(|p| !p.is_empty())
            .collect();
        if parts.len() >= 2 {
            return (SplitKind::And, parts);
        }
    }

    (SplitKind::Single, vec![q.to_string()])
}

/// Time tokens in scope text: month names, or digit tokens that are not part
/// of a product/code token and not adjacent to a capitalized non-month word.
///
/// Rationale: "S22" (letters+digits) is a code, never a time. A pure-digit
/// token next to a capitalized word ("Dell XPS 13") belongs to a name;
/// next to a month ("August 11") or a lowercase word ("10 days") it is a date.
fn scope_time_tokens(scope: &str) -> Vec<String> {
    let words: Vec<&str> = scope
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let mut out = Vec::new();
    for (i, w) in words.iter().enumerate() {
        if w.len() <= 1 {
            continue;
        }
        let lower = w.to_ascii_lowercase();
        if is_month_token(&lower) {
            out.push(lower);
            continue;
        }
        if !w.chars().any(|c| c.is_ascii_digit()) {
            continue;
        }
        // Letter+digit mixes ("S22") are codes, never times.
        if w.chars().any(|c| c.is_ascii_alphabetic()) {
            continue;
        }
        // Pure digits: drop when glued to a capitalized non-month name.
        let neighbor_capitalized_non_month = [i.checked_sub(1), i.checked_add(1)]
            .into_iter()
            .flatten()
            .filter_map(|j| words.get(j))
            .any(|nw| {
                nw.chars().next().is_some_and(|c| c.is_ascii_uppercase())
                    && !is_month_token(&nw.to_ascii_lowercase())
            });
        if !neighbor_capitalized_non_month {
            out.push(lower);
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Standalone time constraint for the whole question: time tokens not covered
/// by any operand entity mention (digits inside product names were already
/// dropped by [`scope_time_tokens`]).
fn time_constraint(question: &str, operands: &[OperandScope]) -> Option<String> {
    let covered: HashSet<&str> = operands
        .iter()
        .flat_map(|o| o.entities.iter())
        .flat_map(|e| e.mention.split_whitespace())
        .collect();
    let mut tokens: Vec<String> = scope_time_tokens(question)
        .into_iter()
        .filter(|t| !covered.contains(t.as_str()))
        .collect();
    tokens.sort();
    tokens.dedup();
    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" "))
    }
}

// ---------------------------------------------------------------------------
// The revealer: question -> operation + operand roles.
// ---------------------------------------------------------------------------

fn has_any(hay: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| hay.contains(n))
}

/// Answer-type signal for single-scope "how much / how many / how long"
/// questions. These stay Lookup (no fan-out — adaptive covers retrieval),
/// but the expected value type is preserved in `answer_shape` instead of
/// being dropped, so downstream ranking can prefer matching evidence
/// (money amounts, counts, durations).
fn single_scope_answer_type(q_lower: &str) -> Option<&'static str> {
    if q_lower.starts_with("how much") {
        // "how much" usually asks for money, but "how much time/longer"
        // asks for a duration and "how much weight" for a quantity.
        if has_any(q_lower, &["how much time", "how much longer"]) {
            Some("single duration")
        } else if q_lower.contains("how much weight") {
            Some("single quantity")
        } else {
            Some("single money amount")
        }
    } else if q_lower.starts_with("how many") || q_lower.contains("how many") {
        Some("single count")
    } else if q_lower.starts_with("how long") {
        Some("single duration")
    } else {
        None
    }
}

/// Reveal the real question: operation, operand roles, answer shape.
/// Pure function of the question text; no index access.
/// Detect "both X and Y <verb>" / "X and Y both <verb>" shared-relation
/// questions. Returns the subject names (original case) and predicate family.
fn detect_shared_relation(question: &str) -> Option<SharedRelationQuery> {
    let q_lower = question.to_lowercase();
    if !q_lower.contains("both") {
        return None;
    }
    // Find the two subject spans around "both ... and ..." or "... and ... both".
    let subjects: Vec<String> = {
        let words: Vec<&str> = question.split_whitespace().collect();
        let lower: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();
        let both_pos = lower
            .iter()
            .position(|w| w.trim_matches(|c: char| !c.is_alphabetic()) == "both")?;
        let and_pos = lower
            .iter()
            .position(|w| w.trim_matches(|c: char| !c.is_alphabetic()) == "and")?;
        let clean = |w: &str| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string();
        let is_name = |w: &str| {
            let c = clean(w);
            c.len() >= 3 && c.chars().next().is_some_and(|ch| ch.is_uppercase())
        };
        if and_pos == both_pos + 2 {
            // "both X and Y": X right after both, Y right after and.
            let a = words.get(both_pos + 1)?;
            let b = words.get(and_pos + 1)?;
            if !(is_name(a) && is_name(b)) {
                return None;
            }
            vec![clean(a), clean(b)]
        } else if both_pos == and_pos + 2 {
            // "X and Y both": X before and, Y between and and both.
            let a = words.get(and_pos.checked_sub(1)?)?;
            let b = words.get(and_pos + 1)?;
            if !(is_name(a) && is_name(b)) {
                return None;
            }
            vec![clean(a), clean(b)]
        } else {
            return None;
        }
    };
    // Predicate family from the verb / activity noun.
    let family = if q_lower.contains("volunteering") {
        PredicateFamily::PlacePresence
    } else if q_lower.contains("visit") {
        PredicateFamily::PlacePresence
    } else {
        return None;
    };
    let expect_place = q_lower.contains("which city")
        || q_lower.contains("what city")
        || q_lower.starts_with("where ");
    Some(SharedRelationQuery {
        subjects,
        family,
        expect_place,
    })
}

pub fn reveal_question(question: &str) -> RevealedQuestion {
    let q = question.trim();
    let q_lower = q.to_lowercase();
    // Shared-relation ("both X and Y") takes precedence over scope splitting.
    if let Some(srq) = detect_shared_relation(q) {
        let answer_shape = format!(
            "intersection of {} objects across {}",
            srq.family.as_str(),
            srq.subjects.join(", ")
        );
        return RevealedQuestion {
            operation: IntentOperation::SharedRelation,
            operands: Vec::new(),
            time_constraint: None,
            answer_shape,
            fallback_reason: None,
            shared_relation: Some(srq),
        };
    }
    let how_many = q_lower.starts_with("how many")
        || q_lower.starts_with("how much")
        || q_lower.starts_with("how long")
        || q_lower.contains("how many");
    let time_unit = has_any(&q_lower, &["day", "week", "month", "year", "hour"]);
    let order_markers = has_any(
        &q_lower,
        &[
            "first",
            "last",
            "earliest",
            "latest",
            "most recent",
            "oldest",
            "newest",
        ],
    );
    let agg_markers = has_any(&q_lower, &["in total", "total", "combined", "altogether"]);

    let (kind, scope_texts) = split_scopes(q);
    let operands: Vec<OperandScope> = scope_texts
        .iter()
        .map(|text| {
            let entities = scope_entities(text);
            // Keywords for scopes with no entities: content words for the
            // sub-query (used by TemporalFilter's class side).
            let keywords = if entities.is_empty() {
                text.split(|ch: char| !ch.is_alphanumeric())
                    .filter(|w| w.len() > 2)
                    .map(|w| normalize_mention(w))
                    .filter(|w| !w.is_empty() && !is_stopword(w, TokenizerMode::Stemmed))
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            OperandScope {
                text: text.clone(),
                entities,
                keywords,
            }
        })
        .collect();

    let (operation, answer_shape, fallback_reason) = match &kind {
        SplitKind::Between if operands.len() == 2 && how_many && time_unit => (
            IntentOperation::TemporalDiff,
            "difference of per-operand timestamps".to_string(),
            None,
        ),
        SplitKind::Between => (
            IntentOperation::Lookup,
            String::new(),
            Some("between without a measurable time-unit question".to_string()),
        ),
        SplitKind::Options if order_markers => (
            IntentOperation::TemporalOrder,
            "argmin/argmax over option timestamps".to_string(),
            None,
        ),
        SplitKind::Options => (
            IntentOperation::Choose,
            "select among options by evidence".to_string(),
            None,
        ),
        SplitKind::And if operands.len() >= 2 && (agg_markers || how_many) => (
            IntentOperation::Aggregate,
            "sum/aggregate of per-operand values".to_string(),
            None,
        ),
        SplitKind::And => (
            IntentOperation::Lookup,
            String::new(),
            Some("and-split without aggregate markers".to_string()),
        ),
        SplitKind::Filter { direction, .. } => (
            IntentOperation::TemporalFilter,
            format!("filter class events {direction} the pivot event"),
            None,
        ),
        SplitKind::Single => (
            IntentOperation::Lookup,
            single_scope_answer_type(&q_lower)
                .unwrap_or_default()
                .to_string(),
            Some("single-scope lookup: adaptive covers it".to_string()),
        ),
    };

    // TemporalFilter operand order: [class, pivot]; keep as-is.
    let time = time_constraint(q, &operands);

    RevealedQuestion {
        operation,
        operands,
        time_constraint: time,
        answer_shape,
        fallback_reason,
        shared_relation: None,
    }
}

// ---------------------------------------------------------------------------
// Corpus validation + per-operand retrieval.
// ---------------------------------------------------------------------------

/// Normalized entity mentions present anywhere in the corpus, from the
/// documents' own `key_entities`. An operand entity only counts when the
/// corpus actually extracted it — this rejects junk spans the way the old
/// join did, but per operand scope.
fn corpus_entity_mentions(index: &SegmentedMemoryIndex) -> HashSet<String> {
    let mut out = HashSet::new();
    for seg in &index.segments {
        for doc in seg.index.docs.values() {
            for entity in &doc.key_entities {
                let norm = normalize_mention(&entity.text);
                if !norm.is_empty() {
                    out.insert(norm);
                }
            }
        }
    }
    out
}

/// Fan out one operand: sub-query from its validated entity displays plus
/// shared context terms, routed through the top segments. Returns
/// (doc_id, raw BM25 score, SearchResult) with NO per-operand normalization:
/// BM25 uses global corpus statistics, so raw scores are directly comparable
/// across operands. (Per-anchor max-normalization was the old join's
/// ranking bug: every operand's best hit scored 1.0 regardless of strength.)
fn fan_out_operand(
    index: &SegmentedMemoryIndex,
    operand: &OperandScope,
    validated: &[ScopeEntity],
    context_terms: &[String],
    strategy: SegmentRoutingStrategy,
    temporal: TemporalQueryContext<'_>,
    per_entity_segment_limit: usize,
    per_entity_top_k: usize,
) -> Vec<(String, f32, SearchResult)> {
    let mut subquery = validated
        .iter()
        .map(|e| e.display.clone())
        .collect::<Vec<_>>()
        .join(" ");
    // TemporalFilter's class side may have no entities: use keyword text.
    if subquery.trim().is_empty() && !operand.keywords.is_empty() {
        subquery = operand.keywords.join(" ");
    }
    for term in context_terms {
        if !validated
            .iter()
            .any(|e| e.mention.split_whitespace().any(|tok| tok == term.as_str()))
            && !subquery.split_whitespace().any(|tok| tok == term)
        {
            subquery.push(' ');
            subquery.push_str(term);
        }
    }
    let routes: Vec<_> =
        index.route_with_temporal_context_and_strategy(&subquery, strategy, temporal);
    let segment_by_id: HashMap<&str, &MemoryIndexSegment> = index
        .segments
        .iter()
        .map(|seg| (seg.segment_id.as_str(), seg))
        .collect();

    let mut hits = Vec::new();
    for route in routes.iter().take(per_entity_segment_limit) {
        let Some(seg) = segment_by_id.get(route.segment_id.as_str()) else {
            continue;
        };
        let (results, _, _) =
            seg.index
                .query_with_temporal_context(&subquery, per_entity_top_k, temporal);
        for r in results {
            hits.push((r.doc_id.clone(), r.score, r));
        }
    }
    // Deterministic order: score desc, then doc_id. Dedup keeps best score.
    hits.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let mut deduped: Vec<(String, f32, SearchResult)> = Vec::new();
    for (doc_id, score, r) in hits {
        if deduped.last().is_some_and(|(d, _, _)| d == &doc_id) {
            continue;
        }
        deduped.push((doc_id, score, r));
    }
    deduped
}

impl SegmentedMemoryIndex {
    /// Intent-revealing multi-operand retrieval. Returns `None` when the
    /// question does not reveal a multi-operand operation, or when fewer
    /// than two operands carry corpus-validated entities (caller should use
    /// the standard retrieval path).
    pub fn query_with_intent(
        &self,
        query: &str,
        top_k: usize,
        per_entity_segment_limit: usize,
        per_entity_top_k: usize,
        strategy: SegmentRoutingStrategy,
        temporal: TemporalQueryContext<'_>,
    ) -> Option<(SegmentQueryOutput, IntentDiagnostics)> {
        let revealed = reveal_question(query);
        if revealed.fallback_reason.is_some() || !revealed.operation.is_multi_operand() {
            return None;
        }

        // Validate operand entities against the corpus entity index.
        let corpus_entities = corpus_entity_mentions(self);
        let mut validated_operands: Vec<(OperandScope, Vec<ScopeEntity>)> = Vec::new();
        for operand in &revealed.operands {
            let validated: Vec<ScopeEntity> = operand
                .entities
                .iter()
                .filter(|e| corpus_entities.contains(&e.mention))
                .cloned()
                .collect();
            // TemporalFilter's class side is keyword-driven, not entity-driven.
            let usable = !validated.is_empty()
                || (revealed.operation == IntentOperation::TemporalFilter
                    && !operand.keywords.is_empty());
            if usable {
                validated_operands.push((operand.clone(), validated));
            }
        }
        if validated_operands.len() < 2 {
            return None;
        }

        // Context terms shared across sub-queries.
        let profile = query_connection_profile(query);
        let mut context_terms: Vec<String> = profile
            .subjects
            .iter()
            .chain(profile.objects.iter())
            .chain(profile.actions.iter())
            .cloned()
            .collect();
        context_terms.sort();
        context_terms.dedup();

        // Per-operand fan-out; union by raw score (max wins per doc).
        let mut best: HashMap<String, (f32, SearchResult)> = HashMap::new();
        for (operand, validated) in &validated_operands {
            for (doc_id, score, r) in fan_out_operand(
                self,
                operand,
                validated,
                &context_terms,
                strategy,
                temporal,
                per_entity_segment_limit,
                per_entity_top_k,
            ) {
                best.entry(doc_id)
                    .and_modify(|(s, _)| {
                        if score > *s {
                            *s = score;
                        }
                    })
                    .or_insert((score, r));
            }
        }
        if best.is_empty() {
            return None;
        }
        let mut ranked: Vec<(String, f32, SearchResult)> = best
            .into_iter()
            .map(|(doc_id, (score, r))| (doc_id, score, r))
            .collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        ranked.truncate(top_k);
        let results: Vec<SearchResult> = ranked
            .into_iter()
            .map(|(_, score, mut r)| {
                r.score = score;
                r
            })
            .collect();

        let diagnostics = IntentDiagnostics {
            operation: revealed.operation.as_str().to_string(),
            revealed: revealed.rendered(),
            operands: revealed.operands.iter().map(|o| o.text.clone()).collect(),
            time_constraint: revealed.time_constraint.clone(),
            validated_operand_count: validated_operands.len(),
            fallback: None,
        };
        let output = SegmentQueryOutput {
            results,
            diagnostics: SegmentQueryDiagnostics {
                snapshot_generation: self.generation,
                ..Default::default()
            },
        };
        Some((output, diagnostics))
    }
}

// ---------------------------------------------------------------------------
// Tests: hand-labeled fixtures from LongMemEval questions.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{DocRecord, Provenance};
    use crate::segments::model::build_memory_index_segment;
    use crate::tier1::Tier1Entity;

    fn mentions(question: &str, scope_idx: usize) -> Vec<String> {
        reveal_question(question).operands[scope_idx]
            .entities
            .iter()
            .map(|e| e.mention.clone())
            .collect()
    }

    #[test]
    fn reveal_aggregate_hawaii_nyc() {
        let q = "How many days did I spend in total traveling in Hawaii and in New York City?";
        let r = reveal_question(q);
        assert_eq!(r.operation, IntentOperation::Aggregate);
        assert!(r.fallback_reason.is_none(), "r: {r:?}");
        assert_eq!(r.operands.len(), 2);
        let m0 = mentions(q, 0);
        let m1 = mentions(q, 1);
        assert!(m0.iter().any(|m| m == "hawaii"), "m0: {m0:?}");
        assert!(m1.iter().any(|m| m.contains("new york")), "m1: {m1:?}");
        assert_eq!(r.time_constraint, None);
        assert_eq!(r.answer_shape, "sum/aggregate of per-operand values");
    }

    #[test]
    fn reveal_temporal_diff_moma_met() {
        let q = "How many days passed between my visit to the Museum of Modern Art (MoMA) and the 'Ancient Civilizations' exhibit at the Metropolitan Museum of Art?";
        let r = reveal_question(q);
        assert_eq!(r.operation, IntentOperation::TemporalDiff);
        assert!(r.fallback_reason.is_none(), "r: {r:?}");
        assert_eq!(r.operands.len(), 2);
        let m0 = mentions(q, 0);
        let m1 = mentions(q, 1);
        assert!(
            m0.iter().any(|m| m.contains("museum of modern art")) || m0.iter().any(|m| m == "moma"),
            "m0: {m0:?}"
        );
        assert!(m1.iter().any(|m| m.contains("ancient civil")), "m1: {m1:?}");
        assert!(
            m1.iter().any(|m| m.contains("metropolitan museum of art")),
            "m1: {m1:?}"
        );
        // No junk "Art"-only anchor and no mangled "Modern Art MoMA".
        assert!(!m0.iter().any(|m| m == "modern art moma"), "m0: {m0:?}");
    }

    #[test]
    fn reveal_temporal_order_device_choice_drops_code_junk() {
        let q = "Which device did I got first, the Samsung Galaxy S22 or the Dell XPS 13?";
        let r = reveal_question(q);
        assert_eq!(r.operation, IntentOperation::TemporalOrder);
        assert!(r.fallback_reason.is_none(), "r: {r:?}");
        assert_eq!(r.operands.len(), 2);
        let m0 = mentions(q, 0);
        let m1 = mentions(q, 1);
        assert!(m0.iter().any(|m| m.contains("samsung")), "m0: {m0:?}");
        assert!(m1.iter().any(|m| m.contains("del")), "m1: {m1:?}");
        // "S22"/"13" are product codes, not time tokens.
        assert_eq!(r.time_constraint, None, "r: {r:?}");
    }

    #[test]
    fn reveal_temporal_order_keeps_real_time_window() {
        let q = "Which vehicle did I take care of first in February, the bike or the car?";
        let r = reveal_question(q);
        // Options carry no entities: lookup fallback (adaptive covers it).
        assert_eq!(r.operation, IntentOperation::Lookup);
        assert!(r.fallback_reason.is_some());
        assert_eq!(r.time_constraint.as_deref(), Some("february"));
    }

    #[test]
    fn reveal_aggregate_marvel_star_wars() {
        let q = "How many weeks did it take me to watch all the Marvel Cinematic Universe movies and the main Star Wars films?";
        let r = reveal_question(q);
        assert_eq!(r.operation, IntentOperation::Aggregate);
        assert_eq!(r.operands.len(), 2);
        let m0 = mentions(q, 0);
        let m1 = mentions(q, 1);
        assert!(m0.iter().any(|m| m.contains("marvel")), "m0: {m0:?}");
        assert!(m1.iter().any(|m| m.contains("star war")), "m1: {m1:?}");
    }

    #[test]
    fn reveal_aggregate_facebook_instagram() {
        let q = "What was the total number of people reached by my Facebook ad campaign and Instagram influencer collaboration?";
        let r = reveal_question(q);
        assert_eq!(r.operation, IntentOperation::Aggregate);
        assert_eq!(r.operands.len(), 2);
    }

    #[test]
    fn reveal_shared_relation_both_visited() {
        let q = "Which city have both Jean and John visited?";
        let r = reveal_question(q);
        assert_eq!(r.operation, IntentOperation::SharedRelation, "q: {q}");
        let srq = r.shared_relation.expect("shared_relation set");
        assert_eq!(srq.subjects, vec!["Jean".to_string(), "John".to_string()]);
        assert_eq!(srq.family, PredicateFamily::PlacePresence);
        assert!(srq.expect_place);
    }

    #[test]
    fn reveal_shared_relation_volunteering_both_after() {
        let q = "What type of volunteering have John and Maria both done?";
        let r = reveal_question(q);
        assert_eq!(r.operation, IntentOperation::SharedRelation, "q: {q}");
        let srq = r.shared_relation.expect("shared_relation set");
        assert_eq!(srq.subjects, vec!["John".to_string(), "Maria".to_string()]);
        assert_eq!(srq.family, PredicateFamily::PlacePresence);
        assert!(!srq.expect_place);
    }

    #[test]
    fn reveal_shared_relation_rejects_non_names() {
        // "both" with non-name operands is not a shared relation.
        let q = "What did both the cat and the dog eat?";
        let r = reveal_question(q);
        assert_ne!(r.operation, IntentOperation::SharedRelation, "q: {q}");
    }

    #[test]
    fn reveal_lookup_falls_back_on_entity_poor_questions() {
        // All four are real both-miss questions: the intent path must decline.
        for q in [
            "How much did I spend on a designer handbag?",
            "What time do I stop checking work emails and messages?",
            "I received a piece of jewelry last Saturday from whom?",
            "What is the total distance of the hikes I did on two consecutive weekends?",
            "How many years older am I than when I graduated from college?",
            "What was the discount I got on my first purchase from the new clothing brand?",
        ] {
            let r = reveal_question(q);
            assert_eq!(r.operation, IntentOperation::Lookup, "q: {q}");
            assert!(r.fallback_reason.is_some(), "q: {q}");
        }
    }

    #[test]
    fn reveal_lookup_how_much_keeps_money_answer_type() {
        let r = reveal_question("how much did I spend on the designer handbag?");
        assert_eq!(r.operation, IntentOperation::Lookup);
        assert!(r.fallback_reason.is_some());
        assert_eq!(r.answer_shape, "single money amount");
    }

    #[test]
    fn reveal_lookup_how_many_keeps_count_answer_type() {
        let r = reveal_question("How many days did I spend traveling in Hawaii?");
        assert_eq!(r.operation, IntentOperation::Lookup);
        assert!(r.fallback_reason.is_some());
        assert_eq!(r.answer_shape, "single count");
    }

    #[test]
    fn reveal_lookup_how_long_keeps_duration_answer_type() {
        let r = reveal_question("How long did the road trip take?");
        assert_eq!(r.operation, IntentOperation::Lookup);
        assert!(r.fallback_reason.is_some());
        assert_eq!(r.answer_shape, "single duration");
    }

    #[test]
    fn reveal_lookup_without_how_marker_has_empty_answer_shape() {
        let r = reveal_question("What time do I stop checking work emails and messages?");
        assert_eq!(r.operation, IntentOperation::Lookup);
        assert!(r.fallback_reason.is_some());
        assert!(r.answer_shape.is_empty());
    }

    #[test]
    fn reveal_curated_adaptive_misses() {
        // All 34 adaptive misses from the full-500 LongMemEval benchmark
        // (segment_full500_intent.json, 2026-09-21, arm
        // adaptive_top_n_segment_enriched, recall_any@5 == 0). Curated
        // regression set: locks in the parser's reveal per miss so future
        // changes are measured against the real miss distribution instead
        // of hand-picked examples. Question text is verbatim, typos kept.
        let cases: &[(&str, IntentOperation, &str)] = &[
            ("How much did I spend on a designer handbag?", IntentOperation::Lookup, "single money amount"),
            ("What was the discount I got on my first purchase from the new clothing brand?", IntentOperation::Lookup, ""),
            ("Where did I attend my cousin's wedding?", IntentOperation::Lookup, ""),
            ("What time do I stop checking work emails and messages?", IntentOperation::Lookup, ""),
            ("What is the name of the music streaming service have I been using lately?", IntentOperation::Lookup, ""),
            ("How long did I wait for the decision on my asylum application?", IntentOperation::Lookup, "single duration"),
            ("How much time do I dedicate to practicing violin every day?", IntentOperation::Lookup, "single duration"),
            ("How many projects have I led or am currently leading?", IntentOperation::Lookup, "single count"),
            ("How many days did it take for my iPad case to arrive after I bought it?", IntentOperation::Lookup, "single count"),
            ("Can you suggest some accessories that would complement my current photography setup?", IntentOperation::Lookup, ""),
            ("Can you recommend some recent publications or conferences that I might find interesting?", IntentOperation::Lookup, ""),
            ("What should I serve for dinner this weekend with my homegrown ingredients?", IntentOperation::Lookup, ""),
            ("I've been thinking about making a cocktail for an upcoming get-together, but I'm not sure which one to choose. Any suggestions?", IntentOperation::Lookup, ""),
            ("I've been having trouble with the battery life on my phone lately. Any tips?", IntentOperation::Lookup, ""),
            ("I'm getting excited about my visit to the music store this weekend. Any tips on what to look for in a new guitar?", IntentOperation::Lookup, ""),
            ("I was thinking of trying a new coffee creamer recipe. Any recommendations?", IntentOperation::Lookup, ""),
            ("Can you suggest some activities I can do during my commute to work?", IntentOperation::Lookup, ""),
            ("What is the total distance of the hikes I did on two consecutive weekends?", IntentOperation::Lookup, ""),
            ("How many years older am I than when I graduated from college?", IntentOperation::Lookup, "single count"),
            ("What is the total number of siblings I have?", IntentOperation::Lookup, ""),
            ("How many days ago did I attend a networking event?", IntentOperation::Lookup, "single count"),
            ("I received a piece of jewelry last Saturday from whom?", IntentOperation::Lookup, ""),
            ("I mentioned that I participated in an art-related event two weeks ago. Where was that event held at?", IntentOperation::Lookup, ""),
            ("What was the the life event of one of my relatives that I participated in a week ago?", IntentOperation::Lookup, ""),
            ("Who did I meet with during the lunch last Tuesday?", IntentOperation::Lookup, ""),
            ("I mentioned cooking something for my friend a couple of days ago. What was it?", IntentOperation::Lookup, ""),
            ("What was the significant buisiness milestone I mentioned four weeks ago?", IntentOperation::Lookup, ""),
            ("What kitchen appliance did I buy 10 days ago?", IntentOperation::Lookup, ""),
            ("Where did I attend the religious activity last week?", IntentOperation::Lookup, ""),
            ("What was the social media activity I participated 5 days ago?", IntentOperation::Lookup, ""),
            ("Which streaming service did I start using most recently?", IntentOperation::Lookup, ""),
            ("Which task did I complete first, fixing the fence or purchasing three cows from Peter?", IntentOperation::TemporalOrder, "argmin/argmax over option timestamps"),
            ("How much weight have I lost since I started going to the gym consistently?", IntentOperation::Lookup, "single quantity"),
            ("How often do I see Dr. Johnson?", IntentOperation::Lookup, ""),
        ];
        assert_eq!(cases.len(), 34);
        for (q, expected_op, expected_shape) in cases {
            let r = reveal_question(q);
            assert_eq!(r.operation, *expected_op, "q: {q}");
            assert_eq!(r.answer_shape, *expected_shape, "q: {q}");
            // Lookup never fires the intent path; structured ops always do.
            assert_eq!(
                r.fallback_reason.is_some(),
                *expected_op == IntentOperation::Lookup,
                "q: {q}"
            );
        }
    }

    #[test]
    fn reveal_temporal_filter_before_pivot() {
        let q = "How many charity events did I participate in before the 'Run for the Cure' event?";
        let r = reveal_question(q);
        assert_eq!(r.operation, IntentOperation::TemporalFilter);
        assert!(r.fallback_reason.is_none(), "r: {r:?}");
        assert_eq!(r.operands.len(), 2);
        // Pivot scope carries the quoted event; class scope is keyword-driven.
        let m1 = mentions(q, 1);
        assert!(
            m1.iter().any(|m| m.contains("run for the cure")),
            "m1: {m1:?}"
        );
        assert!(r.answer_shape.contains("before"));
    }

    #[test]
    fn reveal_temporal_order_quoted_options() {
        let q = "Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?";
        let r = reveal_question(q);
        assert_eq!(r.operation, IntentOperation::TemporalOrder);
        assert_eq!(r.operands.len(), 2);
        let m0 = mentions(q, 0);
        let m1 = mentions(q, 1);
        assert!(
            m0.iter().any(|m| m.contains("effect time manag")),
            "m0: {m0:?}"
        );
        assert!(m1.iter().any(|m| m.contains("data analys")), "m1: {m1:?}");
    }

    #[test]
    fn scope_time_tokens_drops_codes_keeps_dates() {
        assert_eq!(
            scope_time_tokens("the Samsung Galaxy S22 or the Dell XPS 13"),
            Vec::<String>::new()
        );
        assert_eq!(scope_time_tokens("August 11"), vec!["11", "august"]);
        assert_eq!(scope_time_tokens("I bought it 10 days ago"), vec!["10"]);
        assert_eq!(scope_time_tokens("in February"), vec!["february"]);
    }

    #[test]
    fn rendered_reveal_is_readable() {
        let r = reveal_question(
            "How many days did I spend in total traveling in Hawaii and in New York City?",
        );
        let rendered = r.rendered();
        assert!(rendered.starts_with("aggregate("), "{rendered}");
        assert!(rendered.contains("Hawaii"), "{rendered}");
    }

    // --- query_with_intent integration tests on a tiny index ---

    fn entity(text: &str, label: &str) -> Tier1Entity {
        Tier1Entity {
            text: text.to_string(),
            label: label.to_string(),
            start: 0,
            end: text.len(),
            score: Some(1.0),
            source: "test".to_string(),
        }
    }

    fn record(
        doc_id: &str,
        group_id: &str,
        content: &str,
        entities: Vec<Tier1Entity>,
    ) -> DocRecord {
        DocRecord {
            doc_id: doc_id.to_string(),
            source: format!("memory://{doc_id}"),
            content: content.to_string(),
            timestamp: None,
            doc_length: content.len(),
            author_agent: None,
            group_id: Some(group_id.to_string()),
            filters: std::collections::BTreeMap::new(),
            probable_topic: None,
            doc_type_guess: None,
            headings: vec![],
            doc_links: vec![],
            temporal_terms: vec![],
            key_entities: entities,
            important_terms: vec![],
            section_chunks: vec![],
            embedding: None,
            top_claims: vec![],
            provenance: Provenance {
                source: "test".to_string(),
                timestamp: None,
                ner_provider: "test".to_string(),
                term_ranker: "test".to_string(),
                index_version: "test".to_string(),
            },
            content_hash: String::new(),
        }
    }

    /// Two segments: one Hawaii doc, one New York City doc. The answer spans
    /// both by construction: no single doc mentions both operands.
    fn travel_index() -> SegmentedMemoryIndex {
        let seg_a = build_memory_index_segment(
            "seg-hi".to_string(),
            vec![record(
                "doc-hawaii",
                "session-1",
                "I spent 10 days traveling in Hawaii last spring",
                vec![entity("Hawaii", "GPE")],
            )],
        );
        let seg_b = build_memory_index_segment(
            "seg-nyc".to_string(),
            vec![record(
                "doc-nyc",
                "session-2",
                "My New York City trip lasted 5 days in the fall",
                vec![entity("New York City", "GPE")],
            )],
        );
        SegmentedMemoryIndex::from_segments(vec![seg_a, seg_b]).unwrap()
    }

    #[test]
    fn intent_unions_per_operand_evidence() {
        let index = travel_index();
        let temporal = TemporalQueryContext::default();
        let (output, diagnostics) = index
            .query_with_intent(
                "How many days did I spend in total traveling in Hawaii and in New York City?",
                5,
                4,
                5,
                SegmentRoutingStrategy::SparseOverlap,
                temporal,
            )
            .expect("intent should fire");
        assert_eq!(diagnostics.operation, "aggregate");
        assert_eq!(diagnostics.validated_operand_count, 2);
        let ranked: Vec<&str> = output.results.iter().map(|r| r.doc_id.as_str()).collect();
        // Both operand sessions surface: the union is the answer set.
        assert!(ranked.contains(&"doc-hawaii"), "{ranked:?}");
        assert!(ranked.contains(&"doc-nyc"), "{ranked:?}");
    }

    #[test]
    fn intent_declines_when_an_operand_has_no_corpus_entity() {
        // Corpus mentions only Hawaii: the NYC operand cannot validate.
        let seg = build_memory_index_segment(
            "seg-hi".to_string(),
            vec![record(
                "doc-hawaii",
                "session-1",
                "I spent 10 days traveling in Hawaii",
                vec![entity("Hawaii", "GPE")],
            )],
        );
        let index = SegmentedMemoryIndex::from_segments(vec![seg]).unwrap();
        let temporal = TemporalQueryContext::default();
        let out = index.query_with_intent(
            "How many days did I spend in total traveling in Hawaii and in New York City?",
            5,
            4,
            5,
            SegmentRoutingStrategy::SparseOverlap,
            temporal,
        );
        assert!(out.is_none());
    }

    #[test]
    fn intent_declines_on_lookup_questions() {
        let index = travel_index();
        let temporal = TemporalQueryContext::default();
        let out = index.query_with_intent(
            "How much did I spend on a designer handbag?",
            5,
            4,
            5,
            SegmentRoutingStrategy::SparseOverlap,
            temporal,
        );
        assert!(out.is_none());
    }
}
