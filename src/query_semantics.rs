#![allow(unexpected_cfgs)]

use chrono::NaiveDateTime;
use fuzzydate::parse as fuzzy_parse;
use regex::Regex;
use serde::Serialize;
use std::collections::HashSet;

use crate::aggregation::AggregateIntent;
use crate::query_expansion::normalize_for_index;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum QueryKind {
    HowMany,
    HowMuch,
    Who,
    What,
    When,
    Where,
    Which,
    Why,
    Statement,
    Other,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum QuerySpanKind {
    Entity,
    NounPhrase,
    VerbPhrase,
    Subject,
    Object,
    Temporal,
    Quantity,
    QuestionWord,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum QueryTimeHint {
    Past,
    Present,
    Ongoing,
    Mixed,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum QueryRoutingIntent {
    Count,
    Sum,
    Sequence,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuerySpan {
    pub kind: QuerySpanKind,
    pub text: String,
    pub normalized: String,
    pub score: f32,
    pub source: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryTemporal {
    pub phrase: String,
    pub resolved_at: Option<String>,
    pub source: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryPosTag {
    pub word: String,
    pub label: String,
    pub score: f64,
}

#[derive(Debug, Clone)]
struct POSTag {
    word: String,
    label: String,
    score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryAnalysis {
    pub original_query: String,
    pub normalized_query: String,
    pub kind: QueryKind,
    pub quantity_intent: Option<String>,
    pub subject_hint: Option<String>,
    pub object_hint: Option<String>,
    pub temporal: Option<QueryTemporal>,
    pub time_hint: Option<QueryTimeHint>,
    pub query_routing_intent: Option<QueryRoutingIntent>,
    pub raw_pos_tags: Vec<QueryPosTag>,
    pub spans: Vec<QuerySpan>,
    pub augmented_query: String,
}

pub fn analyze_query(query: &str) -> QueryAnalysis {
    let original_query = query.trim().to_string();
    let signals = analyze_query_signals(&original_query);
    build_query_analysis(
        original_query,
        signals.pos_tags,
        signals.entities,
        signals.focused_pos_tags,
        signals.source,
    )
}

#[derive(Debug, Clone)]
struct QuerySignals {
    pos_tags: Vec<POSTag>,
    entities: Vec<QuerySpan>,
    focused_pos_tags: Option<Vec<POSTag>>,
    source: QuerySignalSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuerySignalSource {
    Heuristic,
    RustBert,
}

#[cfg(feature = "rust-bert-query-semantics")]
fn analyze_query_signals(query: &str) -> QuerySignals {
    rust_bert_backend::analyze_query_signals(query)
}

#[cfg(not(feature = "rust-bert-query-semantics"))]
fn analyze_query_signals(query: &str) -> QuerySignals {
    heuristic_query_signals(query)
}

fn heuristic_query_signals(query: &str) -> QuerySignals {
    let pos_tags = heuristic_pos_tags(query);
    let entities = heuristic_entities(query);
    QuerySignals {
        pos_tags,
        entities,
        focused_pos_tags: None,
        source: QuerySignalSource::Heuristic,
    }
}

#[cfg(feature = "rust-bert-query-semantics")]
mod rust_bert_backend {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    struct QueryModels {
        ner: rust_bert::pipelines::ner::NERModel,
        pos: rust_bert::pipelines::pos_tagging::POSModel,
    }

    static MODELS: OnceLock<Mutex<Option<QueryModels>>> = OnceLock::new();

    fn with_models<R>(f: impl FnOnce(&QueryModels) -> R) -> Option<R> {
        let lock = MODELS.get_or_init(|| {
            Mutex::new({
                let ner = rust_bert::pipelines::ner::NERModel::new(Default::default()).ok();
                let pos = rust_bert::pipelines::pos_tagging::POSModel::new(Default::default()).ok();
                match (ner, pos) {
                    (Some(ner), Some(pos)) => Some(QueryModels { ner, pos }),
                    _ => None,
                }
            })
        });
        let guard = lock.lock().ok()?;
        let models = guard.as_ref()?;
        Some(f(models))
    }

    pub fn analyze_query_signals(query: &str) -> QuerySignals {
        let Some(signals) = with_models(|models| {
            let pos_tags = models
                .pos
                .predict(&[query])
                .into_iter()
                .next()
                .unwrap_or_default()
                .into_iter()
                .map(|tag| POSTag {
                    word: tag.word,
                    label: tag.label,
                    score: tag.score,
                })
                .collect::<Vec<_>>();

            let first_sentence = first_sentence_text(query);
            let first_sentence_pos = first_sentence.as_deref().map_or_else(Vec::new, |sentence| {
                models
                    .pos
                    .predict(&[sentence])
                    .into_iter()
                    .next()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|tag| POSTag {
                        word: tag.word,
                        label: tag.label,
                        score: tag.score,
                    })
                    .collect::<Vec<_>>()
            });

            let entities = models
                .ner
                .predict_full_entities(&[query])
                .into_iter()
                .next()
                .unwrap_or_default()
                .into_iter()
                .map(|entity| {
                    let label = entity.label;
                    let text = entity.word;
                    QuerySpan {
                        kind: QuerySpanKind::Entity,
                        text: text.clone(),
                        normalized: normalize_for_index(&text),
                        score: entity.score as f32,
                        source: format!("rust-bert-ner:{label}"),
                    }
                })
                .collect::<Vec<_>>();

            let focused_pos_tags = if first_sentence_pos.is_empty() {
                None
            } else {
                Some(first_sentence_pos)
            };

            // Previous behavior kept the first-sentence model outputs in the main streams:
            //
            // if !first_sentence_pos.is_empty() {
            //     let first_sentence_entities = models
            //         .ner
            //         .predict_full_entities(&[first_sentence.as_deref().unwrap_or(query)])
            //         .into_iter()
            //         .next()
            //         .unwrap_or_default()
            //         .into_iter()
            //         .map(|entity| {
            //             let text = entity.word;
            //             QuerySpan {
            //                 kind: QuerySpanKind::Entity,
            //                 text: text.clone(),
            //                 normalized: normalize_for_index(&text),
            //                 score: entity.score as f32,
            //                 source: "rust-bert".to_string(),
            //             }
            //         })
            //         .collect::<Vec<_>>();
            //     signals.entities.extend(first_sentence_entities);
            //     signals.pos_tags.extend(first_sentence_pos);
            // }

            QuerySignals {
                pos_tags,
                entities,
                focused_pos_tags,
                source: QuerySignalSource::RustBert,
            }
        }) else {
            return super::heuristic_query_signals(query);
        };
        signals
    }

    fn first_sentence_text(query: &str) -> Option<String> {
        let mut parts = query.split_terminator(['.', '!', '?']);
        let first = parts.next()?.trim();
        if first.is_empty() {
            None
        } else {
            Some(first.to_string())
        }
    }
}

fn build_query_analysis(
    original_query: String,
    pos_tags: Vec<POSTag>,
    entities: Vec<QuerySpan>,
    focused_pos_tags: Option<Vec<POSTag>>,
    source: QuerySignalSource,
) -> QueryAnalysis {
    let normalized_query = normalize_for_index(&original_query);
    let kind = classify_query_kind(&original_query);
    let aggregate_intent = quantity_intent(&original_query);
    let quantity_intent = aggregate_intent.as_ref().map(|intent| match intent {
        AggregateIntent::Count => "count".to_string(),
        AggregateIntent::Sum => "sum".to_string(),
    });

    let temporal = extract_temporal(&original_query);
    let query_routing_intent =
        query_routing_intent(&original_query, aggregate_intent.as_ref(), &temporal);
    let mut spans = Vec::new();
    if let Some(qw) = leading_question_word(&original_query) {
        spans.push(QuerySpan {
            kind: QuerySpanKind::QuestionWord,
            text: qw.clone(),
            normalized: normalize_for_index(&qw),
            score: 0.7,
            source: "heuristic".to_string(),
        });
    }

    let raw_pos_tags = pos_tags
        .iter()
        .map(|tag| QueryPosTag {
            word: tag.word.clone(),
            label: tag.label.clone(),
            score: tag.score,
        })
        .collect::<Vec<_>>();
    let time_hint = infer_time_hint_from_pos(&pos_tags);

    spans.extend(entities.iter().cloned());

    let span_tags = focused_pos_tags.as_ref().unwrap_or(&pos_tags);
    let noun_chunks = extract_noun_phrases(span_tags);
    let noun_phrases: Vec<String> = noun_chunks
        .iter()
        .filter_map(|p| clean_noun_phrase_for_query(&p.text))
        .collect();
    let verb_phrases = extract_verb_phrases(span_tags);
    let (subject, object) = infer_subject_object(span_tags, &noun_chunks, &entities);

    let rust_bert_model_signals = source == QuerySignalSource::RustBert;
    let (subject_hint, object_hint) = if rust_bert_model_signals {
        // Keep the model-backed branch close to the original rust-bert flow:
        // infer from the focused POS stream and do not layer text-only fallback
        // hints on top of model-derived subject/object hints.
        (subject, object)
    } else {
        (
            subject.or_else(|| {
                let fallback_phrases = heuristic_noun_phrases(&original_query);
                infer_subject_object_from_text(&original_query, &fallback_phrases).0
            }),
            object.or_else(|| {
                let fallback_phrases = heuristic_noun_phrases(&original_query);
                infer_subject_object_from_text(&original_query, &fallback_phrases).1
            }),
        )
    };

    if rust_bert_model_signals {
        spans.extend(noun_chunks.iter().cloned().map(|phrase| QuerySpan {
            kind: QuerySpanKind::NounPhrase,
            text: phrase.text.clone(),
            normalized: normalize_for_index(&phrase.text),
            score: 0.6,
            source: "rust-bert-pos".to_string(),
        }));

        // Current experiment, kept available for quick restore:
        //
        // spans.extend(verb_phrases.iter().cloned().map(|phrase| QuerySpan {
        //     kind: QuerySpanKind::VerbPhrase,
        //     text: phrase.clone(),
        //     normalized: normalize_for_index(&phrase),
        //     score: 0.35,
        //     source: "rust-bert-pos".to_string(),
        // }));
    } else {
        spans.extend(noun_chunks.iter().cloned().map(|phrase| QuerySpan {
            kind: QuerySpanKind::NounPhrase,
            text: phrase.text.clone(),
            normalized: normalize_for_index(&phrase.text),
            score: 0.4,
            source: "pos".to_string(),
        }));
        spans.extend(verb_phrases.iter().cloned().map(|phrase| QuerySpan {
            kind: QuerySpanKind::VerbPhrase,
            text: phrase.clone(),
            normalized: normalize_for_index(&phrase),
            score: 0.35,
            source: "pos".to_string(),
        }));
    }

    if let Some(ref temporal) = temporal {
        spans.push(QuerySpan {
            kind: QuerySpanKind::Temporal,
            text: temporal.phrase.clone(),
            normalized: normalize_for_index(&temporal.phrase),
            score: 0.8,
            source: temporal.source.clone(),
        });
    }

    if let Some(subject) = subject_hint.clone() {
        spans.push(QuerySpan {
            kind: QuerySpanKind::Subject,
            text: subject.clone(),
            normalized: normalize_for_index(&subject),
            score: 0.75,
            source: "heuristic".to_string(),
        });
    }

    if let Some(object) = object_hint.clone() {
        spans.push(QuerySpan {
            kind: QuerySpanKind::Object,
            text: object.clone(),
            normalized: normalize_for_index(&object),
            score: 0.75,
            source: "heuristic".to_string(),
        });
    }

    if quantity_intent.is_some() {
        let quantity_text =
            quantity_phrase(&original_query).unwrap_or_else(|| original_query.clone());
        spans.push(QuerySpan {
            kind: QuerySpanKind::Quantity,
            normalized: normalize_for_index(&quantity_text),
            text: quantity_text,
            score: 0.85,
            source: "heuristic".to_string(),
        });
    }

    let augmented_verb_phrases: &[String] = if rust_bert_model_signals {
        &[]
    } else {
        &verb_phrases
    };
    let augmented_query = build_augmented_query(
        &original_query,
        &entities,
        &noun_phrases,
        augmented_verb_phrases,
        subject_hint.as_deref(),
        object_hint.as_deref(),
    );

    if rust_bert_model_signals {
        // Previous conservative variant for the model-backed branch:
        //
        // let augmented_query = original_query.clone();
        //
        // Previous permissive experiment:
        //
        // let (subject_hint, object_hint) = (
        //     subject.or_else(|| {
        //         let fallback_phrases = heuristic_noun_phrases(&original_query);
        //         infer_subject_object_from_text(&original_query, &fallback_phrases).0
        //     }),
        //     object.or_else(|| {
        //         let fallback_phrases = heuristic_noun_phrases(&original_query);
        //         infer_subject_object_from_text(&original_query, &fallback_phrases).1
        //     }),
        // );
    }

    QueryAnalysis {
        original_query,
        normalized_query,
        kind,
        quantity_intent,
        subject_hint,
        object_hint,
        temporal,
        time_hint,
        query_routing_intent,
        raw_pos_tags,
        spans: dedup_spans(spans),
        augmented_query,
    }
}

#[derive(Debug, Clone)]
struct NounPhrase {
    text: String,
    start: usize,
    end: usize,
}

fn classify_query_kind(query: &str) -> QueryKind {
    let lower = query.trim().to_lowercase();
    if lower.starts_with("how many") {
        return QueryKind::HowMany;
    }
    if lower.starts_with("how much") {
        return QueryKind::HowMuch;
    }
    if lower.starts_with("who ") || lower == "who" {
        return QueryKind::Who;
    }
    if lower.starts_with("what ") || lower == "what" {
        return QueryKind::What;
    }
    if lower.starts_with("when ") || lower == "when" {
        return QueryKind::When;
    }
    if lower.starts_with("where ") || lower == "where" {
        return QueryKind::Where;
    }
    if lower.starts_with("which ") || lower == "which" {
        return QueryKind::Which;
    }
    if lower.starts_with("why ") || lower == "why" {
        return QueryKind::Why;
    }
    QueryKind::Statement
}

fn quantity_intent(query: &str) -> Option<AggregateIntent> {
    let lower = query.to_lowercase();
    if lower.contains("how many")
        || lower.starts_with("count ")
        || lower.contains(" number of ")
        || lower.contains(" number of")
        || lower.contains("count the ")
        || lower.contains("times did i")
    {
        return Some(AggregateIntent::Count);
    }
    if lower.contains("how much")
        || lower.contains(" total ")
        || lower.starts_with("total ")
        || lower.contains("combined")
        || lower.contains("in all")
        || lower.contains("sum ")
    {
        return Some(AggregateIntent::Sum);
    }
    None
}

fn quantity_phrase(query: &str) -> Option<String> {
    let lower = query.to_lowercase();
    if let Some(m) = Regex::new(r"\bhow\s+(many|much)\b").ok()?.find(&lower) {
        return Some(query[m.start()..m.end()].to_string());
    }
    if let Some(m) = Regex::new(r"\b(number of|count(?: the)?|total)\b")
        .ok()?
        .find(&lower)
    {
        return Some(query[m.start()..m.end()].to_string());
    }
    None
}

fn query_routing_intent(
    query: &str,
    quantity_intent: Option<&AggregateIntent>,
    temporal: &Option<QueryTemporal>,
) -> Option<QueryRoutingIntent> {
    if let Some(intent) = quantity_intent {
        return Some(match intent {
            AggregateIntent::Count => QueryRoutingIntent::Count,
            AggregateIntent::Sum => QueryRoutingIntent::Sum,
        });
    }

    let lower = query.to_lowercase();
    let sequence_markers = [
        "consecutive",
        "earliest",
        "latest",
        "first",
        "last",
        "next",
        "previous",
        "before",
        "after",
        "order",
        "ordered",
        "sequence",
    ];
    if temporal.is_some() && sequence_markers.iter().any(|marker| lower.contains(marker)) {
        return Some(QueryRoutingIntent::Sequence);
    }

    if lower.contains("consecutive days")
        || lower.contains("from earliest to latest")
        || lower.contains("which happened first")
    {
        return Some(QueryRoutingIntent::Sequence);
    }

    None
}

fn heuristic_pos_tags(query: &str) -> Vec<POSTag> {
    let mut tags = Vec::new();
    for raw in query.split_whitespace() {
        let word = raw.trim_matches(|c: char| c.is_ascii_punctuation());
        if word.is_empty() {
            continue;
        }
        let lower = word.to_lowercase();
        let label = if matches!(
            lower.as_str(),
            "how" | "what" | "who" | "when" | "where" | "which" | "why"
        ) {
            "WRB"
        } else if matches!(
            lower.as_str(),
            "i" | "me"
                | "we"
                | "us"
                | "you"
                | "he"
                | "him"
                | "she"
                | "her"
                | "it"
                | "they"
                | "them"
                | "my"
                | "mine"
                | "our"
                | "ours"
                | "your"
                | "yours"
                | "their"
                | "theirs"
        ) {
            "PRP"
        } else if matches!(
            lower.as_str(),
            "a" | "an"
                | "the"
                | "this"
                | "that"
                | "these"
                | "those"
                | "many"
                | "much"
                | "some"
                | "any"
                | "each"
                | "every"
                | "all"
        ) {
            "DT"
        } else if matches!(
            lower.as_str(),
            "of" | "to"
                | "for"
                | "on"
                | "in"
                | "at"
                | "with"
                | "from"
                | "by"
                | "about"
                | "over"
                | "under"
                | "through"
                | "during"
                | "before"
                | "after"
                | "between"
                | "into"
                | "onto"
                | "near"
                | "within"
                | "without"
                | "per"
        ) {
            "IN"
        } else if matches!(lower.as_str(), "and" | "or" | "but" | "nor") {
            "CC"
        } else if matches!(
            lower.as_str(),
            "is" | "am"
                | "are"
                | "was"
                | "were"
                | "be"
                | "been"
                | "being"
                | "have"
                | "has"
                | "had"
                | "do"
                | "does"
                | "did"
                | "can"
                | "could"
                | "should"
                | "would"
                | "will"
                | "shall"
                | "may"
                | "might"
                | "must"
        ) {
            if matches!(
                lower.as_str(),
                "was" | "were" | "did" | "had" | "could" | "should" | "would"
            ) {
                "VBD"
            } else if matches!(lower.as_str(), "been" | "being") {
                "VBN"
            } else if matches!(
                lower.as_str(),
                "am" | "is" | "are" | "has" | "does" | "will" | "shall"
            ) {
                "VBP"
            } else {
                "VB"
            }
        } else if lower.ends_with("ing") && lower.len() > 4 {
            "VBG"
        } else if lower.ends_with("ed") && lower.len() > 3 {
            "VBD"
        } else if lower.ends_with("ly") {
            "RB"
        } else if matches!(
            lower.as_str(),
            "currently" | "now" | "today" | "recently" | "already"
        ) {
            "RB"
        } else if lower.chars().all(|c| c.is_ascii_digit()) {
            "CD"
        } else if word
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '-')
        {
            "NNP"
        } else if word
            .chars()
            .next()
            .map_or(false, |c| c.is_ascii_uppercase())
        {
            "NNP"
        } else if lower.ends_with('s') && lower.len() > 3 {
            "NNS"
        } else {
            "NN"
        };
        tags.push(POSTag {
            word: word.to_string(),
            label: label.to_string(),
            score: 0.55,
        });
    }
    tags
}

fn heuristic_entities(query: &str) -> Vec<QuerySpan> {
    let words: Vec<&str> = query.split_whitespace().collect();
    let mut spans = Vec::new();
    let mut current: Vec<String> = Vec::new();

    let flush = |spans: &mut Vec<QuerySpan>, current: &mut Vec<String>| {
        if current.is_empty() {
            return;
        }
        let text = current.join(" ").trim().to_string();
        if text.is_empty() {
            current.clear();
            return;
        }
        spans.push(QuerySpan {
            kind: QuerySpanKind::Entity,
            normalized: normalize_for_index(&text),
            text,
            score: 0.55,
            source: "heuristic".to_string(),
        });
        current.clear();
    };

    for word in words {
        let token = word.trim_matches(|c: char| c.is_ascii_punctuation());
        if token.is_empty() {
            flush(&mut spans, &mut current);
            continue;
        }
        let looks_like_entity = token.len() > 1
            && (token
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '-')
                || token
                    .chars()
                    .next()
                    .map_or(false, |c| c.is_ascii_uppercase()));
        if looks_like_entity {
            current.push(token.to_string());
        } else {
            flush(&mut spans, &mut current);
        }
    }
    flush(&mut spans, &mut current);
    spans
}

fn extract_verb_phrases(tags: &[POSTag]) -> Vec<String> {
    let mut phrases = Vec::new();
    let mut current = Vec::new();
    let mut seen_verb = false;

    let flush = |phrases: &mut Vec<String>, current: &mut Vec<String>, seen_verb: &mut bool| {
        if *seen_verb && !current.is_empty() {
            phrases.push(current.join(" "));
        }
        current.clear();
        *seen_verb = false;
    };

    for tag in tags {
        if is_punct(&tag.label, &tag.word) {
            flush(&mut phrases, &mut current, &mut seen_verb);
            continue;
        }
        if is_verbish(&tag.label) || (tag.label == "RB" && seen_verb) {
            current.push(tag.word.clone());
            if is_verbish(&tag.label) {
                seen_verb = true;
            }
        } else {
            flush(&mut phrases, &mut current, &mut seen_verb);
        }
    }
    flush(&mut phrases, &mut current, &mut seen_verb);
    phrases
}

#[derive(Debug, Clone)]
struct PhraseMatch {
    text: String,
    start: usize,
    end: usize,
}

fn extract_temporal(query: &str) -> Option<QueryTemporal> {
    for match_ in temporal_candidates(query) {
        let resolved_at = fuzzy_parse(match_.text.as_str())
            .ok()
            .map(|dt: NaiveDateTime| dt.to_string());
        return Some(QueryTemporal {
            phrase: match_.text,
            resolved_at,
            source: "fuzzydate".to_string(),
        });
    }
    None
}

fn temporal_candidates(query: &str) -> Vec<PhraseMatch> {
    let lower = query.to_lowercase();
    let mut out = Vec::new();
    let patterns = [
        r"\b(today|tomorrow|yesterday|tonight)\b",
        r"\b(?:last|this|next)\s+(?:week|month|year|weekend)\b",
        r"\b(?:last|this|next)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:day|days|week|weeks|month|months|year|years)(?:\s+ago)?\b",
        r"\bconsecutive days\b",
    ];

    for pattern in patterns {
        let re = Regex::new(pattern).expect("valid temporal regex");
        for m in re.find_iter(&lower) {
            out.push(PhraseMatch {
                text: query[m.start()..m.end()].to_string(),
                start: m.start(),
                end: m.end(),
            });
        }
    }

    out.sort_by_key(|m| (m.start, m.end));
    dedup_phrase_matches(out)
}

fn dedup_phrase_matches(mut items: Vec<PhraseMatch>) -> Vec<PhraseMatch> {
    let mut seen = HashSet::new();
    items
        .drain(..)
        .filter(|item| seen.insert(item.text.to_lowercase()))
        .collect()
}

fn leading_question_word(query: &str) -> Option<String> {
    let lower = query.trim().to_lowercase();
    let first = lower.split_whitespace().next()?;
    match first {
        "how" => Some("how".to_string()),
        "what" => Some("what".to_string()),
        "who" => Some("who".to_string()),
        "when" => Some("when".to_string()),
        "where" => Some("where".to_string()),
        "which" => Some("which".to_string()),
        "why" => Some("why".to_string()),
        _ => None,
    }
}

fn infer_time_hint_from_pos(tags: &[POSTag]) -> Option<QueryTimeHint> {
    let mut saw_past = false;
    let mut saw_present = false;
    let mut saw_ongoing = false;

    for tag in tags {
        let word = tag.word.to_lowercase();
        match tag.label.as_str() {
            "VBD" | "VBN" => saw_past = true,
            "VBP" | "VBZ" => saw_present = true,
            "VBG" => saw_ongoing = true,
            "RB" if matches!(word.as_str(), "currently" | "now" | "today") => saw_ongoing = true,
            _ => {}
        }
    }

    if saw_past && (saw_present || saw_ongoing) {
        return Some(QueryTimeHint::Mixed);
    }
    if saw_ongoing {
        return Some(QueryTimeHint::Ongoing);
    }
    if saw_past {
        return Some(QueryTimeHint::Past);
    }
    if saw_present {
        return Some(QueryTimeHint::Present);
    }
    None
}

fn is_nounish(label: &str) -> bool {
    label.starts_with("NN") || matches!(label, "CD" | "JJ" | "JJR" | "JJS")
}

fn is_verbish(label: &str) -> bool {
    label.starts_with('V') || matches!(label, "MD")
}

fn is_punct(label: &str, word: &str) -> bool {
    matches!(label, "." | "," | ":" | ";" | "``" | "''")
        || word.chars().all(|c| !c.is_alphanumeric())
}

fn extract_noun_phrases(tags: &[POSTag]) -> Vec<NounPhrase> {
    let mut phrases = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut start = 0usize;

    let flush =
        |phrases: &mut Vec<NounPhrase>, current: &mut Vec<String>, start: usize, end: usize| {
            if current.is_empty() {
                return;
            }
            let text = current.join(" ").trim().to_string();
            if text.is_empty() {
                current.clear();
                return;
            }
            phrases.push(NounPhrase { text, start, end });
            current.clear();
        };

    for (idx, tag) in tags.iter().enumerate() {
        if is_punct(&tag.label, &tag.word) {
            flush(&mut phrases, &mut current, start, idx);
            continue;
        }
        if is_nounish(&tag.label) {
            if current.is_empty() {
                start = idx;
            }
            current.push(tag.word.clone());
        } else {
            flush(&mut phrases, &mut current, start, idx);
        }
    }
    flush(&mut phrases, &mut current, start, tags.len());
    phrases
}

fn infer_subject_object(
    tags: &[POSTag],
    noun_phrases: &[NounPhrase],
    entities: &[QuerySpan],
) -> (Option<String>, Option<String>) {
    let first_verb = tags.iter().position(|tag| is_verbish(&tag.label));
    let subject = noun_phrases
        .iter()
        .find(|phrase| first_verb.map_or(true, |verb_idx| phrase.end <= verb_idx))
        .and_then(|phrase| subject_head(&phrase.text))
        .or_else(|| entities.first().map(|entity| entity.text.clone()));
    let object = noun_phrases
        .iter()
        .rev()
        .find(|phrase| first_verb.map_or(true, |verb_idx| phrase.start >= verb_idx))
        .and_then(|phrase| subject_head(&phrase.text))
        .or_else(|| {
            noun_phrases
                .last()
                .and_then(|phrase| subject_head(&phrase.text))
        });
    (subject, object)
}

fn infer_subject_object_from_text(
    query: &str,
    noun_phrases: &[NounPhrase],
) -> (Option<String>, Option<String>) {
    let lower = query.to_lowercase();
    let first_verb = lower
        .split_whitespace()
        .position(|token| {
            matches!(
                token,
                "is" | "are"
                    | "was"
                    | "were"
                    | "do"
                    | "does"
                    | "did"
                    | "can"
                    | "could"
                    | "should"
                    | "would"
                    | "didn't"
                    | "will"
                    | "has"
                    | "have"
                    | "had"
            )
        })
        .unwrap_or(usize::MAX);
    let subject = noun_phrases
        .iter()
        .find(|phrase| phrase.start <= first_verb)
        .and_then(|phrase| subject_head(&phrase.text));
    let object = noun_phrases
        .iter()
        .rev()
        .find(|phrase| phrase.start >= first_verb)
        .and_then(|phrase| subject_head(&phrase.text))
        .or_else(|| {
            noun_phrases
                .last()
                .and_then(|phrase| subject_head(&phrase.text))
        });
    (subject, object)
}

fn subject_head(text: &str) -> Option<String> {
    let stopwords = [
        "a", "an", "the", "this", "that", "these", "those", "many", "much", "more", "most", "some",
        "any", "each", "every", "all", "my", "your", "his", "her", "its", "our", "their", "me",
        "i", "we", "you", "he", "she", "they", "it",
    ];
    let mut parts = text
        .split_whitespace()
        .filter(|part| {
            let lower = part.to_lowercase();
            !stopwords.contains(&lower.as_str())
        })
        .map(|part| part.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    parts.pop().map(|s| s.to_string())
}

fn clean_noun_phrase_for_query(text: &str) -> Option<String> {
    let stopwords = [
        "a", "an", "the", "this", "that", "these", "those", "many", "much", "more", "most", "some",
        "any", "each", "every", "all", "my", "your", "his", "her", "its", "our", "their", "me",
        "i", "we", "you", "he", "she", "they", "it",
    ];
    let parts = text
        .split_whitespace()
        .map(|part| part.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_'))
        .filter(|part| !part.is_empty())
        .filter(|part| {
            let lower = part.to_lowercase();
            !stopwords.contains(&lower.as_str())
        })
        .collect::<Vec<_>>();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

fn heuristic_noun_phrases(query: &str) -> Vec<NounPhrase> {
    if normalize_for_index(query).is_empty() {
        return Vec::new();
    }
    let words: Vec<&str> = query.split_whitespace().collect();
    let stop_words = [
        "what", "who", "when", "where", "why", "how", "many", "much", "is", "are", "was", "were",
        "do", "does", "did", "the", "a", "an", "of", "to", "for", "on", "in", "at", "with", "from",
        "by", "about", "and",
    ];
    let mut phrases = Vec::new();
    let mut current = Vec::new();
    let mut start = 0usize;
    for (idx, word) in words.iter().enumerate() {
        let token = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_');
        if token.is_empty() || stop_words.contains(&token.to_lowercase().as_str()) {
            if !current.is_empty() {
                phrases.push(NounPhrase {
                    text: current.join(" "),
                    start,
                    end: idx,
                });
                current.clear();
            }
            continue;
        }
        if current.is_empty() {
            start = idx;
        }
        current.push(token.to_string());
    }
    if !current.is_empty() {
        phrases.push(NounPhrase {
            text: current.join(" "),
            start,
            end: words.len(),
        });
    }
    phrases
}

fn build_augmented_query(
    original_query: &str,
    entities: &[QuerySpan],
    noun_phrases: &[String],
    verb_phrases: &[String],
    subject_hint: Option<&str>,
    object_hint: Option<&str>,
) -> String {
    let mut seen = HashSet::new();
    let mut terms = Vec::new();
    let mut push_term = |term: &str| {
        let normalized = normalize_for_index(term);
        if normalized.is_empty() || !seen.insert(normalized.clone()) {
            return;
        }
        terms.push(term.trim().to_string());
    };

    for entity in entities {
        push_term(&entity.text);
    }
    for phrase in noun_phrases {
        push_term(phrase);
    }
    for phrase in verb_phrases {
        push_term(phrase);
    }
    if let Some(subject) = subject_hint {
        push_term(subject);
    }
    if let Some(object) = object_hint {
        push_term(object);
    }

    if terms.is_empty() {
        original_query.to_string()
    } else {
        format!("{} {}", original_query.trim(), terms.join(" "))
    }
}

fn dedup_spans(spans: Vec<QuerySpan>) -> Vec<QuerySpan> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for span in spans {
        let key = format!("{:?}:{}", span.kind, span.normalized);
        if span.normalized.is_empty() || !seen.insert(key) {
            continue;
        }
        out.push(span);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_quantity_intent() {
        assert_eq!(
            quantity_intent("How many days did it take?"),
            Some(AggregateIntent::Count)
        );
        assert_eq!(
            quantity_intent("How much did it cost?"),
            Some(AggregateIntent::Sum)
        );
    }

    #[test]
    fn extracts_temporal_phrase() {
        let temporal = extract_temporal("What happened last Friday?");
        assert!(temporal.is_some());
        assert_eq!(temporal.unwrap().phrase.to_lowercase(), "last friday");
    }

    #[test]
    fn classifies_query_routing_intent() {
        let count = analyze_query("How many projects have I led?");
        assert_eq!(count.query_routing_intent, Some(QueryRoutingIntent::Count));

        let sum = analyze_query("How much did it cost in total?");
        assert_eq!(sum.query_routing_intent, Some(QueryRoutingIntent::Sum));

        let sequence = analyze_query("Which tasks happened in consecutive days?");
        assert_eq!(
            sequence.query_routing_intent,
            Some(QueryRoutingIntent::Sequence)
        );
    }

    #[test]
    fn augments_query_with_terms() {
        let aug = build_augmented_query(
            "what happened",
            &[QuerySpan {
                kind: QuerySpanKind::Entity,
                text: "Project Apollo".to_string(),
                normalized: "project apollo".to_string(),
                score: 1.0,
                source: "test".to_string(),
            }],
            &["Project Apollo".to_string()],
            &[],
            Some("Project Apollo"),
            None,
        );
        assert!(aug.contains("Project Apollo"));
    }
}
