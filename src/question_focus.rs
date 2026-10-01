//! Question focus identification.
//!
//! For a question like "What did John receive a certificate for?", identifies:
//! - question_word: "what"
//! - focus_terms: ["certif"] (what the question is about)
//! - constraint_terms: ["john"] (filters, not the focus)
//!
//! The focus is the expandable concept terms. Constraints are names,
//! pronouns, and other non-expandable terms.
//!
//! This lives in lint-ai (not behood) because question structure
//! understanding is the caller's job. Behood judges entities; lint-ai
//! understands questions.

use std::collections::HashSet;

use crate::lang::Lang;

#[derive(Debug, Clone)]
pub struct QuestionFocus {
    /// The interrogative word (what, who, where, when, which, etc.), if any.
    pub question_word: Option<String>,
    /// What the question is about (expandable concepts).
    pub focus_terms: Vec<String>,
    /// Filters and context (names, pronouns, etc.).
    pub constraint_terms: Vec<String>,
}

/// Identify the focus of a question.
///
/// Uses the same expandable-concept classification as query expansion:
/// terms that can be meaningfully expanded (not question words, pronouns,
/// generic verbs, or common names) are the focus.
pub fn identify_focus(query: &str) -> QuestionFocus {
    // Use unstemmed tokens for the output (Luyi: no stemming for focus words).
    // We still stem each token internally to check against the stemmed
    // exclusion lists (question words, stopwords, expandable concepts).
    let tokens = tokenize_unstemmed(query);
    let lang = Lang::Auto.resolve(query);

    // Spanish interrogatives are checked on the RAW token (accents intact):
    // stemming deunicodes "qué" to "que", which is indistinguishable from
    // the relative pronoun "que" — the accent is the disambiguator.
    let question_word = tokens
        .iter()
        .find(|t| is_spanish_question_word(t))
        .cloned()
        .or_else(|| {
            tokens
                .iter()
                .map(|t| stem_token(t))
                .find(|s| is_question_word(s))
                .and_then(|s| {
                    // Return the original unstemmed form
                    tokens.iter().find(|t| stem_token(t) == s).cloned()
                })
                // Chinese interrogatives are matched as substrings (tokens are
                // bigrams, so single characters like 谁/哪 would never match a
                // token).
                .or_else(|| chinese_question_word(query))
        });

    let mut focus_terms = Vec::new();
    let mut constraint_terms = Vec::new();

    for token in tokens {
        let stemmed = stem_token(&token);
        if is_question_word(&stemmed) || is_chinese_interrogative_token(&token) {
            continue;
        }
        // Skip stopwords (they're structure, not focus or constraints)
        if crate::tokenizer::is_stopword_for_lang(
            &stemmed,
            crate::tokenizer::TokenizerMode::Stemmed,
            lang,
        ) {
            continue;
        }
        if crate::query_expansion::is_expandable_concept(&stemmed, lang) {
            focus_terms.push(token);
        } else {
            constraint_terms.push(token);
        }
    }

    // Deduplicate while preserving order
    let focus_terms = dedup(focus_terms);
    let constraint_terms = dedup(constraint_terms);

    QuestionFocus {
        question_word,
        focus_terms,
        constraint_terms,
    }
}

pub(crate) fn is_question_word(term: &str) -> bool {
    matches!(
        term,
        // English interrogatives.
        "what" | "when" | "where" | "who" | "whom" | "whos" | "why" | "how" | "which" | "would"
        // Korean interrogatives ( Hangul runs are not stemmed, so match
        // the common surface forms; the tokenizer also emits
        // particle-stripped stems, which are listed alongside).
        | "누구" | "누가" | "누구를" | "누구의"        // who
        | "무엇" | "뭐" | "무슨" | "무엇을" | "무엇이"  // what
        | "어디" | "어디에" | "어디서"                // where
        | "언제"                                     // when
        | "얼마" | "얼마나"                           // how much / many
        | "왜"                                       // why
        | "어떻게" | "어떡해"                         // how
        | "어느" | "어떤"                            // which
    )
}

/// Chinese interrogative words. The scan in [`chinese_question_word`]
/// picks the earliest match (ties broken by longer match), so list order
/// does not matter.
const CHINESE_QUESTION_WORDS: &[&str] = &[
    "为什么",
    "怎么样",
    "什么样",
    "多少",
    "什么",
    "怎么",
    "怎样",
    "如何",
    "为何",
    "哪里",
    "哪儿",
    "何时",
    "何处",
    "何地",
    "谁",
    "哪",
    "几",
    "啥",
    "吗",
    "呢",
];

/// Find the Chinese interrogative in `query`, if any. Returns the earliest
/// match (ties broken by longer match), so the question word reflects
/// where the question is asked.
pub(crate) fn chinese_question_word(query: &str) -> Option<String> {
    let mut best: Option<(usize, &str)> = None;
    for word in CHINESE_QUESTION_WORDS {
        if let Some(pos) = query.find(word) {
            let replace = match best {
                None => true,
                Some((best_pos, best_word)) => {
                    pos < best_pos || (pos == best_pos && word.len() > best_word.len())
                }
            };
            if replace {
                best = Some((pos, word));
            }
        }
    }
    best.map(|(_, w)| w.to_string())
}

/// True if a Chinese token is or contains an interrogative word.
/// Tokens are character bigrams, so this checks exact multi-character
/// matches plus single-character interrogatives (谁/哪/几/啥/吗/呢/何).
fn is_chinese_interrogative_token(token: &str) -> bool {
    const SINGLE: &[char] = &['谁', '哪', '几', '啥', '吗', '呢', '何'];
    CHINESE_QUESTION_WORDS.iter().any(|w| token == *w) || token.chars().any(|c| SINGLE.contains(&c))
}

/// Spanish interrogatives, checked against the RAW (accented) token —
/// never the stemmed form, where "qué" and the relative pronoun "que"
/// are indistinguishable. Unaccented "que" is deliberately excluded:
/// without the accent it cannot be told apart from the relative pronoun.
fn is_spanish_question_word(token: &str) -> bool {
    matches!(
        token,
        "qué"
            | "quién"
            | "quiénes"
            | "cuál"
            | "cuáles"
            | "dónde"
            | "cuándo"
            | "cuánto"
            | "cuánta"
            | "cuántos"
            | "cuántas"
            | "cómo"
    )
}

fn dedup(terms: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    terms
        .into_iter()
        .filter(|t| seen.insert(t.clone()))
        .collect()
}

/// Simple stemmed tokenization for focus identification.
/// Uses the same tokenizer as the query path, but does NOT filter stopwords
/// (we need the question words like "what", "when", etc.).
fn tokenize_unstemmed(query: &str) -> Vec<String> {
    use crate::tokenizer::{tokenize, TokenizerMode};
    tokenize(query, TokenizerMode::Unstemmed)
}

/// Stem a single token for exclusion-list checks.
fn stem_token(token: &str) -> String {
    use crate::tokenizer::{tokenize, TokenizerMode};
    // Tokenize in stemmed mode to get the stemmed form for checks
    tokenize(token, TokenizerMode::Stemmed)
        .into_iter()
        .next()
        .unwrap_or_else(|| token.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn korean_question_words_recognized() {
        for w in [
            "누구", "누가", "무엇", "뭐", "무슨", "어디", "언제", "얼마", "얼마나", "왜", "어떻게",
            "어느", "어떤",
        ] {
            assert!(is_question_word(w), "{w} should be a question word");
        }
        assert!(!is_question_word("학교"));
    }

    #[test]
    fn certificate_question_focus_is_certificate() {
        let focus = identify_focus("What did John receive a certificate for?");
        assert_eq!(focus.question_word, Some("what".to_string()));
        assert!(
            focus.focus_terms.contains(&"certificate".to_string()),
            "focus should contain 'certificate', got {:?}",
            focus.focus_terms
        );
        assert!(
            focus.constraint_terms.contains(&"john".to_string()),
            "constraints should contain 'john', got {:?}",
            focus.constraint_terms
        );
        assert!(
            !focus.focus_terms.contains(&"john".to_string()),
            "'john' should not be in focus"
        );
    }

    #[test]
    fn when_question_has_no_explicit_focus() {
        let focus = identify_focus("When did Melanie's family go on a roadtrip?");
        assert_eq!(focus.question_word, Some("when".to_string()));
        // "roadtrip" is the focus (what we're asking about the time of)
        assert!(
            focus.focus_terms.contains(&"roadtrip".to_string()),
            "focus should contain 'roadtrip', got {:?}",
            focus.focus_terms
        );
    }

    #[test]
    fn chinese_what_question_detects_interrogative() {
        let focus = identify_focus("我毕业于哪所大学？");
        assert_eq!(focus.question_word, Some("哪".to_string()));
        // Interrogative bigrams are not focus terms.
        assert!(
            !focus.focus_terms.iter().any(|t| t.contains('哪')),
            "interrogative should not be focus, got {:?}",
            focus.focus_terms
        );
    }

    #[test]
    fn spanish_where_question_focus_is_biblioteca() {
        // Accented interrogative detected on the raw token (stemming would
        // conflate "qué" with the relative pronoun "que").
        let focus = identify_focus("¿Dónde está la biblioteca?");
        assert_eq!(focus.question_word, Some("dónde".to_string()));
        assert!(
            focus.focus_terms.contains(&"biblioteca".to_string()),
            "focus should contain 'biblioteca', got {:?}",
            focus.focus_terms
        );
        // "está"/"la" are Spanish stopwords: neither focus nor constraint.
        assert!(
            !focus.focus_terms.iter().any(|t| t == "está" || t == "la"),
            "stopwords leaked into focus: {:?}",
            focus.focus_terms
        );
    }

    #[test]
    fn chinese_why_question_prefers_longest_earliest() {
        let focus = identify_focus("你为什么学习中文？");
        assert_eq!(focus.question_word, Some("为什么".to_string()));
    }

    #[test]
    fn chinese_how_many_question() {
        let focus = identify_focus("你买了多少本书？");
        assert_eq!(focus.question_word, Some("多少".to_string()));
    }

    #[test]
    fn chinese_non_question_has_no_question_word() {
        let focus = identify_focus("我毕业于清华大学。");
        assert_eq!(focus.question_word, None);
    }

    #[test]
    fn spanish_relative_que_is_not_a_question_word() {
        // No accent, no interrogative: "que" here is a relative pronoun.
        let focus = identify_focus("El libro que compré ayer");
        assert_eq!(focus.question_word, None);
    }
}
