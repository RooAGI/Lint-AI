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

    let question_word = tokens
        .iter()
        .map(|t| stem_token(t))
        .find(|s| is_question_word(s))
        .and_then(|s| {
            // Return the original unstemmed form
            tokens.iter().find(|t| stem_token(t) == s).cloned()
        });

    let mut focus_terms = Vec::new();
    let mut constraint_terms = Vec::new();

    for token in tokens {
        let stemmed = stem_token(&token);
        if is_question_word(&stemmed) {
            continue;
        }
        // Skip stopwords (they're structure, not focus or constraints)
        if crate::tokenizer::is_stopword(&stemmed, crate::tokenizer::TokenizerMode::Stemmed) {
            continue;
        }
        if crate::query_expansion::is_expandable_concept(&stemmed) {
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
}
