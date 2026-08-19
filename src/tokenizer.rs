//! Shared query/document tokenization for the primary lexical/rerank path
//! (`crate::index`) and the experimental segment-routing path
//! (`crate::segments`).
//!
//! The two callers intentionally use different rules — this is not
//! duplication to collapse into one behavior. A LongMemEval-S benchmark
//! (500 scoped queries for the primary path, 133 multi-session queries for
//! segment routing) showed each mode is a real, measured improvement for
//! its own caller and a regression for the other:
//!
//! - Switching segment routing from `Stemmed` to `Unstemmed` cost ~5pp of
//!   recall@5 and ~2pp of MRR on segment-routing variants.
//! - Switching the primary path from `Unstemmed` to `Stemmed` cost
//!   ~0.1-0.3pp across recall/MRR/NDCG.
//!
//! Keep both modes and pick per caller; don't unify them.

use crate::query_expansion::normalize_for_index;
use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerMode {
    /// Regex-bounded terms (`[A-Za-z][A-Za-z0-9_-]{2,}`, min length 3),
    /// lowercased, not stemmed. Used by `crate::index`'s lexical/rerank
    /// path.
    Unstemmed,
    /// Terms split on non-alphanumeric boundaries (min length 2), each
    /// stemmed with an English Porter stemmer via
    /// [`normalize_for_index`]. Used by `crate::segments`'s routing path.
    Stemmed,
}

/// Tokenizes `input` according to `mode`. Order matches input order and
/// duplicates are preserved; callers that need a set should collect into
/// one (as `crate::segments::query_tokens` does).
pub fn tokenize(input: &str, mode: TokenizerMode) -> Vec<String> {
    match mode {
        TokenizerMode::Unstemmed => unstemmed_tokens(input),
        TokenizerMode::Stemmed => stemmed_tokens(input),
    }
}

/// True if `token` is a stopword under `mode`. `token` must already be
/// tokenized/normalized under the same mode — the two stopword lists use
/// different vocabularies (raw words vs. stemmed fragments) and are not
/// interchangeable.
pub fn is_stopword(token: &str, mode: TokenizerMode) -> bool {
    match mode {
        TokenizerMode::Unstemmed => unstemmed_stopwords().contains(token),
        TokenizerMode::Stemmed => stemmed_stopwords().contains(token),
    }
}

fn unstemmed_tokens(input: &str) -> Vec<String> {
    static TOKEN_RE: OnceLock<Regex> = OnceLock::new();
    let token_re =
        TOKEN_RE.get_or_init(|| Regex::new(r"[A-Za-z][A-Za-z0-9_-]{2,}").expect("valid regex"));
    token_re
        .find_iter(input)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

fn stemmed_tokens(input: &str) -> Vec<String> {
    input
        .split(|ch: char| !ch.is_alphanumeric())
        .map(normalize_for_index)
        .filter(|token| token.len() > 1)
        .collect()
}

fn unstemmed_stopwords() -> &'static HashSet<&'static str> {
    static STOP: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STOP.get_or_init(|| {
        [
            "how",
            "many",
            "much",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "did",
            "does",
            "have",
            "has",
            "had",
            "been",
            "being",
            "was",
            "were",
            "are",
            "the",
            "and",
            "or",
            "for",
            "from",
            "with",
            "that",
            "this",
            "these",
            "those",
            "currently",
            "recently",
            "past",
            "last",
            "next",
            "into",
            "onto",
            "about",
            "after",
            "before",
            "over",
            "under",
            "between",
            "during",
            "i",
            "you",
            "we",
            "they",
        ]
        .into_iter()
        .collect()
    })
}

fn stemmed_stopwords() -> &'static HashSet<&'static str> {
    static STOP: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STOP.get_or_init(|| {
        [
            "a", "an", "and", "are", "can", "did", "do", "doe", "for", "from", "had", "have",
            "how", "i", "in", "is", "it", "many", "mani", "me", "my", "of", "on", "or", "that",
            "the", "thi", "this", "to", "wa", "what", "when", "where", "which", "who", "with",
            "you",
        ]
        .into_iter()
        .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unstemmed_matches_manual_regex() {
        let re = Regex::new(r"[A-Za-z][A-Za-z0-9_-]{2,}").unwrap();
        let samples = [
            "What degree did I graduate with?",
            "How many miles did I run last week?",
            "I cooked pasta and watched a movie.",
        ];
        for s in samples {
            let expected: Vec<String> =
                re.find_iter(s).map(|m| m.as_str().to_lowercase()).collect();
            let actual = tokenize(s, TokenizerMode::Unstemmed);
            assert_eq!(expected, actual, "mismatch for {s:?}");
        }
    }

    #[test]
    fn unstemmed_stopword_matches_original_list() {
        for word in ["how", "many", "does", "was", "the", "and"] {
            assert!(
                is_stopword(word, TokenizerMode::Unstemmed),
                "{word} should be a stopword"
            );
        }
        for word in ["degree", "graduate", "miles", "pasta"] {
            assert!(
                !is_stopword(word, TokenizerMode::Unstemmed),
                "{word} should not be a stopword"
            );
        }
    }

    #[test]
    fn stemmed_stopword_matches_segment_router_list() {
        for word in ["doe", "mani", "thi", "wa", "what"] {
            assert!(
                is_stopword(word, TokenizerMode::Stemmed),
                "{word} should be a stopword"
            );
        }
        for word in ["degre", "graduat", "mile", "pasta"] {
            assert!(
                !is_stopword(word, TokenizerMode::Stemmed),
                "{word} should not be a stopword"
            );
        }
    }
}
