//! Shared query/document tokenization for the primary lexical/rerank path
//! (`crate::index`) and the experimental segment-routing path
//! (`crate::segments`).
//!
//! The two callers intentionally use different rules. This is not
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
/// tokenized/normalized under the same mode. The two stopword lists use
/// different vocabularies (raw words vs. stemmed fragments) and are not
/// interchangeable.
pub fn is_stopword(token: &str, mode: TokenizerMode) -> bool {
    match mode {
        TokenizerMode::Unstemmed => {
            unstemmed_stopwords().contains(token) || korean_stopwords().contains(token)
        }
        TokenizerMode::Stemmed => {
            stemmed_stopwords().contains(token) || korean_stopwords().contains(token)
        }
    }
}

fn unstemmed_tokens(input: &str) -> Vec<String> {
    static TOKEN_RE: OnceLock<Regex> = OnceLock::new();
    let token_re =
        TOKEN_RE.get_or_init(|| Regex::new(r"[A-Za-z][A-Za-z0-9_-]{2,}").expect("valid regex"));
    // Script-aware single pass: Latin runs keep the exact historical regex
    // behavior (no regex match can span a Han/Hangul char, so segmenting at
    // script boundaries is byte-identical for Latin); Han runs emit
    // bigrams; Hangul runs emit the eojeol plus a particle-stripped stem.
    let mut out = Vec::new();
    let mut seg = String::new();
    let mut seg_script = Script::Latin;
    let mut started = false;
    for ch in input.chars() {
        let script = script_of(ch);
        if started && script != seg_script {
            push_unstemmed_run(&mut out, &token_re, seg_script, &seg);
            seg.clear();
        }
        seg.push(ch);
        seg_script = script;
        started = true;
    }
    if started {
        push_unstemmed_run(&mut out, &token_re, seg_script, &seg);
    }
    out
}

fn stemmed_tokens(input: &str) -> Vec<String> {
    // Same script-aware split as unstemmed: Han runs become bigrams and
    // Hangul runs become eojeol + stem directly (no Pinyin/romanization, no
    // English stemmer); other runs keep the historical split + normalize
    // behavior.
    let mut out = Vec::new();
    let mut seg = String::new();
    let mut seg_script = Script::Latin;
    let mut started = false;
    for ch in input.chars() {
        let script = script_of(ch);
        if started && script != seg_script {
            push_stemmed_run(&mut out, seg_script, &seg);
            seg.clear();
        }
        seg.push(ch);
        seg_script = script;
        started = true;
    }
    if started {
        push_stemmed_run(&mut out, seg_script, &seg);
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Script {
    Latin,
    Han,
    Hangul,
}

fn script_of(ch: char) -> Script {
    if crate::lang::is_han(ch) {
        Script::Han
    } else if crate::lang::is_hangul(ch) {
        Script::Hangul
    } else {
        Script::Latin
    }
}

fn push_unstemmed_run(
    out: &mut Vec<String>,
    token_re: &Regex,
    script: Script,
    seg: &str,
) {
    match script {
        Script::Latin => {
            out.extend(token_re.find_iter(seg).map(|m| m.as_str().to_lowercase()));
        }
        Script::Han => push_han_bigrams(out, &seg.chars().collect::<Vec<_>>()),
        Script::Hangul => out.extend(hangul_eojeol_tokens(seg)),
    }
}

fn push_stemmed_run(out: &mut Vec<String>, script: Script, seg: &str) {
    match script {
        Script::Latin => {
            out.extend(
                seg.split(|ch: char| !ch.is_alphanumeric())
                    .map(normalize_for_index)
                    .filter(|token| token.len() > 1),
            );
        }
        Script::Han => push_han_bigrams(out, &seg.chars().collect::<Vec<_>>()),
        Script::Hangul => out.extend(hangul_eojeol_tokens(seg)),
    }
}

/// Sliding character bigrams for one Han run: 我毕业 -> 我毕, 毕业.
/// A lone character is emitted as-is so single-character queries match.
fn push_han_bigrams(out: &mut Vec<String>, run: &[char]) {
    if run.len() == 1 {
        out.push(run[0].to_string());
        return;
    }
    for w in run.windows(2) {
        out.push(w.iter().collect());
    }
}

/// Sliding character bigrams over every Han run in `text`, in order.
/// Shared by the tantivy CJK tokenizer and the BM25 query fallback so
/// index-time and query-time segmentation agree.
pub(crate) fn han_bigrams(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut run: Vec<char> = Vec::new();
    for ch in text.chars() {
        if crate::lang::is_han(ch) {
            run.push(ch);
        } else if !run.is_empty() {
            push_han_bigrams(&mut out, &run);
            run.clear();
        }
    }
    if !run.is_empty() {
        push_han_bigrams(&mut out, &run);
    }
    out
}

/// Tokens for one Hangul run (one eojeol — spaces break runs): the eojeol
/// itself plus a particle-stripped stem when stripping changes it, so
/// `학교에` matches a query for `학교` and vice versa.
pub(crate) fn hangul_eojeol_tokens(eojeol: &str) -> Vec<String> {
    let mut out = vec![eojeol.to_string()];
    let stem = strip_korean_particles(eojeol);
    if stem != eojeol {
        out.push(stem);
    }
    out
}

/// Korean particles / case markers / endings, longest first. Up to two
/// suffixes are stripped per eojeol (`학교에서는` -> `학교`); a strip never
/// consumes the whole word.
///
/// Recall-oriented and deliberately simple: both the eojeol and the stem
/// are indexed, so exact matches always work and over-stripping (e.g.
/// `아이` -> `아`) only adds a noisy extra token that single-character
/// queries could match — no morphological analyzer required.
const KOREAN_SUFFIXES: &[&str] = &[
    "에게서", "한테서", // 3-char
    "에서", "에게", "한테", "께서", "부터", "까지", "처럼", "마저", "조차", "으로", "하고", "이랑",
    "이나", // 2-char
    "은", "는", "이", "가", "을", "를", "에", "의", "도", "만", "로", "와", "과", "랑", "나", "야",
    "아", // 1-char
];

fn strip_korean_particles(eojeol: &str) -> String {
    let mut stem = eojeol.to_string();
    for _ in 0..2 {
        match strip_one_korean_suffix(&stem) {
            Some(s) => stem = s,
            None => break,
        }
    }
    stem
}

fn strip_one_korean_suffix(word: &str) -> Option<String> {
    let n = word.chars().count();
    for suf in KOREAN_SUFFIXES {
        let m = suf.chars().count();
        if n > m && word.ends_with(suf) {
            return Some(word.chars().take(n - m).collect());
        }
    }
    None
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

/// Korean function words and particles. Hangul tokens can never collide
/// with the ASCII stopword lists, so this set is unioned into both modes'
/// checks. Interrogatives are included for term statistics; question
/// focus deliberately does not filter stopwords, so they still work there.
fn korean_stopwords() -> &'static HashSet<&'static str> {
    static STOP: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STOP.get_or_init(|| {
        [
            // Particles / case markers.
            "은", "는", "이", "가", "을", "를", "에", "의", "와", "과", "도", "만", "로", "으로",
            "에서", "에게", "한테", "부터", "까지", "처럼", "이랑", "랑", "하고", "나", "야", "아",
            // Demonstratives, bound nouns, quantifiers.
            "그", "저", "것", "거", "수", "등", "및", "한", "두", "세", "더", "또",
            // Negation / adverbs.
            "안", "못", "잘",
            // Pronouns.
            "나", "너", "우리", "저희", "자기", "제", "내", "네",
            // Interrogatives (kept for term stats; focus sees them anyway).
            "뭐", "왜", "언제", "어디", "누구", "누가", "무엇", "무슨", "어떤", "어떻게", "얼마나",
            "얼마", "어느",
            // Deictic adverbs.
            "이렇게", "그렇게", "저렇게", "이런", "그런", "저런", "모든",
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

    #[test]
    fn korean_stopwords_apply_to_both_modes() {
        for word in ["은", "는", "에", "그", "것", "안"] {
            assert!(
                is_stopword(word, TokenizerMode::Unstemmed),
                "{word} should be a stopword"
            );
            assert!(
                is_stopword(word, TokenizerMode::Stemmed),
                "{word} should be a stopword"
            );
        }
        assert!(!is_stopword("학교", TokenizerMode::Unstemmed));
    }

    #[test]
    fn unstemmed_tokenizes_korean_eojeol_with_stem() {
        // 학교에 -> eojeol + stripped stem 학교. 갔다 has no particle
        // suffix (다 is a verb ending, not stripped) -> eojeol only.
        assert_eq!(
            tokenize("학교에 갔다", TokenizerMode::Unstemmed),
            vec!["학교에", "학교", "갔다"],
        );
    }

    #[test]
    fn particle_stripping_cases() {
        assert_eq!(strip_korean_particles("학교에"), "학교");
        assert_eq!(strip_korean_particles("학교에서"), "학교");
        assert_eq!(strip_korean_particles("학교에서는"), "학교");
        assert_eq!(strip_korean_particles("친구와"), "친구");
        assert_eq!(strip_korean_particles("책을"), "책");
        assert_eq!(strip_korean_particles("나의"), "나");
        // Never strips to empty.
        assert_eq!(strip_korean_particles("이"), "이");
        assert_eq!(strip_korean_particles("에"), "에");
        // No suffix: unchanged.
        assert_eq!(strip_korean_particles("학교"), "학교");
    }

    #[test]
    fn han_bigrams_match_zh_convention() {
        assert_eq!(han_bigrams("清华大学"), vec!["清华", "华大", "大学"]);
        assert_eq!(han_bigrams("中"), vec!["中"]);
    }

    #[test]
    fn mixed_script_tokenization() {
        let toks = tokenize("我在学习Rust 학교에", TokenizerMode::Unstemmed);
        assert!(toks.contains(&"rust".to_string()));
        assert!(toks.contains(&"학교에".to_string()));
        assert!(toks.contains(&"학교".to_string()));
        // Han bigrams present.
        assert!(toks.iter().any(|t| t.chars().all(crate::lang::is_han)));
    }
}
