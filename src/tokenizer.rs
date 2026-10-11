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

use crate::lang::Lang;
use crate::query_expansion::normalize_for_index;
use regex::Regex;
use rust_stemmers::{Algorithm, Stemmer};
use std::collections::HashSet;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerMode {
    /// Regex-bounded Latin terms (`[A-Za-z][A-Za-z0-9_-]{2,}`, min length 3),
    /// lowercased, not stemmed; Han runs become sliding character bigrams;
    /// Hangul runs emit the eojeol plus a particle-stripped stem. Used by
    /// `crate::index`'s lexical/rerank path.
    Unstemmed,
    /// Terms split on non-alphanumeric boundaries (min length 2), each
    /// stemmed with an English Porter stemmer via
    /// [`normalize_for_index`]; Han runs become character bigrams and
    /// Hangul runs become eojeol + stem with no stemming. Used by
    /// `crate::segments`'s routing path.
    Stemmed,
}

/// Latin letter class for token regexes: ASCII plus accented Latin
/// (Latin-1 Supplement U+00C0–U+00FF and Latin Extended-A U+0100–U+017F),
/// so Spanish/French/etc. words tokenize as units ("niño" is one token,
/// not "ni"). Pure-ASCII input matches byte-identically to `[A-Za-z]`.
pub(crate) const LATIN_LETTER: &str = r"A-Za-zÀ-ÿĀ-ſ";

/// Strip diacritics from a (lowercased) Latin token: NFD decomposition
/// followed by removal of combining marks. `niño` -> `nino`,
/// `dónde` -> `donde`. Pure-ASCII input is returned unchanged
/// (byte-identical, via the fast path below).
///
/// Used for dual emission (see [`unstemmed_tokens`] and
/// `crate::index::latin_tokenizer`): both the original and the folded
/// form are indexed and queried, so unaccented queries match accented
/// text while exact matches still rank higher. Only combining marks are
/// removed — `ß`, `ø`, `ł` keep their identity (this is not full
/// ASCII-folding).
pub fn fold_diacritics(s: &str) -> String {
    if s.is_ascii() {
        return s.to_string();
    }
    s.nfd().filter(|c| !is_combining_mark(*c)).collect()
}

/// True for Unicode combining marks (diacritics) — the marks that NFD
/// decomposition separates from their base letters.
fn is_combining_mark(c: char) -> bool {
    matches!(c,
        '\u{300}'..='\u{36F}'   // Combining Diacritical Marks
        | '\u{1AB0}'..='\u{1AFF}' // Combining Diacritical Marks Extended
        | '\u{1DC0}'..='\u{1DFF}' // Combining Diacritical Marks Supplement
        | '\u{20D0}'..='\u{20FF}' // Combining Diacritical Marks for Symbols
        | '\u{FE20}'..='\u{FE2F}' // Combining Half Marks
    )
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
            unstemmed_stopwords().contains(token)
                || korean_stopwords().contains(token)
                || chinese_stopwords().contains(token)
        }
        TokenizerMode::Stemmed => {
            stemmed_stopwords().contains(token)
                || korean_stopwords().contains(token)
                || chinese_stopwords().contains(token)
        }
    }
}

/// True if `token` is a Chinese function word. Shared with
/// `crate::query_expansion` (focus classification) and `crate::tier1`.
fn unstemmed_tokens(input: &str) -> Vec<String> {
    static TOKEN_RE: OnceLock<Regex> = OnceLock::new();
    let token_re = TOKEN_RE.get_or_init(|| {
        Regex::new(&format!(r"[{L}][{L}0-9_\-]{{2,}}", L = LATIN_LETTER)).expect("valid regex")
    });
    // Script-aware single pass: Latin runs keep the exact historical regex
    // behavior (no regex match can span a Han/Hangul char, so segmenting at
    // script boundaries is byte-identical for Latin); Han runs emit
    // bigrams; Hangul runs emit the eojeol plus a particle-stripped stem.
    // Latin runs dual-emit the original and the diacritic-folded form
    // (see `push_unstemmed_run`).
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

fn push_unstemmed_run(out: &mut Vec<String>, token_re: &Regex, script: Script, seg: &str) {
    match script {
        Script::Latin => {
            for m in token_re.find_iter(seg) {
                let lowered = m.as_str().to_lowercase();
                // Dual emission for accent-insensitive matching: the
                // original form plus the diacritic-folded form (when
                // different). Index and query both emit both, so `nino`
                // matches a doc containing `niño`, while a query for `niño`
                // matches two terms in an exact doc vs one in a folded-only
                // doc — exact matches rank higher with no boost machinery.
                // Pure-ASCII tokens emit once (byte-identical to before).
                let folded = fold_diacritics(&lowered);
                out.push(lowered);
                if folded != *out.last().unwrap() {
                    out.push(folded);
                }
            }
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
    // Emit each character and each sliding bigram, interleaved by position:
    // c1, c1c2, c2, c2c3, ..., cn. The unigrams let a single-character query
    // term (e.g. 猫) match that character inside an indexed word (e.g. 橘猫);
    // with bigrams alone a unigram query can never hit the index. Interleaving
    // keeps each unigram adjacent to its bigrams so term-rank position scores
    // treat them fairly. Single-character runs stay unigrams (above).
    for (i, ch) in run.iter().enumerate() {
        out.push(ch.to_string());
        if i + 1 < run.len() {
            let mut bigram = String::with_capacity(ch.len_utf8() * 2 + 1);
            bigram.push(*ch);
            bigram.push(run[i + 1]);
            out.push(bigram);
        }
    }
}

/// Interleaved character unigrams and sliding bigrams over every Han run
/// in `text`, in order (c1, c1c2, c2, c2c3, ..., cn). Shared by the tantivy
/// CJK tokenizer and the BM25 query fallback so index-time and query-time
/// segmentation agree.
pub(crate) fn han_tokens(text: &str) -> Vec<String> {
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
    "에게서",
    "한테서", // 3-char
    "에서",
    "에게",
    "한테",
    "께서",
    "부터",
    "까지",
    "처럼",
    "마저",
    "조차",
    "으로",
    "하고",
    "이랑",
    "이나", // 2-char
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "의",
    "도",
    "만",
    "로",
    "와",
    "과",
    "랑",
    "나",
    "야",
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

/// Canonical English stopwords: the vendored spaCy `en` list
/// (`crate::stopwords_data::STOPWORDS_EN`, MIT). Replaces the old ~46-word
/// hand-built list. Shared with the tier-1 term ranker (`crate::tier1`).
pub(crate) fn english_stopwords() -> &'static HashSet<&'static str> {
    static STOP: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STOP.get_or_init(|| {
        crate::stopwords_data::STOPWORDS_EN
            .iter()
            .copied()
            .collect()
    })
}

fn unstemmed_stopwords() -> &'static HashSet<&'static str> {
    english_stopwords()
}

fn stemmed_stopwords() -> &'static HashSet<String> {
    static STOP: OnceLock<HashSet<String>> = OnceLock::new();
    STOP.get_or_init(|| {
        // Systematic: run the same English stemmer the Stemmed token path
        // uses over the canonical English list, so every stemmed stopword
        // is exactly what the tokenizer would emit for that word.
        let stemmer = Stemmer::create(Algorithm::English);
        crate::stopwords_data::STOPWORDS_EN
            .iter()
            .map(|w| stemmer.stem(w).into_owned())
            .collect()
    })
}

/// Korean function words and particles. Hangul tokens can never collide
/// with the ASCII stopword lists, so this set is unioned into both modes'
/// checks. Interrogatives are included for term statistics; question
/// focus deliberately does not filter stopwords, so they still work there.
/// Union of the hand-built list and the vendored spaCy `ko` list
/// (`crate::stopwords_data::STOPWORDS_KO`, MIT).
/// Shared with the tier-1 term ranker (`crate::tier1`).
pub(crate) fn korean_stopwords() -> &'static HashSet<&'static str> {
    static STOP: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STOP.get_or_init(|| {
        let mut set: HashSet<&'static str> = [
            // Particles / case markers.
            "은",
            "는",
            "이",
            "가",
            "을",
            "를",
            "에",
            "의",
            "와",
            "과",
            "도",
            "만",
            "로",
            "으로",
            "에서",
            "에게",
            "한테",
            "부터",
            "까지",
            "처럼",
            "이랑",
            "랑",
            "하고",
            "나",
            "야",
            "아",
            // Demonstratives, bound nouns, quantifiers.
            "그",
            "저",
            "것",
            "거",
            "수",
            "등",
            "및",
            "한",
            "두",
            "세",
            "더",
            "또",
            // Negation / adverbs.
            "안",
            "못",
            "잘",
            // Pronouns.
            "나",
            "너",
            "우리",
            "저희",
            "자기",
            "제",
            "내",
            "네",
            // Interrogatives (kept for term stats; focus sees them anyway).
            "뭐",
            "왜",
            "언제",
            "어디",
            "누구",
            "누가",
            "무엇",
            "무슨",
            "어떤",
            "어떻게",
            "얼마나",
            "얼마",
            "어느",
            // Deictic adverbs.
            "이렇게",
            "그렇게",
            "저렇게",
            "이런",
            "그런",
            "저런",
            "모든",
        ]
        .into_iter()
        .collect();
        set.extend(crate::stopwords_data::STOPWORDS_KO.iter().copied());
        set
    })
}

/// Chinese function words (particles, prepositions, conjunctions,
/// pronouns, modals). Tokens are character bigrams, so the list holds
/// single characters (for lone-character tokens) and common function
/// bigrams. Interrogatives are deliberately excluded — they are detected
/// separately by `crate::question_focus`.
/// Union of the hand-built list and the vendored spaCy `zh` list
/// (`crate::stopwords_data::STOPWORDS_ZH`, MIT; Han-only words, so the
/// two-character entries match bigram tokens directly).
/// Shared with the tier-1 term ranker (`crate::tier1`).
pub(crate) fn chinese_stopwords() -> &'static HashSet<&'static str> {
    static STOP: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STOP.get_or_init(|| {
        let mut set: HashSet<&'static str> = [
            // Single-character function words.
            "的", "了", "着", "过", "在", "是", "有", "和", "与", "或", "但", "而", "就", "都",
            "也", "很", "不", "没", "非", "未", "别", "我", "你", "他", "她", "它", "这", "那",
            "个", "为", "对", "从", "到", "向", "往", "及", "比", "被", "把", "将", "会", "可",
            "应", "能", "够", "以", "之", "其", "些", "每", "各", "该", "此", "若", "如", "乃",
            "则", "然", "故", "因", "虽", "即", "既", "亦", "又", "再", "更", "最", "太", "吗",
            "呢", "吧", "啊", "呀", "哇", "哦", "嗯", // Pronouns and demonstratives.
            "我们", "你们", "他们", "她们", "它们", "我的", "你的", "他的", "她的", "它的", "这是",
            "那是", "这个", "那个", "这些", "那些", "这里", "那里", "这种", "那种", "这样", "那样",
            // Conjunctions.
            "然后", "但是", "因为", "所以", "如果", "虽然", "还是", "或者", "以及", "并且", "而且",
            "不过", "然而", "于是", "因此", "其实", "比如", "例如",
            // Prepositions / coverbs.
            "关于", "对于", "由于", "随着", "通过", "作为", // Modals and auxiliaries.
            "可以", "应该", "必须", "能够", "可能", // Common function bigrams.
            "的是", "在了", "有了", "是的", "的话", "之一", "之间", "之中", "以内", "以外", "以前",
            "以后", "之前", "之后", "正在", "已经", "曾经",
        ]
        .into_iter()
        .collect();
        set.extend(crate::stopwords_data::STOPWORDS_ZH.iter().copied());
        // Current-state deictics are signal for this system, not noise:
        // they anchor the presently-true fact ("现在每天开着上下班").
        // spaCy lists them as stopwords (right for parsing, wrong for
        // current-state retrieval), so they are carved back out. This set
        // is defined by the test suite — the executable spec of measured
        // retrieval — not by hand: every word here is required as a ranked
        // term by a passing test.
        set.retain(|w| !CHINESE_CURRENT_STATE_KEEP.contains(w));
        set
    })
}

/// Words carved out of [`chinese_stopwords`]: deictic markers of current
/// state that spaCy lists as stopwords but this system retrieves on.
/// Required as ranked terms by
/// `tier1::cjk_term_tests::chinese_generous_budget_keeps_late_payload_terms`.
const CHINESE_CURRENT_STATE_KEEP: &[&str] = &["现在"];

/// Spanish function words: the vendored spaCy `es` list
/// (`crate::stopwords_data::STOPWORDS_ES`, MIT) plus mechanical
/// diacritic-folded twins (`está`/`esta`), because dual emission means
/// ranker/query tokens carry both forms. Spanish words can collide with
/// English ones (`no`, `son`, `era`), so unlike the CJK lists this set is
/// NOT unioned into the default `is_stopword` — callers must gate on
/// `Lang::Es`. Shared with the tier-1 term ranker (`crate::tier1`).
pub(crate) fn spanish_stopwords() -> &'static HashSet<String> {
    static STOP: OnceLock<HashSet<String>> = OnceLock::new();
    STOP.get_or_init(|| {
        let mut set: HashSet<String> = crate::stopwords_data::STOPWORDS_ES
            .iter()
            .map(|s| s.to_string())
            .collect();
        // Folded twins: re-inserting an unchanged (pure-ASCII) word is a
        // harmless no-op.
        set.extend(
            crate::stopwords_data::STOPWORDS_ES
                .iter()
                .map(|w| fold_diacritics(w)),
        );
        // Stemmed forms: query-term paths run through the English Porter
        // stemmer (via normalize_for_index), so the set must be closed
        // under stemming. Stemming is not idempotent ("adelante" ->
        // "adelant" -> "adel"), so iterate to a fixed point.
        let stemmer = Stemmer::create(Algorithm::English);
        loop {
            let stemmed: Vec<String> = set
                .iter()
                .map(|w| stemmer.stem(w).to_string())
                .filter(|s| !set.contains(s))
                .collect();
            if stemmed.is_empty() {
                break;
            }
            set.extend(stemmed);
        }
        set
    })
}

/// True if `token` is a stopword for `lang` under `mode`. English behavior
/// is unchanged (`is_stopword`); Spanish adds its function words on top.
/// `lang` must already be resolved — `Auto` falls back to English.
pub fn is_stopword_for_lang(token: &str, mode: TokenizerMode, lang: Lang) -> bool {
    if is_stopword(token, mode) {
        return true;
    }
    match lang {
        Lang::Es => spanish_stopwords().contains(token),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unstemmed_matches_manual_regex() {
        let re = Regex::new(&format!(r"[{L}][{L}0-9_\-]{{2,}}", L = LATIN_LETTER)).unwrap();
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
    fn unstemmed_chinese_emits_bigrams() {
        assert_eq!(
            tokenize("我毕业于清华大学", TokenizerMode::Unstemmed),
            vec![
                "我", "我毕", "毕", "毕业", "业", "业于", "于", "于清", "清", "清华", "华", "华大",
                "大", "大学", "学"
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn unstemmed_mixed_content_keeps_both() {
        // Latin regex behavior is unchanged around Han runs.
        assert_eq!(
            tokenize("我在学习Rust编程", TokenizerMode::Unstemmed),
            vec![
                "我", "我在", "在", "在学", "学", "学习", "习", "rust", "编", "编程", "程"
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn unstemmed_keeps_spanish_accents() {
        // Accented words must tokenize as units (previously "niño" yielded
        // zero tokens and "está" was truncated to "est"). Dual emission:
        // the original form plus the folded form.
        assert_eq!(
            tokenize("¿Dónde está la biblioteca?", TokenizerMode::Unstemmed),
            vec!["dónde", "donde", "está", "esta", "biblioteca"]
        );
        assert_eq!(
            tokenize("El niño juega", TokenizerMode::Unstemmed),
            vec!["niño", "nino", "juega"]
        );
    }

    #[test]
    fn fold_diacritics_strips_marks() {
        assert_eq!(fold_diacritics("niño"), "nino");
        assert_eq!(fold_diacritics("dónde"), "donde");
        assert_eq!(fold_diacritics("sí"), "si");
        assert_eq!(fold_diacritics("año"), "ano");
        assert_eq!(fold_diacritics("Ñoño"), "Nono"); // case preserved, marks stripped
                                                     // Pure ASCII is byte-identical (fast path).
        assert_eq!(fold_diacritics("siesta"), "siesta");
        // Not full ASCII-folding: ß/ø/ł keep their identity.
        assert_eq!(fold_diacritics("straße"), "straße");
        assert_eq!(fold_diacritics("søren"), "søren");
    }

    #[test]
    fn unstemmed_ascii_unchanged() {
        // Pure-ASCII input emits exactly one token per word, as before.
        assert_eq!(
            tokenize("The quick brown fox", TokenizerMode::Unstemmed),
            vec!["the", "quick", "brown", "fox"]
        );
    }

    #[test]
    fn unstemmed_single_han_char_is_kept() {
        assert_eq!(
            tokenize("天", TokenizerMode::Unstemmed),
            vec!["天".to_string()]
        );
    }

    #[test]
    fn stemmed_chinese_emits_bigrams_without_pinyin() {
        let tokens = tokenize("我毕业于清华大学", TokenizerMode::Stemmed);
        assert!(
            tokens.contains(&"清华".to_string()),
            "expected Han bigrams, got {tokens:?}"
        );
        assert!(
            !tokens
                .iter()
                .any(|t| t.chars().all(|c| c.is_ascii_alphabetic())),
            "no Pinyin transliteration expected, got {tokens:?}"
        );
    }

    #[test]
    fn unstemmed_stopwords_use_spacy_english() {
        // Canonical spaCy en list (vendored): the old hand-built words
        // still stop, plus spaCy-only function words.
        for word in [
            "how",
            "many",
            "does",
            "was",
            "the",
            "and",
            "however",
            "therefore",
            "among",
        ] {
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
    fn spanish_stopwords_cover_normalized_forms() {
        // Fixed-point check: every Spanish stopword, once run through the
        // index-time normalization, must land back in the set (or vanish).
        // Otherwise the normalized query-term paths would miss it.
        for word in spanish_stopwords().iter() {
            let normalized = normalize_for_index(word);
            for form in normalized.split_whitespace() {
                assert!(
                    spanish_stopwords().contains(form),
                    "normalized form {form:?} of {word:?} missing from Spanish stopwords"
                );
            }
        }
    }

    #[test]
    fn spanish_stopwords_do_not_leak_into_english() {
        // The per-language gate: English text never consults the Spanish
        // list, so Spanish-only surface forms stay live in English.
        // (Words like "no"/"son"/"era" are also English stopwords, so they
        // can't test the gate — use Spanish-only forms here.)
        for word in ["también", "dónde", "está", "niño", "biblioteca"] {
            assert!(
                !is_stopword(word, TokenizerMode::Unstemmed),
                "{word} must not be an English stopword"
            );
        }
        // But they ARE Spanish stopwords when Lang::Es is passed.
        for word in ["no", "son", "era", "tan", "la", "el"] {
            assert!(
                is_stopword_for_lang(word, TokenizerMode::Unstemmed, Lang::Es),
                "{word} should be a Spanish stopword"
            );
        }
        // Spanish-only forms are not filtered for English.
        for word in ["también", "dónde", "está"] {
            assert!(
                !is_stopword_for_lang(word, TokenizerMode::Unstemmed, Lang::En),
                "{word} must not be filtered for English"
            );
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
        // Stemmed with the SAME Snowball stemmer the Stemmed token path
        // uses, so every entry is exactly what the tokenizer emits.
        // (The old hand-built list carried dead "thi"/"wa" entries from a
        // mismatched Porter stemmer — the runtime stemmer emits "this"/"was".)
        for word in ["doe", "mani", "this", "was", "what", "howev", "therefor"] {
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
    fn chinese_stopwords_apply_to_both_modes() {
        for word in [
            "的", "了", "在", "是", "我们", "你们", "这个", "那个", "因为", "所以", "可以",
        ] {
            assert!(
                is_stopword(word, TokenizerMode::Unstemmed),
                "{word} should be a stopword"
            );
            assert!(
                is_stopword(word, TokenizerMode::Stemmed),
                "{word} should be a stopword"
            );
        }
        for word in ["清华", "学习", "北京"] {
            assert!(
                !is_stopword(word, TokenizerMode::Unstemmed),
                "{word} should not be a stopword"
            );
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
    fn vendored_stopword_lists_are_sorted_and_deduped() {
        use crate::stopwords_data::*;
        for (name, list) in [
            ("en", STOPWORDS_EN),
            ("es", STOPWORDS_ES),
            ("zh", STOPWORDS_ZH),
            ("ko", STOPWORDS_KO),
        ] {
            assert!(!list.is_empty(), "{name} list must be non-empty");
            let mut sorted = list.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                list,
                sorted.as_slice(),
                "{name} list must be sorted and deduped"
            );
        }
    }

    #[test]
    fn fold_diacritics_strips_marks_without_transliteration() {
        assert_eq!(fold_diacritics("niño"), "nino");
        assert_eq!(fold_diacritics("está"), "esta");
        assert_eq!(fold_diacritics("sí"), "si");
        // Not transliterated: identity preserved.
        assert_eq!(fold_diacritics("ß"), "ß");
        assert_eq!(fold_diacritics("ø"), "ø");
        assert_eq!(fold_diacritics("hello"), "hello");
    }

    #[test]
    fn spanish_folded_twins_are_stopped() {
        let stop = spanish_stopwords();
        for word in [
            "está", "esta", "sí", "si", "están", "estan", "también", "tambien",
        ] {
            assert!(stop.contains(word), "{word} should be a Spanish stopword");
        }
        // Content words are not stopwords, folded or not.
        assert!(!stop.contains("niño"));
        assert!(!stop.contains("nino"));
        // Spanish-only: must not leak into the default (English) path.
        assert!(!is_stopword("está", TokenizerMode::Unstemmed));
        assert!(!is_stopword("esta", TokenizerMode::Unstemmed));
    }

    #[test]
    fn spanish_known_spacy_entries_present() {
        let stop = spanish_stopwords();
        for word in [
            "donde", "cuando", "porque", "también", "tambien", "además", "ademas",
        ] {
            assert!(stop.contains(word), "{word} should be a Spanish stopword");
        }
    }

    #[test]
    fn chinese_spacy_entries_union_with_handbuilt() {
        // spaCy-only words (not in the hand-built list) ...
        for word in ["将要", "需要", "进行", "为了"] {
            assert!(
                is_stopword(word, TokenizerMode::Unstemmed),
                "{word} should be a stopword"
            );
            assert!(
                is_stopword(word, TokenizerMode::Stemmed),
                "{word} should be a stopword"
            );
        }
        // ... and the hand-built words still stop.
        for word in ["的", "我们", "因为", "可以"] {
            assert!(is_stopword(word, TokenizerMode::Unstemmed));
        }
    }

    #[test]
    fn korean_spacy_entries_union_with_handbuilt() {
        // spaCy-only word ...
        assert!(is_stopword("그러나", TokenizerMode::Unstemmed));
        assert!(is_stopword("그러나", TokenizerMode::Stemmed));
        // ... and hand-built words still stop.
        for word in ["은", "는", "것"] {
            assert!(is_stopword(word, TokenizerMode::Unstemmed));
        }
    }

    #[test]
    fn english_spacy_entries_present() {
        for word in ["however", "therefore", "among", "whom", "whose"] {
            assert!(
                english_stopwords().contains(word),
                "{word} should be an English stopword"
            );
        }
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
    fn han_tokens_emit_interleaved_unigrams_and_bigrams() {
        // Unigrams let a single-character query term (猫) match that
        // character inside an indexed word (橘猫); bigrams alone made that
        // impossible. Interleaving keeps each unigram adjacent to its
        // bigrams so position-based ranking treats them fairly.
        assert_eq!(
            han_tokens("清华大学"),
            vec!["清", "清华", "华", "华大", "大", "大学", "学"]
        );
        assert_eq!(han_tokens("中"), vec!["中"]);
        // Non-Han text passes through untouched.
        assert_eq!(han_tokens("Rust"), Vec::<String>::new());
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
