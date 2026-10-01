//! Language detection and per-language defaults for lint-ai's
//! language-by-language internationalization.
//!
//! Detection is script-based: count Han vs Hangul vs Latin characters and
//! take the winner; ambiguous or script-free text defaults to English.
//! Tokenization itself is script-aware per run (see [`crate::tokenizer`]),
//! so mixed-language text is handled without needing a single winner —
//! detection is used where a per-text language decision is required
//! (spaCy model selection, normalization).

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// Content language. `Auto` (the default) means "detect from the text".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lang {
    /// Auto-detect from script statistics. Default.
    #[default]
    #[value(name = "auto")]
    Auto,
    #[value(name = "en")]
    En,
    #[value(name = "zh")]
    Zh,
    #[value(name = "ko")]
    Ko,
    #[value(name = "es")]
    Es,
}

impl Lang {
    /// Resolve to a concrete language: `Auto` runs [`detect_lang`].
    pub fn resolve(self, text: &str) -> Lang {
        match self {
            Lang::Auto => detect_lang(text),
            lang => lang,
        }
    }
}

/// True for CJK Unified Ideographs and extensions (but not Hangul,
/// Hiragana, Katakana, or CJK punctuation — 。、「」！？ etc. are
/// separators, not word characters, and must not join bigrams).
pub fn is_han(ch: char) -> bool {
    matches!(ch,
        '\u{3400}'..='\u{4DBF}'   // Ext A
        | '\u{4E00}'..='\u{9FFF}'   // Unified Ideographs
        | '\u{F900}'..='\u{FAFF}'   // Compatibility Ideographs
        | '\u{20000}'..='\u{2A6DF}' // Ext B
        | '\u{2A6E0}'..='\u{2CEAF}' // Ext C
        | '\u{2CEB0}'..='\u{2EBEF}' // Ext D/E/F
        | '\u{30000}'..='\u{3134F}' // Ext G
        | '\u{31350}'..='\u{323AF}' // Ext H
        | '\u{2F800}'..='\u{2FA1D}' // Compatibility Supplement
    )
}

/// True for Hangul syllables and Jamo.
pub fn is_hangul(ch: char) -> bool {
    matches!(ch,
        '\u{1100}'..='\u{11FF}'   // Hangul Jamo
        | '\u{3130}'..='\u{318F}'   // Hangul Compatibility Jamo
        | '\u{A960}'..='\u{A97F}'   // Hangul Jamo Extended-A
        | '\u{AC00}'..='\u{D7AF}'   // Hangul Syllables
        | '\u{D7B0}'..='\u{D7FF}' // Hangul Jamo Extended-B
    )
}

/// Detect the language of `text` from script statistics. Never returns
/// [`Lang::Auto`]; ambiguous or script-free text defaults to English.
pub fn detect_lang(text: &str) -> Lang {
    let mut han = 0u32;
    let mut hangul = 0u32;
    let mut latin = 0u32;
    for ch in text.chars() {
        if is_han(ch) {
            han += 1;
        } else if is_hangul(ch) {
            hangul += 1;
        } else if ch.is_ascii_alphabetic() {
            latin += 1;
        }
    }
    if han > hangul && han > latin {
        Lang::Zh
    } else if hangul > han && hangul > latin {
        Lang::Ko
    } else {
        Lang::En
    }
}

/// The spaCy model to use for `lang` when the user did not pass an explicit
/// `--spacy-model`.
pub fn default_spacy_model_for_lang(lang: Lang) -> &'static str {
    match lang {
        Lang::Zh => "zh_core_web_sm",
        Lang::Ko => "ko_core_news_sm",
        Lang::Es => "es_core_news_sm",
        _ => "en_core_web_sm",
    }
}

/// Value of a single Chinese numeral character, or `None` if `ch` is not
/// one. 两 counts as 2; 〇/零 are 0.
fn chinese_digit_value(ch: char) -> Option<i64> {
    match ch {
        '一' => Some(1),
        '二' | '两' => Some(2),
        '三' => Some(3),
        '四' => Some(4),
        '五' => Some(5),
        '六' => Some(6),
        '七' => Some(7),
        '八' => Some(8),
        '九' => Some(9),
        '〇' | '零' => Some(0),
        _ => None,
    }
}

/// True for any character that can appear in a Chinese numeral run.
fn is_chinese_numeral_char(ch: char) -> bool {
    chinese_digit_value(ch).is_some() || matches!(ch, '十' | '百' | '千' | '万' | '亿')
}

/// Parse a Chinese numeral string (e.g. 二十, 一百二十三, 三千五百万,
/// 二〇二六) to an integer. Returns `None` for empty input, non-numeral
/// input, or a bare unit (万/亿 alone carries no value).
pub fn parse_chinese_numeral(s: &str) -> Option<i64> {
    if s.is_empty() || !s.chars().all(is_chinese_numeral_char) {
        return None;
    }
    let mut total = 0i64;
    let mut current = 0i64; // value of the section below 万/亿
    let mut pending = 0i64; // most recent digit(s) not yet attached to a unit
    let mut has_pending = false;
    // Anything that carries value on its own (digits, or 十/百/千 with
    // their implicit 1). Bare 万/亿 carry nothing.
    let mut saw_value = false;
    for ch in s.chars() {
        if let Some(d) = chinese_digit_value(ch) {
            saw_value = true;
            pending = pending * 10 + d;
            has_pending = true;
            continue;
        }
        // A small unit (十/百/千) with no pending digit means an implicit
        // 1 (十 = 10). 万/亿 fold the whole current section and add no
        // implicit 1 (五百万 = 5,000,000, not 5,000,001).
        match ch {
            '十' | '百' | '千' => {
                saw_value = true;
                let v = if has_pending { pending } else { 1 };
                pending = 0;
                has_pending = false;
                match ch {
                    '十' => current += v * 10,
                    '百' => current += v * 100,
                    _ => current += v * 1000,
                }
            }
            '万' | '亿' => {
                let section = current + if has_pending { pending } else { 0 };
                pending = 0;
                has_pending = false;
                total += section * if ch == '万' { 10_000 } else { 100_000_000 };
                current = 0;
            }
            _ => return None, // unreachable: input pre-validated
        }
    }
    if !saw_value {
        // Bare 万/亿 (or empty, already excluded) carry no value.
        return None;
    }
    total += current + if has_pending { pending } else { 0 };
    Some(total)
}

/// Common Chinese measure words / units. A single-character numeral is
/// only treated as a number when followed by one of these (or 年/月/日/号
/// for dates) — this keeps 一起 ("together") and 一心一意 ("wholehearted")
/// from becoming "1起" and "1心1意".
const CHINESE_MEASURE_WORDS: &[char] = &[
    '个', '位', '名', '只', '条', '张', '把', '件', '本', '块', '元', '角', '分', '头', '匹', '栋',
    '层', '间', '所', '家', '次', '回', '趟', '遍', '顿', '场', '节', '课', '道', '题', '篇', '章',
    '首', '幅', '双', '对', '串', '群', '批', '组', '队', '班', '套', '台', '辆', '架', '艘', '枚',
    '颗', '粒', '滴', '点', '口', '扇', '盏', '枝', '根', '株', '棵', '朵', '片', '页', '封', '袋',
    '包', '箱', '盒', '瓶', '杯', '碗', '盘', '桶', '盆', '罐', '捆', '堆', '人', '口', '户', '家',
    '国', '省', '市', '区', '县', '镇', '村', '路', '街', '号', '楼', '室', '年', '月', '日', '号',
    '天', '周', '岁', '时', '分', '秒', '米', '里', '斤', '两', '吨', '升', '瓦', '倍',
];

/// Replace Chinese numeral runs in `text` with Arabic digits, so
/// downstream English-oriented number handling (text2num, `\d+` regexes)
/// sees them. Multi-character runs are always numbers; a single-character
/// run must be followed by a measure word (see
/// [`CHINESE_MEASURE_WORDS`]) to avoid rewriting idioms like 一起.
pub fn normalize_chinese_numbers(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if !is_chinese_numeral_char(chars[i]) {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        let mut j = i;
        while j < chars.len() && is_chinese_numeral_char(chars[j]) {
            j += 1;
        }
        let run: String = chars[i..j].iter().collect();
        let run_len = j - i;
        let next = chars.get(j).copied();
        // 两 is also a unit of weight (50g); as a numeral run of length 1
        // followed by a non-measure it is left alone by the same rule.
        let replace = if run_len >= 2 {
            true
        } else {
            matches!(next, Some(n) if CHINESE_MEASURE_WORDS.contains(&n))
        };
        if replace {
            if let Some(n) = parse_chinese_numeral(&run) {
                out.push_str(&n.to_string());
                i = j;
                continue;
            }
        }
        // Not a number after all (e.g. bare 万, or 一起): emit verbatim.
        for c in chars[i..j].iter() {
            out.push(*c);
        }
        i = j;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_chinese() {
        assert_eq!(detect_lang("我毕业于清华大学"), Lang::Zh);
        assert_eq!(detect_lang("今天天气如何"), Lang::Zh);
    }

    #[test]
    fn detects_korean() {
        assert_eq!(detect_lang("한국어 테스트입니다"), Lang::Ko);
        assert_eq!(detect_lang("학교에 갔다"), Lang::Ko);
    }

    #[test]
    fn detects_english_and_defaults_on_ambiguous() {
        assert_eq!(detect_lang("What degree did I graduate with?"), Lang::En);
        assert_eq!(detect_lang(""), Lang::En);
        assert_eq!(detect_lang("12345 !!!"), Lang::En);
        // Tie between Han and Latin: ambiguous -> English.
        assert_eq!(detect_lang("中文ab"), Lang::En);
    }

    #[test]
    fn auto_resolves() {
        assert_eq!(Lang::Auto.resolve("我毕业于清华大学"), Lang::Zh);
        assert_eq!(Lang::Auto.resolve("한국어 테스트"), Lang::Ko);
        assert_eq!(Lang::Zh.resolve("hello"), Lang::Zh);
    }

    #[test]
    fn spacy_model_defaults() {
        assert_eq!(default_spacy_model_for_lang(Lang::Zh), "zh_core_web_sm");
        assert_eq!(default_spacy_model_for_lang(Lang::Ko), "ko_core_news_sm");
        assert_eq!(default_spacy_model_for_lang(Lang::Es), "es_core_news_sm");
        assert_eq!(default_spacy_model_for_lang(Lang::En), "en_core_web_sm");
    }

    #[test]
    fn parses_chinese_numerals() {
        for (s, n) in [
            ("一", 1),
            ("二", 2),
            ("两", 2),
            ("十", 10),
            ("十二", 12),
            ("二十", 20),
            ("二十五", 25),
            ("一百", 100),
            ("一百二十三", 123),
            ("三千五百万", 35_000_000),
            ("一亿两千万", 120_000_000),
            ("二〇二六", 2026),
            ("两百", 200),
            ("十万", 100_000),
            ("〇", 0),
        ] {
            assert_eq!(parse_chinese_numeral(s), Some(n), "for {s:?}");
        }
        for s in ["", "万", "亿", "abc", "三a", "一起"] {
            assert_eq!(parse_chinese_numeral(s), None, "for {s:?}");
        }
    }

    #[test]
    fn normalizes_chinese_numbers_in_text() {
        assert_eq!(normalize_chinese_numbers("我买了三本书"), "我买了3本书");
        assert_eq!(normalize_chinese_numbers("二十五天后见"), "25天后见");
        // Single char + measure word.
        assert_eq!(normalize_chinese_numbers("等一个人"), "等1个人");
        // Idioms are left alone.
        assert_eq!(normalize_chinese_numbers("我们一起去"), "我们一起去");
        assert_eq!(normalize_chinese_numbers("一心一意"), "一心一意");
        // Non-numeral text untouched.
        assert_eq!(normalize_chinese_numbers("今天天气很好"), "今天天气很好");
    }
}
