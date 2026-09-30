use crate::index::{MemoryIndex, SearchResult};
use regex::Regex;
use serde::Serialize;
use std::collections::HashSet;
use std::sync::OnceLock;
use text2num::{replace_numbers_in_text, Language};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateIntent {
    Count,
    Sum,
}

#[derive(Debug, Clone, Serialize)]
pub struct AggregateCitation {
    pub doc_id: String,
    pub source: String,
    pub group_id: Option<String>,
    pub score: f32,
    pub evidence_value: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AggregateOutput {
    pub intent: String,
    pub normalized_query: String,
    pub retrieved_candidates: usize,
    pub evidence_count: usize,
    pub answer_value: Option<f64>,
    pub answer_text: String,
    pub citations: Vec<AggregateCitation>,
    pub reasoning: String,
}

pub fn classify_aggregate_intent(query: &str) -> Option<AggregateIntent> {
    let q = query.to_lowercase();
    let asks_for_recommended_quantity = q.contains("how many")
        && [
            " should ",
            " should we",
            " should i",
            " allowed ",
            " maximum ",
            " minimum ",
            " limit ",
        ]
        .iter()
        .any(|cue| q.contains(cue));
    // Chinese: 应该买多少本书 (how many books should I buy) is advice,
    // not a count over memories.
    let asks_for_recommended_quantity_zh = ["应该", "建议", "最多", "最少", "限制"]
        .iter()
        .any(|cue| q.contains(cue))
        && ["多少", "几个", "几次"].iter().any(|cue| q.contains(cue));
    if asks_for_recommended_quantity || asks_for_recommended_quantity_zh {
        return None;
    }
    if q.contains("how many")
        || q.starts_with("count ")
        || q.contains(" number of ")
        || q.contains(" number of")
        || q.contains("count the ")
        || q.contains("times did i")
        || q.contains("多少")
        || q.contains("几个")
        || q.contains("几次")
        || q.contains("数量")
    {
        // 总共/一共/合计 + 多少 = a sum question ("how much in total"),
        // not a count.
        if ["总共", "一共", "合计", "总计", "总额", "总和"]
            .iter()
            .any(|cue| q.contains(cue))
        {
            return Some(AggregateIntent::Sum);
        }
        return Some(AggregateIntent::Count);
    }
    if q.contains("how much")
        || q.contains(" total ")
        || q.starts_with("total ")
        || q.contains("combined")
        || q.contains("in all")
        || q.contains("sum ")
        || q.contains("总共")
        || q.contains("一共")
        || q.contains("合计")
        || q.contains("总计")
        || q.contains("总额")
        || q.contains("总和")
    {
        return Some(AggregateIntent::Sum);
    }
    // Korean aggregate triggers. Token-prefix matching (not substring) so
    // that e.g. "총" does not fire inside "대통령".
    if ko_token_starts_with(&q, &["몇"]) {
        return Some(AggregateIntent::Count);
    }
    if ko_token_starts_with(&q, &["총", "합계", "얼마나"]) {
        return Some(AggregateIntent::Sum);
    }
    None
}

/// True when any whitespace-delimited token of `query` starts with one of
/// `forms`. Korean particles attach inside the eojeol, so prefix matching
/// covers inflected forms (`합계는`, `몇개를`) while keeping the trigger
/// precise.
fn ko_token_starts_with(query: &str, forms: &[&str]) -> bool {
    query
        .split_whitespace()
        .any(|tok| forms.iter().any(|f| tok.starts_with(f)))
}

pub fn normalize_number_words(query: &str) -> String {
    // Chinese numerals first (Rust-side; the Python rule forbids new
    // Python logic), then the historical English text2num pass, then
    // Korean number words.
    let zh_normalized = crate::lang::normalize_chinese_numbers(query);
    let replaced = replace_numbers_in_text(&zh_normalized, &Language::english(), 0.0);
    normalize_korean_number_words(&replaced)
}

/// Replace Korean number words with digits.
///
/// Covers Sino-Korean numerals (`삼백오십` → `350`, `오만` → `50000`) when
/// followed by a counter (`원`, `개`, `명`, …) or a non-Hangul boundary,
/// and common native numerals (`하나` → `1`, `스물` → `20`). Runs that
/// continue into other Hangul (e.g. `일` in `일요일`) are left alone.
///
/// Documented limitations: compound native numbers (`스물다섯`),
/// attributive-only single syllables in running text, and consecutive
/// bare Sino-Korean digits without units are not parsed.
fn normalize_korean_number_words(text: &str) -> String {
    static KO_NUM_RE: OnceLock<Regex> = OnceLock::new();
    // Note: the `regex` crate has no lookahead, so the "not followed by
    // Hangul" boundary is enforced in the replacement closure via group 5:
    // when group 5 matches, the numeral run continues into a larger Hangul
    // word and the match is left unchanged.
    let re = KO_NUM_RE.get_or_init(|| {
        Regex::new(
            r"(하나|둘|셋|넷|다섯|여섯|일곱|여덟|아홉|열|스물|서른|마흔|쉰|예순|일흔|여든|아흔|한|두|세|네)(은|는|이|가|을|를|에|의|과|와|도|만)?([가-힣])?|([일이삼사오육칠팔구]*[십백천만억][일이삼사오육칠팔구십백천만억]*)(원|달러|개|명|번|회|권|대|살|층|호)?([가-힣])?",
        )
        .expect("valid Korean numeral regex")
    });
    re.replace_all(text, |caps: &regex::Captures| {
        // Native word: group 1 (+ particle group 2, + following-Hangul
        // group 3). Sino-Korean run: group 4 (+ counter group 5, +
        // following-Hangul group 6).
        let (num, suffix, following) = if caps.get(1).is_some() {
            (
                caps.get(1).map(|m| m.as_str()).unwrap_or(""),
                caps.get(2).map(|s| s.as_str()).unwrap_or(""),
                caps.get(3).map(|s| s.as_str()).unwrap_or(""),
            )
        } else {
            (
                caps.get(4).map(|m| m.as_str()).unwrap_or(""),
                caps.get(5).map(|s| s.as_str()).unwrap_or(""),
                caps.get(6).map(|s| s.as_str()).unwrap_or(""),
            )
        };
        // A bare numeral run directly followed by Hangul is part of a
        // larger word (not a numeral); leave it unchanged. A trailing
        // particle/counter terminates the numeral, so following Hangul is
        // kept verbatim after the digits.
        if suffix.is_empty() && !following.is_empty() {
            return caps[0].to_string();
        }
        match parse_korean_numeral(num) {
            Some(v) => format!("{v}{suffix}{following}"),
            None => caps[0].to_string(),
        }
    })
    .into_owned()
}

/// Parse a Korean numeral word into a number. Returns `None` for forms
/// outside the supported table (the caller keeps the original text).
fn parse_korean_numeral(word: &str) -> Option<i64> {
    // Native numerals (single words only; compounds like 스물다섯 are out
    // of scope).
    let native = match word {
        "하나" | "한" => 1,
        "둘" | "두" => 2,
        "셋" | "세" => 3,
        "넷" | "네" => 4,
        "다섯" => 5,
        "여섯" => 6,
        "일곱" => 7,
        "여덟" => 8,
        "아홉" => 9,
        "열" => 10,
        "스물" => 20,
        "서른" => 30,
        "마흔" => 40,
        "쉰" => 50,
        "예순" => 60,
        "일흔" => 70,
        "여든" => 80,
        "아흔" => 90,
        _ => -1,
    };
    if native >= 0 {
        return Some(native);
    }
    // Sino-Korean: digits, small units (십백천), big units (만억).
    fn digit(c: char) -> Option<i64> {
        match c {
            '일' => Some(1),
            '이' => Some(2),
            '삼' => Some(3),
            '사' => Some(4),
            '오' => Some(5),
            '육' => Some(6),
            '칠' => Some(7),
            '팔' => Some(8),
            '구' => Some(9),
            _ => None,
        }
    }
    let mut total: i64 = 0;
    let mut section: i64 = 0;
    let mut pending: Option<i64> = None;
    for c in word.chars() {
        if let Some(d) = digit(c) {
            pending = Some(d);
        } else if let Some(unit) = match c {
            '십' => Some(10),
            '백' => Some(100),
            '천' => Some(1000),
            _ => None,
        } {
            section += pending.unwrap_or(1) * unit;
            pending = None;
        } else if let Some(big) = match c {
            '만' => Some(10_000),
            '억' => Some(100_000_000),
            _ => None,
        } {
            let base = section + pending.unwrap_or(0);
            total += if base == 0 { big } else { base * big };
            section = 0;
            pending = None;
        } else {
            return None;
        }
    }
    Some(total + section + pending.unwrap_or(0))
}

pub fn build_aggregate_output(
    index: &MemoryIndex,
    query: &str,
    results: &[SearchResult],
    candidate_limit: usize,
) -> Option<AggregateOutput> {
    let intent = classify_aggregate_intent(query)?;
    let normalized_query = normalize_number_words(query);
    let limit = candidate_limit.max(1).min(results.len().max(1));
    let relevant = &results[..limit.min(results.len())];
    let citations = match intent {
        AggregateIntent::Count => build_count_citations(index, relevant),
        AggregateIntent::Sum => build_sum_citations(index, relevant),
    };

    let answer_value = match intent {
        AggregateIntent::Count => Some(citations.len() as f64),
        AggregateIntent::Sum => {
            let sum = citations
                .iter()
                .filter_map(|c| c.evidence_value)
                .sum::<f64>();
            if sum > 0.0 {
                Some(sum)
            } else {
                None
            }
        }
    };

    let answer_text = match (intent, answer_value) {
        (AggregateIntent::Count, Some(v)) => format!("{v:.0}"),
        (AggregateIntent::Sum, Some(v)) => format!("{v:.2}"),
        _ => "unknown".to_string(),
    };
    let intent_name = match intent {
        AggregateIntent::Count => "count",
        AggregateIntent::Sum => "sum",
    };
    let reasoning = match intent {
        AggregateIntent::Count => format!(
            "counted {} unique evidence items from the top {} retrieved candidates",
            citations.len(),
            limit
        ),
        AggregateIntent::Sum => format!(
            "summed numeric evidence from {} supporting candidates in the top {} retrieved candidates",
            citations.iter().filter(|c| c.evidence_value.is_some()).count(),
            limit
        ),
    };

    Some(AggregateOutput {
        intent: intent_name.to_string(),
        normalized_query,
        retrieved_candidates: limit,
        evidence_count: citations.len(),
        answer_value,
        answer_text,
        citations,
        reasoning,
    })
}

fn build_count_citations(index: &MemoryIndex, results: &[SearchResult]) -> Vec<AggregateCitation> {
    let mut seen = HashSet::new();
    let mut citations = Vec::new();
    for result in results {
        let key = result
            .group_id
            .clone()
            .unwrap_or_else(|| result.doc_id.clone());
        if !seen.insert(key) {
            continue;
        }
        let citation = AggregateCitation {
            doc_id: result.doc_id.clone(),
            source: result.source.clone(),
            group_id: result.group_id.clone(),
            score: result.score,
            evidence_value: None,
        };
        citations.push(citation);
        if citations.len() >= index.docs.len().min(results.len()) {
            break;
        }
    }
    citations
}

fn build_sum_citations(index: &MemoryIndex, results: &[SearchResult]) -> Vec<AggregateCitation> {
    let mut seen = HashSet::new();
    let mut citations = Vec::new();
    for result in results {
        let key = result
            .group_id
            .clone()
            .unwrap_or_else(|| result.doc_id.clone());
        if !seen.insert(key) {
            continue;
        }
        let Some(doc) = index.docs.get(&result.doc_id) else {
            continue;
        };
        let evidence_text = aggregate_text(doc);
        let evidence_value = extract_numeric_value(&evidence_text);
        let citation = AggregateCitation {
            doc_id: result.doc_id.clone(),
            source: result.source.clone(),
            group_id: result.group_id.clone(),
            score: result.score,
            evidence_value,
        };
        citations.push(citation);
    }
    citations
}

fn aggregate_text(doc: &crate::index::DocRecord) -> String {
    let mut parts = vec![doc.headings.join(" ")];
    parts.push(doc.probable_topic.clone().unwrap_or_default());
    parts.push(doc.doc_type_guess.clone().unwrap_or_default());
    parts.push(doc.temporal_terms.join(" "));
    parts.push(doc.content.clone());
    parts.join("\n")
}

fn extract_numeric_value(text: &str) -> Option<f64> {
    let normalized = normalize_number_words(text);
    // Han-aware boundaries: in the `regex` crate \w matches Han letters,
    // so \b never fires between a digit and an adjacent Han char
    // (3本书). Space them apart first; English text has no Han chars and
    // is untouched.
    static HAN_DIGIT_RE: OnceLock<Regex> = OnceLock::new();
    static DIGIT_HAN_RE: OnceLock<Regex> = OnceLock::new();
    let han_digit = HAN_DIGIT_RE
        .get_or_init(|| Regex::new(r"([\u{4e00}-\u{9fff}\u{3400}-\u{4dbf}])(\d)").unwrap());
    let digit_han = DIGIT_HAN_RE
        .get_or_init(|| Regex::new(r"(\d)([\u{4e00}-\u{9fff}\u{3400}-\u{4dbf}])").unwrap());
    let spaced = han_digit.replace_all(&normalized, "$1 $2");
    let spaced = digit_han.replace_all(&spaced, "$1 $2");
    static NUMBER_RE: OnceLock<Regex> = OnceLock::new();
    let re = NUMBER_RE
        .get_or_init(|| Regex::new(r"(?i)\b\d+(?:\.\d+)?\b").expect("valid numeric regex"));
    let mut values = Vec::new();
    for mat in re.find_iter(&spaced) {
        if let Ok(v) = mat.as_str().parse::<f64>() {
            values.push(v);
        }
    }
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{Claim, DocRecord, Provenance, ScoreBreakdown, SearchResult, SectionChunk};
    use crate::tier1::{RankedTerm, Tier1Entity};

    fn sample_doc(doc_id: &str, content: &str, group_id: Option<&str>) -> DocRecord {
        DocRecord {
            doc_id: doc_id.to_string(),
            source: format!("source://{}", doc_id),
            content: content.to_string(),
            timestamp: Some("2024-05-10".to_string()),
            doc_length: content.len(),
            author_agent: None,
            group_id: group_id.map(|g| g.to_string()),
            filters: std::collections::BTreeMap::new(),
            probable_topic: Some("hiking".to_string()),
            doc_type_guess: Some("note".to_string()),
            headings: vec!["Overview".to_string()],
            doc_links: vec![],
            temporal_terms: vec!["date friday".to_string()],
            key_entities: vec![Tier1Entity {
                text: "hike".to_string(),
                label: "PROPN".to_string(),
                start: 0,
                end: 4,
                score: Some(1.0),
                source: "heuristic".to_string(),
            }],
            important_terms: vec![RankedTerm {
                term: "hike".to_string(),
                score: 1.0,
                source: "yake".to_string(),
            }],
            key_phrases: vec![],
            key_phrase_extraction_hash: String::new(),
            section_chunks: vec![SectionChunk {
                chunk_id: format!("{}::chunk0", doc_id),
                heading: "Overview".to_string(),
                content: content.to_string(),
                start_line: 1,
                end_line: 1,
                timestamp: Some("2024-05-10".to_string()),
                key_entities: vec!["hike".to_string()],
                important_terms: vec!["hike".to_string()],
            }],
            embedding: None,
            top_claims: vec![Claim {
                subject: "a".to_string(),
                predicate: "b".to_string(),
                object: "c".to_string(),
                confidence: 1.0,
            }],
            provenance: Provenance {
                source: "source://doc".to_string(),
                timestamp: Some("2024-05-10".to_string()),
                ner_provider: "heuristic".to_string(),
                term_ranker: "yake".to_string(),
                index_version: "v1".to_string(),
            },
            content_hash: String::new(),
        }
    }

    #[test]
    fn classifies_count_and_sum_intents() {
        assert_eq!(
            classify_aggregate_intent("How many hikes did I do?"),
            Some(AggregateIntent::Count)
        );
        assert_eq!(
            classify_aggregate_intent("How much did I spend?"),
            Some(AggregateIntent::Sum)
        );
    }

    #[test]
    fn normalizes_number_words() {
        let out = normalize_number_words("I walked two miles and ate three apples");
        assert!(out.contains("2"));
        assert!(out.contains("3"));
    }

    #[test]
    fn korean_aggregate_triggers() {
        assert_eq!(
            classify_aggregate_intent("이번 달에 몇 개의 약속이 있어?"),
            Some(AggregateIntent::Count)
        );
        assert_eq!(
            classify_aggregate_intent("총 얼마를 썼어?"),
            Some(AggregateIntent::Sum)
        );
        assert_eq!(
            classify_aggregate_intent("합계는 얼마야?"),
            Some(AggregateIntent::Sum)
        );
        // "총" inside another word must not trigger.
        assert_eq!(classify_aggregate_intent("대통령에 대해 알려줘"), None);
    }

    #[test]
    fn korean_numerals_normalize() {
        assert_eq!(
            normalize_number_words("삼천원을 썼다"),
            "3000원을 썼다"
        );
        assert_eq!(normalize_number_words("오만개"), "50000개");
        assert_eq!(normalize_number_words("하나의 사과"), "1의 사과");
        assert_eq!(normalize_number_words("스물 명"), "20 명");
        // Non-numeric Hangul is untouched.
        assert_eq!(normalize_number_words("일요일에 만났다"), "일요일에 만났다");
        assert_eq!(normalize_number_words("이 책"), "이 책");
    }

    #[test]
    fn classifies_chinese_count_and_sum_intents() {
        assert_eq!(
            classify_aggregate_intent("我买了多少本书"),
            Some(AggregateIntent::Count)
        );
        assert_eq!(
            classify_aggregate_intent("我去了几次北京"),
            Some(AggregateIntent::Count)
        );
        assert_eq!(
            classify_aggregate_intent("一共花了多少钱"),
            Some(AggregateIntent::Sum)
        );
        assert_eq!(
            classify_aggregate_intent("总共买了几本书"),
            Some(AggregateIntent::Sum)
        );
        // Advice-seeking is not a count over memories.
        assert_eq!(classify_aggregate_intent("我应该买多少本书"), None);
    }

    #[test]
    fn normalizes_chinese_number_words() {
        let out = normalize_number_words("我买了三本书");
        assert!(out.contains('3'), "expected 3 in {out:?}");
        let out = normalize_number_words("二十五天后见");
        assert!(out.contains("25"), "expected 25 in {out:?}");
    }

    #[test]
    fn extracts_numbers_adjacent_to_han() {
        // Digits touching Han chars have no \b boundary; they must count.
        assert_eq!(extract_numeric_value("买了3本书"), Some(3.0));
        assert_eq!(extract_numeric_value("买了三本书"), Some(3.0));
        assert_eq!(extract_numeric_value("花了25元"), Some(25.0));
        // English behavior unchanged.
        assert_eq!(extract_numeric_value("walked 2 miles"), Some(2.0));
        assert_eq!(extract_numeric_value("no numbers here"), None);
    }

    #[test]
    fn builds_count_aggregation() {
        let index = MemoryIndex::from_records(vec![
            sample_doc("d1", "I walked 2 miles", Some("g1")),
            sample_doc("d2", "I walked 3 miles", Some("g2")),
        ]);
        let results = vec![
            SearchResult {
                doc_id: "d1".to_string(),
                source: "source://d1".to_string(),
                group_id: Some("g1".to_string()),
                score: 1.0,
                score_breakdown: ScoreBreakdown::default(),
                matched_entities: vec![],
                matched_terms: vec![],
                probable_topic: Some("hiking".to_string()),
                doc_type_guess: Some("note".to_string()),
                semantic_status: None,
                superseded_by: None,
                relation_confidence: None,
                relation_evidence: vec![],
            },
            SearchResult {
                doc_id: "d2".to_string(),
                source: "source://d2".to_string(),
                group_id: Some("g2".to_string()),
                score: 0.9,
                score_breakdown: ScoreBreakdown::default(),
                matched_entities: vec![],
                matched_terms: vec![],
                probable_topic: Some("hiking".to_string()),
                doc_type_guess: Some("note".to_string()),
                semantic_status: None,
                superseded_by: None,
                relation_confidence: None,
                relation_evidence: vec![],
            },
        ];
        let agg = build_aggregate_output(&index, "How many miles did I walk?", &results, 5)
            .expect("expected aggregation");
        assert_eq!(agg.intent, "count");
        assert!(agg.answer_value.is_some());
        assert!(!agg.citations.is_empty());
    }
}
