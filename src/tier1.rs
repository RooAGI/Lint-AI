use crate::lang::Lang;
use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::thread::{self, sleep};
use std::time::{Duration, Instant};

const SPACY_SUBPROCESS_TIMEOUT_SECS: u64 = 20;

#[derive(Debug, Clone)]
pub struct Tier1DocInput {
    pub id: String,
    pub source: String,
    pub content: String,
    pub concept: String,
    pub headings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier1Entity {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
    #[serde(default)]
    pub score: Option<f32>,
    pub source: String,
}

/// Provenance marker for key entities that came from the behood noun-phrase
/// (grammar-accepted entity mention) layer rather than NER. The segment
/// summary treats these specially: their literal (unstemmed) tokens join
/// the entity channel so stemming cannot conflate the phrase head
/// ("conference" must not become "confer").
pub const BEHOOD_NP_ENTITY_SOURCE: &str = "behood-np";

#[derive(Serialize)]
pub struct Tier1DocEntities {
    pub id: String,
    pub source: String,
    pub key_entities: Vec<Tier1Entity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedTerm {
    pub term: String,
    pub score: f32,
    pub source: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Tier1DocTerms {
    pub id: String,
    pub source: String,
    pub important_terms: Vec<RankedTerm>,
}

pub trait KeyEntityRanker {
    fn rank_docs(&self, docs: &[Tier1DocInput]) -> Result<HashMap<String, Vec<Tier1Entity>>>;
    fn name(&self) -> &'static str;
}

pub trait ImportantTermRanker {
    fn name(&self) -> &'static str;
    fn rank_terms(&self, doc: &Tier1DocInput) -> Vec<RankedTerm>;
}

pub struct HeuristicKeyEntityRanker;

impl HeuristicKeyEntityRanker {
    fn rank_one(doc: &Tier1DocInput) -> Vec<Tier1Entity> {
        #[derive(Clone, Copy)]
        struct Cand {
            start: usize,
            end: usize,
            mentions: usize,
            heading_hits: usize,
            section_hits: usize,
            first_pos: usize,
            label: &'static str,
        }

        let mut candidates: HashMap<String, Cand> = HashMap::new();
        let mut section_bounds = Vec::new();
        let mut last = 0usize;
        for h in &doc.headings {
            if let Some(pos) = doc.content.find(h) {
                if pos > last {
                    section_bounds.push((last, pos));
                }
                last = pos;
            }
        }
        section_bounds.push((last, doc.content.len()));
        if section_bounds.is_empty() {
            section_bounds.push((0, doc.content.len()));
        }

        let concept = doc.concept.trim().to_string();
        if !concept.is_empty() {
            candidates.insert(
                concept.clone(),
                Cand {
                    start: 0,
                    end: concept.len(),
                    mentions: 1,
                    heading_hits: 0,
                    section_hits: 1,
                    first_pos: 0,
                    label: "CONCEPT",
                },
            );
        }

        for cap in title_case_regex().captures_iter(&doc.content).take(80) {
            let m = cap.get(1).expect("capture exists");
            let text = m.as_str().trim();
            if text.len() < 3 {
                continue;
            }
            let key = text.to_string();
            let section_hits = section_bounds
                .iter()
                .filter(|(s, e)| m.start() >= *s && m.start() < *e)
                .count()
                .max(1);
            let entry = candidates.entry(key).or_insert(Cand {
                start: m.start(),
                end: m.end(),
                mentions: 0,
                heading_hits: 0,
                section_hits: 0,
                first_pos: m.start(),
                label: "PROPN",
            });
            entry.mentions += 1;
            entry.section_hits = entry.section_hits.max(section_hits);
            entry.first_pos = entry.first_pos.min(m.start());
        }

        for m in acronym_regex().find_iter(&doc.content).take(50) {
            let text = m.as_str();
            let key = text.to_string();
            let entry = candidates.entry(key).or_insert(Cand {
                start: m.start(),
                end: m.end(),
                mentions: 0,
                heading_hits: 0,
                section_hits: 1,
                first_pos: m.start(),
                label: "ACRONYM",
            });
            entry.mentions += 1;
            entry.first_pos = entry.first_pos.min(m.start());
        }

        for heading in &doc.headings {
            let heading_l = heading.to_lowercase();
            for (term, cand) in &mut candidates {
                if heading_l.contains(&term.to_lowercase()) {
                    cand.heading_hits += 1;
                }
            }
        }

        let len = doc.content.len().max(1) as f32;
        let mut out: Vec<Tier1Entity> = candidates
            .into_iter()
            .filter_map(|(text, cand)| {
                if text.len() < 3 {
                    return None;
                }
                let pos_bonus = 1.0 + (1.0 - (cand.first_pos as f32 / len));
                let freq_score = (cand.mentions as f32).ln_1p();
                let section_score = cand.section_hits as f32;
                let heading_score = (cand.heading_hits as f32) * 1.5;
                let score =
                    0.8 * freq_score + 0.7 * section_score + 1.2 * heading_score + 0.6 * pos_bonus;
                Some(Tier1Entity {
                    text,
                    label: cand.label.to_string(),
                    start: cand.start,
                    end: cand.end,
                    score: Some(score),
                    source: "heuristic-scored".to_string(),
                })
            })
            .collect();
        out.sort_by(|a, b| {
            b.score
                .unwrap_or(0.0)
                .partial_cmp(&a.score.unwrap_or(0.0))
                .unwrap_or(Ordering::Equal)
        });
        out.truncate(12);
        if out.is_empty() {
            out.push(Tier1Entity {
                text: doc.source.clone(),
                label: "DOC".to_string(),
                start: 0,
                end: doc.source.len(),
                score: Some(0.3),
                source: "heuristic-scored".to_string(),
            });
        }
        out
    }
}

impl KeyEntityRanker for HeuristicKeyEntityRanker {
    fn rank_docs(&self, docs: &[Tier1DocInput]) -> Result<HashMap<String, Vec<Tier1Entity>>> {
        let mut out = HashMap::new();
        for doc in docs {
            out.insert(doc.id.clone(), Self::rank_one(doc));
        }
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "heuristic"
    }
}

pub struct SpacyKeyEntityRanker {
    pub model: String,
    pub script_path: String,
}

/// The spaCy model used when the user did not pass `--spacy-model`.
/// Compared by value (not by "was the flag passed") to decide whether the
/// per-language default applies — see `PipelineOptions::spacy_model_for_text`.
pub const DEFAULT_SPACY_MODEL: &str = "en_core_web_sm";

pub fn default_spacy_script_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/spacy_ner.py")
}

pub fn detect_python_executable() -> String {
    if let Ok(value) = std::env::var("PYTHON_EXECUTABLE") {
        let value = value.trim();
        if !value.is_empty() {
            return value.to_string();
        }
    }
    if let Ok(value) = std::env::var("PYTHON") {
        let value = value.trim();
        if !value.is_empty() {
            return value.to_string();
        }
    }

    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        let mut venv_path = PathBuf::from(venv);
        if cfg!(windows) {
            venv_path.push("Scripts");
            venv_path.push("python.exe");
        } else {
            venv_path.push("bin");
            venv_path.push("python3");
            if !venv_path.exists() {
                venv_path.pop();
                venv_path.push("python");
            }
        }
        if venv_path.exists() {
            return venv_path.to_string_lossy().into_owned();
        }
    }

    "python3".to_string()
}

fn title_case_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b").expect("valid regex"))
}

fn acronym_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b([A-Z]{2,8})\b").expect("valid regex"))
}

fn content_word_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(&format!(
            r"[{L}][{L}0-9_\-]{{2,}}",
            L = crate::tokenizer::LATIN_LETTER
        ))
        .expect("valid regex")
    })
}

fn rake_token_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(&format!(
            r"[{L}][{L}0-9_\-]{{1,}}",
            L = crate::tokenizer::LATIN_LETTER
        ))
        .expect("valid regex")
    })
}

/// Sentence-boundary characters: ASCII plus CJK fullwidth forms.
const SENTENCE_ENDINGS: &[char] = &['.', '!', '?', '。', '！', '？'];

#[derive(Serialize)]
struct SpacyDocInput<'a> {
    id: &'a str,
    text: &'a str,
}

#[derive(Serialize)]
struct SpacyBatchInput<'a> {
    model: &'a str,
    documents: Vec<SpacyDocInput<'a>>,
}

#[derive(Deserialize)]
struct SpacyBatchOutput {
    documents: Vec<SpacyDocOutput>,
}

#[derive(Deserialize)]
struct SpacyDocOutput {
    id: String,
    entities: Vec<SpacyEntityOutput>,
}

#[derive(Deserialize)]
struct SpacyEntityOutput {
    text: String,
    label: String,
    start: usize,
    end: usize,
    #[serde(default)]
    score: Option<f32>,
}

impl KeyEntityRanker for SpacyKeyEntityRanker {
    fn rank_docs(&self, docs: &[Tier1DocInput]) -> Result<HashMap<String, Vec<Tier1Entity>>> {
        let payload = SpacyBatchInput {
            model: &self.model,
            documents: docs
                .iter()
                .map(|d| SpacyDocInput {
                    id: &d.id,
                    text: &d.content,
                })
                .collect(),
        };
        let input_json = serde_json::to_string(&payload)?;

        let python_executable = detect_python_executable();
        let mut child = Command::new(&python_executable)
            .arg("-I")
            .arg(&self.script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        if let Some(stdin) = child.stdin.take() {
            write_spacy_input(stdin, input_json.as_bytes())?;
        }
        let mut stdout = child.stdout.take().expect("stdout pipe configured");
        let mut stderr = child.stderr.take().expect("stderr pipe configured");
        let stdout_reader = thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = stdout.read_to_end(&mut bytes);
            bytes
        });
        let stderr_reader = thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = stderr.read_to_end(&mut bytes);
            bytes
        });
        let timeout = Duration::from_secs(SPACY_SUBPROCESS_TIMEOUT_SECS);
        let start = Instant::now();
        loop {
            if let Some(status) = child.try_wait()? {
                if !status.success() {
                    let stderr_bytes = stderr_reader.join().unwrap_or_default();
                    let stderr = String::from_utf8_lossy(&stderr_bytes);
                    anyhow::bail!("spaCy subprocess failed: {}", stderr.trim());
                }
                break;
            }
            if start.elapsed() >= timeout {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
                anyhow::bail!("spaCy subprocess timed out after {}s", timeout.as_secs());
            }
            sleep(Duration::from_millis(50));
        }
        let stdout = stdout_reader.join().unwrap_or_default();
        let stderr = stderr_reader.join().unwrap_or_default();
        if !child.try_wait()?.is_some_and(|status| status.success()) {
            anyhow::bail!(
                "spaCy subprocess failed: {}",
                String::from_utf8_lossy(&stderr).trim()
            );
        }
        let parsed: SpacyBatchOutput = serde_json::from_slice(&stdout)?;
        let mut out: HashMap<String, Vec<Tier1Entity>> = HashMap::new();
        for doc in parsed.documents {
            let entities = doc
                .entities
                .into_iter()
                .map(|e| Tier1Entity {
                    text: e.text,
                    label: e.label,
                    start: e.start,
                    end: e.end,
                    score: e.score,
                    source: "spacy".to_string(),
                })
                .collect();
            out.insert(doc.id, entities);
        }
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "spacy"
    }
}

fn write_spacy_input<W: Write>(mut writer: W, payload: &[u8]) -> Result<()> {
    writer.write_all(payload)?;
    Ok(())
}

#[cfg(test)]
mod subprocess_tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    struct DropProbe(Arc<AtomicBool>);
    impl Write for DropProbe {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn spacy_input_writer_is_closed_after_payload() {
        let dropped = Arc::new(AtomicBool::new(false));
        write_spacy_input(DropProbe(dropped.clone()), b"{}").unwrap();
        assert!(dropped.load(Ordering::SeqCst));
    }
}

/// Stopwords for the YAKE/RAKE/TextRank term rankers, per language.
/// The English base is the canonical spaCy list
/// (`crate::tokenizer::english_stopwords`); Chinese/Korean function words
/// can never collide with Latin tokens so they are unioned for every
/// language; Spanish is added only for Spanish docs — its words collide
/// with English (`no`, `son`, `era`).
fn default_stopwords_for_lang(lang: Lang) -> HashSet<String> {
    let mut stop: HashSet<String> = crate::tokenizer::english_stopwords()
        .iter()
        .map(|s| s.to_string())
        .collect();
    stop.extend(
        crate::tokenizer::chinese_stopwords()
            .iter()
            .map(|s| s.to_string()),
    );
    stop.extend(
        crate::tokenizer::korean_stopwords()
            .iter()
            .map(|s| s.to_string()),
    );
    if matches!(lang, Lang::Es) {
        stop.extend(crate::tokenizer::spanish_stopwords().iter().cloned());
    }
    stop
}

fn tokenize_words(content: &str) -> Vec<String> {
    // Unstemmed mode keeps the exact historical Latin behavior (the shared
    // tokenizer's Latin path matches the old content-word regex
    // `[A-Za-z][A-Za-z0-9_-]{2,}`) and adds Han bigrams / Hangul eojeol
    // for CJK text.
    crate::tokenizer::tokenize(content, crate::tokenizer::TokenizerMode::Unstemmed)
}

/// RAKE tokens: like the shared tokenizer, but keeps RAKE's historical
/// `{1,}` Latin minimum (so 2-letter English tokens still count) instead
/// of the shared `{2,}`. Script-aware single pass so mixed-language order
/// is preserved for phrase building.
fn rake_tokens(content: &str) -> Vec<String> {
    let rake_re = rake_token_regex();
    let mut out = Vec::new();
    let mut latin = String::new();
    let mut han = String::new();
    let mut hangul = String::new();
    let flush_latin = |latin: &mut String, out: &mut Vec<String>| {
        for m in rake_re.find_iter(latin) {
            out.push(m.as_str().to_lowercase());
        }
        latin.clear();
    };
    for ch in content.chars() {
        if crate::lang::is_han(ch) {
            flush_latin(&mut latin, &mut out);
            if !hangul.is_empty() {
                for t in crate::tokenizer::hangul_eojeol_tokens(&hangul) {
                    out.push(t);
                }
                hangul.clear();
            }
            han.push(ch);
        } else if crate::lang::is_hangul(ch) {
            flush_latin(&mut latin, &mut out);
            if !han.is_empty() {
                out.extend(crate::tokenizer::han_tokens(&han));
                han.clear();
            }
            hangul.push(ch);
        } else {
            if !han.is_empty() {
                out.extend(crate::tokenizer::han_tokens(&han));
                han.clear();
            }
            if !hangul.is_empty() {
                for t in crate::tokenizer::hangul_eojeol_tokens(&hangul) {
                    out.push(t);
                }
                hangul.clear();
            }
            latin.push(ch);
        }
    }
    flush_latin(&mut latin, &mut out);
    if !han.is_empty() {
        out.extend(crate::tokenizer::han_tokens(&han));
    }
    if !hangul.is_empty() {
        for t in crate::tokenizer::hangul_eojeol_tokens(&hangul) {
            out.push(t);
        }
    }
    out
}

fn sentence_count(content: &str) -> usize {
    let count = content
        .split(SENTENCE_ENDINGS)
        .filter(|s| !s.trim().is_empty())
        .count();
    count.max(1)
}

fn sorted_terms(mut terms: Vec<RankedTerm>, top_k: usize) -> Vec<RankedTerm> {
    terms.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    terms.truncate(top_k);
    terms
}

/// True when Han characters make up at least half of the content's
/// non-whitespace characters. Han-dominant docs get a quadrupled term
/// budget: the interleaved unigram+bigram token stream is ~2x the tokens
/// of the same text without unigrams, and Chinese topic-comment order puts
/// the distinctive payload late — a tight budget with an early-position
/// bias truncates exactly the terms questions ask about.
pub(crate) fn han_dominant_content(content: &str) -> bool {
    let mut han = 0usize;
    let mut total = 0usize;
    for ch in content.chars() {
        if ch.is_whitespace() {
            continue;
        }
        total += 1;
        if crate::lang::is_han(ch) {
            han += 1;
        }
    }
    total > 0 && han * 2 >= total
}

/// Term budget for [`sorted_terms`]: quadrupled for Han-dominant content.
/// The interleaved unigram+bigram token stream is ~2x the tokens of the
/// same text without unigrams, and Chinese topic-comment order puts the
/// distinctive payload late — a tight budget with an early-position bias
/// truncates exactly the terms questions ask about (verified on the
/// Chinese memory benchmark: budget 24 left q15/q16/q19/q25 failing).
pub(crate) fn term_budget_for(content: &str) -> usize {
    if han_dominant_content(content) {
        48
    } else {
        12
    }
}

pub struct YakeStyleTermRanker;

impl ImportantTermRanker for YakeStyleTermRanker {
    fn name(&self) -> &'static str {
        "yake-style"
    }

    fn rank_terms(&self, doc: &Tier1DocInput) -> Vec<RankedTerm> {
        let stop = default_stopwords_for_lang(Lang::Auto.resolve(&doc.content));
        let raw_tokens = tokenize_words(&doc.content);
        let total = raw_tokens.len().max(1) as f32;
        let sentences = sentence_count(&doc.content) as f32;
        let mut freq: HashMap<String, usize> = HashMap::new();
        let mut first_pos: HashMap<String, usize> = HashMap::new();
        let mut sent_hits: HashMap<String, usize> = HashMap::new();

        for (i, t) in raw_tokens.iter().enumerate() {
            if stop.contains(t.as_str()) {
                continue;
            }
            *freq.entry(t.clone()).or_insert(0) += 1;
            first_pos.entry(t.clone()).or_insert(i);
        }
        for sent in doc.content.split(SENTENCE_ENDINGS) {
            let s_tokens = tokenize_words(sent);
            let unique: HashSet<String> = s_tokens.into_iter().collect();
            for t in unique {
                if stop.contains(t.as_str()) {
                    continue;
                }
                *sent_hits.entry(t).or_insert(0) += 1;
            }
        }

        let mut out = Vec::new();
        for (term, count) in freq {
            let pos = *first_pos.get(&term).unwrap_or(&0) as f32 / total;
            let pos_bonus = 1.0 + (1.0 - pos);
            let disp = *sent_hits.get(&term).unwrap_or(&1) as f32 / sentences;
            let disp_bonus = 1.0 + disp;
            let score = (count as f32 / total) * pos_bonus * disp_bonus * 100.0;
            out.push(RankedTerm {
                term,
                score,
                source: self.name().to_string(),
            });
        }
        sorted_terms(out, term_budget_for(&doc.content))
    }
}

pub struct RakeStyleTermRanker;

impl ImportantTermRanker for RakeStyleTermRanker {
    fn name(&self) -> &'static str {
        "rake-style"
    }

    fn rank_terms(&self, doc: &Tier1DocInput) -> Vec<RankedTerm> {
        let stop = default_stopwords_for_lang(Lang::Auto.resolve(&doc.content));
        let tokens: Vec<String> = rake_tokens(&doc.content);
        let mut phrases: Vec<Vec<String>> = Vec::new();
        let mut current = Vec::new();
        for t in tokens {
            if stop.contains(t.as_str()) {
                if !current.is_empty() {
                    phrases.push(current.clone());
                    current.clear();
                }
                continue;
            }
            current.push(t);
        }
        if !current.is_empty() {
            phrases.push(current);
        }

        let mut freq: HashMap<String, usize> = HashMap::new();
        let mut degree: HashMap<String, usize> = HashMap::new();
        for phrase in &phrases {
            let len = phrase.len();
            if len == 0 || len > 5 {
                continue;
            }
            for w in phrase {
                *freq.entry(w.clone()).or_insert(0) += 1;
                *degree.entry(w.clone()).or_insert(0) += len.saturating_sub(1);
            }
        }
        let mut word_score: HashMap<String, f32> = HashMap::new();
        for (w, f) in freq {
            let d = degree.get(&w).copied().unwrap_or(0) + f;
            word_score.insert(w, d as f32 / f as f32);
        }

        let mut phrase_scores: HashMap<String, f32> = HashMap::new();
        for phrase in phrases {
            if phrase.is_empty() || phrase.len() > 5 {
                continue;
            }
            let key = phrase.join(" ");
            if key.len() < 4 {
                continue;
            }
            let score: f32 = phrase
                .iter()
                .map(|w| word_score.get(w).copied().unwrap_or(0.0))
                .sum();
            *phrase_scores.entry(key).or_insert(0.0) += score;
        }

        let out = phrase_scores
            .into_iter()
            .map(|(term, score)| RankedTerm {
                term,
                score,
                source: self.name().to_string(),
            })
            .collect();
        sorted_terms(out, term_budget_for(&doc.content))
    }
}

pub struct CValueStyleTermRanker;

impl ImportantTermRanker for CValueStyleTermRanker {
    fn name(&self) -> &'static str {
        "cvalue-style"
    }

    fn rank_terms(&self, doc: &Tier1DocInput) -> Vec<RankedTerm> {
        let tokens = tokenize_words(&doc.content);
        let mut ngram_freq: HashMap<String, usize> = HashMap::new();
        for n in 2..=4 {
            for win in tokens.windows(n) {
                let phrase = win.join(" ");
                *ngram_freq.entry(phrase).or_insert(0) += 1;
            }
        }

        let entries: Vec<(String, usize)> =
            ngram_freq.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let mut out = Vec::new();
        for (phrase, freq) in &entries {
            let words = phrase.split_whitespace().count();
            if words < 2 {
                continue;
            }
            let mut longer_sum = 0usize;
            let mut longer_count = 0usize;
            for (other, other_freq) in &entries {
                if other.len() > phrase.len() && other.contains(phrase) {
                    longer_sum += *other_freq;
                    longer_count += 1;
                }
            }
            let nested_penalty = if longer_count > 0 {
                longer_sum as f32 / longer_count as f32
            } else {
                0.0
            };
            let score = (words as f32).log2() * (*freq as f32 - nested_penalty).max(0.0);
            if score > 0.0 {
                out.push(RankedTerm {
                    term: phrase.clone(),
                    score,
                    source: self.name().to_string(),
                });
            }
        }
        sorted_terms(out, term_budget_for(&doc.content))
    }
}

pub struct TextRankStyleTermRanker;

impl ImportantTermRanker for TextRankStyleTermRanker {
    fn name(&self) -> &'static str {
        "textrank-style"
    }

    fn rank_terms(&self, doc: &Tier1DocInput) -> Vec<RankedTerm> {
        let stop = default_stopwords_for_lang(Lang::Auto.resolve(&doc.content));
        let tokens: Vec<String> = tokenize_words(&doc.content)
            .into_iter()
            .filter(|t| !stop.contains(t.as_str()))
            .collect();

        let mut neighbors: HashMap<String, HashSet<String>> = HashMap::new();
        for win in tokens.windows(3) {
            for i in 0..win.len() {
                for j in 0..win.len() {
                    if i == j {
                        continue;
                    }
                    neighbors
                        .entry(win[i].clone())
                        .or_default()
                        .insert(win[j].clone());
                }
            }
        }

        let mut score: HashMap<String, f32> = neighbors.keys().map(|k| (k.clone(), 1.0)).collect();
        for _ in 0..20 {
            let mut next = HashMap::new();
            for node in neighbors.keys() {
                let mut s = 0.15;
                if let Some(neis) = neighbors.get(node) {
                    for n in neis {
                        let deg = neighbors.get(n).map(|x| x.len()).unwrap_or(1) as f32;
                        let contrib = score.get(n).copied().unwrap_or(1.0) / deg;
                        s += 0.85 * contrib;
                    }
                }
                next.insert(node.clone(), s);
            }
            score = next;
        }

        let out = score
            .into_iter()
            .map(|(term, s)| RankedTerm {
                term,
                score: s,
                source: self.name().to_string(),
            })
            .collect();
        sorted_terms(out, term_budget_for(&doc.content))
    }
}

#[cfg(test)]
mod cjk_term_tests {
    use super::*;

    fn doc(content: &str) -> Tier1DocInput {
        Tier1DocInput {
            id: "t".to_string(),
            source: "test".to_string(),
            content: content.to_string(),
            concept: String::new(),
            headings: Vec::new(),
        }
    }

    #[test]
    fn chinese_content_terms_are_ranked() {
        let terms = YakeStyleTermRanker.rank_terms(&doc("我毕业于清华大学，专业是计算机科学。"));
        let names: Vec<&str> = terms.iter().map(|t| t.term.as_str()).collect();
        assert!(
            names.iter().any(|t| t.contains("清华")),
            "expected a 清华 bigram in ranked terms, got {names:?}"
        );
        // Chinese function words must not surface as content terms.
        assert!(
            !names.iter().any(|t| ["的", "了", "在", "是"].contains(t)),
            "stopwords leaked into terms: {names:?}"
        );
    }

    #[test]
    fn chinese_sentence_splitting_counts_cjk_boundaries() {
        assert_eq!(sentence_count("第一句。第二句！第三句？"), 3);
        assert_eq!(sentence_count("第一句。第二句!"), 2);
        assert_eq!(sentence_count("没有标点"), 1);
    }

    #[test]
    fn rake_tokens_emit_chinese_bigrams() {
        let toks = rake_tokens("我喜欢学习Rust编程");
        assert!(
            toks.iter().any(|t| t == "喜欢"),
            "expected 喜欢 bigram, got {toks:?}"
        );
        assert!(
            toks.iter().any(|t| t == "rust"),
            "expected rust latin token, got {toks:?}"
        );
    }

    #[test]
    fn rake_keeps_two_letter_latin_tokens() {
        let toks = rake_tokens("AI is here");
        assert!(
            toks.iter().any(|t| t == "ai"),
            "RAKE must keep 2-letter Latin tokens, got {toks:?}"
        );
    }

    #[test]
    fn english_ranking_unchanged_by_cjk_work() {
        // Guard: shared-tokenizer switch must not alter Latin behavior.
        let terms = YakeStyleTermRanker.rank_terms(&doc(
            "The certificate program awarded a degree in computer science.",
        ));
        let names: Vec<&str> = terms.iter().map(|t| t.term.as_str()).collect();
        assert!(
            names.iter().any(|t| *t == "certificate" || *t == "degree"),
            "expected content terms, got {names:?}"
        );
    }

    #[test]
    fn han_dominant_detection() {
        assert!(han_dominant_content("八月二十号我改主意了，最后提了一辆比亚迪海豹。"));
        assert!(han_dominant_content("我毕业于清华大学，专业是计算机科学。"));
        assert!(!han_dominant_content(
            "The certificate program awarded a degree in computer science."
        ));
        assert!(!han_dominant_content(""));
        // Mixed: 3 Han of 9 non-ws chars -> not dominant.
        assert!(!han_dominant_content("买特斯拉 Model Y"));
    }

    #[test]
    fn chinese_generous_budget_keeps_late_payload_terms() {
        // Regression: the tight top-12 budget with an early-position bias
        // truncated the distinctive late terms Chinese questions ask about
        // (topic-comment order). Han-dominant docs get a 48-term budget so
        // the payload — 比亚迪海豹, 现在 — survives ranking.
        let terms = YakeStyleTermRanker.rank_terms(&doc(
            "八月二十号我改主意了，最后提了一辆比亚迪海豹，现在每天开着上下班。",
        ));
        let names: Vec<&str> = terms.iter().map(|t| t.term.as_str()).collect();
        // 比亚迪海豹 segments as 比亚/亚迪/海豹 bigrams (+ unigrams); the
        // payload must survive ranking regardless of segmentation.
        for want in ["比亚", "亚迪", "海豹", "现在", "下班"] {
            assert!(
                names.contains(&want),
                "expected late payload term {want:?} in top terms, got {names:?}"
            );
        }
    }

    #[test]
    fn chinese_term_budget_is_quadrupled() {
        assert_eq!(term_budget_for("八月二十号我改主意了。"), 48);
        assert_eq!(term_budget_for("The quick brown fox."), 12);
    }

    #[test]
    fn chinese_unigram_query_term_is_indexable() {
        // 猫 (single char) must be rankable so it can match 橘猫's unigram.
        let terms = YakeStyleTermRanker.rank_terms(&doc("家里养了一只橘猫，名字叫年糕。"));
        let names: Vec<&str> = terms.iter().map(|t| t.term.as_str()).collect();
        assert!(
            names.contains(&"猫"),
            "expected 猫 unigram among ranked terms, got {names:?}"
        );
    }
}

#[cfg(test)]
mod spacy_chinese_tests {
    use super::*;

    /// Chinese NER through the real `spacy_ner.py` with the Rust-selected
    /// `zh_core_web_sm` model. Skips gracefully when spaCy is unavailable;
    /// run with PYTHON_EXECUTABLE=~/workspace/venvs/spacy-ner/bin/python
    /// for the isolated venv that carries the model.
    #[test]
    fn spacy_chinese_ner_extracts_entities() {
        let ranker = SpacyKeyEntityRanker {
            model: "zh_core_web_sm".to_string(),
            script_path: default_spacy_script_path().to_string_lossy().to_string(),
        };
        let docs = vec![Tier1DocInput {
            id: "zh-ner-1".to_string(),
            source: "test".to_string(),
            content: "我毕业于清华大学，专业是计算机科学。".to_string(),
            concept: String::new(),
            headings: Vec::new(),
        }];
        let result = match ranker.rank_docs(&docs) {
            Ok(map) => map,
            Err(e) => {
                println!("SKIPPED: spaCy NER unavailable ({e})");
                return;
            }
        };
        let entities = result.get("zh-ner-1").cloned().unwrap_or_default();
        println!("Chinese NER entities: {entities:?}");
        assert!(
            entities.iter().any(|e| e.text.contains("清华大学")),
            "expected 清华大学 entity, got {entities:?}"
        );
    }
}

#[cfg(test)]
mod stopword_tests {
    use super::*;

    #[test]
    fn per_language_stopwords_use_canonical_lists() {
        let en = default_stopwords_for_lang(Lang::En);
        let es = default_stopwords_for_lang(Lang::Es);
        // Canonical spaCy English base in both.
        for w in ["the", "however", "therefore"] {
            assert!(en.contains(w), "{w} should stop in English");
            assert!(es.contains(w), "{w} should stop in Spanish docs too");
        }
        // CJK unions apply to every language (no collision possible).
        for w in ["的", "은"] {
            assert!(en.contains(w), "{w} should stop in English");
            assert!(es.contains(w), "{w} should stop in Spanish");
        }
        // Spanish gated on Lang::Es: "son"/"era" collide with English words.
        for w in ["está", "esta", "son", "era"] {
            assert!(es.contains(w), "{w} should stop for Spanish docs");
            assert!(!en.contains(w), "{w} must not stop for English docs");
        }
        // "now" is not a stopword: it anchors current-state retrieval
        // ("what is X now" must keep the temporal signal).
        assert!(
            !en.contains("now"),
            "\"now\" must not be an English stopword"
        );
    }

    #[test]
    fn auto_detect_includes_spanish_signals() {
        // Auto-detection uses Spanish signals (accents, ñ, ¿¡) for Latin
        // text, so a Spanish doc via Auto gets the Spanish stop set.
        let es_doc = Tier1DocInput {
            id: "1".into(),
            source: "t".into(),
            content: "El niño está en la escuela porque tiene clases".into(),
            concept: "".into(),
            headings: vec![],
        };
        let stop = default_stopwords_for_lang(Lang::Auto.resolve(&es_doc.content));
        assert!(stop.contains("está"));
        let stop_es = default_stopwords_for_lang(Lang::Es);
        assert!(stop_es.contains("está"));
        // English content words that collide with Spanish stopwords survive.
        let en_doc = Tier1DocInput {
            id: "2".into(),
            source: "t".into(),
            content: "The son went to school in an era of change".into(),
            concept: "".into(),
            headings: vec![],
        };
        let stop_en = default_stopwords_for_lang(Lang::Auto.resolve(&en_doc.content));
        assert!(!stop_en.contains("son"), "English 'son' must survive");
        assert!(!stop_en.contains("era"), "English 'era' must survive");
    }
    use super::*;

    #[test]
    fn spanish_folded_twins_are_stopped() {
        // Dual emission means ranker tokens carry both "está" and "esta";
        // the folded twins of Spanish stopwords must not leak through as
        // content terms.
        let stop = default_stopwords_for_lang(Lang::Es);
        for w in ["sí", "está", "están", "más", "también", "dónde", "qué"] {
            assert!(stop.contains(w), "{w} (raw) not stopped");
            let folded = crate::tokenizer::fold_diacritics(w);
            assert!(
                stop.contains(folded.as_str()),
                "{folded} (folded twin of {w}) not stopped"
            );
        }
        // Every dual-emitted token of a Spanish stopword is covered:
        // tokenize each stopword and check all emissions are stopped.
        for w in ["niño", "está", "dónde"] {
            for t in crate::tokenizer::tokenize(w, crate::tokenizer::TokenizerMode::Unstemmed) {
                // "niño" is content (not a stopword) — only its forms must
                // agree; skip the content word itself.
                if w == "niño" {
                    continue;
                }
                assert!(stop.contains(t.as_str()), "emission {t} of {w} not stopped");
            }
        }
    }

    #[test]
    fn english_stopwords_unchanged() {
        // The English base is untouched by the per-language extension.
        let stop = default_stopwords_for_lang(Lang::En);
        for w in ["the", "and", "of", "is"] {
            assert!(stop.contains(w));
        }
        assert!(!stop.contains("está"));
        assert!(!stop.contains("sí"));
    }
}
