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
    // SPACY_PYTHON: explicit override for the spaCy NER subprocess python,
    // following the BEHOOD_BIN precedent. Points at a Python with spaCy
    // installed (e.g. a uv venv whose own site-packages survive the -I
    // isolated flag below; user site-packages do not). Checked before the
    // legacy PYTHON_EXECUTABLE / PYTHON overrides.
    if let Ok(value) = std::env::var("SPACY_PYTHON") {
        let value = value.trim();
        if !value.is_empty() {
            return value.to_string();
        }
    }
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

fn content_word_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Za-z][A-Za-z0-9_-]{2,}").expect("valid regex"))
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

#[derive(Serialize)]
pub(crate) struct SpacyDocInput<'a> {
    pub(crate) id: &'a str,
    pub(crate) text: &'a str,
}

#[derive(Serialize)]
pub(crate) struct SpacyBatchInput<'a> {
    pub(crate) model: &'a str,
    pub(crate) documents: Vec<SpacyDocInput<'a>>,
}

#[derive(Deserialize)]
pub(crate) struct SpacyBatchOutput {
    pub(crate) documents: Vec<SpacyDocOutput>,
}

#[derive(Deserialize)]
pub(crate) struct SpacyDocOutput {
    pub(crate) id: String,
    pub(crate) entities: Vec<SpacyEntityOutput>,
}

#[derive(Deserialize)]
pub(crate) struct SpacyEntityOutput {
    pub(crate) text: String,
    pub(crate) label: String,
    pub(crate) start: usize,
    pub(crate) end: usize,
    #[serde(default)]
    pub(crate) score: Option<f32>,
}

impl KeyEntityRanker for SpacyKeyEntityRanker {
    fn rank_docs(&self, docs: &[Tier1DocInput]) -> Result<HashMap<String, Vec<Tier1Entity>>> {
        // Fast path: the long-lived NER daemon keeps the spaCy model loaded
        // across calls, so repeated rankings (per-document adds, benchmark
        // batches) pay the interpreter + model load once per process instead
        // of once per call. Fail-open: any daemon failure (missing script,
        // dead child, timeout, lock contention) falls through to the
        // one-shot subprocess below, exactly as before.
        //
        // Only the default script goes through the process-wide daemon; a
        // custom script_path always uses the one-shot path.
        if self.script_path == default_spacy_script_path().display().to_string() {
            let timeout = Duration::from_secs(SPACY_SUBPROCESS_TIMEOUT_SECS);
            if let Some(out) =
                crate::tier1_ner_daemon::NerDaemon::global().rank(&self.model, docs, timeout)
            {
                return Ok(out);
            }
        }
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

fn default_stopwords() -> HashSet<&'static str> {
    let mut set: HashSet<&'static str> = [
        "a", "an", "the", "is", "are", "was", "were", "be", "to", "for", "of", "on", "in", "by",
        "as", "or", "and", "that", "this", "with", "from", "it", "its", "at", "into", "about",
        "over", "under", "also", "can", "could", "should", "would", "will", "may", "might", "do",
        "does", "did", "done", "not", "no", "yes", "if", "then", "than", "there", "their", "we",
        "you", "they", "he", "she", "them", "our", "your",
    ]
    .iter()
    .copied()
    .collect();
    // Korean particles/function words: without these, the term ranker
    // would surface e.g. "것" or "수" as top terms for Korean docs.
    set.extend(crate::tokenizer::korean_stopwords().iter().copied());
    set
}

fn tokenize_words(content: &str) -> Vec<String> {
    let mut tokens: Vec<String> = content_word_regex()
        .find_iter(content)
        .map(|m| m.as_str().to_lowercase())
        .collect();
    // The Latin regex skips Hangul/Han runs entirely; add them via the
    // shared script-aware tokenizer so Korean/Chinese terms participate
    // in term ranking (Han → bigrams, Hangul → eojeol + stem).
    tokens.extend(
        crate::tokenizer::tokenize(content, crate::tokenizer::TokenizerMode::Unstemmed)
            .into_iter()
            .filter(|t| {
                t.chars()
                    .any(|c| crate::lang::is_han(c) || crate::lang::is_hangul(c))
            }),
    );
    tokens
}

fn sentence_count(content: &str) -> usize {
    let count = content
        .split(['.', '!', '?'])
        .filter(|s| !s.trim().is_empty())
        .count();
    count.max(1)
}

fn sorted_terms(mut terms: Vec<RankedTerm>, top_k: usize) -> Vec<RankedTerm> {
    terms.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    terms.truncate(top_k);
    terms
}

pub struct YakeStyleTermRanker;

impl ImportantTermRanker for YakeStyleTermRanker {
    fn name(&self) -> &'static str {
        "yake-style"
    }

    fn rank_terms(&self, doc: &Tier1DocInput) -> Vec<RankedTerm> {
        let stop = default_stopwords();
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
        for sent in doc.content.split(['.', '!', '?']) {
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
        sorted_terms(out, 12)
    }
}

pub struct RakeStyleTermRanker;

impl ImportantTermRanker for RakeStyleTermRanker {
    fn name(&self) -> &'static str {
        "rake-style"
    }

    fn rank_terms(&self, doc: &Tier1DocInput) -> Vec<RankedTerm> {
        let stop = default_stopwords();
        let tokens: Vec<String> = rake_token_regex()
            .find_iter(&doc.content)
            .map(|m| m.as_str().to_lowercase())
            .collect();
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
        sorted_terms(out, 12)
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
        sorted_terms(out, 12)
    }
}

pub struct TextRankStyleTermRanker;

impl ImportantTermRanker for TextRankStyleTermRanker {
    fn name(&self) -> &'static str {
        "textrank-style"
    }

    fn rank_terms(&self, doc: &Tier1DocInput) -> Vec<RankedTerm> {
        let stop = default_stopwords();
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
        sorted_terms(out, 12)
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
