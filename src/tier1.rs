use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

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

#[derive(Serialize)]
pub struct Tier1DocEntities {
    pub id: String,
    pub source: String,
    pub key_entities: Vec<Tier1Entity>,
}

#[derive(Debug, Clone, Serialize)]
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

        if let Ok(title_case_re) = Regex::new(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b") {
            for cap in title_case_re.captures_iter(&doc.content).take(80) {
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
        }

        if let Ok(acronym_re) = Regex::new(r"\b([A-Z]{2,8})\b") {
            for m in acronym_re.find_iter(&doc.content).take(50) {
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

        let mut child = Command::new("python3")
            .arg(&self.script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        if let Some(stdin) = child.stdin.as_mut() {
            stdin.write_all(input_json.as_bytes())?;
        }
        let timeout = Duration::from_secs(20);
        let start = Instant::now();
        loop {
            if let Some(status) = child.try_wait()? {
                if !status.success() {
                    let output = child.wait_with_output()?;
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    anyhow::bail!("spaCy subprocess failed: {}", stderr.trim());
                }
                break;
            }
            if start.elapsed() >= timeout {
                let _ = child.kill();
                let _ = child.wait();
                anyhow::bail!("spaCy subprocess timed out after {}s", timeout.as_secs());
            }
            sleep(Duration::from_millis(50));
        }
        let output = child.wait_with_output()?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("spaCy subprocess failed: {}", stderr.trim());
        }
        let parsed: SpacyBatchOutput = serde_json::from_slice(&output.stdout)?;
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

fn default_stopwords() -> HashSet<&'static str> {
    [
        "a", "an", "the", "is", "are", "was", "were", "be", "to", "for", "of", "on", "in", "by",
        "as", "or", "and", "that", "this", "with", "from", "it", "its", "at", "into", "about",
        "over", "under", "also", "can", "could", "should", "would", "will", "may", "might", "do",
        "does", "did", "done", "not", "no", "yes", "if", "then", "than", "there", "their", "we",
        "you", "they", "he", "she", "them", "our", "your",
    ]
    .iter()
    .copied()
    .collect()
}

fn tokenize_words(content: &str) -> Vec<String> {
    let re = Regex::new(r"[A-Za-z][A-Za-z0-9_-]{2,}").expect("valid regex");
    re.find_iter(content)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

fn sentence_count(content: &str) -> usize {
    let count = content
        .split(|c: char| c == '.' || c == '!' || c == '?')
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
        for sent in doc
            .content
            .split(|c: char| c == '.' || c == '!' || c == '?')
        {
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
        let token_re = Regex::new(r"[A-Za-z][A-Za-z0-9_-]{1,}").expect("valid regex");
        let tokens: Vec<String> = token_re
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
}
