use crate::tier1::{RankedTerm, Tier1Entity};
use anyhow::Result;
use deunicode::deunicode;
use regex::Regex;
use serde::Serialize;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::document::TantivyDocument;
use tantivy::schema::{Field, Schema, STORED, STRING, TEXT};
use tantivy::schema::Value;
use tantivy::{doc, Index, IndexReader};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize)]
pub struct Provenance {
    pub source: String,
    pub timestamp: Option<String>,
    pub ner_provider: String,
    pub term_ranker: String,
    pub index_version: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Claim {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct DocRecord {
    pub doc_id: String,
    pub source: String,
    #[serde(skip_serializing)]
    pub content: String,
    pub timestamp: Option<String>,
    pub doc_length: usize,
    pub author_agent: Option<String>,
    pub probable_topic: Option<String>,
    pub doc_type_guess: Option<String>,
    pub headings: Vec<String>,
    pub key_entities: Vec<Tier1Entity>,
    pub important_terms: Vec<RankedTerm>,
    pub embedding: Option<Vec<f32>>,
    pub top_claims: Vec<Claim>,
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Serialize)]
pub struct EntityPosting {
    pub doc_id: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct TermPosting {
    pub doc_id: String,
    pub score: f32,
}

#[derive(Serialize)]
pub struct MemoryIndex {
    pub docs: HashMap<String, DocRecord>,
    pub entity_to_docs: HashMap<String, Vec<EntityPosting>>,
    pub term_to_docs: HashMap<String, Vec<TermPosting>>,
    pub topic_to_docs: HashMap<String, Vec<String>>,
    pub doc_type_to_docs: HashMap<String, Vec<String>>,
    #[serde(skip_serializing)]
    lexical: Option<LexicalIndex>,
}

struct LexicalIndex {
    index: Index,
    reader: IndexReader,
    doc_id_f: Field,
    content_f: Field,
    headings_f: Field,
    terms_f: Field,
    entities_f: Field,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ScoreBreakdown {
    pub lexical_score: f32,
    pub entity_score: f32,
    pub term_score: f32,
    pub topic_score: f32,
    pub doc_type_score: f32,
    pub recency_score: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub doc_id: String,
    pub source: String,
    pub score: f32,
    pub score_breakdown: ScoreBreakdown,
    pub matched_entities: Vec<String>,
    pub matched_terms: Vec<String>,
    pub probable_topic: Option<String>,
    pub doc_type_guess: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RedactedDocRecord {
    pub doc_id: String,
    pub source: String,
    pub timestamp: Option<String>,
    pub doc_length: usize,
    pub probable_topic: Option<String>,
    pub doc_type_guess: Option<String>,
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Serialize)]
pub struct RedactedMemoryIndex {
    pub docs: HashMap<String, RedactedDocRecord>,
    pub topic_to_docs: HashMap<String, Vec<String>>,
    pub doc_type_to_docs: HashMap<String, Vec<String>>,
}

impl MemoryIndex {
    pub fn from_records(records: Vec<DocRecord>) -> Self {
        let mut docs = HashMap::new();
        let mut entity_to_docs: HashMap<String, Vec<EntityPosting>> = HashMap::new();
        let mut term_to_docs: HashMap<String, Vec<TermPosting>> = HashMap::new();
        let mut topic_to_docs: HashMap<String, Vec<String>> = HashMap::new();
        let mut doc_type_to_docs: HashMap<String, Vec<String>> = HashMap::new();

        for record in records {
            let doc_id = record.doc_id.clone();
            for entity in &record.key_entities {
                let key = normalize_for_index(&entity.text);
                if key.is_empty() {
                    continue;
                }
                let score = entity.score.unwrap_or(0.5);
                entity_to_docs.entry(key).or_default().push(EntityPosting {
                    doc_id: doc_id.clone(),
                    score,
                });
            }
            for term in &record.important_terms {
                let key = term.term.to_lowercase();
                term_to_docs.entry(key).or_default().push(TermPosting {
                    doc_id: doc_id.clone(),
                    score: term.score,
                });
            }
            for heading in &record.headings {
                let heading_tokens = tokenize_query_terms(heading);
                for token in heading_tokens {
                    term_to_docs.entry(token).or_default().push(TermPosting {
                        doc_id: doc_id.clone(),
                        score: 0.4,
                    });
                }
            }
            if let Some(topic) = record.probable_topic.as_ref() {
                topic_to_docs
                    .entry(topic.to_lowercase())
                    .or_default()
                    .push(doc_id.clone());
            }
            if let Some(doc_type) = record.doc_type_guess.as_ref() {
                doc_type_to_docs
                    .entry(doc_type.to_lowercase())
                    .or_default()
                    .push(doc_id.clone());
            }
            docs.insert(doc_id, record);
        }

        for postings in entity_to_docs.values_mut() {
            postings.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        for postings in term_to_docs.values_mut() {
            postings.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let lexical = match Self::build_lexical_index(&docs) {
            Ok(index) => Some(index),
            Err(err) => {
                eprintln!("warning: lexical BM25 index disabled: {}", err);
                None
            }
        };

        Self {
            docs,
            entity_to_docs,
            term_to_docs,
            topic_to_docs,
            doc_type_to_docs,
            lexical,
        }
    }

    fn build_lexical_index(docs: &HashMap<String, DocRecord>) -> Result<LexicalIndex> {
        let mut schema_builder = Schema::builder();
        let doc_id_f = schema_builder.add_text_field("doc_id", STRING | STORED);
        let content_f = schema_builder.add_text_field("content", TEXT);
        let headings_f = schema_builder.add_text_field("headings", TEXT);
        let terms_f = schema_builder.add_text_field("important_terms", TEXT);
        let entities_f = schema_builder.add_text_field("entities", TEXT);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        let mut writer = index.writer(50_000_000)?;
        for doc in docs.values() {
            let headings_text = doc.headings.join(" ");
            let terms_text = doc
                .important_terms
                .iter()
                .map(|t| t.term.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let entities_text = doc
                .key_entities
                .iter()
                .map(|e| e.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            writer.add_document(doc!(
                doc_id_f => doc.doc_id.clone(),
                content_f => doc.content.clone(),
                headings_f => headings_text,
                terms_f => terms_text,
                entities_f => entities_text
            ))?;
        }
        writer.commit()?;
        let reader = index.reader()?;
        Ok(LexicalIndex {
            index,
            reader,
            doc_id_f,
            content_f,
            headings_f,
            terms_f,
            entities_f,
        })
    }

    fn lexical_bm25(&self, query: &str, top_k: usize) -> Result<HashMap<String, f32>> {
        let Some(lex) = self.lexical.as_ref() else {
            return Ok(HashMap::new());
        };
        let searcher = lex.reader.searcher();
        let mut query_parser =
            QueryParser::for_index(&lex.index, vec![lex.content_f, lex.headings_f, lex.terms_f, lex.entities_f]);
        query_parser.set_field_boost(lex.content_f, 1.0);
        query_parser.set_field_boost(lex.headings_f, 1.4);
        query_parser.set_field_boost(lex.terms_f, 2.0);
        query_parser.set_field_boost(lex.entities_f, 2.4);
        let parsed = query_parser.parse_query(query)?;
        let top_docs = searcher.search(&parsed, &TopDocs::with_limit(top_k))?;

        let mut out = HashMap::new();
        for (score, addr) in top_docs {
            let retrieved: TantivyDocument = searcher.doc(addr)?;
            if let Some(v) = retrieved.get_first(lex.doc_id_f) {
                if let Some(doc_id) = v.as_str() {
                    out.insert(doc_id.to_string(), score);
                }
            }
        }
        Ok(out)
    }

    pub fn query(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        const MAX_QUERY_CHARS: usize = 4096;
        const MAX_QUERY_TOKENS: usize = 128;

        let truncated = if query.len() > MAX_QUERY_CHARS {
            let mut cut = 0usize;
            for (idx, _) in query.char_indices() {
                if idx > MAX_QUERY_CHARS {
                    break;
                }
                cut = idx;
            }
            &query[..cut]
        } else {
            query
        };
        let q = normalize_for_index(truncated);
        if q.is_empty() {
            return Vec::new();
        }
        let mut q_terms: Vec<String> = tokenize_query_terms(&q);
        if q_terms.len() > MAX_QUERY_TOKENS {
            q_terms.truncate(MAX_QUERY_TOKENS);
        }
        if q_terms.is_empty() {
            q_terms.push(q.clone());
        }

        let mut scores: HashMap<String, f32> = HashMap::new();
        let mut breakdowns: HashMap<String, ScoreBreakdown> = HashMap::new();
        let mut matched_entities: HashMap<String, Vec<String>> = HashMap::new();
        let mut matched_terms: HashMap<String, Vec<String>> = HashMap::new();
        let query_set: HashSet<String> = q_terms.iter().cloned().collect();

        match self.lexical_bm25(&q, top_k.saturating_mul(5).max(20)) {
            Ok(lexical_hits) => {
                for (doc_id, bm25_score) in lexical_hits {
                    *scores.entry(doc_id.clone()).or_insert(0.0) += bm25_score;
                    breakdowns.entry(doc_id).or_default().lexical_score += bm25_score;
                }
            }
            Err(err) => {
                eprintln!("warning: lexical BM25 query component failed: {}", err);
            }
        }

        // Full-query entity hit.
        if let Some(postings) = self.entity_to_docs.get(&q) {
            for post in postings {
                let delta = 1.5 * post.score;
                *scores.entry(post.doc_id.clone()).or_insert(0.0) += delta;
                breakdowns
                    .entry(post.doc_id.clone())
                    .or_default()
                    .entity_score += delta;
                matched_entities
                    .entry(post.doc_id.clone())
                    .or_default()
                    .push(q.clone());
            }
        }

        for term in &q_terms {
            if let Some(postings) = self.entity_to_docs.get(term) {
                for post in postings {
                    let delta = 1.2 * post.score;
                    *scores.entry(post.doc_id.clone()).or_insert(0.0) += delta;
                    breakdowns
                        .entry(post.doc_id.clone())
                        .or_default()
                        .entity_score += delta;
                    matched_entities
                        .entry(post.doc_id.clone())
                        .or_default()
                        .push(term.clone());
                }
            }
            if let Some(postings) = self.term_to_docs.get(term) {
                for post in postings {
                    let delta = 0.8 * post.score;
                    *scores.entry(post.doc_id.clone()).or_insert(0.0) += delta;
                    breakdowns
                        .entry(post.doc_id.clone())
                        .or_default()
                        .term_score += delta;
                    matched_terms
                        .entry(post.doc_id.clone())
                        .or_default()
                        .push(term.clone());
                }
            }
        }

        for (doc_id, doc) in &self.docs {
            if let Some(topic) = doc.probable_topic.as_ref() {
                let topic_tokens = tokenize_query_terms(topic);
                let overlap = topic_tokens
                    .iter()
                    .filter(|t| query_set.contains(*t))
                    .count();
                if overlap > 0 {
                    let delta = 0.35 * overlap as f32;
                    *scores.entry(doc_id.clone()).or_insert(0.0) += delta;
                    breakdowns.entry(doc_id.clone()).or_default().topic_score += delta;
                }
            }
            if let Some(dt) = doc.doc_type_guess.as_ref() {
                let dt_tokens = tokenize_query_terms(dt);
                let overlap = dt_tokens.iter().filter(|t| query_set.contains(*t)).count();
                if overlap > 0 {
                    let delta = 0.25 * overlap as f32;
                    *scores.entry(doc_id.clone()).or_insert(0.0) += delta;
                    breakdowns.entry(doc_id.clone()).or_default().doc_type_score += delta;
                }
            }
            if doc.timestamp.is_some() {
                let delta = 0.05;
                *scores.entry(doc_id.clone()).or_insert(0.0) += delta;
                breakdowns.entry(doc_id.clone()).or_default().recency_score += delta;
            }
        }

        let mut out: Vec<SearchResult> = scores
            .into_iter()
            .filter_map(|(doc_id, score)| {
                let doc = self.docs.get(&doc_id)?;
                Some(SearchResult {
                    doc_id: doc_id.clone(),
                    source: doc.source.clone(),
                    score,
                    score_breakdown: breakdowns.remove(&doc_id).unwrap_or_default(),
                    matched_entities: matched_entities.remove(&doc_id).unwrap_or_default(),
                    matched_terms: matched_terms.remove(&doc_id).unwrap_or_default(),
                    probable_topic: doc.probable_topic.clone(),
                    doc_type_guess: doc.doc_type_guess.clone(),
                })
            })
            .collect();
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out.truncate(top_k);
        out
    }

    pub fn redacted_for_export(&self) -> RedactedMemoryIndex {
        let docs = self
            .docs
            .iter()
            .map(|(id, d)| {
                (
                    id.clone(),
                    RedactedDocRecord {
                        doc_id: d.doc_id.clone(),
                        source: d.source.clone(),
                        timestamp: d.timestamp.clone(),
                        doc_length: d.doc_length,
                        probable_topic: d.probable_topic.clone(),
                        doc_type_guess: d.doc_type_guess.clone(),
                        provenance: d.provenance.clone(),
                    },
                )
            })
            .collect();
        RedactedMemoryIndex {
            docs,
            topic_to_docs: self.topic_to_docs.clone(),
            doc_type_to_docs: self.doc_type_to_docs.clone(),
        }
    }
}

fn tokenize_query_terms(input: &str) -> Vec<String> {
    let token_re = Regex::new(r"[A-Za-z][A-Za-z0-9_-]{2,}").expect("valid regex");
    token_re
        .find_iter(input)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

fn normalize_for_index(input: &str) -> String {
    let lowered = deunicode(input).to_lowercase();
    let tokens = tokenize_query_terms(&lowered);
    tokens.join(" ")
}
