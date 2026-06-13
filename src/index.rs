use crate::ids::stable_chunk_id;
use crate::query_expansion::{expand_query_terms, normalize_for_index};
use crate::query_semantics::QueryRoutingIntent;
use crate::temporal::{parse_temporal_date, resolve_temporal_target};
use crate::tier1::{RankedTerm, Tier1Entity};
use anyhow::Result;
use chrono::{DateTime, NaiveDate, Utc};
use deunicode::deunicode;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::OnceLock;
use std::time::{Instant, SystemTime};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::document::TantivyDocument;
use tantivy::schema::Value;
use tantivy::schema::{Field, Schema, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexReader};

const LEXICAL_CONTENT_BOOST: f32 = 1.0;
const LEXICAL_HEADINGS_BOOST: f32 = 1.4;
const LEXICAL_TERMS_BOOST: f32 = 2.0;
const LEXICAL_ENTITIES_BOOST: f32 = 2.4;

const FULL_QUERY_ENTITY_WEIGHT: f32 = 1.5;
const ENTITY_TERM_WEIGHT: f32 = 1.2;
const ENTITY_PREFIX_MULTIPLIER: f32 = 0.25;
const IMPORTANT_TERM_WEIGHT: f32 = 0.8;
const IMPORTANT_TERM_PREFIX_MULTIPLIER: f32 = 0.25;
const EXPANDED_ENTITY_WEIGHT: f32 = 0.45;
const EXPANDED_TERM_WEIGHT: f32 = 0.35;
const TOPIC_OVERLAP_WEIGHT: f32 = 0.35;
const DOC_TYPE_OVERLAP_WEIGHT: f32 = 0.25;
const TIMESTAMP_PRESENCE_WEIGHT: f32 = 0.05;
const DOC_LINK_GRAPH_WEIGHT: f32 = 0.22;
const GRAPH_MAX_BOOST: f32 = 0.6;
const ENTITY_GRAPH_WEIGHT: f32 = 0.08;
const ENTITY_GRAPH_MAX_CANDIDATES: usize = 100;
const FINAL_RERANK_WINDOW: usize = 200;
const MAX_DOC_POSTINGS: usize = 500;
const MAX_RESULTS_PER_GROUP: usize = 2;
const TEXT_RERANK_WEIGHT: f32 = 0.08;
const TEXT_RERANK_NGRAM_WEIGHT: f32 = 0.08;
const TEXT_RERANK_LCS_WEIGHT: f32 = 0.05;
const TEXT_RERANK_NGRAM_SIZE: usize = 2;
const TEXT_RERANK_WINDOW: usize = 30;
const TEXT_RERANK_CONTENT_CHARS: usize = 4096;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub source: String,
    pub timestamp: Option<String>,
    pub ner_provider: String,
    pub term_ranker: String,
    pub index_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionChunk {
    pub chunk_id: String,
    pub heading: String,
    pub content: String,
    #[serde(default)]
    pub start_line: usize,
    #[serde(default)]
    pub end_line: usize,
    #[serde(default)]
    pub timestamp: Option<String>,
    pub key_entities: Vec<String>,
    pub important_terms: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocRecord {
    pub doc_id: String,
    pub source: String,
    #[serde(skip_serializing)]
    pub content: String,
    pub timestamp: Option<String>,
    pub doc_length: usize,
    pub author_agent: Option<String>,
    pub group_id: Option<String>,
    #[serde(default)]
    pub filters: std::collections::BTreeMap<String, String>,
    pub probable_topic: Option<String>,
    pub doc_type_guess: Option<String>,
    pub headings: Vec<String>,
    #[serde(default)]
    pub doc_links: Vec<String>,
    #[serde(default)]
    pub temporal_terms: Vec<String>,
    pub key_entities: Vec<Tier1Entity>,
    pub important_terms: Vec<RankedTerm>,
    pub section_chunks: Vec<SectionChunk>,
    pub embedding: Option<Vec<f32>>,
    pub top_claims: Vec<Claim>,
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityPosting {
    pub doc_id: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermPosting {
    pub doc_id: String,
    pub score: f32,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub(crate) struct SemanticChunkState {
    pub chunk_id: String,
    pub doc_id: String,
    pub heading: String,
    pub start_line: usize,
    pub end_line: usize,
    pub key_entities: Vec<String>,
    pub important_terms: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SemanticDocState {
    pub doc_id: String,
    pub chunk_ids: Vec<String>,
    pub chunks: HashMap<String, SemanticChunkState>,
    pub entity_to_docs: HashMap<String, Vec<EntityPosting>>,
    pub term_to_docs: HashMap<String, Vec<TermPosting>>,
    pub claim_to_docs: HashMap<String, Vec<TermPosting>>,
    pub topic: Option<String>,
    pub doc_type: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SemanticAggregate {
    pub chunk_to_doc: HashMap<String, String>,
    pub doc_to_chunks: HashMap<String, Vec<String>>,
    pub chunk_ranges: HashMap<String, (usize, usize)>,
    pub term_to_chunks: HashMap<String, Vec<(String, f32)>>,
    pub entity_to_chunks: HashMap<String, Vec<(String, f32)>>,
    pub entity_to_docs: HashMap<String, Vec<EntityPosting>>,
    pub term_to_docs: HashMap<String, Vec<TermPosting>>,
    pub claim_to_docs: HashMap<String, Vec<TermPosting>>,
    pub topic_to_docs: HashMap<String, Vec<String>>,
    pub doc_type_to_docs: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChunkMeta {
    doc_u32: u32,
    chunk_id: String,
    start_line: usize,
    end_line: usize,
}

#[derive(Debug, Default, Clone)]
struct TrieNode {
    children: HashMap<char, usize>,
    value: Option<u32>,
}

#[derive(Debug, Default, Clone)]
struct LexiconTrie {
    nodes: Vec<TrieNode>,
}

impl LexiconTrie {
    fn new() -> Self {
        Self {
            nodes: vec![TrieNode::default()],
        }
    }

    fn insert(&mut self, key: &str, value: u32) {
        let mut cur = 0usize;
        for ch in key.chars() {
            let next = if let Some(idx) = self.nodes[cur].children.get(&ch) {
                *idx
            } else {
                let idx = self.nodes.len();
                self.nodes.push(TrieNode::default());
                self.nodes[cur].children.insert(ch, idx);
                idx
            };
            cur = next;
        }
        self.nodes[cur].value = Some(value);
    }

    fn get(&self, key: &str) -> Option<u32> {
        let mut cur = 0usize;
        for ch in key.chars() {
            cur = *self.nodes.get(cur)?.children.get(&ch)?;
        }
        self.nodes.get(cur)?.value
    }

    fn prefix_ids(&self, prefix: &str, limit: usize) -> Vec<u32> {
        let mut cur = 0usize;
        for ch in prefix.chars() {
            let Some(next) = self
                .nodes
                .get(cur)
                .and_then(|n| n.children.get(&ch))
                .copied()
            else {
                return Vec::new();
            };
            cur = next;
        }
        let mut out = Vec::new();
        let mut stack = vec![cur];
        while let Some(idx) = stack.pop() {
            if let Some(v) = self.nodes[idx].value {
                out.push(v);
                if out.len() >= limit {
                    break;
                }
            }
            for next in self.nodes[idx].children.values() {
                stack.push(*next);
            }
        }
        out
    }
}

#[derive(Debug, Clone)]
struct IntervalEntry {
    start: usize,
    end: usize,
    chunk_u32: u32,
}

#[derive(Debug, Clone)]
struct IntervalNode {
    center: usize,
    overlaps: Vec<IntervalEntry>,
    left: Option<Box<IntervalNode>>,
    right: Option<Box<IntervalNode>>,
}

#[derive(Debug, Clone, Default)]
struct IntervalTree {
    root: Option<Box<IntervalNode>>,
}

impl IntervalTree {
    fn build(entries: Vec<IntervalEntry>) -> Self {
        fn build_rec(mut entries: Vec<IntervalEntry>) -> Option<Box<IntervalNode>> {
            if entries.is_empty() {
                return None;
            }
            let mut points = entries
                .iter()
                .map(|e| e.start + (e.end.saturating_sub(e.start) / 2))
                .collect::<Vec<_>>();
            points.sort_unstable();
            let center = points[points.len() / 2];
            let mut left = Vec::new();
            let mut right = Vec::new();
            let mut overlaps = Vec::new();
            for e in entries.drain(..) {
                if e.end < center {
                    left.push(e);
                } else if e.start > center {
                    right.push(e);
                } else {
                    overlaps.push(e);
                }
            }
            Some(Box::new(IntervalNode {
                center,
                overlaps,
                left: build_rec(left),
                right: build_rec(right),
            }))
        }
        Self {
            root: build_rec(entries),
        }
    }

    fn query(&self, start: usize, end: usize) -> Vec<u32> {
        fn walk(node: &Option<Box<IntervalNode>>, start: usize, end: usize, out: &mut Vec<u32>) {
            let Some(node) = node else {
                return;
            };
            for e in &node.overlaps {
                if e.start <= end && start <= e.end {
                    out.push(e.chunk_u32);
                }
            }
            if start <= node.center {
                walk(&node.left, start, end, out);
            }
            if end >= node.center {
                walk(&node.right, start, end, out);
            }
        }
        let mut out = Vec::new();
        walk(&self.root, start, end, &mut out);
        out
    }
}

#[derive(Serialize)]
pub struct MemoryIndex {
    pub docs: HashMap<String, DocRecord>,
    pub entity_to_docs: HashMap<String, Vec<EntityPosting>>,
    pub term_to_docs: HashMap<String, Vec<TermPosting>>,
    pub claim_to_docs: HashMap<String, Vec<TermPosting>>,
    pub topic_to_docs: HashMap<String, Vec<String>>,
    pub doc_type_to_docs: HashMap<String, Vec<String>>,
    #[serde(skip_serializing)]
    lexical: Option<LexicalIndex>,
    #[serde(skip_serializing)]
    #[allow(dead_code)]
    doc_id_to_u32: HashMap<String, u32>,
    #[serde(skip_serializing)]
    doc_u32_to_id: Vec<String>,
    #[serde(skip_serializing)]
    #[allow(dead_code)]
    chunk_id_to_u32: HashMap<String, u32>,
    #[serde(skip_serializing)]
    chunks: Vec<ChunkMeta>,
    #[serde(skip_serializing)]
    #[allow(dead_code)]
    doc_to_chunks: Vec<Vec<u32>>,
    #[serde(skip_serializing)]
    #[allow(dead_code)]
    term_lexicon: HashMap<String, u32>,
    #[serde(skip_serializing)]
    #[allow(dead_code)]
    entity_lexicon: HashMap<String, u32>,
    #[serde(skip_serializing)]
    term_postings_chunk: Vec<Vec<(u32, f32)>>,
    #[serde(skip_serializing)]
    entity_postings_chunk: Vec<Vec<(u32, f32)>>,
    #[serde(skip_serializing)]
    entity_postings_doc: Vec<Vec<(u32, f32)>>,
    #[serde(skip_serializing)]
    term_postings_doc: Vec<Vec<(u32, f32)>>,
    #[serde(skip_serializing)]
    #[allow(dead_code)]
    chunk_terms: Vec<Vec<u32>>,
    #[serde(skip_serializing)]
    #[allow(dead_code)]
    chunk_entities: Vec<Vec<u32>>,
    #[serde(skip_serializing)]
    term_trie: LexiconTrie,
    #[serde(skip_serializing)]
    entity_trie: LexiconTrie,
    #[serde(skip_serializing)]
    doc_interval_trees: Vec<IntervalTree>,
    #[serde(skip_serializing)]
    doc_key_entities: Vec<Vec<String>>,
    #[serde(skip_serializing)]
    doc_rerank_texts: Vec<String>,
    #[serde(skip_serializing)]
    doc_rerank_tokens: Vec<Vec<String>>,
    #[serde(skip_serializing)]
    doc_has_number: Vec<bool>,
    #[serde(skip_serializing)]
    claim_scoring: bool,
    #[serde(skip_serializing)]
    text_rerank_ngram: bool,
    #[serde(skip_serializing)]
    text_rerank_lcs: bool,
}

struct LexicalIndex {
    index: Index,
    reader: IndexReader,
    doc_id_f: Field,
    content_f: Field,
    headings_f: Field,
    terms_f: Field,
    entities_f: Field,
    temporal_f: Field,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ScoreBreakdown {
    pub lexical_score: f32,
    pub entity_score: f32,
    pub term_score: f32,
    pub claim_score: f32,
    pub semantic_score: f32,
    pub topic_score: f32,
    pub doc_type_score: f32,
    pub recency_score: f32,
    pub graph_link_score: f32,
    pub entity_graph_score: f32,
    pub sequence_rerank_score: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub doc_id: String,
    pub source: String,
    pub group_id: Option<String>,
    pub score: f32,
    pub score_breakdown: ScoreBreakdown,
    pub matched_entities: Vec<String>,
    pub matched_terms: Vec<String>,
    pub probable_topic: Option<String>,
    pub doc_type_guess: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct QueryTimings {
    pub total_ms: f64,
    pub refresh_ms: f64,
    pub lexical_bm25_ms: f64,
    pub snapshot_query_ms: f64,
    pub rerank_ms: f64,
    pub parse_ms: f64,
    pub sparse_scoring_ms: f64,
    pub lexical_merge_ms: f64,
    pub posting_scoring_ms: f64,
    pub routing_seed_ms: f64,
    pub candidate_accumulation_ms: f64,
    pub candidate_rank_ms: f64,
    pub metadata_ms: f64,
    pub graph_ms: f64,
    pub entity_graph_ms: f64,
    pub sequence_rerank_ms: f64,
    pub evidence_ms: f64,
    pub group_build_ms: f64,
    pub group_sort_ms: f64,
    pub ranking_ms: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct QueryDiagnostics {
    pub query_terms: usize,
    pub expanded_terms: usize,
    pub lexical_hits: usize,
    pub candidates: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct TemporalQueryContext<'a> {
    pub starts_from: Option<&'a str>,
    pub ends_at: Option<&'a str>,
    pub window_days: i64,
    pub hard_filter: bool,
    pub time_hint: Option<TemporalQueryHint>,
    pub query_routing_intent: Option<QueryRoutingIntent>,
    pub has_explicit_temporal: bool,
    pub allowed_doc_ids: Option<&'a HashSet<String>>,
}

impl<'a> Default for TemporalQueryContext<'a> {
    fn default() -> Self {
        Self {
            starts_from: None,
            ends_at: None,
            window_days: 7,
            hard_filter: false,
            time_hint: None,
            query_routing_intent: None,
            has_explicit_temporal: false,
            allowed_doc_ids: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct CandidateState {
    score: f32,
    breakdown: ScoreBreakdown,
    matched_entities: Vec<String>,
    matched_terms: Vec<String>,
}

#[derive(Debug, Clone, Default)]
struct EvidenceFeatures {
    score: f32,
    matched_terms: usize,
    has_number: bool,
    has_unit: bool,
    has_date: bool,
    has_temporal_terms: bool,
    predicate_signal: bool,
    distractor_penalty: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct RedactedDocRecord {
    pub doc_id: String,
    pub source: String,
    pub timestamp: Option<String>,
    pub doc_length: usize,
    pub group_id: Option<String>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalQueryHint {
    Past,
    Present,
    Ongoing,
    Mixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedMemoryCore {
    entity_to_docs: HashMap<String, Vec<EntityPosting>>,
    term_to_docs: HashMap<String, Vec<TermPosting>>,
    #[serde(default)]
    claim_to_docs: HashMap<String, Vec<TermPosting>>,
    topic_to_docs: HashMap<String, Vec<String>>,
    doc_type_to_docs: HashMap<String, Vec<String>>,
    doc_id_to_u32: HashMap<String, u32>,
    doc_u32_to_id: Vec<String>,
    chunk_id_to_u32: HashMap<String, u32>,
    chunks: Vec<ChunkMeta>,
    doc_to_chunks: Vec<Vec<u32>>,
    term_lexicon: HashMap<String, u32>,
    entity_lexicon: HashMap<String, u32>,
    term_postings_chunk: Vec<Vec<(u32, f32)>>,
    entity_postings_chunk: Vec<Vec<(u32, f32)>>,
    chunk_terms: Vec<Vec<u32>>,
    chunk_entities: Vec<Vec<u32>>,
}

impl MemoryIndex {
    pub fn from_records(records: Vec<DocRecord>) -> Self {
        Self::from_records_with_lexical_dir(records, None, false, false, false)
    }

    pub fn from_records_with_lexical_dir(
        records: Vec<DocRecord>,
        lexical_dir: Option<&Path>,
        text_rerank_ngram: bool,
        text_rerank_lcs: bool,
        claim_scoring: bool,
    ) -> Self {
        Self::from_records_internal(
            records,
            lexical_dir,
            None,
            text_rerank_ngram,
            text_rerank_lcs,
            claim_scoring,
        )
    }

    pub(crate) fn from_records_with_semantic_aggregate(
        records: Vec<DocRecord>,
        semantic_aggregate: SemanticAggregate,
        text_rerank_ngram: bool,
        text_rerank_lcs: bool,
        claim_scoring: bool,
    ) -> Self {
        Self::from_records_internal(
            records,
            None,
            Some(semantic_aggregate),
            text_rerank_ngram,
            text_rerank_lcs,
            claim_scoring,
        )
    }

    fn from_records_internal(
        records: Vec<DocRecord>,
        lexical_dir: Option<&Path>,
        semantic_aggregate: Option<SemanticAggregate>,
        text_rerank_ngram: bool,
        text_rerank_lcs: bool,
        claim_scoring: bool,
    ) -> Self {
        let mut docs = HashMap::new();
        let mut entity_to_docs: HashMap<String, Vec<EntityPosting>> = HashMap::new();
        let mut term_to_docs: HashMap<String, Vec<TermPosting>> = HashMap::new();
        let mut claim_to_docs: HashMap<String, Vec<TermPosting>> = HashMap::new();
        let mut topic_to_docs: HashMap<String, Vec<String>> = HashMap::new();
        let mut doc_type_to_docs: HashMap<String, Vec<String>> = HashMap::new();
        let mut doc_id_to_u32: HashMap<String, u32> = HashMap::new();
        let mut doc_u32_to_id: Vec<String> = Vec::new();
        let mut chunk_id_to_u32: HashMap<String, u32> = HashMap::new();
        let mut chunks: Vec<ChunkMeta> = Vec::new();
        let mut doc_to_chunks: Vec<Vec<u32>> = Vec::new();
        let mut term_lexicon: HashMap<String, u32> = HashMap::new();
        let mut entity_lexicon: HashMap<String, u32> = HashMap::new();
        let mut term_postings_chunk: Vec<Vec<(u32, f32)>> = Vec::new();
        let mut entity_postings_chunk: Vec<Vec<(u32, f32)>> = Vec::new();
        let mut chunk_terms: Vec<Vec<u32>> = Vec::new();
        let mut chunk_entities: Vec<Vec<u32>> = Vec::new();
        let mut doc_key_entities: Vec<Vec<String>> = Vec::new();
        let mut doc_rerank_texts: Vec<String> = Vec::new();
        let mut doc_rerank_tokens: Vec<Vec<String>> = Vec::new();
        let mut doc_has_number: Vec<bool> = Vec::new();

        for mut record in records {
            let doc_id = record.doc_id.clone();
            let doc_u32 = doc_u32_to_id.len() as u32;
            doc_id_to_u32.insert(doc_id.clone(), doc_u32);
            doc_u32_to_id.push(doc_id.clone());
            doc_to_chunks.push(Vec::new());
            doc_key_entities.push(normalized_entity_keys(&record.key_entities));
            let (rerank_text, rerank_tokens) = build_doc_rerank_cache(&record);
            doc_has_number.push(contains_number_like(&rerank_text));
            doc_rerank_texts.push(rerank_text);
            doc_rerank_tokens.push(rerank_tokens);
            if record.section_chunks.is_empty() {
                record.section_chunks.push(SectionChunk {
                    chunk_id: stable_chunk_id(
                        &doc_id,
                        record
                            .headings
                            .first()
                            .map(String::as_str)
                            .unwrap_or("(document)"),
                        &record.content,
                        1,
                        record.content.lines().count().max(1),
                    ),
                    heading: record
                        .headings
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "(document)".to_string()),
                    content: record.content.clone(),
                    start_line: 1,
                    end_line: record.content.lines().count().max(1),
                    timestamp: record.timestamp.clone(),
                    key_entities: record
                        .key_entities
                        .iter()
                        .map(|e| normalize_for_index(&e.text))
                        .filter(|v| !v.is_empty())
                        .collect(),
                    important_terms: record
                        .important_terms
                        .iter()
                        .map(|t| normalize_for_index(&t.term))
                        .filter(|v| !v.is_empty())
                        .collect(),
                });
            }
            if semantic_aggregate.is_none() {
                for chunk in &record.section_chunks {
                    let chunk_u32 = chunks.len() as u32;
                    chunk_id_to_u32.insert(chunk.chunk_id.clone(), chunk_u32);
                    chunks.push(ChunkMeta {
                        doc_u32,
                        chunk_id: chunk.chunk_id.clone(),
                        start_line: chunk.start_line,
                        end_line: chunk.end_line,
                    });
                    doc_to_chunks[doc_u32 as usize].push(chunk_u32);
                    chunk_terms.push(Vec::new());
                    chunk_entities.push(Vec::new());

                    for term in &chunk.important_terms {
                        let key = normalize_for_index(term);
                        if key.is_empty() {
                            continue;
                        }
                        let term_u32 = *term_lexicon.entry(key).or_insert_with(|| {
                            term_postings_chunk.push(Vec::new());
                            (term_postings_chunk.len() - 1) as u32
                        });
                        term_postings_chunk[term_u32 as usize].push((chunk_u32, 0.8));
                        chunk_terms[chunk_u32 as usize].push(term_u32);
                    }
                    let heading_tokens = tokenize_query_terms(&chunk.heading);
                    for token in heading_tokens {
                        let term_u32 = *term_lexicon.entry(token).or_insert_with(|| {
                            term_postings_chunk.push(Vec::new());
                            (term_postings_chunk.len() - 1) as u32
                        });
                        term_postings_chunk[term_u32 as usize].push((chunk_u32, 0.4));
                        chunk_terms[chunk_u32 as usize].push(term_u32);
                    }
                    for entity in &chunk.key_entities {
                        let key = normalize_for_index(entity);
                        if key.is_empty() {
                            continue;
                        }
                        let entity_u32 = *entity_lexicon.entry(key).or_insert_with(|| {
                            entity_postings_chunk.push(Vec::new());
                            (entity_postings_chunk.len() - 1) as u32
                        });
                        entity_postings_chunk[entity_u32 as usize].push((chunk_u32, 0.9));
                        chunk_entities[chunk_u32 as usize].push(entity_u32);
                    }
                }
            }
            if semantic_aggregate.is_none() {
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
            }
            if claim_scoring {
                for claim in &record.top_claims {
                    for token in claim_tokens(claim) {
                        claim_to_docs.entry(token).or_default().push(TermPosting {
                            doc_id: doc_id.clone(),
                            score: claim.confidence.max(0.1),
                        });
                    }
                }
            }
            docs.insert(doc_id, record);
        }

        let mut term_u32_to_key: Vec<String> = vec![String::new(); term_lexicon.len()];
        for (k, &v) in &term_lexicon {
            term_u32_to_key[v as usize] = k.clone();
        }
        let mut entity_u32_to_key: Vec<String> = vec![String::new(); entity_lexicon.len()];
        for (k, &v) in &entity_lexicon {
            entity_u32_to_key[v as usize] = k.clone();
        }

        if let Some(semantic_aggregate) = semantic_aggregate {
            for (doc_u32, doc_id) in doc_u32_to_id.iter().enumerate() {
                let mut doc_chunk_ids = semantic_aggregate
                    .doc_to_chunks
                    .get(doc_id)
                    .cloned()
                    .unwrap_or_default();
                doc_chunk_ids.sort();
                for chunk_id in doc_chunk_ids {
                    let Some((start_line, end_line)) =
                        semantic_aggregate.chunk_ranges.get(&chunk_id).copied()
                    else {
                        continue;
                    };
                    let chunk_u32 = chunks.len() as u32;
                    chunk_id_to_u32.insert(chunk_id.clone(), chunk_u32);
                    chunks.push(ChunkMeta {
                        doc_u32: doc_u32 as u32,
                        chunk_id: chunk_id.clone(),
                        start_line,
                        end_line,
                    });
                    doc_to_chunks[doc_u32].push(chunk_u32);
                    chunk_terms.push(Vec::new());
                    chunk_entities.push(Vec::new());
                }
            }
            for (key, postings) in &semantic_aggregate.term_to_chunks {
                let term_u32 = *term_lexicon.entry(key.clone()).or_insert_with(|| {
                    term_postings_chunk.push(Vec::new());
                    (term_postings_chunk.len() - 1) as u32
                });
                for (chunk_id, score) in postings {
                    let Some(&chunk_u32) = chunk_id_to_u32.get(chunk_id) else {
                        continue;
                    };
                    term_postings_chunk[term_u32 as usize].push((chunk_u32, *score));
                    chunk_terms[chunk_u32 as usize].push(term_u32);
                }
            }
            for (key, postings) in &semantic_aggregate.entity_to_chunks {
                let entity_u32 = *entity_lexicon.entry(key.clone()).or_insert_with(|| {
                    entity_postings_chunk.push(Vec::new());
                    (entity_postings_chunk.len() - 1) as u32
                });
                for (chunk_id, score) in postings {
                    let Some(&chunk_u32) = chunk_id_to_u32.get(chunk_id) else {
                        continue;
                    };
                    entity_postings_chunk[entity_u32 as usize].push((chunk_u32, *score));
                    chunk_entities[chunk_u32 as usize].push(entity_u32);
                }
            }
            entity_to_docs = semantic_aggregate.entity_to_docs;
            term_to_docs = semantic_aggregate.term_to_docs;
            if claim_scoring {
                claim_to_docs = semantic_aggregate.claim_to_docs;
            }
            topic_to_docs = semantic_aggregate.topic_to_docs;
            doc_type_to_docs = semantic_aggregate.doc_type_to_docs;
        } else {
            term_to_docs.clear();
            entity_to_docs.clear();
            if !claim_scoring {
                claim_to_docs.clear();
            }
            for (term_u32, postings) in term_postings_chunk.iter().enumerate() {
                let key = &term_u32_to_key[term_u32];
                for (chunk_u32, score) in postings {
                    let doc_u32 = chunks[*chunk_u32 as usize].doc_u32;
                    let doc_id = &doc_u32_to_id[doc_u32 as usize];
                    term_to_docs
                        .entry(key.clone())
                        .or_default()
                        .push(TermPosting {
                            doc_id: doc_id.clone(),
                            score: *score,
                        });
                }
            }
            for (entity_u32, postings) in entity_postings_chunk.iter().enumerate() {
                let key = &entity_u32_to_key[entity_u32];
                for (chunk_u32, score) in postings {
                    let doc_u32 = chunks[*chunk_u32 as usize].doc_u32;
                    let doc_id = &doc_u32_to_id[doc_u32 as usize];
                    entity_to_docs
                        .entry(key.clone())
                        .or_default()
                        .push(EntityPosting {
                            doc_id: doc_id.clone(),
                            score: *score,
                        });
                }
            }
        }

        if claim_scoring {
            for postings in claim_to_docs.values_mut() {
                postings.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        let (term_trie, entity_trie, doc_interval_trees) = Self::build_runtime_helpers(
            &term_lexicon,
            &entity_lexicon,
            &chunks,
            doc_u32_to_id.len(),
        );

        let entity_postings_doc =
            derive_doc_postings(&entity_postings_chunk, &chunks, MAX_DOC_POSTINGS);
        let term_postings_doc =
            derive_doc_postings(&term_postings_chunk, &chunks, MAX_DOC_POSTINGS);

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

        let lexical = match Self::build_lexical_index(&docs, lexical_dir) {
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
            claim_to_docs,
            topic_to_docs,
            doc_type_to_docs,
            lexical,
            doc_id_to_u32,
            doc_u32_to_id,
            chunk_id_to_u32,
            chunks,
            doc_to_chunks,
            term_lexicon,
            entity_lexicon,
            term_postings_chunk,
            entity_postings_chunk,
            entity_postings_doc,
            term_postings_doc,
            chunk_terms,
            chunk_entities,
            term_trie,
            entity_trie,
            doc_interval_trees,
            doc_key_entities,
            doc_rerank_texts,
            doc_rerank_tokens,
            doc_has_number,
            claim_scoring,
            text_rerank_ngram,
            text_rerank_lcs,
        }
    }

    fn build_runtime_helpers(
        term_lexicon: &HashMap<String, u32>,
        entity_lexicon: &HashMap<String, u32>,
        chunks: &[ChunkMeta],
        doc_count: usize,
    ) -> (LexiconTrie, LexiconTrie, Vec<IntervalTree>) {
        let mut term_trie = LexiconTrie::new();
        for (key, &id) in term_lexicon {
            term_trie.insert(key, id);
        }
        let mut entity_trie = LexiconTrie::new();
        for (key, &id) in entity_lexicon {
            entity_trie.insert(key, id);
        }
        let mut interval_entries_by_doc: Vec<Vec<IntervalEntry>> = vec![Vec::new(); doc_count];
        for (chunk_u32, meta) in chunks.iter().enumerate() {
            if meta.end_line >= meta.start_line
                && meta.end_line > 0
                && (meta.doc_u32 as usize) < doc_count
            {
                interval_entries_by_doc[meta.doc_u32 as usize].push(IntervalEntry {
                    start: meta.start_line.max(1),
                    end: meta.end_line,
                    chunk_u32: chunk_u32 as u32,
                });
            }
        }
        let trees = interval_entries_by_doc
            .into_iter()
            .map(IntervalTree::build)
            .collect::<Vec<_>>();
        (term_trie, entity_trie, trees)
    }

    pub fn load_with_binary_core(
        records: Vec<DocRecord>,
        core_path: &Path,
        lexical_dir: Option<&Path>,
        claim_scoring: bool,
    ) -> Result<Self> {
        let bytes = fs::read(core_path)?;
        let core: PersistedMemoryCore = bincode::deserialize(&bytes)?;
        Self::load_from_core_with_options(core, records, lexical_dir, claim_scoring)
    }

    fn load_from_core(
        core: PersistedMemoryCore,
        records: Vec<DocRecord>,
        lexical_dir: Option<&Path>,
    ) -> Result<Self> {
        Self::load_from_core_with_options(core, records, lexical_dir, false)
    }

    fn load_from_core_with_options(
        core: PersistedMemoryCore,
        records: Vec<DocRecord>,
        lexical_dir: Option<&Path>,
        claim_scoring: bool,
    ) -> Result<Self> {
        if core.doc_u32_to_id.is_empty() && !records.is_empty() {
            anyhow::bail!("invalid binary core: empty doc table");
        }
        let mut docs = HashMap::new();
        for mut record in records {
            if record.section_chunks.is_empty() {
                record.section_chunks.push(SectionChunk {
                    chunk_id: stable_chunk_id(
                        &record.doc_id,
                        record
                            .headings
                            .first()
                            .map(String::as_str)
                            .unwrap_or("(document)"),
                        &record.content,
                        1,
                        record.content.lines().count().max(1),
                    ),
                    heading: record
                        .headings
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "(document)".to_string()),
                    content: record.content.clone(),
                    start_line: 1,
                    end_line: record.content.lines().count().max(1),
                    timestamp: record.timestamp.clone(),
                    key_entities: record
                        .key_entities
                        .iter()
                        .map(|e| normalize_for_index(&e.text))
                        .filter(|v| !v.is_empty())
                        .collect(),
                    important_terms: record
                        .important_terms
                        .iter()
                        .map(|t| normalize_for_index(&t.term))
                        .filter(|v| !v.is_empty())
                        .collect(),
                });
            }
            docs.insert(record.doc_id.clone(), record);
        }
        let lexical = match Self::build_lexical_index(&docs, lexical_dir) {
            Ok(index) => Some(index),
            Err(err) => {
                eprintln!("warning: lexical BM25 index disabled: {}", err);
                None
            }
        };
        let (term_trie, entity_trie, doc_interval_trees) = Self::build_runtime_helpers(
            &core.term_lexicon,
            &core.entity_lexicon,
            &core.chunks,
            core.doc_u32_to_id.len(),
        );
        let mut doc_key_entities = vec![Vec::new(); core.doc_u32_to_id.len()];
        let mut doc_rerank_texts = vec![String::new(); core.doc_u32_to_id.len()];
        let mut doc_rerank_tokens = vec![Vec::new(); core.doc_u32_to_id.len()];
        let entity_postings_doc =
            derive_doc_postings(&core.entity_postings_chunk, &core.chunks, MAX_DOC_POSTINGS);
        let term_postings_doc =
            derive_doc_postings(&core.term_postings_chunk, &core.chunks, MAX_DOC_POSTINGS);
        let mut doc_has_number = vec![false; core.doc_u32_to_id.len()];
        for (doc_id, record) in &docs {
            if let Some(&doc_u32) = core.doc_id_to_u32.get(doc_id) {
                doc_key_entities[doc_u32 as usize] = normalized_entity_keys(&record.key_entities);
                let (rerank_text, rerank_tokens) = build_doc_rerank_cache(record);
                doc_has_number[doc_u32 as usize] = contains_number_like(&rerank_text);
                doc_rerank_texts[doc_u32 as usize] = rerank_text;
                doc_rerank_tokens[doc_u32 as usize] = rerank_tokens;
            }
        }
        Ok(Self {
            docs,
            entity_to_docs: core.entity_to_docs,
            term_to_docs: core.term_to_docs,
            claim_to_docs: if claim_scoring {
                core.claim_to_docs
            } else {
                HashMap::new()
            },
            topic_to_docs: core.topic_to_docs,
            doc_type_to_docs: core.doc_type_to_docs,
            lexical,
            doc_id_to_u32: core.doc_id_to_u32,
            doc_u32_to_id: core.doc_u32_to_id,
            chunk_id_to_u32: core.chunk_id_to_u32,
            chunks: core.chunks,
            doc_to_chunks: core.doc_to_chunks,
            term_lexicon: core.term_lexicon,
            entity_lexicon: core.entity_lexicon,
            term_postings_chunk: core.term_postings_chunk,
            entity_postings_chunk: core.entity_postings_chunk,
            entity_postings_doc,
            term_postings_doc,
            chunk_terms: core.chunk_terms,
            chunk_entities: core.chunk_entities,
            term_trie,
            entity_trie,
            doc_interval_trees,
            doc_key_entities,
            doc_rerank_texts,
            doc_rerank_tokens,
            doc_has_number,
            claim_scoring,
            text_rerank_ngram: false,
            text_rerank_lcs: false,
        })
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let core = PersistedMemoryCore {
            entity_to_docs: self.entity_to_docs.clone(),
            term_to_docs: self.term_to_docs.clone(),
            claim_to_docs: self.claim_to_docs.clone(),
            topic_to_docs: self.topic_to_docs.clone(),
            doc_type_to_docs: self.doc_type_to_docs.clone(),
            doc_id_to_u32: self.doc_id_to_u32.clone(),
            doc_u32_to_id: self.doc_u32_to_id.clone(),
            chunk_id_to_u32: self.chunk_id_to_u32.clone(),
            chunks: self.chunks.clone(),
            doc_to_chunks: self.doc_to_chunks.clone(),
            term_lexicon: self.term_lexicon.clone(),
            entity_lexicon: self.entity_lexicon.clone(),
            term_postings_chunk: self.term_postings_chunk.clone(),
            entity_postings_chunk: self.entity_postings_chunk.clone(),
            chunk_terms: self.chunk_terms.clone(),
            chunk_entities: self.chunk_entities.clone(),
        };
        Ok(bincode::serialize(&core)?)
    }

    pub fn from_bytes(bytes: &[u8], records: Vec<DocRecord>) -> Result<Self> {
        let core: PersistedMemoryCore = bincode::deserialize(bytes)?;
        Self::load_from_core(core, records, None)
    }

    pub fn save_binary_core(&self, core_path: &Path) -> Result<()> {
        if let Some(parent) = core_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let core = PersistedMemoryCore {
            entity_to_docs: self.entity_to_docs.clone(),
            term_to_docs: self.term_to_docs.clone(),
            claim_to_docs: self.claim_to_docs.clone(),
            topic_to_docs: self.topic_to_docs.clone(),
            doc_type_to_docs: self.doc_type_to_docs.clone(),
            doc_id_to_u32: self.doc_id_to_u32.clone(),
            doc_u32_to_id: self.doc_u32_to_id.clone(),
            chunk_id_to_u32: self.chunk_id_to_u32.clone(),
            chunks: self.chunks.clone(),
            doc_to_chunks: self.doc_to_chunks.clone(),
            term_lexicon: self.term_lexicon.clone(),
            entity_lexicon: self.entity_lexicon.clone(),
            term_postings_chunk: self.term_postings_chunk.clone(),
            entity_postings_chunk: self.entity_postings_chunk.clone(),
            chunk_terms: self.chunk_terms.clone(),
            chunk_entities: self.chunk_entities.clone(),
        };
        fs::write(core_path, bincode::serialize(&core)?)?;
        Ok(())
    }

    fn build_lexical_index(
        docs: &HashMap<String, DocRecord>,
        lexical_dir: Option<&Path>,
    ) -> Result<LexicalIndex> {
        let mut schema_builder = Schema::builder();
        let doc_id_f = schema_builder.add_text_field("doc_id", STRING | STORED);
        let content_f = schema_builder.add_text_field("content", TEXT);
        let headings_f = schema_builder.add_text_field("headings", TEXT);
        let terms_f = schema_builder.add_text_field("important_terms", TEXT);
        let entities_f = schema_builder.add_text_field("entities", TEXT);
        let temporal_f = schema_builder.add_text_field("temporal_terms", TEXT);
        let schema = schema_builder.build();
        let index = if let Some(dir) = lexical_dir {
            fs::create_dir_all(dir)?;
            match Index::open_in_dir(dir) {
                Ok(existing) => existing,
                Err(_) => {
                    if dir.exists() {
                        let _ = fs::remove_dir_all(dir);
                        fs::create_dir_all(dir)?;
                    }
                    let created = Index::create_in_dir(dir, schema.clone())?;
                    let mut writer = created.writer(50_000_000)?;
                    for doc in docs.values() {
                        let headings_text = doc
                            .section_chunks
                            .iter()
                            .map(|c| c.heading.as_str())
                            .collect::<Vec<_>>()
                            .join(" ");
                        let terms_text = doc
                            .section_chunks
                            .iter()
                            .flat_map(|c| c.important_terms.iter().map(String::as_str))
                            .collect::<Vec<_>>()
                            .join(" ");
                        let entities_text = doc
                            .section_chunks
                            .iter()
                            .flat_map(|c| c.key_entities.iter().map(String::as_str))
                            .collect::<Vec<_>>()
                            .join(" ");
                        let content_text = doc
                            .section_chunks
                            .iter()
                            .map(|c| c.content.as_str())
                            .collect::<Vec<_>>()
                            .join("\n");
                        let temporal_text = doc.temporal_terms.join(" ");
                        writer.add_document(doc!(
                            doc_id_f => doc.doc_id.clone(),
                            content_f => content_text,
                            headings_f => headings_text,
                            terms_f => terms_text,
                            entities_f => entities_text,
                            temporal_f => temporal_text
                        ))?;
                    }
                    writer.commit()?;
                    created
                }
            }
        } else {
            let ram = Index::create_in_ram(schema);
            let mut writer = ram.writer(15_000_001)?;
            for doc in docs.values() {
                let headings_text = doc
                    .section_chunks
                    .iter()
                    .map(|c| c.heading.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                let terms_text = doc
                    .section_chunks
                    .iter()
                    .flat_map(|c| c.important_terms.iter().map(String::as_str))
                    .collect::<Vec<_>>()
                    .join(" ");
                let entities_text = doc
                    .section_chunks
                    .iter()
                    .flat_map(|c| c.key_entities.iter().map(String::as_str))
                    .collect::<Vec<_>>()
                    .join(" ");
                let content_text = doc
                    .section_chunks
                    .iter()
                    .map(|c| c.content.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                let temporal_text = doc.temporal_terms.join(" ");
                writer.add_document(doc!(
                    doc_id_f => doc.doc_id.clone(),
                    content_f => content_text,
                    headings_f => headings_text,
                    terms_f => terms_text,
                    entities_f => entities_text,
                    temporal_f => temporal_text
                ))?;
            }
            writer.commit()?;
            ram
        };
        let reader = index.reader()?;
        let schema_ref = index.schema();
        let doc_id_f = schema_ref.get_field("doc_id")?;
        let content_f = schema_ref.get_field("content")?;
        let headings_f = schema_ref.get_field("headings")?;
        let terms_f = schema_ref.get_field("important_terms")?;
        let entities_f = schema_ref.get_field("entities")?;
        let temporal_f = schema_ref.get_field("temporal_terms")?;
        Ok(LexicalIndex {
            index,
            reader,
            doc_id_f,
            content_f,
            headings_f,
            terms_f,
            entities_f,
            temporal_f,
        })
    }

    fn lexical_bm25(&self, query: &str, top_k: usize) -> Result<HashMap<String, f32>> {
        let Some(lex) = self.lexical.as_ref() else {
            return Ok(HashMap::new());
        };
        let searcher = lex.reader.searcher();
        let mut query_parser = QueryParser::for_index(
            &lex.index,
            vec![
                lex.content_f,
                lex.headings_f,
                lex.terms_f,
                lex.entities_f,
                lex.temporal_f,
            ],
        );
        query_parser.set_field_boost(lex.content_f, LEXICAL_CONTENT_BOOST);
        query_parser.set_field_boost(lex.headings_f, LEXICAL_HEADINGS_BOOST);
        query_parser.set_field_boost(lex.terms_f, LEXICAL_TERMS_BOOST);
        query_parser.set_field_boost(lex.entities_f, LEXICAL_ENTITIES_BOOST);
        query_parser.set_field_boost(lex.temporal_f, 1.1);
        let parsed = match query_parser.parse_query(query) {
            Ok(parsed) => parsed,
            Err(first_err) => {
                let fallback_query = sanitize_bm25_query(query);
                if fallback_query.is_empty() {
                    return Err(first_err.into());
                }
                match query_parser.parse_query(&fallback_query) {
                    Ok(parsed) => parsed,
                    Err(_) => return Err(first_err.into()),
                }
            }
        };
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
        self.query_with_temporal_context(query, top_k, TemporalQueryContext::default())
            .0
    }

    pub fn query_with_temporal_context(
        &self,
        query: &str,
        top_k: usize,
        temporal: TemporalQueryContext<'_>,
    ) -> (Vec<SearchResult>, QueryTimings, QueryDiagnostics) {
        let search_k = match temporal.query_routing_intent {
            Some(_) => top_k.saturating_mul(10).max(25),
            None => top_k.saturating_mul(5).max(20),
        };
        let total_start = Instant::now();
        let (mut results, mut timings, diagnostics) =
            self.query_timed_with_context(query, search_k, temporal);
        let (temporal_start, temporal_end) =
            normalize_temporal_bounds(temporal.starts_from, temporal.ends_at);
        let relative_anchor = temporal_start
            .map(|_| temporal.starts_from)
            .flatten()
            .or(temporal_end.map(|_| temporal.ends_at).flatten())
            .or(temporal.starts_from)
            .or(temporal.ends_at);
        let target = resolve_temporal_target(query, relative_anchor);
        if target.is_none()
            && temporal_start.is_none()
            && temporal_end.is_none()
            && temporal.time_hint.is_none()
        {
            timings.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            results.truncate(top_k);
            return (results, timings, diagnostics);
        }

        let window_days = temporal.window_days.max(1);
        let hard_filter = temporal.hard_filter;
        let time_hint = temporal.time_hint;
        let now_date = DateTime::<Utc>::from(SystemTime::now()).date_naive();
        let mut rescored = Vec::with_capacity(results.len());
        for mut result in results.drain(..) {
            let Some(doc) = self.docs.get(&result.doc_id) else {
                continue;
            };
            let Some(doc_date) = doc_temporal_date(doc) else {
                if hard_filter {
                    continue;
                }
                rescored.push(result);
                continue;
            };
            if let Some(start) = temporal_start {
                if doc_date < start && hard_filter {
                    continue;
                }
            }
            if let Some(end) = temporal_end {
                if doc_date > end && hard_filter {
                    continue;
                }
            }
            let within_explicit_range = temporal_start.is_none_or(|start| doc_date >= start)
                && temporal_end.is_none_or(|end| doc_date <= end);
            if within_explicit_range {
                let explicit_boost = TEMPORAL_RANGE_BOOST;
                result.score += explicit_boost;
                result.score_breakdown.recency_score += explicit_boost;
            }
            if let Some(target) = target {
                let delta_days = (doc_date
                    .signed_duration_since(target.target_date)
                    .num_days())
                .abs();
                if hard_filter && delta_days > window_days {
                    continue;
                }
                if delta_days <= window_days {
                    let proximity = 1.0 - (delta_days as f32 / window_days as f32);
                    let boost = proximity.clamp(0.0, 1.0) * TEMPORAL_PROXIMITY_WEIGHT;
                    result.score += boost;
                    result.score_breakdown.recency_score += boost;
                }
            }
            if temporal.has_explicit_temporal
                && target.is_none()
                && temporal_start.is_none()
                && temporal_end.is_none()
            {
                if let Some(hint) = time_hint {
                    let delta_days = (doc_date.signed_duration_since(now_date).num_days()).abs();
                    let boost = match hint {
                        TemporalQueryHint::Past => {
                            if doc_date <= now_date {
                                0.03
                            } else {
                                0.0
                            }
                        }
                        TemporalQueryHint::Present => {
                            if delta_days <= 30 {
                                let proximity = 1.0 - (delta_days as f32 / 30.0);
                                proximity.clamp(0.0, 1.0) * 0.25
                            } else {
                                0.0
                            }
                        }
                        TemporalQueryHint::Ongoing => {
                            if delta_days <= 14 {
                                let proximity = 1.0 - (delta_days as f32 / 14.0);
                                proximity.clamp(0.0, 1.0) * 0.3
                            } else {
                                0.0
                            }
                        }
                        TemporalQueryHint::Mixed => {
                            if delta_days <= 30 {
                                let proximity = 1.0 - (delta_days as f32 / 30.0);
                                proximity.clamp(0.0, 1.0) * 0.15
                            } else {
                                0.0
                            }
                        }
                    };
                    if boost > 0.0 {
                        result.score += boost;
                        result.score_breakdown.recency_score += boost;
                    }
                }
            }
            rescored.push(result);
        }
        rescored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rescored.truncate(top_k);
        timings.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        (rescored, timings, diagnostics)
    }

    pub fn query_timed(
        &self,
        query: &str,
        top_k: usize,
    ) -> (Vec<SearchResult>, QueryTimings, QueryDiagnostics) {
        self.query_timed_with_context(query, top_k, TemporalQueryContext::default())
    }

    fn query_timed_with_context(
        &self,
        query: &str,
        top_k: usize,
        temporal: TemporalQueryContext<'_>,
    ) -> (Vec<SearchResult>, QueryTimings, QueryDiagnostics) {
        let total_start = Instant::now();
        let lexical_start = Instant::now();
        let lexical_hits = match self.lexical_bm25(query, top_k.saturating_mul(5).max(20)) {
            Ok(hits) => Some(hits),
            Err(err) => {
                eprintln!("warning: lexical BM25 query component failed: {}", err);
                None
            }
        };
        let lexical_bm25_ms = lexical_start.elapsed().as_secs_f64() * 1000.0;
        let snapshot_start = Instant::now();
        let (results, mut timings, diagnostics) = self.query_with_lexical_hits_timed(
            query,
            top_k,
            lexical_hits.as_ref(),
            temporal.query_routing_intent,
            temporal.has_explicit_temporal,
            temporal.allowed_doc_ids,
        );
        let snapshot_query_ms = snapshot_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        timings.snapshot_query_ms = snapshot_query_ms;
        timings.lexical_bm25_ms = lexical_bm25_ms;
        timings.total_ms = total_ms;
        (results, timings, diagnostics)
    }

    pub fn query_with_lexical_hits(
        &self,
        query: &str,
        top_k: usize,
        lexical_hits: Option<&HashMap<String, f32>>,
    ) -> Vec<SearchResult> {
        self.query_with_lexical_hits_timed(query, top_k, lexical_hits, None, false, None)
            .0
    }

    /// Query with exact-match field filtering. All supplied filter key-value pairs must match
    /// a document's `filters` map for it to be included in results.
    pub fn query_with_filters(
        &self,
        query: &str,
        top_k: usize,
        filters: &std::collections::BTreeMap<String, String>,
    ) -> Vec<SearchResult> {
        self.query_with_filters_and_lexical(query, top_k, filters, None)
    }

    pub fn query_with_filters_and_lexical(
        &self,
        query: &str,
        top_k: usize,
        filters: &std::collections::BTreeMap<String, String>,
        lexical_hits: Option<&HashMap<String, f32>>,
    ) -> Vec<SearchResult> {
        if filters.is_empty() {
            return self.query_with_lexical_hits(query, top_k, lexical_hits);
        }
        let allowed: HashSet<String> = self
            .docs
            .iter()
            .filter(|(_, doc)| {
                filters
                    .iter()
                    .all(|(k, v)| doc.filters.get(k).map(|dv| dv == v).unwrap_or(false))
            })
            .map(|(doc_id, _)| doc_id.clone())
            .collect();
        self.query_with_lexical_hits_timed(query, top_k, lexical_hits, None, false, Some(&allowed))
            .0
    }

    /// Multi-term variant: builds the allowed-doc set once from `filters`, then scores each
    /// query against it. At scale (thousands of docs) this avoids rescanning docs N times.
    /// Returns one `Vec<SearchResult>` per input query, in the same order.
    pub fn query_with_filters_multi(
        &self,
        queries: &[&str],
        top_k: usize,
        filters: &std::collections::BTreeMap<String, String>,
        lexical_hits_per_query: &[HashMap<String, f32>],
    ) -> Vec<Vec<SearchResult>> {
        let allowed_opt: Option<HashSet<String>> = if filters.is_empty() {
            None
        } else {
            Some(
                self.docs
                    .iter()
                    .filter(|(_, doc)| {
                        filters
                            .iter()
                            .all(|(k, v)| doc.filters.get(k).map(|dv| dv == v).unwrap_or(false))
                    })
                    .map(|(doc_id, _)| doc_id.clone())
                    .collect(),
            )
        };
        queries
            .iter()
            .zip(lexical_hits_per_query.iter())
            .map(|(q, hits)| {
                self.query_with_lexical_hits_timed(
                    q,
                    top_k,
                    Some(hits),
                    None,
                    false,
                    allowed_opt.as_ref(),
                )
                .0
            })
            .collect()
    }

    fn query_with_lexical_hits_timed(
        &self,
        query: &str,
        top_k: usize,
        lexical_hits: Option<&HashMap<String, f32>>,
        query_routing_intent: Option<QueryRoutingIntent>,
        has_explicit_temporal: bool,
        allowed_doc_ids: Option<&HashSet<String>>,
    ) -> (Vec<SearchResult>, QueryTimings, QueryDiagnostics) {
        const MAX_QUERY_CHARS: usize = 4096;
        const MAX_QUERY_TOKENS: usize = 128;

        let rerank_start = Instant::now();
        let parse_start = Instant::now();
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
            return (
                Vec::new(),
                QueryTimings::default(),
                QueryDiagnostics::default(),
            );
        }
        let mut q_terms: Vec<String> = tokenize_query_terms(&q);
        if q_terms.len() > MAX_QUERY_TOKENS {
            q_terms.truncate(MAX_QUERY_TOKENS);
        }
        if q_terms.is_empty() {
            q_terms.push(q.clone());
        }
        let expanded = expand_query_terms(&q_terms);
        let expanded_terms = expanded.expanded_terms;
        let mut query_entities: HashSet<String> = HashSet::new();
        if self.entity_trie.get(&q).is_some() {
            query_entities.insert(q.clone());
        }
        for term in &q_terms {
            if self.entity_trie.get(term).is_some() {
                query_entities.insert(term.clone());
            }
        }
        if self.doc_u32_to_id.is_empty() {
            return (
                Vec::new(),
                QueryTimings::default(),
                QueryDiagnostics::default(),
            );
        }
        let parse_ms = parse_start.elapsed().as_secs_f64() * 1000.0;
        let sparse_start = Instant::now();
        let mut candidates: HashMap<usize, CandidateState> = HashMap::new();
        let query_set: HashSet<String> = q_terms.iter().cloned().collect();
        let allowed_doc_u32s: Option<HashSet<usize>> = allowed_doc_ids.map(|allowed| {
            allowed
                .iter()
                .filter_map(|doc_id| self.doc_id_to_u32.get(doc_id).copied())
                .map(|doc_u32| doc_u32 as usize)
                .collect()
        });

        fn score_doc<F>(
            candidates: &mut HashMap<usize, CandidateState>,
            doc_u32: usize,
            delta: f32,
            allowed_doc_u32s: Option<&HashSet<usize>>,
            apply: F,
        ) where
            F: FnOnce(&mut CandidateState),
        {
            if let Some(allowed_doc_u32s) = allowed_doc_u32s {
                if !allowed_doc_u32s.contains(&doc_u32) {
                    return;
                }
            }
            let entry = candidates.entry(doc_u32).or_default();
            entry.score += delta;
            apply(entry);
        }

        let lexical_merge_start = Instant::now();
        if let Some(lexical_hits) = lexical_hits {
            for (doc_id, bm25_score) in lexical_hits {
                if let Some(&doc_u32) = self.doc_id_to_u32.get(doc_id) {
                    score_doc(
                        &mut candidates,
                        doc_u32 as usize,
                        *bm25_score,
                        allowed_doc_u32s.as_ref(),
                        |entry| {
                            entry.breakdown.lexical_score += *bm25_score;
                        },
                    );
                }
            }
        }
        let lexical_merge_ms = lexical_merge_start.elapsed().as_secs_f64() * 1000.0;

        let posting_scoring_start = Instant::now();

        // Full-query entity hit.
        if let Some(entity_u32) = self.entity_trie.get(&q) {
            if let Some(postings) = self.entity_postings_doc.get(entity_u32 as usize) {
                for &(doc_u32, post_score) in postings {
                    let mut delta = FULL_QUERY_ENTITY_WEIGHT * post_score;
                    if !query_entities.is_empty() {
                        delta *= 1.25;
                    }
                    score_doc(
                        &mut candidates,
                        doc_u32 as usize,
                        delta,
                        allowed_doc_u32s.as_ref(),
                        |entry| {
                            entry.breakdown.entity_score += delta;
                            entry.matched_entities.push(q.clone());
                        },
                    );
                }
            }
        }

        for term in &q_terms {
            let mut entity_ids = Vec::new();
            if let Some(entity_u32) = self.entity_trie.get(term) {
                entity_ids.push((entity_u32, 1.0f32));
            } else if term.len() >= 4 {
                for entity_u32 in self.entity_trie.prefix_ids(term, 1) {
                    entity_ids.push((entity_u32, ENTITY_PREFIX_MULTIPLIER));
                }
            }
            for (entity_u32, mult) in entity_ids {
                if let Some(postings) = self.entity_postings_doc.get(entity_u32 as usize) {
                    for &(doc_u32, post_score) in postings {
                        let exact_entity_match = query_entities.contains(term);
                        let mut delta = ENTITY_TERM_WEIGHT * post_score * mult;
                        if exact_entity_match {
                            delta *= 1.4;
                        }
                        score_doc(
                            &mut candidates,
                            doc_u32 as usize,
                            delta,
                            allowed_doc_u32s.as_ref(),
                            |entry| {
                                entry.breakdown.entity_score += delta;
                                entry.matched_entities.push(term.clone());
                            },
                        );
                    }
                }
            }
            let mut term_ids = Vec::new();
            if let Some(term_u32) = self.term_trie.get(term) {
                term_ids.push((term_u32, 1.0f32));
            } else if term.len() >= 4 {
                for term_u32 in self.term_trie.prefix_ids(term, 1) {
                    term_ids.push((term_u32, IMPORTANT_TERM_PREFIX_MULTIPLIER));
                }
            }
            for (term_u32, mult) in term_ids {
                if let Some(postings) = self.term_postings_doc.get(term_u32 as usize) {
                    for &(doc_u32, post_score) in postings {
                        let delta = IMPORTANT_TERM_WEIGHT * post_score * mult;
                        score_doc(
                            &mut candidates,
                            doc_u32 as usize,
                            delta,
                            allowed_doc_u32s.as_ref(),
                            |entry| {
                                entry.breakdown.term_score += delta;
                                entry.matched_terms.push(term.clone());
                            },
                        );
                    }
                }
            }
            if self.claim_scoring {
                if let Some(postings) = self.claim_to_docs.get(term) {
                    for posting in postings {
                        if let Some(&doc_u32) = self.doc_id_to_u32.get(&posting.doc_id) {
                            let delta = 0.7 * posting.score;
                            score_doc(
                                &mut candidates,
                                doc_u32 as usize,
                                delta,
                                allowed_doc_u32s.as_ref(),
                                |entry| {
                                    entry.breakdown.claim_score += delta;
                                    entry.matched_terms.push(term.clone());
                                },
                            );
                        }
                    }
                }
            }
        }

        // Expanded semantic terms are lower-weight than original terms.
        for term in &expanded_terms {
            if let Some(entity_u32) = self.entity_trie.get(term) {
                if let Some(postings) = self.entity_postings_doc.get(entity_u32 as usize) {
                    for &(doc_u32, post_score) in postings {
                        let mut delta = EXPANDED_ENTITY_WEIGHT * post_score;
                        if query_entities.contains(term) {
                            delta *= 1.2;
                        }
                        score_doc(
                            &mut candidates,
                            doc_u32 as usize,
                            delta,
                            allowed_doc_u32s.as_ref(),
                            |entry| {
                                entry.breakdown.entity_score += delta;
                            },
                        );
                    }
                }
            }
            if let Some(term_u32) = self.term_trie.get(term) {
                if let Some(postings) = self.term_postings_doc.get(term_u32 as usize) {
                    for &(doc_u32, post_score) in postings {
                        let delta = EXPANDED_TERM_WEIGHT * post_score;
                        score_doc(
                            &mut candidates,
                            doc_u32 as usize,
                            delta,
                            allowed_doc_u32s.as_ref(),
                            |entry| {
                                entry.breakdown.term_score += delta;
                            },
                        );
                    }
                }
            }
            if self.claim_scoring {
                if let Some(postings) = self.claim_to_docs.get(term) {
                    for posting in postings {
                        if let Some(&doc_u32) = self.doc_id_to_u32.get(&posting.doc_id) {
                            let delta = 0.7 * 0.6 * posting.score;
                            score_doc(
                                &mut candidates,
                                doc_u32 as usize,
                                delta,
                                allowed_doc_u32s.as_ref(),
                                |entry| {
                                    entry.breakdown.claim_score += delta;
                                },
                            );
                        }
                    }
                }
            }
        }
        let posting_scoring_ms = posting_scoring_start.elapsed().as_secs_f64() * 1000.0;

        let routing_seed_start = Instant::now();
        if let Some(intent) = query_routing_intent {
            let candidate_doc_ids = candidates.keys().copied().collect::<Vec<_>>();
            seed_routing_candidates(
                self,
                &mut candidates,
                &candidate_doc_ids,
                &q_terms,
                &query_entities,
                intent,
            );
        }
        let routing_seed_ms = routing_seed_start.elapsed().as_secs_f64() * 1000.0;
        let sparse_scoring_ms = sparse_start.elapsed().as_secs_f64() * 1000.0;

        let candidate_accumulation_start = Instant::now();
        let metadata_start = Instant::now();
        let candidate_doc_ids = candidates.keys().copied().collect::<Vec<_>>();
        for doc_u32 in candidate_doc_ids {
            let Some(doc_id) = self.doc_u32_to_id.get(doc_u32) else {
                continue;
            };
            let Some(doc) = self.docs.get(doc_id) else {
                continue;
            };
            let entry = candidates.entry(doc_u32).or_default();
            if let Some(topic) = doc.probable_topic.as_ref() {
                let topic_tokens = tokenize_query_terms(topic);
                let overlap = topic_tokens
                    .iter()
                    .filter(|t| query_set.contains(*t))
                    .count();
                if overlap > 0 {
                    let delta = TOPIC_OVERLAP_WEIGHT * overlap as f32;
                    entry.score += delta;
                    entry.breakdown.topic_score += delta;
                }
            }
            if let Some(dt) = doc.doc_type_guess.as_ref() {
                let dt_tokens = tokenize_query_terms(dt);
                let overlap = dt_tokens.iter().filter(|t| query_set.contains(*t)).count();
                if overlap > 0 {
                    let delta = DOC_TYPE_OVERLAP_WEIGHT * overlap as f32;
                    entry.score += delta;
                    entry.breakdown.doc_type_score += delta;
                }
            }
            if self.claim_scoring && !doc.top_claims.is_empty() {
                let mut best_claim_delta = 0.0f32;
                for claim in &doc.top_claims {
                    let claim_terms = claim_tokens(claim);
                    if claim_terms.is_empty() {
                        continue;
                    }
                    let matched = claim_terms
                        .iter()
                        .filter(|token| query_set.contains(*token))
                        .count();
                    if matched == 0 {
                        continue;
                    }
                    let coverage = matched as f32 / claim_terms.len().max(1) as f32;
                    let mut delta = 0.45 * coverage * claim.confidence.max(0.1);
                    if matched >= 2 {
                        delta += 0.15 * claim.confidence.max(0.1);
                    }
                    if delta > best_claim_delta {
                        best_claim_delta = delta;
                    }
                }
                if best_claim_delta > 0.0 {
                    entry.score += best_claim_delta;
                    entry.breakdown.claim_score += best_claim_delta;
                }
            }
            if doc_has_timestamped_chunk(doc) || doc.timestamp.is_some() {
                let delta = TIMESTAMP_PRESENCE_WEIGHT;
                entry.score += delta;
                entry.breakdown.recency_score += delta;
            }
        }
        let candidate_accumulation_ms =
            candidate_accumulation_start.elapsed().as_secs_f64() * 1000.0;
        let metadata_ms = metadata_start.elapsed().as_secs_f64() * 1000.0;

        let candidate_rank_start = Instant::now();
        let mut ranked_docs: Vec<(usize, f32)> = candidates
            .iter()
            .map(|(doc_u32, state)| (*doc_u32, state.score))
            .collect();
        ranked_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let candidate_rank_ms = candidate_rank_start.elapsed().as_secs_f64() * 1000.0;

        // Graph-aware rerank features with bounded contribution.
        // Use current hybrid score as base so graph signals cannot dominate.
        let disable_followup_boosts = matches!(
            query_routing_intent,
            Some(QueryRoutingIntent::Count) | Some(QueryRoutingIntent::Sum)
        );

        let graph_ms;
        let entity_graph_ms;
        if disable_followup_boosts {
            graph_ms = 0.0;
            entity_graph_ms = 0.0;
        } else {
            let graph_start = Instant::now();
            let anchors = ranked_docs.iter().take(5).cloned().collect::<Vec<_>>();

            // 1-hop doc-link proximity boost.
            for (anchor_idx, anchor_score) in &anchors {
                let anchor_doc_id = &self.doc_u32_to_id[*anchor_idx];
                let Some(anchor_doc) = self.docs.get(anchor_doc_id) else {
                    continue;
                };
                let anchor_weight = (*anchor_score / (1.0 + *anchor_score)).clamp(0.0, 1.0);
                for linked in &anchor_doc.doc_links {
                    if let Some(&doc_u32) = self.doc_id_to_u32.get(linked) {
                        let Some(entry) = candidates.get_mut(&(doc_u32 as usize)) else {
                            continue;
                        };
                        let delta = (DOC_LINK_GRAPH_WEIGHT * anchor_weight)
                            .min(GRAPH_MAX_BOOST - entry.breakdown.graph_link_score);
                        if delta > 0.0 {
                            entry.score += delta;
                            entry.breakdown.graph_link_score += delta;
                        }
                    }
                }
            }
            graph_ms = graph_start.elapsed().as_secs_f64() * 1000.0;

            let entity_graph_start = Instant::now();
            // Entity-graph proximity boost: overlap against anchor-entity set.
            let mut anchor_entities: HashSet<String> = HashSet::new();
            for (doc_u32, _) in &anchors {
                if let Some(keys) = self.doc_key_entities.get(*doc_u32) {
                    for key in keys {
                        if !key.is_empty() {
                            anchor_entities.insert(key.clone());
                        }
                    }
                }
            }
            if !anchor_entities.is_empty() {
                let candidate_doc_ids = ranked_docs
                    .iter()
                    .take(ENTITY_GRAPH_MAX_CANDIDATES)
                    .map(|(doc_u32, _)| *doc_u32)
                    .collect::<Vec<_>>();
                for doc_u32 in candidate_doc_ids {
                    let Some(entry) = candidates.get_mut(&doc_u32) else {
                        continue;
                    };
                    let Some(keys) = self.doc_key_entities.get(doc_u32) else {
                        continue;
                    };
                    let overlap = keys.iter().filter(|k| anchor_entities.contains(*k)).count();
                    if overlap > 0 {
                        let delta = (ENTITY_GRAPH_WEIGHT * overlap as f32)
                            .min(GRAPH_MAX_BOOST - entry.breakdown.entity_graph_score);
                        if delta > 0.0 {
                            entry.score += delta;
                            entry.breakdown.entity_graph_score += delta;
                        }
                    }
                }
            }
            entity_graph_ms = entity_graph_start.elapsed().as_secs_f64() * 1000.0;
        }

        let sequence_rerank_start = Instant::now();
        let rerank_window = ranked_docs
            .iter()
            .take(TEXT_RERANK_WINDOW.min(FINAL_RERANK_WINDOW))
            .map(|(doc_u32, _)| *doc_u32)
            .collect::<Vec<_>>();
        let skip_sequence_rerank = query_routing_intent.is_some() && !has_explicit_temporal;
        if !skip_sequence_rerank && !q_terms.is_empty() {
            for doc_u32 in rerank_window {
                let Some(candidate_tokens) = cached_doc_rerank_tokens(self, doc_u32) else {
                    let Some(doc_id) = self.doc_u32_to_id.get(doc_u32) else {
                        continue;
                    };
                    let Some(doc) = self.docs.get(doc_id) else {
                        continue;
                    };
                    let fallback_text = normalize_for_index(&candidate_rerank_text(doc));
                    let owned_tokens = tokenize_query_terms(&fallback_text);
                    if owned_tokens.is_empty() {
                        continue;
                    }
                    let token_overlap = token_overlap_ratio(&q_terms, &owned_tokens);
                    let mut delta = TEXT_RERANK_WEIGHT * token_overlap;
                    if self.text_rerank_ngram {
                        let ngram_overlap =
                            ngram_overlap_ratio(&q_terms, &owned_tokens, TEXT_RERANK_NGRAM_SIZE);
                        delta += TEXT_RERANK_NGRAM_WEIGHT * ngram_overlap;
                    }
                    if self.text_rerank_lcs {
                        let lcs_overlap = lcs_ratio(&q_terms, &owned_tokens);
                        delta += TEXT_RERANK_LCS_WEIGHT * lcs_overlap;
                    }
                    if delta <= 0.0 {
                        continue;
                    }
                    if let Some(entry) = candidates.get_mut(&doc_u32) {
                        entry.score += delta;
                        entry.breakdown.sequence_rerank_score += delta;
                    }
                    continue;
                };
                if candidate_tokens.is_empty() {
                    continue;
                }
                let token_overlap = token_overlap_ratio(&q_terms, candidate_tokens);
                let mut delta = TEXT_RERANK_WEIGHT * token_overlap;
                if self.text_rerank_ngram {
                    let ngram_overlap =
                        ngram_overlap_ratio(&q_terms, candidate_tokens, TEXT_RERANK_NGRAM_SIZE);
                    delta += TEXT_RERANK_NGRAM_WEIGHT * ngram_overlap;
                }
                if self.text_rerank_lcs {
                    let lcs_overlap = lcs_ratio(&q_terms, candidate_tokens);
                    delta += TEXT_RERANK_LCS_WEIGHT * lcs_overlap;
                }
                if delta <= 0.0 {
                    continue;
                }
                if let Some(entry) = candidates.get_mut(&doc_u32) {
                    entry.score += delta;
                    entry.breakdown.sequence_rerank_score += delta;
                }
            }
        }
        let sequence_rerank_ms = sequence_rerank_start.elapsed().as_secs_f64() * 1000.0;

        let count_query = {
            let lower = query.to_lowercase();
            lower.starts_with("how many")
                || lower.starts_with("count ")
                || lower.contains(" number of ")
                || lower.contains(" number of")
                || lower.contains("count the ")
                || lower.contains("times did i")
        };
        let group_build_start = Instant::now();
        let mut ranked_docs: Vec<(usize, f32)> = candidates
            .iter()
            .filter_map(|(doc_u32, state)| {
                if state.score > 0.0 {
                    Some((*doc_u32, state.score))
                } else {
                    None
                }
            })
            .collect();
        ranked_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let ranking_start = Instant::now();
        let ranked_docs = ranked_docs
            .into_iter()
            .take(FINAL_RERANK_WINDOW.min(self.doc_u32_to_id.len()))
            .collect::<Vec<_>>();
        let mut grouped_ranked_docs: HashMap<String, Vec<(usize, f32)>> = HashMap::new();
        let mut group_metadata: HashMap<String, (Option<String>, String)> = HashMap::new();
        for (doc_u32, score) in ranked_docs {
            let Some(doc_id) = self.doc_u32_to_id.get(doc_u32) else {
                continue;
            };
            let Some(doc) = self.docs.get(doc_id) else {
                continue;
            };
            let group_key = doc
                .group_id
                .clone()
                .unwrap_or_else(|| format!("doc:{}", doc.doc_id));
            grouped_ranked_docs
                .entry(group_key.clone())
                .or_default()
                .push((doc_u32, score));
            group_metadata
                .entry(group_key)
                .or_insert_with(|| (doc.group_id.clone(), doc.source.clone()));
        }
        let group_build_ms = group_build_start.elapsed().as_secs_f64() * 1000.0;

        let mut ranked_groups: Vec<(String, f32, Vec<(usize, f32)>)> = grouped_ranked_docs
            .into_iter()
            .map(|(group_key, mut items)| {
                items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut unique_entities: HashSet<String> = HashSet::new();
                let mut unique_terms: HashSet<String> = HashSet::new();
                let mut dates: Vec<NaiveDate> = Vec::new();
                let support_threshold = (items[0].1 * 0.35).max(0.15);
                let mut supporting_docs = 0usize;

                for (doc_u32, score) in &items {
                    if let Some(state) = candidates.get(doc_u32) {
                        if *score >= support_threshold {
                            supporting_docs += 1;
                        }
                        for entity in &state.matched_entities {
                            if !entity.is_empty() {
                                unique_entities.insert(entity.clone());
                            }
                        }
                        for term in &state.matched_terms {
                            if !term.is_empty() {
                                unique_terms.insert(term.clone());
                            }
                        }
                    }
                    if let Some(doc_id) = self.doc_u32_to_id.get(*doc_u32) {
                        if let Some(doc) = self.docs.get(doc_id) {
                            if let Some(date) = doc_temporal_date(doc) {
                                dates.push(date);
                            }
                        }
                    }
                }

                let mut score = aggregate_group_score(&items, count_query);
                if !query_entities.is_empty() {
                    let exact_entity_matches = items
                        .iter()
                        .filter(|(doc_u32, _)| {
                            candidates
                                .get(doc_u32)
                                .map(|state| {
                                    state
                                        .matched_entities
                                        .iter()
                                        .any(|entity| query_entities.contains(entity.as_str()))
                                })
                                .unwrap_or(false)
                        })
                        .count();
                    score += (exact_entity_matches.min(3) as f32) * 0.10;
                    score += (unique_entities
                        .iter()
                        .filter(|entity| query_entities.contains(*entity))
                        .count()
                        .min(3) as f32)
                        * 0.08;
                }
                score += (unique_entities.len().min(4) as f32) * 0.04;
                score += (unique_terms.len().min(6) as f32) * 0.015;
                score += (supporting_docs.min(4) as f32) * if count_query { 0.06 } else { 0.03 };
                if dates.len() >= 2 {
                    dates.sort();
                    let span_days = dates
                        .last()
                        .zip(dates.first())
                        .map(|(end, start)| end.signed_duration_since(*start).num_days().abs())
                        .unwrap_or(0);
                    if span_days <= 7 {
                        score += if count_query { 0.12 } else { 0.08 };
                    } else if span_days <= 30 {
                        score += if count_query { 0.06 } else { 0.04 };
                    }
                }
                (group_key, score, items)
            })
            .collect();
        let mut group_sort_ms = 0.0;
        let group_sort_start = Instant::now();
        ranked_groups.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        group_sort_ms += group_sort_start.elapsed().as_secs_f64() * 1000.0;
        let evidence_start = Instant::now();
        if let Some(intent) = query_routing_intent {
            let evidence_terms = routing_content_terms(&q_terms);
            let unit_terms = routing_unit_terms(&q_terms);
            for (_group_key, score, items) in ranked_groups.iter_mut() {
                let mut evidence_supporting_docs = 0usize;
                let mut evidence_score = 0.0f32;
                let mut evidence_term_matches = 0usize;
                let mut evidence_numbers = 0usize;
                let mut evidence_units = 0usize;
                let mut evidence_dates = 0usize;
                let mut evidence_predicates = 0usize;
                let mut evidence_penalty = 0.0f32;
                for (doc_u32, _) in items.iter() {
                    if let Some(features) = self.evidence_features_for_doc(
                        *doc_u32,
                        &evidence_terms,
                        &unit_terms,
                        intent,
                    ) {
                        if features.score > 0.0 {
                            evidence_supporting_docs += 1;
                            evidence_score += features.score;
                            evidence_term_matches += features.matched_terms;
                            evidence_numbers += usize::from(features.has_number);
                            evidence_units += usize::from(features.has_unit);
                            evidence_dates +=
                                usize::from(features.has_date || features.has_temporal_terms);
                            evidence_predicates += usize::from(features.predicate_signal);
                            evidence_penalty += features.distractor_penalty;
                        }
                    }
                }
                let evidence_boost = group_evidence_boost(
                    intent,
                    evidence_supporting_docs,
                    evidence_score,
                    evidence_term_matches,
                    evidence_numbers,
                    evidence_units,
                    evidence_dates,
                    evidence_predicates,
                    evidence_penalty,
                );
                *score = *score * 0.80 + evidence_boost;
            }
            let group_sort_start = Instant::now();
            ranked_groups
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            group_sort_ms += group_sort_start.elapsed().as_secs_f64() * 1000.0;
        }
        let evidence_ms = evidence_start.elapsed().as_secs_f64() * 1000.0;

        let mut per_group_counts: HashMap<String, usize> = HashMap::new();
        let mut results = Vec::new();
        for (group_key, _, items) in ranked_groups {
            let (group_id, _) = group_metadata
                .get(&group_key)
                .cloned()
                .unwrap_or((None, String::new()));
            for (doc_u32, score) in items {
                if results.len() >= top_k.min(self.doc_u32_to_id.len()) {
                    break;
                }
                let Some(doc_id) = self.doc_u32_to_id.get(doc_u32).cloned() else {
                    continue;
                };
                let Some(doc) = self.docs.get(&doc_id) else {
                    continue;
                };
                if let Some(group) = group_id.as_ref() {
                    let count = per_group_counts.entry(group.clone()).or_default();
                    if *count >= MAX_RESULTS_PER_GROUP {
                        continue;
                    }
                    *count += 1;
                }
                let state = candidates.get(&doc_u32);
                results.push(SearchResult {
                    doc_id,
                    source: doc.source.clone(),
                    group_id: group_id.clone(),
                    score,
                    score_breakdown: state.map(|s| s.breakdown.clone()).unwrap_or_default(),
                    matched_entities: state
                        .map(|s| {
                            let mut v = s.matched_entities.clone();
                            v.sort();
                            v.dedup();
                            v
                        })
                        .unwrap_or_default(),
                    matched_terms: state
                        .map(|s| {
                            let mut v = s.matched_terms.clone();
                            v.sort();
                            v.dedup();
                            v
                        })
                        .unwrap_or_default(),
                    probable_topic: doc.probable_topic.clone(),
                    doc_type_guess: doc.doc_type_guess.clone(),
                });
            }
            if results.len() >= top_k.min(self.doc_u32_to_id.len()) {
                break;
            }
        }
        let ranking_ms = ranking_start.elapsed().as_secs_f64() * 1000.0;
        let rerank_ms = rerank_start.elapsed().as_secs_f64() * 1000.0;
        (
            results,
            QueryTimings {
                total_ms: rerank_ms,
                refresh_ms: 0.0,
                lexical_bm25_ms: 0.0,
                snapshot_query_ms: rerank_ms,
                rerank_ms,
                parse_ms,
                sparse_scoring_ms,
                lexical_merge_ms,
                posting_scoring_ms,
                routing_seed_ms,
                candidate_accumulation_ms,
                candidate_rank_ms,
                metadata_ms,
                graph_ms,
                entity_graph_ms,
                sequence_rerank_ms,
                evidence_ms,
                group_build_ms,
                group_sort_ms,
                ranking_ms,
            },
            QueryDiagnostics {
                query_terms: q_terms.len(),
                expanded_terms: expanded_terms.len(),
                lexical_hits: lexical_hits.map_or(0, HashMap::len),
                candidates: candidates.len(),
            },
        )
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
                        group_id: d.group_id.clone(),
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

    pub fn impacted_chunks_for_line_range(
        &self,
        doc_id: &str,
        start_line: usize,
        end_line: usize,
    ) -> Vec<String> {
        let Some(&doc_u32) = self.doc_id_to_u32.get(doc_id) else {
            return Vec::new();
        };
        let Some(tree) = self.doc_interval_trees.get(doc_u32 as usize) else {
            return Vec::new();
        };
        let mut chunk_ids = tree
            .query(start_line.max(1), end_line.max(start_line))
            .into_iter()
            .filter_map(|chunk_u32| {
                self.chunks
                    .get(chunk_u32 as usize)
                    .map(|m| m.chunk_id.clone())
            })
            .collect::<Vec<_>>();
        chunk_ids.sort();
        chunk_ids.dedup();
        chunk_ids
    }
}

pub(crate) fn build_semantic_doc_state(
    record: &DocRecord,
    claim_scoring: bool,
) -> SemanticDocState {
    let mut state = SemanticDocState {
        doc_id: record.doc_id.clone(),
        chunk_ids: Vec::new(),
        chunks: HashMap::new(),
        entity_to_docs: HashMap::new(),
        term_to_docs: HashMap::new(),
        claim_to_docs: HashMap::new(),
        topic: record
            .probable_topic
            .as_ref()
            .map(|topic| topic.to_lowercase()),
        doc_type: record
            .doc_type_guess
            .as_ref()
            .map(|doc_type| doc_type.to_lowercase()),
    };

    let chunks = if record.section_chunks.is_empty() {
        vec![SectionChunk {
            chunk_id: stable_chunk_id(
                &record.doc_id,
                record
                    .headings
                    .first()
                    .map(String::as_str)
                    .unwrap_or("(document)"),
                &record.content,
                1,
                record.content.lines().count().max(1),
            ),
            heading: record
                .headings
                .first()
                .cloned()
                .unwrap_or_else(|| "(document)".to_string()),
            content: record.content.clone(),
            start_line: 1,
            end_line: record.content.lines().count().max(1),
            timestamp: record.timestamp.clone(),
            key_entities: record
                .key_entities
                .iter()
                .map(|e| normalize_for_index(&e.text))
                .filter(|v| !v.is_empty())
                .collect(),
            important_terms: record
                .important_terms
                .iter()
                .map(|t| normalize_for_index(&t.term))
                .filter(|v| !v.is_empty())
                .collect(),
        }]
    } else {
        record.section_chunks.clone()
    };

    for chunk in &chunks {
        state.chunk_ids.push(chunk.chunk_id.clone());
        state.chunks.insert(
            chunk.chunk_id.clone(),
            SemanticChunkState {
                chunk_id: chunk.chunk_id.clone(),
                doc_id: record.doc_id.clone(),
                heading: chunk.heading.clone(),
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                key_entities: chunk.key_entities.clone(),
                important_terms: chunk.important_terms.clone(),
            },
        );
        for term in &chunk.important_terms {
            let key = normalize_for_index(term);
            if key.is_empty() {
                continue;
            }
            state
                .term_to_docs
                .entry(key)
                .or_default()
                .push(TermPosting {
                    doc_id: record.doc_id.clone(),
                    score: 0.8,
                });
        }
        for token in tokenize_query_terms(&chunk.heading) {
            state
                .term_to_docs
                .entry(token)
                .or_default()
                .push(TermPosting {
                    doc_id: record.doc_id.clone(),
                    score: 0.4,
                });
        }
        for entity in &chunk.key_entities {
            let key = normalize_for_index(entity);
            if key.is_empty() {
                continue;
            }
            state
                .entity_to_docs
                .entry(key)
                .or_default()
                .push(EntityPosting {
                    doc_id: record.doc_id.clone(),
                    score: 0.9,
                });
        }
    }

    if claim_scoring {
        for claim in &record.top_claims {
            for token in claim_tokens(claim) {
                state
                    .claim_to_docs
                    .entry(token)
                    .or_default()
                    .push(TermPosting {
                        doc_id: record.doc_id.clone(),
                        score: claim.confidence.max(0.1),
                    });
            }
        }
    }

    for postings in state.entity_to_docs.values_mut() {
        postings.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    for postings in state.term_to_docs.values_mut() {
        postings.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    for postings in state.claim_to_docs.values_mut() {
        postings.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    state
}

impl SemanticAggregate {
    pub(crate) fn insert_doc_state(&mut self, state: &SemanticDocState) {
        self.doc_to_chunks
            .insert(state.doc_id.clone(), state.chunk_ids.clone());
        for chunk_id in &state.chunk_ids {
            self.chunk_to_doc
                .insert(chunk_id.clone(), state.doc_id.clone());
            if let Some(chunk) = state.chunks.get(chunk_id) {
                self.chunk_ranges
                    .insert(chunk_id.clone(), (chunk.start_line, chunk.end_line));
                for term in &chunk.important_terms {
                    let key = normalize_for_index(term);
                    if key.is_empty() {
                        continue;
                    }
                    self.term_to_chunks
                        .entry(key)
                        .or_default()
                        .push((chunk_id.clone(), 0.8));
                }
                for token in tokenize_query_terms(&chunk.heading) {
                    self.term_to_chunks
                        .entry(token)
                        .or_default()
                        .push((chunk_id.clone(), 0.4));
                }
                for entity in &chunk.key_entities {
                    let key = normalize_for_index(entity);
                    if key.is_empty() {
                        continue;
                    }
                    self.entity_to_chunks
                        .entry(key)
                        .or_default()
                        .push((chunk_id.clone(), 0.9));
                }
            }
        }
        for (key, postings) in &state.entity_to_docs {
            self.entity_to_docs
                .entry(key.clone())
                .or_default()
                .extend(postings.clone());
            if let Some(entries) = self.entity_to_docs.get_mut(key) {
                entries.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
        for (key, postings) in &state.term_to_docs {
            self.term_to_docs
                .entry(key.clone())
                .or_default()
                .extend(postings.clone());
            if let Some(entries) = self.term_to_docs.get_mut(key) {
                entries.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
        for (key, postings) in &state.claim_to_docs {
            self.claim_to_docs
                .entry(key.clone())
                .or_default()
                .extend(postings.clone());
            if let Some(entries) = self.claim_to_docs.get_mut(key) {
                entries.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
        if let Some(topic) = state.topic.as_ref() {
            self.topic_to_docs
                .entry(topic.clone())
                .or_default()
                .push(state.doc_id.clone());
        }
        if let Some(doc_type) = state.doc_type.as_ref() {
            self.doc_type_to_docs
                .entry(doc_type.clone())
                .or_default()
                .push(state.doc_id.clone());
        }
    }

    pub(crate) fn remove_doc(&mut self, doc_id: &str) {
        if let Some(chunk_ids) = self.doc_to_chunks.remove(doc_id) {
            for chunk_id in chunk_ids {
                self.chunk_to_doc.remove(&chunk_id);
                self.chunk_ranges.remove(&chunk_id);
                for postings in self.term_to_chunks.values_mut() {
                    postings.retain(|(id, _)| id != &chunk_id);
                }
                for postings in self.entity_to_chunks.values_mut() {
                    postings.retain(|(id, _)| id != &chunk_id);
                }
            }
        }
        self.term_to_chunks
            .retain(|_, postings| !postings.is_empty());
        self.entity_to_chunks
            .retain(|_, postings| !postings.is_empty());
        for postings in self.entity_to_docs.values_mut() {
            postings.retain(|posting| posting.doc_id != doc_id);
        }
        self.entity_to_docs
            .retain(|_, postings| !postings.is_empty());
        for postings in self.term_to_docs.values_mut() {
            postings.retain(|posting| posting.doc_id != doc_id);
        }
        self.term_to_docs.retain(|_, postings| !postings.is_empty());
        for postings in self.claim_to_docs.values_mut() {
            postings.retain(|posting| posting.doc_id != doc_id);
        }
        self.claim_to_docs
            .retain(|_, postings| !postings.is_empty());
        for docs in self.topic_to_docs.values_mut() {
            docs.retain(|id| id != doc_id);
        }
        self.topic_to_docs.retain(|_, docs| !docs.is_empty());
        for docs in self.doc_type_to_docs.values_mut() {
            docs.retain(|id| id != doc_id);
        }
        self.doc_type_to_docs.retain(|_, docs| !docs.is_empty());
    }
}

fn tokenize_query_terms(input: &str) -> Vec<String> {
    static TOKEN_RE: OnceLock<Regex> = OnceLock::new();
    let token_re =
        TOKEN_RE.get_or_init(|| Regex::new(r"[A-Za-z][A-Za-z0-9_-]{2,}").expect("valid regex"));
    token_re
        .find_iter(input)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

fn claim_tokens(claim: &Claim) -> Vec<String> {
    let mut tokens = Vec::new();
    tokens.extend(tokenize_query_terms(&normalize_for_index(&claim.subject)));
    tokens.extend(tokenize_query_terms(&normalize_for_index(&claim.predicate)));
    tokens.extend(tokenize_query_terms(&normalize_for_index(&claim.object)));
    tokens.sort();
    tokens.dedup();
    tokens
}

fn sanitize_bm25_query(query: &str) -> String {
    static FALLBACK_RE: OnceLock<Regex> = OnceLock::new();
    let lowered = deunicode(query).to_lowercase();
    let token_re = FALLBACK_RE.get_or_init(|| {
        Regex::new(r"[A-Za-z0-9][A-Za-z0-9_-]*").expect("valid fallback query regex")
    });
    token_re
        .find_iter(&lowered)
        .map(|m| m.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

fn candidate_rerank_text(doc: &DocRecord) -> String {
    let mut parts: Vec<String> = Vec::new();
    if !doc.headings.is_empty() {
        parts.push(doc.headings.join(" "));
    }
    if let Some(topic) = doc.probable_topic.as_ref() {
        parts.push(topic.clone());
    }
    if let Some(doc_type) = doc.doc_type_guess.as_ref() {
        parts.push(doc_type.clone());
    }
    if !doc.important_terms.is_empty() {
        parts.push(
            doc.important_terms
                .iter()
                .map(|t| t.term.clone())
                .collect::<Vec<_>>()
                .join(" "),
        );
    }
    if !doc.key_entities.is_empty() {
        parts.push(
            doc.key_entities
                .iter()
                .map(|e| e.text.clone())
                .collect::<Vec<_>>()
                .join(" "),
        );
    }
    if !doc.temporal_terms.is_empty() {
        parts.push(doc.temporal_terms.join(" "));
    }
    let content_snippet: String = doc
        .content
        .chars()
        .take(TEXT_RERANK_CONTENT_CHARS)
        .collect();
    if !content_snippet.is_empty() {
        parts.push(content_snippet);
    }
    parts.join(" ")
}

fn derive_doc_postings(
    chunk_postings: &[Vec<(u32, f32)>],
    chunks: &[ChunkMeta],
    cap: usize,
) -> Vec<Vec<(u32, f32)>> {
    chunk_postings
        .iter()
        .map(|postings| {
            let mut doc_scores: HashMap<u32, f32> = HashMap::new();
            for &(chunk_u32, score) in postings {
                let doc_u32 = chunks[chunk_u32 as usize].doc_u32;
                doc_scores
                    .entry(doc_u32)
                    .and_modify(|s| *s = s.max(score))
                    .or_insert(score);
            }
            let mut out: Vec<(u32, f32)> = doc_scores.into_iter().collect();
            out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            out.truncate(cap);
            out
        })
        .collect()
}

fn build_doc_rerank_cache(doc: &DocRecord) -> (String, Vec<String>) {
    let rerank_text = candidate_rerank_text(doc);
    let normalized = normalize_for_index(&rerank_text);
    let tokens = tokenize_query_terms(&normalized);
    (normalized, tokens)
}

fn cached_doc_rerank_text<'a>(index: &'a MemoryIndex, doc_u32: usize) -> Option<&'a str> {
    index.doc_rerank_texts.get(doc_u32).map(|s| s.as_str())
}

fn cached_doc_rerank_tokens<'a>(index: &'a MemoryIndex, doc_u32: usize) -> Option<&'a [String]> {
    index
        .doc_rerank_tokens
        .get(doc_u32)
        .map(|tokens| tokens.as_slice())
}

fn doc_has_timestamped_chunk(doc: &DocRecord) -> bool {
    doc.section_chunks
        .iter()
        .any(|chunk| chunk.timestamp.is_some())
}

const TEMPORAL_PROXIMITY_WEIGHT: f32 = 0.45;
const TEMPORAL_RANGE_BOOST: f32 = 0.15;

fn doc_temporal_date(doc: &DocRecord) -> Option<NaiveDate> {
    doc.section_chunks
        .iter()
        .find_map(|chunk| parse_temporal_date(chunk.timestamp.as_deref()))
        .or_else(|| parse_temporal_date(doc.timestamp.as_deref()))
}

fn normalize_temporal_bounds(
    starts_from: Option<&str>,
    ends_at: Option<&str>,
) -> (Option<NaiveDate>, Option<NaiveDate>) {
    let start = parse_temporal_date(starts_from);
    let end = parse_temporal_date(ends_at);
    match (start, end) {
        (Some(start), Some(end)) if start > end => (Some(end), Some(start)),
        (start, end) => (start, end),
    }
}

fn seed_routing_candidates(
    index: &MemoryIndex,
    candidates: &mut HashMap<usize, CandidateState>,
    candidate_doc_ids: &[usize],
    q_terms: &[String],
    query_entities: &HashSet<String>,
    intent: QueryRoutingIntent,
) {
    if candidate_doc_ids.is_empty() {
        return;
    }

    let content_terms = routing_content_terms(q_terms);
    let unit_terms = routing_unit_terms(q_terms);

    for doc_u32 in candidate_doc_ids.iter().copied() {
        let Some(doc_id) = index.doc_u32_to_id.get(doc_u32) else {
            continue;
        };
        let Some(doc) = index.docs.get(doc_id) else {
            continue;
        };
        let Some(doc_terms) = cached_doc_rerank_tokens(index, doc_u32) else {
            continue;
        };
        if doc_terms.is_empty() {
            continue;
        }
        let matched_terms = content_terms
            .iter()
            .filter(|term| doc_terms.iter().any(|tok| tok == *term))
            .count() as f32;
        let doc_entities = index
            .doc_key_entities
            .get(doc_u32)
            .map(|keys| keys.as_slice())
            .unwrap_or(&[]);
        let matched_entities = doc_entities
            .iter()
            .filter(|entity| query_entities.contains(entity.as_str()))
            .count() as f32;
        let has_number = index.doc_has_number.get(doc_u32).copied().unwrap_or(false);
        let has_unit = !unit_terms.is_empty()
            && unit_terms
                .iter()
                .any(|unit| doc_terms.iter().any(|tok| tok == unit));
        let has_date = doc_temporal_date(doc).is_some();
        let has_temporal_terms = !doc.temporal_terms.is_empty();

        let mut score = match intent {
            QueryRoutingIntent::Count => {
                matched_terms * 0.24
                    + matched_entities * 0.35
                    + if has_date { 0.05 } else { 0.0 }
                    + if has_temporal_terms { 0.04 } else { 0.0 }
            }
            QueryRoutingIntent::Sum => {
                matched_terms * 0.20
                    + matched_entities * 0.12
                    + if has_number { 0.30 } else { 0.0 }
                    + if has_unit { 0.18 } else { 0.0 }
                    + if has_date { 0.04 } else { 0.0 }
            }
            QueryRoutingIntent::Sequence => {
                matched_terms * 0.20
                    + matched_entities * 0.12
                    + if has_date { 0.28 } else { 0.0 }
                    + if has_temporal_terms { 0.14 } else { 0.0 }
                    + if has_number { 0.08 } else { 0.0 }
            }
        };

        if score <= 0.0 {
            continue;
        }
        if matches!(intent, QueryRoutingIntent::Sequence) && !has_date && !has_temporal_terms {
            continue;
        }
        if matches!(intent, QueryRoutingIntent::Sum) && !has_number && !has_unit {
            continue;
        }

        score = score.min(0.85);
        let entry = candidates.entry(doc_u32).or_default();
        entry.score += score;
        entry.breakdown.semantic_score += score;
    }
}

impl MemoryIndex {
    fn evidence_features_for_doc(
        &self,
        doc_u32: usize,
        content_terms: &[String],
        unit_terms: &[String],
        intent: QueryRoutingIntent,
    ) -> Option<EvidenceFeatures> {
        let doc_id = self.doc_u32_to_id.get(doc_u32)?;
        let doc = self.docs.get(doc_id)?;
        let normalized = cached_doc_rerank_text(self, doc_u32)?;
        if normalized.is_empty() {
            return None;
        }

        let doc_terms: HashSet<String> = cached_doc_rerank_tokens(self, doc_u32)
            .map(|tokens| tokens.iter().cloned().collect())
            .unwrap_or_else(|| tokenize_query_terms(normalized).into_iter().collect());
        let matched_terms = content_terms
            .iter()
            .filter(|term| doc_terms.contains(term.as_str()))
            .count();
        let has_number = self
            .doc_has_number
            .get(doc_u32)
            .copied()
            .unwrap_or_else(|| contains_number_like(normalized));
        let has_unit = !unit_terms.is_empty()
            && unit_terms
                .iter()
                .any(|unit| doc_terms.contains(unit.as_str()));
        if matched_terms == 0 && !has_number && !has_unit {
            return None;
        }
        let has_date = doc_temporal_date(doc).is_some();
        let has_temporal_terms = !doc.temporal_terms.is_empty();
        let predicate_signal = has_predicate_signal(&normalized, intent);
        let distractor_penalty = routing_distractor_penalty(&normalized, intent);

        let mut score = matched_terms.min(6) as f32 * 0.18;
        if predicate_signal {
            score += 0.55;
        }
        if has_date || has_temporal_terms {
            score += match intent {
                QueryRoutingIntent::Sequence => 0.45,
                _ => 0.12,
            };
        }
        if has_number {
            score += match intent {
                QueryRoutingIntent::Sum => 0.45,
                QueryRoutingIntent::Sequence => 0.12,
                QueryRoutingIntent::Count => 0.08,
            };
        }
        if has_unit {
            score += match intent {
                QueryRoutingIntent::Sum => 0.30,
                QueryRoutingIntent::Sequence => 0.16,
                QueryRoutingIntent::Count => 0.06,
            };
        }
        score -= distractor_penalty;

        if !predicate_signal && matched_terms < 2 {
            score *= 0.25;
        }

        Some(EvidenceFeatures {
            score: score.max(0.0),
            matched_terms,
            has_number,
            has_unit,
            has_date,
            has_temporal_terms,
            predicate_signal,
            distractor_penalty,
        })
    }
}

fn group_evidence_boost(
    intent: QueryRoutingIntent,
    supporting_docs: usize,
    evidence_score: f32,
    evidence_terms: usize,
    evidence_numbers: usize,
    evidence_units: usize,
    evidence_dates: usize,
    evidence_predicates: usize,
    evidence_penalty: f32,
) -> f32 {
    if supporting_docs == 0 {
        return 0.0;
    }

    let mut boost = evidence_score.min(4.0);
    boost += supporting_docs.min(4) as f32 * 0.22;
    boost += evidence_terms.min(8) as f32 * 0.04;
    boost += evidence_predicates.min(3) as f32 * 0.24;

    match intent {
        QueryRoutingIntent::Count => {
            boost += supporting_docs.saturating_sub(1).min(3) as f32 * 0.22;
        }
        QueryRoutingIntent::Sum => {
            boost += evidence_numbers.min(3) as f32 * 0.30;
            boost += evidence_units.min(3) as f32 * 0.18;
        }
        QueryRoutingIntent::Sequence => {
            boost += evidence_dates.min(4) as f32 * 0.30;
            boost += evidence_numbers.min(2) as f32 * 0.12;
        }
    }

    (boost - evidence_penalty.min(1.5)).max(0.0)
}

fn routing_content_terms(q_terms: &[String]) -> Vec<String> {
    let stop = routing_stopwords();
    let mut seen = HashSet::new();
    q_terms
        .iter()
        .filter(|term| term.len() >= 3)
        .filter(|term| !stop.contains(term.as_str()))
        .filter(|term| seen.insert((*term).clone()))
        .take(16)
        .cloned()
        .collect()
}

fn routing_stopwords() -> &'static HashSet<&'static str> {
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

fn routing_unit_terms(query: &[String]) -> Vec<String> {
    let units = [
        "mile",
        "miles",
        "km",
        "kilometer",
        "kilometers",
        "meter",
        "meters",
        "hour",
        "hours",
        "minute",
        "minutes",
        "day",
        "days",
        "week",
        "weeks",
        "month",
        "months",
        "dollar",
        "dollars",
        "usd",
        "pound",
        "pounds",
        "kg",
        "kilogram",
        "kilograms",
        "screen",
        "time",
    ];
    let unit_set: HashSet<&str> = units.into_iter().collect();
    query
        .iter()
        .filter(|term| unit_set.contains(term.as_str()))
        .cloned()
        .collect()
}

fn contains_number_like(text: &str) -> bool {
    static NUMBER_RE: OnceLock<Regex> = OnceLock::new();
    let re = NUMBER_RE.get_or_init(|| {
        Regex::new(
            r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
        )
        .expect("valid number regex")
    });
    re.is_match(text)
}

fn has_predicate_signal(text: &str, intent: QueryRoutingIntent) -> bool {
    let terms = match intent {
        QueryRoutingIntent::Count => [
            "led",
            "lead",
            "leading",
            "visited",
            "attended",
            "bought",
            "purchased",
            "baked",
            "cooked",
            "tried",
            "used",
            "played",
            "learned",
            "presented",
            "participated",
            "sibling",
            "brother",
            "sister",
            "doctor",
            "project",
        ]
        .as_slice(),
        QueryRoutingIntent::Sum => [
            "total",
            "distance",
            "cost",
            "spent",
            "paid",
            "miles",
            "kilometers",
            "hours",
            "days",
        ]
        .as_slice(),
        QueryRoutingIntent::Sequence => [
            "consecutive",
            "before",
            "after",
            "weekend",
            "days",
            "weeks",
            "earliest",
            "latest",
            "first",
            "last",
        ]
        .as_slice(),
    };
    terms.iter().any(|term| text.contains(term))
}

fn routing_distractor_penalty(text: &str, intent: QueryRoutingIntent) -> f32 {
    let generic_markers = [
        "write a story",
        "fiction",
        "character",
        "captain america",
        "area of",
        "formula",
        "table showing",
        "census",
        "survey",
        "article",
        "research studies",
        "here are some",
        "resources",
    ];
    let mut penalty = generic_markers
        .iter()
        .filter(|marker| text.contains(**marker))
        .count() as f32
        * 0.35;

    if matches!(intent, QueryRoutingIntent::Count) {
        if text.contains("how many") && text.contains("here are") {
            penalty += 0.25;
        }
        if text.contains("formula") || text.contains("approximately") {
            penalty += 0.35;
        }
    }

    penalty.min(1.4)
}

fn aggregate_group_score(items: &[(usize, f32)], count_query: bool) -> f32 {
    if items.is_empty() {
        return 0.0;
    }
    let base = items[0].1;
    let support = items
        .iter()
        .skip(1)
        .take(4)
        .enumerate()
        .map(|(idx, (_, score))| {
            let decay = if count_query {
                match idx {
                    0 => 0.45,
                    1 => 0.28,
                    2 => 0.16,
                    _ => 0.08,
                }
            } else {
                match idx {
                    0 => 0.30,
                    1 => 0.18,
                    2 => 0.10,
                    _ => 0.05,
                }
            };
            decay * *score
        })
        .sum::<f32>();
    let supporting_docs = items
        .iter()
        .skip(1)
        .filter(|(_, score)| *score >= (items[0].1 * 0.35).max(0.15))
        .count();
    let coverage = supporting_docs.min(4) as f32 * if count_query { 0.08 } else { 0.05 };
    if count_query {
        base * 0.75 + support + coverage
    } else {
        base + support + coverage
    }
}

fn normalized_entity_keys(entities: &[Tier1Entity]) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for entity in entities {
        let key = normalize_for_index(&entity.text);
        if key.is_empty() || !seen.insert(key.clone()) {
            continue;
        }
        out.push(key);
    }
    out
}

// normalize_for_index moved to query_expansion.rs and reused here.

fn token_overlap_ratio(query_tokens: &[String], candidate_tokens: &[String]) -> f32 {
    if query_tokens.is_empty() || candidate_tokens.is_empty() {
        return 0.0;
    }
    let candidate_set: HashSet<&str> = candidate_tokens.iter().map(String::as_str).collect();
    let matched = query_tokens
        .iter()
        .map(String::as_str)
        .filter(|t| candidate_set.contains(t))
        .count();
    matched as f32 / query_tokens.len().max(1) as f32
}

fn ngram_overlap_ratio(query_tokens: &[String], candidate_tokens: &[String], n: usize) -> f32 {
    if n == 0 || query_tokens.len() < n || candidate_tokens.len() < n {
        return 0.0;
    }
    let query_ngrams = build_ngrams(query_tokens, n);
    if query_ngrams.is_empty() {
        return 0.0;
    }
    let candidate_ngrams = build_ngrams(candidate_tokens, n);
    if candidate_ngrams.is_empty() {
        return 0.0;
    }
    let candidate_set: HashSet<Vec<String>> = candidate_ngrams.into_iter().collect();
    let overlap = query_ngrams
        .into_iter()
        .filter(|g| candidate_set.contains(g))
        .count();
    overlap as f32 / query_tokens.len().saturating_sub(n - 1).max(1) as f32
}

fn build_ngrams(tokens: &[String], n: usize) -> Vec<Vec<String>> {
    if n == 0 || tokens.len() < n {
        return Vec::new();
    }
    tokens
        .windows(n)
        .map(|win| win.iter().cloned().collect::<Vec<_>>())
        .collect()
}

fn lcs_ratio(query_tokens: &[String], candidate_tokens: &[String]) -> f32 {
    if query_tokens.is_empty() || candidate_tokens.is_empty() {
        return 0.0;
    }
    let mut prev = vec![0usize; candidate_tokens.len() + 1];
    let mut cur = vec![0usize; candidate_tokens.len() + 1];
    for q in query_tokens {
        for (j, cand) in candidate_tokens.iter().enumerate() {
            if q == cand {
                cur[j + 1] = prev[j] + 1;
            } else {
                cur[j + 1] = cur[j].max(prev[j + 1]);
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[candidate_tokens.len()] as f32 / query_tokens.len().max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_overlap_prefers_matching_candidate() {
        let query = tokenize_query_terms(&normalize_for_index("What degree did I graduate with?"));
        let good = tokenize_query_terms(&normalize_for_index(
            "I graduated with a computer science degree.",
        ));
        let bad = tokenize_query_terms(&normalize_for_index("I cooked pasta and watched a movie."));

        assert!(token_overlap_ratio(&query, &good) > token_overlap_ratio(&query, &bad));
        assert!(ngram_overlap_ratio(&query, &good, 2) > ngram_overlap_ratio(&query, &bad, 2));
        assert!(lcs_ratio(&query, &good) > lcs_ratio(&query, &bad));
    }

    #[test]
    fn timestamped_chunk_detects_chunk_level_temporal_anchor() {
        let doc = DocRecord {
            doc_id: "doc-1".to_string(),
            source: "source://doc-1".to_string(),
            content: "content".to_string(),
            timestamp: None,
            doc_length: 7,
            author_agent: None,
            group_id: None,
            filters: std::collections::BTreeMap::new(),
            probable_topic: None,
            doc_type_guess: None,
            headings: vec!["Overview".to_string()],
            doc_links: vec![],
            temporal_terms: vec![],
            key_entities: vec![],
            important_terms: vec![],
            section_chunks: vec![SectionChunk {
                chunk_id: "doc-1::0".to_string(),
                heading: "Overview".to_string(),
                content: "content".to_string(),
                start_line: 1,
                end_line: 1,
                timestamp: Some("2024-05-10".to_string()),
                key_entities: vec![],
                important_terms: vec![],
            }],
            embedding: None,
            top_claims: vec![],
            provenance: Provenance {
                source: "source://doc-1".to_string(),
                timestamp: None,
                ner_provider: "heuristic".to_string(),
                term_ranker: "yake".to_string(),
                index_version: "v1".to_string(),
            },
        };

        assert!(doc_has_timestamped_chunk(&doc));
    }

    #[test]
    fn normalize_temporal_bounds_orders_reversed_ranges_and_preserves_same_day() {
        let (start, end) = normalize_temporal_bounds(Some("2024-05-10"), Some("2024-05-01"));
        assert_eq!(
            start,
            Some(chrono::NaiveDate::from_ymd_opt(2024, 5, 1).expect("valid date"))
        );
        assert_eq!(
            end,
            Some(chrono::NaiveDate::from_ymd_opt(2024, 5, 10).expect("valid date"))
        );

        let (same_start, same_end) =
            normalize_temporal_bounds(Some("2024-05-10"), Some("2024-05-10"));
        assert_eq!(same_start, same_end);
        assert_eq!(
            same_start,
            Some(chrono::NaiveDate::from_ymd_opt(2024, 5, 10).expect("valid date"))
        );
    }
}
