use crate::query_expansion::{expand_query_terms, normalize_for_index};
use crate::tier1::{RankedTerm, Tier1Entity};
use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::document::TantivyDocument;
use tantivy::schema::{Field, Schema, STORED, STRING, TEXT};
use tantivy::schema::Value;
use tantivy::{doc, Index, IndexReader};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

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
    pub probable_topic: Option<String>,
    pub doc_type_guess: Option<String>,
    pub headings: Vec<String>,
    #[serde(default)]
    pub doc_links: Vec<String>,
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
            let Some(next) = self.nodes.get(cur).and_then(|n| n.children.get(&ch)).copied() else {
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
    pub graph_link_score: f32,
    pub entity_graph_score: f32,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedMemoryCore {
    entity_to_docs: HashMap<String, Vec<EntityPosting>>,
    term_to_docs: HashMap<String, Vec<TermPosting>>,
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
        Self::from_records_with_lexical_dir(records, None)
    }

    pub fn from_records_with_lexical_dir(records: Vec<DocRecord>, lexical_dir: Option<&Path>) -> Self {
        let mut docs = HashMap::new();
        let mut entity_to_docs: HashMap<String, Vec<EntityPosting>> = HashMap::new();
        let mut term_to_docs: HashMap<String, Vec<TermPosting>> = HashMap::new();
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

        for mut record in records {
            let doc_id = record.doc_id.clone();
            let doc_u32 = doc_u32_to_id.len() as u32;
            doc_id_to_u32.insert(doc_id.clone(), doc_u32);
            doc_u32_to_id.push(doc_id.clone());
            doc_to_chunks.push(Vec::new());
            if record.section_chunks.is_empty() {
                record.section_chunks.push(SectionChunk {
                    chunk_id: format!("{}::0", doc_id),
                    heading: record
                        .headings
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "(document)".to_string()),
                    content: record.content.clone(),
                    start_line: 1,
                    end_line: record.content.lines().count().max(1),
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

        term_to_docs.clear();
        entity_to_docs.clear();
        for (term_u32, postings) in term_postings_chunk.iter().enumerate() {
            let key = &term_u32_to_key[term_u32];
            for (chunk_u32, score) in postings {
                let doc_u32 = chunks[*chunk_u32 as usize].doc_u32;
                let doc_id = &doc_u32_to_id[doc_u32 as usize];
                term_to_docs.entry(key.clone()).or_default().push(TermPosting {
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
                entity_to_docs.entry(key.clone()).or_default().push(EntityPosting {
                    doc_id: doc_id.clone(),
                    score: *score,
                });
            }
        }

        let (term_trie, entity_trie, doc_interval_trees) = Self::build_runtime_helpers(
            &term_lexicon,
            &entity_lexicon,
            &chunks,
            doc_u32_to_id.len(),
        );

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
            chunk_terms,
            chunk_entities,
            term_trie,
            entity_trie,
            doc_interval_trees,
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
            if meta.end_line >= meta.start_line && meta.end_line > 0 && (meta.doc_u32 as usize) < doc_count {
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
    ) -> Result<Self> {
        let bytes = fs::read(core_path)?;
        let core: PersistedMemoryCore = bincode::deserialize(&bytes)?;
        if core.doc_u32_to_id.is_empty() && !records.is_empty() {
            anyhow::bail!("invalid binary core: empty doc table");
        }
        let mut docs = HashMap::new();
        for mut record in records {
            if record.section_chunks.is_empty() {
                record.section_chunks.push(SectionChunk {
                    chunk_id: format!("{}::0", record.doc_id),
                    heading: record
                        .headings
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "(document)".to_string()),
                    content: record.content.clone(),
                    start_line: 1,
                    end_line: record.content.lines().count().max(1),
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
        Ok(Self {
            docs,
            entity_to_docs: core.entity_to_docs,
            term_to_docs: core.term_to_docs,
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
            chunk_terms: core.chunk_terms,
            chunk_entities: core.chunk_entities,
            term_trie,
            entity_trie,
            doc_interval_trees,
        })
    }

    pub fn save_binary_core(&self, core_path: &Path) -> Result<()> {
        if let Some(parent) = core_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let core = PersistedMemoryCore {
            entity_to_docs: self.entity_to_docs.clone(),
            term_to_docs: self.term_to_docs.clone(),
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

    fn build_lexical_index(docs: &HashMap<String, DocRecord>, lexical_dir: Option<&Path>) -> Result<LexicalIndex> {
        let mut schema_builder = Schema::builder();
        let doc_id_f = schema_builder.add_text_field("doc_id", STRING | STORED);
        let content_f = schema_builder.add_text_field("content", TEXT);
        let headings_f = schema_builder.add_text_field("headings", TEXT);
        let terms_f = schema_builder.add_text_field("important_terms", TEXT);
        let entities_f = schema_builder.add_text_field("entities", TEXT);
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
                        writer.add_document(doc!(
                            doc_id_f => doc.doc_id.clone(),
                            content_f => content_text,
                            headings_f => headings_text,
                            terms_f => terms_text,
                            entities_f => entities_text
                        ))?;
                    }
                    writer.commit()?;
                    created
                }
            }
        } else {
            let ram = Index::create_in_ram(schema);
            let mut writer = ram.writer(50_000_000)?;
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
                writer.add_document(doc!(
                    doc_id_f => doc.doc_id.clone(),
                    content_f => content_text,
                    headings_f => headings_text,
                    terms_f => terms_text,
                    entities_f => entities_text
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
        Ok(LexicalIndex { index, reader, doc_id_f, content_f, headings_f, terms_f, entities_f })
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
        let expanded = expand_query_terms(&q_terms);
        let expanded_terms = expanded.expanded_terms;
        let bm25_query = if expanded_terms.is_empty() {
            q.clone()
        } else {
            format!("{} {}", q, expanded_terms.join(" "))
        };

        let doc_count = self.doc_u32_to_id.len();
        if doc_count == 0 {
            return Vec::new();
        }
        let mut scores: Vec<f32> = vec![0.0; doc_count];
        let mut breakdowns: Vec<ScoreBreakdown> = vec![ScoreBreakdown::default(); doc_count];
        let mut matched_entities: Vec<Vec<String>> = vec![Vec::new(); doc_count];
        let mut matched_terms: Vec<Vec<String>> = vec![Vec::new(); doc_count];
        let query_set: HashSet<String> = q_terms.iter().cloned().collect();

        match self.lexical_bm25(&bm25_query, top_k.saturating_mul(5).max(20)) {
            Ok(lexical_hits) => {
                for (doc_id, bm25_score) in lexical_hits {
                    if let Some(&doc_u32) = self.doc_id_to_u32.get(&doc_id) {
                        let idx = doc_u32 as usize;
                        scores[idx] += bm25_score;
                        breakdowns[idx].lexical_score += bm25_score;
                    }
                }
            }
            Err(err) => {
                eprintln!("warning: lexical BM25 query component failed: {}", err);
            }
        }

        // Full-query entity hit.
        if let Some(entity_u32) = self.entity_trie.get(&q) {
            if let Some(postings) = self.entity_postings_chunk.get(entity_u32 as usize) {
                for (chunk_u32, post_score) in postings {
                    let doc_u32 = self.chunks[*chunk_u32 as usize].doc_u32;
                    let idx = doc_u32 as usize;
                    let delta = 1.5 * *post_score;
                    scores[idx] += delta;
                    breakdowns[idx].entity_score += delta;
                    matched_entities[idx].push(q.clone());
                }
            }
        }

        for term in &q_terms {
            let mut entity_ids = Vec::new();
            if let Some(entity_u32) = self.entity_trie.get(term) {
                entity_ids.push((entity_u32, 1.0f32));
            } else if term.len() >= 4 {
                for entity_u32 in self.entity_trie.prefix_ids(term, 1) {
                    entity_ids.push((entity_u32, 0.25));
                }
            }
            for (entity_u32, mult) in entity_ids {
                if let Some(postings) = self.entity_postings_chunk.get(entity_u32 as usize) {
                    for (chunk_u32, post_score) in postings {
                        let doc_u32 = self.chunks[*chunk_u32 as usize].doc_u32;
                        let idx = doc_u32 as usize;
                        let delta = 1.2 * *post_score * mult;
                        scores[idx] += delta;
                        breakdowns[idx].entity_score += delta;
                        matched_entities[idx].push(term.clone());
                    }
                }
            }
            let mut term_ids = Vec::new();
            if let Some(term_u32) = self.term_trie.get(term) {
                term_ids.push((term_u32, 1.0f32));
            } else if term.len() >= 4 {
                for term_u32 in self.term_trie.prefix_ids(term, 1) {
                    term_ids.push((term_u32, 0.25));
                }
            }
            for (term_u32, mult) in term_ids {
                if let Some(postings) = self.term_postings_chunk.get(term_u32 as usize) {
                    for (chunk_u32, post_score) in postings {
                        let doc_u32 = self.chunks[*chunk_u32 as usize].doc_u32;
                        let idx = doc_u32 as usize;
                        let delta = 0.8 * *post_score * mult;
                        scores[idx] += delta;
                        breakdowns[idx].term_score += delta;
                        matched_terms[idx].push(term.clone());
                    }
                }
            }
        }

        // Expanded semantic terms are lower-weight than original terms.
        for term in &expanded_terms {
            if let Some(entity_u32) = self.entity_trie.get(term) {
                if let Some(postings) = self.entity_postings_chunk.get(entity_u32 as usize) {
                    for (chunk_u32, post_score) in postings {
                        let doc_u32 = self.chunks[*chunk_u32 as usize].doc_u32;
                        let idx = doc_u32 as usize;
                        let delta = 0.45 * *post_score;
                        scores[idx] += delta;
                        breakdowns[idx].entity_score += delta;
                    }
                }
            }
            if let Some(term_u32) = self.term_trie.get(term) {
                if let Some(postings) = self.term_postings_chunk.get(term_u32 as usize) {
                    for (chunk_u32, post_score) in postings {
                        let doc_u32 = self.chunks[*chunk_u32 as usize].doc_u32;
                        let idx = doc_u32 as usize;
                        let delta = 0.35 * *post_score;
                        scores[idx] += delta;
                        breakdowns[idx].term_score += delta;
                    }
                }
            }
        }

        for (doc_id, doc) in &self.docs {
            let Some(&doc_u32) = self.doc_id_to_u32.get(doc_id) else {
                continue;
            };
            let idx = doc_u32 as usize;
            if let Some(topic) = doc.probable_topic.as_ref() {
                let topic_tokens = tokenize_query_terms(topic);
                let overlap = topic_tokens
                    .iter()
                    .filter(|t| query_set.contains(*t))
                    .count();
                if overlap > 0 {
                    let delta = 0.35 * overlap as f32;
                    scores[idx] += delta;
                    breakdowns[idx].topic_score += delta;
                }
            }
            if let Some(dt) = doc.doc_type_guess.as_ref() {
                let dt_tokens = tokenize_query_terms(dt);
                let overlap = dt_tokens.iter().filter(|t| query_set.contains(*t)).count();
                if overlap > 0 {
                    let delta = 0.25 * overlap as f32;
                    scores[idx] += delta;
                    breakdowns[idx].doc_type_score += delta;
                }
            }
            if doc.timestamp.is_some() {
                let delta = 0.05;
                scores[idx] += delta;
                breakdowns[idx].recency_score += delta;
            }
        }

        // Graph-aware rerank features with bounded contribution.
        // Use current hybrid score as base so graph signals cannot dominate.
        let base_scores = scores.clone();
        let mut ranked_docs: Vec<(usize, f32)> = base_scores
            .iter()
            .enumerate()
            .filter_map(|(i, s)| if *s > 0.0 { Some((i, *s)) } else { None })
            .collect();
        ranked_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let anchors = ranked_docs.iter().take(5).cloned().collect::<Vec<_>>();
        let anchor_doc_ids = anchors
            .iter()
            .map(|(idx, _)| self.doc_u32_to_id[*idx].clone())
            .collect::<Vec<_>>();

        // 1-hop doc-link proximity boost.
        for (anchor_idx, anchor_score) in &anchors {
            let anchor_doc_id = &self.doc_u32_to_id[*anchor_idx];
            let Some(anchor_doc) = self.docs.get(anchor_doc_id) else {
                continue;
            };
            let anchor_weight = (*anchor_score / (1.0 + *anchor_score)).clamp(0.0, 1.0);
            for linked in &anchor_doc.doc_links {
                if let Some(&doc_u32) = self.doc_id_to_u32.get(linked) {
                    let idx = doc_u32 as usize;
                    if base_scores[idx] <= 0.0 {
                        continue;
                    }
                    let delta = (0.22 * anchor_weight).min(0.6 - breakdowns[idx].graph_link_score);
                    if delta > 0.0 {
                        scores[idx] += delta;
                        breakdowns[idx].graph_link_score += delta;
                    }
                }
            }
        }

        // Entity-graph proximity boost: overlap against anchor-entity set.
        let mut anchor_entities: HashSet<String> = HashSet::new();
        for doc_id in &anchor_doc_ids {
            if let Some(doc) = self.docs.get(doc_id) {
                for e in &doc.key_entities {
                    let k = normalize_for_index(&e.text);
                    if !k.is_empty() {
                        anchor_entities.insert(k);
                    }
                }
            }
        }
        if !anchor_entities.is_empty() {
            for (doc_id, doc) in &self.docs {
                let Some(&doc_u32) = self.doc_id_to_u32.get(doc_id) else {
                    continue;
                };
                let idx = doc_u32 as usize;
                if base_scores[idx] <= 0.0 {
                    continue;
                }
                let overlap = doc
                    .key_entities
                    .iter()
                    .map(|e| normalize_for_index(&e.text))
                    .filter(|k| !k.is_empty() && anchor_entities.contains(k))
                    .count();
                if overlap > 0 {
                    let delta = (0.08 * overlap as f32).min(0.6 - breakdowns[idx].entity_graph_score);
                    if delta > 0.0 {
                        scores[idx] += delta;
                        breakdowns[idx].entity_graph_score += delta;
                    }
                }
            }
        }

        struct MaxSegTree {
            n: usize,
            tree: Vec<(f32, usize)>,
        }
        impl MaxSegTree {
            fn build(values: &[f32]) -> Self {
                let mut n = 1usize;
                while n < values.len() {
                    n <<= 1;
                }
                let mut tree = vec![(f32::NEG_INFINITY, usize::MAX); 2 * n];
                for (i, &v) in values.iter().enumerate() {
                    tree[n + i] = (v, i);
                }
                for i in (1..n).rev() {
                    tree[i] = if tree[i << 1].0 >= tree[i << 1 | 1].0 {
                        tree[i << 1]
                    } else {
                        tree[i << 1 | 1]
                    };
                }
                Self { n, tree }
            }
            fn best(&self) -> (f32, usize) {
                self.tree[1]
            }
            fn set(&mut self, idx: usize, value: f32) {
                let mut p = self.n + idx;
                self.tree[p] = (value, idx);
                while p > 1 {
                    p >>= 1;
                    self.tree[p] = if self.tree[p << 1].0 >= self.tree[p << 1 | 1].0 {
                        self.tree[p << 1]
                    } else {
                        self.tree[p << 1 | 1]
                    };
                }
            }
        }

        let mut seg = MaxSegTree::build(&scores);
        let mut out = Vec::new();
        let limit = top_k.min(doc_count);
        for _ in 0..limit {
            let (best_score, idx) = seg.best();
            if idx == usize::MAX || !best_score.is_finite() || best_score <= 0.0 {
                break;
            }
            let doc_id = &self.doc_u32_to_id[idx];
            let Some(doc) = self.docs.get(doc_id) else {
                seg.set(idx, f32::NEG_INFINITY);
                continue;
            };
            out.push(SearchResult {
                doc_id: doc_id.clone(),
                source: doc.source.clone(),
                score: best_score,
                score_breakdown: breakdowns[idx].clone(),
                matched_entities: {
                    let mut v = matched_entities[idx].clone();
                    v.sort();
                    v.dedup();
                    v
                },
                matched_terms: {
                    let mut v = matched_terms[idx].clone();
                    v.sort();
                    v.dedup();
                    v
                },
                probable_topic: doc.probable_topic.clone(),
                doc_type_guess: doc.doc_type_guess.clone(),
            });
            seg.set(idx, f32::NEG_INFINITY);
        }
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
            .filter_map(|chunk_u32| self.chunks.get(chunk_u32 as usize).map(|m| m.chunk_id.clone()))
            .collect::<Vec<_>>();
        chunk_ids.sort();
        chunk_ids.dedup();
        chunk_ids
    }
}

fn tokenize_query_terms(input: &str) -> Vec<String> {
    static TOKEN_RE: OnceLock<Regex> = OnceLock::new();
    let token_re = TOKEN_RE.get_or_init(|| Regex::new(r"[A-Za-z][A-Za-z0-9_-]{2,}").expect("valid regex"));
    token_re
        .find_iter(input)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

// normalize_for_index moved to query_expansion.rs and reused here.
