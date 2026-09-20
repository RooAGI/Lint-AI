use crate::ids::stable_chunk_id;
use crate::query_expansion::normalize_for_index;
use anyhow::Result;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tantivy::collector::TopDocs;
use tantivy::query::{Bm25StatisticsProvider, Query, QueryParser};
use tantivy::schema::document::TantivyDocument;
use tantivy::schema::{Schema, STORED, STRING, TEXT};
use tantivy::{doc, Index};

use super::helpers::*;
use super::model::*;
use super::query_terms::*;

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
        let mut filter_postings: HashMap<String, HashMap<String, RoaringBitmap>> = HashMap::new();
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
            for (key, value) in &record.filters {
                filter_postings
                    .entry(key.clone())
                    .or_default()
                    .entry(value.clone())
                    .or_default()
                    .insert(doc_u32);
            }
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
            filter_postings,
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
        let (core, _): (PersistedMemoryCore, usize) =
            oxicode::serde::decode_from_slice(&bytes, oxicode::config::standard())?;
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
        let mut filter_postings: HashMap<String, HashMap<String, RoaringBitmap>> = HashMap::new();
        for (doc_id, record) in &docs {
            let Some(&doc_u32) = core.doc_id_to_u32.get(doc_id) else {
                continue;
            };
            for (key, value) in &record.filters {
                filter_postings
                    .entry(key.clone())
                    .or_default()
                    .entry(value.clone())
                    .or_default()
                    .insert(doc_u32);
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
            filter_postings,
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
        Ok(oxicode::serde::encode_to_vec(
            &core,
            oxicode::config::standard(),
        )?)
    }

    pub fn from_bytes(bytes: &[u8], records: Vec<DocRecord>) -> Result<Self> {
        let (core, _): (PersistedMemoryCore, usize) =
            oxicode::serde::decode_from_slice(bytes, oxicode::config::standard())?;
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
        fs::write(
            core_path,
            oxicode::serde::encode_to_vec(&core, oxicode::config::standard())?,
        )?;
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

    pub(crate) fn lexical_bm25(
        &self,
        query: &str,
        top_k: usize,
        statistics: Option<&dyn Bm25StatisticsProvider>,
    ) -> Result<HashMap<String, f32>> {
        let Some(lex) = self.lexical.as_ref() else {
            return Ok(HashMap::new());
        };
        let searcher = lex.reader.searcher();
        let parsed_cache = PARSED_LEXICAL_QUERY_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let cached = {
            let cache = parsed_cache
                .lock()
                .expect("parsed lexical query cache lock poisoned");
            cache.get(query).cloned()
        };
        let parsed = if let Some(parsed) = cached {
            parsed
        } else {
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
            let parsed: Arc<dyn Query> = Arc::from(parsed);
            let mut cache = parsed_cache
                .lock()
                .expect("parsed lexical query cache lock poisoned");
            if cache.len() >= QUERY_TERM_CACHE_CAPACITY {
                if let Some(oldest_key) = cache.keys().next().cloned() {
                    cache.remove(&oldest_key);
                }
            }
            cache.insert(query.to_string(), parsed.clone());
            parsed
        };
        let collector = TopDocs::with_limit(top_k);
        let top_docs = match statistics {
            Some(statistics) => {
                searcher.search_with_statistics_provider(parsed.as_ref(), &collector, statistics)?
            }
            None => searcher.search(parsed.as_ref(), &collector)?,
        };

        let mut out = HashMap::new();
        for (score, addr) in top_docs {
            let retrieved: TantivyDocument = searcher.doc(addr)?;
            if let Some(v) = retrieved.get_first(lex.doc_id_f) {
                // Scoped locally: this trait's blanket impls shadow `String::as_str`
                // if imported at file scope.
                use tantivy::schema::Value as _;
                if let Some(doc_id) = v.as_str() {
                    out.insert(doc_id.to_string(), score);
                }
            }
        }
        Ok(out)
    }
}
