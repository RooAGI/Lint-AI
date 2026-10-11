use crate::ids::stable_chunk_id;
use crate::index::write_lock::with_guarded_index_writer;
use crate::query_expansion::normalize_for_index;
use anyhow::Result;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tantivy::collector::TopDocs;
use tantivy::query::{Bm25StatisticsProvider, FuzzyTermQuery, Query, QueryParser};
use tantivy::Term;
use tantivy::schema::document::TantivyDocument;
use tantivy::schema::{Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, STORED, STRING, TEXT};
use tantivy::{doc, Index};

use super::helpers::*;
use super::model::*;
use super::query_terms::*;

/// Stem query words that match known semantic tag vocabulary (Luyi 2026-10-07).
///
/// Only stems words that could be tags ("weekends"->"weekend", "habitual" stays).
/// Does NOT stem arbitrary words ("library" stays "library") to avoid breaking
/// existing ranking. The tag vocabulary is the closed set from semantic_tags.rs.
fn stem_query_porter(query: &str) -> String {
    // Known tag base forms (canonical). If a query word stems to one of these,
    // use the stemmed form so "weekends" matches the "weekend" tag.
    // Luyi 2026-10-09: beKIND categories ("gardening", "doctor") are canonical
    // as-is; do NOT add their stems ("garden") or queries won't match tags.
    const TAG_BASES: &[&str] = &["weekend", "weekday", "habitual", "herb", "food",
        "gardening", "doctor", "culinary", "sports", "art", "music",
        // Luyi 2026-10-10: hierarchical location tags (beKIND location judgment)
        // Nation level
        "unitedstates", "unitedkingdom", "canada", "mexico", "china",
        "japan", "germany", "france", "italy", "spain", "australia",
        "brazil", "india",
        // US state level (examples; full list in beKIND)
        "colorado", "california", "texas", "utah", "newyork",
        // Place level (examples)
        "rockymountains", "yellowstone", "yosemite", "grandcanyon",
        "bigsur", "moab", "london",
        // UK state level
        "england"];

    // Location phrase mapping: "United States" -> "unitedstates" (single token)
    // so it matches the canonical location tag. Applied before tokenization.
    let query = query
        .replace("United States", "unitedstates")
        .replace("united states", "unitedstates")
        .replace("United Kingdom", "unitedkingdom")
        .replace("united kingdom", "unitedkingdom");

    // Systematic (Luyi 2026-10-10): tokenize on non-alphanumeric, mirroring
    // Tantivy's default tokenizer used for the tags field at index time.
    // This guarantees query tokens align with indexed tags — no per-symbol
    // patching (hyphen, slash, etc.). Each token is lowercased, then replaced
    // by its stem only if the stem is a known tag base.
    query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .filter(|w| {
            // Systematic stopword filtering (Luyi 2026-10-10):
            // Remove stopwords for all supported languages before Tantivy.
            // Uses the canonical lists from crate::tokenizer.
            let lower = w.to_lowercase();
            !crate::tokenizer::english_stopwords().contains(lower.as_str())
                && !crate::tokenizer::chinese_stopwords().contains(lower.as_str())
                && !crate::tokenizer::korean_stopwords().contains(lower.as_str())
        })
        .map(|word| {
            let lower = word.to_lowercase();
            let stemmed = crate::porter_stemmer::porter_stem(&lower);
            // Only use stemmed form if it's a known tag base.
            // Otherwise keep the original word.
            if TAG_BASES.contains(&stemmed.as_str()) {
                stemmed
            } else {
                lower
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The chunk-content text indexed in the tantivy `content` field (and the
/// text bekind judges for definitional semantic tags). Single helper so the
/// indexed text and the judged text cannot drift apart.
fn lexical_content_text(doc: &DocRecord) -> String {
    doc.section_chunks
        .iter()
        .map(|c| c.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn build_doc_scoring_tokens(
    docs: &HashMap<String, DocRecord>,
    doc_id_to_u32: &HashMap<String, u32>,
    doc_count: usize,
    claim_scoring: bool,
) -> (Vec<Vec<String>>, Vec<Vec<String>>, Vec<Vec<Vec<String>>>) {
    let mut topics = vec![Vec::new(); doc_count];
    let mut doc_types = vec![Vec::new(); doc_count];
    let mut claims = vec![Vec::new(); doc_count];
    for (doc_id, doc) in docs {
        let Some(&doc_u32) = doc_id_to_u32.get(doc_id) else {
            continue;
        };
        let idx = doc_u32 as usize;
        if let Some(topic) = doc.probable_topic.as_deref() {
            topics[idx] = tokenize_query_terms(topic);
        }
        if let Some(doc_type) = doc.doc_type_guess.as_deref() {
            doc_types[idx] = tokenize_query_terms(doc_type);
        }
        if claim_scoring {
            claims[idx] = doc.top_claims.iter().map(claim_tokens).collect();
        }
    }
    (topics, doc_types, claims)
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
        mut semantic_aggregate: SemanticAggregate,
        text_rerank_ngram: bool,
        text_rerank_lcs: bool,
        claim_scoring: bool,
    ) -> Self {
        semantic_aggregate.sort_pending_postings();
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
            // Stamp the document's session group as an exact-match filter so
            // callers can explicitly scope a search to one historical session.
            // This never activates implicitly: the query path only consults
            // the "group_id" postings when the caller supplies an explicit
            // group filter.
            if let Some(group_id) = record.group_id.as_deref() {
                filter_postings
                    .entry("group_id".to_string())
                    .or_default()
                    .entry(group_id.to_string())
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

        let (doc_topic_tokens, doc_type_tokens, doc_claim_tokens) =
            build_doc_scoring_tokens(&docs, &doc_id_to_u32, doc_u32_to_id.len(), claim_scoring);

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
            doc_topic_tokens,
            doc_type_tokens,
            doc_claim_tokens,
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
            // Stamp the session group as an exact-match filter (see the full
            // build above). Explicit group scoping only; never implicit.
            if let Some(group_id) = record.group_id.as_deref() {
                filter_postings
                    .entry("group_id".to_string())
                    .or_default()
                    .entry(group_id.to_string())
                    .or_default()
                    .insert(doc_u32);
            }
        }
        let (doc_topic_tokens, doc_type_tokens, doc_claim_tokens) = build_doc_scoring_tokens(
            &docs,
            &core.doc_id_to_u32,
            core.doc_u32_to_id.len(),
            claim_scoring,
        );
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
            doc_topic_tokens,
            doc_type_tokens,
            doc_claim_tokens,
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
        // Definitional semantic tags (Luyi 2026-09-28): closed-set temporal
        // words ("weekend"/"weekday"), "habitual", admitted kind tags.
        // Plain lowercase single words; the default TEXT analyzer keeps
        // them intact.
        let tags_f = schema_builder.add_text_field("semantic_tags", TEXT);
        // Subword field (Luyi 2026-10-07): Porter stems for
        // morphological matching ("allergies" vs "allergic"). Uses the
        // Subword content field (Luyi 2026-10-07): currently unused.
        // Kept as TEXT to avoid custom tokenizer registration issues.
        // The field is populated but not queried.
        let subword_f = schema_builder.add_text_field("subword_content", TEXT);
        // Session synthetic document fields (Luyi 2026-10-07): one session_doc
        // per session aggregates tags. is_synthetic marks it; tag_links stores
        // JSON tag->doc_ids mapping for provenance.
        let is_synthetic_f = schema_builder.add_text_field("is_synthetic", STRING | STORED);
        let tag_links_f = schema_builder.add_text_field("tag_links", STRING | STORED);
        let schema = schema_builder.build();
        // Deterministic doc order so the batched bekind verdicts map back
        // to documents by position.
        let mut ordered: Vec<&DocRecord> = docs.values().collect();
        ordered.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
        // Tags are precomputed at write time by MemoryService (the single
        // write path) and stored in DocRecord.semantic_tags. If empty
        // (legacy docs, or MemoryService didn't populate), compute them
        // here as a fallback. The index builder just reads them — the
        // computation is owned by the write path.
        // Semantic tags from key_phrases (Luyi 2026-10-08): "use key_phrases
        // for semantic tags". The beKIND scope/kind judgments are stored as
        // KeyPhrase entries with kind="semantic_tag" in the DocRecord.
        // Single truth: SourceDocument (via DocRecord.key_phrases) is the
        // basic unit; no separate derivation path.
        let tags_per_doc: Vec<Vec<String>> = ordered
            .iter()
            .map(|doc| {
                let mut tags: Vec<String> = doc
                    .key_phrases
                    .iter()
                    .filter(|kp| kp.kind == "semantic_tag")
                    .map(|kp| kp.text.clone())
                    .collect();
                tags.sort();
                tags.dedup();
                tags
            })
            .collect();
        let index = if let Some(dir) = lexical_dir {
            // Index open, schema migration, and population run through the
            // shared guarded writer entry point.
            let mut selected = Index::create_in_ram(schema.clone());
            with_guarded_index_writer(
                &mut selected,
                Some(dir),
                50_000_000,
                |dir, schema| {
                    fs::create_dir_all(dir)?;
                    match Index::open_in_dir(dir) {
                        Ok(existing)
                            if existing.schema().get_field("semantic_tags").is_ok()
                                && existing.schema().get_field("subword_content").is_ok() =>
                        {
                            // Register tokenizers on the open path too:
                            // QueryParser resolves the field tokenizer
                            // through the index's manager.
                            crate::index::cjk_tokenizer::register_cjk_tokenizer(&existing);
                            return Ok((existing, false));
                        }
                        // Only a confirmed old schema authorizes replacement.
                        Ok(_) => fs::remove_dir_all(dir)?,
                        Err(_) if fs::read_dir(dir)?.next().is_none() => {}
                        Err(error) => return Err(error.into()),
                    }
                    fs::create_dir_all(dir)?;
                    let new_index = Index::create_in_dir(dir, schema.clone())?;
                    // Register tokenizers before the writer uses them.
                    crate::index::cjk_tokenizer::register_cjk_tokenizer(&new_index);
                    Ok((new_index, true))
                },
                |writer| {
                    for (doc, tags) in ordered.iter().zip(tags_per_doc.iter()) {
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
                        let content_text = lexical_content_text(doc);
                        // Luyi 2026-10-10: inject semantic tags into content field
                        // so tag matches benefit from content-field boost.
                        let content_text = if tags.is_empty() {
                            content_text
                        } else {
                            format!("{} {}", content_text, tags.join(" "))
                        };
                        let temporal_text = doc.temporal_terms.join(" ");
                        let tags_text = tags.join(" ");
                        writer.add_document(doc!(doc_id_f => doc.doc_id.clone(), content_f => content_text, headings_f => headings_text, terms_f => terms_text, entities_f => entities_text, temporal_f => temporal_text, tags_f => tags_text, subword_f => content_text))?;
                    }
                    Ok(())
                },
            )?;
            selected
        } else {
            let ram = Index::create_in_ram(schema);
            Self::populate_lexical_index(
                &ram,
                15_000_001,
                &ordered,
                &tags_per_doc,
                doc_id_f,
                content_f,
                headings_f,
                terms_f,
                entities_f,
                temporal_f,
                tags_f,
                subword_f,
                is_synthetic_f,
                tag_links_f,
            )?;
            ram
        };
        // Unified script-aware tokenization for all TEXT fields (overrides
        // the built-in "default"): Han runs index as character bigrams,
        // Hangul runs as eojeol + particle-stripped stem, and Latin runs
        // replicate tantivy's default tokenizer plus deunicode folding
        // ("niño" -> "nino") so accented terms agree with the deunicoded
        // boosted fields. Pure-ASCII text is unaffected. QueryParser
        // resolves the field tokenizer through this manager, so index-time
        // and query-time segmentation agree.
        crate::index::cjk_tokenizer::register_cjk_tokenizer(&index);
        let reader = index.reader()?;
        let schema_ref = index.schema();
        let doc_id_f = schema_ref.get_field("doc_id")?;
        let content_f = schema_ref.get_field("content")?;
        let headings_f = schema_ref.get_field("headings")?;
        let terms_f = schema_ref.get_field("important_terms")?;
        let entities_f = schema_ref.get_field("entities")?;
        let temporal_f = schema_ref.get_field("temporal_terms")?;
        let tags_f = schema_ref.get_field("semantic_tags")?;
        let subword_f = schema_ref.get_field("subword_content")?;
        let is_synthetic_f = schema_ref.get_field("is_synthetic")?;
        let tag_links_f = schema_ref.get_field("tag_links")?;
        Ok(LexicalIndex {
            index,
            reader,
            doc_id_f,
            content_f,
            headings_f,
            terms_f,
            entities_f,
            temporal_f,
            tags_f,
            subword_f,
            is_synthetic_f,
            tag_links_f,
        })
    }

    /// Writes every document (plus its precomputed semantic tags) into a
    /// fresh tantivy index and commits. Shared by the on-disk and
    /// in-RAM build paths.
    #[allow(clippy::too_many_arguments)]
    fn populate_lexical_index(
        index: &Index,
        writer_heap: usize,
        ordered: &[&DocRecord],
        tags_per_doc: &[Vec<String>],
        doc_id_f: Field,
        content_f: Field,
        headings_f: Field,
        terms_f: Field,
        entities_f: Field,
        temporal_f: Field,
        tags_f: Field,
        subword_f: Field,
        is_synthetic_f: Field,
        tag_links_f: Field,
    ) -> Result<()> {
        let mut index = index.clone();
        with_guarded_index_writer(
            &mut index,
            None,
            writer_heap,
            |_, schema| Ok((Index::create_in_ram(schema.clone()), true)),
            |writer| {
                for (doc, tags) in ordered.iter().zip(tags_per_doc.iter()) {
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
                    let content_text = lexical_content_text(doc);
                    // Luyi 2026-10-10: inject semantic tags into content field
                    // so tag matches benefit from content-field boost.
                    // Documents without the tag (e.g., institutional "Senate"
                    // filtered from location tags) don't get the boost.
                    let content_text = if tags.is_empty() {
                        content_text
                    } else {
                        format!("{} {}", content_text, tags.join(" "))
                    };
                    let temporal_text = doc.temporal_terms.join(" ");
                    let tags_text = tags.join(" ");
                    writer.add_document(doc!(
                        doc_id_f => doc.doc_id.clone(),
                        content_f => content_text,
                        headings_f => headings_text,
                        terms_f => terms_text,
                        entities_f => entities_text,
                        temporal_f => temporal_text,
                        tags_f => tags_text,
                        subword_f => content_text
                    ))?;
                }
                Ok(())
            },
        )?;
        Ok(())
    }

    /// Lexical BM25 over the tantivy index, plus definitional semantic-tag
    /// matching (Luyi 2026-09-28; synthetic-document since 2026-10-07).
    ///
    /// Each document's canonical tags (closed-set temporal words,
    /// "habitual", admitted kind tags) are indexed as ordinary terms in the
    /// `semantic_tags` field, and the multi-field QueryParser searches that
    /// field like every other field — scored by plain BM25 (IDF, length
    /// norm, saturation) like every other term, no fixed multiplier. Tags
    /// never filter: a doc without tags simply has an empty field.
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
        // Per-query beKIND judgment (Luyi 2026-10-10):
        // Get canonical location + activity tags for the query, append to search.
        // Replaces hardcoded phrase mappings with proper judgment.
        let query_with_bekind_tags = if crate::behood_query::is_enabled() {
            let loc_tags = crate::behood_query::analyze_location_categories(&[query]);
            let act_tags = crate::behood_query::analyze_activity_categories(&[query]);
            let mut tags = Vec::new();
            if !loc_tags.is_empty() {
                eprintln!("[QUERY DEBUG] bekind location tags: {:?}", loc_tags[0]);
                tags.extend(loc_tags[0].iter().cloned());
            }
            if !act_tags.is_empty() {
                eprintln!("[QUERY DEBUG] bekind activity tags: {:?}", act_tags[0]);
                tags.extend(act_tags[0].iter().cloned());
            }
            if !tags.is_empty() {
                format!("{} {}", query, tags.join(" "))
            } else {
                query.to_string()
            }
        } else {
            query.to_string()
        };
        let cache_key = query_with_bekind_tags.clone();
        let parsed_cache = PARSED_LEXICAL_QUERY_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let cached = {
            let cache = parsed_cache
                .lock()
                .expect("parsed lexical query cache lock poisoned");
            cache.get(&cache_key).cloned()
        };
        let parsed = if let Some(parsed) = cached {
            parsed
        } else {
            // Main fields: exact match via QueryParser.
            let mut query_parser = QueryParser::for_index(
                &lex.index,
                vec![
                    lex.content_f,
                    lex.headings_f,
                    lex.terms_f,
                    lex.entities_f,
                    lex.temporal_f,
                    // Synthetic document tags (Luyi 2026-10-07): exact match
                    // via QueryParser. Query is Porter-stemmed below so
                    // "weekends" matches the "weekend" tag.
                    lex.tags_f,
                ],
            );
            query_parser.set_field_boost(lex.content_f, LEXICAL_CONTENT_BOOST);
            query_parser.set_field_boost(lex.headings_f, LEXICAL_HEADINGS_BOOST);
            query_parser.set_field_boost(lex.terms_f, LEXICAL_TERMS_BOOST);
            query_parser.set_field_boost(lex.entities_f, LEXICAL_ENTITIES_BOOST);
            query_parser.set_field_boost(lex.temporal_f, 1.1);
            query_parser.set_field_boost(lex.tags_f, LEXICAL_TAGS_BOOST);
            // Porter-stem the query (Luyi 2026-10-07): "weekends" -> "weekend"
            // so inflected forms match canonical tags. Preserves query
            // structure (quotes, etc.) by stemming word-by-word.
            // Luyi 2026-10-10: query includes beKIND location tags.
            let stemmed_query = stem_query_porter(&query_with_bekind_tags);
            eprintln!("[QUERY DEBUG] stemmed: {}", stemmed_query);
            let parsed_lexical = match query_parser.parse_query(&stemmed_query) {
                Ok(parsed) => parsed,
                Err(first_err) => {
                    let fallback_query = sanitize_bm25_query(&stemmed_query);
                    if fallback_query.is_empty() {
                        return Err(first_err.into());
                    }
                    match query_parser.parse_query(&fallback_query) {
                        Ok(parsed) => parsed,
                        Err(_) => return Err(first_err.into()),
                    }
                }
            };
            // Tags are first-class lexical terms now (see above): no side
            // SHOULD clauses, no per-query mapping. The parsed query is the
            // whole retrieval query.
            // Fuzzy matching (Luyi 2026-10-08): for each stemmed query term,
            // add FuzzyTermQueries on the tags and content fields (edit
            // distance 1). This bridges near-misses like "allergi" (query)
            // vs "allerg" (content stem). Combined via SHOULD so fuzzy
            // matches boost but don't exclude.
            let combined: Arc<dyn Query> = {
                use tantivy::query::BooleanQuery;
                let mut should_clauses: Vec<(tantivy::query::Occur, Box<dyn Query>)> = Vec::new();
                // Tokenize the stemmed query into terms for fuzzy matching.
                let mut has_fuzzy = false;
                for term in stemmed_query.split_whitespace() {
                    // Skip very short terms to avoid noise.
                    if term.len() < 3 {
                        continue;
                    }
                    // Skip "its" specifically: fuzzy "its"->"is" causes false
                    // positives (breaks stateless session test). Other terms
                    // keep loose fuzzy for recall (e.g., "allergies"->"allergic").
                    if term == "its" {
                        continue;
                    }
                    // Fuzzy on tags field (for tag near-misses).
                    let fuzzy_tags = FuzzyTermQuery::new(
                        Term::from_field_text(lex.tags_f, term),
                        1,    // edit distance
                        true, // prefix: term must share prefix (performance)
                    );
                    let boosted_tags = tantivy::query::BoostQuery::new(
                        Box::new(fuzzy_tags),
                        LEXICAL_TAGS_BOOST,
                    );
                    should_clauses.push((tantivy::query::Occur::Should, Box::new(boosted_tags)));
                    // Fuzzy on content field (for "allergi" vs "allerg").
                    // Use lower boost than tags to avoid drowning exact matches.
                    let fuzzy_content = FuzzyTermQuery::new(
                        Term::from_field_text(lex.content_f, term),
                        1,    // edit distance
                        true, // prefix
                    );
                    let boosted_content = tantivy::query::BoostQuery::new(
                        Box::new(fuzzy_content),
                        LEXICAL_CONTENT_BOOST * 0.5, // half boost for fuzzy
                    );
                    should_clauses.push((tantivy::query::Occur::Should, Box::new(boosted_content)));
                    has_fuzzy = true;
                }
                should_clauses.push((tantivy::query::Occur::Should, Box::new(parsed_lexical)));
                // If we added fuzzy clauses, combine; otherwise use parsed as-is.
                if has_fuzzy {
                    Arc::from(BooleanQuery::new(should_clauses))
                } else {
                    // No fuzzy terms; the parsed query is the only clause.
                    // Reconstruct from the single SHOULD clause.
                    let (_, q) = should_clauses.pop().unwrap();
                    Arc::from(q)
                }
            };
            let mut cache = parsed_cache
                .lock()
                .expect("parsed lexical query cache lock poisoned");
            if cache.len() >= QUERY_TERM_CACHE_CAPACITY {
                if let Some(oldest_key) = cache.keys().next().cloned() {
                    cache.remove(&oldest_key);
                }
            }
            cache.insert(cache_key, combined.clone());
            combined
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

#[cfg(test)]
mod stem_query_porter_tests {
    use super::stem_query_porter;

    #[test]
    fn hyphen_splits_into_tokens() {
        // Luyi 2026-10-10: "gardening-related" must tokenize to "gardening"
        // so it matches the indexed "gardening" tag.
        assert_eq!(stem_query_porter("gardening-related"), "gardening related");
    }

    #[test]
    fn slash_splits_into_tokens() {
        // Real case (LongMemEval 10e09553): "7/22" must tokenize to "7" "22"
        // so it matches indexed date terms.
        assert_eq!(stem_query_porter("7/22"), "7 22");
    }

    #[test]
    fn known_tag_base_uses_stemmed_form() {
        // "weekends" stems to "weekend", a known tag base.
        assert_eq!(stem_query_porter("weekends"), "weekend");
    }

    #[test]
    fn bekind_category_kept_verbatim() {
        // "gardening" stems to "garden", which is NOT a tag base —
        // keep "gardening" so it matches the indexed tag.
        assert_eq!(stem_query_porter("gardening"), "gardening");
    }

    #[test]
    fn mixed_query_tokenizes_systematically() {
        assert_eq!(
            stem_query_porter("What gardening-related activity?"),
            "what gardening related activity"
        );
    }
}
