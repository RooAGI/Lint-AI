use crate::chunking::{
    chunk_document_hybrid, chunk_document_lines, chunk_document_sections, enrich_section_chunks,
};
use crate::claim_extractor::{ClaimExtractor, ConservativeClaimExtractor};
use crate::index::{
    build_semantic_doc_state, DocRecord, MemoryIndex, Provenance, QueryDiagnostics, QueryTimings,
    SearchResult, SemanticAggregate, SemanticDocState,
};
use crate::source::SourceDocument;
use crate::temporal::extract_temporal_terms;
use crate::temporal_fact::TemporalFactStore;
use crate::tier1::{
    default_spacy_script_path, CValueStyleTermRanker, HeuristicKeyEntityRanker,
    ImportantTermRanker, KeyEntityRanker, RakeStyleTermRanker, SpacyKeyEntityRanker,
    TextRankStyleTermRanker, Tier1DocInput, YakeStyleTermRanker,
};
use anyhow::Result;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::document::TantivyDocument;
use tantivy::schema::Value;
use tantivy::schema::{Field, Schema, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, Term};

#[derive(Debug, Clone, ValueEnum)]
pub enum Tier1NerProvider {
    Heuristic,
    Spacy,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum Tier1TermRankerKind {
    Yake,
    Rake,
    Cvalue,
    Textrank,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum ChunkStrategy {
    Heading,
    Line,
    Hybrid,
}

#[derive(Debug, Clone)]
pub enum IndexLocation {
    InMemory,
    UnderCorpusRoot,
    Explicit(PathBuf),
}

#[derive(Debug, Clone)]
pub struct PipelineOptions {
    pub ner_provider: Tier1NerProvider,
    pub spacy_model: String,
    pub term_ranker: Tier1TermRankerKind,
    pub chunk_strategy: ChunkStrategy,
    pub chunk_lines: usize,
    pub chunk_overlap: usize,
    pub chunk_target_tokens: usize,
    pub chunk_max_tokens: usize,
    pub text_rerank_ngram: bool,
    pub text_rerank_lcs: bool,
    pub claim_extraction: bool,
    pub index_location: IndexLocation,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            ner_provider: Tier1NerProvider::Heuristic,
            spacy_model: "en_core_web_sm".to_string(),
            term_ranker: Tier1TermRankerKind::Yake,
            chunk_strategy: ChunkStrategy::Heading,
            chunk_lines: 40,
            chunk_overlap: 10,
            chunk_target_tokens: 450,
            chunk_max_tokens: 800,
            text_rerank_ngram: false,
            text_rerank_lcs: false,
            claim_extraction: false,
            index_location: IndexLocation::InMemory,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StorePaths {
    pub root: Option<PathBuf>,
    pub lexical_dir: Option<PathBuf>,
    pub semantic_dir: Option<PathBuf>,
    pub metadata_path: Option<PathBuf>,
}

const STORE_SCHEMA_VERSION: u32 = 1;
const STORE_LAYOUT_VERSION: &str = "index-store-v1";
const SEMANTIC_RECORDS_FILE: &str = "records.json";
const SEMANTIC_CORE_FILE: &str = "core.bin";
const CHUNK_LIFECYCLE_FILE: &str = "chunk_lifecycle.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkLifecycleMeta {
    pub chunk_id: String,
    pub doc_id: String,
    pub lineage_key: String,
    pub version: u32,
    pub is_latest: bool,
    pub supersedes_chunk_id: Option<String>,
    pub updated_at_ms: u64,
    pub change_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentLifecycleMeta {
    pub doc_id: String,
    pub version: u32,
    pub is_latest: bool,
    pub updated_at_ms: u64,
    pub chunk_count: usize,
    pub latest_chunk_ids: Vec<String>,
    pub superseded_chunk_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoreMetadata {
    schema_version: u32,
    layout_version: String,
    crate_version: String,
    index_location: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedSemanticRecords {
    schema_version: u32,
    layout_version: String,
    records: Vec<PersistedDocRecord>,
    #[serde(default)]
    chunk_lifecycle: Vec<ChunkLifecycleMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedDocRecord {
    doc_id: String,
    source: String,
    content: String,
    timestamp: Option<String>,
    doc_length: usize,
    author_agent: Option<String>,
    group_id: Option<String>,
    #[serde(default)]
    filters: std::collections::BTreeMap<String, String>,
    probable_topic: Option<String>,
    doc_type_guess: Option<String>,
    headings: Vec<String>,
    doc_links: Vec<String>,
    #[serde(default)]
    temporal_terms: Vec<String>,
    key_entities: Vec<crate::tier1::Tier1Entity>,
    important_terms: Vec<crate::tier1::RankedTerm>,
    section_chunks: Vec<crate::index::SectionChunk>,
    embedding: Option<Vec<f32>>,
    top_claims: Vec<crate::index::Claim>,
    provenance: Provenance,
}

impl From<DocRecord> for PersistedDocRecord {
    fn from(record: DocRecord) -> Self {
        Self {
            doc_id: record.doc_id,
            source: record.source,
            content: record.content,
            timestamp: record.timestamp,
            doc_length: record.doc_length,
            author_agent: record.author_agent,
            group_id: record.group_id,
            filters: record.filters,
            probable_topic: record.probable_topic,
            doc_type_guess: record.doc_type_guess,
            headings: record.headings,
            doc_links: record.doc_links,
            temporal_terms: record.temporal_terms,
            key_entities: record.key_entities,
            important_terms: record.important_terms,
            section_chunks: record.section_chunks,
            embedding: record.embedding,
            top_claims: record.top_claims,
            provenance: record.provenance,
        }
    }
}

impl From<PersistedDocRecord> for DocRecord {
    fn from(record: PersistedDocRecord) -> Self {
        Self {
            doc_id: record.doc_id,
            source: record.source,
            content: record.content,
            timestamp: record.timestamp,
            doc_length: record.doc_length,
            author_agent: record.author_agent,
            group_id: record.group_id,
            filters: record.filters,
            probable_topic: record.probable_topic,
            doc_type_guess: record.doc_type_guess,
            headings: record.headings,
            doc_links: record.doc_links,
            temporal_terms: record.temporal_terms,
            key_entities: record.key_entities,
            important_terms: record.important_terms,
            section_chunks: record.section_chunks,
            embedding: record.embedding,
            top_claims: record.top_claims,
            provenance: record.provenance,
        }
    }
}

pub fn resolve_store_paths(
    corpus_root: Option<&Path>,
    options: &PipelineOptions,
) -> Result<StorePaths> {
    match &options.index_location {
        IndexLocation::InMemory => Ok(StorePaths {
            root: None,
            lexical_dir: None,
            semantic_dir: None,
            metadata_path: None,
        }),
        IndexLocation::UnderCorpusRoot => {
            let corpus_root = corpus_root
                .map(Path::to_path_buf)
                .ok_or_else(|| anyhow::anyhow!("corpus root is required for UnderCorpusRoot"))?;
            let root = corpus_root.join(".lint-ai");
            Ok(StorePaths {
                lexical_dir: Some(root.join("lexical")),
                semantic_dir: Some(root.join("semantic")),
                metadata_path: Some(root.join("metadata.json")),
                root: Some(root),
            })
        }
        IndexLocation::Explicit(path) => Ok(StorePaths {
            lexical_dir: Some(path.join("lexical")),
            semantic_dir: Some(path.join("semantic")),
            metadata_path: Some(path.join("metadata.json")),
            root: Some(path.clone()),
        }),
    }
}

pub struct IndexStore {
    options: PipelineOptions,
    store_paths: StorePaths,
    source_docs: HashMap<String, SourceDocument>,
    records: HashMap<String, DocRecord>,
    semantic_docs: HashMap<String, SemanticDocState>,
    semantic_aggregate: SemanticAggregate,
    chunk_lifecycle: HashMap<String, ChunkLifecycleMeta>,
    chunk_latest_by_lineage: HashMap<String, String>,
    temporal_facts: TemporalFactStore,
    dirty_docs: HashSet<String>,
    tombstones: HashSet<String>,
    lexical: LexicalState,
    snapshot: Option<MemoryIndex>,
    snapshot_revision: u64,
    store_revision: u64,
    background_refresh: Option<BackgroundRefresh>,
    dirty: bool,
}

struct BackgroundRefresh {
    target_revision: u64,
    receiver: Receiver<Result<MemoryIndex>>,
}

impl IndexStore {
    pub fn new(options: PipelineOptions) -> Self {
        match Self::try_new(options.clone()) {
            Ok(store) => store,
            Err(err) => {
                eprintln!(
                    "warning: index store initialization failed (falling back to in-memory): {}",
                    err
                );
                let fallback_options = PipelineOptions {
                    index_location: IndexLocation::InMemory,
                    ..options
                };
                let fallback_paths = StorePaths {
                    root: None,
                    lexical_dir: None,
                    semantic_dir: None,
                    metadata_path: None,
                };
                Self::build_with_store_paths(fallback_options, fallback_paths).unwrap_or_else(
                    |fallback_err| {
                        panic!(
                            "in-memory index store initialization failed after fallback: {}",
                            fallback_err
                        )
                    },
                )
            }
        }
    }

    fn try_new(options: PipelineOptions) -> Result<Self> {
        let store_paths = resolve_store_paths(None, &options)?;
        ensure_store_metadata(&store_paths, &options)?;
        Self::build_with_store_paths(options, store_paths)
    }

    pub fn in_memory(mut options: PipelineOptions) -> Self {
        options.index_location = IndexLocation::InMemory;
        Self::new(options)
    }

    pub fn for_corpus(corpus_root: &Path, mut options: PipelineOptions) -> Result<Self> {
        options.index_location = IndexLocation::UnderCorpusRoot;
        let store_paths = resolve_store_paths(Some(corpus_root), &options)?;
        ensure_store_metadata(&store_paths, &options)?;
        Self::build_with_store_paths(options, store_paths)
    }

    pub fn at_path(index_root: &Path, mut options: PipelineOptions) -> Result<Self> {
        options.index_location = IndexLocation::Explicit(index_root.to_path_buf());
        let store_paths = resolve_store_paths(None, &options)?;
        ensure_store_metadata(&store_paths, &options)?;
        Self::build_with_store_paths(options, store_paths)
    }

    fn build_with_store_paths(options: PipelineOptions, store_paths: StorePaths) -> Result<Self> {
        let lexical_index_dir = store_paths.lexical_dir.clone();
        let (source_docs, records, chunk_lifecycle, snapshot) = load_semantic_state(&store_paths)?;
        let mut semantic_docs = HashMap::new();
        let mut semantic_aggregate = SemanticAggregate::default();
        let mut chunk_latest_by_lineage = HashMap::new();
        let temporal_facts = TemporalFactStore::from_records(records.values(), &chunk_lifecycle);
        for record in records.values() {
            let state = build_semantic_doc_state(record, options.claim_extraction);
            semantic_aggregate.insert_doc_state(&state);
            semantic_docs.insert(record.doc_id.clone(), state);
        }
        for meta in chunk_lifecycle.values().filter(|meta| meta.is_latest) {
            chunk_latest_by_lineage.insert(meta.lineage_key.clone(), meta.chunk_id.clone());
        }
        let mut lexical = LexicalState::new(lexical_index_dir)?;
        for record in records.values() {
            lexical.upsert_record(record)?;
        }
        lexical.commit_reload()?;
        Ok(Self {
            options,
            store_paths,
            source_docs,
            records,
            semantic_docs,
            semantic_aggregate,
            chunk_lifecycle,
            chunk_latest_by_lineage,
            temporal_facts,
            dirty_docs: HashSet::new(),
            tombstones: HashSet::new(),
            lexical,
            snapshot,
            snapshot_revision: 0,
            store_revision: 0,
            background_refresh: None,
            dirty: false,
        })
    }

    pub fn with_documents(options: PipelineOptions, docs: Vec<SourceDocument>) -> Self {
        let mut index = Self::new(options);
        for doc in docs {
            index.upsert(doc);
        }
        index
    }

    pub fn upsert(&mut self, doc: SourceDocument) {
        let doc_id = doc.doc_id.clone();
        self.tombstones.remove(&doc_id);
        self.source_docs.insert(doc_id.clone(), doc);
        self.dirty_docs.insert(doc_id);
        self.store_revision = self.store_revision.saturating_add(1);
        self.dirty = true;
    }

    pub fn remove(&mut self, doc_id: &str) -> Option<SourceDocument> {
        let removed = self.source_docs.remove(doc_id);
        if removed.is_some() {
            self.records.remove(doc_id);
            self.semantic_docs.remove(doc_id);
            self.semantic_aggregate.remove_doc(doc_id);
            self.dirty_docs.remove(doc_id);
            self.tombstones.insert(doc_id.to_string());
            self.store_revision = self.store_revision.saturating_add(1);
            self.dirty = true;
        }
        removed
    }

    pub fn len(&self) -> usize {
        self.source_docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.source_docs.is_empty()
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    #[allow(dead_code)]
    fn store_revision(&self) -> u64 {
        self.store_revision
    }

    #[allow(dead_code)]
    fn snapshot_revision(&self) -> u64 {
        self.snapshot_revision
    }

    pub fn tombstones(&self) -> Vec<&str> {
        let mut tombstones = self
            .tombstones
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        tombstones.sort_unstable();
        tombstones
    }

    pub fn source_documents(&self) -> Vec<&SourceDocument> {
        let mut docs: Vec<&SourceDocument> = self.source_docs.values().collect();
        docs.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
        docs
    }

    pub fn records(&self) -> Vec<&DocRecord> {
        let mut records: Vec<&DocRecord> = self.records.values().collect();
        records.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
        records
    }

    pub fn chunk_lifecycle(&self) -> Vec<&ChunkLifecycleMeta> {
        let mut entries: Vec<&ChunkLifecycleMeta> = self.chunk_lifecycle.values().collect();
        entries.sort_by(|a, b| a.chunk_id.cmp(&b.chunk_id));
        entries
    }

    pub fn document_lifecycle(&self) -> Vec<DocumentLifecycleMeta> {
        let mut entries = Vec::new();
        for record in self.records.values() {
            let mut version = 0u32;
            let mut updated_at_ms = 0u64;
            let mut latest_chunk_ids = Vec::new();
            let mut superseded_chunk_ids = Vec::new();
            for chunk in &record.section_chunks {
                if let Some(meta) = self.chunk_lifecycle.get(&chunk.chunk_id) {
                    version = version.max(meta.version);
                    updated_at_ms = updated_at_ms.max(meta.updated_at_ms);
                    if meta.is_latest {
                        latest_chunk_ids.push(meta.chunk_id.clone());
                    } else {
                        superseded_chunk_ids.push(meta.chunk_id.clone());
                    }
                }
            }
            latest_chunk_ids.sort();
            superseded_chunk_ids.sort();
            let chunk_count = record.section_chunks.len();
            entries.push(DocumentLifecycleMeta {
                doc_id: record.doc_id.clone(),
                version: version.max(1),
                is_latest: chunk_count > 0 && superseded_chunk_ids.is_empty(),
                updated_at_ms,
                chunk_count,
                latest_chunk_ids,
                superseded_chunk_ids,
            });
        }
        entries.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
        entries
    }

    pub fn temporal_facts(&self) -> Vec<&crate::temporal_fact::TemporalFact> {
        self.temporal_facts.facts().iter().collect::<Vec<_>>()
    }

    pub fn temporal_facts_as_of(&self, date: &str) -> Vec<&crate::temporal_fact::TemporalFact> {
        self.temporal_facts.as_of(date)
    }

    pub fn temporal_timeline(&self, subject: &str) -> Vec<&crate::temporal_fact::TemporalFact> {
        self.temporal_facts.timeline(subject)
    }

    pub fn temporal_timeline_window_around(
        &self,
        anchor: &str,
        before: usize,
        after: usize,
    ) -> Vec<crate::temporal_fact::TimelineEvent<'_>> {
        self.temporal_facts
            .timeline_window_around(anchor, before, after)
    }

    pub fn temporal_events_between(
        &self,
        start: &str,
        end: &str,
    ) -> Vec<crate::temporal_fact::TimelineEvent<'_>> {
        self.temporal_facts.timeline_events_between(start, end)
    }

    pub fn temporal_adjacent_pairs_between(
        &self,
        start: &str,
        end: &str,
        max_gap_days: Option<i64>,
    ) -> Vec<crate::temporal_fact::TimelinePair<'_>> {
        self.temporal_facts
            .adjacent_pairs_between(start, end, max_gap_days)
    }

    pub fn refresh(&mut self) -> Result<&MemoryIndex> {
        self.poll_background_refresh()?;
        if self.dirty || self.snapshot.is_none() {
            self.prepare_pending_changes()?;
            let mut records = self.records.values().cloned().collect::<Vec<DocRecord>>();
            records.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
            let snapshot = MemoryIndex::from_records_with_semantic_aggregate(
                records,
                self.semantic_aggregate.clone(),
                self.options.text_rerank_ngram,
                self.options.text_rerank_lcs,
                self.options.claim_extraction,
            );
            self.snapshot = Some(snapshot);
            self.snapshot_revision = self.store_revision;
            persist_store_metadata(&self.store_paths, &self.options)?;
            persist_semantic_state(
                &self.store_paths,
                self.snapshot.as_ref().expect("snapshot should exist"),
                &self.records,
                &self.chunk_lifecycle,
            )?;
            self.dirty = false;
        }
        Ok(self
            .snapshot
            .as_ref()
            .expect("snapshot should exist after refresh"))
    }

    #[allow(dead_code)]
    fn refresh_async(&mut self) -> Result<()> {
        self.poll_background_refresh()?;
        if self.background_refresh.is_some() {
            return Ok(());
        }
        if !self.dirty && self.snapshot.is_some() {
            return Ok(());
        }
        self.prepare_pending_changes()?;
        let target_revision = self.store_revision;
        let mut records = self.records.values().cloned().collect::<Vec<DocRecord>>();
        records.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
        let (sender, receiver) = mpsc::channel();
        let semantic_aggregate = self.semantic_aggregate.clone();
        let text_rerank_ngram = self.options.text_rerank_ngram;
        let text_rerank_lcs = self.options.text_rerank_lcs;
        thread::spawn(move || {
            let _ = sender.send(Ok(MemoryIndex::from_records_with_semantic_aggregate(
                records,
                semantic_aggregate,
                text_rerank_ngram,
                text_rerank_lcs,
                false,
            )));
        });
        self.background_refresh = Some(BackgroundRefresh {
            target_revision,
            receiver,
        });
        Ok(())
    }

    #[allow(dead_code)]
    fn query_latest(&mut self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        self.poll_background_refresh()?;
        if self.snapshot.is_none() {
            self.refresh()?;
        } else if self.dirty {
            self.refresh_async()?;
        }
        let lexical_hits = self
            .lexical
            .search(query, top_k.saturating_mul(5).max(20))?;
        Ok(self
            .snapshot
            .as_ref()
            .expect("snapshot should exist after latest query preparation")
            .query_with_lexical_hits(query, top_k, Some(&lexical_hits)))
    }

    fn query_fresh(&mut self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        self.refresh()?;
        let lexical_hits = self
            .lexical
            .search(query, top_k.saturating_mul(5).max(20))?;
        Ok(self
            .snapshot
            .as_ref()
            .expect("snapshot should exist after refresh")
            .query_with_lexical_hits(query, top_k, Some(&lexical_hits)))
    }

    pub fn query(&mut self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        self.query_fresh(query, top_k)
    }

    pub fn query_timed(
        &mut self,
        query: &str,
        top_k: usize,
    ) -> Result<(Vec<SearchResult>, QueryTimings, QueryDiagnostics)> {
        let refresh_start = std::time::Instant::now();
        self.refresh()?;
        let refresh_ms = refresh_start.elapsed().as_secs_f64() * 1000.0;
        let (results, mut timings, diagnostics) = self
            .snapshot
            .as_ref()
            .expect("snapshot should exist after refresh")
            .query_timed(query, top_k);
        timings.refresh_ms = refresh_ms;
        timings.total_ms += refresh_ms;
        Ok((results, timings, diagnostics))
    }

    pub fn query_filtered(
        &mut self,
        query: &str,
        top_k: usize,
        filters: &std::collections::BTreeMap<String, String>,
    ) -> Result<Vec<SearchResult>> {
        self.refresh()?;
        let lexical_hits = self.lexical.search(query, top_k.saturating_mul(5).max(20))?;
        Ok(self
            .snapshot
            .as_ref()
            .expect("snapshot should exist after refresh")
            .query_with_filters_and_lexical(query, top_k, filters, Some(&lexical_hits)))
    }

    /// Multi-term variant: builds the allowed-doc set once from `filters`, then scores every
    /// query. BM25 (tantivy) is still called once per term — that can't be collapsed — but the
    /// filter scan over all docs happens only once regardless of how many queries are given.
    /// Returns one `Vec<SearchResult>` per input query, in the same order.
    pub fn query_filtered_multi(
        &mut self,
        queries: &[&str],
        top_k: usize,
        filters: &std::collections::BTreeMap<String, String>,
    ) -> Result<Vec<Vec<SearchResult>>> {
        self.refresh()?;
        let candidate_k = top_k.saturating_mul(5).max(20);
        let mut lexical_hits_per_query = Vec::with_capacity(queries.len());
        for q in queries {
            lexical_hits_per_query.push(self.lexical.search(q, candidate_k)?);
        }
        Ok(self
            .snapshot
            .as_ref()
            .expect("snapshot should exist after refresh")
            .query_with_filters_multi(queries, top_k, filters, &lexical_hits_per_query))
    }

    fn prepare_pending_changes(&mut self) -> Result<()> {
        let dirty_doc_ids = self.dirty_docs.iter().cloned().collect::<Vec<String>>();
        for doc_id in &dirty_doc_ids {
            let source_doc = self
                .source_docs
                .get(doc_id)
                .expect("dirty doc should still exist in source docs");
            let record = build_doc_record(source_doc, &self.options)?;
            self.semantic_aggregate.remove_doc(doc_id);
            let semantic_state = build_semantic_doc_state(&record, self.options.claim_extraction);
            self.semantic_aggregate.insert_doc_state(&semantic_state);
            self.semantic_docs.insert(doc_id.clone(), semantic_state);
            let previous = self.records.get(doc_id).cloned();
            self.records.insert(doc_id.clone(), record);
            if let Some(current) = self.records.get(doc_id).cloned() {
                self.update_chunk_lifecycle_for_doc(doc_id, previous.as_ref(), &current);
            }
            self.dirty_docs.remove(doc_id);
        }
        let tombstoned = self.tombstones.iter().cloned().collect::<Vec<_>>();
        for doc_id in tombstoned {
            self.remove_chunk_lifecycle_for_doc(&doc_id);
        }
        for doc_id in &self.tombstones {
            self.lexical.remove_doc(doc_id)?;
        }
        self.temporal_facts =
            TemporalFactStore::from_records(self.records.values(), &self.chunk_lifecycle);
        let lexical_upserts = if self.snapshot.is_none() && self.snapshot_revision == 0 {
            self.records
                .iter()
                .filter(|(doc_id, _)| self.source_docs.contains_key(*doc_id))
                .filter(|(doc_id, _)| !self.tombstones.contains(*doc_id))
                .map(|(_, record)| record)
                .collect::<Vec<_>>()
        } else {
            dirty_doc_ids
                .iter()
                .filter_map(|doc_id| self.records.get(doc_id))
                .collect::<Vec<_>>()
        };
        for record in lexical_upserts {
            self.lexical.upsert_record(record)?;
        }
        self.lexical.commit_reload()?;
        Ok(())
    }

    fn poll_background_refresh(&mut self) -> Result<()> {
        let Some(background) = self.background_refresh.as_ref() else {
            return Ok(());
        };
        match background.receiver.try_recv() {
            Ok(result) => {
                let target_revision = background.target_revision;
                self.background_refresh = None;
                let snapshot = result?;
                if target_revision == self.store_revision {
                    self.snapshot = Some(snapshot);
                    self.snapshot_revision = target_revision;
                    persist_store_metadata(&self.store_paths, &self.options)?;
                    persist_semantic_state(
                        &self.store_paths,
                        self.snapshot.as_ref().expect("snapshot should exist"),
                        &self.records,
                        &self.chunk_lifecycle,
                    )?;
                    self.dirty = false;
                }
                Ok(())
            }
            Err(TryRecvError::Empty) => Ok(()),
            Err(TryRecvError::Disconnected) => {
                self.background_refresh = None;
                anyhow::bail!("background semantic refresh disconnected")
            }
        }
    }

    fn update_chunk_lifecycle_for_doc(
        &mut self,
        doc_id: &str,
        previous_record: Option<&DocRecord>,
        current_record: &DocRecord,
    ) {
        let now = current_time_ms();
        let mut current_lineage_keys = HashSet::new();
        for chunk in &current_record.section_chunks {
            let lineage_key = chunk_lineage_key(doc_id, chunk);
            current_lineage_keys.insert(lineage_key.clone());

            if let Some(existing) = self.chunk_lifecycle.get_mut(&chunk.chunk_id) {
                existing.doc_id = doc_id.to_string();
                existing.lineage_key = lineage_key.clone();
                existing.is_latest = true;
                existing.updated_at_ms = now;
                self.chunk_latest_by_lineage
                    .insert(lineage_key.clone(), chunk.chunk_id.clone());
                continue;
            }

            let previous_latest_chunk_id = self.chunk_latest_by_lineage.get(&lineage_key).cloned();
            let (version, supersedes_chunk_id) =
                if let Some(prev_chunk_id) = previous_latest_chunk_id {
                    if let Some(prev_meta) = self.chunk_lifecycle.get_mut(&prev_chunk_id) {
                        prev_meta.is_latest = false;
                        prev_meta.updated_at_ms = now;
                        (
                            prev_meta.version.saturating_add(1),
                            Some(prev_meta.chunk_id.clone()),
                        )
                    } else {
                        (1, None)
                    }
                } else {
                    (1, None)
                };

            self.chunk_lifecycle.insert(
                chunk.chunk_id.clone(),
                ChunkLifecycleMeta {
                    chunk_id: chunk.chunk_id.clone(),
                    doc_id: doc_id.to_string(),
                    lineage_key: lineage_key.clone(),
                    version,
                    is_latest: true,
                    supersedes_chunk_id,
                    updated_at_ms: now,
                    change_reason: Some("upsert".to_string()),
                },
            );
            self.chunk_latest_by_lineage
                .insert(lineage_key, chunk.chunk_id.clone());
        }

        if let Some(previous_record) = previous_record {
            for chunk in &previous_record.section_chunks {
                let lineage_key = chunk_lineage_key(doc_id, chunk);
                if !current_lineage_keys.contains(&lineage_key) {
                    if let Some(meta) = self.chunk_lifecycle.get_mut(&chunk.chunk_id) {
                        meta.is_latest = false;
                        meta.updated_at_ms = now;
                    }
                    self.chunk_latest_by_lineage.remove(&lineage_key);
                }
            }
        }
    }

    fn remove_chunk_lifecycle_for_doc(&mut self, doc_id: &str) {
        let chunk_ids = self
            .chunk_lifecycle
            .values()
            .filter(|meta| meta.doc_id == doc_id)
            .map(|meta| meta.chunk_id.clone())
            .collect::<Vec<_>>();
        for chunk_id in chunk_ids {
            if let Some(meta) = self.chunk_lifecycle.remove(&chunk_id) {
                if self
                    .chunk_latest_by_lineage
                    .get(&meta.lineage_key)
                    .map(|id| id == &meta.chunk_id)
                    .unwrap_or(false)
                {
                    self.chunk_latest_by_lineage.remove(&meta.lineage_key);
                }
            }
        }
    }
}

fn index_location_name(index_location: &IndexLocation) -> String {
    match index_location {
        IndexLocation::InMemory => "in_memory".to_string(),
        IndexLocation::UnderCorpusRoot => "under_corpus_root".to_string(),
        IndexLocation::Explicit(_) => "explicit".to_string(),
    }
}

fn current_store_metadata(options: &PipelineOptions) -> StoreMetadata {
    StoreMetadata {
        schema_version: STORE_SCHEMA_VERSION,
        layout_version: STORE_LAYOUT_VERSION.to_string(),
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
        index_location: index_location_name(&options.index_location),
    }
}

fn load_store_metadata(metadata_path: &Path) -> Result<Option<StoreMetadata>> {
    if !metadata_path.exists() {
        return Ok(None);
    }
    let contents = fs::read_to_string(metadata_path)?;
    Ok(Some(serde_json::from_str(&contents)?))
}

fn persist_store_metadata(store_paths: &StorePaths, options: &PipelineOptions) -> Result<()> {
    let Some(metadata_path) = store_paths.metadata_path.as_ref() else {
        return Ok(());
    };
    let metadata = current_store_metadata(options);
    write_text_file_atomic(metadata_path, &serde_json::to_string_pretty(&metadata)?)?;
    Ok(())
}

fn ensure_store_metadata(store_paths: &StorePaths, options: &PipelineOptions) -> Result<()> {
    let Some(metadata_path) = store_paths.metadata_path.as_ref() else {
        return Ok(());
    };
    if let Some(root) = store_paths.root.as_ref() {
        fs::create_dir_all(root)?;
    }
    if let Some(existing) = load_store_metadata(metadata_path)? {
        let current = current_store_metadata(options);
        if existing.schema_version != current.schema_version {
            anyhow::bail!(
                "index schema mismatch at {}: found {}, expected {}",
                metadata_path.display(),
                existing.schema_version,
                current.schema_version
            );
        }
        if existing.layout_version != current.layout_version {
            anyhow::bail!(
                "index layout mismatch at {}: found {}, expected {}",
                metadata_path.display(),
                existing.layout_version,
                current.layout_version
            );
        }
        return Ok(());
    }
    persist_store_metadata(store_paths, options)
}

fn is_directory_empty(dir: &Path) -> Result<bool> {
    let mut entries = fs::read_dir(dir)?;
    Ok(entries.next().is_none())
}

fn semantic_records_path(store_paths: &StorePaths) -> Option<PathBuf> {
    store_paths
        .semantic_dir
        .as_ref()
        .map(|dir| dir.join(SEMANTIC_RECORDS_FILE))
}

fn semantic_core_path(store_paths: &StorePaths) -> Option<PathBuf> {
    store_paths
        .semantic_dir
        .as_ref()
        .map(|dir| dir.join(SEMANTIC_CORE_FILE))
}

fn chunk_lifecycle_path(store_paths: &StorePaths) -> Option<PathBuf> {
    store_paths
        .semantic_dir
        .as_ref()
        .map(|dir| dir.join(CHUNK_LIFECYCLE_FILE))
}

fn chunk_lineage_key(doc_id: &str, chunk: &crate::index::SectionChunk) -> String {
    format!(
        "{}::{}::{}::{}",
        doc_id,
        chunk.start_line,
        chunk.end_line,
        chunk.heading.trim().to_lowercase()
    )
}

fn current_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn source_document_from_record(record: &DocRecord) -> SourceDocument {
    SourceDocument {
        doc_id: record.doc_id.clone(),
        source: record.source.clone(),
        content: record.content.clone(),
        concept: record
            .headings
            .first()
            .cloned()
            .or_else(|| record.probable_topic.clone())
            .unwrap_or_else(|| record.doc_id.clone()),
        group_id: record.group_id.clone(),
        filters: record.filters.clone(),
        headings: record.headings.clone(),
        links: record.doc_links.clone(),
        timestamp: record.timestamp.clone(),
        doc_length: record.doc_length,
        author_agent: record.author_agent.clone(),
    }
}

fn load_semantic_state(
    store_paths: &StorePaths,
) -> Result<(
    HashMap<String, SourceDocument>,
    HashMap<String, DocRecord>,
    HashMap<String, ChunkLifecycleMeta>,
    Option<MemoryIndex>,
)> {
    let Some(records_path) = semantic_records_path(store_paths) else {
        return Ok((HashMap::new(), HashMap::new(), HashMap::new(), None));
    };
    let Some(core_path) = semantic_core_path(store_paths) else {
        return Ok((HashMap::new(), HashMap::new(), HashMap::new(), None));
    };
    if !records_path.exists() || !core_path.exists() {
        return Ok((HashMap::new(), HashMap::new(), HashMap::new(), None));
    }

    let persisted: PersistedSemanticRecords =
        serde_json::from_str(&fs::read_to_string(&records_path)?)?;
    if persisted.schema_version != STORE_SCHEMA_VERSION {
        anyhow::bail!(
            "semantic record schema mismatch at {}: found {}, expected {}",
            records_path.display(),
            persisted.schema_version,
            STORE_SCHEMA_VERSION
        );
    }
    if persisted.layout_version != STORE_LAYOUT_VERSION {
        anyhow::bail!(
            "semantic record layout mismatch at {}: found {}, expected {}",
            records_path.display(),
            persisted.layout_version,
            STORE_LAYOUT_VERSION
        );
    }

    let restored_records = persisted
        .records
        .into_iter()
        .map(Into::into)
        .collect::<Vec<DocRecord>>();
    let snapshot =
        MemoryIndex::load_with_binary_core(restored_records.clone(), &core_path, None, false)?;
    let mut source_docs = HashMap::new();
    let mut records = HashMap::new();
    for record in restored_records {
        source_docs.insert(record.doc_id.clone(), source_document_from_record(&record));
        records.insert(record.doc_id.clone(), record);
    }
    let mut chunk_lifecycle = persisted
        .chunk_lifecycle
        .into_iter()
        .map(|meta| (meta.chunk_id.clone(), meta))
        .collect::<HashMap<_, _>>();
    if chunk_lifecycle.is_empty() {
        for record in records.values() {
            for chunk in &record.section_chunks {
                let chunk_id = chunk.chunk_id.clone();
                chunk_lifecycle.insert(
                    chunk_id.clone(),
                    ChunkLifecycleMeta {
                        chunk_id,
                        doc_id: record.doc_id.clone(),
                        lineage_key: chunk_lineage_key(&record.doc_id, chunk),
                        version: 1,
                        is_latest: true,
                        supersedes_chunk_id: None,
                        updated_at_ms: current_time_ms(),
                        change_reason: Some("bootstrap".to_string()),
                    },
                );
            }
        }
    } else if let Some(chunk_lifecycle_path) = chunk_lifecycle_path(store_paths) {
        if !chunk_lifecycle_path.exists() {
            write_text_file_atomic(
                &chunk_lifecycle_path,
                &serde_json::to_string_pretty(
                    &chunk_lifecycle.values().cloned().collect::<Vec<_>>(),
                )?,
            )?;
        }
    }
    Ok((source_docs, records, chunk_lifecycle, Some(snapshot)))
}

fn persist_semantic_state(
    store_paths: &StorePaths,
    snapshot: &MemoryIndex,
    records_map: &HashMap<String, DocRecord>,
    chunk_lifecycle_map: &HashMap<String, ChunkLifecycleMeta>,
) -> Result<()> {
    let Some(semantic_dir) = store_paths.semantic_dir.as_ref() else {
        return Ok(());
    };
    fs::create_dir_all(semantic_dir)?;
    let records_path = semantic_records_path(store_paths)
        .expect("records path should exist when semantic dir exists");
    let core_path =
        semantic_core_path(store_paths).expect("core path should exist when semantic dir exists");
    let mut records = records_map.values().cloned().collect::<Vec<_>>();
    records.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
    let payload = PersistedSemanticRecords {
        schema_version: STORE_SCHEMA_VERSION,
        layout_version: STORE_LAYOUT_VERSION.to_string(),
        records: records.into_iter().map(PersistedDocRecord::from).collect(),
        chunk_lifecycle: chunk_lifecycle_map
            .values()
            .cloned()
            .collect::<Vec<ChunkLifecycleMeta>>(),
    };
    write_text_file_atomic(&records_path, &serde_json::to_string_pretty(&payload)?)?;
    if let Some(lifecycle_path) = chunk_lifecycle_path(store_paths) {
        let mut lifecycle = chunk_lifecycle_map
            .values()
            .cloned()
            .collect::<Vec<ChunkLifecycleMeta>>();
        lifecycle.sort_by(|a, b| a.chunk_id.cmp(&b.chunk_id));
        write_text_file_atomic(&lifecycle_path, &serde_json::to_string_pretty(&lifecycle)?)?;
    }
    save_binary_core_atomic(snapshot, &core_path)?;
    Ok(())
}

fn ensure_safe_output_path(path: &Path) -> Result<()> {
    if path.is_dir() {
        anyhow::bail!("refusing to write: output path is a directory");
    }
    if let Ok(meta) = fs::symlink_metadata(path) {
        if meta.file_type().is_symlink() {
            anyhow::bail!("refusing to write: output path is a symlink");
        }
    }
    if let Some(parent) = path.parent() {
        let mut cur = if parent.is_absolute() {
            PathBuf::from("/")
        } else {
            std::env::current_dir()?
        };
        for comp in parent.components() {
            use std::path::Component;
            match comp {
                Component::RootDir | Component::CurDir => continue,
                Component::ParentDir => {
                    anyhow::bail!("refusing to write: parent traversal is not allowed")
                }
                Component::Normal(seg) => {
                    cur.push(seg);
                    if let Ok(meta) = fs::symlink_metadata(&cur) {
                        if meta.file_type().is_symlink() {
                            anyhow::bail!(
                                "refusing to write: parent path component is a symlink ({})",
                                cur.display()
                            );
                        }
                    }
                }
                Component::Prefix(_) => {}
            }
        }
    }
    Ok(())
}

fn atomic_temp_path(path: &Path) -> Result<PathBuf> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("output");
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    for attempt in 0..1000u32 {
        let candidate = parent.join(format!(".{}.{}.{}.tmp", stem, pid, nanos + attempt as u128));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    anyhow::bail!(
        "unable to allocate temporary file path for {}",
        path.display()
    )
}

fn write_text_file_atomic(path: &Path, content: &str) -> Result<()> {
    ensure_safe_output_path(path)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temp_path = atomic_temp_path(path)?;
    {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp_path)?;
        file.write_all(content.as_bytes())?;
        file.sync_all().ok();
    }
    fs::rename(&temp_path, path)?;
    Ok(())
}

fn save_binary_core_atomic(snapshot: &MemoryIndex, path: &Path) -> Result<()> {
    ensure_safe_output_path(path)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temp_path = atomic_temp_path(path)?;
    snapshot.save_binary_core(&temp_path)?;
    fs::rename(&temp_path, path)?;
    Ok(())
}

struct LexicalState {
    index: Index,
    writer: IndexWriter,
    reader: IndexReader,
    doc_id_f: Field,
    content_f: Field,
    headings_f: Field,
    terms_f: Field,
    entities_f: Field,
}

impl LexicalState {
    fn new(index_dir: Option<PathBuf>) -> Result<Self> {
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("doc_id", STRING | STORED);
        schema_builder.add_text_field("content", TEXT);
        schema_builder.add_text_field("headings", TEXT);
        schema_builder.add_text_field("important_terms", TEXT);
        schema_builder.add_text_field("entities", TEXT);
        let schema = schema_builder.build();

        let index = match index_dir.as_deref() {
            Some(dir) => Self::open_or_create_on_disk(dir, &schema)?,
            None => Index::create_in_ram(schema),
        };
        let writer = index.writer(50_000_000)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        let schema_ref = index.schema();
        let doc_id_f = schema_ref.get_field("doc_id")?;
        let content_f = schema_ref.get_field("content")?;
        let headings_f = schema_ref.get_field("headings")?;
        let terms_f = schema_ref.get_field("important_terms")?;
        let entities_f = schema_ref.get_field("entities")?;
        Ok(Self {
            index,
            writer,
            reader,
            doc_id_f,
            content_f,
            headings_f,
            terms_f,
            entities_f,
        })
    }

    fn open_or_create_on_disk(dir: &Path, schema: &Schema) -> Result<Index> {
        fs::create_dir_all(dir)?;
        if is_directory_empty(dir)? {
            return Ok(Index::create_in_dir(dir, schema.clone())?);
        }
        match Index::open_in_dir(dir) {
            Ok(index) => Ok(index),
            Err(err) => Err(anyhow::anyhow!(
                "failed to open tantivy index at {}: {}",
                dir.display(),
                err
            )),
        }
    }

    fn upsert_record(&mut self, record: &DocRecord) -> Result<()> {
        let headings_text = record
            .section_chunks
            .iter()
            .map(|c| c.heading.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let terms_text = record
            .section_chunks
            .iter()
            .flat_map(|c| c.important_terms.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join(" ");
        let entities_text = record
            .section_chunks
            .iter()
            .flat_map(|c| c.key_entities.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join(" ");
        let content_text = record
            .section_chunks
            .iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        self.writer
            .delete_term(Term::from_field_text(self.doc_id_f, &record.doc_id));
        self.writer.add_document(doc!(
            self.doc_id_f => record.doc_id.clone(),
            self.content_f => content_text,
            self.headings_f => headings_text,
            self.terms_f => terms_text,
            self.entities_f => entities_text
        ))?;
        Ok(())
    }

    fn remove_doc(&mut self, doc_id: &str) -> Result<()> {
        self.writer
            .delete_term(Term::from_field_text(self.doc_id_f, doc_id));
        Ok(())
    }

    fn commit_reload(&mut self) -> Result<()> {
        self.writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }

    fn search(&self, query: &str, top_k: usize) -> Result<HashMap<String, f32>> {
        let searcher = self.reader.searcher();
        let mut query_parser = QueryParser::for_index(
            &self.index,
            vec![
                self.content_f,
                self.headings_f,
                self.terms_f,
                self.entities_f,
            ],
        );
        query_parser.set_field_boost(self.content_f, 1.0);
        query_parser.set_field_boost(self.headings_f, 1.4);
        query_parser.set_field_boost(self.terms_f, 2.0);
        query_parser.set_field_boost(self.entities_f, 2.4);
        let parsed = query_parser.parse_query(query)?;
        let top_docs = searcher.search(&parsed, &TopDocs::with_limit(top_k))?;
        let mut out = HashMap::new();
        for (score, addr) in top_docs {
            let retrieved: TantivyDocument = searcher.doc(addr)?;
            if let Some(v) = retrieved.get_first(self.doc_id_f) {
                if let Some(doc_id) = v.as_str() {
                    out.insert(doc_id.to_string(), score);
                }
            }
        }
        Ok(out)
    }
}

fn select_term_ranker(ranker_kind: &Tier1TermRankerKind) -> Box<dyn ImportantTermRanker> {
    match ranker_kind {
        Tier1TermRankerKind::Yake => Box::new(YakeStyleTermRanker),
        Tier1TermRankerKind::Rake => Box::new(RakeStyleTermRanker),
        Tier1TermRankerKind::Cvalue => Box::new(CValueStyleTermRanker),
        Tier1TermRankerKind::Textrank => Box::new(TextRankStyleTermRanker),
    }
}

fn guess_doc_type(headings: &[String], content: &str) -> Option<String> {
    let joined_headings = headings.join(" ").to_lowercase();
    let content_l = content.to_lowercase();
    if joined_headings.contains("incident") || content_l.contains("postmortem") {
        Some("incident".to_string())
    } else if joined_headings.contains("runbook") || content_l.contains("playbook") {
        Some("runbook".to_string())
    } else if joined_headings.contains("changelog") || content_l.contains("release notes") {
        Some("changelog".to_string())
    } else if joined_headings.contains("reference") {
        Some("reference".to_string())
    } else if joined_headings.contains("tutorial") || joined_headings.contains("quick start") {
        Some("tutorial".to_string())
    } else if joined_headings.contains("decision") || content_l.contains("adr") {
        Some("decision".to_string())
    } else {
        None
    }
}

pub fn source_documents_to_tier1_inputs(docs: &[SourceDocument]) -> Vec<Tier1DocInput> {
    docs.iter()
        .map(|doc| Tier1DocInput {
            id: doc.doc_id.clone(),
            source: doc.source.clone(),
            content: doc.content.clone(),
            concept: doc.concept.clone(),
            headings: doc.headings.clone(),
        })
        .collect()
}

fn build_doc_records(
    source_docs: &[SourceDocument],
    options: &PipelineOptions,
) -> Result<Vec<DocRecord>> {
    let docs = source_documents_to_tier1_inputs(source_docs);

    let heuristic = HeuristicKeyEntityRanker;
    let entities_by_doc = match &options.ner_provider {
        Tier1NerProvider::Heuristic => heuristic.rank_docs(&docs)?,
        Tier1NerProvider::Spacy => {
            let spacy = SpacyKeyEntityRanker {
                model: options.spacy_model.clone(),
                script_path: default_spacy_script_path().display().to_string(),
            };
            match spacy.rank_docs(&docs) {
                Ok(out) => out,
                Err(err) => {
                    eprintln!(
                        "warning: {} ranker unavailable ({}), falling back to heuristic",
                        spacy.name(),
                        err
                    );
                    heuristic.rank_docs(&docs).unwrap_or_default()
                }
            }
        }
    };

    let term_ranker = select_term_ranker(&options.term_ranker);
    let term_ranker_name = term_ranker.name().to_string();
    let ner_provider_name = match &options.ner_provider {
        Tier1NerProvider::Heuristic => "heuristic".to_string(),
        Tier1NerProvider::Spacy => format!("spacy:{}", options.spacy_model),
    };

    let source_by_id: HashMap<&str, &SourceDocument> = source_docs
        .iter()
        .map(|doc| (doc.doc_id.as_str(), doc))
        .collect();

    let mut records = Vec::new();
    for doc in docs {
        let source_doc = source_by_id
            .get(doc.id.as_str())
            .copied()
            .expect("source document should exist for tier1 input");
        let key_entities = entities_by_doc.get(&doc.id).cloned().unwrap_or_default();
        let important_terms = term_ranker.rank_terms(&doc);
        records.push(assemble_doc_record(
            source_doc,
            &doc,
            key_entities,
            important_terms,
            &ner_provider_name,
            &term_ranker_name,
            options,
        ));
    }

    Ok(records)
}

fn build_doc_record(source_doc: &SourceDocument, options: &PipelineOptions) -> Result<DocRecord> {
    let docs = source_documents_to_tier1_inputs(std::slice::from_ref(source_doc));
    let doc = docs
        .into_iter()
        .next()
        .expect("single source document should yield one tier1 input");

    let heuristic = HeuristicKeyEntityRanker;
    let key_entities = match &options.ner_provider {
        Tier1NerProvider::Heuristic => heuristic
            .rank_docs(std::slice::from_ref(&doc))?
            .remove(&doc.id)
            .unwrap_or_default(),
        Tier1NerProvider::Spacy => {
            let spacy = SpacyKeyEntityRanker {
                model: options.spacy_model.clone(),
                script_path: default_spacy_script_path().display().to_string(),
            };
            match spacy.rank_docs(std::slice::from_ref(&doc)) {
                Ok(mut out) => out.remove(&doc.id).unwrap_or_default(),
                Err(err) => {
                    eprintln!(
                        "warning: {} ranker unavailable ({}), falling back to heuristic",
                        spacy.name(),
                        err
                    );
                    heuristic
                        .rank_docs(std::slice::from_ref(&doc))
                        .unwrap_or_default()
                        .remove(&doc.id)
                        .unwrap_or_default()
                }
            }
        }
    };

    let term_ranker = select_term_ranker(&options.term_ranker);
    let important_terms = term_ranker.rank_terms(&doc);
    let ner_provider_name = match &options.ner_provider {
        Tier1NerProvider::Heuristic => "heuristic".to_string(),
        Tier1NerProvider::Spacy => format!("spacy:{}", options.spacy_model),
    };
    let term_ranker_name = term_ranker.name().to_string();

    Ok(assemble_doc_record(
        source_doc,
        &doc,
        key_entities,
        important_terms,
        &ner_provider_name,
        &term_ranker_name,
        options,
    ))
}

fn assemble_doc_record(
    source_doc: &SourceDocument,
    doc: &Tier1DocInput,
    key_entities: Vec<crate::tier1::Tier1Entity>,
    important_terms: Vec<crate::tier1::RankedTerm>,
    ner_provider_name: &str,
    term_ranker_name: &str,
    options: &PipelineOptions,
) -> DocRecord {
    let probable_topic = if let Some(first_heading) = doc.headings.first() {
        Some(first_heading.clone())
    } else {
        important_terms.first().map(|t| t.term.clone())
    };
    let base_chunks = match &options.chunk_strategy {
        ChunkStrategy::Heading => chunk_document_sections(&doc.content, &doc.id),
        ChunkStrategy::Line => chunk_document_lines(
            &doc.content,
            &doc.id,
            options.chunk_lines,
            options.chunk_overlap,
        ),
        ChunkStrategy::Hybrid => chunk_document_hybrid(
            &doc.content,
            &doc.id,
            options.chunk_lines,
            options.chunk_overlap,
            options.chunk_target_tokens,
            options.chunk_max_tokens,
        ),
    };
    let mut section_chunks = enrich_section_chunks(base_chunks, &key_entities, &important_terms);
    for chunk in &mut section_chunks {
        if chunk.timestamp.is_none() {
            chunk.timestamp = source_doc.timestamp.clone();
        }
    }
    let temporal_terms =
        extract_temporal_terms(source_doc.timestamp.as_deref(), &doc.content, &doc.headings);
    let mut record = DocRecord {
        doc_id: doc.id.clone(),
        source: doc.source.clone(),
        content: doc.content.clone(),
        timestamp: source_doc.timestamp.clone(),
        doc_length: source_doc.doc_length,
        author_agent: source_doc.author_agent.clone(),
        group_id: source_doc.group_id.clone(),
        filters: source_doc.filters.clone(),
        probable_topic,
        doc_type_guess: guess_doc_type(&doc.headings, &doc.content),
        headings: doc.headings.clone(),
        doc_links: source_doc.links.clone(),
        temporal_terms,
        key_entities,
        important_terms,
        section_chunks,
        embedding: None,
        top_claims: Vec::new(),
        provenance: Provenance {
            source: doc.source.clone(),
            timestamp: source_doc.timestamp.clone(),
            ner_provider: ner_provider_name.to_string(),
            term_ranker: term_ranker_name.to_string(),
            index_version: "v1-memory-hybrid".to_string(),
        },
    };

    if options.claim_extraction {
        let extractor = ConservativeClaimExtractor;
        record.top_claims = extractor.extract(&record).claims;
    }
    record
}

pub fn build_query_snapshot_from_records(
    records: &[DocRecord],
    options: &PipelineOptions,
) -> Result<MemoryIndex> {
    let store_paths = resolve_store_paths(None, options)?;
    Ok(MemoryIndex::from_records_with_lexical_dir(
        records.to_vec(),
        store_paths.lexical_dir.as_deref(),
        options.text_rerank_ngram,
        options.text_rerank_lcs,
        options.claim_extraction,
    ))
}

pub fn build_query_snapshot(
    source_docs: &[SourceDocument],
    options: &PipelineOptions,
) -> Result<MemoryIndex> {
    let records = build_doc_records(source_docs, options)?;
    build_query_snapshot_from_records(&records, options)
}

pub fn build_query_snapshot_from_source_documents(
    source_docs: &[SourceDocument],
    provider: &Tier1NerProvider,
    spacy_model: &str,
    ranker_kind: &Tier1TermRankerKind,
    chunk_strategy: &ChunkStrategy,
    chunk_lines: usize,
    chunk_overlap: usize,
    chunk_target_tokens: usize,
    chunk_max_tokens: usize,
    text_rerank_ngram: bool,
    text_rerank_lcs: bool,
) -> Result<MemoryIndex> {
    let options = PipelineOptions {
        ner_provider: provider.clone(),
        spacy_model: spacy_model.to_string(),
        term_ranker: ranker_kind.clone(),
        chunk_strategy: chunk_strategy.clone(),
        chunk_lines,
        chunk_overlap,
        chunk_target_tokens,
        chunk_max_tokens,
        text_rerank_ngram,
        text_rerank_lcs,
        claim_extraction: false,
        index_location: IndexLocation::InMemory,
    };
    build_query_snapshot(source_docs, &options)
}

pub fn build_index_store(
    source_docs: &[SourceDocument],
    options: &PipelineOptions,
) -> Result<IndexStore> {
    let mut index = IndexStore::with_documents(options.clone(), source_docs.to_vec());
    index.refresh()?;
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn sample_doc(id: &str, content: &str) -> SourceDocument {
        SourceDocument {
            doc_id: id.to_string(),
            source: format!("artifact://{}", id),
            content: content.to_string(),
            concept: id.to_string(),
            group_id: None,
            headings: vec!["Overview".to_string()],
            links: vec![],
            timestamp: None,
            doc_length: content.len(),
            author_agent: None,
        }
    }

    fn sample_doc_with_group(id: &str, group_id: &str, content: &str) -> SourceDocument {
        SourceDocument {
            group_id: Some(group_id.to_string()),
            ..sample_doc(id, content)
        }
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("lint-ai-{prefix}-{nanos}"))
    }

    #[test]
    fn build_memory_index_works_with_defaults() {
        let docs = vec![sample_doc("doc-1", "docker install on linux")];
        let index =
            build_query_snapshot(&docs, &PipelineOptions::default()).expect("index should build");
        let results = index.query("docker", 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "doc-1");
    }

    #[test]
    fn chunk_ids_are_stable_for_same_input() {
        let content = "# Intro\nDocker install on linux\n# Usage\nRun docker info";
        let first = crate::chunking::chunk_document_sections(content, "doc-1");
        let second = crate::chunking::chunk_document_sections(content, "doc-1");
        assert_eq!(first.len(), second.len());
        let first_ids = first
            .into_iter()
            .map(|chunk| chunk.chunk_id)
            .collect::<Vec<_>>();
        let second_ids = second
            .into_iter()
            .map(|chunk| chunk.chunk_id)
            .collect::<Vec<_>>();
        assert_eq!(first_ids, second_ids);
    }

    #[test]
    fn artifact_index_upsert_remove_and_query() {
        let mut artifact_index = IndexStore::new(PipelineOptions::default());
        artifact_index.upsert(sample_doc("doc-1", "docker install guide"));
        let results = artifact_index
            .query("docker", 5)
            .expect("query should succeed");
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "doc-1");
        assert!(!artifact_index.is_dirty());

        artifact_index.upsert(sample_doc("doc-2", "kubernetes setup guide"));
        assert!(artifact_index.is_dirty());
        let results = artifact_index
            .query("kubernetes", 5)
            .expect("query should succeed");
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "doc-2");

        let removed = artifact_index.remove("doc-2");
        assert!(removed.is_some());
        let results = artifact_index
            .query("kubernetes", 5)
            .expect("query should succeed");
        assert!(results.is_empty() || results[0].doc_id != "doc-2");
        assert_eq!(artifact_index.tombstones(), vec!["doc-2"]);
    }

    #[test]
    fn query_results_diversify_by_group_id() {
        let docs = vec![
            sample_doc_with_group("doc-a1", "group-a", "shared ranking token"),
            sample_doc_with_group("doc-a2", "group-a", "shared ranking token"),
            sample_doc_with_group("doc-a3", "group-a", "shared ranking token"),
            sample_doc_with_group("doc-a4", "group-a", "shared ranking token"),
            sample_doc_with_group("doc-b1", "group-b", "shared ranking token"),
            sample_doc_with_group("doc-b2", "group-b", "shared ranking token"),
            sample_doc_with_group("doc-b3", "group-b", "shared ranking token"),
            sample_doc_with_group("doc-b4", "group-b", "shared ranking token"),
        ];
        let index =
            build_query_snapshot(&docs, &PipelineOptions::default()).expect("index should build");
        let results = index.query("shared", 6);
        assert!(!results.is_empty());
        let mut counts: HashMap<String, usize> = HashMap::new();
        for result in results {
            let group = result.group_id.expect("group id should be preserved");
            *counts.entry(group).or_default() += 1;
        }
        assert!(counts.values().all(|count| *count <= 3));
        assert!(counts.len() >= 2);
    }

    #[test]
    fn index_store_query_uses_incremental_lexical_updates() {
        let mut index = IndexStore::new(PipelineOptions::default());
        index.upsert(sample_doc("doc-1", "docker install guide"));
        let docker_results = index.query("docker", 5).expect("query should succeed");
        assert!(!docker_results.is_empty());
        assert_eq!(docker_results[0].doc_id, "doc-1");

        index.upsert(sample_doc("doc-2", "kubernetes cluster operations"));
        let kube_results = index.query("kubernetes", 5).expect("query should succeed");
        assert!(!kube_results.is_empty());
        assert_eq!(kube_results[0].doc_id, "doc-2");

        let docker_results = index.query("docker", 5).expect("query should succeed");
        assert!(!docker_results.is_empty());
        assert_eq!(docker_results[0].doc_id, "doc-1");
    }

    #[test]
    fn refresh_is_idempotent_when_no_documents_change() {
        let mut index = IndexStore::new(PipelineOptions::default());
        index.upsert(sample_doc("doc-1", "refresh idempotence check"));

        index.refresh().expect("initial refresh should succeed");
        assert!(!index.is_dirty());
        let snapshot_revision = index.snapshot_revision();
        let store_revision = index.store_revision();

        index.refresh().expect("second refresh should succeed");
        assert_eq!(index.snapshot_revision(), snapshot_revision);
        assert_eq!(index.store_revision(), store_revision);
        assert!(!index.is_dirty());
    }

    #[test]
    fn index_store_remove_deletes_lexical_doc() {
        let mut index = IndexStore::new(PipelineOptions::default());
        index.upsert(sample_doc("doc-1", "redis cache operations"));
        let before_remove = index.query("redis", 5).expect("query should succeed");
        assert!(!before_remove.is_empty());
        assert_eq!(before_remove[0].doc_id, "doc-1");

        index.remove("doc-1");
        let after_remove = index.query("redis", 5).expect("query should succeed");
        assert!(after_remove.iter().all(|result| result.doc_id != "doc-1"));
    }

    #[test]
    fn index_store_with_lexical_dir_persists_queries_across_instances() {
        let index_root = unique_temp_dir("lexical-root");
        let options = PipelineOptions {
            index_location: IndexLocation::Explicit(index_root.clone()),
            ..PipelineOptions::default()
        };

        let mut first = IndexStore::at_path(&index_root, options.clone())
            .expect("explicit-path store should initialize");
        first.upsert(sample_doc("doc-1", "persistent lexical search"));
        let first_results = first.query("persistent", 5).expect("query should succeed");
        assert!(!first_results.is_empty());
        assert_eq!(first_results[0].doc_id, "doc-1");
        drop(first);

        let mut second = IndexStore::at_path(&index_root, options)
            .expect("explicit-path store should initialize");
        second.upsert(sample_doc("doc-1", "persistent lexical search"));
        let second_results = second.query("persistent", 5).expect("query should succeed");
        assert!(!second_results.is_empty());
        assert_eq!(second_results[0].doc_id, "doc-1");

        let _ = fs::remove_dir_all(index_root);
    }

    #[test]
    fn resolve_store_paths_under_corpus_root() {
        let corpus_root = unique_temp_dir("corpus-root");
        let options = PipelineOptions {
            index_location: IndexLocation::UnderCorpusRoot,
            ..PipelineOptions::default()
        };
        let paths =
            resolve_store_paths(Some(&corpus_root), &options).expect("paths should resolve");
        assert_eq!(paths.root, Some(corpus_root.join(".lint-ai")));
        assert_eq!(
            paths.lexical_dir,
            Some(corpus_root.join(".lint-ai").join("lexical"))
        );
        assert_eq!(
            paths.semantic_dir,
            Some(corpus_root.join(".lint-ai").join("semantic"))
        );
        assert_eq!(
            paths.metadata_path,
            Some(corpus_root.join(".lint-ai").join("metadata.json"))
        );
    }

    #[test]
    fn index_store_for_corpus_uses_corpus_local_lexical_dir() {
        let corpus_root = unique_temp_dir("corpus-store");
        let mut index = IndexStore::for_corpus(&corpus_root, PipelineOptions::default())
            .expect("corpus-backed store should initialize");
        index.upsert(sample_doc("doc-1", "corpus rooted lexical index"));
        let results = index.query("lexical", 5).expect("query should succeed");
        assert!(!results.is_empty());
        assert!(corpus_root.join(".lint-ai").join("lexical").exists());
        let _ = fs::remove_dir_all(corpus_root.join(".lint-ai"));
    }

    #[test]
    fn index_store_for_corpus_writes_metadata() {
        let corpus_root = unique_temp_dir("corpus-metadata");
        let mut index = IndexStore::for_corpus(&corpus_root, PipelineOptions::default())
            .expect("corpus-backed store should initialize");
        index.upsert(sample_doc("doc-1", "metadata persistence test"));
        index.refresh().expect("refresh should succeed");

        let metadata_path = corpus_root.join(".lint-ai").join("metadata.json");
        assert!(metadata_path.exists());
        let metadata = fs::read_to_string(&metadata_path).expect("metadata should be readable");
        assert!(metadata.contains("\"schema_version\": 1"));
        assert!(metadata.contains("\"layout_version\": \"index-store-v1\""));

        let _ = fs::remove_dir_all(corpus_root.join(".lint-ai"));
    }

    #[test]
    fn index_store_persists_and_reloads_semantic_state() {
        let corpus_root = unique_temp_dir("semantic-state");
        let mut first = IndexStore::for_corpus(&corpus_root, PipelineOptions::default())
            .expect("corpus-backed store should initialize");
        first.upsert(sample_doc("doc-1", "semantic persistence works"));
        let first_results = first.query("persistence", 5).expect("query should succeed");
        assert!(!first_results.is_empty());
        assert_eq!(first_results[0].doc_id, "doc-1");
        drop(first);

        let semantic_dir = corpus_root.join(".lint-ai").join("semantic");
        assert!(semantic_dir.join("records.json").exists());
        assert!(semantic_dir.join("core.bin").exists());

        let mut second = IndexStore::for_corpus(&corpus_root, PipelineOptions::default())
            .expect("corpus-backed store should initialize");
        let second_results = second
            .query("persistence", 5)
            .expect("query should succeed");
        assert!(!second_results.is_empty());
        assert_eq!(second_results[0].doc_id, "doc-1");

        let _ = fs::remove_dir_all(corpus_root.join(".lint-ai"));
    }

    #[test]
    fn index_store_rejects_metadata_schema_mismatch() {
        let index_root = unique_temp_dir("schema-mismatch");
        fs::create_dir_all(&index_root).expect("index root should be creatable");
        fs::write(
            index_root.join("metadata.json"),
            r#"{"schema_version":999,"layout_version":"index-store-v1","crate_version":"0.1.6","index_location":"explicit"}"#,
        )
        .expect("metadata file should be writable");

        let result = IndexStore::at_path(&index_root, PipelineOptions::default());
        assert!(result.is_err());

        let _ = fs::remove_dir_all(index_root);
    }

    #[test]
    fn index_store_does_not_delete_invalid_lexical_directory() {
        let index_root = unique_temp_dir("invalid-lexical-dir");
        let lexical_dir = index_root.join("lexical");
        fs::create_dir_all(&lexical_dir).expect("lexical dir should be creatable");
        let sentinel = lexical_dir.join("keep.txt");
        fs::write(&sentinel, "do not delete").expect("sentinel file should be writable");

        let result = IndexStore::at_path(&index_root, PipelineOptions::default());
        assert!(result.is_err());
        assert!(sentinel.exists());

        let _ = fs::remove_dir_all(index_root);
    }

    #[test]
    fn index_store_new_falls_back_to_in_memory_on_invalid_explicit_root() {
        let index_root = unique_temp_dir("new-fallback");
        fs::write(&index_root, "not a directory").expect("invalid root file should be writable");

        let options = PipelineOptions {
            index_location: IndexLocation::Explicit(index_root.clone()),
            ..PipelineOptions::default()
        };
        let mut index = IndexStore::new(options);
        index.upsert(sample_doc("doc-1", "fallback in memory works"));

        let results = index.query("fallback", 5).expect("query should succeed");
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "doc-1");
        assert!(!index_root.join("lexical").exists());
        assert!(index.store_paths.root.is_none());
        assert!(index.store_paths.lexical_dir.is_none());
        assert!(index.store_paths.semantic_dir.is_none());
        assert!(index.store_paths.metadata_path.is_none());

        let _ = fs::remove_file(index_root);
    }

    #[test]
    fn index_store_new_falls_back_to_in_memory_when_corpus_root_is_missing() {
        let options = PipelineOptions {
            index_location: IndexLocation::UnderCorpusRoot,
            ..PipelineOptions::default()
        };

        let mut index = IndexStore::new(options);
        index.upsert(sample_doc("doc-1", "missing corpus root fallback"));

        let results = index.query("corpus", 5).expect("query should succeed");
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "doc-1");
        assert!(index.store_paths.root.is_none());
        assert!(index.store_paths.lexical_dir.is_none());
        assert!(index.store_paths.semantic_dir.is_none());
        assert!(index.store_paths.metadata_path.is_none());
    }

    #[test]
    fn chunk_lifecycle_increments_version_and_tracks_latest() {
        let mut index = IndexStore::new(PipelineOptions::default());
        index.upsert(sample_doc("doc-1", "first chunk body"));
        index.refresh().expect("refresh should succeed");

        let first_record = index
            .records()
            .into_iter()
            .find(|record| record.doc_id == "doc-1")
            .expect("doc-1 record should exist");
        assert_eq!(first_record.section_chunks.len(), 1);
        let first_chunk_id = first_record.section_chunks[0].chunk_id.clone();

        let first_meta = index
            .chunk_lifecycle()
            .into_iter()
            .find(|meta| meta.chunk_id == first_chunk_id)
            .expect("first chunk lifecycle should exist");
        assert_eq!(first_meta.version, 1);
        assert!(first_meta.is_latest);
        assert!(first_meta.supersedes_chunk_id.is_none());

        index.upsert(sample_doc("doc-1", "second chunk body"));
        index.refresh().expect("refresh should succeed");

        let second_record = index
            .records()
            .into_iter()
            .find(|record| record.doc_id == "doc-1")
            .expect("doc-1 record should exist");
        assert_eq!(second_record.section_chunks.len(), 1);
        let second_chunk_id = second_record.section_chunks[0].chunk_id.clone();
        assert_ne!(first_chunk_id, second_chunk_id);

        let metas = index
            .chunk_lifecycle()
            .into_iter()
            .filter(|meta| meta.doc_id == "doc-1")
            .collect::<Vec<_>>();
        assert_eq!(metas.len(), 2);

        let latest = metas
            .iter()
            .find(|meta| meta.chunk_id == second_chunk_id)
            .expect("new chunk metadata should exist");
        assert!(latest.is_latest);
        assert_eq!(latest.version, 2);
        assert_eq!(
            latest.supersedes_chunk_id.as_deref(),
            Some(first_chunk_id.as_str())
        );

        let previous = metas
            .iter()
            .find(|meta| meta.chunk_id == first_chunk_id)
            .expect("previous chunk metadata should exist");
        assert!(!previous.is_latest);
        assert_eq!(previous.version, 1);
    }

    #[test]
    fn section_chunks_inherit_parent_timestamp_and_document_lifecycle_is_derived() {
        let mut doc = sample_doc("doc-1", "timestamp inheritance body");
        doc.timestamp = Some("2024-05-10T12:34:56Z".to_string());

        let mut index = IndexStore::new(PipelineOptions::default());
        index.upsert(doc);
        index.refresh().expect("refresh should succeed");

        let record = index
            .records()
            .into_iter()
            .find(|record| record.doc_id == "doc-1")
            .expect("doc-1 record should exist");
        assert_eq!(record.section_chunks.len(), 1);
        assert_eq!(
            record.section_chunks[0].timestamp.as_deref(),
            Some("2024-05-10T12:34:56Z")
        );

        let doc_lifecycle = index
            .document_lifecycle()
            .into_iter()
            .find(|meta| meta.doc_id == "doc-1")
            .expect("doc lifecycle should exist");
        assert_eq!(doc_lifecycle.chunk_count, 1);
        assert_eq!(doc_lifecycle.latest_chunk_ids.len(), 1);
        assert!(doc_lifecycle.is_latest);
        assert!(doc_lifecycle.updated_at_ms > 0);
    }

    #[test]
    fn chunk_lifecycle_is_removed_when_doc_is_removed() {
        let mut index = IndexStore::new(PipelineOptions::default());
        index.upsert(sample_doc("doc-1", "redis lifecycle cleanup"));
        index.refresh().expect("refresh should succeed");
        assert!(index
            .chunk_lifecycle()
            .into_iter()
            .any(|meta| meta.doc_id == "doc-1"));

        index.remove("doc-1");
        index.refresh().expect("refresh should succeed");
        assert!(!index
            .chunk_lifecycle()
            .into_iter()
            .any(|meta| meta.doc_id == "doc-1"));
    }
}
