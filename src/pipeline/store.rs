use super::{
    build_doc_record, chunk_lineage_key, current_time_ms, doc_record_content_hash,
    ensure_store_metadata, execute_prepared_on_snapshot_parts, inspect_memory_index_snapshot,
    load_segment_manifest, load_semantic_state, persist_segment_manifest, persist_semantic_state,
    persist_store_metadata, source_document_from_record, IndexLocation, IndexStoreInspection,
    LexicalState, MemoryIndexLayout, MemoryIndexSnapshot, PipelineOptions,
};
use crate::index::{
    build_semantic_doc_state, DocRecord, MemoryIndex, Provenance, QueryDiagnostics, QueryTimings,
    SearchResult, SemanticAggregate, SemanticDocState,
};
use crate::query_plan::PreparedQuery;
use crate::segments::{MemoryIndexSegment, SegmentedMemoryIndex};
use crate::semantic_relations::SemanticRelationStore;
use crate::source::SourceDocument;
use crate::temporal_fact::TemporalFactStore;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread;
use tantivy::doc;
#[derive(Debug, Clone)]
pub struct StorePaths {
    pub root: Option<PathBuf>,
    pub lexical_dir: Option<PathBuf>,
    pub semantic_dir: Option<PathBuf>,
    pub metadata_path: Option<PathBuf>,
}

pub(crate) const STORE_SCHEMA_VERSION: u32 = 1;
pub(crate) const STORE_LAYOUT_VERSION: &str = "index-store-v1";
pub(crate) const SEMANTIC_RECORDS_FILE: &str = "records.json";
pub(crate) const SEMANTIC_CORE_FILE: &str = "core.bin";
pub(crate) const SEGMENT_MANIFEST_FILE: &str = "segments.json";
pub(crate) const CHUNK_LIFECYCLE_FILE: &str = "chunk_lifecycle.json";

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
pub(crate) struct DocumentLifecycleMeta {
    pub doc_id: String,
    pub version: u32,
    pub is_latest: bool,
    pub updated_at_ms: u64,
    pub chunk_count: usize,
    pub latest_chunk_ids: Vec<String>,
    pub superseded_chunk_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct StoreMetadata {
    pub(crate) schema_version: u32,
    pub(crate) layout_version: String,
    pub(crate) crate_version: String,
    pub(crate) index_location: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PersistedSemanticRecords {
    pub(crate) schema_version: u32,
    pub(crate) layout_version: String,
    pub(crate) records: Vec<PersistedDocRecord>,
    #[serde(default)]
    pub(crate) chunk_lifecycle: Vec<ChunkLifecycleMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PersistedDocRecord {
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
    #[serde(default)]
    content_hash: String,
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
            content_hash: record.content_hash,
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
            content_hash: record.content_hash,
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

/// Opaque byte buffers produced by `IndexStore::dump()` and consumed by
/// `IndexStore::load_from_dump()`. Intended for storage in an external system
/// (e.g. Postgres `BYTEA` / `JSONB` columns). Callers should treat the contents
/// as opaque. The schema version embedded in `records_json` guards against
/// version mismatches on restore.
#[derive(Debug, Clone)]
pub struct IndexDump {
    /// JSON-serialized `PersistedSemanticRecords` (doc records + chunk lifecycle).
    pub records_json: Vec<u8>,
    /// Bincode-serialized `PersistedMemoryCore` (BM25 structures, posting lists).
    pub core_bytes: Vec<u8>,
}

pub struct IndexStore {
    options: PipelineOptions,
    pub(crate) store_paths: StorePaths,
    source_docs: HashMap<String, SourceDocument>,
    records: HashMap<String, DocRecord>,
    semantic_docs: HashMap<String, SemanticDocState>,
    semantic_aggregate: SemanticAggregate,
    chunk_lifecycle: HashMap<String, ChunkLifecycleMeta>,
    chunk_latest_by_lineage: HashMap<String, String>,
    temporal_facts: TemporalFactStore,
    semantic_relations: SemanticRelationStore,
    dirty_docs: HashSet<String>,
    tombstones: HashSet<String>,
    lexical: LexicalState,
    snapshot: Option<Arc<MemoryIndexSnapshot>>,
    snapshot_revision: u64,
    store_revision: u64,
    background_refresh: Option<BackgroundRefresh>,
    dirty: bool,
}

/// Immutable, cheaply clonable view of the last fully published index generation.
/// It intentionally excludes all mutable writer and persistence state.
#[derive(Clone)]
pub struct PublishedIndexSnapshot {
    options: PipelineOptions,
    source_docs: Arc<HashMap<String, SourceDocument>>,
    records: Arc<HashMap<String, DocRecord>>,
    semantic_relations: Arc<SemanticRelationStore>,
    snapshot: Option<Arc<MemoryIndexSnapshot>>,
}

impl PublishedIndexSnapshot {
    pub fn query_prepared(
        &self,
        prepared: &PreparedQuery,
        top_k: usize,
        filters: &std::collections::BTreeMap<String, String>,
    ) -> Result<Vec<SearchResult>> {
        execute_prepared_on_snapshot_parts(
            &self.options,
            &self.source_docs,
            &self.records,
            &self.semantic_relations,
            self.snapshot.as_deref(),
            prepared,
            top_k,
            filters,
        )
        .map(|(results, _, _)| results)
    }

    pub fn source_document_by_id(&self, doc_id: &str) -> Option<&SourceDocument> {
        self.source_docs.get(doc_id)
    }
}

struct BackgroundRefresh {
    target_revision: u64,
    receiver: Mutex<Receiver<Result<MemoryIndexSnapshot>>>,
}

fn build_memory_index_snapshot(
    records: Vec<DocRecord>,
    global_index: Option<MemoryIndex>,
    layout: &MemoryIndexLayout,
    generation: u64,
) -> MemoryIndexSnapshot {
    match layout {
        MemoryIndexLayout::Single => MemoryIndexSnapshot::Single(
            global_index.expect("single-index snapshots require a global index"),
        ),
        MemoryIndexLayout::Segmented { .. } | MemoryIndexLayout::AdaptiveSegmented { .. } => {
            MemoryIndexSnapshot::Segmented(
                SegmentedMemoryIndex::from_records_by_group_id_with_generation(
                    &records, generation,
                ),
            )
        }
    }
}

fn merge_composed_map<T>(
    target: &mut HashMap<String, T>,
    additions: HashMap<String, T>,
    label: &str,
) -> Result<()> {
    for (id, value) in additions {
        if target.insert(id.clone(), value).is_some() {
            anyhow::bail!("cannot compose stores with duplicate {label} id: {id}");
        }
    }
    Ok(())
}

fn take_segmented_snapshot(
    snapshot: Option<Arc<MemoryIndexSnapshot>>,
    namespace: &str,
) -> Result<Vec<MemoryIndexSegment>> {
    let Some(snapshot) = snapshot else {
        return Ok(Vec::new());
    };
    let snapshot = Arc::try_unwrap(snapshot).map_err(|_| {
        anyhow::anyhow!("cannot compose an index snapshot while it is still shared")
    })?;
    let MemoryIndexSnapshot::Segmented(mut segmented) = snapshot else {
        anyhow::bail!("composed stores require segmented snapshots");
    };
    for segment in &mut segmented.segments {
        segment.segment_id = format!("{namespace}:{}", segment.segment_id);
    }
    Ok(segmented.segments)
}

impl IndexStore {
    pub fn published_snapshot(&self) -> PublishedIndexSnapshot {
        PublishedIndexSnapshot {
            options: self.options.clone(),
            source_docs: Arc::new(self.source_docs.clone()),
            records: Arc::new(self.records.clone()),
            semantic_relations: Arc::new(self.semantic_relations.clone()),
            snapshot: self.snapshot.clone(),
        }
    }
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

    /// Compose already-published segmented stores without rebuilding document
    /// records. This is used by provider MCP servers: the persistent workspace
    /// and a provider's private memory retain their own storage, while the
    /// request process gets one in-memory query view with corpus-wide BM25
    /// statistics.
    pub fn compose_segmented(workspace: Self, provider_memory: Option<Self>) -> Result<Self> {
        if !matches!(
            workspace.options.memory_index_layout,
            MemoryIndexLayout::Segmented { .. } | MemoryIndexLayout::AdaptiveSegmented { .. }
        ) {
            anyhow::bail!("composed stores require a segmented memory index layout");
        }

        let IndexStore {
            mut source_docs,
            mut records,
            mut chunk_lifecycle,
            snapshot: workspace_snapshot,
            options,
            ..
        } = workspace;
        let mut segments = take_segmented_snapshot(workspace_snapshot, "workspace")?;

        if let Some(provider_memory) = provider_memory {
            if !matches!(
                provider_memory.options.memory_index_layout,
                MemoryIndexLayout::Segmented { .. } | MemoryIndexLayout::AdaptiveSegmented { .. }
            ) {
                anyhow::bail!("provider memory must use a segmented memory index layout");
            }
            let IndexStore {
                source_docs: provider_docs,
                records: provider_records,
                chunk_lifecycle: provider_lifecycle,
                snapshot: provider_snapshot,
                ..
            } = provider_memory;
            merge_composed_map(&mut source_docs, provider_docs, "source document")?;
            merge_composed_map(&mut records, provider_records, "record")?;
            merge_composed_map(&mut chunk_lifecycle, provider_lifecycle, "chunk lifecycle")?;
            segments.extend(take_segmented_snapshot(provider_snapshot, "provider")?);
        }

        let mut semantic_docs = HashMap::new();
        let mut semantic_aggregate = SemanticAggregate::default();
        let mut chunk_latest_by_lineage = HashMap::new();
        for record in records.values() {
            let state = build_semantic_doc_state(record, options.claim_extraction);
            semantic_aggregate.insert_doc_state(&state);
            semantic_docs.insert(record.doc_id.clone(), state);
        }
        for meta in chunk_lifecycle.values().filter(|meta| meta.is_latest) {
            chunk_latest_by_lineage.insert(meta.lineage_key.clone(), meta.chunk_id.clone());
        }
        let temporal_facts = TemporalFactStore::from_records(records.values(), &chunk_lifecycle);
        let semantic_relations =
            SemanticRelationStore::try_from_documents(source_docs.values(), options.supersession)?;
        let mut lexical = LexicalState::new(None)?;
        for record in records.values() {
            lexical.upsert_record(record)?;
        }
        lexical.commit_reload()?;

        let snapshot = (!segments.is_empty())
            .then(|| SegmentedMemoryIndex::from_segments_with_generation(segments, 1))
            .transpose()
            .map_err(anyhow::Error::msg)?
            .map(MemoryIndexSnapshot::Segmented)
            .map(Arc::new);
        Ok(Self {
            options: PipelineOptions {
                index_location: IndexLocation::InMemory,
                ..options
            },
            store_paths: StorePaths {
                root: None,
                lexical_dir: None,
                semantic_dir: None,
                metadata_path: None,
            },
            source_docs,
            records,
            semantic_docs,
            semantic_aggregate,
            chunk_lifecycle,
            chunk_latest_by_lineage,
            temporal_facts,
            semantic_relations,
            dirty_docs: HashSet::new(),
            tombstones: HashSet::new(),
            lexical,
            snapshot_revision: usize::from(snapshot.is_some()) as u64,
            store_revision: usize::from(snapshot.is_some()) as u64,
            snapshot,
            background_refresh: None,
            dirty: false,
        })
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
        let (source_docs, records, chunk_lifecycle, loaded_snapshot) =
            load_semantic_state(&store_paths, &options.memory_index_layout)?;
        let snapshot =
            if let Some(global) = loaded_snapshot {
                let records = records.values().cloned().collect::<Vec<_>>();
                Some(build_memory_index_snapshot(
                    records,
                    Some(global),
                    &options.memory_index_layout,
                    0,
                ))
            } else if matches!(
                options.memory_index_layout,
                MemoryIndexLayout::Segmented { .. } | MemoryIndexLayout::AdaptiveSegmented { .. }
            ) && !records.is_empty()
            {
                let records = records.values().cloned().collect::<Vec<_>>();
                let manifest = load_segment_manifest(&store_paths)?;
                let snapshot = manifest.and_then(|manifest| {
                    match SegmentedMemoryIndex::from_records_by_manifest(&records, &manifest, 0) {
                        Ok(segmented) => Some(MemoryIndexSnapshot::Segmented(segmented)),
                        Err(error) => {
                            eprintln!(
                                "warning: ignoring invalid persisted segment manifest: {error}"
                            );
                            None
                        }
                    }
                });
                Some(snapshot.unwrap_or_else(|| {
                    build_memory_index_snapshot(records, None, &options.memory_index_layout, 0)
                }))
            } else {
                None
            };
        let mut semantic_docs = HashMap::new();
        let mut semantic_aggregate = SemanticAggregate::default();
        let mut chunk_latest_by_lineage = HashMap::new();
        let temporal_facts = TemporalFactStore::from_records(records.values(), &chunk_lifecycle);
        let semantic_relations =
            SemanticRelationStore::try_from_documents(source_docs.values(), options.supersession)?;
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
            semantic_relations,
            dirty_docs: HashSet::new(),
            tombstones: HashSet::new(),
            lexical,
            snapshot: snapshot.map(Arc::new),
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

    fn build_compatibility_index(&self) -> MemoryIndex {
        let mut records = self.records.values().cloned().collect::<Vec<_>>();
        records.sort_by(|left, right| left.doc_id.cmp(&right.doc_id));
        MemoryIndex::from_records_with_semantic_aggregate(
            records,
            self.semantic_aggregate.clone(),
            self.options.text_rerank_ngram,
            self.options.text_rerank_lcs,
            self.options.claim_extraction,
        )
    }

    fn persist_compatibility_state(&self) -> Result<()> {
        if self.store_paths.semantic_dir.is_none() {
            return Ok(());
        }
        match self.snapshot.as_deref().expect("snapshot should exist") {
            MemoryIndexSnapshot::Single(index) => persist_semantic_state(
                &self.store_paths,
                Some(index),
                &self.records,
                &self.chunk_lifecycle,
            ),
            MemoryIndexSnapshot::Segmented(segmented) => {
                persist_semantic_state(
                    &self.store_paths,
                    None,
                    &self.records,
                    &self.chunk_lifecycle,
                )?;
                persist_segment_manifest(&self.store_paths, &segmented.manifest())
            }
        }
    }

    /// Serialize the current index state to a pair of opaque byte buffers suitable
    /// for storage in an external system (e.g. Postgres BYTEA / JSONB columns).
    /// Call `load_from_dump` to restore.
    pub(crate) fn dump(&mut self) -> Result<IndexDump> {
        self.refresh()?;
        let compatibility_index = self.build_compatibility_index();
        let core_bytes = compatibility_index.to_bytes()?;
        let records: Vec<PersistedDocRecord> = self
            .records
            .values()
            .cloned()
            .map(PersistedDocRecord::from)
            .collect();
        let chunk_lifecycle: Vec<ChunkLifecycleMeta> =
            self.chunk_lifecycle.values().cloned().collect();
        let persisted = PersistedSemanticRecords {
            schema_version: STORE_SCHEMA_VERSION,
            layout_version: STORE_LAYOUT_VERSION.to_string(),
            records,
            chunk_lifecycle,
        };
        let records_json = serde_json::to_vec(&persisted)?;
        Ok(IndexDump {
            records_json,
            core_bytes,
        })
    }

    /// Restore an `IndexStore` from a dump produced by `dump()`.
    /// Returns an error if the schema version does not match.
    pub(crate) fn load_from_dump(dump: IndexDump, options: PipelineOptions) -> Result<Self> {
        let persisted: PersistedSemanticRecords = serde_json::from_slice(&dump.records_json)?;
        if persisted.schema_version != STORE_SCHEMA_VERSION {
            anyhow::bail!(
                "index dump schema mismatch: found {}, expected {}",
                persisted.schema_version,
                STORE_SCHEMA_VERSION
            );
        }
        if persisted.layout_version != STORE_LAYOUT_VERSION {
            anyhow::bail!(
                "index dump layout mismatch: found {}, expected {}",
                persisted.layout_version,
                STORE_LAYOUT_VERSION
            );
        }
        let restored_records: Vec<DocRecord> =
            persisted.records.into_iter().map(Into::into).collect();
        let global_index = MemoryIndex::from_bytes(&dump.core_bytes, restored_records.clone())?;
        let snapshot = build_memory_index_snapshot(
            restored_records.clone(),
            Some(global_index),
            &options.memory_index_layout,
            0,
        );
        let mut source_docs = HashMap::new();
        let mut records = HashMap::new();
        let mut chunk_lifecycle: HashMap<String, ChunkLifecycleMeta> = persisted
            .chunk_lifecycle
            .into_iter()
            .map(|m| (m.chunk_id.clone(), m))
            .collect();
        for record in &restored_records {
            source_docs.insert(record.doc_id.clone(), source_document_from_record(record));
            records.insert(record.doc_id.clone(), record.clone());
        }
        if chunk_lifecycle.is_empty() {
            for record in records.values() {
                for chunk in &record.section_chunks {
                    chunk_lifecycle.insert(
                        chunk.chunk_id.clone(),
                        ChunkLifecycleMeta {
                            chunk_id: chunk.chunk_id.clone(),
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
        }
        let chunk_latest_by_lineage = chunk_lifecycle
            .values()
            .filter(|m| m.is_latest)
            .map(|m| (m.lineage_key.clone(), m.chunk_id.clone()))
            .collect();
        let temporal_facts = TemporalFactStore::from_records(records.values(), &chunk_lifecycle);
        let semantic_relations =
            SemanticRelationStore::try_from_documents(source_docs.values(), options.supersession)?;
        let mut semantic_docs = HashMap::new();
        let mut semantic_aggregate = SemanticAggregate::default();
        for record in records.values() {
            let state = build_semantic_doc_state(record, options.claim_extraction);
            semantic_aggregate.insert_doc_state(&state);
            semantic_docs.insert(record.doc_id.clone(), state);
        }
        let mut lexical = LexicalState::new(None)?;
        for record in records.values() {
            lexical.upsert_record(record)?;
        }
        lexical.commit_reload()?;
        let store_paths = StorePaths {
            root: None,
            lexical_dir: None,
            semantic_dir: None,
            metadata_path: None,
        };
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
            semantic_relations,
            dirty_docs: HashSet::new(),
            tombstones: HashSet::new(),
            lexical,
            snapshot: Some(Arc::new(snapshot)),
            snapshot_revision: 1,
            store_revision: 1,
            background_refresh: None,
            dirty: false,
        })
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
    pub(crate) fn store_revision(&self) -> u64 {
        self.store_revision
    }

    #[allow(dead_code)]
    pub(crate) fn snapshot_revision(&self) -> u64 {
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

    pub fn source_document_by_id(&self, doc_id: &str) -> Option<&SourceDocument> {
        self.source_docs.get(doc_id)
    }

    pub fn records(&self) -> Vec<&DocRecord> {
        let mut records: Vec<&DocRecord> = self.records.values().collect();
        records.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
        records
    }

    pub fn record_by_id(&self, doc_id: &str) -> Option<&DocRecord> {
        self.records.get(doc_id)
    }

    /// Test-only: wipes a stored record's content hash to simulate a legacy
    /// persisted record (written before hashing existed).
    #[cfg(test)]
    pub(crate) fn clear_record_content_hash_for_test(&mut self, doc_id: &str) {
        if let Some(record) = self.records.get_mut(doc_id) {
            record.content_hash.clear();
        }
    }

    pub fn memory_index_snapshot(&self) -> Option<&MemoryIndexSnapshot> {
        self.snapshot.as_deref()
    }

    pub fn inspection(&self) -> IndexStoreInspection {
        IndexStoreInspection {
            source_document_count: self.source_docs.len(),
            record_count: self.records.len(),
            dirty: self.dirty,
            store_revision: self.store_revision,
            snapshot_revision: self.snapshot_revision,
            tombstones: self.tombstones().into_iter().map(str::to_string).collect(),
            snapshot: self.snapshot.as_deref().map(inspect_memory_index_snapshot),
        }
    }

    pub fn chunk_lifecycle(&self) -> Vec<&ChunkLifecycleMeta> {
        let mut entries: Vec<&ChunkLifecycleMeta> = self.chunk_lifecycle.values().collect();
        entries.sort_by(|a, b| a.chunk_id.cmp(&b.chunk_id));
        entries
    }

    pub(crate) fn document_lifecycle(&self) -> Vec<DocumentLifecycleMeta> {
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

    pub fn semantic_claims(&self) -> &[crate::semantic_relations::SemanticClaim] {
        self.semantic_relations.claims()
    }

    pub fn semantic_relations(&self) -> &[crate::semantic_relations::SemanticRelation] {
        self.semantic_relations.relations()
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

    pub fn refresh(&mut self) -> Result<()> {
        self.poll_background_refresh()?;
        if self.dirty || self.snapshot.is_none() {
            // Only documents whose content actually changed get fresh
            // records (see prepare_pending_changes), so only segments
            // containing one of the returned ids must be rebuilt.
            let reprocessed_doc_ids = self.prepare_pending_changes()?;
            let mut records = self.records.values().cloned().collect::<Vec<DocRecord>>();
            records.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
            if !self.try_refresh_incremental(&records, &reprocessed_doc_ids)? {
                let global_index =
                    matches!(self.options.memory_index_layout, MemoryIndexLayout::Single)
                        .then(|| self.build_compatibility_index());
                self.snapshot = Some(Arc::new(build_memory_index_snapshot(
                    records,
                    global_index,
                    &self.options.memory_index_layout,
                    self.store_revision,
                )));
            }
            self.snapshot_revision = self.store_revision;
            persist_store_metadata(&self.store_paths, &self.options)?;
            self.persist_compatibility_state()?;
            self.dirty = false;
        }
        Ok(())
    }

    /// Rebuild only the segments touched by staged changes, sharing the rest
    /// with the previous snapshot. Returns `Ok(true)` when the incremental
    /// path ran; `Ok(false)` means the caller must do a full rebuild (first
    /// build, or a non-segmented layout). Sharing is structural — segments
    /// hold their index behind `Arc` — so there is no ownership dance and no
    /// silent fallback: a concurrent reader simply keeps the old snapshot.
    fn try_refresh_incremental(
        &mut self,
        records: &[DocRecord],
        reprocessed_doc_ids: &HashSet<String>,
    ) -> Result<bool> {
        if !matches!(
            self.options.memory_index_layout,
            MemoryIndexLayout::Segmented { .. } | MemoryIndexLayout::AdaptiveSegmented { .. }
        ) {
            return Ok(false);
        }
        let previous = match self.snapshot.as_deref() {
            Some(MemoryIndexSnapshot::Segmented(previous)) => previous,
            _ => return Ok(false),
        };
        let next = SegmentedMemoryIndex::refresh_incremental(
            previous,
            records,
            reprocessed_doc_ids,
            self.store_revision,
        )
        .map_err(|error| anyhow::anyhow!("incremental segment refresh failed: {error}"))?;
        self.snapshot = Some(Arc::new(MemoryIndexSnapshot::Segmented(next)));
        Ok(true)
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
        let claim_extraction = self.options.claim_extraction;
        let memory_index_layout = self.options.memory_index_layout.clone();
        thread::spawn(move || {
            let global_index =
                matches!(memory_index_layout, MemoryIndexLayout::Single).then(|| {
                    MemoryIndex::from_records_with_semantic_aggregate(
                        records.clone(),
                        semantic_aggregate,
                        text_rerank_ngram,
                        text_rerank_lcs,
                        claim_extraction,
                    )
                });
            let _ = sender.send(Ok(build_memory_index_snapshot(
                records,
                global_index,
                &memory_index_layout,
                target_revision,
            )));
        });
        self.background_refresh = Some(BackgroundRefresh {
            target_revision,
            receiver: Mutex::new(receiver),
        });
        Ok(())
    }

    #[allow(dead_code)]
    // Delegates to deprecated MemoryIndex helpers until they are removed in 0.2.0.
    #[allow(deprecated)]
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
        match self
            .snapshot
            .as_deref()
            .expect("snapshot should exist after latest query preparation")
        {
            MemoryIndexSnapshot::Single(index) => {
                Ok(index.query_with_lexical_hits(query, top_k, Some(&lexical_hits)))
            }
            MemoryIndexSnapshot::Segmented(_) => self.query(query, top_k),
        }
    }

    pub fn query(&mut self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        let prepared = PreparedQuery::new(query);
        self.query_prepared(&prepared, top_k, &std::collections::BTreeMap::new())
    }

    pub fn query_timed(
        &mut self,
        query: &str,
        top_k: usize,
    ) -> Result<(Vec<SearchResult>, QueryTimings, QueryDiagnostics)> {
        let prepared = PreparedQuery::new(query);
        self.query_prepared_timed(&prepared, top_k, &std::collections::BTreeMap::new())
    }

    /// Searches with the canonical query treatment: one [`PreparedQuery`] owns
    /// intent inference, query augmentation, temporal context, and the optional
    /// reference clock. `IndexStore` adds only semantic/filter scoping and
    /// snapshot layout routing.
    pub fn query_prepared(
        &mut self,
        prepared: &PreparedQuery,
        top_k: usize,
        filters: &std::collections::BTreeMap<String, String>,
    ) -> Result<Vec<SearchResult>> {
        self.refresh()?;
        self.execute_prepared_on_snapshot(prepared, top_k, filters)
            .map(|(results, _, _)| results)
    }

    pub fn query_prepared_timed(
        &mut self,
        prepared: &PreparedQuery,
        top_k: usize,
        filters: &std::collections::BTreeMap<String, String>,
    ) -> Result<(Vec<SearchResult>, QueryTimings, QueryDiagnostics)> {
        let refresh_start = std::time::Instant::now();
        self.refresh()?;
        let refresh_ms = refresh_start.elapsed().as_secs_f64() * 1000.0;
        let (results, mut timings, diagnostics) =
            self.execute_prepared_on_snapshot(prepared, top_k, filters)?;
        timings.refresh_ms += refresh_ms;
        timings.total_ms += refresh_ms;
        Ok((results, timings, diagnostics))
    }

    /// Query the current immutable snapshot without attempting a refresh.
    /// Callers may use this from concurrent read paths after writes have
    /// refreshed the store. An empty store has no snapshot and returns no hits.
    pub fn query_prepared_cached(
        &self,
        prepared: &PreparedQuery,
        top_k: usize,
        filters: &std::collections::BTreeMap<String, String>,
    ) -> Result<Vec<SearchResult>> {
        if self.snapshot.is_none() {
            return Ok(Vec::new());
        }
        self.execute_prepared_on_snapshot(prepared, top_k, filters)
            .map(|(results, _, _)| results)
    }

    pub fn query_filtered(
        &mut self,
        query: &str,
        top_k: usize,
        filters: &std::collections::BTreeMap<String, String>,
    ) -> Result<Vec<SearchResult>> {
        let prepared = PreparedQuery::new(query);
        self.query_prepared(&prepared, top_k, filters)
    }

    fn execute_prepared_on_snapshot(
        &self,
        prepared: &PreparedQuery,
        top_k: usize,
        filters: &std::collections::BTreeMap<String, String>,
    ) -> Result<(Vec<SearchResult>, QueryTimings, QueryDiagnostics)> {
        execute_prepared_on_snapshot_parts(
            &self.options,
            &self.source_docs,
            &self.records,
            &self.semantic_relations,
            self.snapshot.as_deref(),
            prepared,
            top_k,
            filters,
        )
    }

    /// Multi-query convenience wrapper over the same prepared-query executor.
    /// Refresh happens once, then each query uses the current immutable snapshot.
    pub fn query_filtered_multi(
        &mut self,
        queries: &[&str],
        top_k: usize,
        filters: &std::collections::BTreeMap<String, String>,
    ) -> Result<Vec<Vec<SearchResult>>> {
        self.refresh()?;
        queries
            .iter()
            .map(|query| {
                let prepared = PreparedQuery::new(query);
                self.execute_prepared_on_snapshot(&prepared, top_k, filters)
                    .map(|(results, _, _)| results)
            })
            .collect()
    }

    /// Stages pending source changes into records and derived state.
    ///
    /// Returns the ids of the documents whose records were actually rebuilt.
    /// A dirty document re-upserted with byte-identical content reuses its
    /// stored record (gated by `content_hash`): its segments, lexical
    /// entries, semantic state, and chunk lifecycle are already correct, so
    /// the rebuild is skipped entirely. Skipped docs are still cleared from
    /// `dirty_docs`.
    fn prepare_pending_changes(&mut self) -> Result<HashSet<String>> {
        let dirty_doc_ids = self.dirty_docs.iter().cloned().collect::<Vec<String>>();
        let mut reprocessed_doc_ids = HashSet::new();
        for doc_id in &dirty_doc_ids {
            let incoming_hash = {
                let source_doc = self
                    .source_docs
                    .get(doc_id)
                    .expect("dirty doc should still exist in source docs");
                doc_record_content_hash(source_doc, &self.options)
            };
            let stored_hash = self
                .records
                .get(doc_id)
                .map(|record| record.content_hash.clone())
                .unwrap_or_default();
            if !stored_hash.is_empty() && stored_hash == incoming_hash {
                self.dirty_docs.remove(doc_id);
                continue;
            }
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
            reprocessed_doc_ids.insert(doc_id.clone());
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
        self.semantic_relations = SemanticRelationStore::try_from_documents(
            self.source_docs.values(),
            self.options.supersession,
        )?;
        let lexical_upserts = if self.snapshot.is_none() && self.snapshot_revision == 0 {
            self.records
                .iter()
                .filter(|(doc_id, _)| self.source_docs.contains_key(*doc_id))
                .filter(|(doc_id, _)| !self.tombstones.contains(*doc_id))
                .map(|(_, record)| record)
                .collect::<Vec<_>>()
        } else {
            // Only actually-rebuilt docs need lexical upserts: skipped docs
            // already have correct lexical entries from their last build.
            reprocessed_doc_ids
                .iter()
                .filter_map(|doc_id| self.records.get(doc_id))
                .collect::<Vec<_>>()
        };
        for record in lexical_upserts {
            self.lexical.upsert_record(record)?;
        }
        self.lexical.commit_reload()?;
        Ok(reprocessed_doc_ids)
    }

    fn poll_background_refresh(&mut self) -> Result<()> {
        let Some(background) = self.background_refresh.as_ref() else {
            return Ok(());
        };
        let result = background
            .receiver
            .lock()
            .map_err(|_| anyhow::anyhow!("background refresh lock poisoned"))?
            .try_recv();
        match result {
            Ok(result) => {
                let target_revision = background.target_revision;
                self.background_refresh = None;
                let snapshot = result?;
                if target_revision == self.store_revision {
                    self.snapshot = Some(Arc::new(snapshot));
                    self.snapshot_revision = target_revision;
                    persist_store_metadata(&self.store_paths, &self.options)?;
                    self.persist_compatibility_state()?;
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
