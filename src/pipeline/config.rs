use crate::index::MemoryIndex;
use crate::segments::{SegmentRoutingStrategy, SegmentedMemoryIndex};
use crate::semantic_relations::SupersessionOptions;
use clap::ValueEnum;
use serde::Serialize;
use std::path::PathBuf;
use tantivy::doc;
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

#[derive(Debug, Clone, Default)]
pub enum MemoryIndexLayout {
    #[default]
    Single,
    Segmented {
        query_top_n: usize,
        routing_strategy: SegmentRoutingStrategy,
    },
    /// Adaptive segmented routing starts at `query_top_n` and expands up to
    /// `max_query_n` when routed segments do not cover enough query evidence.
    AdaptiveSegmented {
        query_top_n: usize,
        max_query_n: usize,
        routing_strategy: SegmentRoutingStrategy,
    },
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
    pub supersession: SupersessionOptions,
    pub index_location: IndexLocation,
    pub memory_index_layout: MemoryIndexLayout,
    /// When true (default), a multi-segment query also runs the corpus-wide
    /// all-segments arm and fuses it with the routed arm via reciprocal rank
    /// fusion. When false, only the routed arm runs: lower latency, slightly
    /// lower recall on weak routers.
    pub fuse_global_arm: bool,
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
            supersession: SupersessionOptions::default(),
            index_location: IndexLocation::InMemory,
            memory_index_layout: MemoryIndexLayout::Single,
            fuse_global_arm: false,
        }
    }
}

#[allow(clippy::large_enum_variant)]
pub enum MemoryIndexSnapshot {
    Single(MemoryIndex),
    Segmented(SegmentedMemoryIndex),
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryIndexSegmentInspection {
    pub segment_id: String,
    pub document_count: usize,
    pub doc_ids: Vec<String>,
    pub profile_term_count: usize,
    pub profile_entity_count: usize,
    pub profile_topic_count: usize,
    pub profile_local_memory_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryIndexSnapshotInspection {
    pub layout: String,
    pub segment_count: usize,
    pub global_document_count: usize,
    pub segments: Vec<MemoryIndexSegmentInspection>,
}

#[derive(Debug, Clone, Serialize)]
pub struct IndexStoreInspection {
    pub source_document_count: usize,
    pub record_count: usize,
    pub dirty: bool,
    pub store_revision: u64,
    pub snapshot_revision: u64,
    pub tombstones: Vec<String>,
    pub snapshot: Option<MemoryIndexSnapshotInspection>,
}

impl MemoryIndexSnapshot {
    pub fn single_index(&self) -> Option<&MemoryIndex> {
        match self {
            Self::Single(index) => Some(index),
            Self::Segmented(_) => None,
        }
    }

    pub fn is_segmented(&self) -> bool {
        matches!(self, Self::Segmented(_))
    }

    pub fn segment_count(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Segmented(index) => index.len(),
        }
    }
}
