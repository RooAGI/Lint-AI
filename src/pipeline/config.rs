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
    /// When true (default), a follow-up query inside a session is retrieved
    /// deep (top-200) and rescored with the two-stage conversational rerank
    /// (session selection, then neighbor-context / speaker-match pinpoint).
    /// When false, session follow-ups keep the base ranking.
    pub conversational_rerank: bool,
    /// When true (default), search consults the dependency-parse relation
    /// index for structured fact questions ("which place did both X and Y
    /// visit", "where was X between <dates>") and blends that evidence
    /// ahead of the lexical results. The index builds lazily on first use
    /// from the indexed documents (one batched extractor run, bounded by a
    /// timeout) and is fail-open: extraction failures fall back to the
    /// lexical path. The write path never pays for it.
    pub structured_fact_retrieval: bool,
    /// When true (default), newly written documents are enriched with
    /// grammar-accepted entity key phrases in the background: a worker
    /// thread batches pending documents through the extractor's
    /// `key_phrases_only` mode and backfills each document's `key_phrases`,
    /// which the next refresh folds into the cap-exempt segment entity
    /// channel (literal, unstemmed). Writes stay fast; enrichment is
    /// fail-open and applies on the next refresh cycle. When false, no
    /// background extraction runs and documents keep empty key phrases.
    pub key_phrase_enrichment: bool,
    /// Override path for the extractor script. `None` (default) uses the
    /// bundled `scripts/spacy_relations.py`; set to a test double in tests.
    pub extractor_script: Option<std::path::PathBuf>,
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
            conversational_rerank: true,
            structured_fact_retrieval: true,
            key_phrase_enrichment: true,
            extractor_script: None,
        }
    }
}

/// Base per-query segment breadth for the production server: each search
/// routes to this many candidate segments before adaptive expansion.
/// Shared by the server and the retrieval benchmark so both see the same
/// candidate pool.
pub const DEFAULT_SEGMENT_QUERY_TOP_N: usize = 5;

/// The pipeline options the production server uses by default: segmented
/// layout at [`DEFAULT_SEGMENT_QUERY_TOP_N`] with the gated coverage-local
/// router (the measured best recall-per-latency trade-off). The server
/// binary applies its CLI flags as overrides on top of this; the retrieval
/// benchmark builds from it directly so benchmark numbers track production
/// instead of a hand-duplicated copy of its defaults.
pub fn default_production_pipeline_options() -> PipelineOptions {
    PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: DEFAULT_SEGMENT_QUERY_TOP_N,
            routing_strategy: SegmentRoutingStrategy::TypedEvidenceMultiplicative,
        },
        ..PipelineOptions::default()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_defaults_are_segmented_top5_gated() {
        let options = default_production_pipeline_options();
        match options.memory_index_layout {
            MemoryIndexLayout::Segmented {
                query_top_n,
                routing_strategy,
            } => {
                assert_eq!(query_top_n, DEFAULT_SEGMENT_QUERY_TOP_N);
                assert!(matches!(
                    routing_strategy,
                    SegmentRoutingStrategy::TypedEvidenceMultiplicative
                ));
            }
            other => panic!("expected Segmented layout, got {other:?}"),
        }
        assert!(options.conversational_rerank);
        assert!(options.structured_fact_retrieval);
        assert!(options.key_phrase_enrichment);
        assert!(!options.fuse_global_arm);
    }
}
