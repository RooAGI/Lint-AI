//! Lint AI — semantic linting for Markdown documentation and related semantic graphs.
//!
//! This crate provides the core pipeline for building a concept inventory from
//! a Markdown corpus, matching mentions, building symbol and ownership graphs,
//! and reporting missing cross-references and orphan/unreachable pages.
//!
//! Basic usage:
//! ```no_run
//! use lint_ai::graph::Graph;
//! use lint_ai::report::Report;
//! use lint_ai::rules::{cross_refs::check_cross_refs, orphan_pages::check_orphans};
//! use lint_ai::config::Config;
//!
//! let graph = Graph::build("docs", 5_000_000, 50_000, 20, 100_000_000).unwrap();
//! let mut report = Report::new();
//! let cfg = Config::default();
//! check_orphans(&graph, &mut report);
//! check_cross_refs(&graph, &mut report, &cfg);
//! ```

pub mod adapters;
pub mod aggregation;
pub mod chunking;
pub mod claim_extractor;
pub mod cli;
pub mod corpus_graph;
pub mod config;
pub mod engine;
pub mod filters;
pub mod graph;
pub mod ids;
pub mod index;
pub mod ownership;
pub mod pipeline;
pub mod query_expansion;
pub mod query_semantics;
pub mod report;
pub mod review;
pub mod rules;
pub mod source;
pub mod symbols;
pub mod temporal;
pub mod temporal_fact;
pub mod tier1;
pub mod usage;

pub use crate::claim_extractor::{ClaimExtractor, ConservativeClaimExtractor, ExtractedClaims};
pub use crate::corpus_graph::CorpusGraph;
pub use crate::ids::{stable_chunk_id, stable_doc_id_from_source};
pub use crate::index::{
    MemoryIndex, QueryDiagnostics, QueryTimings, SearchResult, TemporalQueryContext,
};
pub use crate::ownership::{
    FlowEdge, FlowEdgeKind, FlowState, LeakFinding, OwnershipKind, OwnershipRecord,
};
pub use crate::pipeline::{
    build_index_store, build_query_snapshot, build_query_snapshot_from_source_documents,
    resolve_store_paths, ChunkStrategy, IndexLocation, IndexStore, PipelineOptions, StorePaths,
    Tier1NerProvider, Tier1TermRankerKind,
};
pub use crate::review::{
    DocumentSummary, OwnershipSummary, ReviewCategory, ReviewContext, ReviewDiff,
    ReviewDiffSummary, ReviewEvidence, ReviewFileChange, ReviewFinding, ReviewHunk, ReviewPacket,
    ReviewRepoRef, ReviewSeverity, ReviewUsageSummary, SymbolSummary,
};
pub use crate::source::SourceDocument;
pub use crate::symbols::{
    CorpusIndex, IngestBundle, SymbolKind, SymbolLocation, SymbolRecord, SymbolRelationKind,
    SymbolStore,
};
pub use crate::temporal_fact::{TemporalFact, TemporalFactStore, TimelineEvent, TimelinePair};
pub use crate::usage::{UsageEdge, UsageEdgeKind, UsageGraph, UsageNode, UsageNodeKind};
