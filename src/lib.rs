//! Lint-AI: persistent memory layer for AI coding agents.
//!
//! Lint-AI gives coding agents (Claude Code, Codex, Gemini CLI, Muse, Agy)
//! long-term memory with a property most stores can't offer: retrieval knows
//! what is *still true*. Memories carry timestamps, and newer memories can
//! explicitly supersede older ones, so recall returns the current state of
//! the world instead of a pile of contradictions.
//!
//! The core workflow is three steps:
//!
//! 1. **Record** — capture session content as [`SourceDocument`]s.
//! 2. **Index** — build an [`IndexStore`] with [`build_index_store`].
//! 3. **Recall** — query through [`memory_api::MemoryService`] (`add` /
//!    `search`) or [`MemoryIndex`] directly; superseded memories are
//!    filtered automatically.
//!
//! Basic usage:
//! ```no_run
//! use lint_ai::{build_index_store, PipelineOptions, SourceDocument};
//!
//! let docs = vec![SourceDocument::with_stable_doc_id_from_source(
//!     "docs/getting-started.md".to_string(),
//!     "# Getting started\n\nInstall with `cargo install lint-ai`.".to_string(),
//!     "getting-started".to_string(),
//!     None,
//!     vec!["Getting started".to_string()],
//!     vec![],
//!     None,
//!     None,
//! )];
//!
//! let store = build_index_store(&docs, &PipelineOptions::default()).unwrap();
//! ```
//!
//! Agents usually don't touch this crate directly — they talk to the
//! `lint-ai` binary over MCP (`--claude-code-serve`, `--codex-serve`, …).
//! The library API is for embedding: custom hosts, the Python bindings,
//! and the HTTP server binary.

pub mod cli;
mod config;
mod ids;
pub mod index;
#[cfg(any(
    feature = "claude-code",
    feature = "codex",
    feature = "gemini-cli",
    feature = "agy",
    feature = "muse-code"
))]
mod integrations;
pub mod memory_api;
pub mod pipeline;
pub mod query_plan;
mod remote_query;
pub mod segments;
pub mod semantic_relations;
pub mod source;
pub mod telemetry;
pub mod temporal_fact;

// Internal implementation details. These modules are intentionally not part
// of the public API; use the re-exports above instead.
mod adapters;
mod aggregation;
mod chunking;
mod claim_extractor;
mod corpus_graph;
mod engine;
mod filters;
mod graph;
mod ownership;
mod query_expansion;
mod query_semantics;
mod report;
mod review;
mod rules;
mod symbols;
mod temporal;
mod tokenizer;
mod tier1;
mod usage;

pub use crate::ids::{stable_chunk_id, stable_doc_id_from_source};
pub use crate::index::{
    GlobalBm25Statistics, MemoryIndex, QueryDiagnostics, QueryTimings, SearchResult,
    TemporalQueryContext,
};
pub use crate::pipeline::{
    build_index_store, build_query_snapshot, build_query_snapshot_from_source_documents,
    resolve_store_paths, ChunkStrategy, IndexDump, IndexLocation, IndexStore, IndexStoreInspection,
    MemoryIndexLayout, MemoryIndexSegmentInspection, MemoryIndexSnapshot,
    MemoryIndexSnapshotInspection, PipelineOptions, PublishedIndexSnapshot, StorePaths,
    Tier1NerProvider, Tier1TermRankerKind,
};
pub use crate::segments::{
    SegmentManifest, SegmentManifestEntry, ShardQueryCompleteness, ShardQueryFailure,
};
pub use crate::semantic_relations::{
    DocumentSemanticState, SemanticClaim, SemanticRelation, SemanticRelationKind,
    SemanticRelationStore, SemanticStatus, SupersessionOptions,
};
pub use crate::source::SourceDocument;
pub use crate::temporal_fact::{TemporalFact, TemporalFactStore, TimelineEvent, TimelinePair};
// Re-exported so the public `index::DocRecord` struct can be constructed by
// downstream users (`key_entities` / `important_terms` fields).
pub use crate::tier1::{RankedTerm, Tier1Entity};
// Date helper for building timestamped documents (used by benchmarks; also
// useful for anyone constructing `SourceDocument`s with timestamps).
pub use crate::temporal::parse_temporal_date;
// Query-pipeline utilities used by the benchmark binaries and useful for
// power users driving `MemoryIndex` directly.
pub use crate::aggregation::{build_aggregate_output, AggregateOutput};
pub use crate::query_expansion::normalize_for_index;
pub use crate::query_semantics::{analyze_query, QueryAnalysis, QueryTimeHint};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(feature = "python")]
#[pyclass(name = "IndexStore", unsendable)]
struct PyIndexStore {
    inner: IndexStore,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyIndexStore {
    #[new]
    fn new() -> Self {
        Self {
            inner: IndexStore::in_memory(PipelineOptions::default()),
        }
    }

    #[pyo3(signature = (doc_id, content, source=None, timestamp=None, group_id=None))]
    fn upsert(
        &mut self,
        doc_id: String,
        content: String,
        source: Option<String>,
        timestamp: Option<String>,
        group_id: Option<String>,
    ) {
        let source = source.unwrap_or_else(|| format!("memory://{}", doc_id));
        let doc = SourceDocument {
            source,
            concept: "memory".to_string(),
            headings: vec![],
            links: vec![],
            timestamp,
            doc_length: content.len(),
            author_agent: None,
            filters: std::collections::BTreeMap::new(),
            group_id,
            doc_id,
            content,
        };

        self.inner.upsert(doc);
    }

    fn query(&mut self, py: Python<'_>, query: &str, top_k: usize) -> PyResult<Py<PyAny>> {
        let results = self
            .inner
            .query(query, top_k)
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
        let json = serde_json::to_string(&results)
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
        let json_module = py.import("json")?;
        Ok(json_module.getattr("loads")?.call1((json,))?.unbind())
    }

    fn remove(&mut self, doc_id: &str) -> bool {
        self.inner.remove(doc_id).is_some()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn is_dirty(&self) -> bool {
        self.inner.is_dirty()
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn lint_ai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<PyIndexStore>()?;
    Ok(())
}
