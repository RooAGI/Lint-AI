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
pub mod conversation_state;
mod conversational_rerank;
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
pub mod session_prepare;
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
mod tier1;
mod tokenizer;
mod usage;

pub use crate::ids::{stable_chunk_id, stable_doc_id_from_source};
pub use crate::index::{
    GlobalBm25Statistics, MemoryIndex, QueryDiagnostics, QueryTimings, SearchResult,
    TemporalQueryContext,
};
pub use crate::pipeline::{
    build_doc_records, build_index_store, build_query_snapshot, build_query_snapshot_from_records,
    build_query_snapshot_from_source_documents, resolve_store_paths, ChunkStrategy, IndexDump,
    IndexLocation, IndexStore, IndexStoreInspection, MemoryIndexLayout,
    MemoryIndexSegmentInspection, MemoryIndexSnapshot, MemoryIndexSnapshotInspection,
    PipelineOptions, PublishedIndexSnapshot, StorePaths, Tier1NerProvider, Tier1TermRankerKind,
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
pub use crate::query_semantics::{
    analyze_query, parse_reference_date, resolve_anchor_window, resolve_temporal_anchor,
    temporal_anchor_is_span, QueryAnalysis, QueryTimeHint,
};

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyAny;
#[cfg(feature = "python")]
use reqwest::blocking::Client;
#[cfg(feature = "python")]
use serde::de::DeserializeOwned;
#[cfg(feature = "python")]
use std::path::Path;

#[cfg(feature = "python")]
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(feature = "python")]
struct RemoteMemoryClient {
    client: Client,
    base_url: String,
    api_key: Option<String>,
}

#[cfg(feature = "python")]
impl RemoteMemoryClient {
    fn new(base_url: String, api_key: Option<String>) -> anyhow::Result<Self> {
        let base_url = base_url.trim_end_matches('/').to_string();
        anyhow::ensure!(!base_url.is_empty(), "base_url must not be empty");
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;
        Ok(Self {
            client,
            base_url,
            api_key,
        })
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::blocking::RequestBuilder {
        let mut request = self
            .client
            .request(method, format!("{}{}", self.base_url, path));
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }
        request
    }

    fn decode<T: DeserializeOwned>(response: reqwest::blocking::Response) -> anyhow::Result<T> {
        let status = response.status();
        let body = response.text()?;
        if !status.is_success() {
            anyhow::bail!("remote memory request failed ({status}): {body}");
        }
        Ok(serde_json::from_str(&body)?)
    }

    fn add(&self, request: &memory_api::AddRequest) -> anyhow::Result<memory_api::AddResponse> {
        Self::decode(
            self.request(reqwest::Method::POST, "/v1/memories")
                .json(request)
                .send()?,
        )
    }

    fn search(
        &self,
        request: &memory_api::SearchRequest,
    ) -> anyhow::Result<memory_api::SearchResponse> {
        Self::decode(
            self.request(reqwest::Method::POST, "/v1/memories/search")
                .json(request)
                .send()?,
        )
    }

    fn get(
        &self,
        request: &memory_api::GetRequest,
    ) -> anyhow::Result<Option<memory_api::MemoryRecord>> {
        let response = self
            .request(
                reqwest::Method::GET,
                &format!("/v1/memories/{}", encode_path_segment(&request.memory_id)),
            )
            .query(&[
                ("user_id", request.user_id.as_str()),
                (
                    "include_inactive",
                    if request.include_inactive {
                        "true"
                    } else {
                        "false"
                    },
                ),
            ])
            .send()?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        Self::decode(response).map(Some)
    }

    fn list(&self, request: &memory_api::ListRequest) -> anyhow::Result<memory_api::ListResponse> {
        Self::decode(
            self.request(reqwest::Method::GET, "/v1/memories")
                .query(request)
                .send()?,
        )
    }

    fn update(
        &self,
        request: &memory_api::UpdateRequest,
    ) -> anyhow::Result<Option<memory_api::MemoryRecord>> {
        let response = self
            .request(
                reqwest::Method::PATCH,
                &format!("/v1/memories/{}", encode_path_segment(&request.memory_id)),
            )
            .json(request)
            .send()?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        Self::decode(response).map(Some)
    }

    fn delete(&self, user_id: &str, memory_id: &str) -> anyhow::Result<bool> {
        let response = self
            .request(
                reqwest::Method::DELETE,
                &format!("/v1/memories/{}", encode_path_segment(memory_id)),
            )
            .query(&[("user_id", user_id)])
            .send()?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(false);
        }
        let status = response.status();
        if !status.is_success() {
            anyhow::bail!("remote memory deletion failed ({status})");
        }
        Ok(true)
    }

    fn refresh(&self) -> anyhow::Result<()> {
        let response = self
            .request(reqwest::Method::POST, "/v1/memories/refresh")
            .send()?;
        let status = response.status();
        if !status.is_success() {
            anyhow::bail!("remote memory refresh failed ({status})");
        }
        Ok(())
    }
}

#[cfg(feature = "python")]
enum MemoryBackend {
    Local(memory_api::MemoryService),
    Remote(RemoteMemoryClient),
}

#[cfg(feature = "python")]
struct PyMemoryCore {
    backend: MemoryBackend,
}

#[cfg(feature = "python")]
impl PyMemoryCore {
    fn local(path: Option<String>) -> PyResult<Self> {
        let store = match path {
            Some(path) => {
                memory_api::MemoryService::at_path(Path::new(&path), PipelineOptions::default())
                    .map_err(runtime_error)?
            }
            None => memory_api::MemoryService::in_memory(PipelineOptions::default()),
        };
        Ok(Self {
            backend: MemoryBackend::Local(store),
        })
    }

    fn remote(base_url: String, api_key: Option<String>) -> PyResult<Self> {
        Ok(Self {
            backend: MemoryBackend::Remote(
                RemoteMemoryClient::new(base_url, api_key).map_err(runtime_error)?,
            ),
        })
    }

    fn add(
        &mut self,
        py: Python<'_>,
        request_id: String,
        user_id: String,
        session_id: String,
        messages: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let messages = python_value_to_json(py, messages)?;
        let messages = serde_json::from_value(messages).map_err(json_error)?;
        let request = memory_api::AddRequest {
            request_id,
            messages,
            user_id,
            session_id,
        };
        let response = match &mut self.backend {
            MemoryBackend::Local(service) => service.add(request).map_err(runtime_error)?,
            MemoryBackend::Remote(client) => {
                py.detach(|| client.add(&request)).map_err(runtime_error)?
            }
        };
        json_to_python(py, &response)
    }

    fn search(
        &mut self,
        py: Python<'_>,
        query: String,
        user_id: String,
        top_k: usize,
    ) -> PyResult<Py<PyAny>> {
        let request = memory_api::SearchRequest {
            query,
            options: None,
            user_id,
            top_k,
        };
        let response = match &mut self.backend {
            MemoryBackend::Local(service) => service.search(request).map_err(runtime_error)?,
            MemoryBackend::Remote(client) => py
                .detach(|| client.search(&request))
                .map_err(runtime_error)?,
        };
        json_to_python(py, &response.data)
    }

    fn get(
        &self,
        py: Python<'_>,
        memory_id: String,
        user_id: String,
        include_inactive: bool,
    ) -> PyResult<Py<PyAny>> {
        let request = memory_api::GetRequest {
            user_id,
            memory_id,
            include_inactive,
        };
        let response = match &self.backend {
            MemoryBackend::Local(service) => service.get(request).map_err(runtime_error)?,
            MemoryBackend::Remote(client) => {
                py.detach(|| client.get(&request)).map_err(runtime_error)?
            }
        };
        json_to_python(py, &response)
    }

    fn list(
        &self,
        py: Python<'_>,
        user_id: String,
        session_id: Option<String>,
        limit: usize,
        cursor: Option<String>,
        include_inactive: bool,
    ) -> PyResult<Py<PyAny>> {
        let request = memory_api::ListRequest {
            user_id,
            session_id,
            limit,
            cursor,
            include_inactive,
        };
        let response = match &self.backend {
            MemoryBackend::Local(service) => service.list(request).map_err(runtime_error)?,
            MemoryBackend::Remote(client) => {
                py.detach(|| client.list(&request)).map_err(runtime_error)?
            }
        };
        json_to_python(py, &response)
    }

    fn update(
        &mut self,
        py: Python<'_>,
        memory_id: String,
        user_id: String,
        content: String,
        role: Option<String>,
        timestamp: Option<i64>,
        expires_at_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        let request = memory_api::UpdateRequest {
            user_id,
            memory_id,
            content,
            role,
            timestamp,
            expires_at_ms,
        };
        let response = match &mut self.backend {
            MemoryBackend::Local(service) => service.update(request).map_err(runtime_error)?,
            MemoryBackend::Remote(client) => py
                .detach(|| client.update(&request))
                .map_err(runtime_error)?,
        };
        json_to_python(py, &response)
    }

    fn delete(&mut self, py: Python<'_>, user_id: String, memory_id: String) -> PyResult<bool> {
        match &mut self.backend {
            MemoryBackend::Local(service) => {
                service.delete(&user_id, &memory_id).map_err(runtime_error)
            }
            MemoryBackend::Remote(client) => py
                .detach(|| client.delete(&user_id, &memory_id))
                .map_err(runtime_error),
        }
    }

    fn refresh(&mut self, py: Python<'_>) -> PyResult<()> {
        match &mut self.backend {
            MemoryBackend::Local(service) => service.refresh().map_err(runtime_error),
            MemoryBackend::Remote(client) => py.detach(|| client.refresh()).map_err(runtime_error),
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Memory", unsendable)]
struct PyMemory {
    inner: PyMemoryCore,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyMemory {
    #[new]
    #[pyo3(signature = (path=None, base_url=None, api_key=None))]
    fn new(
        path: Option<String>,
        base_url: Option<String>,
        api_key: Option<String>,
    ) -> PyResult<Self> {
        if path.is_some() && base_url.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "path and base_url cannot both be set",
            ));
        }
        let inner = match base_url {
            Some(base_url) => PyMemoryCore::remote(base_url, api_key)?,
            None => PyMemoryCore::local(path)?,
        };
        Ok(Self { inner })
    }

    #[pyo3(signature = (request_id, user_id, session_id, messages))]
    fn add(
        &mut self,
        py: Python<'_>,
        request_id: String,
        user_id: String,
        session_id: String,
        messages: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.inner
            .add(py, request_id, user_id, session_id, messages)
    }

    fn search(
        &mut self,
        py: Python<'_>,
        query: String,
        user_id: String,
        top_k: usize,
    ) -> PyResult<Py<PyAny>> {
        self.inner.search(py, query, user_id, top_k)
    }

    #[pyo3(signature = (memory_id, user_id, include_inactive=false))]
    fn get(
        &self,
        py: Python<'_>,
        memory_id: String,
        user_id: String,
        include_inactive: bool,
    ) -> PyResult<Py<PyAny>> {
        self.inner.get(py, memory_id, user_id, include_inactive)
    }

    #[pyo3(signature = (user_id, session_id=None, limit=100, cursor=None, include_inactive=false))]
    fn list(
        &self,
        py: Python<'_>,
        user_id: String,
        session_id: Option<String>,
        limit: usize,
        cursor: Option<String>,
        include_inactive: bool,
    ) -> PyResult<Py<PyAny>> {
        self.inner
            .list(py, user_id, session_id, limit, cursor, include_inactive)
    }

    #[pyo3(signature = (memory_id, user_id, content, role=None, timestamp=None, expires_at_ms=None))]
    fn update(
        &mut self,
        py: Python<'_>,
        memory_id: String,
        user_id: String,
        content: String,
        role: Option<String>,
        timestamp: Option<i64>,
        expires_at_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        self.inner.update(
            py,
            memory_id,
            user_id,
            content,
            role,
            timestamp,
            expires_at_ms,
        )
    }

    fn delete(&mut self, py: Python<'_>, user_id: String, memory_id: String) -> PyResult<bool> {
        self.inner.delete(py, user_id, memory_id)
    }

    fn refresh(&mut self, py: Python<'_>) -> PyResult<()> {
        self.inner.refresh(py)
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "RemoteMemory", unsendable)]
struct PyRemoteMemory {
    inner: PyMemoryCore,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyRemoteMemory {
    #[new]
    #[pyo3(signature = (base_url, api_key=None))]
    fn new(base_url: String, api_key: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: PyMemoryCore::remote(base_url, api_key)?,
        })
    }

    #[pyo3(signature = (request_id, user_id, session_id, messages))]
    fn add(
        &mut self,
        py: Python<'_>,
        request_id: String,
        user_id: String,
        session_id: String,
        messages: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.inner
            .add(py, request_id, user_id, session_id, messages)
    }

    fn search(
        &mut self,
        py: Python<'_>,
        query: String,
        user_id: String,
        top_k: usize,
    ) -> PyResult<Py<PyAny>> {
        self.inner.search(py, query, user_id, top_k)
    }

    #[pyo3(signature = (memory_id, user_id, include_inactive=false))]
    fn get(
        &self,
        py: Python<'_>,
        memory_id: String,
        user_id: String,
        include_inactive: bool,
    ) -> PyResult<Py<PyAny>> {
        self.inner.get(py, memory_id, user_id, include_inactive)
    }

    #[pyo3(signature = (user_id, session_id=None, limit=100, cursor=None, include_inactive=false))]
    fn list(
        &self,
        py: Python<'_>,
        user_id: String,
        session_id: Option<String>,
        limit: usize,
        cursor: Option<String>,
        include_inactive: bool,
    ) -> PyResult<Py<PyAny>> {
        self.inner
            .list(py, user_id, session_id, limit, cursor, include_inactive)
    }

    #[pyo3(signature = (memory_id, user_id, content, role=None, timestamp=None, expires_at_ms=None))]
    fn update(
        &mut self,
        py: Python<'_>,
        memory_id: String,
        user_id: String,
        content: String,
        role: Option<String>,
        timestamp: Option<i64>,
        expires_at_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        self.inner.update(
            py,
            memory_id,
            user_id,
            content,
            role,
            timestamp,
            expires_at_ms,
        )
    }

    fn delete(&mut self, py: Python<'_>, user_id: String, memory_id: String) -> PyResult<bool> {
        self.inner.delete(py, user_id, memory_id)
    }

    fn refresh(&mut self, py: Python<'_>) -> PyResult<()> {
        self.inner.refresh(py)
    }
}

#[cfg(feature = "python")]
fn encode_path_segment(value: &str) -> String {
    value
        .bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                (byte as char).to_string()
            }
            _ => format!("%{byte:02X}"),
        })
        .collect()
}

#[cfg(feature = "python")]
fn python_value_to_json(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    let json_module = py.import("json")?;
    let encoded: String = json_module.getattr("dumps")?.call1((value,))?.extract()?;
    serde_json::from_str(&encoded).map_err(json_error)
}

#[cfg(feature = "python")]
fn json_to_python<T: serde::Serialize>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>> {
    let encoded = serde_json::to_string(value).map_err(json_error)?;
    let json_module = py.import("json")?;
    Ok(json_module.getattr("loads")?.call1((encoded,))?.unbind())
}

#[cfg(feature = "python")]
fn runtime_error(error: anyhow::Error) -> pyo3::PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
}

#[cfg(feature = "python")]
fn json_error(error: impl std::fmt::Display) -> pyo3::PyErr {
    pyo3::exceptions::PyValueError::new_err(error.to_string())
}

#[cfg(feature = "python")]
#[pymodule]
fn lint_ai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<PyMemory>()?;
    m.add_class::<PyRemoteMemory>()?;
    Ok(())
}
