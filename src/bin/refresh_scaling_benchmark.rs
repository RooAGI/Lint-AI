//! Measure IndexStore::refresh() scaling: cold build vs single-write refresh
//! as the corpus grows. This answers whether one document write costs
//! O(total corpus) under the segmented snapshot layout.
//!
//! Two corpus modes:
//! - synthetic (default): generated sessions, any size.
//! - --longmemeval <path>: real LongMemEval-S haystack sessions (one document
//!   per session, group_id = session id, like a promoted session). Sizes beyond
//!   the dataset's unique sessions tile it with replica suffixes (distinct
//!   segment ids, real text); the output notes how many sessions are real.

use anyhow::{Context, Result};
use clap::Parser;
use lint_ai::pipeline::{IndexLocation, IndexStore, MemoryIndexLayout, PipelineOptions};
use lint_ai::segments::SegmentRoutingStrategy;
use lint_ai::SourceDocument;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(about = "Measure refresh() scaling with corpus size")]
struct Args {
    /// Corpus sizes (documents) to benchmark.
    #[arg(long, value_delimiter = ',', default_value = "1000,2500,5000,10000")]
    sizes: Vec<usize>,
    /// Use the LongMemEval-S haystack sessions at this path as corpus text.
    #[arg(long)]
    longmemeval: Option<PathBuf>,
    /// Documents per session segment (synthetic mode only).
    #[arg(long, default_value_t = 25)]
    docs_per_session: usize,
    /// Repetitions of the single-write refresh; median is reported.
    #[arg(long, default_value_t = 3)]
    reps: usize,
}

#[derive(Deserialize)]
struct LmEntry {
    #[serde(default)]
    haystack_session_ids: Vec<String>,
    #[serde(default)]
    haystack_sessions: Vec<Vec<LmTurn>>,
}

#[derive(Deserialize, Clone)]
struct LmTurn {
    role: String,
    content: String,
}

struct SessionText {
    id: String,
    content: String,
}

fn load_longmemeval(path: &PathBuf) -> Result<Vec<SessionText>> {
    let raw: Vec<LmEntry> = serde_json::from_str(&fs::read_to_string(path)?)
        .context("failed to parse LongMemEval JSON")?;
    let mut sessions = Vec::new();
    let mut seen = HashSet::new();
    for entry in &raw {
        for (idx, id) in entry.haystack_session_ids.iter().enumerate() {
            if !seen.insert(id.clone()) {
                continue;
            }
            let content = entry
                .haystack_sessions
                .get(idx)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .map(|t| format!("{}: {}", t.role, t.content))
                .collect::<Vec<_>>()
                .join("\n");
            sessions.push(SessionText {
                id: id.clone(),
                content,
            });
        }
    }
    Ok(sessions)
}

// Deterministic pseudo-random word picker so synthetic runs are comparable.
fn word(i: usize, salt: usize) -> &'static str {
    const VOCAB: &[&str] = &[
        "session",
        "memory",
        "project",
        "agent",
        "context",
        "retrieval",
        "index",
        "segment",
        "provider",
        "model",
        "token",
        "query",
        "document",
        "record",
        "pipeline",
        "cache",
        "refresh",
        "snapshot",
        "routing",
        "catalog",
        "decision",
        "schema",
        "migration",
        "config",
        "endpoint",
        "timeout",
        "retry",
        "fallback",
        "latency",
        "throughput",
        "benchmark",
        "baseline",
        "regression",
        "release",
        "branch",
        "commit",
        "review",
        "merge",
    ];
    VOCAB[(i.wrapping_mul(2654435761).wrapping_add(salt)) % VOCAB.len()]
}

fn make_synthetic_doc(doc_idx: usize, session_idx: usize, provider: &str) -> SourceDocument {
    let mut content = String::new();
    for para in 0..4 {
        for w in 0..60 {
            content.push_str(word(doc_idx * 7 + para * 131 + w, session_idx));
            content.push(' ');
        }
        content.push_str(". ");
    }
    let group_id = format!("{provider}-session:{session_idx:05}");
    SourceDocument {
        doc_id: format!("session-recording:{provider}:{session_idx:05}:{doc_idx:06}"),
        source: format!("{provider}://session/{session_idx:05}"),
        content: content.clone(),
        concept: "conversation".into(),
        group_id: Some(group_id),
        filters: Default::default(),
        headings: vec!["Session notes".into()],
        links: vec![],
        timestamp: None,
        doc_length: content.len(),
        author_agent: Some(provider.into()),
    }
}

fn make_longmemeval_doc(session: &SessionText, replica: usize) -> SourceDocument {
    let suffix = if replica == 0 {
        String::new()
    } else {
        format!("#r{replica}")
    };
    let group_id = format!("longmemeval-session:{}{}", session.id, suffix);
    let content = session.content.clone();
    SourceDocument {
        doc_id: format!("longmemeval:{}{}", session.id, suffix),
        source: format!("longmemeval://{}{}", session.id, suffix),
        content: content.clone(),
        concept: "conversation".into(),
        group_id: Some(group_id),
        filters: Default::default(),
        headings: vec!["Conversation".into()],
        links: vec![],
        timestamp: None,
        doc_length: content.len(),
        author_agent: None,
    }
}

fn options() -> PipelineOptions {
    PipelineOptions {
        index_location: IndexLocation::InMemory,
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 3,
            routing_strategy: SegmentRoutingStrategy::SparseOverlap,
        },
        ..PipelineOptions::default()
    }
}

fn median(mut xs: Vec<u128>) -> u128 {
    xs.sort_unstable();
    xs[xs.len() / 2]
}

fn main() -> Result<()> {
    let args = Args::parse();
    const PROVIDERS: &[&str] = &["claude", "codex", "gemini-cli", "agy", "muse"];

    let lm_sessions: Option<Vec<SessionText>> = args
        .longmemeval
        .as_ref()
        .map(load_longmemeval)
        .transpose()?;
    if let Some(sessions) = &lm_sessions {
        eprintln!(
            "loaded {} unique LongMemEval-S sessions (sizes beyond this tile with replica ids)",
            sessions.len()
        );
    }

    println!(
        "{:>8} {:>9} {:>18} {:>22}",
        "docs", "segments", "cold_refresh_ms", "single_write_refresh_ms"
    );
    for &size in &args.sizes {
        let mut store = IndexStore::new(options());
        let mut segments = 0usize;
        if let Some(sessions) = &lm_sessions {
            // One document per session; tile with replica suffixes past the dataset size.
            for doc_idx in 0..size {
                let session = &sessions[doc_idx % sessions.len()];
                let replica = doc_idx / sessions.len();
                store.upsert(make_longmemeval_doc(session, replica));
            }
            segments = size; // one segment per document
        } else {
            segments = size.div_ceil(args.docs_per_session);
            for doc_idx in 0..size {
                let session_idx = doc_idx / args.docs_per_session;
                let provider = PROVIDERS[session_idx % PROVIDERS.len()];
                store.upsert(make_synthetic_doc(doc_idx, session_idx, provider));
            }
        }

        let t = Instant::now();
        store.refresh()?;
        let cold_ms = t.elapsed().as_millis();

        // Single write into an existing segment: the incremental case.
        let mut single_ms = Vec::with_capacity(args.reps);
        for rep in 0..args.reps {
            if let Some(sessions) = &lm_sessions {
                let session = &sessions[rep % sessions.len()];
                // New write into an existing session's segment.
                let mut doc = make_longmemeval_doc(session, 0);
                doc.doc_id = format!("{}:write{rep}", doc.doc_id);
                store.upsert(doc);
            } else {
                let session_idx = rep % segments;
                let provider = PROVIDERS[session_idx % PROVIDERS.len()];
                store.upsert(make_synthetic_doc(size + rep, session_idx, provider));
            }
            let t = Instant::now();
            store.refresh()?;
            single_ms.push(t.elapsed().as_millis());
        }

        println!(
            "{:>8} {:>9} {:>18} {:>22}",
            size,
            segments,
            cold_ms,
            median(single_ms)
        );
    }
    Ok(())
}
