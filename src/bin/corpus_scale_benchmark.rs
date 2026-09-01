//! Measure retrieval build/query behavior as the LongMemEval corpus grows.

use anyhow::{Context, Result};
use clap::Parser;
use lint_ai::{build_query_snapshot, PipelineOptions, SourceDocument, TemporalQueryContext};
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(about = "Measure Lint-AI retrieval scaling on LongMemEval-S")]
struct Args {
    #[arg(long)]
    longmemeval: PathBuf,
    /// Corpus sizes (unique sessions) to benchmark.
    #[arg(long, value_delimiter = ',', default_value = "1000,5000,10000,25000")]
    sizes: Vec<usize>,
    #[arg(long, default_value_t = 100)]
    queries: usize,
    #[arg(long, default_value_t = 10)]
    top_k: usize,
}

#[derive(Deserialize)]
struct Entry {
    question: String,
    #[serde(default)]
    answer_session_ids: Vec<String>,
    #[serde(default)]
    haystack_session_ids: Vec<String>,
    #[serde(default)]
    haystack_sessions: Vec<Vec<Turn>>,
}
#[derive(Deserialize, Clone)]
struct Turn {
    role: String,
    content: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let raw: Vec<Entry> = serde_json::from_str(&fs::read_to_string(&args.longmemeval)?)
        .context("failed to parse LongMemEval JSON")?;
    let mut sessions = Vec::<SourceDocument>::new();
    let mut seen = HashSet::new();
    for entry in &raw {
        for (idx, id) in entry.haystack_session_ids.iter().enumerate() {
            if !seen.insert(id.clone()) {
                continue;
            }
            let turns = entry
                .haystack_sessions
                .get(idx)
                .cloned()
                .unwrap_or_default();
            let content = turns
                .into_iter()
                .map(|t| format!("{}: {}", t.role, t.content))
                .collect::<Vec<_>>()
                .join("\n");
            sessions.push(SourceDocument {
                doc_id: id.clone(),
                source: format!("longmemeval://{id}"),
                content: content.clone(),
                concept: "conversation".into(),
                group_id: Some(id.clone()),
                filters: Default::default(),
                headings: vec!["Conversation".into()],
                links: vec![],
                timestamp: None,
                doc_length: content.len(),
                author_agent: None,
            });
        }
    }
    eprintln!(
        "loaded {} unique sessions from {} questions",
        sessions.len(),
        raw.len()
    );
    let mut sizes = args.sizes;
    sizes.sort_unstable();
    sizes.dedup();
    println!("size,eligible_queries,build_ms,query_p50_ms,query_p95_ms,recall_any_at_k");
    for size in sizes {
        let docs = sessions
            .iter()
            .take(size.min(sessions.len()))
            .cloned()
            .collect::<Vec<_>>();
        let ids = docs
            .iter()
            .map(|d| d.doc_id.as_str())
            .collect::<HashSet<_>>();
        // Score only questions whose complete labeled answer set is in this
        // corpus slice; partial gold sets would make recall incomparable.
        let eligible = raw
            .iter()
            .filter(|e| {
                !e.answer_session_ids.is_empty()
                    && e.answer_session_ids
                        .iter()
                        .all(|id| ids.contains(id.as_str()))
            })
            .take(args.queries)
            .collect::<Vec<_>>();
        let build_start = Instant::now();
        let index = build_query_snapshot(&docs, &PipelineOptions::default())?;
        let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;
        let mut latencies = Vec::with_capacity(eligible.len());
        let mut hits = 0usize;
        for entry in &eligible {
            let start = Instant::now();
            let allowed = entry
                .haystack_session_ids
                .iter()
                .filter(|id| ids.contains(id.as_str()))
                .cloned()
                .collect::<HashSet<_>>();
            let temporal = TemporalQueryContext {
                allowed_doc_ids: Some(&allowed),
                ..TemporalQueryContext::default()
            };
            let (results, _, _) =
                index.query_with_temporal_context(&entry.question, args.top_k, temporal);
            latencies.push(start.elapsed().as_secs_f64() * 1000.0);
            if results.iter().any(|r| {
                r.group_id
                    .as_deref()
                    .map(|id| entry.answer_session_ids.iter().any(|answer| answer == id))
                    .unwrap_or_else(|| entry.answer_session_ids.contains(&r.doc_id))
            }) {
                hits += 1;
            }
        }
        latencies.sort_by(f64::total_cmp);
        let percentile = |p: f64| -> f64 {
            latencies
                .get(((latencies.len() as f64 * p).ceil() as usize).saturating_sub(1))
                .copied()
                .unwrap_or(0.0)
        };
        let recall = if eligible.is_empty() {
            0.0
        } else {
            hits as f64 / eligible.len() as f64
        };
        println!(
            "{},{},{:.2},{:.3},{:.3},{:.4}",
            docs.len(),
            eligible.len(),
            build_ms,
            percentile(0.50),
            percentile(0.95),
            recall
        );
    }
    Ok(())
}
