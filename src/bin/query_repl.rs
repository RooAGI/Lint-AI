//! Interactive query harness: index a question's haystack ONCE, then run
//! queries repeatedly without rebuilding.
//!
//! Usage:
//!     query_repl --longmemeval <dataset> --question-id <id> [--bekind]
//!
//! Builds the MemoryService index from the question's haystack sessions,
//! then enters a REPL: type a query, see top retrieved sessions with scores.
//! Type `!ref <date>` to set reference date, `!quit` to exit.
//!
//! Query-side code changes (tokenization, ranking) take effect on the next
//! query without reindexing. Index-side changes require restarting.

use crate::memory_api::{AddRequest, MemoryService, Message, SearchRequest};
use crate::{
    behood_query, parse_reference_date, ChunkStrategy, MemoryIndexLayout, PipelineOptions,
    Tier1TermRankerKind,
};
use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::collections::{BTreeMap, HashSet};
use std::io::{BufRead, Write};

/// Fixed user ID for all benchmark documents.
const BENCHMARK_USER_ID: &str = "benchmark-user";

#[derive(Debug, Parser)]
#[command(name = "query-repl")]
#[command(about = "Index a haystack once, then interactively query it")]
struct Args {
    #[arg(long)]
    longmemeval: String,
    #[arg(long)]
    question_id: String,
    #[arg(long, default_value_t = false)]
    bekind: bool,
    #[arg(long, default_value_t = 10)]
    top_k: usize,
}

#[derive(Deserialize)]
struct Entry {
    question_id: String,
    question: String,
    question_date: String,
    haystack_session_ids: Vec<String>,
    #[serde(default)]
    haystack_dates: Vec<String>,
    haystack_sessions: Vec<Vec<Turn>>,
    #[serde(default)]
    answer_session_ids: Vec<String>,
}

#[derive(Deserialize)]
struct Turn {
    #[serde(default)]
    content: String,
}

pub(crate) fn main() -> Result<()> {
    let args = Args::parse();
    if args.bekind {
        behood_query::set_enabled(true);
    }

    let raw = std::fs::read_to_string(&args.longmemeval)
        .with_context(|| format!("failed to read {}", args.longmemeval))?;
    let entries: Vec<Entry> = serde_json::from_str(&raw)?;
    let entry = entries
        .into_iter()
        .find(|e| e.question_id == args.question_id)
        .with_context(|| format!("question_id {} not found", args.question_id))?;

    println!(
        "Indexing {} haystack sessions for '{}'...",
        entry.haystack_session_ids.len(),
        entry.question_id
    );

    let options = PipelineOptions {
        ner_provider: crate::Tier1NerProvider::Heuristic,
        spacy_model: "en_core_web_sm".to_string(),
        term_ranker: Tier1TermRankerKind::Yake,
        chunk_strategy: ChunkStrategy::Heading,
        chunk_lines: 40,
        chunk_overlap: 10,
        chunk_target_tokens: 450,
        chunk_max_tokens: 800,
        memory_index_layout: MemoryIndexLayout::Single,
        ..PipelineOptions::default()
    };
    let mut service = MemoryService::in_memory(options);

    let mut seen = HashSet::new();
    let mut add_requests = Vec::new();
    for (sess_idx, (session_id, turns)) in entry
        .haystack_session_ids
        .iter()
        .zip(entry.haystack_sessions.iter())
        .enumerate()
    {
        if turns.is_empty() || !seen.insert(session_id.clone()) {
            continue;
        }
        let session_date = entry
            .haystack_dates
            .get(sess_idx)
            .cloned()
            .unwrap_or_else(|| entry.question_date.clone());
        let timestamp_ms = parse_reference_date(&session_date)
            .and_then(|d| d.and_hms_opt(0, 0, 0))
            .map(|dt| dt.and_utc().timestamp_millis());
        let messages: Vec<Message> = turns
            .iter()
            .filter(|turn| !turn.content.trim().is_empty())
            .enumerate()
            .map(|(i, turn)| Message {
                role: if i % 2 == 0 {
                    "user".to_string()
                } else {
                    "assistant".to_string()
                },
                content: turn.content.clone(),
                timestamp: timestamp_ms,
                expires_at_ms: None,
                supersedes_id: None,
            })
            .collect();
        if messages.is_empty() {
            continue;
        }
        add_requests.push(AddRequest {
            user_id: BENCHMARK_USER_ID.to_string(),
            request_id: format!("{}-{}", entry.question_id, session_id),
            session_id: session_id.clone(),
            messages,
        });
    }
    service.add_batch(add_requests)?;
    println!("Indexed. Answer sessions: {:?}", entry.answer_session_ids);
    println!("Original question: {}", entry.question);
    println!("Commands: !ref <date>  !quit");
    println!();

    let mut reference_date = Some(entry.question_date.clone());
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        print!("query> ");
        std::io::stdout().flush()?;
        let line = match lines.next() {
            Some(Ok(l)) => l,
            _ => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "!quit" {
            break;
        }
        if let Some(date) = line.strip_prefix("!ref ") {
            reference_date = Some(date.trim().to_string());
            println!("reference_date = {:?}", reference_date);
            continue;
        }

        let mut filters = BTreeMap::new();
        filters.insert("user".to_string(), BENCHMARK_USER_ID.to_string());
        let start = std::time::Instant::now();
        let response = service.search(SearchRequest {
            query: line.to_string(),
            user_id: BENCHMARK_USER_ID.to_string(),
            top_k: args.top_k,
            session_id: None,
            scope: None,
            filters: Some(filters),
            options: None,
            reference_date: reference_date.clone(),
        })?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;

        let mut seen_sess = HashSet::new();
        let mut ranked: Vec<(String, f32)> = Vec::new();
        for m in &response.data {
            if let Some(sid) = &m.session_id {
                if seen_sess.insert(sid.clone()) {
                    ranked.push((sid.clone(), m.score));
                }
            }
            if ranked.len() >= args.top_k {
                break;
            }
        }
        println!("({:.1}ms) top sessions:", ms);
        for (i, (sid, score)) in ranked.iter().enumerate() {
            let mark = if entry.answer_session_ids.contains(sid) {
                "  <-- ANSWER"
            } else {
                ""
            };
            println!("  {}. {} (score {:.4}){}", i + 1, sid, score, mark);
        }
        println!();
    }
    Ok(())
}
