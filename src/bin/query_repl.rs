//! Interactive query harness: index a question's haystack ONCE, then run
//! queries repeatedly without rebuilding.
//!
//! Usage:
//!     query_repl --longmemeval <dataset> --question-id <id> [--bekind] [--show-native-claims]
//!
//! Builds the MemoryService index from the question's haystack sessions,
//! then enters a REPL: type a query, see top retrieved sessions with scores.
//! Type `!ref <date>` to set reference date, `!quit` to exit.
//!
//! Query-side code changes (tokenization, ranking) take effect on the next
//! query without reindexing. Index-side changes require restarting.

use crate::memory_api::{AddRequest, MemoryService, Message, SearchRequest};
use crate::query_plan::PreparedQuery;
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
    /// Print native event and temporal claims stored for the answer sessions.
    #[arg(long, default_value_t = false)]
    show_native_claims: bool,
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
    #[serde(default = "default_turn_role")]
    role: String,
    #[serde(default)]
    content: String,
}

fn default_turn_role() -> String {
    "user".to_string()
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
            .map(|turn| Message {
                role: turn.role.clone(),
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
    if args.bekind {
        println!("Bekind diagnostics: enabled");
        print_indexed_bekind_tags(&service, &entry.answer_session_ids);
    } else {
        println!("Bekind diagnostics: disabled (pass --bekind to inspect query tags)");
    }
    if args.show_native_claims {
        print_indexed_native_claims(&service, &entry.answer_session_ids);
    }
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
        filters.insert("memory_user_id".to_string(), BENCHMARK_USER_ID.to_string());
        let prepared = match reference_date.as_deref() {
            Some(date) => PreparedQuery::new_at(line, date),
            None => PreparedQuery::new(line),
        };
        print_query_processing(&prepared)?;
        if args.bekind {
            print_query_bekind_tags(line, &prepared);
        }
        let start = std::time::Instant::now();
        let response = service.search(SearchRequest {
            query: line.to_string(),
            user_id: BENCHMARK_USER_ID.to_string(),
            top_k: args.top_k,
            session_id: None,
            scope: None,
            filters: Some(filters.clone()),
            options: None,
            reference_date: reference_date.clone(),
        })?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;

        // Inspect the raw index stage with a deep window. The normal service
        // response below still shows the complete MemoryService result order.
        let (index_results, index_timings, index_diagnostics) =
            service.query_index_with_diagnostics(&prepared, 200, &filters)?;
        print_evidence_diagnostics(
            index_results.len(),
            response.data.len(),
            &index_timings,
            &index_diagnostics,
            &entry.answer_session_ids,
        );
        print_answer_index_presence(&index_results, &entry.answer_session_ids, "deep index");
        println!("  MemoryService hit ranking:");
        for (rank, hit) in response.data.iter().enumerate() {
            let mark = if entry
                .answer_session_ids
                .iter()
                .any(|answer| hit.session_id.as_deref() == Some(answer.as_str()))
            {
                " ANSWER"
            } else {
                ""
            };
            println!(
                "    {}. id={} session={:?} score={:.3} structured_evidence={:?}{}",
                rank + 1,
                hit.id,
                hit.session_id,
                hit.score,
                hit.relation_evidence,
                mark
            );
        }
        let (service_depth_results, service_depth_timings, service_depth_diagnostics) =
            service.query_index_with_diagnostics(&prepared, args.top_k, &filters)?;
        println!("Index stage at MemoryService top_k={}", args.top_k);
        print_evidence_diagnostics(
            service_depth_results.len(),
            0,
            &service_depth_timings,
            &service_depth_diagnostics,
            &entry.answer_session_ids,
        );
        print_answer_index_presence(
            &service_depth_results,
            &entry.answer_session_ids,
            "service-depth index",
        );
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

fn print_answer_index_presence(
    results: &[crate::SearchResult],
    answer_session_ids: &[String],
    stage: &str,
) {
    println!("  {stage} gold-session presence:");
    for answer in answer_session_ids {
        if let Some((rank, result)) = results
            .iter()
            .enumerate()
            .find(|(_, result)| result.group_id.as_deref() == Some(answer.as_str()))
        {
            println!(
                "    {answer}: candidate rank={} score={:.3} doc_id={}",
                rank + 1,
                result.score,
                result.doc_id
            );
        } else {
            println!("    {answer}: absent from {stage} results");
        }
    }
    println!();
}

fn print_evidence_diagnostics(
    index_result_count: usize,
    service_result_count: usize,
    timings: &crate::index::QueryTimings,
    diagnostics: &crate::index::QueryDiagnostics,
    answer_session_ids: &[String],
) {
    let answers: HashSet<&str> = answer_session_ids.iter().map(String::as_str).collect();
    let mut groups = diagnostics.evidence_groups.clone();
    groups.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.group_id.cmp(&b.group_id))
    });
    println!(
        "Retrieval stages: MemoryService results={} raw index results={} lexical_hits={} candidates={} evidence_groups={} evidence_ms={:.2}",
        service_result_count,
        index_result_count,
        diagnostics.lexical_hits,
        diagnostics.candidates,
        groups.len(),
        timings.evidence_ms
    );
    println!("  post-evidence group ranking (top 15 plus answer sessions):");
    for (rank, group) in groups.iter().enumerate() {
        if rank >= 15 && !answers.contains(group.group_id.as_str()) {
            continue;
        }
        let answer_mark = if answers.contains(group.group_id.as_str()) {
            " ANSWER"
        } else {
            ""
        };
        println!(
            "  {:>3}. {} base={:.3} evidence={:.3} boost={:.3} final={:.3} pre={} support={} terms={} num={} unit={} date={} pred={} required_relation_docs={} penalty={:.3}{}",
            rank + 1,
            group.group_id,
            group.base_score,
            group.evidence_score,
            group.evidence_boost,
            group.final_score,
            group.pre_evidence_rank,
            group.supporting_docs,
            group.term_matches,
            group.numbers,
            group.units,
            group.dates,
            group.predicates,
            group.required_relation_docs,
            group.penalty,
            answer_mark,
        );
    }
    for answer in answer_session_ids {
        if !groups.iter().any(|group| group.group_id == *answer) {
            println!(
                "  ANSWER {answer}: absent from index evidence groups; the evidence stage had no candidates"
            );
        }
    }
    println!();
}

fn print_query_bekind_tags(original_query: &str, prepared: &PreparedQuery) {
    let augmented_query = prepared.search_query();
    let texts = if augmented_query == original_query {
        vec![original_query]
    } else {
        vec![original_query, augmented_query]
    };
    let tags = behood_query::analyze_query_tags(&texts);

    println!("Bekind query tags:");
    for (text, (locations, activities)) in texts.iter().zip(tags) {
        let label = if *text == original_query {
            "original"
        } else {
            "augmented/search"
        };
        println!("  {label} text: {text}");
        println!("    location tags: {locations:?}");
        println!("    activity sibling tags: {activities:?}");
    }
    println!();
}

fn print_indexed_bekind_tags(service: &MemoryService, answer_session_ids: &[String]) {
    let answer_sessions: HashSet<&str> = answer_session_ids.iter().map(String::as_str).collect();
    let mut tags_by_session: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for document in service.source_documents() {
        let Some(session_id) = document.group_id.as_deref() else {
            continue;
        };
        if !answer_sessions.contains(session_id) {
            continue;
        }
        let tags = tags_by_session.entry(session_id.to_string()).or_default();
        if let Some(record) = service.record_by_id(&document.doc_id) {
            tags.extend(record.semantic_tags.iter().cloned());
        }
    }

    println!("Bekind semantic tags stored on answer sessions:");
    for session_id in answer_session_ids {
        let mut tags = tags_by_session.remove(session_id).unwrap_or_default();
        tags.sort();
        tags.dedup();
        println!("  {session_id}: {tags:?}");
    }

    println!("Structured evidence stored in existing top_claims:");
    let mut evidence_inputs = Vec::new();
    for session_id in answer_session_ids {
        let mut evidence = Vec::new();
        for document in service.source_documents() {
            if document.group_id.as_deref() != Some(session_id.as_str()) {
                continue;
            }
            if let Some(record) = service.record_by_id(&document.doc_id) {
                evidence_inputs.push((session_id.clone(), record.content.clone()));
                evidence.extend(
                    record
                        .top_claims
                        .iter()
                        .filter(|claim| {
                            claim.predicate == "bekind:participated_in"
                                || claim.predicate == "participated_in"
                        })
                        .map(|claim| {
                            format!(
                                "{} -> {} ({:.2})",
                                claim.subject, claim.object, claim.confidence
                            )
                        }),
                );
            }
        }
        println!("  {session_id}: {evidence:?}");
    }
    let input_texts: Vec<&str> = evidence_inputs
        .iter()
        .map(|(_, text)| text.as_str())
        .collect();
    let extracted = behood_query::analyze_evidence_signals_bulk(&input_texts);
    println!("Bekind evidence extractor output on indexed answer records:");
    for ((session_id, _), signals) in evidence_inputs.iter().zip(extracted) {
        if !signals.is_empty() {
            println!("  {session_id}: {signals:?}");
        }
    }
}

fn print_indexed_native_claims(service: &MemoryService, answer_session_ids: &[String]) {
    println!("Native structured claims stored on answer sessions:");
    for session_id in answer_session_ids {
        let mut claims = Vec::new();
        let mut timestamp = None;
        for document in service.source_documents() {
            if document.group_id.as_deref() != Some(session_id.as_str()) {
                continue;
            }
            if let Some(record) = service.record_by_id(&document.doc_id) {
                timestamp = timestamp.or_else(|| record.timestamp.clone());
                claims.extend(record.top_claims.iter().filter(|claim| {
                    matches!(
                        claim.predicate.as_str(),
                        "participated_in"
                            | "participant_role"
                            | "event_type"
                            | "event_of"
                            | "relative_of"
                            | "event_time"
                    )
                }));
            }
        }
        println!("  {session_id} timestamp={timestamp:?}");
        if claims.is_empty() {
            println!("    (no native event/temporal claims)");
        } else {
            for claim in claims {
                println!(
                    "    {} -[{}]-> {} (confidence={:.2})",
                    claim.subject, claim.predicate, claim.object, claim.confidence
                );
            }
        }
    }
    println!();
}

fn print_query_processing(prepared: &PreparedQuery) -> Result<()> {
    let analysis = prepared.analysis();
    let temporal = prepared.temporal_context();
    let anchor_window = temporal
        .anchor_window
        .map(|(start, end)| format!("{}..{}", start.format("%Y-%m-%d"), end.format("%Y-%m-%d")));

    println!("Query processing:");
    println!("  input: {}", analysis.original_query);
    println!("  reference date: {:?}", prepared.reference_date());
    println!("  normalized query: {}", analysis.normalized_query);
    println!("  augmented/search query: {}", prepared.search_query());
    println!("  kind: {:?}", analysis.kind);
    println!("  analyzed intent: {:?}", analysis.intent);
    println!("  routing intent: {:?}", temporal.query_routing_intent);
    println!(
        "  evidence requirements: {:?}",
        analysis.evidence_requirements
    );
    println!("  time hint: {:?}", temporal.time_hint);
    println!(
        "  temporal phrase: {:?}",
        analysis.temporal.as_ref().map(|t| &t.phrase)
    );
    println!(
        "  temporal source: {:?}",
        analysis.temporal.as_ref().map(|t| &t.source)
    );
    println!("  resolved temporal anchor: {:?}", temporal.anchor_date);
    println!(
        "  temporal window: {:?}..{:?}",
        temporal.starts_from, temporal.ends_at
    );
    println!("  anchor date: {:?}", temporal.anchor_date);
    println!("  anchor window: {:?}", anchor_window);
    println!("  window days: {}", temporal.window_days);
    println!("  explicit temporal: {}", temporal.has_explicit_temporal);
    println!("  hard temporal filter: {}", temporal.hard_filter);
    println!("  spans:");
    for span in &analysis.spans {
        println!(
            "    {:?}: {:?} => {:?} (source={}, score={:.3})",
            span.kind, span.text, span.normalized, span.source, span.score
        );
    }
    println!();
    Ok(())
}
