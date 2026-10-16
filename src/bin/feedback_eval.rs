//! Read-only offline evaluation of previously submitted MCP search feedback.
//!
//! This binary deliberately does not import the Membrick library, open a
//! workspace, or write any files. Pass copied telemetry files explicitly.

use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Default)]
struct Counts {
    reviews: usize,
    joined_searches: usize,
    usable_searches: usize,
    judged_documents: usize,
    excluded_reviews: usize,
    malformed_history_lines: usize,
    overall_labels: BTreeMap<String, usize>,
    relevance_labels: BTreeMap<String, usize>,
}

fn usage() -> &'static str {
    "Usage: cargo run --features experimental --bin feedback_eval -- --history PATH --reviews PATH [--k 5,10]\n\nReads copied MCP history and search-review state files and prints aggregate, read-only evaluation results."
}

fn arguments() -> Result<(PathBuf, PathBuf, Vec<usize>), String> {
    let mut history = None;
    let mut reviews = None;
    let mut cutoffs = vec![5, 10];
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--history" => history = args.next().map(PathBuf::from),
            "--reviews" => reviews = args.next().map(PathBuf::from),
            "--k" => {
                let raw = args
                    .next()
                    .ok_or("--k requires comma-separated positive integers")?;
                cutoffs = raw
                    .split(',')
                    .map(|part| {
                        part.parse::<usize>()
                            .ok()
                            .filter(|value| *value > 0)
                            .ok_or_else(|| format!("invalid cutoff: {part}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                cutoffs.sort_unstable();
                cutoffs.dedup();
            }
            "-h" | "--help" => return Err(usage().to_string()),
            _ => return Err(format!("unknown argument: {arg}\n\n{}", usage())),
        }
    }
    Ok((
        history.ok_or_else(|| format!("--history is required\n\n{}", usage()))?,
        reviews.ok_or_else(|| format!("--reviews is required\n\n{}", usage()))?,
        cutoffs,
    ))
}

fn mcp_result(record: &Value) -> Option<Value> {
    let text = record
        .pointer("/response/result/content/0/text")?
        .as_str()?;
    serde_json::from_str(text).ok()
}

fn relevance_gain(value: &str) -> Option<f64> {
    match value {
        "relevant" => Some(3.0),
        "partially_relevant" => Some(1.0),
        "not_relevant" => Some(0.0),
        _ => None,
    }
}

fn ndcg(gains: &[f64], cutoff: usize) -> Option<f64> {
    if gains.is_empty() {
        return None;
    }
    let dcg = |values: &[f64]| {
        values
            .iter()
            .take(cutoff)
            .enumerate()
            .map(|(index, gain)| gain / ((index + 2) as f64).log2())
            .sum::<f64>()
    };
    let mut ideal = gains.to_vec();
    ideal.sort_by(|left, right| right.total_cmp(left));
    let ideal_dcg = dcg(&ideal);
    (ideal_dcg > 0.0).then(|| dcg(gains) / ideal_dcg)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("feedback evaluation failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let (history_path, reviews_path, cutoffs) = arguments()?;
    let history_text = fs::read_to_string(&history_path)
        .map_err(|error| format!("cannot read {}: {error}", history_path.display()))?;
    let state_text = fs::read_to_string(&reviews_path)
        .map_err(|error| format!("cannot read {}: {error}", reviews_path.display()))?;
    let state: Value = serde_json::from_str(&state_text)
        .map_err(|error| format!("invalid review state: {error}"))?;
    let reviews = state
        .get("reviews")
        .and_then(Value::as_object)
        .ok_or("review state must contain an object at `reviews`")?;

    let mut counts = Counts::default();
    let mut search_by_id = HashMap::<String, Value>::new();
    for line in history_text.lines().filter(|line| !line.trim().is_empty()) {
        let record: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(_) => {
                counts.malformed_history_lines += 1;
                continue;
            }
        };
        if record.get("tool").and_then(Value::as_str) != Some("search") {
            continue;
        }
        if let Some(id) = record.get("trace_id").and_then(Value::as_str) {
            search_by_id.insert(id.to_string(), record);
        }
    }

    let mut ndcg_totals = BTreeMap::<usize, (f64, usize)>::new();
    let mut reciprocal_rank_total = 0.0;
    let mut reciprocal_rank_count = 0;
    for (search_id, entry) in reviews {
        if entry.get("status").and_then(Value::as_str) != Some("reviewed") {
            continue;
        }
        let Some(feedback) = entry.get("feedback") else {
            continue;
        };
        counts.reviews += 1;
        if let Some(label) = feedback.get("label").and_then(Value::as_str) {
            *counts.overall_labels.entry(label.to_string()).or_default() += 1;
        }
        let Some(record) = search_by_id.get(search_id) else {
            counts.excluded_reviews += 1;
            continue;
        };
        counts.joined_searches += 1;
        let Some(result) = mcp_result(record) else {
            counts.excluded_reviews += 1;
            continue;
        };
        let Some(hits) = result.get("results").and_then(Value::as_array) else {
            counts.excluded_reviews += 1;
            continue;
        };
        let Some(judgments) = feedback.get("document_judgments").and_then(Value::as_array) else {
            counts.excluded_reviews += 1;
            continue;
        };
        let mut labels = HashMap::<String, (usize, f64)>::new();
        let mut valid = judgments.len() == hits.len();
        for judgment in judgments {
            let Some(doc_id) = judgment.get("doc_id").and_then(Value::as_str) else {
                valid = false;
                break;
            };
            let Some(rank) = judgment
                .get("rank")
                .and_then(Value::as_u64)
                .map(|number| number as usize)
            else {
                valid = false;
                break;
            };
            let Some(relevance) = judgment.get("relevance").and_then(Value::as_str) else {
                valid = false;
                break;
            };
            let Some(gain) = relevance_gain(relevance) else {
                valid = false;
                break;
            };
            if labels.insert(doc_id.to_string(), (rank, gain)).is_some() {
                valid = false;
                break;
            }
            *counts
                .relevance_labels
                .entry(relevance.to_string())
                .or_default() += 1;
        }
        if !valid {
            counts.excluded_reviews += 1;
            continue;
        }
        let mut ranked = Vec::with_capacity(hits.len());
        for hit in hits {
            let Some(doc_id) = hit
                .get("doc_id")
                .or_else(|| hit.get("id"))
                .and_then(Value::as_str)
            else {
                valid = false;
                break;
            };
            let Some((rank, gain)) = labels.get(doc_id) else {
                valid = false;
                break;
            };
            ranked.push((*rank, *gain));
        }
        if !valid || labels.len() != hits.len() {
            counts.excluded_reviews += 1;
            continue;
        }
        ranked.sort_by_key(|(rank, _)| *rank);
        if ranked
            .iter()
            .enumerate()
            .any(|(index, (rank, _))| *rank != index + 1)
        {
            counts.excluded_reviews += 1;
            continue;
        }
        let gains = ranked.iter().map(|(_, gain)| *gain).collect::<Vec<_>>();
        for cutoff in &cutoffs {
            if let Some(value) = ndcg(&gains, *cutoff) {
                let total = ndcg_totals.entry(*cutoff).or_default();
                total.0 += value;
                total.1 += 1;
            }
        }
        if let Some(rank) = ranked
            .iter()
            .find(|(_, gain)| *gain == 3.0)
            .map(|(rank, _)| *rank)
        {
            reciprocal_rank_total += 1.0 / rank as f64;
        }
        reciprocal_rank_count += 1;
        counts.usable_searches += 1;
        counts.judged_documents += judgments.len();
    }

    println!("Offline search-feedback evaluation (read-only)");
    println!("History: {}", history_path.display());
    println!("Reviews: {}", reviews_path.display());
    println!("Reviewed searches: {}", counts.reviews);
    println!("Joined to search records: {}", counts.joined_searches);
    println!(
        "Complete usable judgments: {} searches, {} query-document labels",
        counts.usable_searches, counts.judged_documents
    );
    println!("Excluded or unmatched reviews: {}", counts.excluded_reviews);
    println!(
        "Malformed history lines skipped: {}",
        counts.malformed_history_lines
    );
    println!("Overall labels: {}", format_counts(&counts.overall_labels));
    println!(
        "Document relevance labels: {}",
        format_counts(&counts.relevance_labels)
    );
    for cutoff in &cutoffs {
        match ndcg_totals.get(cutoff) {
            Some((sum, count)) if *count > 0 => println!(
                "Baseline macro NDCG@{cutoff}: {:.4} ({count} searches)",
                sum / *count as f64
            ),
            _ => println!("Baseline macro NDCG@{cutoff}: unavailable"),
        }
    }
    if reciprocal_rank_count > 0 {
        println!(
            "Baseline MRR (relevant only): {:.4} ({reciprocal_rank_count} searches)",
            reciprocal_rank_total / reciprocal_rank_count as f64
        );
    } else {
        println!("Baseline MRR (relevant only): unavailable");
    }
    println!("Interpretation: metrics describe only returned documents with complete feedback; they are not corpus-wide recall and do not measure a trained model.");
    Ok(())
}

fn format_counts(counts: &BTreeMap<String, usize>) -> String {
    if counts.is_empty() {
        return "none".to_string();
    }
    counts
        .iter()
        .map(|(label, count)| format!("{label}={count}"))
        .collect::<Vec<_>>()
        .join(", ")
}
