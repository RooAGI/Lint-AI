use anyhow::Result;
use chrono::{DateTime, Utc};
use clap::Parser;
use lint_ai::index::DocRecord;
use lint_ai::{
    build_query_snapshot, temporal::parse_temporal_date, PipelineOptions, SourceDocument,
};
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

#[derive(Debug, Parser)]
#[command(about = "Evaluate freshness on a persisted Lint-AI project memory corpus")]
struct Args {
    /// Path to semantic/records.json produced by IndexStore.
    #[arg(long, default_value = ".lint-ai/codex-memory/semantic/records.json")]
    records: PathBuf,
}

#[derive(Debug, Deserialize)]
struct PersistedRecords {
    records: Vec<DocRecord>,
}

const QUERIES: &[&str] = &[
    "memory",
    "temporal",
    "hooks",
    "benchmark",
    "Codex",
    "AGY",
    "skills",
    "installation",
];

fn age_bucket(timestamp: Option<&str>, today: chrono::NaiveDate) -> &'static str {
    let Some(date) = parse_temporal_date(timestamp) else {
        return "unknown";
    };
    let age = today.signed_duration_since(date).num_days().max(0);
    match age {
        0..=30 => "fresh",
        31..=90 => "warm",
        _ => "cold",
    }
}

fn source_document(record: DocRecord) -> SourceDocument {
    SourceDocument {
        doc_id: record.doc_id,
        source: record.source,
        content: record.content,
        concept: record.probable_topic.unwrap_or_default(),
        group_id: record.group_id,
        filters: record.filters,
        headings: record.headings,
        links: record.doc_links,
        timestamp: record.timestamp,
        doc_length: record.doc_length,
        author_agent: record.author_agent,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let payload: PersistedRecords = serde_json::from_str(&fs::read_to_string(&args.records)?)?;
    let now = DateTime::<Utc>::from(SystemTime::now());
    let today = now.date_naive();
    let timestamp_by_id = payload
        .records
        .iter()
        .map(|record| (record.doc_id.clone(), record.timestamp.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut corpus_distribution = BTreeMap::<&str, usize>::new();
    for timestamp in timestamp_by_id.values() {
        *corpus_distribution
            .entry(age_bucket(timestamp.as_deref(), today))
            .or_default() += 1;
    }

    let documents = payload
        .records
        .into_iter()
        .map(source_document)
        .collect::<Vec<_>>();
    let index = build_query_snapshot(&documents, &PipelineOptions::default())?;
    let mut retrieved_distribution = BTreeMap::<&str, usize>::new();
    let mut top_1_distribution = BTreeMap::<&str, usize>::new();
    let mut query_reports = Vec::new();

    for query in QUERIES {
        let results = index.query(query, 5);
        for result in &results {
            let bucket = age_bucket(
                timestamp_by_id
                    .get(&result.doc_id)
                    .and_then(|timestamp| timestamp.as_deref()),
                today,
            );
            *retrieved_distribution.entry(bucket).or_default() += 1;
        }
        if let Some(result) = results.first() {
            let bucket = age_bucket(
                timestamp_by_id
                    .get(&result.doc_id)
                    .and_then(|timestamp| timestamp.as_deref()),
                today,
            );
            *top_1_distribution.entry(bucket).or_default() += 1;
            query_reports.push(json!({
                "query": query,
                "top_1_doc_id": result.doc_id,
                "top_1_bucket": bucket,
                "top_1_recency_score": result.score_breakdown.recency_score,
            }));
        }
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "benchmark": "lint-ai-project-memory-recency",
            "records_path": args.records,
            "record_count": timestamp_by_id.len(),
            "query_count": QUERIES.len(),
            "age_buckets": {"fresh": "0-30 days", "warm": "31-90 days", "cold": "91+ days"},
            "corpus_freshness_distribution": corpus_distribution,
            "retrieved_top_5_freshness_distribution": retrieved_distribution,
            "retrieved_top_1_freshness_distribution": top_1_distribution,
            "queries": query_reports,
            "note": "This evaluates the current persisted project memory corpus. It measures freshness distribution, not semantic answer correctness."
        }))?
    );
    Ok(())
}
