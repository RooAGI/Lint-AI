use anyhow::Result;
use chrono::{Duration, Utc};
use lint_ai::{build_query_snapshot, PipelineOptions, SourceDocument};
use serde_json::json;
use std::collections::BTreeMap;
use std::time::SystemTime;

const TOPICS: &[&str] = &["aurora", "beacon", "comet", "delta", "eclipse"];
const AGE_BUCKETS: &[(&str, i64)] = &[("fresh", 7), ("warm", 45), ("cold", 180)];

fn document(topic: &str, bucket: &str, age_days: i64, today: chrono::NaiveDate) -> SourceDocument {
    let date = today - Duration::days(age_days);
    SourceDocument {
        doc_id: format!("conversation-{topic}-{bucket}"),
        source: format!("conversation://{topic}/{bucket}"),
        content: format!(
            "Conversation memory about project {topic}. The team decided to keep the {topic} rollout plan and review it with the team."
        ),
        concept: topic.to_string(),
        group_id: Some(format!("session-{topic}-{bucket}")),
        filters: BTreeMap::new(),
        headings: vec!["Conversation memory".to_string()],
        links: Vec::new(),
        timestamp: Some(format!("{date}T12:00:00Z")),
        doc_length: 0,
        author_agent: Some("recency-benchmark".to_string()),
    }
}

fn bucket(doc_id: &str) -> &'static str {
    AGE_BUCKETS
        .iter()
        .find_map(|(bucket, _)| doc_id.ends_with(bucket).then_some(*bucket))
        .unwrap_or("unknown")
}

fn main() -> Result<()> {
    let today = chrono::DateTime::<Utc>::from(SystemTime::now()).date_naive();
    let documents = TOPICS
        .iter()
        .flat_map(|topic| {
            AGE_BUCKETS
                .iter()
                .map(move |(bucket, age)| document(topic, bucket, *age, today))
        })
        .collect::<Vec<_>>();
    let index = build_query_snapshot(&documents, &PipelineOptions::default())?;

    let mut latest_at_1 = 0usize;
    let mut latest_at_3 = 0usize;
    let mut latest_rank_sum = 0usize;
    let mut bucket_at_1 = BTreeMap::<&str, usize>::new();
    let mut bucket_at_3 = BTreeMap::<&str, usize>::new();
    let mut boost_sum = 0.0f32;

    for topic in TOPICS {
        let results = index.query(&format!("conversation memory project {topic}"), 3);
        let latest_rank = results
            .iter()
            .position(|result| result.doc_id == format!("conversation-{topic}-fresh"));
        if let Some(rank) = latest_rank {
            latest_rank_sum += rank + 1;
            if rank == 0 {
                latest_at_1 += 1;
            }
            if rank < 3 {
                latest_at_3 += 1;
            }
        }
        if let Some(result) = results.first() {
            *bucket_at_1.entry(bucket(&result.doc_id)).or_default() += 1;
            boost_sum += result.score_breakdown.recency_score;
        }
        for result in results.iter().take(3) {
            *bucket_at_3.entry(bucket(&result.doc_id)).or_default() += 1;
        }
    }

    let query_count = TOPICS.len();
    assert_eq!(
        latest_at_1, query_count,
        "fresh conversation memory must rank first for every controlled query"
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "benchmark": "conversation-recency",
            "query_count": query_count,
            "document_count": documents.len(),
            "age_buckets_days": {
                "fresh": "0-30",
                "warm": "31-90",
                "cold": "91+"
            },
            "metrics": {
                "latest_memory_at_1": latest_at_1 as f64 / query_count as f64,
                "latest_memory_at_3": latest_at_3 as f64 / query_count as f64,
                "mean_latest_memory_rank": latest_rank_sum as f64 / query_count as f64,
                "mean_top_1_recency_score": boost_sum as f64 / query_count as f64,
                "top_1_freshness_distribution": bucket_at_1,
                "top_3_freshness_distribution": bucket_at_3
            },
            "note": "Controlled retrieval benchmark; each topic has identical fresh, warm, and cold conversation memories."
        }))?
    );
    Ok(())
}
