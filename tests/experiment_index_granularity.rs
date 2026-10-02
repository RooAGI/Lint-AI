//! Experiment: turn-level vs session-level indexing for conv-47_q1.
//!
//! Question: "When will John start his new job?"
//! Reference: "In July, 2022" (from session_13: "I'm starting next month" on June 13)
//!
//! Hypothesis: Turn-level indexing fragments the answer across documents,
//! so the retriever can't connect "dream job" (turn A) with "starting next
//! month" (turn C). Session-level indexing keeps them together.

use lint_ai::memory_api::{MemoryService, SearchRequest};
use lint_ai::PipelineOptions;
use lint_ai::SourceDocument;
use std::collections::BTreeMap;

fn make_service() -> MemoryService {
    MemoryService::in_memory(PipelineOptions::default())
}

fn source_doc(doc_id: &str, content: &str, group_id: &str) -> SourceDocument {
    SourceDocument {
        doc_id: doc_id.to_string(),
        source: format!("test/{doc_id}"),
        content: content.to_string(),
        concept: "test-turn".to_string(),
        group_id: Some(group_id.to_string()),
        headings: vec![],
        links: vec![],
        timestamp: Some("4:30 pm on 13 June, 2022".to_string()),
        doc_length: content.len(),
        author_agent: None,
        filters: BTreeMap::new(),
        key_phrases: vec![],
        key_phrase_extraction_hash: String::new(),
    }
}

#[test]
fn experiment_turn_vs_session_indexing() {
    let query = "When will John start his new job?";

    // Session 13 turns (the relevant session)
    let turn_a = "John: I finally got my dream job! After lots of interviews and late nights, I got the offer and was ecstatic. Can't wait to start my journey!";
    let turn_b = "James: Wow, John! Congrats on getting your dream job. I'm super stoked for you. When do you start?";
    let turn_c = "John: Thank you! I'm starting next month.";

    // Distractor: session 18 turn (has "job" but wrong context)
    let distractor = "John: Hey James, good catching up! Been a while huh? I made a huge call - recently left my IT job after 3 years. It was tough but I wanted something that made a difference.";

    // --- Turn-level index (current) ---
    let mut turn_svc = make_service();
    turn_svc.upsert(source_doc("s13::turn0", turn_a, "s13"));
    turn_svc.upsert(source_doc("s13::turn1", turn_b, "s13"));
    turn_svc.upsert(source_doc("s13::turn2", turn_c, "s13"));
    turn_svc.upsert(source_doc("s18::turn0", distractor, "s18"));
    turn_svc.refresh().unwrap();

    let turn_results = turn_svc
        .search(SearchRequest {
            query: query.to_string(),
            options: None,
            user_id: String::new(),
            top_k: 5,
            session_id: None,
            scope: Some("test".to_string()),
            filters: None,
        })
        .unwrap();

    println!("\n=== Turn-level index ===");
    for (i, r) in turn_results.data.iter().enumerate() {
        println!(
            "{}. {} (score {:.2}): {}",
            i + 1,
            r.id,
            r.score,
            r.content.chars().take(60).collect::<String>()
        );
    }

    // --- Session-level index (proposed) ---
    let mut sess_svc = make_service();
    let session_13 = format!("{turn_a}\n{turn_b}\n{turn_c}");
    sess_svc.upsert(source_doc("s13", &session_13, "s13"));
    sess_svc.upsert(source_doc("s18", distractor, "s18"));
    sess_svc.refresh().unwrap();

    let sess_results = sess_svc
        .search(SearchRequest {
            query: query.to_string(),
            options: None,
            user_id: String::new(),
            top_k: 5,
            session_id: None,
            scope: Some("test".to_string()),
            filters: None,
        })
        .unwrap();

    println!("\n=== Session-level index ===");
    for (i, r) in sess_results.data.iter().enumerate() {
        println!(
            "{}. {} (score {:.2}): {}",
            i + 1,
            r.id,
            r.score,
            r.content.chars().take(60).collect::<String>()
        );
    }

    // The session-level index should rank s13 higher because it contains
    // "job", "start", AND "next month" together.
    let turn_s13_rank = turn_results
        .data
        .iter()
        .position(|r| r.id.starts_with("s13"));
    let sess_s13_rank = sess_results.data.iter().position(|r| r.id == "s13");

    println!("\nTurn-level s13 rank: {turn_s13_rank:?}");
    println!("Session-level s13 rank: {sess_s13_rank:?}");
}
