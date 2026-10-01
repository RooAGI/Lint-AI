//! Corpus-wide validation: spaCy index health + Behood metadata audit.
//!
//! Usage: cargo run --release --bin validate_corpus -- /path/to/locomo10.json
//!
//! Reports:
//! 1. SpaCy health: total turns, relations extracted, turns with zero relations
//! 2. Coverage: persons, predicates, sessions
//! 3. Behood metadata: kind distribution, relations per person, gaps

use lint_ai::segments::relations::{extract_relations_via_spacy, RelationTurn};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: validate_corpus <locomo10.json>");
    let data = std::fs::read_to_string(&path).expect("failed to read file");

    // Flatten to RelationTurns
    let mut turns = Vec::new();

    // Parse generically to handle the actual structure
    let v: serde_json::Value = serde_json::from_str(&data).expect("parse as Value");
    let convs = v.as_array().expect("expected array");

    for conv in convs {
        let sample_id = conv
            .get("sample_id")
            .and_then(|s| s.as_str())
            .unwrap_or("?");
        // conversation is an object with session_N keys and session_N_date_time keys
        if let Some(conv_obj) = conv.get("conversation").and_then(|c| c.as_object()) {
            for (key, session_val) in conv_obj {
                // Only process session_N keys (not session_N_date_time)
                if !key.starts_with("session_") || key.contains("date_time") {
                    continue;
                }
                // Get the date for this session
                let date_key = format!("{}_date_time", key);
                let session_date = conv_obj
                    .get(&date_key)
                    .and_then(|d| d.as_str())
                    .map(|s| s.to_string());

                if let Some(turns_arr) = session_val.as_array() {
                    for (turn_idx, turn) in turns_arr.iter().enumerate() {
                        let speaker = turn
                            .get("speaker")
                            .and_then(|s| s.as_str())
                            .unwrap_or("?")
                            .to_string();
                        let text = turn
                            .get("text")
                            .and_then(|s| s.as_str())
                            .unwrap_or("")
                            .to_string();
                        let session_id = format!("{}::{}", sample_id, key);
                        turns.push(RelationTurn {
                            speaker,
                            text,
                            session_id: session_id.clone(),
                            turn_idx,
                            doc_id: format!("{}::{}", session_id, turn_idx),
                            session_date: session_date.clone(),
                        });
                    }
                }
            }
        }
    }

    println!("=== CORPUS STATS ===");
    println!("Total turns: {}", turns.len());

    let speakers: HashSet<_> = turns.iter().map(|t| t.speaker.as_str()).collect();
    println!("Unique speakers: {}", speakers.len());

    let sessions: HashSet<_> = turns.iter().map(|t| t.session_id.as_str()).collect();
    println!("Unique sessions: {}", sessions.len());

    println!("\n=== RUNNING SPACY EXTRACTION ===");
    let output = extract_relations_via_spacy(&turns, Duration::from_secs(300), "en_core_web_sm");

    println!("\n=== SPACY HEALTH ===");
    println!("Relations extracted: {}", output.relations.len());
    println!("Key phrases extracted: {}", output.key_phrases.len());

    let rels_per_turn = output.relations.len() as f64 / turns.len() as f64;
    println!("Avg relations per turn: {:.2}", rels_per_turn);

    // Turns with relations vs without
    let turns_with_rels: HashSet<_> = output.relations.iter().map(|r| r.doc_id.as_str()).collect();
    let turns_without = turns.len() - turns_with_rels.len();
    println!("Turns with ≥1 relation: {}", turns_with_rels.len());
    println!(
        "Turns with 0 relations: {} ({:.1}%)",
        turns_without,
        100.0 * turns_without as f64 / turns.len() as f64
    );

    println!("\n=== PREDICATE DISTRIBUTION (top 20) ===");
    let mut pred_counts: HashMap<&str, usize> = HashMap::new();
    for r in &output.relations {
        *pred_counts.entry(r.predicate.as_str()).or_default() += 1;
    }
    let mut preds: Vec<_> = pred_counts.iter().collect();
    preds.sort_by(|a, b| b.1.cmp(a.1));
    for (pred, count) in preds.iter().take(20) {
        println!("  {}: {}", pred, count);
    }

    println!("\n=== BEHOOD KIND DISTRIBUTION ===");
    let mut kind_counts: HashMap<&str, usize> = HashMap::new();
    for r in &output.relations {
        *kind_counts.entry(r.object_kind.as_str()).or_default() += 1;
    }
    let mut kinds: Vec<_> = kind_counts.iter().collect();
    kinds.sort_by(|a, b| b.1.cmp(a.1));
    for (kind, count) in kinds {
        println!("  {}: {}", kind, count);
    }

    println!("\n=== RELATIONS PER PERSON (top 20) ===");
    let mut person_counts: HashMap<&str, usize> = HashMap::new();
    for r in &output.relations {
        *person_counts.entry(r.subject.as_str()).or_default() += 1;
    }
    let mut persons: Vec<_> = person_counts.iter().collect();
    persons.sort_by(|a, b| b.1.cmp(a.1));
    for (person, count) in persons.iter().take(20) {
        println!("  {}: {}", person, count);
    }
    println!("Total unique persons in relations: {}", person_counts.len());

    println!("\n=== PLACE/EVENT OBJECTS (potential answer candidates) ===");
    let place_event: Vec<_> = output
        .relations
        .iter()
        .filter(|r| r.is_place || r.object_kind == "place" || r.object_kind == "event")
        .collect();
    println!("Place/event relations: {}", place_event.len());

    // Sample a few
    for r in place_event.iter().take(10) {
        println!(
            "  {} -> {} -> '{}' ({})",
            r.subject, r.predicate, r.object, r.object_kind
        );
    }

    println!("\n=== VALIDATION COMPLETE ===");

    // Sample zero-relation turns for diagnosis
    if std::env::args().any(|a| a == "--sample-zero") {
        println!("\n=== ZERO-RELATION TURN SAMPLES (first 20) ===");
        let mut shown = 0;
        for t in &turns {
            if !turns_with_rels.contains(t.doc_id.as_str()) && shown < 20 {
                println!(
                    "[{}] {}: {}",
                    t.session_id,
                    t.speaker,
                    t.text.chars().take(120).collect::<String>()
                );
                shown += 1;
            }
        }
    }

    // Dump all thing-kind object texts for offline analysis
    if let Some(path) = std::env::args()
        .skip_while(|a| a != "--dump-thing-objects")
        .nth(1)
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&path).expect("create dump file");
        for r in &output.relations {
            if r.object_kind == "thing" {
                writeln!(f, "{}", r.object).ok();
            }
        }
        println!("Dumped thing objects to {}", path);
    }

    // Sample thing-kind objects to see why they're not classified
    if std::env::args().any(|a| a == "--sample-thing") {
        println!("\n=== THING-KIND OBJECT SAMPLES (first 20) ===");
        let mut shown = 0;
        for r in &output.relations {
            if r.object_kind == "thing" && shown < 20 {
                println!("{} -> {} -> '{}'", r.subject, r.predicate, r.object);
                shown += 1;
            }
        }
    }
}
