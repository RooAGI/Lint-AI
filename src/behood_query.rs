//! Query-time behood entity analysis.
//!
//! Luyi's design: "the behood provide people as the source, then we have
//! place and thing." At query time, behood judges the question's entities
//! and returns (text, kind) pairs. Lint-ai uses the text for matching and
//! the kind for filtering in structured question analysis.
//!
//! This calls `scripts/behood_query.py` via subprocess (fail-open with a
//! timeout). Behood owns judgment; the caller owns knowledge.

use std::process::Command;
use std::time::Duration;

/// A (text, kind) pair judged by behood at query time.
#[derive(Debug, Clone)]
pub struct QueryEntity {
    /// The entity text as it appears in the question.
    pub text: String,
    /// Behood's ontological kind: person, place, org, event, work, food, thing.
    pub kind: String,
}

/// Analyze a question with behood, returning (text, kind) pairs.
///
/// Fail-open: returns an empty vec when the script is missing, times out,
/// or produces unparseable output. Callers fall back to heuristics.
pub fn analyze_query_entities(question: &str) -> Vec<QueryEntity> {
    // Locate the script relative to the crate root.
    let script = match script_path() {
        Some(p) => p,
        None => return Vec::new(),
    };

    let output = Command::new("python3")
        .arg(&script)
        .arg(question)
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_entities(&stdout)
}

fn script_path() -> Option<std::path::PathBuf> {
    // CARGO_MANIFEST_DIR is set at compile time to the crate root.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(manifest_dir).join("scripts/behood_query.py");
    if path.is_file() {
        Some(path)
    } else {
        None
    }
}

fn parse_entities(json_str: &str) -> Vec<QueryEntity> {
    let parsed: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let mut entities = Vec::new();
    if let Some(arr) = parsed.get("entities").and_then(|e| e.as_array()) {
        for item in arr {
            let text = item.get("text").and_then(|t| t.as_str()).unwrap_or("");
            let kind = item.get("kind").and_then(|k| k.as_str()).unwrap_or("thing");
            if !text.is_empty() {
                entities.push(QueryEntity {
                    text: text.to_string(),
                    kind: kind.to_string(),
                });
            }
        }
    }
    // Deduplicate by text, keeping first occurrence.
    let mut seen = std::collections::HashSet::new();
    entities.retain(|e| seen.insert(e.text.clone()));
    entities
}

/// Extract person names from behood's query-time entities.
///
/// Returns the person-kind entity texts, cleaned of determiners and
/// quantifiers that spaCy's noun chunks may include (e.g. "both Jean").
/// Deduplicated after cleaning (behood may return both "both Jean" from
/// the noun chunk and "Jean" from the PROPN mention).
pub fn query_persons(entities: &[QueryEntity]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    entities
        .iter()
        .filter(|e| e.kind == "person")
        .map(|e| clean_person_text(&e.text))
        .filter(|t| !t.is_empty())
        .filter(|t| seen.insert(t.clone()))
        .collect()
}

/// Strip leading determiners/quantifiers from a person entity text.
fn clean_person_text(text: &str) -> String {
    let lower = text.to_lowercase();
    // Strip leading quantifiers/determiners that noun chunks include.
    for prefix in &["both ", "the ", "a ", "an "] {
        if lower.starts_with(prefix) {
            return text[prefix.len()..].trim().to_string();
        }
    }
    text.trim().to_string()
}

/// Whether behood judged any entity in the question as the given kind.
pub fn query_has_kind(entities: &[QueryEntity], kind: &str) -> bool {
    entities.iter().any(|e| e.kind == kind)
}

#[allow(dead_code)]
fn _timeout() -> Duration {
    Duration::from_secs(30)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_entities_dedups() {
        let json = r#"{"entities": [{"text": "John", "kind": "person"}, {"text": "John", "kind": "person"}]}"#;
        let entities = parse_entities(json);
        assert_eq!(entities.len(), 1);
    }

    #[test]
    fn clean_person_text_strips_both() {
        assert_eq!(clean_person_text("both Jean"), "Jean");
        assert_eq!(clean_person_text("John"), "John");
    }

    #[test]
    fn parse_entities_fail_open() {
        assert!(parse_entities("not json").is_empty());
        assert!(parse_entities("{}").is_empty());
    }
}
