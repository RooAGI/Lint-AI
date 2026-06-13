use lint_ai::config::Config;
use lint_ai::graph::Graph;
use lint_ai::index::{DocRecord, MemoryIndex, Provenance};
use lint_ai::report::Report;
use lint_ai::rules::cross_refs::check_cross_refs;
use lint_ai::rules::orphan_pages::check_orphans;
use lint_ai::tier1::{RankedTerm, Tier1Entity};
use std::fs;
use std::path::PathBuf;

const MAX_BYTES: usize = 5_000_000;
const MAX_FILES: usize = 50_000;
const MAX_DEPTH: usize = 20;
const MAX_TOTAL_BYTES: usize = 50_000_000;

fn setup_fixture() -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "lint_ai_fixture_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("docs")).unwrap();

    fs::write(
        root.join("docs").join("alpha.md"),
        r#"
# Alpha

See Gamma.

## Related
Beta also appears here.
        "#,
    )
    .unwrap();

    fs::write(
        root.join("docs").join("beta.md"),
        r#"
# Beta

Mentions Alpha.
        "#,
    )
    .unwrap();

    fs::write(
        root.join("docs").join("gamma.md"),
        r#"
# Gamma

Code sample:
```
Alpha Beta
```
        "#,
    )
    .unwrap();

    root
}

#[test]
fn lint_reports_orphans_and_missing_links() {
    let root = setup_fixture();
    let graph = Graph::build(
        root.to_str().unwrap(),
        MAX_BYTES,
        MAX_FILES,
        MAX_DEPTH,
        MAX_TOTAL_BYTES,
    )
    .unwrap();
    let cfg = Config::default();
    let mut report = Report::new();

    check_orphans(&graph, &mut report);
    check_cross_refs(&graph, &mut report, &cfg);

    let text = report.to_string();
    assert!(text.contains("Orphan page: docs/gamma.md"));
    assert!(
        text.contains("Missing cross-ref in docs/alpha.md -> [[gamma]]"),
        "report:\n{}",
        text
    );
    assert!(
        text.contains("Missing cross-ref in docs/beta.md -> [[alpha]]"),
        "report:\n{}",
        text
    );
}

#[test]
fn ignore_related_section_for_crossrefs() {
    let root = setup_fixture();
    let graph = Graph::build(
        root.to_str().unwrap(),
        MAX_BYTES,
        MAX_FILES,
        MAX_DEPTH,
        MAX_TOTAL_BYTES,
    )
    .unwrap();
    let mut cfg = Config::default();
    cfg.ignore_crossref_sections = vec!["related".to_string()];
    let mut report = Report::new();

    check_cross_refs(&graph, &mut report, &cfg);
    let text = report.to_string();
    assert!(!text.contains("Missing cross-ref in docs/alpha.md -> [[beta]]"));
    assert!(text.contains("Missing cross-ref in docs/alpha.md -> [[gamma]]"));
}

#[test]
fn allowlist_limits_crossrefs() {
    let root = setup_fixture();
    let graph = Graph::build(
        root.to_str().unwrap(),
        MAX_BYTES,
        MAX_FILES,
        MAX_DEPTH,
        MAX_TOTAL_BYTES,
    )
    .unwrap();
    let mut cfg = Config::default();
    cfg.allowlist_concepts = vec!["gamma".to_string()];
    let mut report = Report::new();

    check_cross_refs(&graph, &mut report, &cfg);
    let text = report.to_string();
    assert!(!text.contains("Missing cross-ref in docs/alpha.md -> [[beta]]"));
    assert!(text.contains("Missing cross-ref in docs/alpha.md -> [[gamma]]"));
}

#[test]
fn analyze_suggests_config() {
    let root = setup_fixture();
    let graph = Graph::build(
        root.to_str().unwrap(),
        MAX_BYTES,
        MAX_FILES,
        MAX_DEPTH,
        MAX_TOTAL_BYTES,
    )
    .unwrap();
    let cfg = Config::default();

    let output = lint_ai::engine::analyze_for_tests(&graph, &cfg);
    assert!(output.contains("\"ignore_sections\""));
    assert!(output.contains("\"ignore_crossref_sections\""));
    assert!(output.contains("top concepts:"));
    assert!(output.contains("pages:"));
}

#[test]
fn query_baseline_still_works_without_semantic_match() {
    let index = MemoryIndex::from_records(vec![DocRecord {
        doc_id: "d1".to_string(),
        source: "d1.md".to_string(),
        content: "docker install on linux".to_string(),
        timestamp: None,
        doc_length: 24,
        author_agent: None,
        group_id: None,
        filters: std::collections::BTreeMap::new(),
        probable_topic: Some("Install".to_string()),
        doc_type_guess: None,
        headings: vec!["Install".to_string()],
        doc_links: vec![],
        temporal_terms: vec![],
        key_entities: vec![Tier1Entity {
            text: "docker".to_string(),
            label: "CONCEPT".to_string(),
            start: 0,
            end: 6,
            score: Some(1.0),
            source: "test".to_string(),
        }],
        important_terms: vec![RankedTerm {
            term: "install".to_string(),
            score: 2.0,
            source: "test".to_string(),
        }],
        section_chunks: vec![],
        embedding: None,
        top_claims: vec![],
        provenance: Provenance {
            source: "d1.md".to_string(),
            timestamp: None,
            ner_provider: "heuristic".to_string(),
            term_ranker: "test".to_string(),
            index_version: "test".to_string(),
        },
    }]);

    let results = index.query("docker", 10);
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, "d1");
}

#[test]
fn semantic_expansion_improves_recall_for_synonyms() {
    let index = MemoryIndex::from_records(vec![DocRecord {
        doc_id: "d2".to_string(),
        source: "d2.md".to_string(),
        content: "job role and occupation details".to_string(),
        timestamp: None,
        doc_length: 31,
        author_agent: None,
        group_id: None,
        filters: std::collections::BTreeMap::new(),
        probable_topic: Some("Occupation".to_string()),
        doc_type_guess: None,
        headings: vec!["Occupation".to_string()],
        doc_links: vec![],
        temporal_terms: vec![],
        key_entities: vec![],
        important_terms: vec![RankedTerm {
            term: "job".to_string(),
            score: 3.0,
            source: "test".to_string(),
        }],
        section_chunks: vec![],
        embedding: None,
        top_claims: vec![],
        provenance: Provenance {
            source: "d2.md".to_string(),
            timestamp: None,
            ner_provider: "heuristic".to_string(),
            term_ranker: "test".to_string(),
            index_version: "test".to_string(),
        },
    }]);

    // "occupation" expands to "job" in bundled lexical subsets.
    let results = index.query("occupation", 10);
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, "d2");
}
