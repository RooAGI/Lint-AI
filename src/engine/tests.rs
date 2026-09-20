use super::*;

#[test]
fn normalize_heading_rules() {
    assert_eq!(normalize_heading("Related"), "related");
    assert_eq!(normalize_heading("Quick start"), "setup");
    assert_eq!(normalize_heading("Security notes"), "security");
    assert_eq!(normalize_heading("Routing & Sessions"), "routing");
    assert_eq!(normalize_heading(""), "unscoped");
}

#[test]
fn common_prefix() {
    let paths = vec![
        "docs/channels/a.md".to_string(),
        "docs/channels/b.md".to_string(),
    ];
    assert_eq!(common_dir_prefix(&paths).as_deref(), Some("docs/channels/"));
    assert_eq!(common_dir_prefix(&[]), None);
}

// White-box tests for the document-graph linting pipeline. These exercise
// crate-internal modules (graph/rules/report), so they live here rather
// than in tests/.
mod lint_pipeline {
    use crate::config::Config;
    use crate::engine::analyze_for_tests;
    use crate::graph::Graph;
    use crate::report::Report;
    use crate::rules::cross_refs::check_cross_refs;
    use crate::rules::orphan_pages::check_orphans;
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
        let cfg = Config {
            ignore_crossref_sections: vec!["related".to_string()],
            ..Default::default()
        };
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
        let cfg = Config {
            allowlist_concepts: vec!["gamma".to_string()],
            ..Default::default()
        };
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

        let output = analyze_for_tests(&graph, &cfg);
        assert!(output.contains("\"ignore_sections\""));
        assert!(output.contains("\"ignore_crossref_sections\""));
        assert!(output.contains("top concepts:"));
        assert!(output.contains("pages:"));
    }
}
