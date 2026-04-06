use lint_ai::config::Config;
use lint_ai::graph::Graph;
use lint_ai::report::Report;
use lint_ai::rules::cross_refs::check_cross_refs;
use lint_ai::rules::orphan_pages::check_orphans;
use std::fs;
use std::path::PathBuf;

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
    let graph = Graph::build(root.to_str().unwrap(), 5_000_000, 50_000, 20, 50_000_000).unwrap();
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
    let graph = Graph::build(root.to_str().unwrap(), 5_000_000, 50_000, 20, 50_000_000).unwrap();
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
    let graph = Graph::build(root.to_str().unwrap(), 5_000_000, 50_000, 20, 50_000_000).unwrap();
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
    let graph = Graph::build(root.to_str().unwrap(), 5_000_000, 50_000, 20, 50_000_000).unwrap();
    let cfg = Config::default();

    let output = lint_ai::engine::analyze_for_tests(&graph, &cfg);
    assert!(output.contains("\"ignore_sections\""));
    assert!(output.contains("\"ignore_crossref_sections\""));
    assert!(output.contains("top concepts:"));
    assert!(output.contains("pages:"));
}
