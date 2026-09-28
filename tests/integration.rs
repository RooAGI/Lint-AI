// Public-API integration tests for the curated `lint_ai` surface.
// White-box tests for crate-internal modules live in `src/` unit tests.
use lint_ai::index::{DocRecord, MemoryIndex, Provenance};
use lint_ai::{RankedTerm, Tier1Entity};

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
        key_phrases: vec![],
        key_phrase_extraction_hash: String::new(),
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
        content_hash: String::new(),
    }]);

    let results = index.query("docker", 10);
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, "d1");
}

#[test]
fn semantic_policy_hides_automatically_superseded_documents_and_exposes_history() {
    let documents = vec![
        lint_ai::source::SourceDocument::with_stable_doc_id_from_source(
            "decisions/legacy.md".to_string(),
            "The platform team owns the control surface.".to_string(),
            "ownership decision".to_string(),
            None,
            vec![],
            vec![],
            Some("2026-01-01".to_string()),
            None,
        ),
        lint_ai::source::SourceDocument::with_stable_doc_id_from_source(
            "decisions/current.md".to_string(),
            "The reliability team owns the control surface.".to_string(),
            "ownership decision".to_string(),
            None,
            vec![],
            vec![],
            Some("2026-02-01".to_string()),
            None,
        ),
    ];
    let mut index =
        lint_ai::IndexStore::with_documents(lint_ai::PipelineOptions::default(), documents);

    let current = index.query("who owns the control surface", 10).unwrap();
    assert!(current
        .iter()
        .any(|result| result.source == "decisions/current.md"));
    assert!(current
        .iter()
        .all(|result| result.source != "decisions/legacy.md"));

    let history = index
        .query("what changed about who owns the control surface", 10)
        .unwrap();
    let legacy = history
        .iter()
        .find(|result| result.source == "decisions/legacy.md")
        .expect("historical query should retain the superseded source");
    assert_eq!(
        legacy.semantic_status,
        Some(lint_ai::SemanticStatus::Historical)
    );
}

#[test]
fn reveals_bug_superseded_claim_hides_unrelated_current_content() {
    let documents = vec![
        lint_ai::source::SourceDocument::with_stable_doc_id_from_source(
            "decisions/architecture.md".to_string(),
            concat!(
                "The platform team owns the control surface.\n",
                "The deployment strategy uses blue-green releases."
            )
            .to_string(),
            "architecture decisions".to_string(),
            None,
            vec![],
            vec![],
            Some("2026-01-01".to_string()),
            None,
        ),
        lint_ai::source::SourceDocument::with_stable_doc_id_from_source(
            "decisions/ownership-update.md".to_string(),
            "The reliability team owns the control surface.".to_string(),
            "ownership decision".to_string(),
            None,
            vec![],
            vec![],
            Some("2026-02-01".to_string()),
            None,
        ),
    ];
    let mut index =
        lint_ai::IndexStore::with_documents(lint_ai::PipelineOptions::default(), documents);

    let results = index.query("blue-green deployment strategy", 10).unwrap();

    let architecture = results
        .iter()
        .find(|result| result.source == "decisions/architecture.md")
        .unwrap_or_else(|| {
            panic!(
                "replacing the ownership claim must not hide the still-current deployment guidance: {results:?}"
            )
        });
    assert_eq!(
        architecture.semantic_status,
        Some(lint_ai::SemanticStatus::Conflicted),
        "partial inferred supersession should remain visible as a conflict"
    );
}

#[test]
fn reveals_bug_operational_before_query_exposes_superseded_guidance() {
    let old = lint_ai::source::SourceDocument::with_stable_doc_id_from_source(
        "runbooks/legacy-deployment.md".to_string(),
        "Before deployment, operators must run the legacy smoke tests.".to_string(),
        "legacy deployment procedure".to_string(),
        None,
        vec![],
        vec![],
        Some("2026-01-01".to_string()),
        None,
    );
    let mut current = lint_ai::source::SourceDocument::with_stable_doc_id_from_source(
        "runbooks/current-deployment.md".to_string(),
        "Before deployment, operators must run the current safety checks.".to_string(),
        "current deployment procedure".to_string(),
        None,
        vec![],
        vec![],
        Some("2026-02-01".to_string()),
        None,
    );
    current
        .filters
        .insert("supersedes_id".to_string(), old.doc_id.clone());
    let legacy_id = old.doc_id.clone();
    let current_id = current.doc_id.clone();
    let mut index = lint_ai::IndexStore::with_documents(
        lint_ai::PipelineOptions::default(),
        vec![old, current],
    );

    let results = index
        .query("what must happen before deployment", 10)
        .unwrap();

    assert!(
        results.iter().any(|result| result.doc_id == current_id),
        "current deployment guidance should be returned: {results:?}"
    );
    assert!(
        results.iter().all(|result| result.doc_id != legacy_id),
        "an operational use of 'before' must not expose superseded guidance: {results:?}"
    );
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
        key_phrases: vec![],
        key_phrase_extraction_hash: String::new(),
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
        content_hash: String::new(),
    }]);

    // "occupation" expands to "job" in bundled lexical subsets.
    let results = index.query("occupation", 10);
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, "d2");
}
