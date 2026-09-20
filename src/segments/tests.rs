use super::*;
use crate::index::{
    DocRecord, GlobalBm25Statistics, MemoryIndex, Provenance, ScoreBreakdown, SearchResult,
    TemporalQueryContext, TemporalQueryHint,
};
use crate::tier1::{RankedTerm, Tier1Entity};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[test]
fn segment_candidate_limit_oversamples_without_overflow() {
    assert_eq!(segment_candidate_limit(0), 0);
    assert_eq!(segment_candidate_limit(3), 6);
    assert_eq!(segment_candidate_limit(usize::MAX), usize::MAX);
}

fn record(doc_id: &str, group_id: &str, content: &str, terms: &[&str]) -> DocRecord {
    DocRecord {
        doc_id: doc_id.to_string(),
        source: format!("memory://{doc_id}"),
        content: content.to_string(),
        timestamp: None,
        doc_length: content.len(),
        author_agent: None,
        group_id: Some(group_id.to_string()),
        filters: std::collections::BTreeMap::new(),
        probable_topic: terms.first().map(|term| (*term).to_string()),
        doc_type_guess: None,
        headings: vec![group_id.to_string()],
        doc_links: vec![],
        temporal_terms: vec![],
        key_entities: terms
            .iter()
            .map(|term| Tier1Entity {
                text: (*term).to_string(),
                label: "KEY".to_string(),
                start: 0,
                end: term.len(),
                score: Some(1.0),
                source: "test".to_string(),
            })
            .collect(),
        important_terms: terms
            .iter()
            .map(|term| RankedTerm {
                term: (*term).to_string(),
                score: 1.0,
                source: "test".to_string(),
            })
            .collect(),
        section_chunks: vec![],
        embedding: None,
        top_claims: vec![],
        provenance: Provenance {
            source: "test".to_string(),
            timestamp: None,
            ner_provider: "test".to_string(),
            term_ranker: "test".to_string(),
            index_version: "test".to_string(),
        },
        content_hash: String::new(),
    }
}

fn search_result(doc_id: &str, score: f32) -> SearchResult {
    SearchResult {
        doc_id: doc_id.to_string(),
        source: format!("memory://{doc_id}"),
        group_id: None,
        score,
        score_breakdown: ScoreBreakdown::default(),
        matched_entities: Vec::new(),
        matched_terms: Vec::new(),
        probable_topic: None,
        doc_type_guess: None,
        semantic_status: None,
        superseded_by: None,
        relation_confidence: None,
        relation_evidence: vec![],
    }
}

/// Deterministic projection of a routing catalog: sorted so HashMap
/// iteration order cannot affect the comparison.
fn normalized_catalog(catalog: &SegmentCatalog) -> String {
    let mut segment_ids = catalog.ordered_segment_ids.clone();
    segment_ids.sort();
    let mut parts = Vec::new();
    for segment_id in segment_ids {
        let summary = &catalog.summaries[&segment_id];
        let mut terms: Vec<(String, u32)> = summary
            .terms
            .iter()
            .chain(summary.entities.iter())
            .chain(summary.topics.iter())
            .chain(summary.local_memory.iter())
            .map(|(term, weight)| (term.clone(), weight.to_bits()))
            .collect();
        terms.sort();
        let profile = &catalog.connection_profiles[&segment_id];
        let mut evidence: Vec<String> = profile
            .people
            .iter()
            .chain(profile.subjects.iter())
            .chain(profile.times.iter())
            .chain(profile.actions.iter())
            .chain(profile.objects.iter())
            .cloned()
            .collect();
        evidence.sort();
        parts.push(format!("{segment_id}|{terms:?}|{evidence:?}"));
    }
    let mut term_map: Vec<(String, Vec<String>)> = catalog
        .term_to_segments
        .iter()
        .map(|(term, ids)| {
            let mut sorted_ids = ids.clone();
            sorted_ids.sort();
            (term.clone(), sorted_ids)
        })
        .collect();
    term_map.sort();
    parts.push(format!("{term_map:?}"));
    parts.join("\n")
}

#[test]
fn refresh_incremental_reuses_only_untouched_segments() {
    let records = vec![
        record("keep-1", "keep", "keep alpha", &["alpha"]),
        record("keep-2", "keep", "keep beta", &["beta"]),
        record("grow-1", "grow", "grow gamma", &["gamma"]),
        record("drop-1", "drop", "drop delta", &["delta"]),
    ];
    let previous = SegmentedMemoryIndex::from_records_by_group_id(&records);
    let previous_index = |id: &str| {
        Arc::as_ptr(
            &previous
                .segments
                .iter()
                .find(|segment| segment.segment_id == id)
                .unwrap()
                .index,
        )
    };

    // "grow" gains a document, "drop" loses its only document, and one
    // document of "keep" is re-processed: only an untouched "keep" copy
    // would be reusable, so nothing is reused here.
    let mut next_records: Vec<DocRecord> = records
        .into_iter()
        .filter(|record| record.doc_id != "drop-1")
        .collect();
    next_records.push(record("grow-2", "grow", "grow epsilon", &["epsilon"]));
    let next_map: HashMap<String, DocRecord> = next_records
        .iter()
        .map(|record| (record.doc_id.clone(), record.clone()))
        .collect();
    let reprocessed: HashSet<String> = ["keep-1".to_string()].into_iter().collect();
    let next = SegmentedMemoryIndex::refresh_incremental(&previous, &next_map, &reprocessed, 1)
        .expect("incremental refresh should succeed");

    let next_index = |snapshot: &SegmentedMemoryIndex, id: &str| {
        Arc::as_ptr(
            &snapshot
                .segments
                .iter()
                .find(|segment| segment.segment_id == id)
                .unwrap()
                .index,
        )
    };
    assert_eq!(next.segments.len(), 2);
    assert!(next
        .segments
        .iter()
        .all(|segment| segment.segment_id != "drop"));
    // Every surviving segment changed, so every index is rebuilt.
    assert_ne!(next_index(&next, "keep"), previous_index("keep"));
    assert_ne!(next_index(&next, "grow"), previous_index("grow"));

    // A refresh with no changes reuses every segment's index by pointer.
    let idle = SegmentedMemoryIndex::refresh_incremental(&next, &next_map, &HashSet::new(), 2)
        .expect("idle incremental refresh should succeed");
    assert_eq!(idle.segments.len(), 2);
    assert!(Arc::ptr_eq(
        &idle
            .segments
            .iter()
            .find(|segment| segment.segment_id == "keep")
            .unwrap()
            .index,
        &next
            .segments
            .iter()
            .find(|segment| segment.segment_id == "keep")
            .unwrap()
            .index
    ));
    assert!(Arc::ptr_eq(
        &idle
            .segments
            .iter()
            .find(|segment| segment.segment_id == "grow")
            .unwrap()
            .index,
        &next
            .segments
            .iter()
            .find(|segment| segment.segment_id == "grow")
            .unwrap()
            .index
    ));

    // Differential: the incremental catalog must equal a clean full
    // rebuild's catalog, or query routing would diverge.
    let rebuilt = SegmentedMemoryIndex::from_records_by_group_id(&next_records);
    assert_eq!(
        normalized_catalog(&next.catalog),
        normalized_catalog(&rebuilt.catalog)
    );
}

/// Differential oracle for incremental refresh: the incremental result must
/// be indistinguishable from a full rebuild — same segment membership and
/// same catalog (including postings order of every derived map). Query
/// results are intentionally not compared (see below).
fn assert_incremental_matches_full_rebuild(
    base: Vec<DocRecord>,
    mutate: impl FnOnce(Vec<DocRecord>) -> (Vec<DocRecord>, HashSet<String>),
) {
    let previous = SegmentedMemoryIndex::from_records_by_group_id_with_generation(&base, 7);
    let (next_records, reprocessed) = mutate(base);
    let next_map: HashMap<String, DocRecord> = next_records
        .iter()
        .map(|record| (record.doc_id.clone(), record.clone()))
        .collect();
    let next = SegmentedMemoryIndex::refresh_incremental(&previous, &next_map, &reprocessed, 8)
        .expect("incremental refresh should succeed");
    let oracle = SegmentedMemoryIndex::from_records_by_group_id_with_generation(&next_records, 8);

    assert_eq!(
        format!("{:?}", next.manifest()),
        format!("{:?}", oracle.manifest()),
        "segment membership diverged"
    );
    assert_eq!(
        normalized_catalog(&next.catalog),
        normalized_catalog(&oracle.catalog),
        "catalog summaries/profiles diverged"
    );
    assert_eq!(
        next.catalog.derived_maps_snapshot(),
        oracle.catalog.derived_maps_snapshot(),
        "derived routing maps diverged"
    );

    // Query results are intentionally NOT compared here: the query path has
    // pre-existing nondeterministic top-k tie-breaking (two identical full
    // rebuilds can select different tied documents; see
    // pipeline::tests::incremental_refresh_matches_full_rebuild). The
    // deterministic routing inputs — segment manifest, catalog summaries,
    // connection profiles, and every derived routing map bit-for-bit — are
    // asserted above, which is the complete input surface this incremental
    // refresh is responsible for.
}

fn differential_base_records() -> Vec<DocRecord> {
    vec![
        record("a-1", "alpha", "alpha one two", &["alpha"]),
        record("a-2", "alpha", "alpha three four", &["bravo"]),
        record("b-1", "beta", "beta one two", &["charlie"]),
        record("b-2", "beta", "beta three four", &["delta"]),
        record("c-1", "gamma", "gamma one two", &["echo"]),
        record("c-2", "gamma", "gamma three four", &["foxtrot"]),
    ]
}

#[test]
fn incremental_refresh_matches_full_rebuild_across_mutations() {
    // Insert into an existing segment.
    assert_incremental_matches_full_rebuild(differential_base_records(), |mut records| {
        records.push(record("a-3", "alpha", "alpha five six", &["golf"]));
        (records, HashSet::new())
    });

    // Insert creating a brand-new segment.
    assert_incremental_matches_full_rebuild(differential_base_records(), |mut records| {
        records.push(record("d-1", "delta-seg", "hotel india", &["hotel"]));
        (records, HashSet::new())
    });

    // Edit within a segment (content changes, membership does not).
    assert_incremental_matches_full_rebuild(differential_base_records(), |mut records| {
        let edited = record("a-1", "alpha", "alpha one two JULIET", &["juliet"]);
        let slot = records
            .iter_mut()
            .find(|record| record.doc_id == "a-1")
            .unwrap();
        *slot = edited;
        (records, HashSet::from(["a-1".to_string()]))
    });

    // Document moves between groups.
    assert_incremental_matches_full_rebuild(differential_base_records(), |mut records| {
        let moved = record("b-1", "alpha", "beta one two", &["charlie"]);
        let slot = records
            .iter_mut()
            .find(|record| record.doc_id == "b-1")
            .unwrap();
        *slot = moved;
        (records, HashSet::from(["b-1".to_string()]))
    });

    // Deletion that keeps the segment alive.
    assert_incremental_matches_full_rebuild(differential_base_records(), |mut records| {
        records.retain(|record| record.doc_id != "b-2");
        (records, HashSet::new())
    });

    // Deletion removing the segment's last document.
    assert_incremental_matches_full_rebuild(differential_base_records(), |mut records| {
        records.retain(|record| !record.doc_id.starts_with("c-"));
        (records, HashSet::new())
    });

    // No-op refresh: identical records, nothing reprocessed.
    assert_incremental_matches_full_rebuild(differential_base_records(), |records| {
        (records, HashSet::new())
    });

    // Multiple affected segments at once: insert, edit, move, and delete.
    assert_incremental_matches_full_rebuild(differential_base_records(), |mut records| {
        records.push(record("a-3", "alpha", "alpha five six", &["golf"]));
        records.retain(|record| record.doc_id != "b-2");
        let moved = record("c-1", "beta", "gamma one two", &["echo"]);
        *records
            .iter_mut()
            .find(|record| record.doc_id == "c-1")
            .unwrap() = moved;
        let edited = record("a-2", "alpha", "alpha three four KILO", &["kilo"]);
        *records
            .iter_mut()
            .find(|record| record.doc_id == "a-2")
            .unwrap() = edited;
        (
            records,
            HashSet::from(["c-1".to_string(), "a-2".to_string()]),
        )
    });
}

#[test]
fn raw_constructor_rejects_duplicate_segment_ids() {
    let segments = vec![
        build_memory_index_segment(
            "duplicate".into(),
            vec![record("doc-a", "a", "alpha", &["alpha"])],
        ),
        build_memory_index_segment(
            "duplicate".into(),
            vec![record("doc-b", "b", "beta", &["beta"])],
        ),
    ];
    let error = SegmentedMemoryIndex::from_segments(segments).err().unwrap();
    assert!(error.contains("duplicate segment id"));
}

#[test]
fn raw_constructor_rejects_documents_assigned_to_multiple_segments() {
    let segments = vec![
        build_memory_index_segment(
            "segment-a".into(),
            vec![record("shared", "a", "alpha", &["alpha"])],
        ),
        build_memory_index_segment(
            "segment-b".into(),
            vec![record("shared", "b", "beta", &["beta"])],
        ),
    ];
    let error = SegmentedMemoryIndex::from_segments(segments).err().unwrap();
    assert!(error.contains("multiple segments"));
}

#[test]
fn raw_constructor_rejects_empty_segments() {
    let segment = build_memory_index_segment("empty".into(), Vec::new());
    let error = SegmentedMemoryIndex::from_segments(vec![segment])
        .err()
        .unwrap();
    assert!(error.contains("segment is empty"));
}

#[test]
fn raw_constructor_rejects_doc_id_index_mismatch() {
    let mut segment = build_memory_index_segment(
        "segment-a".into(),
        vec![record("doc-a", "a", "alpha", &["alpha"])],
    );
    segment.doc_ids = vec!["different-doc".into()];
    let error = SegmentedMemoryIndex::from_segments(vec![segment])
        .err()
        .unwrap();
    assert!(error.contains("do not match"));
}

#[test]
fn bounded_candidate_strategies_do_not_route_the_full_corpus() {
    let records = (0..100)
        .map(|index| {
            record(
                &format!("doc-{index:03}"),
                &format!("session-{index:03}"),
                "Common bought widget",
                &["common", "bought", "widget"],
            )
        })
        .collect::<Vec<_>>();
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

    for strategy in [
        SegmentRoutingStrategy::LocalDistinctiveness,
        SegmentRoutingStrategy::CoverageLocalDistinctiveness,
        SegmentRoutingStrategy::TeamCoverageLocalDistinctiveness,
        SegmentRoutingStrategy::CoverageTeamSelection,
        SegmentRoutingStrategy::TypedEvidence,
    ] {
        let routes = segmented.route_with_strategy("Common bought widget", strategy);
        assert!(!routes.is_empty(), "{strategy:?} returned no candidates");
        assert!(
            routes.len() <= ROUTING_CANDIDATE_POOL_LIMIT,
            "{strategy:?} routed {} candidates",
            routes.len()
        );
        assert!(routes.len() < segmented.len());
    }

    let exact = segmented.query_all_segments_with_diagnostics("Common bought widget", 5);
    assert_eq!(exact.diagnostics.queried_segment_count, segmented.len());
}

#[test]
fn routes_and_queries_one_group_segment() {
    let records = vec![
        record(
            "doc-a",
            "session-a",
            "docker install guide for linux",
            &["docker", "install", "linux"],
        ),
        record(
            "doc-b",
            "session-b",
            "kubernetes cluster operations",
            &["kubernetes", "cluster"],
        ),
    ];
    let segments = build_segments_by_group_id(&records);

    assert_eq!(segments.len(), 2);
    let routes = route_segments("docker install", &segments);
    assert_eq!(routes[0].segment_id, "session-a");

    let results = query_top_segment("docker install", 5, &segments);
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, "doc-a");
}

#[test]
fn queries_top_n_segments_and_merges_results() {
    let records = vec![
        record(
            "doc-a",
            "session-a",
            "docker install guide for linux",
            &["docker", "install", "linux"],
        ),
        record(
            "doc-b",
            "session-b",
            "docker compose troubleshooting",
            &["docker", "compose", "troubleshooting"],
        ),
        record(
            "doc-c",
            "session-c",
            "kubernetes cluster operations",
            &["kubernetes", "cluster"],
        ),
    ];
    let segments = build_segments_by_group_id(&records);

    let one_segment = query_top_segments("docker", 5, &segments, 1);
    assert_eq!(one_segment.len(), 1);

    let two_segments = query_top_segments("docker", 5, &segments, 2);
    let doc_ids = two_segments
        .iter()
        .map(|result| result.doc_id.as_str())
        .collect::<HashSet<_>>();
    assert_eq!(two_segments.len(), 2);
    assert!(doc_ids.contains("doc-a"));
    assert!(doc_ids.contains("doc-b"));
    assert!(!doc_ids.contains("doc-c"));
}

#[test]
fn segmented_memory_index_returns_diagnostics() {
    let records = vec![
        record(
            "doc-a",
            "session-a",
            "docker install guide for linux",
            &["docker", "install", "linux"],
        ),
        record(
            "doc-b",
            "session-b",
            "kubernetes cluster operations",
            &["kubernetes", "cluster"],
        ),
    ];
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

    assert_eq!(segmented.len(), 2);
    let routes = segmented.route("docker install");
    assert_eq!(routes[0].segment_id, "session-a");

    let output = segmented.query_with_diagnostics("docker install", 5, 1);
    assert_eq!(output.diagnostics.queried_segment_count, 1);
    let completeness = output
        .diagnostics
        .shard_completeness
        .as_ref()
        .expect("routed shard queries report completeness");
    assert!(completeness.is_complete());
    assert_eq!(completeness.expected_segments, vec!["session-a"]);
    assert_eq!(completeness.successful_segments, vec!["session-a"]);
    assert_eq!(
        output.diagnostics.selected_segments[0].segment_id,
        "session-a"
    );
    assert_eq!(output.diagnostics.final_result_count, output.results.len());
    assert!(output.diagnostics.merged_result_count >= output.diagnostics.final_result_count);
    assert_eq!(
        output
            .diagnostics
            .per_segment_result_counts
            .get("session-a")
            .copied(),
        Some(output.results.len())
    );
    assert_eq!(
        output.diagnostics.query_terms,
        vec!["docker".to_string(), "instal".to_string()]
    );
    assert_eq!(
        output.diagnostics.covered_query_terms,
        vec!["docker".to_string(), "instal".to_string()]
    );
    assert!(output.diagnostics.uncovered_query_terms.is_empty());
    assert_eq!(
        output.diagnostics.segments_with_results,
        vec!["session-a".to_string()]
    );
}

#[test]
fn diagnostics_report_query_coverage_across_more_segments() {
    let records = vec![
        record("doc-a", "session-a", "docker install guide", &["docker"]),
        record(
            "doc-b",
            "session-b",
            "compose troubleshooting guide",
            &["compose"],
        ),
        record(
            "doc-c",
            "session-c",
            "kubernetes cluster operations",
            &["kubernetes"],
        ),
    ];
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

    let top_one = segmented.query_with_diagnostics("docker compose", 5, 1);
    assert_eq!(
        top_one.diagnostics.covered_query_terms,
        vec!["docker".to_string()]
    );
    assert_eq!(
        top_one.diagnostics.uncovered_query_terms,
        vec!["compos".to_string()]
    );
    assert_eq!(
        top_one.diagnostics.segments_with_results,
        vec!["session-a".to_string()]
    );

    let top_two = segmented.query_with_diagnostics("docker compose", 5, 2);
    assert_eq!(
        top_two.diagnostics.covered_query_terms,
        vec!["compos".to_string(), "docker".to_string()]
    );
    assert!(top_two.diagnostics.uncovered_query_terms.is_empty());
    assert_eq!(
        top_two.diagnostics.segments_with_results,
        vec!["session-a".to_string(), "session-b".to_string()]
    );
}

#[test]
fn kl_router_prefers_closest_segment_distribution() {
    let records = vec![
        record(
            "doc-a",
            "session-a",
            "docker compose install troubleshooting",
            &["docker", "compose", "install"],
        ),
        record(
            "doc-b",
            "session-b",
            "kubernetes cluster node scheduling",
            &["kubernetes", "cluster", "node"],
        ),
    ];
    let segments = build_segments_by_group_id(&records);

    let routes = route_segments_with_strategy(
        "docker compose",
        &segments,
        SegmentRoutingStrategy::KlDivergence,
    );

    assert_eq!(routes[0].segment_id, "session-a");
    assert!(routes[0].score > routes[1].score);
}

#[test]
fn local_router_prefers_segment_specific_differentiators() {
    let records = vec![
        record(
            "doc-a",
            "session-a",
            "general study planning notes",
            &["study", "notes", "planning"],
        ),
        record(
            "doc-b",
            "session-b",
            "undergraduate graduate GPA transcript",
            &["study", "undergraduate", "graduate", "GPA"],
        ),
        record(
            "doc-c",
            "session-c",
            "study schedule and reading list",
            &["study", "schedule", "reading"],
        ),
    ];
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

    let output = segmented.query_with_diagnostics_and_strategy(
        "what is my average GPA from undergraduate and graduate studies",
        5,
        1,
        SegmentRoutingStrategy::LocalDistinctiveness,
    );

    assert_eq!(
        output.diagnostics.selected_segments[0].segment_id,
        "session-b"
    );
    let local_terms = output.diagnostics.local_evidence[0]
        .differentiators
        .iter()
        .map(|differentiator| differentiator.term.as_str())
        .collect::<HashSet<_>>();
    assert!(local_terms.contains("gpa"));
    assert!(local_terms.contains("undergradu"));
    assert!(local_terms.contains("graduat"));
    assert!(!output.diagnostics.query_terms.contains(&"what".to_string()));
    assert!(!output.diagnostics.query_terms.contains(&"my".to_string()));
}

#[test]
fn coverage_local_router_prefers_rare_term_coverage() {
    let records = vec![
        record(
            "doc-a",
            "session-a",
            "GPA GPA GPA GPA GPA academic score notes",
            &["GPA"],
        ),
        record(
            "doc-b",
            "session-b",
            "undergraduate graduate transcript academic record",
            &["undergraduate", "graduate", "transcript"],
        ),
        record(
            "doc-c",
            "session-c",
            "general academic planning and study notes",
            &["academic", "planning"],
        ),
    ];
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

    let output = segmented.query_with_diagnostics_and_strategy(
        "GPA undergraduate graduate transcript",
        5,
        1,
        SegmentRoutingStrategy::CoverageLocalDistinctiveness,
    );

    assert_eq!(
        output.diagnostics.selected_segments[0].segment_id,
        "session-b"
    );
    let local_terms = output.diagnostics.local_evidence[0]
        .differentiators
        .iter()
        .map(|differentiator| differentiator.term.as_str())
        .collect::<HashSet<_>>();
    assert!(local_terms.contains("undergradu"));
    assert!(local_terms.contains("graduat"));
    assert!(local_terms.contains("transcript"));
}

#[test]
fn team_coverage_router_prefers_complementary_segments() {
    let records = vec![
        record(
            "doc-a",
            "session-a",
            "GPA undergraduate academic record",
            &["GPA", "undergraduate"],
        ),
        record(
            "doc-b",
            "session-b",
            "GPA undergraduate admission note",
            &["GPA", "undergraduate"],
        ),
        record(
            "doc-c",
            "session-c",
            "graduate jewelry repair appointment",
            &["graduate", "jewelry", "appointment"],
        ),
    ];
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

    let output = segmented.query_with_diagnostics_and_strategy(
        "GPA undergraduate graduate jewelry appointment",
        5,
        2,
        SegmentRoutingStrategy::TeamCoverageLocalDistinctiveness,
    );
    let selected = output
        .diagnostics
        .selected_segments
        .iter()
        .map(|route| route.segment_id.as_str())
        .collect::<HashSet<_>>();

    assert_eq!(selected.len(), 2);
    assert!(selected.contains("session-c"));
    assert!(selected.contains("session-a") || selected.contains("session-b"));
    assert!(output
        .diagnostics
        .covered_query_terms
        .contains(&"appoint".to_string()));
    assert!(output
        .diagnostics
        .covered_query_terms
        .contains(&"gpa".to_string()));
    assert!(output
        .diagnostics
        .covered_query_terms
        .contains(&"jewelri".to_string()));
}

#[test]
fn adaptive_segment_enrichment_expands_when_query_coverage_is_low() {
    let records = vec![
        record(
            "doc-a",
            "session-a",
            "GPA undergraduate academic record",
            &["GPA", "undergraduate"],
        ),
        record(
            "doc-b",
            "session-b",
            "graduate jewelry repair appointment",
            &["graduate", "jewelry", "appointment"],
        ),
    ];
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

    let (output, _) = segmented
        .query_with_adaptive_segment_enrichment_temporal_context_and_strategy(
            "GPA undergraduate graduate jewelry appointment",
            5,
            1,
            2,
            SegmentRoutingStrategy::CoverageLocalDistinctiveness,
            TemporalQueryContext::default(),
            None,
        );
    let selected = output
        .diagnostics
        .selected_segments
        .iter()
        .map(|route| route.segment_id.as_str())
        .collect::<HashSet<_>>();

    assert_eq!(selected.len(), 2);
    assert!(selected.contains("session-a"));
    assert!(selected.contains("session-b"));
    assert!(output.diagnostics.uncovered_query_terms.is_empty());
}

#[test]
fn adaptive_segment_enrichment_stays_at_base_when_coverage_is_sufficient() {
    let records = vec![record(
        "doc-a",
        "session-a",
        "GPA undergraduate academic record",
        &["GPA", "undergraduate"],
    )];
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

    let (output, _) = segmented
        .query_with_adaptive_segment_enrichment_temporal_context_and_strategy(
            "GPA undergraduate",
            5,
            1,
            8,
            SegmentRoutingStrategy::CoverageLocalDistinctiveness,
            TemporalQueryContext::default(),
            None,
        );

    assert_eq!(output.diagnostics.selected_segments.len(), 1);
    assert!(output.diagnostics.uncovered_query_terms.is_empty());
}

#[test]
fn route_aware_rerank_prefers_new_evidence_coverage() {
    let candidates = vec![
        RouteAwareCandidate {
            result: search_result("doc-common", 1.0),
            segment_id: "session-common".to_string(),
            base_score: 1.0,
            evidence_terms: HashSet::from(["appointment".to_string()]),
        },
        RouteAwareCandidate {
            result: search_result("doc-specific", 0.9),
            segment_id: "session-specific".to_string(),
            base_score: 0.9,
            evidence_terms: HashSet::from(["gpa".to_string(), "jewelri".to_string()]),
        },
    ];

    let selected = select_route_aware_top_k(candidates, 1);

    assert_eq!(selected[0].doc_id, "doc-specific");
    assert!(selected[0].score > 0.9);
}

#[test]
fn session_aggregation_combines_segment_evidence_by_group() {
    let mut session_a_primary = search_result("session-a::turn0", 1.0);
    session_a_primary.group_id = Some("session-a".to_string());
    session_a_primary.matched_terms = vec!["gpa".to_string()];
    let mut session_a_support = search_result("session-a::turn1", 0.8);
    session_a_support.group_id = Some("session-a".to_string());
    session_a_support.matched_entities = vec!["jewelry".to_string()];
    let mut session_b = search_result("session-b::turn0", 1.1);
    session_b.group_id = Some("session-b".to_string());

    let aggregated =
        aggregate_segment_results_by_session(vec![session_b, session_a_support, session_a_primary]);

    assert_eq!(aggregated[0].group_id.as_deref(), Some("session-a"));
    assert!(aggregated[0].score > 1.1);
}

#[test]
fn local_memory_router_uses_rolling_content_window() {
    let records = vec![
        record(
            "doc-a",
            "session-a",
            "the day before my doctor appointment I went to bed early",
            &["appointment"],
        ),
        record(
            "doc-b",
            "session-b",
            "appointment calendar reminder and scheduling notes",
            &["appointment"],
        ),
    ];
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

    let output = segmented.query_with_diagnostics_and_strategy(
        "what time did I go to bed on the day before the doctor appointment",
        5,
        1,
        SegmentRoutingStrategy::LocalDistinctiveness,
    );

    assert_eq!(
        output.diagnostics.selected_segments[0].segment_id,
        "session-a"
    );
    let local_evidence = &output.diagnostics.local_evidence[0].differentiators;
    assert!(local_evidence
        .iter()
        .any(|differentiator| differentiator.term == "bed"
            && differentiator
                .evidence_types
                .contains(&"local_memory".to_string())));
    assert!(local_evidence
        .iter()
        .any(|differentiator| differentiator.term == "doctor"
            && differentiator
                .evidence_types
                .contains(&"local_memory".to_string())));
}

#[test]
fn segment_enriched_query_uses_each_selected_segments_local_context() {
    let records = vec![
        record(
            "doc-a",
            "session-a",
            "doctor appointment clinic notes before going to bed early",
            &["appointment", "doctor", "clinic"],
        ),
        record(
            "doc-b",
            "session-b",
            "appointment calendar reminder for product planning meeting",
            &["appointment", "calendar", "meeting"],
        ),
    ];
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

    let (output, enrichment) = segmented.query_with_segment_enrichment_and_strategy(
        "appointment",
        5,
        2,
        SegmentRoutingStrategy::LocalDistinctiveness,
    );

    assert_eq!(output.diagnostics.queried_segment_count, 2);
    assert_eq!(enrichment.selected_segments.len(), 2);
    let by_segment = enrichment
        .selected_segments
        .iter()
        .map(|diagnostic| (diagnostic.segment_id.as_str(), diagnostic))
        .collect::<HashMap<_, _>>();
    let session_a = by_segment.get("session-a").expect("session-a enrichment");
    let session_b = by_segment.get("session-b").expect("session-b enrichment");

    assert_ne!(session_a.enriched_query, session_b.enriched_query);
    assert!(session_a.added_terms.iter().any(|term| term == "doctor"));
    assert!(session_b.added_terms.iter().any(|term| term == "calendar"));
    assert!(enrichment.average_added_terms > 0.0);
    assert!(output
        .diagnostics
        .local_evidence
        .iter()
        .any(|evidence| !evidence.differentiators.is_empty()));
}

#[test]
fn segment_enriched_query_reports_temporal_local_context() {
    let mut older = record(
        "doc-a",
        "session-a",
        "doctor appointment notes from last week",
        &["appointment", "doctor"],
    );
    older.timestamp = Some("2024-05-01".to_string());
    older.temporal_terms = vec!["date 2024-05-01".to_string(), "last week".to_string()];
    let mut current = record(
        "doc-b",
        "session-b",
        "doctor appointment notes from today",
        &["appointment", "doctor"],
    );
    current.timestamp = Some("2024-05-10".to_string());
    current.temporal_terms = vec!["date 2024-05-10".to_string(), "today".to_string()];
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&[older, current]);

    let (_output, enrichment) = segmented
        .query_with_segment_enrichment_temporal_context_and_strategy(
            "what happened at the appointment today",
            5,
            2,
            SegmentRoutingStrategy::LocalDistinctiveness,
            TemporalQueryContext {
                ends_at: Some("2024-05-10"),
                time_hint: Some(TemporalQueryHint::Present),
                has_explicit_temporal: true,
                ..TemporalQueryContext::default()
            },
        );

    let current = enrichment
        .selected_segments
        .iter()
        .find(|diagnostic| diagnostic.segment_id == "session-b")
        .expect("session-b enrichment");
    assert!(current.temporal_signal);
    assert!(!current.temporal_added_terms.is_empty());
    assert!(current
        .temporal_evidence
        .iter()
        .any(|evidence| evidence == "temporal_near_query_date"));
}

#[test]
fn temporal_path_enrichment_expands_to_nearby_before_segment() {
    let mut anchor = record(
        "doc-anchor",
        "session-anchor",
        "doctor appointment happened at the clinic",
        &["doctor", "appointment", "clinic"],
    );
    anchor.timestamp = Some("2024-05-10".to_string());
    anchor.temporal_terms = vec!["date 2024-05-10".to_string()];
    let mut before = record(
        "doc-before",
        "session-before",
        "bought toothpaste before the visit",
        &["toothpaste", "visit"],
    );
    before.timestamp = Some("2024-05-09".to_string());
    before.temporal_terms = vec!["date 2024-05-09".to_string(), "before".to_string()];
    let mut unrelated = record(
        "doc-unrelated",
        "session-unrelated",
        "weekly grocery list and errands",
        &["grocery", "errands"],
    );
    unrelated.timestamp = Some("2024-04-01".to_string());
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&[anchor, before, unrelated]);

    let (output, enrichment) = segmented.query_with_temporal_path_enrichment_and_strategy(
        "what happened before the doctor appointment",
        5,
        1,
        SegmentRoutingStrategy::LocalDistinctiveness,
        TemporalQueryContext {
            ends_at: Some("2024-05-10"),
            window_days: 2,
            time_hint: Some(TemporalQueryHint::Past),
            has_explicit_temporal: true,
            ..TemporalQueryContext::default()
        },
    );

    assert!(output
        .diagnostics
        .selected_segments
        .iter()
        .any(|route| route.segment_id == "session-before"));
    assert!(enrichment
        .temporal_expanded_segments
        .iter()
        .any(|expansion| expansion.segment_id == "session-before"
            && expansion.source_segment_id == "session-anchor"));
    assert!(output
        .results
        .iter()
        .any(|result| result.doc_id == "doc-before"));
}

#[test]
fn connected_segment_enrichment_swaps_in_explicitly_related_session() {
    let anchor = record(
        "doc-anchor",
        "session-anchor",
        "Alice mentioned toothpaste during the doctor appointment",
        &["Alice", "doctor", "appointment", "toothpaste"],
    );
    let connected = record(
        "doc-connected",
        "session-connected",
        "Alice bought toothpaste before the clinic visit",
        &["Alice", "bought", "toothpaste", "visit"],
    );
    let decoy = record(
        "doc-decoy",
        "session-decoy",
        "calendar reminder and weekly planning",
        &["calendar", "planning"],
    );
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&[anchor, connected, decoy]);

    let (output, enrichment) = segmented.query_with_connected_segment_enrichment_and_strategy(
        "doctor appointment",
        5,
        1,
        SegmentRoutingStrategy::CoverageLocalDistinctiveness,
        TemporalQueryContext::default(),
    );

    assert_eq!(output.diagnostics.selected_segments.len(), 1);
    assert_eq!(
        output.diagnostics.selected_segments[0].segment_id,
        "session-connected"
    );
    let expansion = enrichment
        .connected_expanded_segments
        .iter()
        .find(|expansion| expansion.segment_id == "session-connected")
        .expect("connected expansion");
    assert!(expansion.action.starts_with("swapped_out:"));
    assert!(expansion
        .shared_subjects
        .iter()
        .any(|term| term == "alic" || term == "alice"));
    assert!(expansion
        .shared_objects
        .iter()
        .any(|term| term == "toothpast"));
}

#[test]
fn missing_coverage_recovery_replaces_weak_segment() {
    let strong = record(
        "doc-strong",
        "session-strong",
        "GPA application admissions notes",
        &["GPA", "application", "admissions"],
    );
    let weak = record(
        "doc-weak",
        "session-weak",
        "GPA application general notes repeated",
        &["GPA", "application"],
    );
    let recovered = record(
        "doc-recovered",
        "session-recovered",
        "undergraduate graduate transcript details",
        &["undergraduate", "graduate", "transcript"],
    );
    let segments = build_segments_by_group_id(&[strong, weak, recovered]);
    let query_terms = query_tokens("GPA undergraduate graduate application");
    let routes = vec![
        SegmentRoute {
            segment_id: "session-strong".to_string(),
            score: 10.0,
            fallback: false,
        },
        SegmentRoute {
            segment_id: "session-weak".to_string(),
            score: 2.0,
            fallback: false,
        },
        SegmentRoute {
            segment_id: "session-recovered".to_string(),
            score: 1.5,
            fallback: false,
        },
    ];
    let corpus_stats = SegmentCorpusStats::from_segments(&segments, 0);

    let (selected, events) = recover_missing_coverage_segments(
        &query_terms,
        &routes[..2],
        &routes,
        &segments,
        &corpus_stats,
    );
    let selected_ids = selected
        .iter()
        .map(|route| route.segment_id.as_str())
        .collect::<HashSet<_>>();

    assert_eq!(selected.len(), 2);
    assert!(selected_ids.contains("session-strong"));
    assert!(selected_ids.contains("session-recovered"));
    assert!(!selected_ids.contains("session-weak"));
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].source_segment_id, "session-weak");
    assert!(events[0]
        .shared_subjects
        .iter()
        .any(|term| term == "undergradu"));
    assert!(events[0]
        .action
        .starts_with("recovered_missing_coverage:swapped_out:"));
}

#[test]
fn typed_evidence_score_requires_explicit_multi_bucket_match() {
    let query = query_connection_profile("Alice bought toothpaste today");
    let mut loose = SegmentConnectionProfile::default();
    loose.subjects.insert("toothpast".to_string());

    let mut explicit = loose.clone();
    explicit.objects.insert("toothpast".to_string());
    explicit.people.insert("alic".to_string());
    explicit.actions.insert("bought".to_string());
    explicit.times.insert("today".to_string());

    assert_eq!(typed_evidence_route_score(&query, &loose), 0.0);
    assert!(typed_evidence_route_score(&query, &explicit) > 0.0);
}

#[test]
fn sparse_router_does_not_query_zero_signal_padding_segments() {
    let records = vec![
        record("doc-a", "session-a", "docker install guide", &["docker"]),
        record("doc-b", "session-b", "kubernetes cluster", &["kubernetes"]),
        record("doc-c", "session-c", "postgres index tuning", &["postgres"]),
    ];
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

    assert_eq!(segmented.route("docker").len(), 1);
    let output = segmented.query_with_diagnostics("docker", 5, 3);

    assert_eq!(output.diagnostics.selected_segments.len(), 1);
    assert_eq!(
        output.diagnostics.selected_segments[0].segment_id,
        "session-a"
    );
    assert_eq!(output.diagnostics.queried_segment_count, 1);
    assert!(output.diagnostics.fallback_segments.is_empty());
    assert!(!output.diagnostics.routing_fallback);
    assert_eq!(output.diagnostics.routing_fallback_reason, None);
    assert_eq!(
        output
            .results
            .iter()
            .map(|result| result.doc_id.as_str())
            .collect::<Vec<_>>(),
        vec!["doc-a"]
    );
    assert!(output
        .diagnostics
        .fallback_segments
        .iter()
        .all(|route| route.fallback));
}

#[test]
fn sparse_router_executes_bounded_fallback_when_there_is_no_signal() {
    let records = vec![
        record("doc-a", "session-a", "docker install guide", &["docker"]),
        record("doc-b", "session-b", "kubernetes cluster", &["kubernetes"]),
    ];
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

    let output = segmented.query_with_diagnostics("espresso grinder", 5, 1);

    assert!(output.results.is_empty());
    assert!(output.diagnostics.selected_segments.is_empty());
    assert_eq!(output.diagnostics.queried_segment_count, 1);
    assert_eq!(output.diagnostics.fallback_segments.len(), 1);
    assert!(output.diagnostics.routing_fallback);
    assert_eq!(
        output.diagnostics.routing_fallback_reason.as_deref(),
        Some("no_signal_routes")
    );
}

#[test]
fn enriched_segment_queries_report_shard_completeness() {
    let records = vec![record(
        "doc-a",
        "session-a",
        "docker install guide for linux",
        &["docker", "install", "linux"],
    )];
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

    let (output, _) = segmented.query_with_segment_enrichment_and_strategy(
        "docker install",
        5,
        1,
        SegmentRoutingStrategy::SparseOverlap,
    );

    let completeness = output
        .diagnostics
        .shard_completeness
        .as_ref()
        .expect("enriched shard queries report completeness");
    assert!(completeness.is_complete());
    assert_eq!(completeness.expected_segments, vec!["session-a"]);
    assert_eq!(completeness.successful_segments, vec!["session-a"]);
}

#[test]
fn shard_completeness_marks_failed_segments_as_partial() {
    let completeness = ShardQueryCompleteness {
        expected_segments: vec!["session-a".to_string(), "session-b".to_string()],
        successful_segments: vec!["session-a".to_string()],
        failures: vec![ShardQueryFailure {
            segment_id: "session-b".to_string(),
            message: "deadline exceeded".to_string(),
        }],
    };

    assert!(!completeness.is_complete());
    assert_eq!(completeness.failures[0].segment_id, "session-b");
}

#[test]
fn segmented_query_passes_temporal_allowed_doc_ids_to_inner_indexes() {
    let records = vec![
        record(
            "doc-a",
            "session-a",
            "docker install guide for linux",
            &["docker", "install", "linux"],
        ),
        record(
            "doc-b",
            "session-b",
            "docker compose troubleshooting",
            &["docker", "compose", "troubleshooting"],
        ),
    ];
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);
    let allowed_doc_ids = HashSet::from(["doc-b".to_string()]);

    let output = segmented.query_with_temporal_context_and_diagnostics_and_strategy(
        "docker",
        5,
        2,
        SegmentRoutingStrategy::SparseOverlap,
        TemporalQueryContext {
            allowed_doc_ids: Some(&allowed_doc_ids),
            ..TemporalQueryContext::default()
        },
    );

    assert_eq!(output.results.len(), 1);
    assert_eq!(output.results[0].doc_id, "doc-b");
    assert_eq!(
        output
            .diagnostics
            .per_segment_result_counts
            .get("session-a")
            .copied(),
        Some(0)
    );
    assert_eq!(
        output
            .diagnostics
            .per_segment_result_counts
            .get("session-b")
            .copied(),
        Some(1)
    );
}

#[test]
fn all_segment_query_matches_global_memory_index_on_asymmetric_corpus() {
    let records = vec![
        record(
            "doc-a1",
            "session-a",
            "docker install guide for linux",
            &["docker", "install", "linux"],
        ),
        record(
            "doc-a2",
            "session-a",
            "docker setup notes and troubleshooting",
            &["docker", "setup", "troubleshooting"],
        ),
        record(
            "doc-a3",
            "session-a",
            "linux package manager notes",
            &["linux", "package"],
        ),
        record(
            "doc-b1",
            "session-b",
            "docker compose production incident",
            &["docker", "compose", "production"],
        ),
    ];
    let global = MemoryIndex::from_records(records.clone());
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);
    let statistics = GlobalBm25Statistics::from_indexes(
        segmented
            .segments
            .iter()
            .map(|segment| segment.index.as_ref()),
    );

    assert_eq!(statistics.shard_count(), 2);
    assert_eq!(
        tantivy::query::Bm25StatisticsProvider::total_num_docs(&statistics).unwrap(),
        4
    );

    let global_doc_ids = global
        .query("docker compose troubleshooting", 4)
        .into_iter()
        .map(|result| result.doc_id)
        .collect::<Vec<_>>();
    let segmented_doc_ids = segmented
        .query_all_segments("docker compose troubleshooting", 4)
        .into_iter()
        .map(|result| result.doc_id)
        .collect::<Vec<_>>();

    assert_eq!(segmented_doc_ids, global_doc_ids);
}

#[test]
fn all_segment_query_is_stable_after_rebuilding_segments() {
    let records = vec![
        record(
            "doc-a",
            "session-a",
            "docker install guide for linux",
            &["docker", "install", "linux"],
        ),
        record(
            "doc-b",
            "session-b",
            "docker compose troubleshooting",
            &["docker", "compose", "troubleshooting"],
        ),
        record(
            "doc-c",
            "session-c",
            "postgres index tuning",
            &["postgres", "index", "tuning"],
        ),
    ];
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);
    let segments = build_segments_by_group_id(&records);
    let rebuilt = SegmentedMemoryIndex {
        catalog: SegmentCorpusStats::from_segments(&segments, 0),
        global_statistics: GlobalBm25Statistics::from_indexes(
            segments.iter().map(|segment| segment.index.as_ref()),
        ),
        generation: 0,
        segments,
    };

    let expected = segmented
        .query_all_segments("docker compose troubleshooting", 3)
        .into_iter()
        .map(|result| result.doc_id)
        .collect::<Vec<_>>();
    let actual = rebuilt
        .query_all_segments_with_diagnostics("docker compose troubleshooting", 3)
        .results
        .into_iter()
        .map(|result| result.doc_id)
        .collect::<Vec<_>>();

    assert_eq!(actual, expected);
}

#[test]
fn all_segment_query_executes_zero_signal_segments() {
    let records = vec![
        record(
            "doc-a",
            "session-a",
            "docker compose troubleshooting",
            &["docker", "compose"],
        ),
        record(
            "doc-b",
            "session-b",
            "postgres index tuning",
            &["postgres", "index"],
        ),
        record(
            "doc-c",
            "session-c",
            "calendar appointment notes",
            &["calendar", "appointment"],
        ),
    ];
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

    let output = segmented.query_all_segments_with_diagnostics("docker compose", 3);

    assert_eq!(output.diagnostics.queried_segment_count, 3);
    assert_eq!(output.diagnostics.selected_segments.len(), 3);
    assert_eq!(
        output
            .diagnostics
            .shard_completeness
            .as_ref()
            .expect("all-segment completeness")
            .expected_segments
            .len(),
        3
    );
    assert!(output
        .diagnostics
        .fallback_segments
        .iter()
        .any(|route| route.segment_id == "session-b"));
    assert!(output
        .diagnostics
        .fallback_segments
        .iter()
        .any(|route| route.segment_id == "session-c"));
}

#[test]
fn all_segment_query_only_executes_filter_eligible_segments() {
    let records = vec![
        record("doc-a", "session-a", "docker compose", &["docker"]),
        record("doc-b", "session-b", "postgres tuning", &["postgres"]),
    ];
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);
    let allowed = HashSet::from(["doc-a".to_string()]);

    let output = segmented.query_all_segments_with_temporal_context_and_diagnostics(
        "docker",
        2,
        TemporalQueryContext {
            allowed_doc_ids: Some(&allowed),
            ..TemporalQueryContext::default()
        },
    );

    assert_eq!(output.diagnostics.queried_segment_count, 1);
    assert_eq!(
        output
            .diagnostics
            .shard_completeness
            .as_ref()
            .expect("filtered all-segment completeness")
            .expected_segments,
        vec!["session-a".to_string()]
    );
    assert_eq!(
        output
            .diagnostics
            .per_segment_result_counts
            .get("session-b"),
        Some(&0)
    );
}

#[test]
fn sparse_router_marks_empty_query_terms_as_fallback() {
    let records = vec![
        record("doc-a", "session-a", "docker install guide", &["docker"]),
        record("doc-b", "session-b", "kubernetes cluster", &["kubernetes"]),
    ];
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);

    let output = segmented.query_with_diagnostics("the and to", 5, 2);

    assert!(output.results.is_empty());
    assert!(output.diagnostics.selected_segments.is_empty());
    assert_eq!(output.diagnostics.queried_segment_count, 2);
    assert!(output.diagnostics.routing_fallback);
    assert_eq!(
        output.diagnostics.routing_fallback_reason.as_deref(),
        Some("empty_query_terms")
    );
    assert_eq!(output.diagnostics.fallback_segments.len(), 2);
}
