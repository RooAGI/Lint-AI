use super::*;
use crate::index::{
    DocRecord, GlobalBm25Statistics, MemoryIndex, Provenance, ScoreBreakdown, SearchResult,
    TemporalQueryContext, TemporalQueryHint,
};
use crate::query_semantics::{
    parse_reference_date, resolve_anchor_window, temporal_anchor_is_span,
};
use crate::tier1::{RankedTerm, Tier1Entity, BEHOOD_NP_ENTITY_SOURCE};
use chrono::NaiveDate;
use chrono::TimeDelta;
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
/// be indistinguishable from a full rebuild — same segment membership, same
/// catalog (including postings order of every derived map), and bit-identical
/// query results.
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

    // Query results must match bit-for-bit: the query path is fully
    // deterministic (float sums run in sorted-key order and top-k
    // tie-breaking falls back to doc_id), so any divergence here is a real
    // incremental-refresh bug, not test flakiness.
    for query in ["alpha", "beta one two", "gamma three four", "one two"] {
        assert_eq!(
            projected_results(&next.query(query, 5, 8)),
            projected_results(&oracle.query(query, 5, 8)),
            "query results diverged for query {query:?}"
        );
    }
}

/// (doc_id, score bits) projection of query results for bit-for-bit comparison.
fn projected_results(results: &[SearchResult]) -> Vec<(String, u32)> {
    results
        .iter()
        .map(|result| (result.doc_id.clone(), result.score.to_bits()))
        .collect()
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
        SegmentRoutingStrategy::TypedEvidenceMultiplicative,
        SegmentRoutingStrategy::CoverageTeamTypedMultiplicative,
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
fn multiplicative_typed_evidence_gates_on_content() {
    // Typed evidence must never elect a zero-content segment.
    assert_eq!(combine_typed_evidence_multiplicative(0.0, 10.0), 0.0);
    assert_eq!(combine_typed_evidence_multiplicative(0.0, 0.0), 0.0);
    // No typed evidence leaves the base score untouched.
    assert_eq!(combine_typed_evidence_multiplicative(2.5, 0.0), 2.5);
    // Typed evidence strictly amplifies a positive base score.
    let amplified = combine_typed_evidence_multiplicative(2.5, 4.0);
    assert!(amplified > 2.5, "expected amplification, got {amplified}");
    // The amplification saturates: typed evidence can at most double the base.
    let saturated = combine_typed_evidence_multiplicative(2.5, 1e6);
    assert!(
        saturated < 5.0,
        "expected saturation below 2x, got {saturated}"
    );
    assert!(
        saturated > amplified,
        "expected monotonic growth, got {saturated}"
    );
}

#[test]
fn expansion_terms_participate_in_segment_routing() {
    // "diploma" appears in no segment, but the lexical store expands it to
    // "degree". The router must select the degree segment instead of falling
    // back to arbitrary segments; otherwise the per-segment expansion
    // vocabulary never reaches the segment that holds it.
    let records = vec![
        record(
            "doc-degree",
            "session-degree",
            "congratulations on finishing your degree",
            &["degree"],
        ),
        record(
            "doc-other",
            "session-other",
            "unrelated project discussion",
            &["unrelated", "project"],
        ),
    ];
    let segments = build_segments_by_group_id(&records);
    assert_eq!(segments.len(), 2);

    let routes = route_segments("diploma", &segments);
    assert_eq!(routes[0].segment_id, "session-degree");
    assert!(
        routes[0].score > 0.0,
        "expansion routing should give the degree segment a positive score"
    );

    let routes = route_segments_with_strategy(
        "diploma",
        &segments,
        SegmentRoutingStrategy::TypedEvidenceMultiplicative,
    );
    assert_eq!(routes[0].segment_id, "session-degree");
    assert!(routes[0].score > 0.0);
}

#[test]
fn literal_query_terms_outrank_expansion_noise_in_routing() {
    // Regression test: routing on the expanded vocabulary let noisy
    // wrong-sense expansions outrank segments matching the literal query
    // terms ("game" -> "bathroom"/"gospel" misrouted real benchmark queries).
    // Routing must prefer literal-term signal; expansion is only a recall
    // fallback when literal terms match nothing (see
    // expansion_terms_participate_in_segment_routing).
    let raw_terms = query_tokens("game");
    let expanded_terms = query_tokens_expanded("game");
    // Keep only expansions that survive a second stemming pass, so the test
    // is robust to the stemmer's non-idempotency ("degre" -> "degr").
    let mut noise_terms: Vec<String> = expanded_terms
        .difference(&raw_terms)
        .filter(|term| query_tokens(term).into_iter().collect::<Vec<_>>() == vec![(*term).clone()])
        .cloned()
        .collect();
    noise_terms.sort();
    assert!(
        noise_terms.len() >= 2,
        "test needs noisy expansions of 'game' from the lexical store"
    );
    let noise_refs: Vec<&str> = noise_terms.iter().map(String::as_str).collect();
    let records = vec![
        record(
            "doc-gold",
            "session-gold",
            "I bought a new game yesterday",
            &["game"],
        ),
        record(
            "doc-noise",
            "session-noise",
            &noise_terms.join(" "),
            &noise_refs,
        ),
    ];
    let segments = build_segments_by_group_id(&records);
    assert_eq!(segments.len(), 2);

    let routes = route_segments("game", &segments);
    assert_eq!(
        routes[0].segment_id, "session-gold",
        "literal-term segment must route first despite expansion noise"
    );
    assert!(routes[0].score > 0.0);
    // The expansion-only segment must not draw any routing score while the
    // literal terms match: under expanded-vocabulary routing it overlapped
    // the noisy expansions and outranked the literal segment.
    for route in &routes {
        if route.segment_id != "session-gold" {
            assert_eq!(
                route.score, 0.0,
                "expansion-only segment '{}' must get no routing score",
                route.segment_id
            );
        }
    }
}

/// A DocRecord whose key entities include grammar-accepted phrases
/// (behood-np provenance), alongside ordinary terms.
fn record_with_phrases(
    doc_id: &str,
    group_id: &str,
    content: &str,
    terms: &[&str],
    phrases: &[(&str, &str)],
) -> DocRecord {
    let mut rec = record(doc_id, group_id, content, terms);
    rec.key_entities
        .extend(phrases.iter().map(|(text, kind)| Tier1Entity {
            text: (*text).to_string(),
            label: (*kind).to_string(),
            start: 0,
            end: 0,
            // Score 2.0 matches assemble_doc_record: grammar-accepted mentions
            // are higher precision than heuristic NER.
            score: Some(2.0),
            source: BEHOOD_NP_ENTITY_SOURCE.to_string(),
        }));
    rec
}

#[test]
fn key_phrase_literal_tokens_join_summary_entity_channel() {
    // The Porter stem conflates the phrase head "conference" with the verb
    // "confer". Grammar-accepted phrases must therefore be indexed literally
    // in the entity channel, which (unlike local_memory) is never pruned.
    let rec = record_with_phrases(
        "doc-13",
        "session-13",
        "Last week I went to a Harry Potter conference in the UK",
        &["went", "week"],
        &[("Harry Potter conference", "event"), ("UK", "place")],
    );
    let summary = SegmentRoutingSummary::from_records(std::slice::from_ref(&rec));
    assert!(
        summary.entities.contains_key("conference"),
        "entity channel must index the literal phrase head, not just 'confer'"
    );
    assert!(summary.entities.contains_key("harry"));
    assert!(summary.entities.contains_key("potter"));
    // The stemmed form is still indexed too, for the existing stemmed path.
    assert!(summary.entities.contains_key("confer"));
}

#[test]
fn literal_phrase_evidence_outranks_acronym_only_evidence() {
    // The conv-43_q5 miss: "Which week did Tim visit the UK for the Harry
    // Potter Conference?" must route to the session holding the
    // grammar-accepted phrase, not to a session that merely mentions "UK".
    // "conference" is rare (high IDF); "uk" is common (low IDF), so the
    // literal phrase evidence dominates the acronym-only evidence.
    //
    // The verb-"confer" distractor proves the literal channel disambiguates:
    // the stemmed form "confer" matches both the phrase head and the verb,
    // but only the phrase session carries the literal "conference".
    let gold = record_with_phrases(
        "doc-13",
        "session-13",
        "Last week I went to a Harry Potter conference in the UK",
        &["went", "week"],
        &[("Harry Potter conference", "event"), ("UK", "place")],
    );
    let other = record(
        "doc-7",
        "session-7",
        "UK weather and travel discussion",
        &["uk", "weather", "travel"],
    );
    let verb_distractor = record(
        "doc-9",
        "session-9",
        "the committee will confer the award",
        &["confer", "award", "committee"],
    );
    let segments = build_segments_by_group_id(&[gold, other, verb_distractor]);
    assert_eq!(segments.len(), 3);

    let routes = route_segments_with_strategy(
        "Which week did Tim visit the UK for the Harry Potter Conference?",
        &segments,
        SegmentRoutingStrategy::TypedEvidenceMultiplicative,
    );
    assert_eq!(
        routes[0].segment_id, "session-13",
        "literal phrase evidence must outrank acronym-only evidence"
    );
    assert!(routes[0].score > 0.0);
}

#[test]
fn literal_query_tokens_are_unstemmed() {
    // Premise check: the stemmer really does conflate "conference" with the
    // verb "confer", which is why the literal channel exists.
    assert_eq!(
        query_tokens("conference"),
        HashSet::from(["confer".to_string()])
    );
    assert_eq!(
        literal_query_tokens("conference"),
        vec!["conference".to_string()]
    );
    assert_eq!(
        literal_query_tokens("Which week did Tim visit the UK?"),
        // "uk" is below the tokenizer's 3-char minimum; stopwords filtered.
        vec!["week".to_string(), "tim".to_string(), "visit".to_string(),]
    );
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
        vec![
            "dock".to_string(),
            "docker".to_string(),
            "dockhand".to_string(),
            "episod".to_string(),
            "establish".to_string(),
            "facil".to_string(),
            "instal".to_string(),
            "wallop".to_string(),
            "worker".to_string(),
        ]
    );
    assert_eq!(
        output.diagnostics.covered_query_terms,
        vec!["docker".to_string(), "instal".to_string()]
    );
    assert_eq!(
        output.diagnostics.uncovered_query_terms,
        vec![
            "dock".to_string(),
            "dockhand".to_string(),
            "episod".to_string(),
            "establish".to_string(),
            "facil".to_string(),
            "wallop".to_string(),
            "worker".to_string(),
        ]
    );
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
        vec![
            "compos".to_string(),
            "dock".to_string(),
            "dockhand".to_string(),
            "wallop".to_string(),
            "worker".to_string(),
        ]
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
    assert_eq!(
        top_two.diagnostics.uncovered_query_terms,
        vec![
            "dock".to_string(),
            "dockhand".to_string(),
            "wallop".to_string(),
            "worker".to_string(),
        ]
    );
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
fn route_aware_rerank_selects_highest_relevance_without_diversity_bias() {
    // Relevance wins: a higher base_score doc is selected even when a
    // lower-scoring doc would add more "new" evidence terms or come from an
    // unrepresented segment. Diversity bonuses during selection demote the
    // single best doc, which is exactly wrong for a recall@k objective.
    let candidates = vec![
        RouteAwareCandidate {
            result: search_result("doc-best", 1.0),
            segment_id: "session-common".to_string(),
            base_score: 1.0,
        },
        RouteAwareCandidate {
            result: search_result("doc-diverse", 0.9),
            segment_id: "session-specific".to_string(),
            base_score: 0.9,
        },
    ];

    let selected = select_route_aware_top_k(candidates, 1);

    assert_eq!(selected[0].doc_id, "doc-best");
    assert_eq!(selected[0].score, 1.0);
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

#[test]
fn reciprocal_rank_fusion_prefers_docs_ranked_by_both_modes() {
    let mode_a = vec![search_result("a", 0.1), search_result("b", 0.9)];
    let mode_b = vec![search_result("b", 0.1), search_result("c", 0.9)];
    // b: 1/62 + 1/61 > a: 1/61 > c: 1/62 ... a outranks c (rank 0 vs rank 1)
    let fused = reciprocal_rank_fusion(&[mode_a.as_slice(), mode_b.as_slice()], 10);
    let ids: Vec<&str> = fused.iter().map(|r| r.doc_id.as_str()).collect();
    assert_eq!(ids, vec!["b", "a", "c"]);
}

#[test]
fn reciprocal_rank_fusion_ignores_cross_mode_scores() {
    // "low" has a huge raw score but is ranked last in its mode; "high" has a
    // tiny score but is ranked first. Ranks win: score scales cannot crowd out
    // a mode's ranking.
    let mode_a = vec![search_result("high", 0.01), search_result("low", 999.0)];
    let mode_b = vec![search_result("high", 0.01)];
    let fused = reciprocal_rank_fusion(&[mode_a.as_slice(), mode_b.as_slice()], 10);
    let ids: Vec<&str> = fused.iter().map(|r| r.doc_id.as_str()).collect();
    assert_eq!(ids, vec!["high", "low"]);
}

#[test]
fn reciprocal_rank_fusion_truncates_and_breaks_ties_by_doc_id() {
    let mode_a = vec![search_result("b", 1.0)];
    let mode_b = vec![search_result("a", 1.0)];
    // Equal RRF weight (both rank 0 in one mode): doc_id ascending.
    let fused = reciprocal_rank_fusion(&[mode_a.as_slice(), mode_b.as_slice()], 10);
    let ids: Vec<&str> = fused.iter().map(|r| r.doc_id.as_str()).collect();
    assert_eq!(ids, vec!["a", "b"]);
    let truncated = reciprocal_rank_fusion(&[mode_a.as_slice(), mode_b.as_slice()], 1);
    assert_eq!(truncated.len(), 1);
    assert_eq!(truncated[0].doc_id, "a");
}

#[test]
fn reciprocal_rank_fusion_handles_empty_modes() {
    let empty: Vec<SearchResult> = vec![];
    assert!(reciprocal_rank_fusion(&[], 10).is_empty());
    assert!(reciprocal_rank_fusion(&[empty.as_slice()], 10).is_empty());
    let mode_a = vec![search_result("a", 1.0)];
    let fused = reciprocal_rank_fusion(&[mode_a.as_slice(), empty.as_slice()], 10);
    assert_eq!(fused.len(), 1);
    assert_eq!(fused[0].doc_id, "a");
}

fn temporal_boost_segment(group_id: &str, timestamps: &[&str]) -> MemoryIndexSegment {
    let records: Vec<DocRecord> = timestamps
        .iter()
        .enumerate()
        .map(|(i, timestamp)| {
            let mut rec = record(
                &format!("{group_id}-doc-{i}"),
                group_id,
                "some content without temporal terms",
                &["content"],
            );
            rec.timestamp = Some((*timestamp).to_string());
            rec
        })
        .collect();
    let segmented = SegmentedMemoryIndex::from_records_by_group_id(&records);
    assert_eq!(segmented.segments.len(), 1);
    // `MemoryIndexSegment` is not `Clone`; rebuild is cheap enough for tests.
    // Instead of cloning, move the single segment out via segment id lookup.
    let id = segmented.segments[0].segment_id.clone();
    let mut segments = segmented.segments;
    let position = segments.iter().position(|s| s.segment_id == id).unwrap();
    segments.swap_remove(position)
}

fn anchored_temporal_context(anchor: Option<&'static str>) -> TemporalQueryContext<'static> {
    let anchor_window = anchor.and_then(|date| {
        let parsed = parse_reference_date(date)?;
        Some((
            parsed.checked_sub_signed(TimeDelta::days(7))?,
            parsed.checked_add_signed(TimeDelta::days(7))?,
        ))
    });
    TemporalQueryContext {
        anchor_date: anchor,
        anchor_window,
        ends_at: Some("2024-05-10"),
        time_hint: None,
        has_explicit_temporal: true,
        ..TemporalQueryContext::default()
    }
}

fn anchored_span_context(start: &'static str, end: &'static str) -> TemporalQueryContext<'static> {
    TemporalQueryContext {
        anchor_date: None,
        anchor_window: Some((
            parse_reference_date(start).expect("valid start"),
            parse_reference_date(end).expect("valid end"),
        )),
        ends_at: Some(end),
        time_hint: None,
        has_explicit_temporal: true,
        ..TemporalQueryContext::default()
    }
}

#[test]
fn temporal_route_boost_is_zero_without_temporal_context() {
    let segment = temporal_boost_segment("g", &["2024-05-10"]);
    assert_eq!(
        segment_temporal_route_boost(&segment, TemporalQueryContext::default()),
        0.0
    );
}

#[test]
fn temporal_route_boost_uses_max_not_sum_across_records() {
    // Twenty in-window records must not beat one same-day record: the factor
    // comes from the segment's best temporal evidence, not the record count.
    let many = temporal_boost_segment(
        "many",
        &[
            "2024-05-09",
            "2024-05-09",
            "2024-05-09",
            "2024-05-09",
            "2024-05-09",
            "2024-05-09",
            "2024-05-09",
            "2024-05-09",
            "2024-05-09",
            "2024-05-09",
            "2024-05-09",
            "2024-05-09",
            "2024-05-09",
            "2024-05-09",
            "2024-05-09",
            "2024-05-09",
            "2024-05-09",
            "2024-05-09",
            "2024-05-09",
            "2024-05-09",
        ],
    );
    let one = temporal_boost_segment("one", &["2024-05-10"]);
    let temporal = anchored_temporal_context(Some("2024-05-10"));
    let many_factor = segment_temporal_route_boost(&many, temporal);
    let one_factor = segment_temporal_route_boost(&one, temporal);
    // Same-day proximity is 1.0; one-day-away proximity is 1 - 1/7.
    assert!((one_factor - 1.0).abs() < 1e-6, "one_factor={one_factor}");
    assert!(
        (many_factor - (1.0 - 1.0 / 7.0)).abs() < 1e-6,
        "many_factor={many_factor}"
    );
    assert!(many_factor < one_factor);
}

#[test]
fn temporal_route_boost_is_bounded_and_ignores_out_of_window_records() {
    let segment = temporal_boost_segment("g", &["2023-01-01", "2025-12-31"]);
    let factor =
        segment_temporal_route_boost(&segment, anchored_temporal_context(Some("2024-05-10")));
    assert_eq!(factor, 0.0);
    // Even a pathological segment cannot exceed the 1.0 factor cap.
    let saturated = temporal_boost_segment(
        "s",
        &[
            "2024-05-10",
            "2024-05-10",
            "2024-05-10",
            "2024-05-10",
            "2024-05-10",
            "2024-05-10",
            "2024-05-10",
            "2024-05-10",
            "2024-05-10",
            "2024-05-10",
        ],
    );
    let saturated_factor =
        segment_temporal_route_boost(&saturated, anchored_temporal_context(Some("2024-05-10")));
    assert!(
        saturated_factor <= 1.0,
        "saturated_factor={saturated_factor}"
    );
}

#[test]
fn temporal_route_boost_applies_hint_from_best_record_only() {
    // Past hint: a record 30 days before the anchor is outside the 7-day
    // window but still earns the hint weight once, not per record.
    let segment = temporal_boost_segment("g", &["2024-04-10", "2024-04-10", "2024-04-10"]);
    let temporal = TemporalQueryContext {
        time_hint: Some(TemporalQueryHint::Past),
        ..anchored_temporal_context(Some("2024-05-10"))
    };
    let factor = segment_temporal_route_boost(&segment, temporal);
    assert!((factor - 0.18).abs() < 1e-6, "factor={factor}");
}

fn anchored_prefilter_segments() -> Vec<MemoryIndexSegment> {
    let mut in_window = record(
        "in-doc-0",
        "in-window",
        "lunch meeting on tuesday with the team",
        &["lunch", "meeting", "tuesday"],
    );
    in_window.timestamp = Some("2024-05-07".to_string());
    let mut out_window = record(
        "out-doc-0",
        "out-window",
        "lunch meeting notes from last year",
        &["lunch", "meeting"],
    );
    out_window.timestamp = Some("2023-01-01".to_string());
    build_segments_by_group_id(&[in_window, out_window])
}

fn route_with_temporal_for_prefilter_test(
    segments: &[MemoryIndexSegment],
    temporal: TemporalQueryContext<'_>,
) -> Vec<SegmentRoute> {
    let corpus_stats = SegmentCorpusStats::from_segments(segments, 0);
    route_segments_with_temporal_context_and_corpus_stats(
        "lunch meeting tuesday",
        segments,
        SegmentRoutingStrategy::TypedEvidence,
        temporal,
        &corpus_stats,
    )
}

#[test]
fn anchored_prefilter_restricts_routing_to_in_window_segments() {
    let segments = anchored_prefilter_segments();
    let routes = route_with_temporal_for_prefilter_test(
        &segments,
        anchored_temporal_context(Some("2024-05-07")),
    );
    // The out-of-window segment matches the content terms but must not be
    // routed when the anchor is resolved.
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].segment_id, "in-window");
}

#[test]
fn anchored_prefilter_falls_back_when_nothing_is_in_window() {
    let segments = anchored_prefilter_segments();
    let routes = route_with_temporal_for_prefilter_test(
        &segments,
        anchored_temporal_context(Some("2025-01-01")),
    );
    // No segment holds a record in the anchor window: route everything rather
    // than returning an empty route set.
    let ids: HashSet<&str> = routes
        .iter()
        .map(|route| route.segment_id.as_str())
        .collect();
    assert!(ids.contains("in-window"));
    assert!(ids.contains("out-window"));
}

#[test]
fn anchored_prefilter_is_inactive_without_anchor_date() {
    let segments = anchored_prefilter_segments();
    let routes = route_with_temporal_for_prefilter_test(&segments, anchored_temporal_context(None));
    // No resolved anchor: both segments stay routable.
    let ids: HashSet<&str> = routes
        .iter()
        .map(|route| route.segment_id.as_str())
        .collect();
    assert!(ids.contains("in-window"));
    assert!(ids.contains("out-window"));
}

#[test]
fn temporal_anchor_is_span_classifies_point_and_range_phrases() {
    // Points: named weekdays and explicit offsets win over nearby range words.
    assert!(!temporal_anchor_is_span(
        "last Tuesday",
        "Who did I meet with during the lunch last Tuesday?"
    ));
    assert!(!temporal_anchor_is_span(
        "four weeks ago",
        "I mentioned an investment for a competition four weeks ago?"
    ));
    assert!(!temporal_anchor_is_span(
        "yesterday",
        "What did I do yesterday?"
    ));
    // Ranges: explicit markers, or a bare quantity with range context.
    assert!(temporal_anchor_is_span(
        "two months",
        "What is the order of the concerts I attended in the past two months?"
    ));
    assert!(temporal_anchor_is_span(
        "past two months",
        "What is the order of the concerts I attended in the past two months?"
    ));
    // "most recent" is not range context.
    assert!(!temporal_anchor_is_span(
        "two months",
        "What is the most recent thing I bought two months after moving?"
    ));
}

#[test]
fn resolve_anchor_window_returns_point_and_range_windows() {
    let reference = parse_reference_date("2024-05-10").unwrap();
    let (start, end) =
        resolve_anchor_window("last Tuesday", "lunch last Tuesday?", reference).unwrap();
    assert_eq!(
        (start.to_string(), end.to_string()),
        ("2024-04-30".to_string(), "2024-05-14".to_string())
    );
    let (start, end) = resolve_anchor_window(
        "two months",
        "concerts I attended in the past two months",
        reference,
    )
    .unwrap();
    assert_eq!(
        (start.to_string(), end.to_string()),
        ("2024-03-11".to_string(), "2024-05-10".to_string())
    );
    assert!(
        resolve_anchor_window("most recently", "what happened most recently?", reference).is_none()
    );
}

#[test]
fn span_prefilter_keeps_mid_range_segments_and_drops_out_of_range() {
    let mut in_range = record(
        "in-doc-0",
        "in-range",
        "concerts and musical events attended",
        &["concerts", "musical", "events"],
    );
    in_range.timestamp = Some("2024-04-15".to_string());
    let mut out_of_range = record(
        "out-doc-0",
        "out-range",
        "concerts and musical events attended",
        &["concerts", "musical", "events"],
    );
    out_of_range.timestamp = Some("2023-01-01".to_string());
    let segments = build_segments_by_group_id(&[in_range, out_of_range]);
    let corpus_stats = SegmentCorpusStats::from_segments(&segments, 0);
    let routes = route_segments_with_temporal_context_and_corpus_stats(
        "concerts musical events",
        &segments,
        SegmentRoutingStrategy::TypedEvidence,
        anchored_span_context("2024-03-11", "2024-05-10"),
        &corpus_stats,
    );
    // A point-window around the range start would have dropped the mid-range
    // segment; the range window keeps it.
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].segment_id, "in-range");
}

#[test]
fn segment_record_dates_are_cached_sorted_and_unique() {
    let mut dated = record(
        "dated-doc-0",
        "dated-group",
        "lunch meeting on tuesday",
        &["lunch", "meeting"],
    );
    dated.timestamp = Some("2024-05-07".to_string());
    let mut duplicate = record(
        "dated-doc-1",
        "dated-group",
        "another lunch meeting",
        &["lunch"],
    );
    duplicate.timestamp = Some("2024/05/07".to_string());
    let mut undated = record(
        "dated-doc-2",
        "dated-group",
        "no timestamp on this one",
        &["meeting"],
    );
    undated.timestamp = Some("not-a-date".to_string());
    let segments = build_segments_by_group_id(&[dated, duplicate, undated]);
    assert_eq!(segments.len(), 1);
    let segment = &segments[0];
    // Two records share the same date in different formats; the unparseable
    // timestamp is excluded. Dates are parsed once and cached.
    let first = segment.record_dates() as *const Vec<NaiveDate>;
    let second = segment.record_dates() as *const Vec<NaiveDate>;
    assert_eq!(first, second, "record dates must be computed once");
    assert_eq!(segment.record_dates().len(), 1);
    assert_eq!(
        segment.record_dates()[0],
        parse_reference_date("2024-05-07").unwrap()
    );
    // The cached dates drive the anchor-window check: in-window matches,
    // out-of-window does not.
    let in_window = (
        parse_reference_date("2024-05-01").unwrap(),
        parse_reference_date("2024-05-10").unwrap(),
    );
    assert!(segment_has_record_in_anchor_window(segment, in_window));
    let out_window = (
        parse_reference_date("2023-01-01").unwrap(),
        parse_reference_date("2023-12-31").unwrap(),
    );
    assert!(!segment_has_record_in_anchor_window(segment, out_window));
}
