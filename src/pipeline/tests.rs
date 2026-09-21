use super::*;
use crate::index::TemporalQueryContext;
use crate::query_plan::PreparedQuery;
use crate::segments::{SegmentManifest, SegmentRoutingStrategy};
use crate::semantic_relations::SupersessionOptions;
use crate::source::SourceDocument;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

fn sample_doc(id: &str, content: &str) -> SourceDocument {
    SourceDocument {
        doc_id: id.to_string(),
        source: format!("artifact://{}", id),
        content: content.to_string(),
        concept: id.to_string(),
        group_id: None,
        filters: std::collections::BTreeMap::new(),
        headings: vec!["Overview".to_string()],
        links: vec![],
        timestamp: None,
        doc_length: content.len(),
        author_agent: None,
    }
}

#[test]
fn workspace_watcher_publishes_relative_file_change_events_without_content() {
    let root = std::env::temp_dir().join(format!(
        "lint-ai-workspace-watcher-{}-{}",
        std::process::id(),
        workspace_now_ms()
    ));
    fs::create_dir_all(root.join(".lint-ai")).unwrap();
    let watcher = WorkspaceWatcher::new(&root, &[]).unwrap();
    fs::write(root.join("notes.md"), "private content").unwrap();
    fs::write(root.join(".lint-ai").join("internal.json"), "internal").unwrap();

    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    let events = loop {
        let events = watcher.take_events();
        if !events.is_empty() || std::time::Instant::now() >= deadline {
            break events;
        }
        thread::sleep(Duration::from_millis(25));
    };

    assert!(
        events.iter().any(|event| event.file_path == "notes.md"),
        "unexpected watcher events: {events:?}"
    );
    assert!(events
        .iter()
        .all(|event| event.file_path != ".lint-ai/internal.json"));
    assert!(events.iter().all(|event| event.event.starts_with("file_")));
    drop(watcher);
    fs::remove_dir_all(root).unwrap();
}

fn sample_doc_with_group(id: &str, group_id: &str, content: &str) -> SourceDocument {
    SourceDocument {
        group_id: Some(group_id.to_string()),
        ..sample_doc(id, content)
    }
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    std::env::current_dir()
        .expect("current directory should be available")
        .join("target")
        .join(format!("lint-ai-{prefix}-{nanos}"))
}

fn reopen_store_after_writer_drop(index_root: &Path, options: PipelineOptions) -> IndexStore {
    let mut last_error = None;
    for _ in 0..20 {
        match IndexStore::at_path(index_root, options.clone()) {
            Ok(store) => return store,
            Err(error) => last_error = Some(error),
        }
        thread::sleep(Duration::from_millis(10));
    }
    panic!(
        "persistent index should reopen after its previous writer is dropped: {}",
        last_error.expect("at least one reopen attempt should fail")
    );
}

#[test]
fn build_memory_index_works_with_defaults() {
    let docs = vec![sample_doc("doc-1", "docker install on linux")];
    let index =
        build_query_snapshot(&docs, &PipelineOptions::default()).expect("index should build");
    let results = index.query("docker", 5);
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, "doc-1");
}

#[test]
fn query_prefers_latest_equally_relevant_conversation_memory() {
    let today = chrono::DateTime::<chrono::Utc>::from(SystemTime::now()).date_naive();
    let mut latest = sample_doc(
        "conversation-latest",
        "Conversation memory: the project Aurora rollout decision was to ship in phases.",
    );
    latest.timestamp = Some(format!("{today}T12:00:00Z"));
    let mut old = sample_doc(
        "conversation-old",
        "Conversation memory: the project Aurora rollout decision was to ship in phases.",
    );
    old.timestamp = Some(format!("{}T12:00:00Z", today - chrono::Duration::days(180)));

    let index = build_query_snapshot(&[old, latest], &PipelineOptions::default())
        .expect("conversation memories should build");
    let results = index.query("Aurora rollout decision", 2);

    assert_eq!(
        results.first().map(|result| result.doc_id.as_str()),
        Some("conversation-latest")
    );
    assert!(results[0].score_breakdown.recency_score > results[1].score_breakdown.recency_score);
}

#[test]
fn query_prefers_current_markdown_guidance_when_wording_changes() {
    let today = chrono::DateTime::<chrono::Utc>::from(SystemTime::now()).date_naive();
    let mut old = sample_doc(
        "ownership-history",
        "The national structure assigned valve control to the National Park POD. This was the prior owner assignment.",
    );
    old.timestamp = Some(format!("{}T12:00:00Z", today - chrono::Duration::days(180)));

    let mut current = sample_doc(
        "ownership-current",
        "Current domain responsibility: the Controls POD owns valve control. Effective today, use the Controls POD for this work.",
    );
    current.timestamp = Some(format!("{today}T12:00:00Z"));

    let index = build_query_snapshot(&[old, current], &PipelineOptions::default())
        .expect("Markdown-like ownership guidance should build");
    let results = index.query("who is responsible for valve control", 2);

    assert_eq!(
        results.first().map(|result| result.doc_id.as_str()),
        Some("ownership-current"),
        "current guidance should outrank the older, still-relevant wording: {results:?}"
    );
    assert!(
        results[0].score_breakdown.recency_score > results[1].score_breakdown.recency_score,
        "current guidance should receive the stronger temporal signal: {results:?}"
    );
}

#[test]
fn chunk_ids_are_stable_for_same_input() {
    let content = "# Intro\nDocker install on linux\n# Usage\nRun docker info";
    let first = crate::chunking::chunk_document_sections(content, "doc-1");
    let second = crate::chunking::chunk_document_sections(content, "doc-1");
    assert_eq!(first.len(), second.len());
    let first_ids = first
        .into_iter()
        .map(|chunk| chunk.chunk_id)
        .collect::<Vec<_>>();
    let second_ids = second
        .into_iter()
        .map(|chunk| chunk.chunk_id)
        .collect::<Vec<_>>();
    assert_eq!(first_ids, second_ids);
}

#[test]
fn artifact_index_upsert_remove_and_query() {
    let mut artifact_index = IndexStore::new(PipelineOptions::default());
    artifact_index.upsert(sample_doc("doc-1", "docker install guide"));
    let results = artifact_index
        .query("docker", 5)
        .expect("query should succeed");
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, "doc-1");
    assert!(!artifact_index.is_dirty());

    artifact_index.upsert(sample_doc("doc-2", "kubernetes setup guide"));
    assert!(artifact_index.is_dirty());
    let results = artifact_index
        .query("kubernetes", 5)
        .expect("query should succeed");
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, "doc-2");

    let removed = artifact_index.remove("doc-2");
    assert!(removed.is_some());
    let results = artifact_index
        .query("kubernetes", 5)
        .expect("query should succeed");
    assert!(results.is_empty() || results[0].doc_id != "doc-2");
    assert_eq!(artifact_index.tombstones(), vec!["doc-2"]);
}

#[test]
fn index_store_publishes_single_snapshot_by_default() {
    let mut index = IndexStore::new(PipelineOptions::default());
    index.upsert(sample_doc("doc-1", "docker install guide"));
    index.refresh().expect("refresh should succeed");

    let snapshot = index
        .memory_index_snapshot()
        .expect("snapshot should be published");
    assert!(!snapshot.is_segmented());
    assert_eq!(snapshot.segment_count(), 1);
}

#[test]
fn single_and_segmented_queries_share_ranking_and_filter_semantics() {
    let segmented_options = PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 8,
            routing_strategy: SegmentRoutingStrategy::SparseOverlap,
        },
        ..PipelineOptions::default()
    };
    let documents = [
        sample_doc_with_group("docker-1", "session-docker", "docker compose containers"),
        sample_doc_with_group("docker-2", "session-docker", "docker deployment guide"),
        sample_doc_with_group(
            "database-1",
            "session-database",
            "database migration schema",
        ),
    ];
    let mut single = IndexStore::new(PipelineOptions::default());
    let mut segmented = IndexStore::new(segmented_options);
    for document in documents {
        single.upsert(document.clone());
        segmented.upsert(document);
    }

    let single_ids = single
        .query("docker deployment", 5)
        .expect("single query should succeed")
        .into_iter()
        .map(|result| result.doc_id)
        .collect::<Vec<_>>();
    let segmented_ids = segmented
        .query("docker deployment", 5)
        .expect("segmented query should succeed")
        .into_iter()
        .map(|result| result.doc_id)
        .collect::<Vec<_>>();
    assert_eq!(segmented_ids, single_ids);

    let mut filters = std::collections::BTreeMap::new();
    filters.insert("missing-scope".to_string(), "true".to_string());
    assert!(single
        .query_filtered("docker", 5, &filters)
        .expect("single filtered query should succeed")
        .is_empty());
    assert!(segmented
        .query_filtered("docker", 5, &filters)
        .expect("segmented filtered query should succeed")
        .is_empty());
}

#[test]
fn segmented_routing_applies_filters_before_selecting_segments() {
    let options = PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 1,
            routing_strategy: SegmentRoutingStrategy::SparseOverlap,
        },
        ..PipelineOptions::default()
    };
    let mut single = IndexStore::new(PipelineOptions::default());
    let mut segmented = IndexStore::new(options);
    let mut excluded = sample_doc_with_group("session-a-doc", "session-a", "docker guide");
    excluded
        .filters
        .insert("scope".to_string(), "excluded".to_string());
    let mut allowed = sample_doc_with_group("session-b-doc", "session-b", "docker guide");
    allowed
        .filters
        .insert("scope".to_string(), "allowed".to_string());
    for document in [excluded, allowed] {
        single.upsert(document.clone());
        segmented.upsert(document);
    }
    let mut filters = std::collections::BTreeMap::new();
    filters.insert("scope".to_string(), "allowed".to_string());

    let single_results = single
        .query_filtered("docker", 1, &filters)
        .expect("single filtered query should succeed");
    let segmented_results = segmented
        .query_filtered("docker", 1, &filters)
        .expect("segmented filtered query should succeed");
    assert_eq!(
        single_results
            .iter()
            .map(|result| result.doc_id.as_str())
            .collect::<Vec<_>>(),
        vec!["session-b-doc"]
    );
    assert_eq!(
        segmented_results
            .iter()
            .map(|result| result.doc_id.as_str())
            .collect::<Vec<_>>(),
        vec!["session-b-doc"]
    );
}

#[test]
fn index_store_publishes_and_queries_segmented_snapshot() {
    let options = PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 1,
            routing_strategy: SegmentRoutingStrategy::SparseOverlap,
        },
        ..PipelineOptions::default()
    };
    let mut index = IndexStore::new(options);
    index.upsert(sample_doc_with_group(
        "docker-doc",
        "session-docker",
        "docker compose installation containers",
    ));
    index.upsert(sample_doc_with_group(
        "database-doc",
        "session-database",
        "postgres schema migration database",
    ));

    let results = index
        .query("docker containers", 5)
        .expect("query should succeed");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].doc_id, "docker-doc");

    let (_, _, query_diagnostics) = index
        .query_timed("docker containers", 5)
        .expect("timed query should succeed");
    assert_eq!(
        query_diagnostics.snapshot_generation,
        index.snapshot_revision()
    );
    let completeness = query_diagnostics
        .shard_completeness
        .expect("segmented diagnostics should expose completeness");
    assert_eq!(completeness.expected_segments, vec!["session-docker"]);
    assert!(completeness.is_complete());

    let snapshot = index
        .memory_index_snapshot()
        .expect("snapshot should be published");
    assert!(snapshot.is_segmented());
    assert_eq!(snapshot.segment_count(), 2);
    assert!(snapshot.single_index().is_none());
    assert_eq!(index.snapshot_revision(), 2);
    let output = match snapshot {
        MemoryIndexSnapshot::Segmented(segmented) => {
            segmented.query_with_diagnostics("docker containers", 5, 1)
        }
        MemoryIndexSnapshot::Single(_) => panic!("expected segmented snapshot"),
    };
    assert_eq!(
        output.diagnostics.snapshot_generation,
        index.snapshot_revision()
    );
    let enrichment_output = match snapshot {
        MemoryIndexSnapshot::Segmented(segmented) => {
            segmented
                .query_with_segment_enrichment_temporal_context_and_strategy(
                    "docker containers",
                    5,
                    1,
                    SegmentRoutingStrategy::SparseOverlap,
                    TemporalQueryContext::default(),
                )
                .0
        }
        MemoryIndexSnapshot::Single(_) => panic!("expected segmented snapshot"),
    };
    assert_eq!(
        enrichment_output.diagnostics.snapshot_generation,
        index.snapshot_revision()
    );

    let inspection = index.inspection();
    let snapshot = inspection
        .snapshot
        .expect("inspection should include snapshot");
    assert_eq!(snapshot.layout, "segmented");
    assert_eq!(snapshot.segment_count, 2);
    assert_eq!(snapshot.global_document_count, 2);
    assert_eq!(
        snapshot
            .segments
            .iter()
            .map(|segment| segment.segment_id.as_str())
            .collect::<Vec<_>>(),
        vec!["session-database", "session-docker"]
    );
    assert!(snapshot
        .segments
        .iter()
        .all(|segment| segment.document_count == 1));
}

#[test]
fn segmented_snapshot_generation_tracks_publication_revision() {
    let options = PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 2,
            routing_strategy: SegmentRoutingStrategy::SparseOverlap,
        },
        ..PipelineOptions::default()
    };
    let mut index = IndexStore::new(options);
    index.upsert(sample_doc_with_group(
        "doc-1",
        "session-1",
        "docker installation",
    ));
    index
        .query("docker", 1)
        .expect("initial query should succeed");

    let first_generation = match index.memory_index_snapshot().unwrap() {
        MemoryIndexSnapshot::Segmented(segmented) => segmented.generation(),
        MemoryIndexSnapshot::Single(_) => panic!("expected segmented snapshot"),
    };
    assert_eq!(first_generation, index.snapshot_revision());

    index.upsert(sample_doc_with_group(
        "doc-2",
        "session-2",
        "postgres database",
    ));
    index
        .query("postgres", 1)
        .expect("replacement query should succeed");

    let second_generation = match index.memory_index_snapshot().unwrap() {
        MemoryIndexSnapshot::Segmented(segmented) => segmented.generation(),
        MemoryIndexSnapshot::Single(_) => panic!("expected segmented snapshot"),
    };
    assert!(second_generation > first_generation);
    assert_eq!(second_generation, index.snapshot_revision());
}

#[test]
fn segmented_index_store_rebuilds_segments_after_upsert() {
    let options = PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 1,
            routing_strategy: SegmentRoutingStrategy::SparseOverlap,
        },
        ..PipelineOptions::default()
    };
    let mut index = IndexStore::new(options);
    index.upsert(sample_doc_with_group(
        "docker-doc",
        "session-docker",
        "docker compose installation",
    ));
    index.refresh().expect("initial refresh should succeed");
    assert_eq!(index.memory_index_snapshot().unwrap().segment_count(), 1);

    index.upsert(sample_doc_with_group(
        "database-doc",
        "session-database",
        "postgres schema migration",
    ));
    let results = index
        .query("postgres migration", 5)
        .expect("query should succeed");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].doc_id, "database-doc");
    assert_eq!(index.memory_index_snapshot().unwrap().segment_count(), 2);
}

#[test]
fn query_results_diversify_by_group_id() {
    let docs = vec![
        sample_doc_with_group("doc-a1", "group-a", "shared ranking token"),
        sample_doc_with_group("doc-a2", "group-a", "shared ranking token"),
        sample_doc_with_group("doc-a3", "group-a", "shared ranking token"),
        sample_doc_with_group("doc-a4", "group-a", "shared ranking token"),
        sample_doc_with_group("doc-b1", "group-b", "shared ranking token"),
        sample_doc_with_group("doc-b2", "group-b", "shared ranking token"),
        sample_doc_with_group("doc-b3", "group-b", "shared ranking token"),
        sample_doc_with_group("doc-b4", "group-b", "shared ranking token"),
    ];
    let index =
        build_query_snapshot(&docs, &PipelineOptions::default()).expect("index should build");
    let results = index.query("shared", 6);
    assert!(!results.is_empty());
    let mut counts: HashMap<String, usize> = HashMap::new();
    for result in results {
        let group = result.group_id.expect("group id should be preserved");
        *counts.entry(group).or_default() += 1;
    }
    assert!(counts.values().all(|count| *count <= 3));
    assert!(counts.len() >= 2);
}

#[test]
fn index_store_query_uses_incremental_lexical_updates() {
    let mut index = IndexStore::new(PipelineOptions::default());
    index.upsert(sample_doc("doc-1", "docker install guide"));
    let docker_results = index.query("docker", 5).expect("query should succeed");
    assert!(!docker_results.is_empty());
    assert_eq!(docker_results[0].doc_id, "doc-1");

    index.upsert(sample_doc("doc-2", "kubernetes cluster operations"));
    let kube_results = index.query("kubernetes", 5).expect("query should succeed");
    assert!(!kube_results.is_empty());
    assert_eq!(kube_results[0].doc_id, "doc-2");

    let docker_results = index.query("docker", 5).expect("query should succeed");
    assert!(!docker_results.is_empty());
    assert_eq!(docker_results[0].doc_id, "doc-1");
}

#[test]
fn refresh_is_idempotent_when_no_documents_change() {
    let mut index = IndexStore::new(PipelineOptions::default());
    index.upsert(sample_doc("doc-1", "refresh idempotence check"));

    index.refresh().expect("initial refresh should succeed");
    assert!(!index.is_dirty());
    let snapshot_revision = index.snapshot_revision();
    let store_revision = index.store_revision();

    index.refresh().expect("second refresh should succeed");
    assert_eq!(index.snapshot_revision(), snapshot_revision);
    assert_eq!(index.store_revision(), store_revision);
    assert!(!index.is_dirty());
}

#[test]
fn incremental_refresh_matches_full_rebuild() {
    fn segmented_options() -> PipelineOptions {
        PipelineOptions {
            memory_index_layout: MemoryIndexLayout::Segmented {
                query_top_n: 8,
                routing_strategy: SegmentRoutingStrategy::SparseOverlap,
            },
            ..PipelineOptions::default()
        }
    }
    fn manifest_of(store: &IndexStore) -> Vec<(String, Vec<String>)> {
        match store.memory_index_snapshot() {
            Some(MemoryIndexSnapshot::Segmented(segmented)) => {
                let mut entries: Vec<(String, Vec<String>)> = segmented
                    .manifest()
                    .segments
                    .into_iter()
                    .map(|entry| {
                        let mut doc_ids = entry.doc_ids;
                        doc_ids.sort();
                        (entry.segment_id, doc_ids)
                    })
                    .collect();
                entries.sort();
                entries
            }
            other => panic!(
                "expected segmented snapshot, got {}",
                if other.is_some() {
                    "a non-segmented snapshot"
                } else {
                    "no snapshot"
                }
            ),
        }
    }
    fn segment_records(store: &IndexStore) -> Vec<(String, serde_json::Value)> {
        // Record-level comparison: the stored records are exactly
        // deterministic, and scores are a pure function of these records plus
        // the global statistics, which are rebuilt from the same segment
        // searchers on both paths. Query results are compared separately
        // below — the query path is deterministic (float sums run in
        // sorted-key order and top-k ties break on doc_id).
        match store.memory_index_snapshot() {
            Some(MemoryIndexSnapshot::Segmented(segmented)) => {
                let mut out: Vec<(String, serde_json::Value)> = segmented
                    .segments
                    .iter()
                    .map(|segment| {
                        let records = serde_json::to_value(&segment.index.docs)
                            .expect("records should serialize");
                        (segment.segment_id.clone(), records)
                    })
                    .collect();
                out.sort_by(|left, right| left.0.cmp(&right.0));
                out
            }
            other => panic!(
                "expected segmented snapshot, got {}",
                if other.is_some() {
                    "a non-segmented snapshot"
                } else {
                    "no snapshot"
                }
            ),
        }
    }

    // Six segments of eight documents; the first refresh is a full build.
    let mut store = IndexStore::new(segmented_options());
    let mut final_docs = std::collections::HashMap::new();
    for segment in 0..6 {
        for doc in 0..8 {
            let id = format!("doc-{segment}-{doc}");
            let document = sample_doc_with_group(
                &id,
                &format!("session-{segment}"),
                &format!("segment {segment} document {doc} about rust testing pipelines"),
            );
            final_docs.insert(id, document.clone());
            store.upsert(document);
        }
    }
    store.refresh().expect("initial refresh should succeed");

    // Mutations covering every incremental case: re-processed content in
    // an existing segment, a new document in an existing segment, a brand
    // new segment, a removed document, and a removed segment.
    let updated = sample_doc_with_group(
        "doc-0-0",
        "session-0",
        "segment 0 document 0 rewritten about migration",
    );
    final_docs.insert("doc-0-0".to_string(), updated.clone());
    store.upsert(updated);
    let added = sample_doc_with_group(
        "doc-0-8",
        "session-0",
        "segment 0 document 8 about rust testing",
    );
    final_docs.insert("doc-0-8".to_string(), added.clone());
    store.upsert(added);
    let new_segment = sample_doc_with_group(
        "doc-6-0",
        "session-6",
        "segment 6 document 0 about pipelines",
    );
    final_docs.insert("doc-6-0".to_string(), new_segment.clone());
    store.upsert(new_segment);
    final_docs.remove("doc-1-0");
    store.remove("doc-1-0");
    for doc in 0..8 {
        let id = format!("doc-5-{doc}");
        final_docs.remove(&id);
        store.remove(&id);
    }
    store.refresh().expect("incremental refresh should succeed");

    let incremental_manifest = manifest_of(&store);
    let incremental_records = segment_records(&store);

    // Same final documents in a fresh store: full rebuild, no reuse.
    let mut fresh = IndexStore::new(segmented_options());
    let mut ids: Vec<String> = final_docs.keys().cloned().collect();
    ids.sort();
    for id in ids.iter() {
        fresh.upsert(final_docs[id].clone());
    }
    fresh.refresh().expect("full rebuild should succeed");

    // Control experiment: two identical full rebuilds must agree exactly.
    let mut fresh2 = IndexStore::new(segmented_options());
    for id in ids.iter() {
        fresh2.upsert(final_docs[id].clone());
    }
    fresh2
        .refresh()
        .expect("second full rebuild should succeed");
    assert_eq!(segment_records(&fresh), segment_records(&fresh2));

    assert_eq!(incremental_manifest, manifest_of(&fresh));
    assert_eq!(incremental_records, segment_records(&fresh));

    // Query results must also match bit-for-bit. Comparing the two identical
    // full rebuilds pins down query-path determinism; comparing the
    // incremental refresh against the rebuild pins down refresh correctness.
    fn projected(results: &[crate::index::SearchResult]) -> Vec<(String, u32)> {
        results
            .iter()
            .map(|result| (result.doc_id.clone(), result.score.to_bits()))
            .collect()
    }
    for query in ["rust testing", "pipelines", "migration"] {
        let incremental = projected(&store.query(query, 8).expect("query should succeed"));
        let rebuilt = projected(&fresh.query(query, 8).expect("query should succeed"));
        let rebuilt_again = projected(&fresh2.query(query, 8).expect("query should succeed"));
        assert_eq!(
            incremental, rebuilt,
            "incremental query results diverged for {query:?}"
        );
        assert_eq!(
            rebuilt, rebuilt_again,
            "two identical rebuilds diverged for {query:?}"
        );
    }
}

#[test]
fn index_store_remove_deletes_lexical_doc() {
    let mut index = IndexStore::new(PipelineOptions::default());
    index.upsert(sample_doc("doc-1", "redis cache operations"));
    let before_remove = index.query("redis", 5).expect("query should succeed");
    assert!(!before_remove.is_empty());
    assert_eq!(before_remove[0].doc_id, "doc-1");

    index.remove("doc-1");
    let after_remove = index.query("redis", 5).expect("query should succeed");
    assert!(after_remove.iter().all(|result| result.doc_id != "doc-1"));
}

#[test]
fn index_store_with_lexical_dir_persists_queries_across_instances() {
    let index_root = unique_temp_dir("lexical-root");
    let options = PipelineOptions {
        index_location: IndexLocation::Explicit(index_root.clone()),
        ..PipelineOptions::default()
    };

    let mut first = IndexStore::at_path(&index_root, options.clone())
        .expect("explicit-path store should initialize");
    first.upsert(sample_doc("doc-1", "persistent lexical search"));
    let first_results = first.query("persistent", 5).expect("query should succeed");
    assert!(!first_results.is_empty());
    assert_eq!(first_results[0].doc_id, "doc-1");
    drop(first);

    let mut second = reopen_store_after_writer_drop(&index_root, options);
    second.upsert(sample_doc("doc-1", "persistent lexical search"));
    let second_results = second.query("persistent", 5).expect("query should succeed");
    assert!(!second_results.is_empty());
    assert_eq!(second_results[0].doc_id, "doc-1");

    let _ = fs::remove_dir_all(index_root);
}

#[test]
fn resolve_store_paths_under_corpus_root() {
    let corpus_root = unique_temp_dir("corpus-root");
    let options = PipelineOptions {
        index_location: IndexLocation::UnderCorpusRoot,
        ..PipelineOptions::default()
    };
    let paths = resolve_store_paths(Some(&corpus_root), &options).expect("paths should resolve");
    assert_eq!(paths.root, Some(corpus_root.join(".lint-ai")));
    assert_eq!(
        paths.lexical_dir,
        Some(corpus_root.join(".lint-ai").join("lexical"))
    );
    assert_eq!(
        paths.semantic_dir,
        Some(corpus_root.join(".lint-ai").join("semantic"))
    );
    assert_eq!(
        paths.metadata_path,
        Some(corpus_root.join(".lint-ai").join("metadata.json"))
    );
}

#[test]
fn index_store_for_corpus_uses_corpus_local_lexical_dir() {
    let corpus_root = unique_temp_dir("corpus-store");
    let mut index = IndexStore::for_corpus(&corpus_root, PipelineOptions::default())
        .expect("corpus-backed store should initialize");
    index.upsert(sample_doc("doc-1", "corpus rooted lexical index"));
    let results = index.query("lexical", 5).expect("query should succeed");
    assert!(!results.is_empty());
    assert!(corpus_root.join(".lint-ai").join("lexical").exists());
    let _ = fs::remove_dir_all(corpus_root.join(".lint-ai"));
}

#[test]
fn index_store_for_corpus_writes_metadata() {
    let corpus_root = unique_temp_dir("corpus-metadata");
    let mut index = IndexStore::for_corpus(&corpus_root, PipelineOptions::default())
        .expect("corpus-backed store should initialize");
    index.upsert(sample_doc("doc-1", "metadata persistence test"));
    index.refresh().expect("refresh should succeed");

    let metadata_path = corpus_root.join(".lint-ai").join("metadata.json");
    assert!(metadata_path.exists());
    let metadata = fs::read_to_string(&metadata_path).expect("metadata should be readable");
    assert!(metadata.contains("\"schema_version\": 1"));
    assert!(metadata.contains("\"layout_version\": \"index-store-v1\""));

    let _ = fs::remove_dir_all(corpus_root.join(".lint-ai"));
}

#[test]
fn index_store_persists_and_reloads_semantic_state() {
    let corpus_root = unique_temp_dir("semantic-state");
    let mut first = IndexStore::for_corpus(&corpus_root, PipelineOptions::default())
        .expect("corpus-backed store should initialize");
    first.upsert(sample_doc("doc-1", "semantic persistence works"));
    let first_results = first.query("persistence", 5).expect("query should succeed");
    assert!(!first_results.is_empty());
    assert_eq!(first_results[0].doc_id, "doc-1");
    drop(first);

    let semantic_dir = corpus_root.join(".lint-ai").join("semantic");
    assert!(semantic_dir.join("records.json").exists());
    assert!(semantic_dir.join("core.bin").exists());

    let mut second =
        reopen_store_after_writer_drop(&corpus_root.join(".lint-ai"), PipelineOptions::default());
    let second_results = second
        .query("persistence", 5)
        .expect("query should succeed");
    assert!(!second_results.is_empty());
    assert_eq!(second_results[0].doc_id, "doc-1");

    let _ = fs::remove_dir_all(corpus_root.join(".lint-ai"));
}

#[test]
fn segmented_store_reloads_without_a_compatibility_core() {
    let index_root = unique_temp_dir("segmented-semantic-state");
    let options = PipelineOptions {
        index_location: IndexLocation::Explicit(index_root.clone()),
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 2,
            routing_strategy: SegmentRoutingStrategy::SparseOverlap,
        },
        ..PipelineOptions::default()
    };

    let mut first = IndexStore::at_path(&index_root, options.clone())
        .expect("segmented explicit-path store should initialize");
    first.upsert(sample_doc_with_group(
        "doc-1",
        "session-1",
        "segmented persistence works",
    ));
    first.upsert(sample_doc_with_group(
        "doc-2",
        "session-2",
        "manifest preserves segment assignment",
    ));
    assert!(!first
        .query("persistence", 5)
        .expect("query should succeed")
        .is_empty());
    drop(first);

    let semantic_dir = index_root.join("semantic");
    assert!(semantic_dir.join("records.json").exists());
    assert!(!semantic_dir.join("core.bin").exists());
    let manifest: SegmentManifest = serde_json::from_str(
        &fs::read_to_string(semantic_dir.join("segments.json"))
            .expect("segment manifest should be persisted"),
    )
    .expect("segment manifest should be valid JSON");
    assert_eq!(manifest.segments.len(), 2);

    // Prove that a valid manifest is authoritative rather than merely
    // reproducing the default group-id partitioning.
    let custom_manifest = SegmentManifest {
        generation: manifest.generation,
        segments: vec![crate::segments::SegmentManifestEntry {
            segment_id: "combined".to_string(),
            doc_ids: vec!["doc-1".to_string(), "doc-2".to_string()],
        }],
    };
    fs::write(
        semantic_dir.join("segments.json"),
        serde_json::to_string_pretty(&custom_manifest).expect("manifest should serialize"),
    )
    .expect("custom manifest should be writable");

    let mut second = reopen_store_after_writer_drop(&index_root, options);
    let results = second
        .query("persistence", 5)
        .expect("reloaded segmented query should succeed");
    assert_eq!(
        results.first().map(|result| result.doc_id.as_str()),
        Some("doc-1")
    );
    let snapshot = second
        .memory_index_snapshot()
        .expect("reloaded snapshot should be published");
    let diagnostics = match snapshot {
        MemoryIndexSnapshot::Segmented(segmented) => {
            segmented.query_with_diagnostics("manifest", 5, 1)
        }
        MemoryIndexSnapshot::Single(_) => panic!("expected segmented snapshot"),
    };
    assert_eq!(diagnostics.diagnostics.queried_segment_count, 1);
    assert_eq!(
        diagnostics.diagnostics.selected_segments[0].segment_id,
        "combined"
    );

    let _ = fs::remove_dir_all(index_root);
}

#[test]
fn adaptive_segmented_pipeline_expands_routed_segments() {
    let options = PipelineOptions {
        memory_index_layout: MemoryIndexLayout::AdaptiveSegmented {
            query_top_n: 1,
            max_query_n: 2,
            routing_strategy: SegmentRoutingStrategy::CoverageLocalDistinctiveness,
        },
        ..PipelineOptions::default()
    };
    let mut store = IndexStore::with_documents(
        options,
        vec![
            sample_doc_with_group("doc-a", "session-a", "GPA undergraduate record"),
            sample_doc_with_group("doc-b", "session-b", "graduate jewelry appointment"),
        ],
    );
    let prepared = PreparedQuery::new("GPA undergraduate graduate jewelry appointment");
    let (_, _, diagnostics) = store
        .query_prepared_timed(&prepared, 5, &std::collections::BTreeMap::new())
        .expect("adaptive pipeline query should succeed");
    assert_eq!(
        diagnostics
            .shard_completeness
            .as_ref()
            .map(|s| s.expected_segments.len()),
        Some(2)
    );
}

#[test]
fn adaptive_segmented_store_reloads_from_persisted_records() {
    let index_root = unique_temp_dir("adaptive-segmented");
    let options = PipelineOptions {
        index_location: IndexLocation::Explicit(index_root.clone()),
        memory_index_layout: MemoryIndexLayout::AdaptiveSegmented {
            query_top_n: 1,
            max_query_n: 2,
            routing_strategy: SegmentRoutingStrategy::CoverageLocalDistinctiveness,
        },
        ..PipelineOptions::default()
    };
    let mut first = IndexStore::at_path(&index_root, options.clone())
        .expect("adaptive store should initialize");
    first.upsert(sample_doc_with_group(
        "doc-a",
        "session-a",
        "GPA undergraduate record",
    ));
    first.upsert(sample_doc_with_group(
        "doc-b",
        "session-b",
        "graduate jewelry appointment",
    ));
    assert!(!first
        .query("GPA", 5)
        .expect("query should succeed")
        .is_empty());
    drop(first);

    let mut restored = reopen_store_after_writer_drop(&index_root, options);
    assert!(matches!(
        restored
            .memory_index_snapshot()
            .expect("adaptive snapshot should be published during reload"),
        MemoryIndexSnapshot::Segmented(_)
    ));
    let cached = restored
        .query_prepared_cached(
            &PreparedQuery::new("jewelry"),
            5,
            &std::collections::BTreeMap::new(),
        )
        .expect("cold-start cached query should succeed");
    assert!(!cached.is_empty());
    assert!(!restored
        .query("jewelry", 5)
        .expect("reloaded query should succeed")
        .is_empty());
    assert!(matches!(
        restored
            .memory_index_snapshot()
            .expect("snapshot should exist"),
        MemoryIndexSnapshot::Segmented(_)
    ));
    let _ = fs::remove_dir_all(index_root);
}

#[test]
fn segmented_dump_round_trips_without_retaining_a_global_snapshot() {
    let options = PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 2,
            routing_strategy: SegmentRoutingStrategy::SparseOverlap,
        },
        ..PipelineOptions::default()
    };
    let mut first = IndexStore::new(options.clone());
    first.upsert(sample_doc_with_group(
        "docker-doc",
        "session-docker",
        "docker compose installation",
    ));
    first.upsert(sample_doc_with_group(
        "database-doc",
        "session-database",
        "postgres schema migration",
    ));

    let dump = first.dump().expect("segmented dump should succeed");
    let mut restored =
        IndexStore::load_from_dump(dump, options).expect("segmented dump should load");

    assert!(restored
        .memory_index_snapshot()
        .expect("snapshot should be restored")
        .single_index()
        .is_none());
    let results = restored
        .query("docker compose", 5)
        .expect("restored segmented query should succeed");
    assert_eq!(
        results.first().map(|result| result.doc_id.as_str()),
        Some("docker-doc")
    );
}

#[test]
fn index_store_rejects_metadata_schema_mismatch() {
    let index_root = unique_temp_dir("schema-mismatch");
    fs::create_dir_all(&index_root).expect("index root should be creatable");
    fs::write(
        index_root.join("metadata.json"),
        r#"{"schema_version":999,"layout_version":"index-store-v1","crate_version":"0.1.6","index_location":"explicit"}"#,
    )
    .expect("metadata file should be writable");

    let result = IndexStore::at_path(&index_root, PipelineOptions::default());
    assert!(result.is_err());

    let _ = fs::remove_dir_all(index_root);
}

#[test]
fn reveals_bug_invalid_supersession_thresholds_are_silently_accepted() {
    let index_root = unique_temp_dir("invalid-supersession-options");
    let options = PipelineOptions {
        supersession: SupersessionOptions {
            suppress_confidence: 1.2,
            ..SupersessionOptions::default()
        },
        ..PipelineOptions::default()
    };

    let result = IndexStore::at_path(&index_root, options);

    match result {
        Ok(store) => {
            drop(store);
            panic!("invalid supersession thresholds must fail IndexStore construction");
        }
        Err(error) => assert!(
            error
                .to_string()
                .contains("confidence thresholds must be between 0 and 1"),
            "constructor should report the invalid threshold clearly: {error}"
        ),
    }

    let _ = fs::remove_dir_all(index_root);
}

#[test]
fn reveals_bug_fixed_candidate_window_returns_fewer_than_top_k_current_results() {
    let mut index = IndexStore::new(PipelineOptions::default());

    for i in 0..20 {
        let old_id = format!("old-{i}");
        let mut old = sample_doc(&old_id, "needle highrank highrank");
        old.filters.insert("semantic_scope".into(), "test".into());
        index.upsert(old);

        let replacement_id = format!("replacement-{i}");
        let mut replacement = sample_doc(&replacement_id, "replacement guidance");
        replacement
            .filters
            .insert("semantic_scope".into(), "test".into());
        replacement.filters.insert("supersedes_id".into(), old_id);
        index.upsert(replacement);
    }

    for i in 0..3 {
        let mut current = sample_doc(&format!("current-{i}"), "needle");
        current
            .filters
            .insert("semantic_scope".into(), "test".into());
        index.upsert(current);
    }

    let results = index
        .query("needle highrank", 3)
        .expect("query should succeed");
    assert_eq!(
        results.len(),
        3,
        "eligible current results should fill top_k before ranking"
    );
    assert!(
        results
            .iter()
            .all(|result| result.doc_id.starts_with("current-")),
        "superseded documents must not consume candidate slots: {results:?}"
    );
}

#[test]
fn index_store_does_not_delete_invalid_lexical_directory() {
    let index_root = unique_temp_dir("invalid-lexical-dir");
    let lexical_dir = index_root.join("lexical");
    fs::create_dir_all(&lexical_dir).expect("lexical dir should be creatable");
    let sentinel = lexical_dir.join("keep.txt");
    fs::write(&sentinel, "do not delete").expect("sentinel file should be writable");

    let result = IndexStore::at_path(&index_root, PipelineOptions::default());
    assert!(result.is_err());
    assert!(sentinel.exists());

    let _ = fs::remove_dir_all(index_root);
}

#[test]
fn index_store_new_falls_back_to_in_memory_on_invalid_explicit_root() {
    let index_root = unique_temp_dir("new-fallback");
    fs::write(&index_root, "not a directory").expect("invalid root file should be writable");

    let options = PipelineOptions {
        index_location: IndexLocation::Explicit(index_root.clone()),
        ..PipelineOptions::default()
    };
    let mut index = IndexStore::new(options);
    index.upsert(sample_doc("doc-1", "fallback in memory works"));

    let results = index.query("fallback", 5).expect("query should succeed");
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, "doc-1");
    assert!(!index_root.join("lexical").exists());
    assert!(index.store_paths.root.is_none());
    assert!(index.store_paths.lexical_dir.is_none());
    assert!(index.store_paths.semantic_dir.is_none());
    assert!(index.store_paths.metadata_path.is_none());

    let _ = fs::remove_file(index_root);
}

#[test]
fn index_store_new_falls_back_to_in_memory_when_corpus_root_is_missing() {
    let options = PipelineOptions {
        index_location: IndexLocation::UnderCorpusRoot,
        ..PipelineOptions::default()
    };

    let mut index = IndexStore::new(options);
    index.upsert(sample_doc("doc-1", "missing corpus root fallback"));

    let results = index.query("corpus", 5).expect("query should succeed");
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, "doc-1");
    assert!(index.store_paths.root.is_none());
    assert!(index.store_paths.lexical_dir.is_none());
    assert!(index.store_paths.semantic_dir.is_none());
    assert!(index.store_paths.metadata_path.is_none());
}

#[test]
fn chunk_lifecycle_increments_version_and_tracks_latest() {
    let mut index = IndexStore::new(PipelineOptions::default());
    index.upsert(sample_doc("doc-1", "first chunk body"));
    index.refresh().expect("refresh should succeed");

    let first_record = index
        .records()
        .into_iter()
        .find(|record| record.doc_id == "doc-1")
        .expect("doc-1 record should exist");
    assert_eq!(first_record.section_chunks.len(), 1);
    let first_chunk_id = first_record.section_chunks[0].chunk_id.clone();

    let first_meta = index
        .chunk_lifecycle()
        .into_iter()
        .find(|meta| meta.chunk_id == first_chunk_id)
        .expect("first chunk lifecycle should exist");
    assert_eq!(first_meta.version, 1);
    assert!(first_meta.is_latest);
    assert!(first_meta.supersedes_chunk_id.is_none());

    index.upsert(sample_doc("doc-1", "second chunk body"));
    index.refresh().expect("refresh should succeed");

    let second_record = index
        .records()
        .into_iter()
        .find(|record| record.doc_id == "doc-1")
        .expect("doc-1 record should exist");
    assert_eq!(second_record.section_chunks.len(), 1);
    let second_chunk_id = second_record.section_chunks[0].chunk_id.clone();
    assert_ne!(first_chunk_id, second_chunk_id);

    let metas = index
        .chunk_lifecycle()
        .into_iter()
        .filter(|meta| meta.doc_id == "doc-1")
        .collect::<Vec<_>>();
    assert_eq!(metas.len(), 2);

    let latest = metas
        .iter()
        .find(|meta| meta.chunk_id == second_chunk_id)
        .expect("new chunk metadata should exist");
    assert!(latest.is_latest);
    assert_eq!(latest.version, 2);
    assert_eq!(
        latest.supersedes_chunk_id.as_deref(),
        Some(first_chunk_id.as_str())
    );

    let previous = metas
        .iter()
        .find(|meta| meta.chunk_id == first_chunk_id)
        .expect("previous chunk metadata should exist");
    assert!(!previous.is_latest);
    assert_eq!(previous.version, 1);
}

#[test]
fn section_chunks_inherit_parent_timestamp_and_document_lifecycle_is_derived() {
    let mut doc = sample_doc("doc-1", "timestamp inheritance body");
    doc.timestamp = Some("2024-05-10T12:34:56Z".to_string());

    let mut index = IndexStore::new(PipelineOptions::default());
    index.upsert(doc);
    index.refresh().expect("refresh should succeed");

    let record = index
        .records()
        .into_iter()
        .find(|record| record.doc_id == "doc-1")
        .expect("doc-1 record should exist");
    assert_eq!(record.section_chunks.len(), 1);
    assert_eq!(
        record.section_chunks[0].timestamp.as_deref(),
        Some("2024-05-10T12:34:56Z")
    );

    let doc_lifecycle = index
        .document_lifecycle()
        .into_iter()
        .find(|meta| meta.doc_id == "doc-1")
        .expect("doc lifecycle should exist");
    assert_eq!(doc_lifecycle.chunk_count, 1);
    assert_eq!(doc_lifecycle.latest_chunk_ids.len(), 1);
    assert!(doc_lifecycle.is_latest);
    assert!(doc_lifecycle.updated_at_ms > 0);
}

#[test]
fn chunk_lifecycle_is_removed_when_doc_is_removed() {
    let mut index = IndexStore::new(PipelineOptions::default());
    index.upsert(sample_doc("doc-1", "redis lifecycle cleanup"));
    index.refresh().expect("refresh should succeed");
    assert!(index
        .chunk_lifecycle()
        .into_iter()
        .any(|meta| meta.doc_id == "doc-1"));

    index.remove("doc-1");
    index.refresh().expect("refresh should succeed");
    assert!(!index
        .chunk_lifecycle()
        .into_iter()
        .any(|meta| meta.doc_id == "doc-1"));
}

fn segmented_test_options() -> PipelineOptions {
    PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 1,
            routing_strategy: SegmentRoutingStrategy::SparseOverlap,
        },
        ..PipelineOptions::default()
    }
}

/// Raw pointers to each segment's shared index: stable across refreshes only
/// when the segment is reused rather than rebuilt.
fn segment_index_ptrs(index: &IndexStore) -> Vec<*const crate::index::MemoryIndex> {
    use std::sync::Arc;
    match index
        .memory_index_snapshot()
        .expect("snapshot should exist")
    {
        MemoryIndexSnapshot::Segmented(segmented) => segmented
            .segments
            .iter()
            .map(|segment| Arc::as_ptr(&segment.index))
            .collect(),
        MemoryIndexSnapshot::Single(_) => panic!("expected a segmented snapshot"),
    }
}

#[test]
fn doc_record_content_hash_matches_built_record() {
    let options = PipelineOptions::default();
    let doc = sample_doc("doc-1", "hash agreement probe");
    let record = build_doc_record(&doc, &options).expect("build should succeed");
    assert_eq!(record.content_hash.len(), 64);
    assert_eq!(
        record.content_hash,
        doc_record_content_hash(&doc, &options),
        "stamped hash must equal the cheap pre-build hash"
    );
}

#[test]
fn doc_record_content_hash_is_stable_and_sensitive() {
    let options = PipelineOptions::default();
    let doc = sample_doc("doc-1", "hello world");
    let baseline = doc_record_content_hash(&doc, &options);
    assert_eq!(baseline, doc_record_content_hash(&doc, &options));

    let mut changed = doc.clone();
    changed.content = "hello mars".to_string();
    changed.doc_length = changed.content.len();
    assert_ne!(baseline, doc_record_content_hash(&changed, &options));

    let mut filtered = doc.clone();
    filtered
        .filters
        .insert("team".to_string(), "infra".to_string());
    assert_ne!(baseline, doc_record_content_hash(&filtered, &options));

    let mut concept = doc.clone();
    concept.concept = "different concept".to_string();
    assert_ne!(
        baseline,
        doc_record_content_hash(&concept, &options),
        "concept feeds the key-entity ranker"
    );

    let mut headed = doc.clone();
    headed.headings = vec!["Changed".to_string()];
    assert_ne!(baseline, doc_record_content_hash(&headed, &options));

    let mut claim_options = PipelineOptions::default();
    claim_options.claim_extraction = true;
    assert_ne!(
        baseline,
        doc_record_content_hash(&doc, &claim_options),
        "extraction options must invalidate the hash"
    );
}

#[test]
fn doc_record_content_hash_build_version_invalidates() {
    let options = PipelineOptions::default();
    let doc = sample_doc("doc-1", "hello world");
    let v1 = doc_record_content_hash_with_version(&doc, &options, 1);
    let v2 = doc_record_content_hash_with_version(&doc, &options, 2);
    assert_ne!(
        v1, v2,
        "a build version bump must invalidate every stored hash"
    );
    assert_eq!(
        doc_record_content_hash(&doc, &options),
        doc_record_content_hash_with_version(&doc, &options, DOC_RECORD_BUILD_VERSION),
        "public wrapper must agree with the versioned core at the current build version"
    );
}

#[test]
fn identical_content_reupsert_skips_record_rebuild() {
    let mut index = IndexStore::new(segmented_test_options());
    let doc = sample_doc_with_group("doc-1", "group-a", "the quick brown fox jumps");
    index.upsert(doc.clone());
    index.refresh().expect("initial refresh should succeed");
    let before_ptrs = segment_index_ptrs(&index);
    let before_hash = index
        .record_by_id("doc-1")
        .expect("record should exist")
        .content_hash
        .clone();
    assert!(!before_hash.is_empty());

    // Re-upsert byte-identical content: the record rebuild is skipped, so all
    // segments are shared with the previous snapshot.
    index.upsert(doc);
    index.refresh().expect("refresh should succeed");
    assert_eq!(
        before_ptrs,
        segment_index_ptrs(&index),
        "identical re-upsert must not rebuild any segment"
    );
    assert_eq!(
        index
            .record_by_id("doc-1")
            .expect("record should exist")
            .content_hash,
        before_hash
    );
    assert!(
        !index.inspection().dirty,
        "skipped doc must be cleared from dirty state"
    );
    let results = index
        .query("quick brown fox", 5)
        .expect("query should succeed");
    assert!(results.iter().any(|result| result.doc_id == "doc-1"));
}

#[test]
fn changed_content_reupsert_rebuilds_record() {
    let mut index = IndexStore::new(segmented_test_options());
    index.upsert(sample_doc_with_group(
        "doc-1",
        "group-a",
        "the quick brown fox jumps",
    ));
    index.refresh().expect("initial refresh should succeed");
    let before_ptrs = segment_index_ptrs(&index);
    let before_hash = index
        .record_by_id("doc-1")
        .expect("record should exist")
        .content_hash
        .clone();

    index.upsert(sample_doc_with_group(
        "doc-1",
        "group-a",
        "a completely different sentence about databases",
    ));
    index.refresh().expect("refresh should succeed");
    assert_ne!(
        before_ptrs,
        segment_index_ptrs(&index),
        "changed content must rebuild the segment"
    );
    let record = index.record_by_id("doc-1").expect("record should exist");
    assert_ne!(record.content_hash, before_hash);
    assert_eq!(
        record.content,
        "a completely different sentence about databases"
    );
}

#[test]
fn legacy_record_without_content_hash_is_rebuilt() {
    let mut index = IndexStore::new(segmented_test_options());
    let doc = sample_doc_with_group("doc-1", "group-a", "legacy hash simulation");
    index.upsert(doc.clone());
    index.refresh().expect("initial refresh should succeed");
    let before_ptrs = segment_index_ptrs(&index);

    // Simulate a record persisted before hashing existed.
    index.clear_record_content_hash_for_test("doc-1");
    index.upsert(doc);
    index.refresh().expect("refresh should succeed");
    assert_ne!(
        before_ptrs,
        segment_index_ptrs(&index),
        "empty content hash must force a rebuild, never a reuse"
    );
    assert!(
        !index
            .record_by_id("doc-1")
            .expect("record should exist")
            .content_hash
            .is_empty(),
        "rebuilt record must carry a fresh hash"
    );
}
