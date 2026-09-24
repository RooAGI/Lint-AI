use super::{
    ChunkLifecycleMeta, IndexLocation, MemoryIndexLayout, MemoryIndexSegmentInspection,
    MemoryIndexSnapshot, MemoryIndexSnapshotInspection, PersistedDocRecord,
    PersistedSemanticRecords, PipelineOptions, StoreMetadata, StorePaths, CHUNK_LIFECYCLE_FILE,
    SEGMENT_MANIFEST_FILE, SEMANTIC_CORE_FILE, SEMANTIC_RECORDS_FILE, STORE_LAYOUT_VERSION,
    STORE_SCHEMA_VERSION,
};
use crate::index::{DocRecord, MemoryIndex, QueryDiagnostics, QueryTimings, SearchResult};
use crate::query_plan::PreparedQuery;
use crate::query_semantics::{
    parse_reference_date, resolve_anchor_window, resolve_temporal_anchor,
};
use crate::segments::SegmentManifest;
use crate::semantic_relations::SemanticRelationStore;
use crate::source::SourceDocument;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
pub(crate) fn execute_prepared_on_snapshot_parts(
    options: &PipelineOptions,
    source_docs: &HashMap<String, SourceDocument>,
    records: &HashMap<String, DocRecord>,
    semantic_relations: &SemanticRelationStore,
    snapshot: Option<&MemoryIndexSnapshot>,
    prepared: &PreparedQuery,
    top_k: usize,
    filters: &std::collections::BTreeMap<String, String>,
) -> Result<(Vec<SearchResult>, QueryTimings, QueryDiagnostics)> {
    let profile = std::env::var_os("LINT_AI_QUERY_TIMINGS").is_some();
    let query_started = std::time::Instant::now();
    let Some(snapshot) = snapshot else {
        return Ok((
            Vec::new(),
            QueryTimings::default(),
            QueryDiagnostics::default(),
        ));
    };
    let filter_started = std::time::Instant::now();
    let filter_bitmap = match snapshot {
        MemoryIndexSnapshot::Single(index) => index.doc_bitmap_matching_filters(filters),
        MemoryIndexSnapshot::Segmented(_) => None,
    };
    let filter_allowed = if filters.is_empty() {
        None
    } else if let MemoryIndexSnapshot::Segmented(segmented) = snapshot {
        // Build the routing scope from the per-segment bitmap postings. This
        // preserves filter-aware segment selection without scanning records.
        // A one-segment snapshot has no routing decision to make; the local
        // bitmap is sufficient and avoids materializing 23k string IDs.
        if segmented.segment_count() == 1 {
            None
        } else {
            segmented.doc_ids_matching_filters(filters)
        }
    } else if let MemoryIndexSnapshot::Single(index) = snapshot {
        // The bitmap is the canonical filter representation. Keep the legacy
        // string allow-list only when semantic suppression must be intersected
        // with it; otherwise this avoids a full ID materialization per query.
        if semantic_relations.is_empty() {
            None
        } else {
            index.doc_ids_matching_filters(filters)
        }
    } else {
        Some(
            records
                .iter()
                .filter(|(_, record)| {
                    filters.iter().all(|(key, value)| {
                        record
                            .filters
                            .get(key)
                            .is_some_and(|actual| actual == value)
                    })
                })
                .map(|(doc_id, _)| doc_id.clone())
                .collect::<HashSet<_>>(),
        )
    };
    let filter_segment_bitmaps = match snapshot {
        MemoryIndexSnapshot::Segmented(segmented) => {
            Some(segmented.doc_bitmaps_matching_filters(filters))
        }
        MemoryIndexSnapshot::Single(_) => None,
    };
    if profile {
        eprintln!(
            "query_timing filter_ms={:.3}",
            filter_started.elapsed().as_secs_f64() * 1000.0
        );
    }
    let document_ids = source_docs.keys().cloned().collect::<Vec<_>>();
    let allowed_doc_ids =
        prepared.semantic_allowed_doc_ids(filter_allowed, semantic_relations, &document_ids);

    let (results, timings, diagnostics) = match snapshot {
        MemoryIndexSnapshot::Segmented(segmented) => {
            let (query_top_n, max_query_n, routing_strategy, adaptive) =
                match &options.memory_index_layout {
                    MemoryIndexLayout::Segmented {
                        query_top_n,
                        routing_strategy,
                    } => (*query_top_n, *query_top_n, *routing_strategy, false),
                    MemoryIndexLayout::AdaptiveSegmented {
                        query_top_n,
                        max_query_n,
                        routing_strategy,
                    } => (*query_top_n, *max_query_n, *routing_strategy, true),
                    MemoryIndexLayout::Single => {
                        anyhow::bail!("segmented snapshot published under single-index options")
                    }
                };
            let mut context = prepared.temporal_context();
            // Resolve the query's relative time anchor ("last Tuesday",
            // "10 days ago") against the reference date so temporal scoring
            // centers on the anchor instead of the question date itself, and
            // pre-filtering can restrict routing to the anchor window (a
            // point window for "last Tuesday", the [anchor, reference] range
            // for "in the past two months"). No-op when the query carries no
            // resolvable anchor phrase or no reference date.
            let (anchor_date_string, anchor_window) = prepared
                .analysis()
                .temporal
                .as_ref()
                .and_then(|temporal| {
                    let reference = prepared.reference_date().and_then(parse_reference_date)?;
                    let anchor = resolve_temporal_anchor(&temporal.phrase, reference)?;
                    let window = resolve_anchor_window(
                        &temporal.phrase,
                        prepared.search_query(),
                        reference,
                    )?;
                    Some((anchor.format("%Y-%m-%d").to_string(), window))
                })
                .unzip();
            // Only overwrite the context's anchor when this path resolves one:
            // temporal_context() may already carry a seeded absolute anchor
            // (no reference clock needed), and a None here must not wipe it.
            if anchor_date_string.is_some() {
                context.anchor_date = anchor_date_string.as_deref();
            }
            if anchor_window.is_some() {
                context.anchor_window = anchor_window;
            }
            context.allowed_doc_ids = allowed_doc_ids.as_ref();
            context.allowed_doc_bitmap = filter_bitmap.as_ref();
            context.allowed_segment_doc_bitmaps = filter_segment_bitmaps.as_ref();
            let started = std::time::Instant::now();
            let multi_segment = segmented.segment_count() > 1;
            let output = if !multi_segment {
                let local_bitmap = filter_segment_bitmaps
                    .as_ref()
                    .and_then(|maps| maps.values().next());
                let mut local_context = context;
                local_context.allowed_doc_ids = None;
                local_context.allowed_doc_bitmap = local_bitmap;
                let (results, _) = segmented
                    .query_single_segment(
                        prepared.search_query(),
                        top_k,
                        local_context,
                        prepared.reference_date(),
                    )
                    .unwrap_or_default();
                crate::segments::SegmentQueryOutput {
                    results,
                    diagnostics: Default::default(),
                }
            } else if adaptive {
                segmented
                    .query_with_adaptive_segment_enrichment_temporal_context_and_strategy(
                        prepared.search_query(),
                        top_k,
                        query_top_n.max(1),
                        max_query_n.max(query_top_n).max(1),
                        routing_strategy,
                        context,
                        prepared.reference_date(),
                    )
                    .0
            } else {
                segmented.query_with_temporal_context_at_and_diagnostics_and_strategy(
                    prepared.search_query(),
                    top_k,
                    query_top_n.max(1),
                    routing_strategy,
                    context,
                    prepared.reference_date(),
                )
            };
            // The corpus-wide ("global") arm is fused with the routed arm via
            // reciprocal rank fusion when `fuse_global_arm` is set: a routed
            // segment arm can miss when the router selects the wrong segments,
            // so its ranking is fused with an all-segments ranking. Label-free
            // (ranks only), so the two arms' score scales never crowd each
            // other out. With a strong router the global arm rescues little
            // and costs most of the query latency, so it can be disabled.
            let fuse_global = options.fuse_global_arm && multi_segment;
            let (results, global_diagnostics) = if fuse_global {
                let global_output = segmented
                    .query_all_segments_with_temporal_context_and_diagnostics(
                        prepared.search_query(),
                        top_k,
                        context,
                    );
                let fused = crate::segments::reciprocal_rank_fusion(
                    &[output.results.as_slice(), global_output.results.as_slice()],
                    top_k,
                );
                (fused, Some(global_output.diagnostics))
            } else {
                (output.results, None)
            };
            let timings = QueryTimings {
                total_ms: started.elapsed().as_secs_f64() * 1000.0,
                ..QueryTimings::default()
            };
            // Diagnostics account for both arms: the fused ranking is drawn
            // from the routed arm's candidates plus the global arm's.
            let global_merged = global_diagnostics
                .as_ref()
                .map(|d| d.merged_result_count)
                .unwrap_or(0);
            let global_queried = global_diagnostics
                .as_ref()
                .map(|d| d.queried_segment_count)
                .unwrap_or(0);
            if profile {
                eprintln!(
                    "query_timing segmented_ms={:.3} total_ms={:.3} segments={}",
                    timings.total_ms,
                    query_started.elapsed().as_secs_f64() * 1000.0,
                    output.diagnostics.queried_segment_count + global_queried
                );
            }
            let diagnostics = QueryDiagnostics {
                candidates: output.diagnostics.merged_result_count + global_merged,
                snapshot_generation: output.diagnostics.snapshot_generation,
                shard_completeness: output.diagnostics.shard_completeness,
                ..QueryDiagnostics::default()
            };
            (
                prepared.annotate_semantic_results(results, semantic_relations, top_k),
                timings,
                diagnostics,
            )
        }
        MemoryIndexSnapshot::Single(index) => {
            let (results, timings, diagnostics) =
                if filter_bitmap.is_some() && semantic_relations.is_empty() {
                    prepared.execute_on_index_with_bitmap(index, top_k, filter_bitmap.as_ref())
                } else {
                    prepared.execute_on_index(index, top_k, allowed_doc_ids.as_ref())
                };
            (
                prepared.annotate_semantic_results(results, semantic_relations, top_k),
                timings,
                diagnostics,
            )
        }
    };

    Ok((results, timings, diagnostics))
}

pub(crate) fn inspect_memory_index_snapshot(
    snapshot: &MemoryIndexSnapshot,
) -> MemoryIndexSnapshotInspection {
    match snapshot {
        MemoryIndexSnapshot::Single(index) => {
            let mut doc_ids = index.docs.keys().cloned().collect::<Vec<_>>();
            doc_ids.sort();
            MemoryIndexSnapshotInspection {
                layout: "single".to_string(),
                segment_count: 1,
                global_document_count: index.docs.len(),
                segments: vec![MemoryIndexSegmentInspection {
                    segment_id: "global".to_string(),
                    document_count: doc_ids.len(),
                    doc_ids,
                    profile_term_count: 0,
                    profile_entity_count: 0,
                    profile_topic_count: 0,
                    profile_local_memory_count: 0,
                }],
            }
        }
        MemoryIndexSnapshot::Segmented(index) => MemoryIndexSnapshotInspection {
            layout: "segmented".to_string(),
            segment_count: index.segments.len(),
            global_document_count: index
                .segments
                .iter()
                .map(|segment| segment.doc_ids.len())
                .sum(),
            segments: index
                .segments
                .iter()
                .map(|segment| {
                    let (term_count, entity_count, topic_count, local_memory_count) =
                        index.routing_summary_counts(&segment.segment_id);
                    MemoryIndexSegmentInspection {
                        segment_id: segment.segment_id.clone(),
                        document_count: segment.doc_ids.len(),
                        doc_ids: segment.doc_ids.clone(),
                        profile_term_count: term_count,
                        profile_entity_count: entity_count,
                        profile_topic_count: topic_count,
                        profile_local_memory_count: local_memory_count,
                    }
                })
                .collect(),
        },
    }
}

fn index_location_name(index_location: &IndexLocation) -> String {
    match index_location {
        IndexLocation::InMemory => "in_memory".to_string(),
        IndexLocation::UnderCorpusRoot => "under_corpus_root".to_string(),
        IndexLocation::Explicit(_) => "explicit".to_string(),
    }
}

fn current_store_metadata(options: &PipelineOptions) -> StoreMetadata {
    StoreMetadata {
        schema_version: STORE_SCHEMA_VERSION,
        layout_version: STORE_LAYOUT_VERSION.to_string(),
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
        index_location: index_location_name(&options.index_location),
    }
}

fn load_store_metadata(metadata_path: &Path) -> Result<Option<StoreMetadata>> {
    if !metadata_path.exists() {
        return Ok(None);
    }
    let contents = fs::read_to_string(metadata_path)?;
    Ok(Some(serde_json::from_str(&contents)?))
}

pub(crate) fn persist_store_metadata(
    store_paths: &StorePaths,
    options: &PipelineOptions,
) -> Result<()> {
    let Some(metadata_path) = store_paths.metadata_path.as_ref() else {
        return Ok(());
    };
    let metadata = current_store_metadata(options);
    write_text_file_atomic(metadata_path, &serde_json::to_string_pretty(&metadata)?)?;
    Ok(())
}

pub(crate) fn ensure_store_metadata(
    store_paths: &StorePaths,
    options: &PipelineOptions,
) -> Result<()> {
    let Some(metadata_path) = store_paths.metadata_path.as_ref() else {
        return Ok(());
    };
    if let Some(root) = store_paths.root.as_ref() {
        fs::create_dir_all(root)?;
    }
    if let Some(existing) = load_store_metadata(metadata_path)? {
        let current = current_store_metadata(options);
        if existing.schema_version != current.schema_version {
            anyhow::bail!(
                "index schema mismatch at {}: found {}, expected {}",
                metadata_path.display(),
                existing.schema_version,
                current.schema_version
            );
        }
        if existing.layout_version != current.layout_version {
            anyhow::bail!(
                "index layout mismatch at {}: found {}, expected {}",
                metadata_path.display(),
                existing.layout_version,
                current.layout_version
            );
        }
        return Ok(());
    }
    persist_store_metadata(store_paths, options)
}

pub(crate) fn is_directory_empty(dir: &Path) -> Result<bool> {
    let mut entries = fs::read_dir(dir)?;
    Ok(entries.next().is_none())
}

fn semantic_records_path(store_paths: &StorePaths) -> Option<PathBuf> {
    store_paths
        .semantic_dir
        .as_ref()
        .map(|dir| dir.join(SEMANTIC_RECORDS_FILE))
}

fn semantic_core_path(store_paths: &StorePaths) -> Option<PathBuf> {
    store_paths
        .semantic_dir
        .as_ref()
        .map(|dir| dir.join(SEMANTIC_CORE_FILE))
}

fn segment_manifest_path(store_paths: &StorePaths) -> Option<PathBuf> {
    store_paths
        .semantic_dir
        .as_ref()
        .map(|dir| dir.join(SEGMENT_MANIFEST_FILE))
}

pub(crate) fn load_segment_manifest(store_paths: &StorePaths) -> Result<Option<SegmentManifest>> {
    let Some(path) = segment_manifest_path(store_paths) else {
        return Ok(None);
    };
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(&fs::read_to_string(path)?)?))
}

pub(crate) fn persist_segment_manifest(
    store_paths: &StorePaths,
    manifest: &SegmentManifest,
) -> Result<()> {
    let Some(semantic_dir) = store_paths.semantic_dir.as_ref() else {
        return Ok(());
    };
    fs::create_dir_all(semantic_dir)?;
    let path = segment_manifest_path(store_paths)
        .expect("segment manifest path should exist when semantic dir exists");
    write_text_file_atomic(&path, &serde_json::to_string_pretty(manifest)?)?;
    Ok(())
}

fn chunk_lifecycle_path(store_paths: &StorePaths) -> Option<PathBuf> {
    store_paths
        .semantic_dir
        .as_ref()
        .map(|dir| dir.join(CHUNK_LIFECYCLE_FILE))
}

pub(crate) fn chunk_lineage_key(doc_id: &str, chunk: &crate::index::SectionChunk) -> String {
    format!(
        "{}::{}::{}::{}",
        doc_id,
        chunk.start_line,
        chunk.end_line,
        chunk.heading.trim().to_lowercase()
    )
}

pub(crate) fn current_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub(crate) fn source_document_from_record(record: &DocRecord) -> SourceDocument {
    SourceDocument {
        doc_id: record.doc_id.clone(),
        source: record.source.clone(),
        content: record.content.clone(),
        concept: record
            .headings
            .first()
            .cloned()
            .or_else(|| record.probable_topic.clone())
            .unwrap_or_else(|| record.doc_id.clone()),
        group_id: record.group_id.clone(),
        filters: record.filters.clone(),
        headings: record.headings.clone(),
        links: record.doc_links.clone(),
        timestamp: record.timestamp.clone(),
        doc_length: record.doc_length,
        author_agent: record.author_agent.clone(),
        key_phrases: Vec::new(),
    }
}

type SemanticState = (
    HashMap<String, SourceDocument>,
    HashMap<String, DocRecord>,
    HashMap<String, ChunkLifecycleMeta>,
    Option<MemoryIndex>,
);

pub(crate) fn load_semantic_state(
    store_paths: &StorePaths,
    layout: &MemoryIndexLayout,
) -> Result<SemanticState> {
    let Some(records_path) = semantic_records_path(store_paths) else {
        return Ok((HashMap::new(), HashMap::new(), HashMap::new(), None));
    };
    if !records_path.exists() {
        return Ok((HashMap::new(), HashMap::new(), HashMap::new(), None));
    }
    let core_path = semantic_core_path(store_paths);
    if matches!(layout, MemoryIndexLayout::Single)
        && !core_path.as_ref().is_some_and(|p| p.exists())
    {
        return Ok((HashMap::new(), HashMap::new(), HashMap::new(), None));
    }

    let persisted: PersistedSemanticRecords =
        serde_json::from_str(&fs::read_to_string(&records_path)?)?;
    if persisted.schema_version != STORE_SCHEMA_VERSION {
        anyhow::bail!(
            "semantic record schema mismatch at {}: found {}, expected {}",
            records_path.display(),
            persisted.schema_version,
            STORE_SCHEMA_VERSION
        );
    }
    if persisted.layout_version != STORE_LAYOUT_VERSION {
        anyhow::bail!(
            "semantic record layout mismatch at {}: found {}, expected {}",
            records_path.display(),
            persisted.layout_version,
            STORE_LAYOUT_VERSION
        );
    }

    let restored_records = persisted
        .records
        .into_iter()
        .map(Into::into)
        .collect::<Vec<DocRecord>>();
    let snapshot = match layout {
        MemoryIndexLayout::Single => Some(MemoryIndex::load_with_binary_core(
            restored_records.clone(),
            core_path
                .as_ref()
                .expect("single layout requires a core path"),
            None,
            false,
        )?),
        MemoryIndexLayout::Segmented { .. } | MemoryIndexLayout::AdaptiveSegmented { .. } => None,
    };
    let mut source_docs = HashMap::new();
    let mut records = HashMap::new();
    for record in restored_records {
        source_docs.insert(record.doc_id.clone(), source_document_from_record(&record));
        records.insert(record.doc_id.clone(), record);
    }
    let mut chunk_lifecycle = persisted
        .chunk_lifecycle
        .into_iter()
        .map(|meta| (meta.chunk_id.clone(), meta))
        .collect::<HashMap<_, _>>();
    if chunk_lifecycle.is_empty() {
        for record in records.values() {
            for chunk in &record.section_chunks {
                let chunk_id = chunk.chunk_id.clone();
                chunk_lifecycle.insert(
                    chunk_id.clone(),
                    ChunkLifecycleMeta {
                        chunk_id,
                        doc_id: record.doc_id.clone(),
                        lineage_key: chunk_lineage_key(&record.doc_id, chunk),
                        version: 1,
                        is_latest: true,
                        supersedes_chunk_id: None,
                        updated_at_ms: current_time_ms(),
                        change_reason: Some("bootstrap".to_string()),
                    },
                );
            }
        }
    } else if let Some(chunk_lifecycle_path) = chunk_lifecycle_path(store_paths) {
        if !chunk_lifecycle_path.exists() {
            write_text_file_atomic(
                &chunk_lifecycle_path,
                &serde_json::to_string_pretty(
                    &chunk_lifecycle.values().cloned().collect::<Vec<_>>(),
                )?,
            )?;
        }
    }
    Ok((source_docs, records, chunk_lifecycle, snapshot))
}

pub(crate) fn persist_semantic_state(
    store_paths: &StorePaths,
    snapshot: Option<&MemoryIndex>,
    records_map: &HashMap<String, DocRecord>,
    chunk_lifecycle_map: &HashMap<String, ChunkLifecycleMeta>,
) -> Result<()> {
    let Some(semantic_dir) = store_paths.semantic_dir.as_ref() else {
        return Ok(());
    };
    fs::create_dir_all(semantic_dir)?;
    let records_path = semantic_records_path(store_paths)
        .expect("records path should exist when semantic dir exists");
    let mut records = records_map.values().cloned().collect::<Vec<_>>();
    records.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
    let payload = PersistedSemanticRecords {
        schema_version: STORE_SCHEMA_VERSION,
        layout_version: STORE_LAYOUT_VERSION.to_string(),
        records: records.into_iter().map(PersistedDocRecord::from).collect(),
        chunk_lifecycle: chunk_lifecycle_map
            .values()
            .cloned()
            .collect::<Vec<ChunkLifecycleMeta>>(),
    };
    write_text_file_atomic(&records_path, &serde_json::to_string_pretty(&payload)?)?;
    if let Some(lifecycle_path) = chunk_lifecycle_path(store_paths) {
        let mut lifecycle = chunk_lifecycle_map
            .values()
            .cloned()
            .collect::<Vec<ChunkLifecycleMeta>>();
        lifecycle.sort_by(|a, b| a.chunk_id.cmp(&b.chunk_id));
        write_text_file_atomic(&lifecycle_path, &serde_json::to_string_pretty(&lifecycle)?)?;
    }
    if let Some(snapshot) = snapshot {
        let core_path = semantic_core_path(store_paths)
            .expect("core path should exist when persisting a single index");
        save_binary_core_atomic(snapshot, &core_path)?;
    }
    Ok(())
}

fn ensure_safe_output_path(path: &Path) -> Result<()> {
    if path.is_dir() {
        anyhow::bail!("refusing to write: output path is a directory");
    }
    if let Ok(meta) = fs::symlink_metadata(path) {
        if meta.file_type().is_symlink() {
            anyhow::bail!("refusing to write: output path is a symlink");
        }
    }
    if let Some(parent) = path.parent() {
        let mut cur = if parent.is_absolute() {
            PathBuf::from("/")
        } else {
            std::env::current_dir()?
        };
        for comp in parent.components() {
            use std::path::Component;
            match comp {
                Component::RootDir | Component::CurDir => continue,
                Component::ParentDir => {
                    anyhow::bail!("refusing to write: parent traversal is not allowed")
                }
                Component::Normal(seg) => {
                    cur.push(seg);
                    if let Ok(meta) = fs::symlink_metadata(&cur) {
                        if meta.file_type().is_symlink() {
                            anyhow::bail!(
                                "refusing to write: parent path component is a symlink ({})",
                                cur.display()
                            );
                        }
                    }
                }
                Component::Prefix(_) => {}
            }
        }
    }
    Ok(())
}

fn atomic_temp_path(path: &Path) -> Result<PathBuf> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("output");
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    for attempt in 0..1000u32 {
        let candidate = parent.join(format!(".{}.{}.{}.tmp", stem, pid, nanos + attempt as u128));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    anyhow::bail!(
        "unable to allocate temporary file path for {}",
        path.display()
    )
}

fn write_text_file_atomic(path: &Path, content: &str) -> Result<()> {
    ensure_safe_output_path(path)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temp_path = atomic_temp_path(path)?;
    {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp_path)?;
        file.write_all(content.as_bytes())?;
        file.sync_all().ok();
    }
    fs::rename(&temp_path, path)?;
    Ok(())
}

fn save_binary_core_atomic(snapshot: &MemoryIndex, path: &Path) -> Result<()> {
    ensure_safe_output_path(path)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temp_path = atomic_temp_path(path)?;
    snapshot.save_binary_core(&temp_path)?;
    fs::rename(&temp_path, path)?;
    Ok(())
}
