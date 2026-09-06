from pathlib import Path

p = Path("src/engine.rs")
s = p.read_text()

# Use the canonical adapter converter so Tier0 relationship metadata survives.
start = s.index("fn graph_to_source_documents(graph: &Graph) -> Vec<SourceDocument> {")
end = s.index("\nfn surface_forms", start)
s = s[:start] + """fn graph_to_source_documents(graph: &Graph) -> Vec<SourceDocument> {
    crate::adapters::graph_to_source_documents(graph)
}
""" + s[end:]

# Build the lightweight graph/source-doc view even on a ranking-cache hit so
# semantic current-state policy is always computed from fresh source metadata.
start = s.index("        let lexical_dir = query_cache_lexical_dir(&cache_settings);")
end_marker = "        let query_value = args.llm_context.as_deref().or(args.query.as_deref());"
end = s.index(end_marker, start)
replacement = """        let lexical_dir = query_cache_lexical_dir(&cache_settings);
        let mut graph = Graph::build(
            &args.path,
            args.max_bytes,
            args.max_files,
            args.max_depth,
            args.max_total_bytes,
        )?;
        if !cfg.ignore_paths.is_empty() {
            let ignore = normalize_list(&cfg.ignore_paths);
            graph.pages.retain(|p| {
                let rel = p.rel_path.to_lowercase();
                !ignore.iter().any(|pat| rel.contains(pat))
            });
            let retained: HashSet<String> =
                graph.pages.iter().map(|p| p.rel_path.clone()).collect();
            graph.tier0_records.retain(|r| retained.contains(&r.source));
        }
        let source_docs = graph_to_source_documents(&graph);
        let semantic_relations = crate::semantic_relations::SemanticRelationStore::try_from_documents(
            source_docs.iter(),
            crate::semantic_relations::SupersessionOptions::default(),
        )?;

        let index = if let Some(cached) =
            load_cached_query_index(&cache_settings, &corpus_fingerprint)
        {
            cached
        } else {
            let built = build_memory_index(
                &graph,
                &args.tier1_ner_provider,
                &args.spacy_model,
                &args.tier1_term_ranker,
                &args.chunk_strategy,
                args.chunk_lines,
                args.chunk_overlap,
                args.chunk_target_tokens,
                args.chunk_max_tokens,
                Some(&lexical_dir),
            )?;
            if let Err(err) = save_cached_query_index(&cache_settings, &corpus_fingerprint, &built)
            {
                eprintln!("warning: unable to persist query cache: {}", err);
            }
            built
        };
"""
s = s[:start] + replacement + s[end:]

old = "            let temporal_context = prepared.temporal_context();"
new = """            let mut temporal_context = prepared.temporal_context();
            let historical_query = crate::semantic_relations::is_historical_query(query);
            let has_superseded = source_docs.iter().any(|doc| {
                semantic_relations.document_state(&doc.doc_id).status
                    == Some(crate::semantic_relations::SemanticStatus::Superseded)
            });
            let allowed_doc_ids = if historical_query || !has_superseded {
                None
            } else {
                Some(
                    source_docs
                        .iter()
                        .filter(|doc| {
                            semantic_relations.document_state(&doc.doc_id).status
                                != Some(crate::semantic_relations::SemanticStatus::Superseded)
                        })
                        .map(|doc| doc.doc_id.clone())
                        .collect::<HashSet<_>>(),
                )
            };
            temporal_context.allowed_doc_ids = allowed_doc_ids.as_ref();"""
if old not in s:
    raise SystemExit("temporal context marker not found")
s = s.replace(old, new, 1)

old = """                let results = index
                    .query_with_temporal_context(
                        &search_query,
                        DEFAULT_QUERY_TOP_K,
                        temporal_context,
                    )
                    .0;
                let elapsed_ms = started.elapsed().as_millis();"""
new = """                let mut results = index
                    .query_with_temporal_context(
                        &search_query,
                        DEFAULT_QUERY_TOP_K,
                        temporal_context,
                    )
                    .0;
                for result in &mut results {
                    let state = semantic_relations.document_state(&result.doc_id);
                    result.semantic_status = if historical_query
                        && state.status == Some(crate::semantic_relations::SemanticStatus::Superseded)
                    {
                        Some(crate::semantic_relations::SemanticStatus::Historical)
                    } else {
                        state.status
                    };
                    result.superseded_by = state.superseded_by;
                    result.relation_confidence = state.relation_confidence;
                    result.relation_evidence = state.evidence;
                }
                let elapsed_ms = started.elapsed().as_millis();"""
if old not in s:
    raise SystemExit("normal query marker not found")
s = s.replace(old, new, 1)
p.write_text(s)

# Recommended/configured quantities are not counts of retrieved evidence rows.
p = Path("src/aggregation.rs")
s = p.read_text()
marker = """pub fn classify_aggregate_intent(query: &str) -> Option<AggregateIntent> {
    let q = query.to_lowercase();
"""
replacement = """pub fn classify_aggregate_intent(query: &str) -> Option<AggregateIntent> {
    let q = query.to_lowercase();
    let asks_for_recommended_quantity = q.contains("how many")
        && [
            " should ",
            " should we",
            " should i",
            " allowed ",
            " maximum ",
            " minimum ",
            " limit ",
        ]
        .iter()
        .any(|cue| q.contains(cue));
    if asks_for_recommended_quantity {
        return None;
    }
"""
if marker not in s:
    raise SystemExit("aggregate classifier marker not found")
s = s.replace(marker, replacement, 1)
p.write_text(s)

Path("tests/cli_supersession.rs").write_text(r'''use serde_json::Value;
use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir() -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("lint-ai-cli-supersession-{nanos}"));
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn cli_query_and_llm_context_suppress_explicitly_superseded_markdown() {
    let root = temp_dir();
    fs::write(
        root.join("decision-a.md"),
        "# Gateway Retry Policy\n\nGateway timeout retry attempts: 5.\n",
    )
    .unwrap();
    fs::write(
        root.join("decision-b.md"),
        "---\nsupersedes: decision-a.md\n---\n# Gateway Retry Policy\n\nGateway timeout retry attempts: 2.\n",
    )
    .unwrap();

    let bin = env!("CARGO_BIN_EXE_lint-ai");
    let query = "How many retry attempts should we use for gateway timeouts?";

    let output = Command::new(bin)
        .current_dir(&root)
        .args(["--query", query, root.to_str().unwrap()])
        .output()
        .expect("CLI query should run");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).expect("query should return JSON");
    let results = payload["results"].as_array().expect("results array");
    assert_eq!(
        results.len(),
        1,
        "superseded evidence must be filtered: {payload:#}"
    );
    assert_eq!(results[0]["doc_id"], "decision-b.md");
    assert_eq!(results[0]["semantic_status"], "current");
    assert!(
        payload["aggregation"].is_null(),
        "recommended quantity must not count evidence rows: {payload:#}"
    );

    let output = Command::new(bin)
        .current_dir(&root)
        .args([
            "--llm-context",
            query,
            "--result-count",
            "5",
            root.to_str().unwrap(),
        ])
        .output()
        .expect("LLM context query should run");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value =
        serde_json::from_slice(&output.stdout).expect("LLM context should return JSON");
    let chunks = payload["top_chunks"].as_array().expect("top_chunks array");
    assert!(!chunks.is_empty());
    assert!(
        chunks.iter().all(|chunk| chunk["doc_id"] != "decision-a.md"),
        "superseded evidence leaked into LLM context: {payload:#}"
    );
    assert!(chunks.iter().any(|chunk| {
        chunk["doc_id"] == "decision-b.md"
            && chunk["text"]
                .as_str()
                .unwrap_or("")
                .contains("retry attempts: 2")
    }));

    let _ = fs::remove_dir_all(root);
}
''')
