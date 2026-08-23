use anyhow::Result;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use crate::config::normalize_list;
use crate::graph::{Graph, Tier0Record};
use crate::source::SourceDocument;

#[derive(Debug, Clone)]
pub struct AdapterInput<'a> {
    pub root: &'a Path,
    pub max_bytes: usize,
    pub max_files: usize,
    pub max_depth: usize,
    pub max_total_bytes: usize,
}

pub trait SourceAdapter {
    fn name(&self) -> &'static str;
    fn supports(&self, path: &Path) -> bool;
    fn ingest(&self, input: &AdapterInput<'_>) -> Result<Vec<SourceDocument>>;
}

pub mod markdown;

pub fn default_source_adapters() -> Vec<Box<dyn SourceAdapter>> {
    vec![Box::new(markdown::MarkdownAdapter)]
}

/// Build the project graph for a corpus root.
pub fn build_project_graph(input: &AdapterInput<'_>) -> Result<Graph> {
    Graph::build(
        &input.root.to_string_lossy(),
        input.max_bytes,
        input.max_files,
        input.max_depth,
        input.max_total_bytes,
    )
}

/// Drop pages whose relative path contains any of the configured ignore
/// fragments, along with the Tier0 records that described them.
pub fn apply_ignore_paths(mut graph: Graph, ignore_paths: &[String]) -> Graph {
    if ignore_paths.is_empty() {
        return graph;
    }
    let ignore = normalize_list(ignore_paths);
    graph.pages.retain(|p| {
        let rel = p.rel_path.to_lowercase();
        !ignore.iter().any(|pat| rel.contains(pat))
    });
    let retained: HashSet<String> = graph.pages.iter().map(|p| p.rel_path.clone()).collect();
    graph.tier0_records.retain(|r| retained.contains(&r.source));
    graph
}

/// Convert a project graph into the source documents an index store ingests.
pub fn graph_to_source_documents(graph: &Graph) -> Vec<SourceDocument> {
    let mut tier0_by_source = HashMap::<String, &Tier0Record>::new();
    for record in &graph.tier0_records {
        if tier0_by_source
            .insert(record.source.clone(), record)
            .is_some()
        {
            debug_assert!(false, "duplicate Tier0 source: {}", record.source);
        }
    }
    let concept_to_rel: HashMap<String, String> = graph
        .pages
        .iter()
        .map(|p| (p.concept.clone(), p.rel_path.clone()))
        .collect();

    graph
        .pages
        .iter()
        .map(|p| {
            let t0 = tier0_by_source.get(&p.rel_path).copied();
            SourceDocument {
                doc_id: p.rel_path.clone(),
                source: p.rel_path.clone(),
                content: p.content.clone(),
                concept: p.raw_concept.clone(),
                group_id: None,
                filters: BTreeMap::new(),
                headings: p.headings.clone(),
                links: p
                    .links
                    .iter()
                    .filter_map(|c| concept_to_rel.get(c).cloned())
                    .collect(),
                timestamp: t0.and_then(|r| r.timestamp.clone()),
                doc_length: t0.map(|r| r.doc_length).unwrap_or(p.content.len()),
                author_agent: t0.and_then(|r| r.author_agent.clone()),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("lint-ai-adapters-{name}-{nanos}"));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn ignore_paths_drop_vendored_documents_and_their_tier0_records() {
        let root = temp_dir("ignore-paths");
        fs::write(root.join("readme.md"), "# Readme\nthe project itself").unwrap();
        fs::create_dir_all(root.join("node_modules").join("left-pad")).unwrap();
        fs::write(
            root.join("node_modules").join("left-pad").join("readme.md"),
            "# Left Pad\nthe project itself",
        )
        .unwrap();

        let input = AdapterInput {
            root: &root,
            max_bytes: 5_000_000,
            max_files: 50_000,
            max_depth: 20,
            max_total_bytes: 100_000_000,
        };
        let unfiltered = build_project_graph(&input).unwrap();
        assert_eq!(unfiltered.pages.len(), 2);

        let filtered = apply_ignore_paths(unfiltered, &["node_modules".to_string()]);
        let documents = graph_to_source_documents(&filtered);
        assert_eq!(documents.len(), 1);
        assert!(!documents[0].source.contains("node_modules"));
        assert!(!filtered
            .tier0_records
            .iter()
            .any(|record| record.source.contains("node_modules")));

        fs::remove_dir_all(&root).unwrap();
    }
}
