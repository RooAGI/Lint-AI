use crate::adapters::markdown::build_markdown_corpus;
use anyhow::Result;
use deunicode::deunicode;
use petgraph::graph::{DiGraph, NodeIndex};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Clone)]
pub struct Page {
    /// Absolute path to the file.
    pub path: String,
    /// Path relative to the root used for traversal.
    pub rel_path: String,
    /// Normalized concept name derived from the file name.
    pub concept: String,
    /// Raw concept name derived from the file name.
    pub raw_concept: String,
    /// Full file contents.
    pub content: String,
    /// Normalized outbound link targets.
    pub links: HashSet<String>,
    /// Markdown headings extracted from the file.
    pub headings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ChunkNode {
    /// Stable chunk id (<rel_path>::<idx>)
    pub chunk_id: String,
    /// Parent document rel path.
    pub doc_rel_path: String,
    /// Section heading for this chunk.
    pub heading: String,
    /// 1-based start line in parent doc.
    pub start_line: usize,
    /// 1-based end line in parent doc.
    pub end_line: usize,
    /// Chunk text.
    pub content: String,
}

#[derive(Debug, Clone, Copy)]
pub enum ChunkEdgeKind {
    /// Sequence edge between adjacent chunks in same doc.
    Next,
    /// Cross-doc edge projected from doc-level links.
    DocLink,
}

#[derive(Debug, Clone, Copy)]
pub enum EntityEdgeKind {
    /// Two entities co-occurred in at least one document.
    CoOccurs,
    /// A document-level link projected to concept entities.
    DocLink,
}

pub struct Graph {
    /// All parsed pages.
    pub pages: Vec<Page>,
    /// Tier 0 ingestion records for each parsed document.
    pub tier0_records: Vec<Tier0Record>,
    /// Map of concept -> node index in the graph.
    pub index: HashMap<String, NodeIndex>,
    /// Directed graph of page links.
    pub graph: DiGraph<String, ()>,
    /// All chunk nodes across the corpus.
    pub chunks: Vec<ChunkNode>,
    /// Map of chunk_id -> chunk node index.
    pub chunk_index: HashMap<String, NodeIndex>,
    /// Map of doc rel_path -> ordered chunk ids.
    pub doc_to_chunks: HashMap<String, Vec<String>>,
    /// Directed chunk graph.
    pub chunk_graph: DiGraph<String, ChunkEdgeKind>,
    /// Map of canonical entity -> node index.
    pub entity_index: HashMap<String, NodeIndex>,
    /// Canonical entity graph.
    pub entity_graph: DiGraph<String, EntityEdgeKind>,
    /// Document -> canonical entities mentioned in that document.
    pub doc_entities: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier0Record {
    /// Stable document id for ingestion.
    pub id: String,
    /// Source path for the document (relative to scan root).
    pub source: String,
    /// Last-modified timestamp (unix seconds as string), when available.
    pub timestamp: Option<String>,
    /// Author/agent from frontmatter, when available.
    pub author_agent: Option<String>,
    /// Document length in bytes.
    pub doc_length: usize,
    /// Additional lightweight metadata for later tiers.
    pub metadata: HashMap<String, String>,
}

/// Normalize a concept string for matching (unicode + deunicode + case fold).
pub(crate) fn normalize_concept(s: &str) -> String {
    let normalized: String = s
        .nfc()
        .collect::<String>()
        .trim()
        .to_lowercase()
        .replace('_', " ")
        .replace('-', " ");
    deunicode(&normalized).to_lowercase()
}

pub(crate) fn strip_anchor(target: &str) -> &str {
    target.split('#').next().unwrap_or(target)
}

pub(crate) fn concept_from_link_target(target: &str) -> Option<String> {
    let target = strip_anchor(target).trim();
    if target.is_empty() || target.starts_with("http://") || target.starts_with("https://") {
        return None;
    }
    if target.starts_with("mailto:") || target.starts_with("tel:") {
        return None;
    }
    if target.starts_with('#') {
        return None;
    }

    let target = target.trim_end_matches('/');
    let path = Path::new(target);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or(target);
    if stem.is_empty() {
        return None;
    }
    Some(normalize_concept(stem))
}

pub(crate) fn rel_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn chunk_page_sections(content: &str, rel_path: &str) -> Vec<ChunkNode> {
    let heading_re = Regex::new(r"(?m)^#{1,6}\s+(.*)$").expect("valid heading regex");
    let mut chunks = Vec::new();
    let mut last_start = 0usize;
    let mut current_heading = "(document)".to_string();
    let mut idx = 0usize;

    for cap in heading_re.captures_iter(content) {
        let m = match cap.get(0) {
            Some(v) => v,
            None => continue,
        };
        if m.start() > last_start {
            let body = content[last_start..m.start()].trim();
            if !body.is_empty() {
                let start_line = content[..last_start].lines().count().max(1);
                let end_line = content[..m.start()].lines().count().max(start_line);
                chunks.push(ChunkNode {
                    chunk_id: format!("{}::{}", rel_path, idx),
                    doc_rel_path: rel_path.to_string(),
                    heading: current_heading.clone(),
                    start_line,
                    end_line,
                    content: body.to_string(),
                });
                idx += 1;
            }
        }
        current_heading = cap
            .get(1)
            .map(|v| v.as_str().trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "(section)".to_string());
        last_start = m.end();
    }

    let tail = content[last_start..].trim();
    if !tail.is_empty() {
        let start_line = content[..last_start].lines().count().max(1);
        let end_line = content.lines().count().max(start_line);
        chunks.push(ChunkNode {
            chunk_id: format!("{}::{}", rel_path, idx),
            doc_rel_path: rel_path.to_string(),
            heading: current_heading,
            start_line,
            end_line,
            content: tail.to_string(),
        });
    }

    if chunks.is_empty() {
        chunks.push(ChunkNode {
            chunk_id: format!("{}::0", rel_path),
            doc_rel_path: rel_path.to_string(),
            heading: "(document)".to_string(),
            start_line: 1,
            end_line: content.lines().count().max(1),
            content: content.to_string(),
        });
    }
    chunks
}

fn extract_doc_entities(content: &str, concept: &str) -> HashSet<String> {
    let mut out: HashSet<String> = HashSet::new();
    let concept_norm = normalize_concept(concept);
    if !concept_norm.is_empty() {
        out.insert(concept_norm);
    }

    let title_case_re =
        Regex::new(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b").expect("valid entity regex");
    for cap in title_case_re.captures_iter(content).take(100) {
        if let Some(m) = cap.get(1) {
            let e = normalize_concept(m.as_str());
            if e.len() >= 3 {
                out.insert(e);
            }
        }
    }
    let acronym_re = Regex::new(r"\b([A-Z]{2,8})\b").expect("valid acronym regex");
    for m in acronym_re.find_iter(content).take(80) {
        let e = normalize_concept(m.as_str());
        if e.len() >= 2 {
            out.insert(e);
        }
    }
    out
}

impl Graph {
    /// Build a graph from the given path, applying size and depth limits.
    ///
    /// Example:
    /// ```no_run
    /// use lint_ai::graph::Graph;
    /// let graph = Graph::build("docs", 5_000_000, 50_000, 20, 100_000_000).unwrap();
    /// println!("pages: {}", graph.pages.len());
    /// ```
    pub fn build(
        path: &str,
        max_bytes: usize,
        max_files: usize,
        max_depth: usize,
        max_total_bytes: usize,
    ) -> Result<Self> {
        let root = Path::new(path);
        let corpus = build_markdown_corpus(root, max_bytes, max_files, max_depth, max_total_bytes)?;
        let pages: Vec<Page> = corpus
            .pages
            .into_iter()
            .map(|page| Page {
                path: page.path,
                rel_path: page.rel_path,
                concept: page.concept,
                raw_concept: page.raw_concept,
                content: page.content,
                links: page.links,
                headings: page.headings,
            })
            .collect();
        let tier0_records = corpus.tier0_records;

        let mut graph = DiGraph::<String, ()>::new();
        let mut index: HashMap<String, NodeIndex> = HashMap::new();
        for page in &pages {
            let node = graph.add_node(page.rel_path.clone());
            index.insert(page.concept.clone(), node);
        }
        for page in &pages {
            if let Some(&from) = index.get(&page.concept) {
                for link in &page.links {
                    if let Some(&to) = index.get(link) {
                        graph.add_edge(from, to, ());
                    }
                }
            }
        }
        // Build chunk graph on top of parsed pages.
        let mut chunks = Vec::new();
        let mut chunk_index: HashMap<String, NodeIndex> = HashMap::new();
        let mut doc_to_chunks: HashMap<String, Vec<String>> = HashMap::new();
        let mut chunk_graph = DiGraph::<String, ChunkEdgeKind>::new();

        // Concept -> rel_path for link projection onto first chunk of target doc.
        let concept_to_rel_path: HashMap<String, String> = pages
            .iter()
            .map(|p| (p.concept.clone(), p.rel_path.clone()))
            .collect();

        for page in &pages {
            let page_chunks = chunk_page_sections(&page.content, &page.rel_path);
            let mut ordered_ids = Vec::new();
            for ch in page_chunks {
                let node = chunk_graph.add_node(ch.chunk_id.clone());
                chunk_index.insert(ch.chunk_id.clone(), node);
                ordered_ids.push(ch.chunk_id.clone());
                chunks.push(ch);
            }
            // Intra-doc sequence edges.
            for pair in ordered_ids.windows(2) {
                if let [a, b] = pair {
                    if let (Some(&from), Some(&to)) = (chunk_index.get(a), chunk_index.get(b)) {
                        chunk_graph.add_edge(from, to, ChunkEdgeKind::Next);
                    }
                }
            }
            doc_to_chunks.insert(page.rel_path.clone(), ordered_ids);
        }

        // Project doc links to first chunk -> first chunk.
        for page in &pages {
            let Some(from_chunks) = doc_to_chunks.get(&page.rel_path) else {
                continue;
            };
            let Some(from_first) = from_chunks.first() else {
                continue;
            };
            let Some(&from_node) = chunk_index.get(from_first) else {
                continue;
            };
            for link in &page.links {
                let Some(target_rel) = concept_to_rel_path.get(link) else {
                    continue;
                };
                let Some(target_chunks) = doc_to_chunks.get(target_rel) else {
                    continue;
                };
                let Some(target_first) = target_chunks.first() else {
                    continue;
                };
                if let Some(&to_node) = chunk_index.get(target_first) {
                    chunk_graph.add_edge(from_node, to_node, ChunkEdgeKind::DocLink);
                }
            }
        }

        // Build entity graph.
        let mut entity_graph = DiGraph::<String, EntityEdgeKind>::new();
        let mut entity_index: HashMap<String, NodeIndex> = HashMap::new();
        let mut doc_entities: HashMap<String, Vec<String>> = HashMap::new();

        for page in &pages {
            let mut ents = extract_doc_entities(&page.content, &page.concept)
                .into_iter()
                .collect::<Vec<_>>();
            ents.sort();
            ents.dedup();
            for e in &ents {
                if !entity_index.contains_key(e) {
                    let n = entity_graph.add_node(e.clone());
                    entity_index.insert(e.clone(), n);
                }
            }
            doc_entities.insert(page.rel_path.clone(), ents);
        }

        for ents in doc_entities.values() {
            for i in 0..ents.len() {
                for j in (i + 1)..ents.len() {
                    let a = &ents[i];
                    let b = &ents[j];
                    let (Some(&na), Some(&nb)) = (entity_index.get(a), entity_index.get(b)) else {
                        continue;
                    };
                    entity_graph.add_edge(na, nb, EntityEdgeKind::CoOccurs);
                }
            }
        }

        // Project doc links as concept-entity edges.
        for page in &pages {
            let from_concept = normalize_concept(&page.concept);
            let Some(&from_node) = entity_index.get(&from_concept) else {
                continue;
            };
            for link in &page.links {
                let to_concept = normalize_concept(link);
                if let Some(&to_node) = entity_index.get(&to_concept) {
                    entity_graph.add_edge(from_node, to_node, EntityEdgeKind::DocLink);
                }
            }
        }

        Ok(Self {
            pages,
            tier0_records,
            index,
            graph,
            chunks,
            chunk_index,
            doc_to_chunks,
            chunk_graph,
            entity_index,
            entity_graph,
            doc_entities,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::markdown::parse_frontmatter_kv;

    #[test]
    fn normalize_concept_basic() {
        assert_eq!(normalize_concept("Group-Messages"), "group messages");
        assert_eq!(normalize_concept("Group_Messages"), "group messages");
        assert_eq!(normalize_concept("Café-Menu"), "cafe menu");
    }

    #[test]
    fn link_target_concept_parsing() {
        assert_eq!(
            concept_from_link_target("docs/channels/discord.md").as_deref(),
            Some("discord")
        );
        assert_eq!(
            concept_from_link_target("docs/channels/discord.md#setup").as_deref(),
            Some("discord")
        );
        assert_eq!(concept_from_link_target("https://example.com"), None);
        assert_eq!(concept_from_link_target("mailto:test@example.com"), None);
    }

    #[test]
    fn parse_frontmatter_metadata() {
        let content = "---\nauthor: lint-bot\nagent: reviewer-v1\ntopic: docs\n---\n# Title";
        let parsed = parse_frontmatter_kv(content);
        assert_eq!(parsed.get("author").map(|s| s.as_str()), Some("lint-bot"));
        assert_eq!(parsed.get("agent").map(|s| s.as_str()), Some("reviewer-v1"));
        assert_eq!(parsed.get("topic").map(|s| s.as_str()), Some("docs"));
    }
}
