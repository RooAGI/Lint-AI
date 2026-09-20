use super::Tier0IndexFile;
use crate::graph::{normalize_concept, Graph, Tier0Record};
use crate::index::MemoryIndex;
use anyhow::Result;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn write_tier0_index(
    path: &str,
    records: &[Tier0Record],
    scanned_path: &str,
) -> Result<String> {
    let mut output_path = Path::new(path).to_path_buf();
    if output_path.extension().is_none() {
        output_path.set_extension("json");
    }
    let mut documents_by_id: BTreeMap<String, Tier0Record> = BTreeMap::new();
    for record in records {
        documents_by_id.insert(record.id.clone(), record.clone());
    }
    let generated_at_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let payload = Tier0IndexFile {
        tier: "tier0".to_string(),
        generated_at_unix,
        path: scanned_path.to_string(),
        document_count: documents_by_id.len(),
        documents_by_id,
    };
    let json = serde_json::to_string_pretty(&payload)?;
    safe_write_text_file(&output_path.display().to_string(), &json)?;
    Ok(output_path.display().to_string())
}

fn dot_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\"', "\\\"")
}

fn js_string_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        // Prevent HTML parser from terminating inline <script> blocks.
        .replace("</", "<\\/")
}

pub(crate) fn safe_write_text_file(out_path: &str, content: &str) -> Result<()> {
    let path = Path::new(out_path);
    if path.is_dir() {
        anyhow::bail!("refusing to write: output path is a directory");
    }
    if let Ok(meta) = fs::symlink_metadata(path) {
        if meta.file_type().is_symlink() {
            anyhow::bail!("refusing to write: output path is a symlink");
        }
    }
    if let Some(parent) = path.parent() {
        // Reject symlink components in the existing parent chain.
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
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(path, content)?;
    Ok(())
}

pub(crate) fn export_graph_dot(graph: &Graph, out_path: &str) -> Result<String> {
    let mut lines = Vec::new();
    lines.push("digraph lint_ai_graph {".to_string());
    lines.push("  rankdir=LR;".to_string());
    lines.push("  node [shape=box, style=rounded];".to_string());

    let mut concept_to_rel: HashMap<String, String> = HashMap::new();
    for page in &graph.pages {
        concept_to_rel.insert(page.concept.clone(), page.rel_path.clone());
    }

    for page in &graph.pages {
        let node_id = format!("n_{}", dot_escape(&page.concept.replace(' ', "_")));
        let label = format!(
            "{}\\n({})",
            dot_escape(&page.rel_path),
            dot_escape(&page.concept)
        );
        lines.push(format!("  \"{}\" [label=\"{}\"];", node_id, label));
    }

    for page in &graph.pages {
        let from_id = format!("n_{}", dot_escape(&page.concept.replace(' ', "_")));
        for link in &page.links {
            if concept_to_rel.contains_key(link) {
                let to_id = format!("n_{}", dot_escape(&link.replace(' ', "_")));
                lines.push(format!("  \"{}\" -> \"{}\";", from_id, to_id));
            }
        }
    }

    lines.push("}".to_string());
    safe_write_text_file(out_path, &lines.join("\n"))?;
    Ok(out_path.to_string())
}

pub(crate) fn export_chunk_graph_dot(graph: &Graph, out_path: &str) -> Result<String> {
    let mut lines = Vec::new();
    lines.push("digraph lint_ai_chunk_graph {".to_string());
    lines.push("  rankdir=LR;".to_string());
    lines.push("  node [shape=box, style=rounded];".to_string());
    for chunk in &graph.chunks {
        let node_id = dot_escape(&chunk.chunk_id);
        let label = format!(
            "{}\\n{}:{}-{}",
            dot_escape(&chunk.doc_rel_path),
            dot_escape(&chunk.heading),
            chunk.start_line,
            chunk.end_line
        );
        lines.push(format!("  \"{}\" [label=\"{}\"];", node_id, label));
    }
    for edge in graph.chunk_graph.raw_edges() {
        let from = &graph.chunk_graph[edge.source()];
        let to = &graph.chunk_graph[edge.target()];
        lines.push(format!(
            "  \"{}\" -> \"{}\";",
            dot_escape(from),
            dot_escape(to)
        ));
    }
    lines.push("}".to_string());
    safe_write_text_file(out_path, &lines.join("\n"))?;
    Ok(out_path.to_string())
}

pub(crate) fn export_entity_graph_dot(graph: &Graph, out_path: &str) -> Result<String> {
    let mut lines = Vec::new();
    lines.push("digraph lint_ai_entity_graph {".to_string());
    lines.push("  rankdir=LR;".to_string());
    lines.push("  node [shape=ellipse, style=filled, fillcolor=\"#e8f1ff\"];".to_string());
    for ent in graph.entity_index.keys() {
        let id = dot_escape(ent);
        lines.push(format!("  \"{}\" [label=\"{}\"];", id, id));
    }
    for edge in graph.entity_graph.raw_edges() {
        let from = &graph.entity_graph[edge.source()];
        let to = &graph.entity_graph[edge.target()];
        let label = match edge.weight {
            crate::graph::EntityEdgeKind::CoOccurs => "co_occurs",
            crate::graph::EntityEdgeKind::DocLink => "doc_link",
        };
        lines.push(format!(
            "  \"{}\" -> \"{}\" [label=\"{}\"];",
            dot_escape(from),
            dot_escape(to),
            label
        ));
    }
    lines.push("}".to_string());
    safe_write_text_file(out_path, &lines.join("\n"))?;
    Ok(out_path.to_string())
}

#[derive(Serialize)]
struct GraphJsonNode {
    id: String,
    concept: String,
    source: String,
}

#[derive(Serialize)]
struct GraphJsonEdge {
    source: String,
    target: String,
}

#[derive(Serialize)]
struct GraphJsonExport {
    nodes: Vec<GraphJsonNode>,
    edges: Vec<GraphJsonEdge>,
}

fn build_graph_json_export(graph: &Graph) -> GraphJsonExport {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut concept_exists: HashSet<String> = HashSet::new();

    for page in &graph.pages {
        concept_exists.insert(page.concept.clone());
        nodes.push(GraphJsonNode {
            id: page.concept.clone(),
            concept: page.concept.clone(),
            source: page.rel_path.clone(),
        });
    }

    for page in &graph.pages {
        for link in &page.links {
            if concept_exists.contains(link) {
                edges.push(GraphJsonEdge {
                    source: page.concept.clone(),
                    target: link.clone(),
                });
            }
        }
    }

    GraphJsonExport { nodes, edges }
}

pub(crate) fn export_graph_json(graph: &Graph, out_path: &str) -> Result<String> {
    let payload = build_graph_json_export(graph);
    safe_write_text_file(out_path, &serde_json::to_string_pretty(&payload)?)?;
    Ok(out_path.to_string())
}

#[derive(Serialize)]
struct ChunkGraphJsonNode {
    id: String,
    chunk_id: String,
    doc_id: String,
    heading: String,
    start_line: usize,
    end_line: usize,
}

#[derive(Serialize)]
struct ChunkGraphJsonEdge {
    source: String,
    target: String,
    kind: String,
}

#[derive(Serialize)]
struct ChunkGraphJsonExport {
    nodes: Vec<ChunkGraphJsonNode>,
    edges: Vec<ChunkGraphJsonEdge>,
}

fn build_chunk_graph_json_export(graph: &Graph) -> ChunkGraphJsonExport {
    let nodes = graph
        .chunks
        .iter()
        .map(|c| ChunkGraphJsonNode {
            id: c.chunk_id.clone(),
            chunk_id: c.chunk_id.clone(),
            doc_id: c.doc_rel_path.clone(),
            heading: c.heading.clone(),
            start_line: c.start_line,
            end_line: c.end_line,
        })
        .collect::<Vec<_>>();
    let edges = graph
        .chunk_graph
        .raw_edges()
        .iter()
        .map(|e| {
            let kind = match e.weight {
                crate::graph::ChunkEdgeKind::Next => "next",
                crate::graph::ChunkEdgeKind::DocLink => "doc_link",
            };
            ChunkGraphJsonEdge {
                source: graph.chunk_graph[e.source()].clone(),
                target: graph.chunk_graph[e.target()].clone(),
                kind: kind.to_string(),
            }
        })
        .collect::<Vec<_>>();
    ChunkGraphJsonExport { nodes, edges }
}

pub(crate) fn export_chunk_graph_json(graph: &Graph, out_path: &str) -> Result<String> {
    let payload = build_chunk_graph_json_export(graph);
    safe_write_text_file(out_path, &serde_json::to_string_pretty(&payload)?)?;
    Ok(out_path.to_string())
}

#[derive(Serialize)]
struct EntityGraphJsonNode {
    id: String,
    label: String,
}

#[derive(Serialize)]
struct EntityGraphJsonEdge {
    source: String,
    target: String,
    kind: String,
}

#[derive(Serialize)]
struct EntityGraphJsonExport {
    nodes: Vec<EntityGraphJsonNode>,
    edges: Vec<EntityGraphJsonEdge>,
}

fn build_entity_graph_json_export(graph: &Graph) -> EntityGraphJsonExport {
    let nodes = graph
        .entity_index
        .keys()
        .map(|e| EntityGraphJsonNode {
            id: e.clone(),
            label: e.clone(),
        })
        .collect::<Vec<_>>();
    let edges = graph
        .entity_graph
        .raw_edges()
        .iter()
        .map(|e| {
            let kind = match e.weight {
                crate::graph::EntityEdgeKind::CoOccurs => "co_occurs",
                crate::graph::EntityEdgeKind::DocLink => "doc_link",
            };
            EntityGraphJsonEdge {
                source: graph.entity_graph[e.source()].clone(),
                target: graph.entity_graph[e.target()].clone(),
                kind: kind.to_string(),
            }
        })
        .collect::<Vec<_>>();
    EntityGraphJsonExport { nodes, edges }
}

pub(crate) fn export_entity_graph_json(graph: &Graph, out_path: &str) -> Result<String> {
    let payload = build_entity_graph_json_export(graph);
    safe_write_text_file(out_path, &serde_json::to_string_pretty(&payload)?)?;
    Ok(out_path.to_string())
}

pub(crate) fn export_graph_cytoscape_html(graph: &Graph, out_path: &str) -> Result<String> {
    let payload = build_graph_json_export(graph);
    let nodes_js = payload
        .nodes
        .iter()
        .map(|n| {
            format!(
                "{{ data: {{ id: \"{}\", label: \"{}\", source: \"{}\" }} }}",
                js_string_escape(&n.id),
                js_string_escape(&n.concept),
                js_string_escape(&n.source)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let edges_js = payload
        .edges
        .iter()
        .map(|e| {
            format!(
                "{{ data: {{ source: \"{}\", target: \"{}\" }} }}",
                js_string_escape(&e.source),
                js_string_escape(&e.target)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");

    let html = format!(
        r#"<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Lint-AI Graph</title>
  <script src="./cytoscape.min.js"></script>
  <style>
    html, body {{ width: 100%; height: 100%; margin: 0; }}
    #cy {{ width: 100%; height: 100%; }}
  </style>
</head>
<body>
  <div id="cy"></div>
  <script>
    const elements = {{
      nodes: [{nodes_js}],
      edges: [{edges_js}]
    }};
    const cy = cytoscape({{
      container: document.getElementById('cy'),
      elements,
      layout: {{ name: 'cose', animate: false }},
      style: [
        {{
          selector: 'node',
          style: {{
            'label': 'data(label)',
            'font-size': 10,
            'background-color': '#1f77b4',
            'color': '#111'
          }}
        }},
        {{
          selector: 'edge',
          style: {{
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'line-color': '#888',
            'target-arrow-color': '#888',
            'width': 1
          }}
        }}
      ]
    }});
  </script>
</body>
</html>"#
    );
    safe_write_text_file(out_path, &html)?;
    Ok(out_path.to_string())
}

pub(crate) fn export_chunk_graph_cytoscape_html(graph: &Graph, out_path: &str) -> Result<String> {
    let payload = build_chunk_graph_json_export(graph);
    let nodes_js = payload
        .nodes
        .iter()
        .map(|n| {
            format!(
                "{{ data: {{ id: \"{}\", label: \"{}\", source: \"{}\" }} }}",
                js_string_escape(&n.id),
                js_string_escape(&n.heading),
                js_string_escape(&n.doc_id)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let edges_js = payload
        .edges
        .iter()
        .map(|e| {
            format!(
                "{{ data: {{ source: \"{}\", target: \"{}\", kind: \"{}\" }} }}",
                js_string_escape(&e.source),
                js_string_escape(&e.target),
                js_string_escape(&e.kind)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let html = format!(
        r#"<!doctype html>
<html><head><meta charset="utf-8" /><title>Lint-AI Chunk Graph</title>
<script src="./cytoscape.min.js"></script>
<style>html, body {{ width: 100%; height: 100%; margin: 0; }} #cy {{ width:100%; height:100%; }}</style>
</head><body><div id="cy"></div><script>
const elements = {{ nodes:[{nodes_js}], edges:[{edges_js}] }};
cytoscape({{
  container: document.getElementById('cy'),
  elements,
  layout: {{ name: 'cose', animate: false }},
  style: [
    {{ selector: 'node', style: {{ 'label': 'data(label)', 'font-size': 9, 'background-color': '#2a9d8f' }} }},
    {{ selector: 'edge', style: {{ 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'line-color': '#777', 'target-arrow-color': '#777', 'width': 1 }} }}
  ]
}});
</script></body></html>"#
    );
    safe_write_text_file(out_path, &html)?;
    Ok(out_path.to_string())
}

pub(crate) fn export_entity_graph_cytoscape_html(graph: &Graph, out_path: &str) -> Result<String> {
    let payload = build_entity_graph_json_export(graph);
    let nodes_js = payload
        .nodes
        .iter()
        .map(|n| {
            format!(
                "{{ data: {{ id: \"{}\", label: \"{}\" }} }}",
                js_string_escape(&n.id),
                js_string_escape(&n.label)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let edges_js = payload
        .edges
        .iter()
        .map(|e| {
            format!(
                "{{ data: {{ source: \"{}\", target: \"{}\", kind: \"{}\" }} }}",
                js_string_escape(&e.source),
                js_string_escape(&e.target),
                js_string_escape(&e.kind)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let html = format!(
        r#"<!doctype html>
<html><head><meta charset="utf-8" /><title>Lint-AI Entity Graph</title>
<script src="./cytoscape.min.js"></script>
<style>html, body {{ width:100%; height:100%; margin:0; }} #cy {{ width:100%; height:100%; }}</style>
</head><body><div id="cy"></div><script>
const elements = {{ nodes:[{nodes_js}], edges:[{edges_js}] }};
cytoscape({{
  container: document.getElementById('cy'),
  elements,
  layout: {{ name: 'cose', animate: false }},
  style: [
    {{ selector: 'node', style: {{ 'label': 'data(label)', 'font-size': 10, 'background-color': '#457b9d' }} }},
    {{ selector: 'edge', style: {{ 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'line-color': '#666', 'target-arrow-color': '#666', 'width': 1 }} }}
  ]
}});
</script></body></html>"#
    );
    safe_write_text_file(out_path, &html)?;
    Ok(out_path.to_string())
}

#[derive(Serialize)]
pub(crate) struct ChunkGraphStats {
    chunk_nodes: usize,
    chunk_edges: usize,
    next_edges: usize,
    doc_link_edges: usize,
    docs_with_chunks: usize,
    avg_chunks_per_doc: f32,
}

pub(crate) fn chunk_graph_stats(graph: &Graph) -> ChunkGraphStats {
    let mut next_edges = 0usize;
    let mut doc_link_edges = 0usize;
    for e in graph.chunk_graph.raw_edges() {
        match e.weight {
            crate::graph::ChunkEdgeKind::Next => next_edges += 1,
            crate::graph::ChunkEdgeKind::DocLink => doc_link_edges += 1,
        }
    }
    let docs_with_chunks = graph.doc_to_chunks.len();
    let avg_chunks_per_doc = if docs_with_chunks == 0 {
        0.0
    } else {
        graph.chunks.len() as f32 / docs_with_chunks as f32
    };
    ChunkGraphStats {
        chunk_nodes: graph.chunks.len(),
        chunk_edges: graph.chunk_graph.edge_count(),
        next_edges,
        doc_link_edges,
        docs_with_chunks,
        avg_chunks_per_doc,
    }
}

#[derive(Serialize)]
struct OntologyNode {
    id: String,
    node_type: String,
    label: String,
    metadata: serde_json::Value,
}

#[derive(Serialize)]
struct OntologyEdge {
    source: String,
    target: String,
    edge_type: String,
    weight: f32,
    confidence: f32,
    provenance: serde_json::Value,
}

#[derive(Serialize)]
struct OntologyExport {
    version: String,
    generated_at_unix: u64,
    corpus_path: String,
    node_count: usize,
    edge_count: usize,
    nodes: Vec<OntologyNode>,
    edges: Vec<OntologyEdge>,
}

fn canonical_entity_id(text: &str) -> String {
    let base = normalize_concept(text);
    let key = if base.trim().is_empty() {
        "unknown".to_string()
    } else {
        base.replace(' ', "_")
    };
    format!("entity:{}", key)
}

pub(crate) fn export_ontology_json(
    index: &MemoryIndex,
    out_path: &str,
    corpus_path: &str,
) -> Result<String> {
    let mut nodes: Vec<OntologyNode> = Vec::new();
    let mut edges: Vec<OntologyEdge> = Vec::new();
    let mut node_ids: HashSet<String> = HashSet::new();
    let mut entity_aliases: HashMap<String, HashSet<String>> = HashMap::new();
    let mut cooccur_counts: HashMap<(String, String), usize> = HashMap::new();

    for doc in index.docs.values() {
        let doc_node_id = format!("doc:{}", doc.doc_id);
        if node_ids.insert(doc_node_id.clone()) {
            nodes.push(OntologyNode {
                id: doc_node_id.clone(),
                node_type: "doc".to_string(),
                label: doc.source.clone(),
                metadata: serde_json::json!({
                    "doc_id": doc.doc_id,
                    "source": doc.source,
                    "timestamp": doc.timestamp,
                    "probable_topic": doc.probable_topic,
                    "doc_type_guess": doc.doc_type_guess,
                }),
            });
        }

        for chunk in &doc.section_chunks {
            let chunk_node_id = format!("chunk:{}", chunk.chunk_id);
            if node_ids.insert(chunk_node_id.clone()) {
                nodes.push(OntologyNode {
                    id: chunk_node_id.clone(),
                    node_type: "chunk".to_string(),
                    label: chunk.heading.clone(),
                    metadata: serde_json::json!({
                        "chunk_id": chunk.chunk_id,
                        "doc_id": doc.doc_id,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    }),
                });
            }
            edges.push(OntologyEdge {
                source: chunk_node_id.clone(),
                target: doc_node_id.clone(),
                edge_type: "belongs_to".to_string(),
                weight: 1.0,
                confidence: 1.0,
                provenance: serde_json::json!({"source":"chunk-structure"}),
            });

            let mut canonical_in_chunk: Vec<String> = Vec::new();
            for ent in &chunk.key_entities {
                let eid = canonical_entity_id(ent);
                entity_aliases
                    .entry(eid.clone())
                    .or_default()
                    .insert(ent.clone());
                canonical_in_chunk.push(eid.clone());
                if node_ids.insert(eid.clone()) {
                    nodes.push(OntologyNode {
                        id: eid.clone(),
                        node_type: "entity".to_string(),
                        label: ent.clone(),
                        metadata: serde_json::json!({}),
                    });
                }
                edges.push(OntologyEdge {
                    source: chunk_node_id.clone(),
                    target: eid,
                    edge_type: "mentions".to_string(),
                    weight: 1.0,
                    confidence: 0.8,
                    provenance: serde_json::json!({"source":"tier1_chunk_entities"}),
                });
            }
            canonical_in_chunk.sort();
            canonical_in_chunk.dedup();
            for i in 0..canonical_in_chunk.len() {
                for j in (i + 1)..canonical_in_chunk.len() {
                    let a = canonical_in_chunk[i].clone();
                    let b = canonical_in_chunk[j].clone();
                    let key = if a <= b { (a, b) } else { (b, a) };
                    *cooccur_counts.entry(key).or_insert(0) += 1;
                }
            }
        }
    }

    for (eid, aliases) in entity_aliases {
        if let Some(node) = nodes.iter_mut().find(|n| n.id == eid) {
            let mut alias_list = aliases.into_iter().collect::<Vec<_>>();
            alias_list.sort();
            node.metadata = serde_json::json!({ "aliases": alias_list });
        }
    }

    for ((a, b), count) in cooccur_counts {
        edges.push(OntologyEdge {
            source: a,
            target: b,
            edge_type: "co_occurs".to_string(),
            weight: count as f32,
            confidence: 0.6,
            provenance: serde_json::json!({"source":"chunk_cooccurrence","count":count}),
        });
    }

    let generated_at_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let payload = OntologyExport {
        version: "v0.1-seed".to_string(),
        generated_at_unix,
        corpus_path: corpus_path.to_string(),
        node_count: nodes.len(),
        edge_count: edges.len(),
        nodes,
        edges,
    };
    safe_write_text_file(out_path, &serde_json::to_string_pretty(&payload)?)?;
    Ok(out_path.to_string())
}
