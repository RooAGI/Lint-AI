use crate::ownership::{OwnershipKind, OwnershipRecord};
use crate::symbols::{SymbolLocation, SymbolRecord};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UsageNodeKind {
    Symbol,
    Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UsageEdgeKind {
    Declares,
    References,
    Uses,
    Calls,
    Allocates,
    Transfers,
    Releases,
    Imports,
    Inherits,
    Contains,
    Anchors,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UsageNode {
    pub node_id: String,
    pub kind: UsageNodeKind,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub doc_id: Option<String>,
    #[serde(default)]
    pub symbol_id: Option<String>,
    #[serde(default)]
    pub location: Option<SymbolLocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UsageEdge {
    pub from: String,
    pub to: String,
    pub kind: UsageEdgeKind,
    #[serde(default)]
    pub doc_id: Option<String>,
    #[serde(default)]
    pub symbol_id: Option<String>,
    #[serde(default)]
    pub location: Option<SymbolLocation>,
    #[serde(default)]
    pub confidence: f32,
    #[serde(default)]
    pub attributes: HashMap<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageGraph {
    pub nodes: Vec<UsageNode>,
    pub edges: Vec<UsageEdge>,
    #[serde(default)]
    pub by_node: HashMap<String, Vec<usize>>,
    #[serde(default)]
    pub by_doc: HashMap<String, Vec<usize>>,
    #[serde(default)]
    pub by_symbol: HashMap<String, Vec<usize>>,
    #[serde(default)]
    pub by_kind: HashMap<UsageEdgeKind, Vec<usize>>,
}

impl UsageGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a usage graph from symbol records.
    ///
    /// Documents are not represented as graph nodes; document provenance stays
    /// on edges through `doc_id`.
    pub fn from_bundle(bundle: &crate::symbols::IngestBundle) -> Self {
        let mut graph = Self::new();
        graph.ingest_symbols(bundle.symbols.iter().cloned());
        graph
    }

    pub fn insert_node(&mut self, node: UsageNode) -> usize {
        let index = self.nodes.len();
        self.nodes.push(node);
        index
    }

    pub fn insert_edge(&mut self, edge: UsageEdge) {
        let index = self.edges.len();
        let from = edge.from.clone();
        let to = edge.to.clone();
        let doc_id = edge.doc_id.clone();
        let symbol_id = edge.symbol_id.clone();
        let kind = edge.kind;
        self.edges.push(edge);
        self.by_node.entry(from).or_default().push(index);
        self.by_node.entry(to).or_default().push(index);
        if let Some(doc_id) = doc_id {
            self.by_doc.entry(doc_id).or_default().push(index);
        }
        if let Some(symbol_id) = symbol_id {
            self.by_symbol.entry(symbol_id).or_default().push(index);
        }
        self.by_kind.entry(kind).or_default().push(index);
    }

    pub fn lookup_by_node(&self, node_id: &str) -> Vec<&UsageEdge> {
        self.lookup_indices(self.by_node.get(node_id))
    }

    pub fn lookup_by_doc(&self, doc_id: &str) -> Vec<&UsageEdge> {
        self.lookup_indices(self.by_doc.get(doc_id))
    }

    pub fn lookup_by_symbol(&self, symbol_id: &str) -> Vec<&UsageEdge> {
        self.lookup_indices(self.by_symbol.get(symbol_id))
    }

    pub fn lookup_by_kind(&self, kind: UsageEdgeKind) -> Vec<&UsageEdge> {
        self.lookup_indices(self.by_kind.get(&kind))
    }

    fn lookup_indices(&self, indices: Option<&Vec<usize>>) -> Vec<&UsageEdge> {
        indices
            .into_iter()
            .flat_map(|indices| indices.iter())
            .filter_map(|index| self.edges.get(*index))
            .collect()
    }

    pub fn ingest_symbols(&mut self, symbols: impl IntoIterator<Item = SymbolRecord>) {
        for symbol in symbols {
            let node_id = format!("symbol:{}", symbol.symbol_id);
            self.insert_node(UsageNode {
                node_id: node_id.clone(),
                kind: UsageNodeKind::Symbol,
                label: Some(symbol.name.clone()),
                doc_id: Some(symbol.doc_id.clone()),
                symbol_id: Some(symbol.symbol_id.clone()),
                location: symbol.location.clone(),
            });

            let doc_node_id = format!("doc:{}", symbol.doc_id);
            self.insert_edge(UsageEdge {
                from: doc_node_id,
                to: node_id,
                kind: match symbol.relation {
                    crate::symbols::SymbolRelationKind::Declares
                    | crate::symbols::SymbolRelationKind::Defines => UsageEdgeKind::Declares,
                    crate::symbols::SymbolRelationKind::References => UsageEdgeKind::References,
                    crate::symbols::SymbolRelationKind::Uses => UsageEdgeKind::Uses,
                    crate::symbols::SymbolRelationKind::Calls => UsageEdgeKind::Calls,
                    crate::symbols::SymbolRelationKind::Imports => UsageEdgeKind::Imports,
                    crate::symbols::SymbolRelationKind::Inherits => UsageEdgeKind::Inherits,
                    crate::symbols::SymbolRelationKind::Contains => UsageEdgeKind::Contains,
                },
                doc_id: Some(symbol.doc_id),
                symbol_id: Some(symbol.symbol_id),
                location: symbol.location,
                confidence: 1.0,
                attributes: HashMap::new(),
            });
        }
    }

    pub fn ingest_ownership(&mut self, records: impl IntoIterator<Item = OwnershipRecord>) {
        for record in records {
            let kind = match record.kind {
                OwnershipKind::Allocates | OwnershipKind::Managed => UsageEdgeKind::Allocates,
                OwnershipKind::Transfers => UsageEdgeKind::Transfers,
                OwnershipKind::Releases => UsageEdgeKind::Releases,
                OwnershipKind::Borrows => UsageEdgeKind::Uses,
                OwnershipKind::Aliases => UsageEdgeKind::Uses,
                OwnershipKind::Escapes => UsageEdgeKind::Uses,
                OwnershipKind::Invalidates => UsageEdgeKind::Anchors,
            };
            let from = record
                .symbol_id
                .clone()
                .unwrap_or_else(|| format!("doc:{}", record.doc_id));
            let to = record
                .target
                .clone()
                .unwrap_or_else(|| format!("doc:{}", record.doc_id));
            self.insert_edge(UsageEdge {
                from,
                to,
                kind,
                doc_id: Some(record.doc_id),
                symbol_id: record.symbol_id,
                location: record.location,
                confidence: record.confidence,
                attributes: record.attributes,
            });
        }
    }

    /// Return node ids that are connected to the given symbol through usage or
    /// ownership edges.
    pub fn connected_symbols(&self, symbol_id: &str) -> Vec<String> {
        let mut out = HashSet::new();
        for edge in self.lookup_by_symbol(symbol_id) {
            if self.nodes.iter().any(|node| node.node_id == edge.from) {
                out.insert(edge.from.clone());
            }
            if self.nodes.iter().any(|node| node.node_id == edge.to) {
                out.insert(edge.to.clone());
            }
        }
        out.into_iter().collect()
    }

    /// Summarize the edges and adjacent nodes associated with a symbol.
    pub fn summary_for_symbol(&self, symbol_id: &str) -> UsageSummary {
        let edges = self.lookup_by_symbol(symbol_id);
        let mut counts = HashMap::new();
        for kind in [
            UsageEdgeKind::Declares,
            UsageEdgeKind::References,
            UsageEdgeKind::Uses,
            UsageEdgeKind::Calls,
            UsageEdgeKind::Allocates,
            UsageEdgeKind::Transfers,
            UsageEdgeKind::Releases,
            UsageEdgeKind::Imports,
            UsageEdgeKind::Inherits,
            UsageEdgeKind::Contains,
            UsageEdgeKind::Anchors,
        ] {
            counts.insert(kind, 0usize);
        }
        for edge in edges {
            *counts.entry(edge.kind).or_insert(0) += 1;
        }
        UsageSummary {
            symbol_id: symbol_id.to_string(),
            edge_count: self.lookup_by_symbol(symbol_id).len(),
            related_nodes: self.connected_symbols(symbol_id),
            edge_counts: counts,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UsageSummary {
    pub symbol_id: String,
    pub edge_count: usize,
    pub related_nodes: Vec<String>,
    pub edge_counts: HashMap<UsageEdgeKind, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_by_node_is_bidirectional() {
        let mut graph = UsageGraph::new();
        graph.insert_edge(UsageEdge {
            from: "node:a".to_string(),
            to: "node:b".to_string(),
            kind: UsageEdgeKind::Calls,
            doc_id: Some("doc-1".to_string()),
            symbol_id: Some("sym-1".to_string()),
            location: None,
            confidence: 1.0,
            attributes: HashMap::new(),
        });

        assert_eq!(graph.lookup_by_node("node:a").len(), 1);
        assert_eq!(graph.lookup_by_node("node:b").len(), 1);
    }

    #[test]
    fn summary_for_symbol_counts_edges_and_connected_nodes() {
        let mut graph = UsageGraph::new();
        graph.insert_node(UsageNode {
            node_id: "symbol:a".to_string(),
            kind: UsageNodeKind::Symbol,
            label: Some("Alpha".to_string()),
            doc_id: Some("doc-1".to_string()),
            symbol_id: Some("sym-a".to_string()),
            location: None,
        });
        graph.insert_node(UsageNode {
            node_id: "symbol:b".to_string(),
            kind: UsageNodeKind::Symbol,
            label: Some("Beta".to_string()),
            doc_id: Some("doc-2".to_string()),
            symbol_id: Some("sym-b".to_string()),
            location: None,
        });
        graph.insert_edge(UsageEdge {
            from: "symbol:a".to_string(),
            to: "symbol:b".to_string(),
            kind: UsageEdgeKind::Calls,
            doc_id: Some("doc-1".to_string()),
            symbol_id: Some("sym-a".to_string()),
            location: None,
            confidence: 1.0,
            attributes: HashMap::new(),
        });

        let summary = graph.summary_for_symbol("sym-a");
        assert_eq!(summary.symbol_id, "sym-a");
        assert_eq!(summary.edge_count, 1);
        assert_eq!(
            summary.edge_counts.get(&UsageEdgeKind::Calls).copied(),
            Some(1)
        );
        assert!(summary.related_nodes.contains(&"symbol:a".to_string()));
        assert!(summary.related_nodes.contains(&"symbol:b".to_string()));
    }
}
