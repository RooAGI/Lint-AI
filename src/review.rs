use crate::ownership::OwnershipRecord;
use crate::source::SourceDocument;
use crate::symbols::{SymbolLocation, SymbolRecord};
use crate::usage::{UsageEdge, UsageGraph};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReviewSeverity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReviewCategory {
    Ownership,
    ApiMisuse,
    UsageRegression,
    DeadCode,
    MissingReference,
    RippleEffect,
    BehaviorChange,
    StyleOrNaming,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReviewRepoRef {
    pub name: String,
    #[serde(default)]
    pub base_ref: Option<String>,
    #[serde(default)]
    pub head_ref: Option<String>,
    #[serde(default)]
    pub commit_sha: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReviewHunk {
    pub hunk_id: String,
    pub path: String,
    #[serde(default)]
    pub old_start: Option<usize>,
    #[serde(default)]
    pub old_end: Option<usize>,
    #[serde(default)]
    pub new_start: Option<usize>,
    #[serde(default)]
    pub new_end: Option<usize>,
    #[serde(default)]
    pub added_lines: Vec<String>,
    #[serde(default)]
    pub removed_lines: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReviewFileChange {
    pub path: String,
    pub status: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub hunks: Vec<ReviewHunk>,
    #[serde(default)]
    pub changed_symbols: Vec<String>,
    #[serde(default)]
    pub symbol_ids: Vec<String>,
    #[serde(default)]
    pub ownership_fact_ids: Vec<String>,
    #[serde(default)]
    pub usage_edge_kinds: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReviewFileSummary {
    pub path: String,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReviewSummary {
    pub files_changed: usize,
    pub lines_added: usize,
    pub lines_removed: usize,
    pub symbols_touched: usize,
    #[serde(default)]
    pub files: Vec<ReviewFileSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReviewDiffSummary {
    pub files_changed: usize,
    pub lines_added: usize,
    pub lines_removed: usize,
    pub symbols_touched: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentSummary {
    pub doc_id: String,
    pub source: String,
    pub concept: String,
    #[serde(default)]
    pub headings: Vec<String>,
    #[serde(default)]
    pub links: Vec<String>,
    pub doc_length: usize,
}

impl From<&SourceDocument> for DocumentSummary {
    fn from(value: &SourceDocument) -> Self {
        Self {
            doc_id: value.doc_id.clone(),
            source: value.source.clone(),
            concept: value.concept.clone(),
            headings: value.headings.clone(),
            links: value.links.clone(),
            doc_length: value.doc_length,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SymbolSummary {
    pub symbol_id: String,
    pub doc_id: String,
    pub name: String,
    pub kind: String,
    pub relation: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub container: Option<String>,
    #[serde(default)]
    pub location: Option<SymbolLocation>,
    #[serde(default)]
    pub signature: Option<String>,
}

impl From<&SymbolRecord> for SymbolSummary {
    fn from(value: &SymbolRecord) -> Self {
        Self {
            symbol_id: value.symbol_id.clone(),
            doc_id: value.doc_id.clone(),
            name: value.name.clone(),
            kind: format!("{:?}", value.kind),
            relation: format!("{:?}", value.relation),
            language: value.language.clone(),
            container: value.container.clone(),
            location: value.location.clone(),
            signature: value.signature.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OwnershipSummary {
    pub fact_id: String,
    pub doc_id: String,
    #[serde(default)]
    pub symbol_id: Option<String>,
    pub kind: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub location: Option<SymbolLocation>,
    #[serde(default)]
    pub confidence: f32,
}

impl From<&OwnershipRecord> for OwnershipSummary {
    fn from(value: &OwnershipRecord) -> Self {
        Self {
            fact_id: value.fact_id.clone(),
            doc_id: value.doc_id.clone(),
            symbol_id: value.symbol_id.clone(),
            kind: format!("{:?}", value.kind),
            source: value.source.clone(),
            target: value.target.clone(),
            location: value.location.clone(),
            confidence: value.confidence,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReviewUsageNodeSummary {
    #[serde(default)]
    pub node_id: Option<String>,
    pub kind: crate::usage::UsageNodeKind,
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
pub struct ReviewUsageSummary {
    #[serde(default)]
    pub nodes: Vec<ReviewUsageNodeSummary>,
    #[serde(default)]
    pub edges: Vec<UsageEdge>,
}

impl From<&UsageGraph> for ReviewUsageSummary {
    fn from(value: &UsageGraph) -> Self {
        Self {
            nodes: value
                .nodes
                .iter()
                .map(|node| ReviewUsageNodeSummary {
                    node_id: Some(node.node_id.clone()),
                    kind: node.kind,
                    label: node.label.clone(),
                    doc_id: node.doc_id.clone(),
                    symbol_id: node.symbol_id.clone(),
                    location: node.location.clone(),
                })
                .collect(),
            edges: value.edges.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReviewEvidence {
    #[serde(default)]
    pub doc_id: Option<String>,
    #[serde(default)]
    pub symbol_id: Option<String>,
    #[serde(default)]
    pub location: Option<SymbolLocation>,
    #[serde(default)]
    pub snippet: Option<String>,
    pub kind: String,
    #[serde(default)]
    pub attributes: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReviewFinding {
    pub finding_id: String,
    pub severity: ReviewSeverity,
    pub category: ReviewCategory,
    pub summary: String,
    pub rationale: String,
    pub confidence: f32,
    #[serde(default)]
    pub evidence: Vec<ReviewEvidence>,
    #[serde(default)]
    pub recommendation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReviewContext {
    #[serde(default)]
    pub documents: Vec<DocumentSummary>,
    #[serde(default)]
    pub symbols: Vec<SymbolSummary>,
    #[serde(default)]
    pub ownership: Vec<OwnershipSummary>,
    #[serde(default)]
    pub usage: Option<ReviewUsageSummary>,
    #[serde(default)]
    pub source_spans: Vec<ReviewEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReviewPacket {
    pub review_id: String,
    pub repo: ReviewRepoRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReviewSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff: Option<ReviewDiff>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<ReviewContext>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub findings: Vec<ReviewFinding>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub instructions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReviewDiff {
    pub summary: ReviewDiffSummary,
    #[serde(default)]
    pub files: Vec<ReviewFileChange>,
}

impl ReviewPacket {
    pub fn new(
        review_id: impl Into<String>,
        repo: ReviewRepoRef,
        summary: Option<ReviewSummary>,
        diff: Option<ReviewDiff>,
        context: Option<ReviewContext>,
    ) -> Self {
        Self {
            review_id: review_id.into(),
            repo,
            summary,
            diff,
            context,
            findings: Vec::new(),
            instructions: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::usage::{UsageEdgeKind, UsageNode, UsageNodeKind};

    #[test]
    fn summary_conversions_preserve_source_fields() {
        let doc = SourceDocument {
            doc_id: "doc-1".to_string(),
            source: "docs/install.md".to_string(),
            content: "install guide".to_string(),
            concept: "install guide".to_string(),
            group_id: None,
            headings: vec!["Overview".to_string()],
            links: vec!["docs/setup.md".to_string()],
            timestamp: None,
            doc_length: 13,
            author_agent: None,
        };
        let symbol = SymbolRecord::declared(
            "doc-1",
            "install",
            crate::symbols::SymbolKind::Function,
            Some("cpp".to_string()),
        );
        let ownership = OwnershipRecord {
            fact_id: "fact-1".to_string(),
            doc_id: "doc-1".to_string(),
            symbol_id: Some(symbol.symbol_id.clone()),
            kind: crate::ownership::OwnershipKind::Allocates,
            source: "allocate()".to_string(),
            target: Some("buffer".to_string()),
            location: None,
            confidence: 0.95,
            state: None,
            attributes: HashMap::new(),
        };

        let doc_summary = DocumentSummary::from(&doc);
        assert_eq!(doc_summary.doc_id, doc.doc_id);
        assert_eq!(doc_summary.source, doc.source);
        assert_eq!(doc_summary.links, doc.links);

        let symbol_summary = SymbolSummary::from(&symbol);
        assert_eq!(symbol_summary.symbol_id, symbol.symbol_id);
        assert_eq!(symbol_summary.kind, "Function");
        assert_eq!(symbol_summary.relation, "Declares");

        let ownership_summary = OwnershipSummary::from(&ownership);
        assert_eq!(ownership_summary.fact_id, ownership.fact_id);
        assert_eq!(ownership_summary.symbol_id, ownership.symbol_id);
        assert_eq!(ownership_summary.kind, "Allocates");
    }

    #[test]
    fn review_usage_summary_copies_nodes_and_edges() {
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

        let summary = ReviewUsageSummary::from(&graph);
        assert_eq!(summary.nodes.len(), 2);
        assert_eq!(summary.edges.len(), 1);
        assert_eq!(summary.nodes[0].node_id.as_deref(), Some("symbol:a"));
        assert_eq!(summary.edges[0].kind, UsageEdgeKind::Calls);
    }

    #[test]
    fn review_packet_new_initializes_empty_lists() {
        let packet = ReviewPacket::new(
            "review-1",
            ReviewRepoRef {
                name: "owner/repo".to_string(),
                base_ref: Some("main".to_string()),
                head_ref: Some("feature".to_string()),
                commit_sha: Some("abc123".to_string()),
            },
            None,
            None,
            None,
        );

        assert_eq!(packet.review_id, "review-1");
        assert!(packet.summary.is_none());
        assert!(packet.diff.is_none());
        assert!(packet.context.is_none());
        assert!(packet.findings.is_empty());
        assert!(packet.instructions.is_empty());
    }
}
