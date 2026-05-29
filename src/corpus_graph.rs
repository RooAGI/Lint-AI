use crate::index::MemoryIndex;
use crate::ownership::OwnershipRecord;
use crate::pipeline::{build_query_snapshot, PipelineOptions};
use crate::symbols::{IngestBundle, SymbolRecord, SymbolStore};
use crate::usage::{UsageGraph, UsageSummary};
use anyhow::Result;

pub struct CorpusGraph {
    pub documents: MemoryIndex,
    pub symbols: SymbolStore,
    pub usage: UsageGraph,
}

impl CorpusGraph {
    pub fn new(documents: MemoryIndex, symbols: SymbolStore, usage: UsageGraph) -> Self {
        Self {
            documents,
            symbols,
            usage,
        }
    }

    pub fn from_bundle(bundle: IngestBundle, options: &PipelineOptions) -> Result<Self> {
        let IngestBundle { documents: source_documents, symbols } = bundle;
        let documents = build_query_snapshot(&source_documents, options)?;
        let usage_bundle = IngestBundle::new(source_documents, symbols.clone());
        let symbols = SymbolStore::from_records(symbols);
        let usage = UsageGraph::from_bundle(&usage_bundle);
        Ok(Self {
            documents,
            symbols,
            usage,
        })
    }

    pub fn with_ownership(mut self, records: impl IntoIterator<Item = OwnershipRecord>) -> Self {
        self.usage.ingest_ownership(records);
        self
    }

    pub fn symbol_usage_summary(&self, symbol_id: &str) -> UsageSummary {
        self.usage.summary_for_symbol(symbol_id)
    }

    pub fn symbols_for_doc(&self, doc_id: &str) -> Vec<&SymbolRecord> {
        self.symbols.lookup_by_doc(doc_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::SourceDocument;

    #[test]
    fn from_bundle_builds_documents_symbols_and_usage() {
        let document = SourceDocument {
            doc_id: "doc-1".to_string(),
            source: "docs/install.md".to_string(),
            content: "install guide for linux hosts".to_string(),
            concept: "install guide".to_string(),
            group_id: None,
            headings: vec!["Overview".to_string()],
            links: vec![],
            timestamp: None,
            doc_length: "install guide for linux hosts".len(),
            author_agent: None,
        };
        let symbol = SymbolRecord::declared(
            "doc-1",
            "install",
            crate::symbols::SymbolKind::Function,
            Some("cpp".to_string()),
        );
        let bundle = IngestBundle::new(vec![document], vec![symbol.clone()]);

        let graph = CorpusGraph::from_bundle(bundle, &PipelineOptions::default()).unwrap();

        assert_eq!(graph.symbols_for_doc("doc-1").len(), 1);
        let summary = graph.symbol_usage_summary(&symbol.symbol_id);
        assert_eq!(summary.edge_count, 1);
        assert_eq!(
            summary.edge_counts.get(&crate::usage::UsageEdgeKind::Declares).copied(),
            Some(1)
        );
    }
}
