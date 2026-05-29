use crate::index::MemoryIndex;
use crate::pipeline::{build_query_snapshot, PipelineOptions};
use crate::query_expansion::normalize_for_index;
use crate::source::SourceDocument;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    Namespace,
    Type,
    Class,
    Struct,
    Enum,
    Function,
    Method,
    Field,
    Variable,
    Module,
    File,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SymbolRelationKind {
    Declares,
    References,
    Defines,
    Uses,
    Imports,
    Inherits,
    Calls,
    Contains,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SymbolLocation {
    pub source: String,
    #[serde(default)]
    pub start_line: Option<usize>,
    #[serde(default)]
    pub end_line: Option<usize>,
    #[serde(default)]
    pub start_byte: Option<usize>,
    #[serde(default)]
    pub end_byte: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SymbolRecord {
    pub symbol_id: String,
    pub doc_id: String,
    pub name: String,
    pub kind: SymbolKind,
    pub relation: SymbolRelationKind,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub container: Option<String>,
    #[serde(default)]
    pub location: Option<SymbolLocation>,
    #[serde(default)]
    pub signature: Option<String>,
    #[serde(default)]
    pub modifiers: Vec<String>,
    #[serde(default)]
    pub scope_path: Vec<String>,
    #[serde(default)]
    pub targets: Vec<String>,
    #[serde(default)]
    pub attributes: HashMap<String, Value>,
}

impl SymbolRecord {
    pub fn new(
        symbol_id: impl Into<String>,
        doc_id: impl Into<String>,
        name: impl Into<String>,
        kind: SymbolKind,
        relation: SymbolRelationKind,
    ) -> Self {
        Self {
            symbol_id: symbol_id.into(),
            doc_id: doc_id.into(),
            name: name.into(),
            kind,
            relation,
            language: None,
            container: None,
            location: None,
            signature: None,
            modifiers: Vec::new(),
            scope_path: Vec::new(),
            targets: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    pub fn declared(
        doc_id: impl Into<String>,
        name: impl Into<String>,
        kind: SymbolKind,
        language: Option<String>,
    ) -> Self {
        let doc_id = doc_id.into();
        let name = name.into();
        let symbol_id = symbol_id_for(&doc_id, &name, &kind, &SymbolRelationKind::Declares);
        Self {
            language,
            ..Self::new(symbol_id, doc_id, name, kind, SymbolRelationKind::Declares)
        }
    }

    pub fn referenced(
        doc_id: impl Into<String>,
        name: impl Into<String>,
        kind: SymbolKind,
        language: Option<String>,
    ) -> Self {
        let doc_id = doc_id.into();
        let name = name.into();
        let symbol_id = symbol_id_for(&doc_id, &name, &kind, &SymbolRelationKind::References);
        Self {
            language,
            ..Self::new(
                symbol_id,
                doc_id,
                name,
                kind,
                SymbolRelationKind::References,
            )
        }
    }

    pub fn with_attribute(mut self, key: impl Into<String>, value: Value) -> Self {
        self.attributes.insert(key.into(), value);
        self
    }
}

fn symbol_id_for(
    doc_id: &str,
    name: &str,
    kind: &SymbolKind,
    relation: &SymbolRelationKind,
) -> String {
    format!("{doc_id}::{name}::{kind:?}::{relation:?}")
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolStore {
    pub records: Vec<SymbolRecord>,
    #[serde(default)]
    pub by_name: HashMap<String, Vec<usize>>,
    #[serde(default)]
    pub by_normalized_name: HashMap<String, Vec<usize>>,
    #[serde(default)]
    pub by_doc: HashMap<String, Vec<usize>>,
    #[serde(default)]
    pub by_kind: HashMap<SymbolKind, Vec<usize>>,
    #[serde(default)]
    pub by_relation: HashMap<SymbolRelationKind, Vec<usize>>,
}

impl SymbolStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_records(records: impl IntoIterator<Item = SymbolRecord>) -> Self {
        let mut store = Self::new();
        store.extend(records);
        store
    }

    pub fn extend(&mut self, records: impl IntoIterator<Item = SymbolRecord>) {
        for record in records {
            self.insert(record);
        }
    }

    pub fn insert(&mut self, record: SymbolRecord) {
        let index = self.records.len();
        let exact_name = record.name.clone();
        let normalized_name = normalize_for_index(&exact_name);
        let doc_id = record.doc_id.clone();
        let kind = record.kind.clone();
        let relation = record.relation.clone();

        self.records.push(record);
        self.by_name.entry(exact_name).or_default().push(index);
        if !normalized_name.is_empty() {
            self.by_normalized_name
                .entry(normalized_name)
                .or_default()
                .push(index);
        }
        self.by_doc.entry(doc_id).or_default().push(index);
        self.by_kind.entry(kind).or_default().push(index);
        self.by_relation.entry(relation).or_default().push(index);
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn lookup_by_name<'a>(&'a self, name: &str) -> Vec<&'a SymbolRecord> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        self.append_matches(name, &mut seen, &mut out);
        let normalized = normalize_for_index(name);
        if normalized != name {
            self.append_normalized_matches(&normalized, &mut seen, &mut out);
        }
        out
    }

    pub fn lookup_by_doc<'a>(&'a self, doc_id: &str) -> Vec<&'a SymbolRecord> {
        self.lookup_indices(self.by_doc.get(doc_id))
    }

    pub fn lookup_by_kind<'a>(&'a self, kind: &SymbolKind) -> Vec<&'a SymbolRecord> {
        self.lookup_indices(self.by_kind.get(kind))
    }

    pub fn lookup_by_relation<'a>(
        &'a self,
        relation: &SymbolRelationKind,
    ) -> Vec<&'a SymbolRecord> {
        self.lookup_indices(self.by_relation.get(relation))
    }

    pub fn lookup_by_attribute<'a>(&'a self, key: &str, value: &Value) -> Vec<&'a SymbolRecord> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for (idx, record) in self.records.iter().enumerate() {
            if record.attributes.get(key) == Some(value) && seen.insert(idx) {
                out.push(record);
            }
        }
        out
    }

    fn append_matches<'a>(
        &'a self,
        name: &str,
        seen: &mut HashSet<usize>,
        out: &mut Vec<&'a SymbolRecord>,
    ) {
        if let Some(indices) = self.by_name.get(name) {
            self.append_indices(indices, seen, out);
        }
    }

    fn append_normalized_matches<'a>(
        &'a self,
        name: &str,
        seen: &mut HashSet<usize>,
        out: &mut Vec<&'a SymbolRecord>,
    ) {
        if let Some(indices) = self.by_normalized_name.get(name) {
            self.append_indices(indices, seen, out);
        }
    }

    fn lookup_indices<'a>(&'a self, indices: Option<&Vec<usize>>) -> Vec<&'a SymbolRecord> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        if let Some(indices) = indices {
            self.append_indices(indices, &mut seen, &mut out);
        }
        out
    }

    fn append_indices<'a>(
        &'a self,
        indices: &[usize],
        seen: &mut HashSet<usize>,
        out: &mut Vec<&'a SymbolRecord>,
    ) {
        for &index in indices {
            if seen.insert(index) {
                if let Some(record) = self.records.get(index) {
                    out.push(record);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct IngestBundle {
    pub documents: Vec<SourceDocument>,
    pub symbols: Vec<SymbolRecord>,
}

impl IngestBundle {
    pub fn new(documents: Vec<SourceDocument>, symbols: Vec<SymbolRecord>) -> Self {
        Self { documents, symbols }
    }
}

pub struct CorpusIndex {
    pub documents: MemoryIndex,
    pub symbols: SymbolStore,
}

impl CorpusIndex {
    pub fn new(documents: MemoryIndex, symbols: SymbolStore) -> Self {
        Self { documents, symbols }
    }

    pub fn from_bundle(bundle: IngestBundle, options: &PipelineOptions) -> Result<Self> {
        let documents = build_query_snapshot(&bundle.documents, options)?;
        let symbols = SymbolStore::from_records(bundle.symbols);
        Ok(Self { documents, symbols })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_source_document() -> SourceDocument {
        SourceDocument {
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
        }
    }

    #[test]
    fn lookup_by_name_matches_exact_and_normalized_forms() {
        let exact = SymbolRecord::declared(
            "doc-1",
            "virtual machine",
            SymbolKind::Class,
            Some("cpp".to_string()),
        );
        let hyphenated = SymbolRecord::declared(
            "doc-2",
            "virtual-machine",
            SymbolKind::Class,
            Some("cpp".to_string()),
        );
        let store = SymbolStore::from_records(vec![exact, hyphenated]);

        let matches = store.lookup_by_name("virtual machine");
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().all(|record| record.kind == SymbolKind::Class));
    }

    #[test]
    fn lookup_by_doc_and_attribute_return_matching_records() {
        let record = SymbolRecord::referenced(
            "doc-1",
            "install",
            SymbolKind::Function,
            Some("cpp".to_string()),
        )
        .with_attribute("visibility", json!("public"));
        let store = SymbolStore::from_records(vec![record.clone()]);

        let by_doc = store.lookup_by_doc("doc-1");
        assert_eq!(by_doc.len(), 1);
        assert_eq!(by_doc[0].symbol_id, record.symbol_id);

        let by_attr = store.lookup_by_attribute("visibility", &json!("public"));
        assert_eq!(by_attr.len(), 1);
        assert_eq!(by_attr[0].symbol_id, record.symbol_id);
    }

    #[test]
    fn corpus_index_from_bundle_preserves_documents_and_symbols() {
        let doc = sample_source_document();
        let symbol = SymbolRecord::declared(
            "doc-1",
            "install",
            SymbolKind::Function,
            Some("cpp".to_string()),
        );
        let bundle = IngestBundle::new(vec![doc], vec![symbol.clone()]);

        let index = CorpusIndex::from_bundle(bundle, &PipelineOptions::default()).unwrap();

        assert_eq!(index.symbols.lookup_by_doc("doc-1").len(), 1);
        assert!(!index.documents.query("install", 5).is_empty());
    }
}
