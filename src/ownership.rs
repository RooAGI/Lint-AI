use crate::symbols::SymbolLocation;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OwnershipKind {
    Allocates,
    Releases,
    Transfers,
    Borrows,
    Aliases,
    Managed,
    Escapes,
    Invalidates,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FlowState {
    Unknown,
    Owned,
    Borrowed,
    Shared,
    Escaped,
    Released,
    Dangling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FlowEdgeKind {
    Allocates,
    Transfers,
    Borrows,
    Aliases,
    Releases,
    Escapes,
    Invalidates,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FlowEdge {
    pub from: String,
    pub to: String,
    pub kind: FlowEdgeKind,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OwnershipRecord {
    pub fact_id: String,
    pub doc_id: String,
    #[serde(default)]
    pub symbol_id: Option<String>,
    pub kind: OwnershipKind,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub location: Option<SymbolLocation>,
    #[serde(default)]
    pub confidence: f32,
    #[serde(default)]
    pub state: Option<FlowState>,
    #[serde(default)]
    pub attributes: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LeakFinding {
    pub doc_id: String,
    #[serde(default)]
    pub target: Option<String>,
    pub allocation_fact_id: String,
    #[serde(default)]
    pub release_fact_ids: Vec<String>,
    #[serde(default)]
    pub location: Option<SymbolLocation>,
    #[serde(default)]
    pub confidence: f32,
    pub summary: String,
}
