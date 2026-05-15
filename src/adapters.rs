use anyhow::Result;
use std::path::Path;

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
