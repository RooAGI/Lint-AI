use anyhow::Result;
use comrak::{nodes::NodeValue, parse_document, Arena, ComrakOptions};
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;
use walkdir::WalkDir;

use crate::adapters::{AdapterInput, SourceAdapter};
use crate::graph::{concept_from_link_target, normalize_concept, rel_path, Tier0Record};
use crate::source::SourceDocument;

#[derive(Debug, Clone)]
pub struct SourcePage {
    pub path: String,
    pub rel_path: String,
    pub concept: String,
    pub raw_concept: String,
    pub content: String,
    pub links: std::collections::HashSet<String>,
    pub headings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MarkdownCorpus {
    pub pages: Vec<SourcePage>,
    pub tier0_records: Vec<Tier0Record>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MarkdownAdapter;

impl SourceAdapter for MarkdownAdapter {
    fn name(&self) -> &'static str {
        "markdown"
    }

    fn supports(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|s| s.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("md"))
            .unwrap_or(false)
    }

    fn ingest(&self, input: &AdapterInput<'_>) -> Result<Vec<SourceDocument>> {
        let corpus = build_markdown_corpus(
            input.root,
            input.max_bytes,
            input.max_files,
            input.max_depth,
            input.max_total_bytes,
        )?;
        Ok(corpus
            .pages
            .into_iter()
            .map(|page| {
                let doc_length = page.content.len();
                SourceDocument {
                    doc_id: page.rel_path.clone(),
                    source: page.rel_path,
                    content: page.content,
                    concept: page.raw_concept,
                    group_id: None,
                    filters: std::collections::BTreeMap::new(),
                    headings: page.headings,
                    links: page.links.into_iter().collect(),
                    timestamp: None,
                    doc_length,
                    author_agent: None,
                }
            })
            .collect())
    }
}

pub fn build_markdown_corpus(
    root: &Path,
    max_bytes: usize,
    max_files: usize,
    max_depth: usize,
    max_total_bytes: usize,
) -> Result<MarkdownCorpus> {
    let base = docs_dir(root);
    let base_walk = base.clone();
    let single_file = if root.is_file() {
        Some(root.to_path_buf())
    } else {
        None
    };
    let rel_root = if root.is_file() { base.as_path() } else { root };

    let wiki_link_re = Regex::new(r"\[\[(.*?)\]\]")?;
    let md_link_re = Regex::new(r"\[([^\]]+)\]\(([^)]+)\)")?;
    let md_heading_re = Regex::new(r"(?m)^#{1,6}\s+(.*)$")?;

    let mut pages = Vec::new();
    let mut tier0_records = Vec::new();
    let mut files_seen = 0usize;
    let mut total_bytes = 0usize;

    for entry in WalkDir::new(base_walk).max_depth(max_depth) {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        if let Some(ref only) = single_file {
            if entry.path() != only {
                continue;
            }
        }

        let ext = entry
            .path()
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if ext != "md" {
            continue;
        }

        files_seen += 1;
        if files_seen > max_files {
            break;
        }

        let metadata = entry.metadata()?;
        if metadata.len() as usize > max_bytes {
            continue;
        }
        total_bytes = total_bytes.saturating_add(metadata.len() as usize);
        if total_bytes > max_total_bytes {
            break;
        }

        let content = fs::read_to_string(entry.path())?;
        let raw_concept = entry
            .path()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let concept = normalize_concept(&raw_concept);

        let mut links = std::collections::HashSet::new();
        let mut headings = Vec::new();

        for cap in wiki_link_re.captures_iter(&content) {
            if let Some(concept) = concept_from_link_target(cap[1].trim()) {
                links.insert(concept);
            }
        }

        for cap in md_link_re.captures_iter(&content) {
            let target = cap.get(2).map(|m| m.as_str()).unwrap_or("");
            if let Some(concept) = concept_from_link_target(target) {
                links.insert(concept);
            }
        }

        for cap in md_heading_re.captures_iter(&content) {
            let heading = cap[1].trim();
            if !heading.is_empty() {
                headings.push(heading.to_string());
            }
        }

        if headings.is_empty() {
            let arena = Arena::new();
            let ast = parse_document(&arena, &content, &ComrakOptions::default());
            let mut stack = vec![ast];
            while let Some(node) = stack.pop() {
                for child in node.children() {
                    stack.push(child);
                }
                if let NodeValue::Heading(ref heading) = node.data.borrow().value {
                    let mut text = String::new();
                    for child in node.children() {
                        if let NodeValue::Text(ref t) = child.data.borrow().value {
                            text.push_str(t);
                        }
                    }
                    if !text.is_empty() {
                        headings.push(text);
                    } else if heading.level > 0 {
                        headings.push(format!("(heading level {})", heading.level));
                    }
                }
            }
        }

        let page = SourcePage {
            path: entry.path().display().to_string(),
            rel_path: rel_path(rel_root, entry.path()),
            concept,
            raw_concept,
            content,
            links,
            headings,
        };

        let mut basic_metadata: HashMap<String, String> = HashMap::new();
        basic_metadata.insert("concept".to_string(), page.concept.clone());
        basic_metadata.insert("raw_concept".to_string(), page.raw_concept.clone());
        basic_metadata.insert("file_ext".to_string(), "md".to_string());
        basic_metadata.insert("heading_count".to_string(), page.headings.len().to_string());
        basic_metadata.insert(
            "outbound_link_count".to_string(),
            page.links.len().to_string(),
        );
        basic_metadata.insert("path".to_string(), page.path.clone());

        let file_size = metadata.len() as usize;
        let timestamp = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs().to_string());
        let frontmatter = parse_frontmatter_kv(&page.content);
        let author_agent = frontmatter
            .get("author")
            .cloned()
            .or_else(|| frontmatter.get("agent").cloned())
            .or_else(|| frontmatter.get("author_agent").cloned())
            .or_else(|| frontmatter.get("created_by").cloned());
        if frontmatter.contains_key("author") {
            basic_metadata.insert("frontmatter_author".to_string(), "true".to_string());
        }
        if frontmatter.contains_key("agent") {
            basic_metadata.insert("frontmatter_agent".to_string(), "true".to_string());
        }
        basic_metadata.insert("file_size_bytes".to_string(), file_size.to_string());

        tier0_records.push(Tier0Record {
            id: page.rel_path.clone(),
            source: page.rel_path.clone(),
            timestamp,
            author_agent,
            doc_length: file_size,
            metadata: basic_metadata,
        });
        pages.push(page);
    }

    Ok(MarkdownCorpus {
        pages,
        tier0_records,
    })
}

fn docs_dir(root: &Path) -> PathBuf {
    if root.is_file() {
        return root.parent().unwrap_or(root).to_path_buf();
    }
    let docs = root.join("docs");
    if docs.is_dir() {
        docs
    } else {
        root.to_path_buf()
    }
}

pub(crate) fn parse_frontmatter_kv(content: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut lines = content.lines();
    if lines.next() != Some("---") {
        return out;
    }
    for line in lines {
        let trimmed = line.trim();
        if trimmed == "---" {
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            let key = k.trim().to_lowercase();
            let value = v.trim().trim_matches('"').trim_matches('\'').to_string();
            if !key.is_empty() && !value.is_empty() {
                out.insert(key, value);
            }
        }
    }
    out
}
