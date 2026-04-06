use anyhow::Result;
use comrak::{nodes::NodeValue, parse_document, Arena, ComrakOptions};
use deunicode::deunicode;
use regex::Regex;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use unicode_normalization::UnicodeNormalization;
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct Page {
    pub path: String,
    pub rel_path: String,
    pub concept: String,
    pub raw_concept: String,
    pub content: String,
    pub links: HashSet<String>,
    pub headings: Vec<String>,
}

pub struct Graph {
    pub pages: Vec<Page>,
}

pub fn normalize_concept(s: &str) -> String {
    let normalized: String = s
        .nfc()
        .collect::<String>()
        .trim()
        .to_lowercase()
        .replace('_', " ")
        .replace('-', " ");
    deunicode(&normalized).to_lowercase()
}

fn strip_anchor(target: &str) -> &str {
    target.split('#').next().unwrap_or(target)
}

fn concept_from_link_target(target: &str) -> Option<String> {
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
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(target);
    if stem.is_empty() {
        return None;
    }
    Some(normalize_concept(stem))
}

fn rel_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
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

impl Graph {
    pub fn build(
        path: &str,
        max_bytes: usize,
        max_files: usize,
        max_depth: usize,
        max_total_bytes: usize,
    ) -> Result<Self> {
        let root = Path::new(path);
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

            let ext = entry.path().extension().and_then(|s| s.to_str()).unwrap_or("");
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

            let mut links = HashSet::new();
            let mut headings = Vec::new();

            for cap in wiki_link_re.captures_iter(&content) {
                let target = cap[1].trim();
                if let Some(concept) = concept_from_link_target(target) {
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
                            // Fallback to keep a placeholder for structure.
                            headings.push(format!("(heading level {})", heading.level));
                        }
                    }
                }
            }

            let page = Page {
                path: entry.path().display().to_string(),
                rel_path: rel_path(rel_root, entry.path()),
                concept,
                raw_concept,
                content,
                links,
                headings,
            };

            pages.push(page);
        }

        Ok(Self { pages })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
