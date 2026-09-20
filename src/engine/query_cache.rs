use super::safe_write_text_file;
use crate::index::{DocRecord, MemoryIndex};
use crate::pipeline::{ChunkStrategy, Tier1NerProvider, Tier1TermRankerKind};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use walkdir::WalkDir;

#[derive(Debug, Serialize, Deserialize)]
struct CachedQueryIndex {
    version: String,
    path: String,
    ner_provider: String,
    term_ranker: String,
    spacy_model: String,
    chunk_strategy: String,
    chunk_lines: usize,
    chunk_overlap: usize,
    chunk_target_tokens: usize,
    chunk_max_tokens: usize,
    corpus_fingerprint: String,
    records: Vec<DocRecord>,
}

const QUERY_CACHE_VERSION: &str = "v1-query-cache";
pub(crate) const DEFAULT_QUERY_TOP_K: usize = 5;
pub(crate) const LLM_CONTEXT_CANDIDATE_TOP_K: usize = 20;
pub(crate) const LLM_CONTEXT_DUPLICATE_DOC_PENALTY: f32 = 0.35;
pub(crate) const MAX_RESULT_COUNT: usize = 50;

#[derive(Debug, Clone)]
pub(crate) struct CacheSettings<'a> {
    pub(crate) root_path: &'a str,
    pub(crate) ner_provider: &'a Tier1NerProvider,
    pub(crate) term_ranker: &'a Tier1TermRankerKind,
    pub(crate) spacy_model: &'a str,
    pub(crate) chunk_strategy: &'a ChunkStrategy,
    pub(crate) chunk_lines: usize,
    pub(crate) chunk_overlap: usize,
    pub(crate) chunk_target_tokens: usize,
    pub(crate) chunk_max_tokens: usize,
}

fn query_cache_key(s: &CacheSettings<'_>) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.root_path.hash(&mut hasher);
    format!("{:?}", s.ner_provider)
        .to_lowercase()
        .hash(&mut hasher);
    format!("{:?}", s.term_ranker)
        .to_lowercase()
        .hash(&mut hasher);
    s.spacy_model.hash(&mut hasher);
    format!("{:?}", s.chunk_strategy)
        .to_lowercase()
        .hash(&mut hasher);
    s.chunk_lines.hash(&mut hasher);
    s.chunk_overlap.hash(&mut hasher);
    s.chunk_target_tokens.hash(&mut hasher);
    s.chunk_max_tokens.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn query_cache_file(s: &CacheSettings<'_>) -> PathBuf {
    Path::new(".lint-ai-cache").join(format!("{}.json", query_cache_key(s)))
}

pub(crate) fn query_cache_lexical_dir(s: &CacheSettings<'_>) -> PathBuf {
    Path::new(".lint-ai-cache").join(format!("{}.tantivy", query_cache_key(s)))
}

fn query_cache_core_file(s: &CacheSettings<'_>) -> PathBuf {
    Path::new(".lint-ai-cache").join(format!("{}.core.bin", query_cache_key(s)))
}

fn fingerprint_base_path(root_path: &str) -> PathBuf {
    let root = Path::new(root_path);
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

pub(crate) fn compute_corpus_fingerprint(
    root_path: &str,
    max_files: usize,
    max_depth: usize,
    max_total_bytes: usize,
) -> String {
    let root = Path::new(root_path);
    let base = fingerprint_base_path(root_path);
    let single_file = if root.is_file() {
        Some(root.to_path_buf())
    } else {
        None
    };
    let rel_root = if root.is_file() {
        base.clone()
    } else {
        root.to_path_buf()
    };

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    root_path.hash(&mut hasher);
    let mut files_seen = 0usize;
    let mut total_bytes = 0usize;
    let mut items: Vec<(String, u64, u64, u64)> = Vec::new();

    for entry in WalkDir::new(base).max_depth(max_depth) {
        let Ok(entry) = entry else {
            continue;
        };
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
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        total_bytes = total_bytes.saturating_add(metadata.len() as usize);
        if total_bytes > max_total_bytes {
            break;
        }
        let rel = entry
            .path()
            .strip_prefix(&rel_root)
            .unwrap_or(entry.path())
            .display()
            .to_string();
        let mut mtime_secs = 0u64;
        if let Ok(modified) = metadata.modified() {
            if let Ok(dur) = modified.duration_since(UNIX_EPOCH) {
                mtime_secs = dur.as_secs();
            }
        }
        let content_hash = match fs::read(entry.path()) {
            Ok(bytes) => {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                bytes.hash(&mut h);
                h.finish()
            }
            Err(_) => 0,
        };
        items.push((rel, metadata.len(), mtime_secs, content_hash));
    }

    items.sort_by(|a, b| a.0.cmp(&b.0));
    for (rel, len, mtime_secs, content_hash) in items {
        rel.hash(&mut hasher);
        len.hash(&mut hasher);
        mtime_secs.hash(&mut hasher);
        content_hash.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

pub(crate) fn load_cached_query_index(
    s: &CacheSettings<'_>,
    corpus_fingerprint: &str,
) -> Option<MemoryIndex> {
    let cache_file = query_cache_file(s);
    let data = fs::read_to_string(&cache_file).ok()?;
    let cached: CachedQueryIndex = serde_json::from_str(&data).ok()?;
    if cached.version != QUERY_CACHE_VERSION
        || cached.path != s.root_path
        || cached.ner_provider != format!("{:?}", s.ner_provider).to_lowercase()
        || cached.term_ranker != format!("{:?}", s.term_ranker).to_lowercase()
        || cached.spacy_model != s.spacy_model
        || cached.chunk_strategy != format!("{:?}", s.chunk_strategy).to_lowercase()
        || cached.chunk_lines != s.chunk_lines
        || cached.chunk_overlap != s.chunk_overlap
        || cached.chunk_target_tokens != s.chunk_target_tokens
        || cached.chunk_max_tokens != s.chunk_max_tokens
        || cached.corpus_fingerprint != corpus_fingerprint
    {
        return None;
    }
    let lexical_dir = query_cache_lexical_dir(s);
    let core_file = query_cache_core_file(s);
    match MemoryIndex::load_with_binary_core(
        cached.records.clone(),
        &core_file,
        Some(&lexical_dir),
        false,
    ) {
        Ok(index) => return Some(index),
        Err(err) => {
            eprintln!(
                "warning: binary core cache load failed (falling back): {}",
                err
            );
        }
    }
    Some(MemoryIndex::from_records_with_lexical_dir(
        cached.records,
        Some(&lexical_dir),
        false,
        false,
        false,
    ))
}

pub(crate) fn save_cached_query_index(
    s: &CacheSettings<'_>,
    corpus_fingerprint: &str,
    index: &MemoryIndex,
) -> Result<()> {
    let cache_file = query_cache_file(s);
    if let Some(parent) = cache_file.parent() {
        fs::create_dir_all(parent)?;
    }
    let records = index.docs.values().cloned().collect::<Vec<_>>();
    let payload = CachedQueryIndex {
        version: QUERY_CACHE_VERSION.to_string(),
        path: s.root_path.to_string(),
        ner_provider: format!("{:?}", s.ner_provider).to_lowercase(),
        term_ranker: format!("{:?}", s.term_ranker).to_lowercase(),
        spacy_model: s.spacy_model.to_string(),
        chunk_strategy: format!("{:?}", s.chunk_strategy).to_lowercase(),
        chunk_lines: s.chunk_lines,
        chunk_overlap: s.chunk_overlap,
        chunk_target_tokens: s.chunk_target_tokens,
        chunk_max_tokens: s.chunk_max_tokens,
        corpus_fingerprint: corpus_fingerprint.to_string(),
        records,
    };
    safe_write_text_file(
        &cache_file.display().to_string(),
        &serde_json::to_string(&payload)?,
    )?;
    let core_file = query_cache_core_file(s);
    save_binary_core_atomic(index, &core_file)?;
    Ok(())
}

fn ensure_safe_output_path(path: &Path) -> Result<()> {
    if path.is_dir() {
        anyhow::bail!("refusing to write: output path is a directory");
    }
    if let Ok(meta) = fs::symlink_metadata(path) {
        if meta.file_type().is_symlink() {
            anyhow::bail!("refusing to write: output path is a symlink");
        }
    }
    if let Some(parent) = path.parent() {
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
    Ok(())
}

fn atomic_temp_path(path: &Path) -> Result<PathBuf> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("output");
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    for attempt in 0..1000u32 {
        let candidate = parent.join(format!(".{}.{}.{}.tmp", stem, pid, nanos + attempt as u128));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    anyhow::bail!(
        "unable to allocate temporary file path for {}",
        path.display()
    )
}

fn save_binary_core_atomic(index: &MemoryIndex, path: &Path) -> Result<()> {
    ensure_safe_output_path(path)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temp_path = atomic_temp_path(path)?;
    index.save_binary_core(&temp_path)?;
    fs::rename(&temp_path, path)?;
    Ok(())
}
