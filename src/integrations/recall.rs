//! One-shot chunk-level recall over the unified store.
//!
//! The MCP servers answer the same question over a long-lived stdio session.
//! A remote worker cannot hold that session open, so this module runs the same
//! retrieval once, writes JSON to stdout and returns, which makes `lint-ai`
//! usable as a plain subprocess.

use crate::adapters::{
    apply_ignore_paths, build_project_graph, graph_to_source_documents, AdapterInput,
};
use crate::index::{DocRecord, SearchResult, SectionChunk};
use crate::integrations::mcp_index;
use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

/// The project index is shared with the MCP servers rather than kept under a
/// name of its own. Both paths build it from the same graph with the same
/// options, so a second copy would only double the disk cost and the rebuild
/// work each time the workspace moves on.
/// A single chunk of a long session memory can run to tens of kilobytes, and
/// every result crosses a gRPC boundary before anyone reads it, so the text is
/// cut to a window a reader (human or model) actually consumes. Cutting is done
/// by whole lines so the reported line range stays true to the text emitted.
const MAX_CHUNK_TEXT_BYTES: usize = 1_600;

/// Directories holding code the project did not write. A recall over a project
/// root without ignores answers with a vendored README rather than anything the
/// caller asked about, and a worker on another machine has no chance to notice.
/// The configured list is added to this rather than replacing it, because no
/// query for a project's own memory is improved by its dependencies.
const ALWAYS_IGNORED: [&str; 7] = [
    "node_modules",
    "target",
    "dist",
    "build",
    "vendor",
    "coverage",
    ".git",
];

/// Options for a one-shot recall over a project root.
#[derive(Debug, Clone)]
pub struct RecallOptions<'a> {
    pub root: &'a Path,
    pub query: &'a str,
    pub result_count: usize,
    pub ignore_paths: &'a [String],
    pub max_bytes: usize,
    pub max_files: usize,
    pub max_depth: usize,
    pub max_total_bytes: usize,
    pub index_name: &'a str,
    pub memory_name: &'a str,
}

/// The wire contract consumed by lint-service: a `results` array whose every
/// element carries a numeric `score`. Fields beyond that pass through the
/// dispatcher untouched.
#[derive(Debug, Clone, Serialize)]
pub struct RecallOutput {
    pub query: String,
    pub root: String,
    pub elapsed_ms: u128,
    pub results: Vec<RecallHit>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecallHit {
    pub score: f32,
    /// Documents keep their repo-relative path here; the dispatcher rewrites
    /// relative paths against the worker's base path, so they are never
    /// absolutised on this side. A memory is not a file, so it carries the
    /// corpus it lives in — the dispatcher turns that into a real absolute path
    /// too, which is what a reader on another machine can act on.
    pub doc_id: String,
    pub source: String,
    /// The record's own identity. It is kept out of `doc_id` because the
    /// dispatcher treats every relative-looking value there as a path and
    /// prefixes it, which turned `doc:2df221ab…` into a file that never existed.
    pub record_id: String,
    /// Where a memory came from, verbatim, for the same reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    pub kind: RecallKind,
    pub chunk: RecallChunk,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RecallKind {
    Document,
    Memory,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecallChunk {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_id: Option<String>,
    /// Null when the record carries no honest line numbers. A wrong range is
    /// worse than none: the desktop app jumps the editor to it.
    pub start_line: Option<usize>,
    pub end_line: Option<usize>,
    /// True when the chunk was longer than the cap and only a window is here.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

/// Query the unified store — project documents and recorded memories together —
/// and return chunk-level hits.
pub fn recall(options: &RecallOptions<'_>) -> Result<RecallOutput> {
    let started = Instant::now();
    let root = options
        .root
        .canonicalize()
        .with_context(|| format!("unable to resolve project root {}", options.root.display()))?;

    let mut ignore_paths = options.ignore_paths.to_vec();
    for fragment in ALWAYS_IGNORED {
        if !ignore_paths.iter().any(|known| known == fragment) {
            ignore_paths.push(fragment.to_string());
        }
    }
    let input = AdapterInput {
        root: &root,
        max_bytes: options.max_bytes,
        max_files: options.max_files,
        max_depth: options.max_depth,
        max_total_bytes: options.max_total_bytes,
    };
    let mut store = mcp_index::open_persistent_store(
        &root,
        options.index_name,
        options.memory_name,
        &ignore_paths,
        || {
            let graph = build_project_graph(&input)?;
            let graph = apply_ignore_paths(graph, &ignore_paths);
            Ok(graph_to_source_documents(&graph))
        },
    )
    .with_context(|| format!("unable to open the index for {}", root.display()))?;

    let results = store
        .query(options.query, options.result_count)
        .with_context(|| format!("query failed: {}", options.query))?;

    let hits = results
        .iter()
        .filter_map(|result| {
            store
                .record_by_id(&result.doc_id)
                .map(|record| hit_from(result, record, options.query))
        })
        .collect();

    Ok(RecallOutput {
        query: options.query.to_string(),
        root: root.to_string_lossy().to_string(),
        elapsed_ms: started.elapsed().as_millis(),
        results: hits,
    })
}

fn hit_from(result: &SearchResult, record: &DocRecord, query: &str) -> RecallHit {
    let filter = |key: &str| record.filters.get(key).cloned();
    // Only the integrations stamp these filters onto what they capture, so
    // their presence is what separates a recorded memory from a project file.
    let kind = if record.filters.contains_key("session_id")
        || record.filters.contains_key("document_type")
    {
        RecallKind::Memory
    } else {
        RecallKind::Document
    };
    let is_memory = kind == RecallKind::Memory;
    RecallHit {
        score: result.score,
        doc_id: if is_memory {
            memory_corpus_path(&record.source)
        } else {
            result.doc_id.clone()
        },
        source: if is_memory {
            memory_corpus_path(&record.source)
        } else {
            record.source.clone()
        },
        record_id: result.doc_id.clone(),
        origin: is_memory.then(|| record.source.clone()),
        kind,
        chunk: best_chunk(record, query, &result.matched_terms),
        session_id: filter("session_id"),
        branch: filter("branch"),
        revision: filter("revision"),
        document_type: filter("document_type"),
        timestamp: record.timestamp.clone(),
    }
}

/// Return the provider-specific corpus path for a recorded memory.
fn memory_corpus_path(source: &str) -> String {
    let name = match source.split_once("://").map(|(provider, _)| provider) {
        Some("codex") => "codex-memory",
        Some("gemini-cli") => "gemini-cli-memory",
        Some("agy") => "agy-memory",
        _ => "claude-memory",
    };
    format!(".lint-ai/{name}")
}

/// Pick the section chunk that best answers the query and bound its text.
fn best_chunk(record: &DocRecord, query: &str, matched_terms: &[String]) -> RecallChunk {
    let terms = excerpt_terms(query, matched_terms);
    let selected = record
        .section_chunks
        .iter()
        .max_by_key(|chunk| chunk_score(chunk, &terms));
    match selected {
        Some(chunk) => {
            let (content, first_line) =
                skip_recorder_header(&chunk.content, &terms, chunk.start_line.max(1));
            let (text, start_line, end_line, truncated) =
                bounded_window(content, first_line, &terms);
            RecallChunk {
                text,
                chunk_id: Some(chunk.chunk_id.clone()),
                start_line,
                end_line,
                truncated,
            }
        }
        // Chunking can legitimately produce nothing (an empty or unheaded
        // document). The record content still starts at line one, so a window
        // over it carries a range we can stand behind.
        None => {
            let (content, first_line) = skip_recorder_header(&record.content, &terms, 1);
            let (text, start_line, end_line, truncated) =
                bounded_window(content, first_line, &terms);
            RecallChunk {
                text,
                chunk_id: None,
                start_line,
                end_line,
                truncated,
            }
        }
    }
}

fn chunk_score(chunk: &SectionChunk, terms: &HashSet<String>) -> usize {
    let haystack = format!("{}\n{}", chunk.heading, chunk.content).to_ascii_lowercase();
    terms
        .iter()
        .filter(|term| haystack.contains(term.as_str()))
        .count()
}

/// Take a contiguous run of lines that fits the byte cap, centred on the
/// densest match, and report the line numbers that run really occupies.
/// The recorder writes a machine header above what it concluded — the agent
/// that wrote the memory and the type of the record. An excerpt that opens on
/// those lines spends its budget saying nothing, so leading header lines are
/// dropped when they carry none of what was asked about. The reported range
/// moves with them; a range that no longer matches its text would be worse than
/// the boilerplate.
fn skip_recorder_header<'a>(
    content: &'a str,
    terms: &HashSet<String>,
    first_line: usize,
) -> (&'a str, usize) {
    let mut offset = 0;
    let mut skipped = 0;
    for line in content.lines() {
        let trimmed = line.trim();
        let is_header = trimmed.starts_with("Claude Code ")
            || trimmed.starts_with("Codex ")
            || trimmed.starts_with("Gemini CLI ")
            || trimmed.starts_with("Memory type:");
        if !is_header {
            break;
        }
        let lower = trimmed.to_ascii_lowercase();
        if terms.iter().any(|term| lower.contains(term.as_str())) {
            break;
        }
        offset += line.len() + 1;
        skipped += 1;
    }
    if offset >= content.len() {
        return (content, first_line);
    }
    (&content[offset..], first_line + skipped)
}

fn bounded_window(
    content: &str,
    first_line: usize,
    terms: &HashSet<String>,
) -> (String, Option<usize>, Option<usize>, bool) {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return (String::new(), None, None, false);
    }
    // A stored chunk numbered from zero would push every reported range off by
    // one, so fall back to one-based numbering rather than emit a wrong range.
    let first_line = first_line.max(1);
    if content.len() <= MAX_CHUNK_TEXT_BYTES {
        return (
            content.trim_end().to_string(),
            Some(first_line),
            Some(first_line + lines.len() - 1),
            false,
        );
    }

    let best = lines
        .iter()
        .enumerate()
        .max_by_key(|(index, line)| {
            let lower = line.to_ascii_lowercase();
            let score = terms.iter().filter(|term| lower.contains(*term)).count();
            (score, std::cmp::Reverse(*index))
        })
        .map(|(index, _)| index)
        .unwrap_or(0);

    let mut start = best;
    let mut end = best;
    let mut used = lines[best].len();
    // Grow outwards a line at a time so the window stays contiguous, which is
    // what makes the reported range honest.
    loop {
        // Each neighbouring line costs its own bytes plus the newline the join
        // will put back between them.
        let after_cost = lines.get(end + 1).map(|line| line.len() + 1);
        let grew_after = matches!(after_cost, Some(cost) if used + cost <= MAX_CHUNK_TEXT_BYTES);
        if grew_after {
            end += 1;
            used += lines[end].len() + 1;
        }
        let before_cost = start.checked_sub(1).map(|index| lines[index].len() + 1);
        let grew_before = matches!(before_cost, Some(cost) if used + cost <= MAX_CHUNK_TEXT_BYTES);
        if grew_before {
            start -= 1;
            used += lines[start].len() + 1;
        }
        if !grew_after && !grew_before {
            break;
        }
    }

    let joined = lines[start..=end].join("\n");
    let joined_len = joined.len();
    let text = if joined_len > MAX_CHUNK_TEXT_BYTES {
        truncate_utf8(&joined, MAX_CHUNK_TEXT_BYTES.saturating_sub(3))
    } else {
        joined
    };
    let truncated = start > 0 || end + 1 < lines.len() || joined_len > MAX_CHUNK_TEXT_BYTES;
    (
        text.trim_end().to_string(),
        Some(first_line + start),
        Some(first_line + end),
        truncated,
    )
}

/// The terms an excerpt is scored against: what the user asked plus what the
/// index says actually matched.
pub(crate) fn excerpt_terms(query: &str, matched_terms: &[String]) -> HashSet<String> {
    let mut terms = query_terms(query);
    terms.extend(matched_terms.iter().flat_map(|term| query_terms(term)));
    terms
}

pub(crate) fn query_terms(value: &str) -> HashSet<String> {
    const STOP_WORDS: &[&str] = &[
        "about", "after", "again", "also", "and", "are", "before", "from", "have", "into", "our",
        "that", "the", "their", "this", "was", "what", "when", "which", "with", "without",
    ];
    value
        .split(|character: char| {
            !character.is_alphanumeric() && character != '-' && character != '_'
        })
        .map(str::to_ascii_lowercase)
        .filter(|term| term.len() >= 3 && !STOP_WORDS.contains(&term.as_str()))
        .collect()
}

#[allow(dead_code)]
pub(crate) fn truncate_utf8(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", value[..end].trim_end())
}

/// The lines of `content` that carry the query, joined for injection into a
/// prompt. Unlike [`best_chunk`] this drops the gaps between the lines it
/// keeps, so it reports no line range.
#[allow(dead_code)]
pub(crate) fn relevant_excerpt(
    content: &str,
    query: &str,
    matched_terms: &[String],
    max_bytes: usize,
) -> String {
    let terms = excerpt_terms(query, matched_terms);
    let mut lines = content
        .lines()
        .enumerate()
        .map(|(index, line)| {
            let lower = line.to_ascii_lowercase();
            let score = terms.iter().filter(|term| lower.contains(*term)).count();
            (index, score, line.trim())
        })
        .filter(|(_, _, line)| !line.is_empty())
        .collect::<Vec<_>>();
    lines.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));

    let mut selected = lines
        .iter()
        .filter(|(_, score, _)| *score > 0)
        .take(3)
        .cloned()
        .collect::<Vec<_>>();
    if selected.is_empty() {
        selected.extend(lines.into_iter().take(2));
    }
    selected.sort_by_key(|(index, _, _)| *index);
    truncate_utf8(
        &selected
            .into_iter()
            .map(|(_, _, line)| line)
            .collect::<Vec<_>>()
            .join("\n"),
        max_bytes,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{Provenance, ScoreBreakdown};
    use crate::pipeline::{IndexStore, PipelineOptions};
    use crate::source::SourceDocument;
    use std::collections::BTreeMap;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn provenance() -> Provenance {
        Provenance {
            source: "test".to_string(),
            timestamp: None,
            ner_provider: "heuristic".to_string(),
            term_ranker: "yake-style".to_string(),
            index_version: "test".to_string(),
        }
    }

    fn chunk(content: &str, start_line: usize, end_line: usize) -> SectionChunk {
        SectionChunk {
            chunk_id: "chunk:test".to_string(),
            heading: "(document)".to_string(),
            content: content.to_string(),
            start_line,
            end_line,
            timestamp: None,
            key_entities: Vec::new(),
            important_terms: Vec::new(),
        }
    }

    fn record(
        doc_id: &str,
        source: &str,
        content: &str,
        filters: BTreeMap<String, String>,
        section_chunks: Vec<SectionChunk>,
    ) -> DocRecord {
        DocRecord {
            doc_id: doc_id.to_string(),
            source: source.to_string(),
            content: content.to_string(),
            timestamp: Some("2026-08-21T23:46:12+00:00".to_string()),
            doc_length: content.len(),
            author_agent: None,
            group_id: None,
            filters,
            probable_topic: None,
            doc_type_guess: None,
            headings: Vec::new(),
            doc_links: Vec::new(),
            temporal_terms: Vec::new(),
            key_entities: Vec::new(),
            important_terms: Vec::new(),
            section_chunks,
            embedding: None,
            top_claims: Vec::new(),
            provenance: provenance(),
        }
    }

    fn search_result(doc_id: &str, source: &str, matched_terms: &[&str]) -> SearchResult {
        SearchResult {
            doc_id: doc_id.to_string(),
            source: source.to_string(),
            group_id: None,
            score: 88.4,
            score_breakdown: ScoreBreakdown::default(),
            matched_entities: Vec::new(),
            matched_terms: matched_terms.iter().map(|t| t.to_string()).collect(),
            probable_topic: None,
            doc_type_guess: None,
        }
    }

    fn memory_filters() -> BTreeMap<String, String> {
        BTreeMap::from([
            ("session_id".to_string(), "6cccf167".to_string()),
            ("branch".to_string(), "master".to_string()),
            ("revision".to_string(), "d4d78b7".to_string()),
            ("document_type".to_string(), "outcome".to_string()),
        ])
    }

    #[test]
    fn project_recall_uses_each_provider_store() {
        let providers = [
            ("claude-code", "claude-mcp-index", "claude-memory"),
            ("codex", "codex-mcp-index", "codex-memory"),
            ("gemini-cli", "gemini-mcp-index", "gemini-cli-memory"),
            ("agy", "agy-mcp-index", "agy-memory"),
        ];

        for (provider, index_name, memory_name) in providers {
            let root = std::env::temp_dir().join(format!(
                "lint-ai-recall-{provider}-{}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir_all(root.join("docs")).unwrap();
            fs::write(
                root.join("docs").join("architecture.md"),
                "# Architecture\nThe project architecture uses provider-aware recall.",
            )
            .unwrap();

            let memory_root = root.join(".lint-ai").join(memory_name);
            let mut memory = IndexStore::at_path(&memory_root, PipelineOptions::default()).unwrap();
            memory.upsert(SourceDocument {
                doc_id: format!("{provider}-session-1"),
                source: format!("{provider}://project/session-1/outcome"),
                content: "The provider-aware recall decision was recorded in this session."
                    .to_string(),
                concept: "outcome".to_string(),
                group_id: Some(format!("{provider}-session:session-1")),
                filters: memory_filters(),
                headings: vec!["Outcome".to_string()],
                links: vec![],
                timestamp: Some("2026-08-22T00:00:00Z".to_string()),
                doc_length: 0,
                author_agent: Some(provider.to_string()),
            });
            memory.refresh().unwrap();
            drop(memory);

            let options = RecallOptions {
                root: &root,
                query: "provider-aware recall",
                result_count: 10,
                ignore_paths: &[],
                max_bytes: 5_000_000,
                max_files: 50_000,
                max_depth: 20,
                max_total_bytes: 100_000_000,
                index_name,
                memory_name,
            };
            let output = recall(&options).unwrap();

            assert!(
                output
                    .results
                    .iter()
                    .any(|hit| hit.kind == RecallKind::Document
                        && hit.source == "docs/architecture.md"),
                "{provider} recall did not return the project document: {:?}",
                output.results
            );
            assert!(
                output.results.iter().any(|hit| {
                    hit.kind == RecallKind::Memory
                        && hit.doc_id == format!(".lint-ai/{memory_name}")
                        && hit.origin.as_deref()
                            == Some(&format!("{provider}://project/session-1/outcome"))
                }),
                "{provider} recall did not return its memory corpus: {:?}",
                output.results
            );

            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn memory_hit_carries_session_branch_and_revision() {
        let record = record(
            "doc:2df221abf669e91a",
            "claude-code://project/session/outcome",
            "we dropped embeddings",
            memory_filters(),
            vec![chunk("we dropped embeddings", 12, 12)],
        );
        let hit = hit_from(
            &search_result("doc:2df221abf669e91a", "claude-code://x", &["embeddings"]),
            &record,
            "why did we drop embeddings",
        );

        assert_eq!(hit.kind, RecallKind::Memory);
        assert_eq!(hit.session_id.as_deref(), Some("6cccf167"));
        assert_eq!(hit.branch.as_deref(), Some("master"));
        assert_eq!(hit.revision.as_deref(), Some("d4d78b7"));
        assert_eq!(hit.document_type.as_deref(), Some("outcome"));
        assert_eq!(hit.chunk.start_line, Some(12));
        assert!(hit.chunk.text.contains("embeddings"));
    }

    #[test]
    fn document_hit_carries_no_session_metadata() {
        let record = record(
            "docs/architecture.md",
            "docs/architecture.md",
            "# Architecture\nembeddings were dropped",
            BTreeMap::new(),
            vec![chunk("embeddings were dropped", 88, 88)],
        );
        let hit = hit_from(
            &search_result("docs/architecture.md", "docs/architecture.md", &[]),
            &record,
            "embeddings",
        );

        assert_eq!(hit.kind, RecallKind::Document);
        assert!(hit.session_id.is_none());
        assert!(hit.branch.is_none());
        assert!(hit.revision.is_none());
        // Repo-relative, because the dispatcher absolutises it against the
        // worker's base path.
        assert_eq!(hit.doc_id, "docs/architecture.md");
    }

    #[test]
    fn chunk_text_is_capped_and_the_reported_range_matches_it() {
        let mut lines: Vec<String> = (0..400)
            .map(|index| format!("filler line {index}"))
            .collect();
        lines[300] = "the verdict of the gauntlet critic".to_string();
        let body = lines.join("\n");
        let record = record(
            "doc:long",
            "claude-code://project/session/outcome",
            &body,
            memory_filters(),
            vec![chunk(&body, 1, 400)],
        );
        let hit = hit_from(
            &search_result("doc:long", "claude-code://x", &["verdict"]),
            &record,
            "gauntlet critic verdict",
        );

        assert!(hit.chunk.text.len() <= MAX_CHUNK_TEXT_BYTES);
        assert!(hit.chunk.truncated);
        assert!(hit.chunk.text.contains("verdict"));
        let start = hit.chunk.start_line.expect("a window has a start");
        let end = hit.chunk.end_line.expect("a window has an end");
        assert_eq!(end - start + 1, hit.chunk.text.lines().count());
        // The window must really sit where it says it does.
        assert_eq!(
            hit.chunk.text.lines().next(),
            body.lines().nth(start - 1),
            "reported start_line does not match the emitted text"
        );
    }

    #[test]
    fn a_single_long_line_is_still_bounded() {
        let body = format!("prefix {} suffix", "x".repeat(MAX_CHUNK_TEXT_BYTES * 2));
        let record = record(
            "doc:long-line",
            "claude-code://project/session/outcome",
            &body,
            memory_filters(),
            vec![chunk(&body, 7, 7)],
        );
        let hit = hit_from(
            &search_result("doc:long-line", "claude-code://x", &["prefix"]),
            &record,
            "prefix",
        );

        assert!(hit.chunk.text.len() <= MAX_CHUNK_TEXT_BYTES);
        assert!(hit.chunk.truncated);
        assert_eq!(hit.chunk.start_line, Some(7));
        assert_eq!(hit.chunk.end_line, Some(7));
    }

    #[test]
    fn a_record_without_chunks_still_reports_an_honest_range() {
        let record = record(
            "doc:unchunked",
            "claude-code://project/session/outcome",
            "first line\nsecond line",
            memory_filters(),
            Vec::new(),
        );
        let hit = hit_from(
            &search_result("doc:unchunked", "claude-code://x", &[]),
            &record,
            "second",
        );

        assert!(hit.chunk.chunk_id.is_none());
        assert_eq!(hit.chunk.start_line, Some(1));
        assert_eq!(hit.chunk.end_line, Some(2));
    }

    #[test]
    fn the_chunk_carrying_the_query_wins() {
        let record = record(
            "doc:multi",
            "claude-code://project/session/outcome",
            "unrelated\nembeddings were dropped",
            memory_filters(),
            vec![
                chunk("unrelated prose about packaging", 1, 1),
                chunk("embeddings were dropped", 2, 2),
            ],
        );
        let hit = hit_from(
            &search_result("doc:multi", "claude-code://x", &["embeddings"]),
            &record,
            "why did we drop embeddings",
        );

        assert_eq!(hit.chunk.start_line, Some(2));
        assert!(hit.chunk.text.contains("embeddings"));
    }
}
