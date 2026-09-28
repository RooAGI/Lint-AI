//! OpenClaw hook capture documents.
//!
//! Mirrors `crate::integrations::claude_code::document` so that captures from
//! OpenClaw hooks land in the store with the same document-type convention
//! the Claude/Codex lifecycle uses: Checkpoint (pre-compaction), Outcome
//! (a finished unit of work), SessionSummary (session close).

use crate::ids::stable_doc_id_from_source;
use crate::source::SourceDocument;
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Capture document types, following the Claude/Codex lifecycle convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenClawDocumentType {
    /// Pre-compaction checkpoint. The compaction hooks that would trigger this
    /// were not observed on a live OpenClaw host (see the probe report), so no
    /// hook kind wires this yet — the type exists so the capture path is ready.
    Checkpoint,
    /// A finished unit of work (per-turn capture from `agent_end`).
    Outcome,
    /// Session close (authoritative capture from `before_reset`).
    SessionSummary,
}

impl OpenClawDocumentType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Checkpoint => "checkpoint",
            Self::Outcome => "outcome",
            Self::SessionSummary => "session-summary",
        }
    }
}

#[derive(Debug, Clone)]
pub struct OpenClawDocument {
    pub event_id: String,
    pub session_id: String,
    pub document_type: OpenClawDocumentType,
    pub content: String,
    pub cwd: PathBuf,
    pub timestamp: Option<String>,
    /// OpenClaw channel/surface that produced the session, e.g. "webchat".
    pub channel: Option<String>,
}

impl OpenClawDocument {
    pub fn into_source_document(self) -> Result<SourceDocument> {
        let project_id = project_id(&self.cwd)?;
        let document_type = self.document_type.as_str();
        let source = format!(
            "openclaw://{}/{}/{}",
            project_id, self.session_id, document_type
        );
        let mut filters = BTreeMap::from([
            ("integration".to_string(), "openclaw".to_string()),
            ("project_id".to_string(), project_id.clone()),
            ("session_id".to_string(), self.session_id.clone()),
            ("document_type".to_string(), document_type.to_string()),
        ]);
        if let Some(channel) = self.channel.filter(|c| !c.trim().is_empty()) {
            filters.insert("channel".to_string(), channel);
        }

        let content = format!("OpenClaw {document_type}\n{}", self.content.trim());
        let doc_length = content.len();
        Ok(SourceDocument {
            doc_id: stable_doc_id_from_source(&format!("{source}/{}", self.event_id)),
            source,
            content,
            concept: document_type.to_string(),
            group_id: Some(format!("openclaw-session:{project_id}:{}", self.session_id)),
            headings: vec![document_type.to_string()],
            links: vec![],
            timestamp: self.timestamp,
            doc_length,
            author_agent: Some("openclaw".to_string()),
            key_phrases: Vec::new(),
            key_phrase_extraction_hash: String::new(),
            filters,
        })
    }
}

fn project_id(cwd: &Path) -> Result<String> {
    let canonical = cwd.canonicalize().with_context(|| {
        format!(
            "failed to canonicalize OpenClaw working directory {}",
            cwd.display()
        )
    })?;
    Ok(stable_doc_id_from_source(&canonical.to_string_lossy())
        .trim_start_matches("doc:")
        .to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc() -> OpenClawDocument {
        OpenClawDocument {
            event_id: "run-1".to_string(),
            session_id: "session-1".to_string(),
            document_type: OpenClawDocumentType::Outcome,
            content: "User asked for a summary; assistant summarized.".to_string(),
            cwd: std::env::current_dir().unwrap(),
            timestamp: Some("2026-09-28T15:00:00Z".to_string()),
            channel: Some("webchat".to_string()),
        }
    }

    #[test]
    fn openclaw_document_maps_to_session_source_document() {
        let source = doc().into_source_document().unwrap();
        assert!(source.source.starts_with("openclaw://"));
        assert!(source.group_id.unwrap().starts_with("openclaw-session:"));
        assert_eq!(source.author_agent.as_deref(), Some("openclaw"));
        assert_eq!(source.filters["integration"], "openclaw");
        assert_eq!(source.filters["document_type"], "outcome");
        assert_eq!(source.filters["channel"], "webchat");
        assert!(source.content.starts_with("OpenClaw outcome\n"));
    }

    #[test]
    fn document_types_follow_lifecycle_convention() {
        assert_eq!(OpenClawDocumentType::Checkpoint.as_str(), "checkpoint");
        assert_eq!(OpenClawDocumentType::Outcome.as_str(), "outcome");
        assert_eq!(
            OpenClawDocumentType::SessionSummary.as_str(),
            "session-summary"
        );
    }

    #[test]
    fn event_id_makes_document_id_deterministic() {
        let a = doc().into_source_document().unwrap().doc_id;
        let b = doc().into_source_document().unwrap().doc_id;
        assert_eq!(a, b);
    }
}
