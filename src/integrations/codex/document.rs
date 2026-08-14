use crate::ids::stable_doc_id_from_source;
use crate::source::SourceDocument;
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexDocumentType {
    Checkpoint,
    Outcome,
    SessionSummary,
}

impl CodexDocumentType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Checkpoint => "checkpoint",
            Self::Outcome => "outcome",
            Self::SessionSummary => "session-summary",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CodexDocument {
    pub event_id: String,
    pub session_id: String,
    pub document_type: CodexDocumentType,
    pub content: String,
    pub cwd: PathBuf,
    pub timestamp: Option<String>,
    pub command_name: Option<String>,
    pub command_args: Option<String>,
    pub affected_paths: Vec<String>,
    pub branch: Option<String>,
    pub revision: Option<String>,
}

impl CodexDocument {
    pub fn into_source_document(self) -> Result<SourceDocument> {
        let project_id = project_id(&self.cwd)?;
        let document_type = self.document_type.as_str();
        let source = format!(
            "codex://{}/{}/{}",
            project_id, self.session_id, document_type
        );
        let mut filters = BTreeMap::from([
            ("integration".to_string(), "codex".to_string()),
            ("project_id".to_string(), project_id.clone()),
            ("session_id".to_string(), self.session_id.clone()),
            ("document_type".to_string(), document_type.to_string()),
        ]);
        insert_optional(&mut filters, "command_name", self.command_name);
        insert_optional(&mut filters, "command_args", self.command_args);
        insert_optional(&mut filters, "branch", self.branch);
        insert_optional(&mut filters, "revision", self.revision);

        let content = format!("Codex Code {document_type}\n{}", self.content.trim());
        let doc_length = content.len();
        Ok(SourceDocument {
            doc_id: stable_doc_id_from_source(&format!("{source}/{}", self.event_id)),
            source,
            content,
            concept: document_type.to_string(),
            group_id: Some(format!("codex-session:{project_id}:{}", self.session_id)),
            headings: vec![document_type.to_string()],
            links: self.affected_paths,
            timestamp: self.timestamp,
            doc_length,
            author_agent: Some("codex".to_string()),
            filters,
        })
    }
}

pub fn project_id(cwd: &Path) -> Result<String> {
    let canonical = cwd.canonicalize().with_context(|| {
        format!(
            "failed to canonicalize Codex working directory {}",
            cwd.display()
        )
    })?;
    Ok(stable_doc_id_from_source(&canonical.to_string_lossy())
        .trim_start_matches("doc:")
        .to_string())
}

fn insert_optional(filters: &mut BTreeMap<String, String>, key: &str, value: Option<String>) {
    if let Some(value) = value.filter(|value| !value.trim().is_empty()) {
        filters.insert(key.to_string(), value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_document_maps_to_session_source_document() {
        let cwd = std::env::current_dir().unwrap();
        let source = CodexDocument {
            event_id: "event-1".to_string(),
            session_id: "session-1".to_string(),
            document_type: CodexDocumentType::Outcome,
            content: "Implemented segmented retrieval".to_string(),
            cwd,
            timestamp: Some("2026-07-13T12:00:00Z".to_string()),
            command_name: Some("review".to_string()),
            command_args: None,
            affected_paths: vec!["src/pipeline.rs".to_string()],
            branch: Some("feature/hooks".to_string()),
            revision: Some("abc123".to_string()),
        }
        .into_source_document()
        .unwrap();

        assert!(source.group_id.unwrap().contains(":session-1"));
        assert_eq!(source.author_agent.as_deref(), Some("codex"));
        assert_eq!(source.filters["document_type"], "outcome");
        assert_eq!(source.filters["command_name"], "review");
        assert_eq!(source.links, vec!["src/pipeline.rs"]);
        assert!(source.content.contains("Implemented segmented retrieval"));
    }

    #[test]
    fn event_id_makes_document_id_deterministic() {
        let cwd = std::env::current_dir().unwrap();
        let build = || CodexDocument {
            event_id: "same-event".to_string(),
            session_id: "session-1".to_string(),
            document_type: CodexDocumentType::Checkpoint,
            content: "checkpoint".to_string(),
            cwd: cwd.clone(),
            timestamp: None,
            command_name: None,
            command_args: None,
            affected_paths: vec![],
            branch: None,
            revision: None,
        };

        assert_eq!(
            build().into_source_document().unwrap().doc_id,
            build().into_source_document().unwrap().doc_id
        );
    }
}
