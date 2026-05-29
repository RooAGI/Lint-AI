use serde::{Deserialize, Serialize};

use crate::ids::stable_doc_id_from_source;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceDocument {
    pub doc_id: String,
    pub source: String,
    pub content: String,
    pub concept: String,
    #[serde(default)]
    pub group_id: Option<String>,
    #[serde(default)]
    pub headings: Vec<String>,
    #[serde(default)]
    pub links: Vec<String>,
    pub timestamp: Option<String>,
    pub doc_length: usize,
    pub author_agent: Option<String>,
}

impl SourceDocument {
    pub fn with_stable_doc_id_from_source(
        source: String,
        content: String,
        concept: String,
        group_id: Option<String>,
        headings: Vec<String>,
        links: Vec<String>,
        timestamp: Option<String>,
        author_agent: Option<String>,
    ) -> Self {
        let doc_id = stable_doc_id_from_source(&source);
        let doc_length = content.len();
        Self {
            doc_id,
            source,
            content,
            concept,
            group_id,
            headings,
            links,
            timestamp,
            doc_length,
            author_agent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_stable_doc_id_sets_doc_length_from_content() {
        let doc = SourceDocument::with_stable_doc_id_from_source(
            "docs/install.md".to_string(),
            "install guide for linux hosts".to_string(),
            "install guide".to_string(),
            None,
            vec!["Overview".to_string()],
            vec!["docs/setup.md".to_string()],
            Some("2026-05-29".to_string()),
            Some("agent".to_string()),
        );

        assert_eq!(doc.doc_length, "install guide for linux hosts".len());
        assert_eq!(doc.source, "docs/install.md");
        assert_eq!(doc.concept, "install guide");
        assert_eq!(doc.headings, vec!["Overview".to_string()]);
        assert_eq!(doc.links, vec!["docs/setup.md".to_string()]);
        assert_eq!(doc.author_agent.as_deref(), Some("agent"));
        assert_eq!(doc.timestamp.as_deref(), Some("2026-05-29"));
        assert!(!doc.doc_id.is_empty());
    }

    #[test]
    fn with_stable_doc_id_is_deterministic_for_same_source() {
        let first = SourceDocument::with_stable_doc_id_from_source(
            "docs/install.md".to_string(),
            "content one".to_string(),
            "install guide".to_string(),
            None,
            vec![],
            vec![],
            None,
            None,
        );
        let second = SourceDocument::with_stable_doc_id_from_source(
            "docs/install.md".to_string(),
            "different content".to_string(),
            "install guide".to_string(),
            None,
            vec![],
            vec![],
            None,
            None,
        );

        assert_eq!(first.doc_id, second.doc_id);
    }

    #[test]
    fn source_document_round_trips_through_json() {
        let doc = SourceDocument::with_stable_doc_id_from_source(
            "docs/install.md".to_string(),
            "install guide for linux hosts".to_string(),
            "install guide".to_string(),
            Some("group-1".to_string()),
            vec!["Overview".to_string()],
            vec!["docs/setup.md".to_string()],
            Some("2026-05-29".to_string()),
            Some("agent".to_string()),
        );

        let json = serde_json::to_string(&doc).unwrap();
        let decoded: SourceDocument = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.doc_id, doc.doc_id);
        assert_eq!(decoded.source, doc.source);
        assert_eq!(decoded.content, doc.content);
        assert_eq!(decoded.concept, doc.concept);
        assert_eq!(decoded.group_id, doc.group_id);
        assert_eq!(decoded.headings, doc.headings);
        assert_eq!(decoded.links, doc.links);
        assert_eq!(decoded.timestamp, doc.timestamp);
        assert_eq!(decoded.doc_length, doc.doc_length);
        assert_eq!(decoded.author_agent, doc.author_agent);
    }
}
