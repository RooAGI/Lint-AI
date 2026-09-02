use crate::source::SourceDocument;
use crate::temporal::parse_temporal_date;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SupersessionOptions {
    pub enabled: bool,
    pub suppress_confidence: f32,
    pub conflict_confidence: f32,
}

impl Default for SupersessionOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            suppress_confidence: 0.90,
            conflict_confidence: 0.65,
        }
    }
}

impl SupersessionOptions {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(0.0..=1.0).contains(&self.conflict_confidence)
            || !(0.0..=1.0).contains(&self.suppress_confidence)
        {
            anyhow::bail!("semantic relationship confidence thresholds must be between 0 and 1");
        }
        if self.conflict_confidence > self.suppress_confidence {
            anyhow::bail!("conflict confidence must not exceed suppress confidence");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticRelationKind {
    Supersedes,
    ConflictsWith,
    Confirms,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticStatus {
    Current,
    Superseded,
    Conflicted,
    Historical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticClaim {
    pub claim_id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub source_doc_id: String,
    pub evidence: String,
    pub effective_at: Option<String>,
    pub scope: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticRelation {
    pub relation_id: String,
    pub kind: SemanticRelationKind,
    /// The newer or confirming claim.
    pub source_claim_id: String,
    /// The older or conflicting claim.
    pub target_claim_id: String,
    pub source_doc_id: String,
    pub target_doc_id: String,
    pub confidence: f32,
    pub method: String,
    pub evidence: Vec<String>,
    pub scope: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DocumentSemanticState {
    pub status: Option<SemanticStatus>,
    pub superseded_by: Option<String>,
    pub relation_confidence: Option<f32>,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct SemanticRelationStore {
    claims: Vec<SemanticClaim>,
    relations: Vec<SemanticRelation>,
    document_states: HashMap<String, DocumentSemanticState>,
}

impl SemanticRelationStore {
    pub fn from_documents<'a, I>(documents: I, options: SupersessionOptions) -> Self
    where
        I: IntoIterator<Item = &'a SourceDocument>,
    {
        Self::try_from_documents(documents, options).expect("invalid semantic supersession options")
    }

    pub fn try_from_documents<'a, I>(
        documents: I,
        options: SupersessionOptions,
    ) -> anyhow::Result<Self>
    where
        I: IntoIterator<Item = &'a SourceDocument>,
    {
        if !options.enabled {
            return Ok(Self::default());
        }
        options.validate()?;

        let documents = documents.into_iter().collect::<Vec<_>>();
        let by_id = documents
            .iter()
            .map(|doc| (doc.doc_id.as_str(), *doc))
            .collect::<HashMap<_, _>>();
        let mut claims = documents
            .iter()
            .flat_map(|doc| extract_claims(doc))
            .collect::<Vec<_>>();
        claims.sort_by(|a, b| {
            claim_date(a)
                .cmp(&claim_date(b))
                .then_with(|| a.source_doc_id.cmp(&b.source_doc_id))
                .then_with(|| a.claim_id.cmp(&b.claim_id))
        });
        let claim_by_id = claims
            .iter()
            .map(|claim| (claim.claim_id.clone(), claim.clone()))
            .collect::<HashMap<_, _>>();

        let mut relations = Vec::new();
        let mut seen_relations = HashSet::new();

        // Explicit links are authoritative evidence, regardless of source type.
        for doc in &documents {
            let Some(target_id) = doc.filters.get("supersedes_id") else {
                continue;
            };
            let Some(target) = by_id.get(target_id.as_str()) else {
                continue;
            };
            if semantic_scope(doc) != semantic_scope(target) {
                continue;
            }
            let source_claim = claims
                .iter()
                .find(|claim| claim.source_doc_id == doc.doc_id)
                .cloned()
                .unwrap_or_else(|| document_claim(doc));
            let target_claim = claims
                .iter()
                .find(|claim| claim.source_doc_id == target.doc_id)
                .cloned()
                .unwrap_or_else(|| document_claim(target));
            push_relation(
                &mut relations,
                &mut seen_relations,
                &source_claim,
                &target_claim,
                SemanticRelationKind::Supersedes,
                1.0,
                "explicit",
                vec![format!(
                    "{} explicitly supersedes {}",
                    doc.doc_id, target.doc_id
                )],
            );
        }

        let mut latest_by_key = HashMap::<(String, String, String), String>::new();
        for claim in &claims {
            let key = (
                claim.scope.clone(),
                normalize(&claim.subject),
                normalize(&claim.predicate),
            );
            if let Some(previous_id) = latest_by_key.get(&key) {
                let Some(previous) = claim_by_id.get(previous_id) else {
                    continue;
                };
                if previous.source_doc_id == claim.source_doc_id {
                    latest_by_key.insert(key, claim.claim_id.clone());
                    continue;
                }
                if normalize(&previous.object) == normalize(&claim.object) {
                    push_relation(
                        &mut relations,
                        &mut seen_relations,
                        claim,
                        previous,
                        SemanticRelationKind::Confirms,
                        0.95,
                        "canonical_claim",
                        vec!["same canonical claim and value".to_string()],
                    );
                } else {
                    let source_doc = by_id.get(claim.source_doc_id.as_str()).copied();
                    let target_doc = by_id.get(previous.source_doc_id.as_str()).copied();
                    let direct_correction =
                        source_doc.is_some_and(|doc| has_correction_cue(&doc.content));
                    let chronological = claim_date(claim)
                        .zip(claim_date(previous))
                        .is_some_and(|(newer, older)| newer > older);
                    let same_source_kind = source_doc
                        .zip(target_doc)
                        .is_some_and(|(source, target)| source_kind(source) == source_kind(target));
                    let (kind, confidence, method) = if direct_correction {
                        (SemanticRelationKind::Supersedes, 0.95, "correction_cue")
                    } else if chronological && same_source_kind {
                        (
                            SemanticRelationKind::Supersedes,
                            0.90,
                            "canonical_claim_and_time",
                        )
                    } else {
                        (
                            SemanticRelationKind::ConflictsWith,
                            0.75,
                            "canonical_claim_conflict",
                        )
                    };
                    push_relation(
                        &mut relations,
                        &mut seen_relations,
                        claim,
                        previous,
                        kind,
                        confidence,
                        method,
                        vec![format!(
                            "{} changed from '{}' to '{}'",
                            claim.subject, previous.object, claim.object
                        )],
                    );
                }
            }
            latest_by_key.insert(key, claim.claim_id.clone());
        }

        let document_states = build_document_states(&documents, &claims, &relations, options);
        Ok(Self {
            claims,
            relations,
            document_states,
        })
    }

    pub fn claims(&self) -> &[SemanticClaim] {
        &self.claims
    }

    pub fn relations(&self) -> &[SemanticRelation] {
        &self.relations
    }

    pub fn document_state(&self, doc_id: &str) -> DocumentSemanticState {
        self.document_states
            .get(doc_id)
            .cloned()
            .unwrap_or_default()
    }
}

fn build_document_states(
    documents: &[&SourceDocument],
    claims: &[SemanticClaim],
    relations: &[SemanticRelation],
    options: SupersessionOptions,
) -> HashMap<String, DocumentSemanticState> {
    let mut states = documents
        .iter()
        .map(|doc| {
            (
                doc.doc_id.clone(),
                DocumentSemanticState {
                    status: Some(SemanticStatus::Current),
                    ..DocumentSemanticState::default()
                },
            )
        })
        .collect::<HashMap<_, _>>();
    for relation in relations {
        match relation.kind {
            SemanticRelationKind::Supersedes
                if relation.confidence >= options.suppress_confidence =>
            {
                let state = states.entry(relation.target_doc_id.clone()).or_default();
                if relation.method != "explicit"
                    && !document_contains_only_target_claim(documents, claims, relation)
                {
                    if state.status != Some(SemanticStatus::Superseded) {
                        state.status = Some(SemanticStatus::Conflicted);
                        state.relation_confidence = Some(relation.confidence);
                        state.evidence = relation.evidence.clone();
                    }
                    continue;
                }
                if state.relation_confidence.unwrap_or_default() <= relation.confidence {
                    state.status = Some(SemanticStatus::Superseded);
                    state.superseded_by = Some(relation.source_doc_id.clone());
                    state.relation_confidence = Some(relation.confidence);
                    state.evidence = relation.evidence.clone();
                }
            }
            SemanticRelationKind::ConflictsWith
                if relation.confidence >= options.conflict_confidence =>
            {
                for doc_id in [&relation.source_doc_id, &relation.target_doc_id] {
                    let state = states.entry(doc_id.clone()).or_default();
                    if state.status != Some(SemanticStatus::Superseded) {
                        state.status = Some(SemanticStatus::Conflicted);
                        state.relation_confidence = Some(relation.confidence);
                        state.evidence = relation.evidence.clone();
                    }
                }
            }
            _ => {}
        }
    }
    states
}

fn document_contains_only_target_claim(
    documents: &[&SourceDocument],
    claims: &[SemanticClaim],
    relation: &SemanticRelation,
) -> bool {
    let mut target_claims = claims
        .iter()
        .filter(|claim| claim.source_doc_id == relation.target_doc_id);
    let Some(target_claim) = target_claims.next() else {
        return false;
    };
    if target_claims.next().is_some() || target_claim.claim_id != relation.target_claim_id {
        return false;
    }
    documents
        .iter()
        .find(|document| document.doc_id == relation.target_doc_id)
        .is_some_and(|document| normalize(&document.content) == normalize(&target_claim.evidence))
}

fn push_relation(
    relations: &mut Vec<SemanticRelation>,
    seen: &mut HashSet<String>,
    source: &SemanticClaim,
    target: &SemanticClaim,
    kind: SemanticRelationKind,
    confidence: f32,
    method: &str,
    evidence: Vec<String>,
) {
    let relation_id = format!("{}::{:?}::{}", source.claim_id, kind, target.claim_id);
    if !seen.insert(relation_id.clone()) {
        return;
    }
    relations.push(SemanticRelation {
        relation_id,
        kind,
        source_claim_id: source.claim_id.clone(),
        target_claim_id: target.claim_id.clone(),
        source_doc_id: source.source_doc_id.clone(),
        target_doc_id: target.source_doc_id.clone(),
        confidence,
        method: method.to_string(),
        evidence,
        scope: source.scope.clone(),
    });
}

fn extract_claims(doc: &SourceDocument) -> Vec<SemanticClaim> {
    let mut claims = Vec::new();
    for sentence in sentences(&doc.content) {
        if let Some((subject, predicate, object)) = ownership_claim(sentence) {
            claims.push(make_claim(
                doc,
                sentence,
                &subject,
                predicate,
                &object,
                claims.len(),
            ));
        }
        if let Some((subject, predicate, object)) = assignment_claim(sentence) {
            claims.push(make_claim(
                doc,
                sentence,
                &subject,
                predicate,
                &object,
                claims.len(),
            ));
        }
        if let Some((subject, predicate, object)) = usage_claim(sentence) {
            claims.push(make_claim(
                doc,
                sentence,
                &subject,
                predicate,
                &object,
                claims.len(),
            ));
        }
    }
    dedupe_claims(&mut claims);
    claims
}

fn document_claim(doc: &SourceDocument) -> SemanticClaim {
    make_claim(
        doc,
        &doc.content.chars().take(240).collect::<String>(),
        &doc.concept,
        "document_state",
        &doc.doc_id,
        0,
    )
}

fn make_claim(
    doc: &SourceDocument,
    evidence: &str,
    subject: &str,
    predicate: &str,
    object: &str,
    index: usize,
) -> SemanticClaim {
    let subject = clean_phrase(subject);
    let object = clean_phrase(object);
    SemanticClaim {
        claim_id: format!("{}::claim-{}", doc.doc_id, index),
        subject,
        predicate: predicate.to_string(),
        object,
        source_doc_id: doc.doc_id.clone(),
        evidence: evidence.trim().to_string(),
        effective_at: doc.timestamp.clone(),
        scope: semantic_scope(doc),
        confidence: 0.85,
    }
}

fn ownership_claim(sentence: &str) -> Option<(String, &'static str, String)> {
    static OWNS: OnceLock<Regex> = OnceLock::new();
    static RESPONSIBLE: OnceLock<Regex> = OnceLock::new();
    static OWNED_BY: OnceLock<Regex> = OnceLock::new();
    let owns = OWNS.get_or_init(|| {
        Regex::new(r"(?i)(?:^|[:;])\s*(?:the\s+)?([a-z][a-z0-9 _/-]{1,60}?)\s+owns\s+(?:the\s+)?([a-z][a-z0-9 _/-]{1,80})")
            .expect("valid ownership regex")
    });
    if let Some(caps) = owns.captures(sentence) {
        return Some((caps[2].to_string(), "owner", caps[1].to_string()));
    }
    let responsible = RESPONSIBLE.get_or_init(|| {
        Regex::new(r"(?i)(?:^|[:;])\s*(?:the\s+)?([a-z][a-z0-9 _/-]{1,60}?)\s+is\s+(?:now\s+)?responsible\s+for\s+(?:the\s+)?([a-z][a-z0-9 _/-]{1,80})")
            .expect("valid responsibility regex")
    });
    if let Some(caps) = responsible.captures(sentence) {
        return Some((caps[2].to_string(), "owner", caps[1].to_string()));
    }
    let owned_by = OWNED_BY.get_or_init(|| {
        Regex::new(r"(?i)(?:^|[:;])\s*(?:the\s+)?([a-z][a-z0-9 _/-]{1,80}?)\s+is\s+(?:now\s+)?owned\s+by\s+(?:the\s+)?([a-z][a-z0-9 _/-]{1,60})")
            .expect("valid owned-by regex")
    });
    owned_by
        .captures(sentence)
        .map(|caps| (caps[1].to_string(), "owner", caps[2].to_string()))
}

fn assignment_claim(sentence: &str) -> Option<(String, &'static str, String)> {
    static ASSIGNED: OnceLock<Regex> = OnceLock::new();
    let assigned = ASSIGNED.get_or_init(|| {
        Regex::new(r"(?i)assigned\s+(?:the\s+)?([a-z][a-z0-9 _/-]{1,80}?)\s+to\s+(?:the\s+)?([a-z][a-z0-9 _/-]{1,60})")
            .expect("valid assignment regex")
    });
    assigned
        .captures(sentence)
        .map(|caps| (caps[1].to_string(), "owner", caps[2].to_string()))
}

fn usage_claim(sentence: &str) -> Option<(String, &'static str, String)> {
    static USE_FOR: OnceLock<Regex> = OnceLock::new();
    static USES: OnceLock<Regex> = OnceLock::new();
    let use_for = USE_FOR.get_or_init(|| {
        Regex::new(r"(?i)(?:use|adopt|choose|selected)\s+(?:the\s+)?([a-z][a-z0-9 _/.-]{1,60}?)\s+for\s+(?:the\s+)?([a-z][a-z0-9 _/-]{1,80})")
            .expect("valid use-for regex")
    });
    if let Some(caps) = use_for.captures(sentence) {
        return Some((caps[2].to_string(), "implementation", caps[1].to_string()));
    }
    let uses = USES.get_or_init(|| {
        Regex::new(r"(?i)(?:^|[:;])\s*(?:the\s+)?([a-z][a-z0-9 _/-]{1,80}?)\s+uses\s+(?:the\s+)?([a-z][a-z0-9 _/.-]{1,60})")
            .expect("valid uses regex")
    });
    uses.captures(sentence)
        .map(|caps| (caps[1].to_string(), "implementation", caps[2].to_string()))
}

fn sentences(content: &str) -> Vec<&str> {
    content
        .split(|ch| matches!(ch, '\n' | '.' | '!' | '?'))
        .map(str::trim)
        .filter(|sentence| !sentence.is_empty())
        .collect()
}

fn clean_phrase(value: &str) -> String {
    let value = value
        .trim_matches(|ch: char| ch.is_ascii_punctuation() || ch.is_whitespace())
        .to_lowercase();
    let value = value
        .strip_prefix("the ")
        .or_else(|| value.strip_prefix("current "))
        .or_else(|| value.strip_prefix("new "))
        .unwrap_or(&value);
    value.trim().to_string()
}

fn normalize(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .filter(|token| !matches!(*token, "the" | "a" | "an" | "current" | "new" | "old"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn dedupe_claims(claims: &mut Vec<SemanticClaim>) {
    let mut seen = HashSet::new();
    claims.retain(|claim| {
        !claim.subject.is_empty()
            && !claim.object.is_empty()
            && seen.insert((
                normalize(&claim.subject),
                claim.predicate.clone(),
                normalize(&claim.object),
            ))
    });
}

fn semantic_scope(doc: &SourceDocument) -> String {
    doc.filters
        .get("semantic_scope")
        .map(|scope| format!("scope:{scope}"))
        .or_else(|| {
            doc.filters
                .get("memory_user_id")
                .map(|user| format!("user:{user}"))
        })
        .unwrap_or_else(|| "store".to_string())
}

fn source_kind(doc: &SourceDocument) -> &'static str {
    if doc.source.starts_with("codex://")
        || doc.source.starts_with("claude://")
        || doc.source.starts_with("claude-code://")
        || doc.source.starts_with("gemini-cli://")
        || doc.source.starts_with("agy://")
        || doc.source.starts_with("lint-ai://")
        || doc.filters.contains_key("document_type")
    {
        "memory"
    } else {
        "document"
    }
}

fn has_correction_cue(content: &str) -> bool {
    let lower = content.to_lowercase();
    [
        "supersedes",
        "replaces",
        "instead of",
        "no longer",
        "changed from",
        "moved from",
        "previously",
        "formerly",
    ]
    .iter()
    .any(|cue| lower.contains(cue))
}

fn claim_date(claim: &SemanticClaim) -> Option<chrono::NaiveDate> {
    claim
        .effective_at
        .as_deref()
        .and_then(|value| parse_temporal_date(Some(value)))
}

pub fn is_historical_query(query: &str) -> bool {
    let lower = query.to_lowercase();
    [
        "history",
        "historical",
        "previous",
        "previously",
        "formerly",
        "superseded",
        "what changed",
        "why did we change",
        "what was true before",
        "who owned this before",
        "who was responsible before",
        "before the change",
        "before we changed",
        "timeline",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn doc(id: &str, content: &str, timestamp: Option<&str>, source: &str) -> SourceDocument {
        SourceDocument {
            doc_id: id.to_string(),
            source: source.to_string(),
            content: content.to_string(),
            concept: "decision".to_string(),
            group_id: None,
            headings: vec!["Decision".to_string()],
            links: vec![],
            timestamp: timestamp.map(str::to_string),
            doc_length: content.len(),
            author_agent: None,
            filters: BTreeMap::new(),
        }
    }

    #[test]
    fn newer_same_kind_ownership_claim_supersedes_old_claim() {
        let old = doc(
            "old",
            "The National Park POD owns valve control.",
            Some("2026-01-01"),
            "docs/old.md",
        );
        let new = doc(
            "new",
            "The Controls POD owns valve control.",
            Some("2026-06-01"),
            "docs/new.md",
        );
        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());
        assert_eq!(
            store.document_state("old").status,
            Some(SemanticStatus::Superseded)
        );
        assert_eq!(
            store.document_state("old").superseded_by.as_deref(),
            Some("new")
        );
    }

    #[test]
    fn timestamp_only_cross_source_change_is_a_visible_conflict() {
        let old = doc(
            "old",
            "The National Park POD owns valve control.",
            Some("2026-01-01"),
            "docs/old.md",
        );
        let new = doc(
            "new",
            "The Controls POD owns valve control.",
            Some("2026-06-01"),
            "codex://project/session/outcome",
        );
        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());
        assert_eq!(
            store.document_state("old").status,
            Some(SemanticStatus::Conflicted)
        );
        assert_eq!(
            store.document_state("new").status,
            Some(SemanticStatus::Conflicted)
        );
    }

    #[test]
    fn explicit_supersession_respects_user_scope() {
        let mut old = doc("old", "old decision", None, "memory://old");
        old.filters.insert("memory_user_id".into(), "a".into());
        let mut new = doc("new", "new decision", None, "memory://new");
        new.filters.insert("memory_user_id".into(), "b".into());
        new.filters.insert("supersedes_id".into(), "old".into());
        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());
        assert_ne!(
            store.document_state("old").status,
            Some(SemanticStatus::Superseded)
        );
    }

    #[test]
    fn explicit_supersession_without_extractable_claims_is_authoritative() {
        let old = doc(
            "old",
            "Original architecture decision.",
            None,
            "docs/old.md",
        );
        let mut new = doc("new", "Updated architecture decision.", None, "docs/new.md");
        new.filters.insert("supersedes_id".into(), "old".into());

        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());

        let relation = store
            .relations()
            .iter()
            .find(|relation| relation.kind == SemanticRelationKind::Supersedes)
            .expect("explicit metadata should create a supersession relation");
        assert_eq!(relation.method, "explicit");
        assert_eq!(relation.confidence, 1.0);
        assert_eq!(
            store.document_state("old").superseded_by.as_deref(),
            Some("new")
        );
    }

    #[test]
    fn explicit_document_supersession_still_hides_a_multi_claim_document() {
        let old = doc(
            "old",
            concat!(
                "The Platform POD owns valve control. ",
                "The deployment strategy uses blue-green releases."
            ),
            None,
            "docs/old.md",
        );
        let mut new = doc("new", "Updated architecture decision.", None, "docs/new.md");
        new.filters.insert("supersedes_id".into(), "old".into());

        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());

        assert_eq!(
            store.document_state("old").status,
            Some(SemanticStatus::Superseded)
        );
        assert_eq!(
            store.document_state("old").superseded_by.as_deref(),
            Some("new")
        );
    }

    #[test]
    fn correction_language_can_supersede_without_timestamps() {
        let old = doc(
            "a-old",
            "The National Park POD owns valve control.",
            None,
            "docs/old.md",
        );
        let new = doc(
            "z-new",
            "This replaces the previous decision. The Controls POD owns valve control.",
            None,
            "docs/new.md",
        );

        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());

        assert_eq!(
            store.document_state("a-old").status,
            Some(SemanticStatus::Superseded)
        );
        assert!(store.relations().iter().any(|relation| {
            relation.kind == SemanticRelationKind::Supersedes && relation.method == "correction_cue"
        }));
    }

    #[test]
    fn changed_claim_without_time_or_correction_stays_visible_as_conflict() {
        let old = doc(
            "a-old",
            "The National Park POD owns valve control.",
            None,
            "docs/old.md",
        );
        let new = doc(
            "z-new",
            "The Controls POD owns valve control.",
            None,
            "docs/new.md",
        );

        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());

        assert_eq!(
            store.document_state("a-old").status,
            Some(SemanticStatus::Conflicted)
        );
        assert_eq!(
            store.document_state("z-new").status,
            Some(SemanticStatus::Conflicted)
        );
        assert!(store
            .relations()
            .iter()
            .all(|relation| relation.kind != SemanticRelationKind::Supersedes));
    }

    #[test]
    fn equivalent_wording_confirms_the_existing_claim() {
        let first = doc(
            "first",
            "The Controls POD owns valve control.",
            Some("2026-01-01"),
            "docs/first.md",
        );
        let second = doc(
            "second",
            "Valve control is owned by the Controls POD.",
            Some("2026-02-01"),
            "docs/second.md",
        );

        let store = SemanticRelationStore::from_documents(
            [&first, &second],
            SupersessionOptions::default(),
        );

        assert!(store
            .relations()
            .iter()
            .any(|relation| relation.kind == SemanticRelationKind::Confirms));
        assert_eq!(
            store.document_state("first").status,
            Some(SemanticStatus::Current)
        );
        assert_eq!(
            store.document_state("second").status,
            Some(SemanticStatus::Current)
        );
    }

    #[test]
    fn semantic_scope_prevents_automatic_cross_project_relations() {
        let mut old = doc(
            "old",
            "The National Park POD owns valve control.",
            Some("2026-01-01"),
            "docs/old.md",
        );
        old.filters
            .insert("semantic_scope".into(), "project-a".into());
        let mut new = doc(
            "new",
            "The Controls POD owns valve control.",
            Some("2026-06-01"),
            "docs/new.md",
        );
        new.filters
            .insert("semantic_scope".into(), "project-b".into());

        let store =
            SemanticRelationStore::from_documents([&old, &new], SupersessionOptions::default());

        assert!(store.relations().is_empty());
        assert_eq!(
            store.document_state("old").status,
            Some(SemanticStatus::Current)
        );
        assert_eq!(
            store.document_state("new").status,
            Some(SemanticStatus::Current)
        );
    }

    #[test]
    fn disabled_supersession_produces_no_claims_relations_or_states() {
        let old = doc(
            "old",
            "The National Park POD owns valve control.",
            Some("2026-01-01"),
            "docs/old.md",
        );
        let new = doc(
            "new",
            "The Controls POD owns valve control.",
            Some("2026-06-01"),
            "docs/new.md",
        );
        let options = SupersessionOptions {
            enabled: false,
            suppress_confidence: 1.2,
            conflict_confidence: -0.1,
        };

        let store = SemanticRelationStore::from_documents([&old, &new], options);

        assert!(store.claims().is_empty());
        assert!(store.relations().is_empty());
        assert_eq!(
            store.document_state("old"),
            DocumentSemanticState::default()
        );
    }

    #[test]
    fn enabled_supersession_rejects_invalid_thresholds() {
        let document = doc(
            "decision",
            "The Controls POD owns valve control.",
            None,
            "doc.md",
        );
        let options = SupersessionOptions {
            suppress_confidence: 1.2,
            ..SupersessionOptions::default()
        };

        let error = SemanticRelationStore::try_from_documents([&document], options)
            .expect_err("enabled supersession must reject invalid thresholds");

        assert!(error
            .to_string()
            .contains("confidence thresholds must be between 0 and 1"));
    }

    #[test]
    fn suppression_threshold_can_keep_detected_relation_visible() {
        let old = doc(
            "old",
            "The National Park POD owns valve control.",
            Some("2026-01-01"),
            "docs/old.md",
        );
        let new = doc(
            "new",
            "The Controls POD owns valve control.",
            Some("2026-06-01"),
            "docs/new.md",
        );
        let options = SupersessionOptions {
            suppress_confidence: 0.95,
            ..SupersessionOptions::default()
        };

        let store = SemanticRelationStore::from_documents([&old, &new], options);

        assert!(store
            .relations()
            .iter()
            .any(|relation| relation.kind == SemanticRelationKind::Supersedes));
        assert_eq!(
            store.document_state("old").status,
            Some(SemanticStatus::Current)
        );
    }

    #[test]
    fn historical_query_detection_covers_history_but_not_current_state() {
        for query in [
            "show the decision history",
            "what changed about ownership",
            "who owned this before",
            "show the superseded guidance",
            "ownership timeline",
        ] {
            assert!(
                is_historical_query(query),
                "expected historical query: {query}"
            );
        }
        for query in [
            "who owns valve control now",
            "what must happen before deployment",
            "validate inputs before saving",
            "check authorization before updating status",
        ] {
            assert!(
                !is_historical_query(query),
                "expected current-state query: {query}"
            );
        }
    }
}
