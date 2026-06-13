use crate::index::{Claim, DocRecord, SectionChunk};

#[derive(Debug, Clone, Default)]
pub struct ExtractedClaims {
    pub claims: Vec<Claim>,
}

pub trait ClaimExtractor {
    fn extract(&self, record: &DocRecord) -> ExtractedClaims;
    fn name(&self) -> &'static str;
}

#[derive(Debug, Clone, Default)]
pub struct ConservativeClaimExtractor;

impl ConservativeClaimExtractor {
    fn from_existing_claims(record: &DocRecord, out: &mut Vec<Claim>) {
        out.extend(
            record
                .top_claims
                .iter()
                .filter(|claim| is_valid_claim(claim))
                .cloned(),
        );
    }

    fn from_chunks(record: &DocRecord, out: &mut Vec<Claim>) {
        for chunk in &record.section_chunks {
            out.extend(chunk_claims(record, chunk));
        }
    }
}

impl ClaimExtractor for ConservativeClaimExtractor {
    fn extract(&self, record: &DocRecord) -> ExtractedClaims {
        let mut claims = Vec::new();
        Self::from_existing_claims(record, &mut claims);
        Self::from_chunks(record, &mut claims);
        dedupe_claims(&mut claims);
        ExtractedClaims { claims }
    }

    fn name(&self) -> &'static str {
        "conservative"
    }
}

fn chunk_claims(record: &DocRecord, chunk: &SectionChunk) -> Vec<Claim> {
    let mut out = Vec::new();

    if let Some(topic) = record
        .probable_topic
        .as_ref()
        .filter(|s| !s.trim().is_empty())
    {
        if chunk.heading.to_lowercase().contains(&topic.to_lowercase())
            || chunk.content.to_lowercase().contains(&topic.to_lowercase())
        {
            out.push(Claim {
                subject: topic.clone(),
                predicate: "mentions".to_string(),
                object: chunk.heading.clone(),
                confidence: 0.55,
            });
        }
    }

    if let Some(ts) = chunk.timestamp.as_ref().or(record.timestamp.as_ref()) {
        if !ts.trim().is_empty() {
            out.push(Claim {
                subject: chunk.heading.clone(),
                predicate: "timestamp".to_string(),
                object: ts.clone(),
                confidence: 0.5,
            });
        }
    }

    for ent in &chunk.key_entities {
        out.push(Claim {
            subject: chunk.heading.clone(),
            predicate: "mentions_entity".to_string(),
            object: ent.clone(),
            confidence: 0.45,
        });
    }

    for term in &chunk.important_terms {
        out.push(Claim {
            subject: chunk.heading.clone(),
            predicate: "important_term".to_string(),
            object: term.clone(),
            confidence: 0.4,
        });
    }

    out
}

fn is_valid_claim(claim: &Claim) -> bool {
    !claim.subject.trim().is_empty()
        && !claim.predicate.trim().is_empty()
        && !claim.object.trim().is_empty()
        && claim.confidence.is_finite()
}

fn dedupe_claims(claims: &mut Vec<Claim>) {
    claims.retain(is_valid_claim);
    claims.sort_by(|a, b| {
        a.subject
            .cmp(&b.subject)
            .then_with(|| a.predicate.cmp(&b.predicate))
            .then_with(|| a.object.cmp(&b.object))
            .then_with(|| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    claims.dedup_by(|a, b| {
        a.subject == b.subject && a.predicate == b.predicate && a.object == b.object
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{Claim, Provenance, SectionChunk};
    use crate::tier1::{RankedTerm, Tier1Entity};

    fn sample_record() -> DocRecord {
        DocRecord {
            doc_id: "doc-1".to_string(),
            source: "source://doc-1".to_string(),
            content: "Alice works at Acme".to_string(),
            timestamp: Some("2024-05-01".to_string()),
            doc_length: 20,
            author_agent: None,
            group_id: Some("group-a".to_string()),
            filters: std::collections::BTreeMap::new(),
            probable_topic: Some("Alice".to_string()),
            doc_type_guess: Some("note".to_string()),
            headings: vec!["Overview".to_string()],
            doc_links: vec![],
            temporal_terms: vec![],
            key_entities: vec![Tier1Entity {
                text: "Alice".to_string(),
                label: "PROPN".to_string(),
                start: 0,
                end: 5,
                score: Some(1.0),
                source: "heuristic".to_string(),
            }],
            important_terms: vec![RankedTerm {
                term: "acme".to_string(),
                score: 1.0,
                source: "yake".to_string(),
            }],
            section_chunks: vec![SectionChunk {
                chunk_id: "doc-1::chunk-0".to_string(),
                heading: "Overview".to_string(),
                content: "Alice works at Acme".to_string(),
                start_line: 1,
                end_line: 1,
                timestamp: Some("2024-05-01".to_string()),
                key_entities: vec!["Alice".to_string()],
                important_terms: vec!["acme".to_string()],
            }],
            embedding: None,
            top_claims: vec![Claim {
                subject: "Alice".to_string(),
                predicate: "works_at".to_string(),
                object: "Acme".to_string(),
                confidence: 0.9,
            }],
            provenance: Provenance {
                source: "source://doc-1".to_string(),
                timestamp: Some("2024-05-01".to_string()),
                ner_provider: "heuristic".to_string(),
                term_ranker: "yake".to_string(),
                index_version: "v1".to_string(),
            },
        }
    }

    #[test]
    fn conservative_extractor_uses_existing_claims_and_chunk_signals() {
        let extractor = ConservativeClaimExtractor;
        let record = sample_record();
        let extracted = extractor.extract(&record);
        assert!(extracted
            .claims
            .iter()
            .any(|c| c.predicate == "works_at" && c.object == "Acme"));
        assert!(extracted
            .claims
            .iter()
            .any(|c| c.predicate == "mentions_entity" && c.object == "Alice"));
        assert!(extracted
            .claims
            .iter()
            .any(|c| c.predicate == "important_term" && c.object == "acme"));
        assert!(extracted
            .claims
            .iter()
            .any(|c| c.predicate == "timestamp" && c.object == "2024-05-01"));
    }
}
