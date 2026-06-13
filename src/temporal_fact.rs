use crate::index::{Claim, DocRecord};
use crate::pipeline::ChunkLifecycleMeta;
use crate::temporal::parse_temporal_date;
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimelineEvent<'a> {
    pub fact: &'a TemporalFact,
    pub date: NaiveDate,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimelinePair<'a> {
    pub first: TimelineEvent<'a>,
    pub second: TimelineEvent<'a>,
    pub gap_days: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TemporalFact {
    pub fact_id: String,
    pub subject: String,
    pub predicate: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_to: Option<String>,
    pub source_doc_id: String,
    pub source_chunk_id: String,
    pub source_chunk_version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_timestamp: Option<String>,
    pub confidence: f32,
    pub is_latest: bool,
}

impl TemporalFact {
    pub fn normalized_subject(&self) -> String {
        normalize_component(&self.subject)
    }

    pub fn normalized_predicate(&self) -> String {
        normalize_component(&self.predicate)
    }

    pub fn normalized_scope(&self) -> Option<String> {
        self.scope.as_deref().map(normalize_component)
    }

    fn semantic_key(&self) -> String {
        let scope = self.normalized_scope().unwrap_or_default();
        format!(
            "{}::{}::{}",
            self.normalized_subject(),
            self.normalized_predicate(),
            scope
        )
    }

    fn semantic_value_key(&self) -> String {
        let object = self.object.as_deref().unwrap_or_default();
        let value = self.value.as_deref().unwrap_or_default();
        let unit = self.unit.as_deref().unwrap_or_default();
        format!(
            "{}::{}::{}",
            normalize_component(object),
            normalize_component(value),
            normalize_component(unit)
        )
    }

    fn valid_from_date(&self) -> Option<NaiveDate> {
        self.valid_from
            .as_deref()
            .and_then(|value| parse_temporal_date(Some(value)))
            .or_else(|| {
                self.chunk_timestamp
                    .as_deref()
                    .and_then(|value| parse_temporal_date(Some(value)))
            })
    }

    fn valid_to_date(&self) -> Option<NaiveDate> {
        self.valid_to
            .as_deref()
            .and_then(|value| parse_temporal_date(Some(value)))
    }

    fn active_at(&self, date: NaiveDate) -> bool {
        let Some(start) = self.valid_from_date() else {
            return false;
        };
        if date < start {
            return false;
        }
        if let Some(end) = self.valid_to_date() {
            date < end
        } else {
            true
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemporalFactStore {
    facts: Vec<TemporalFact>,
    #[serde(skip)]
    latest_by_key: HashMap<String, usize>,
}

impl TemporalFactStore {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn from_records<'a, I>(
        records: I,
        chunk_lifecycle: &HashMap<String, ChunkLifecycleMeta>,
    ) -> Self
    where
        I: IntoIterator<Item = &'a DocRecord>,
    {
        let mut ordered = records.into_iter().collect::<Vec<_>>();
        ordered.sort_by(|a, b| {
            let a_date = a
                .timestamp
                .as_deref()
                .and_then(|value| parse_temporal_date(Some(value)));
            let b_date = b
                .timestamp
                .as_deref()
                .and_then(|value| parse_temporal_date(Some(value)));
            a_date.cmp(&b_date).then_with(|| a.doc_id.cmp(&b.doc_id))
        });

        let mut store = Self::default();
        for record in ordered {
            let (source_chunk_id, source_chunk_version, chunk_timestamp) =
                best_source_chunk(record, chunk_lifecycle);
            for claim in &record.top_claims {
                store.ingest_claim(
                    record,
                    &claim,
                    source_chunk_id.as_deref(),
                    source_chunk_version,
                    chunk_timestamp.as_deref(),
                );
            }
        }
        store
    }

    pub fn facts(&self) -> &[TemporalFact] {
        &self.facts
    }

    pub fn ingest_claim(
        &mut self,
        record: &DocRecord,
        claim: &Claim,
        source_chunk_id: Option<&str>,
        source_chunk_version: u32,
        chunk_timestamp: Option<&str>,
    ) {
        let fact = TemporalFact {
            fact_id: make_fact_id(
                &record.doc_id,
                source_chunk_id.unwrap_or(&record.doc_id),
                &claim.subject,
                &claim.predicate,
                claim.object.as_str(),
                record.group_id.as_deref(),
            ),
            subject: claim.subject.trim().to_string(),
            predicate: claim.predicate.trim().to_string(),
            object: (!claim.object.trim().is_empty()).then(|| claim.object.trim().to_string()),
            value: None,
            unit: None,
            scope: record.group_id.clone(),
            valid_from: chunk_timestamp
                .map(str::to_string)
                .or_else(|| record.timestamp.clone()),
            valid_to: None,
            source_doc_id: record.doc_id.clone(),
            source_chunk_id: source_chunk_id.unwrap_or(&record.doc_id).to_string(),
            source_chunk_version,
            chunk_timestamp: chunk_timestamp
                .map(str::to_string)
                .or_else(|| record.timestamp.clone()),
            confidence: claim.confidence,
            is_latest: true,
        };
        self.ingest_fact(fact);
    }

    pub fn ingest_fact(&mut self, mut fact: TemporalFact) {
        fact.subject = fact.subject.trim().to_string();
        fact.predicate = fact.predicate.trim().to_string();
        if let Some(object) = fact.object.as_mut() {
            *object = object.trim().to_string();
            if object.is_empty() {
                *object = String::new();
            }
        }
        if let Some(value) = fact.value.as_mut() {
            *value = value.trim().to_string();
            if value.is_empty() {
                *value = String::new();
            }
        }
        if let Some(unit) = fact.unit.as_mut() {
            *unit = unit.trim().to_string();
            if unit.is_empty() {
                *unit = String::new();
            }
        }

        let semantic_key = fact.semantic_key();
        let semantic_value_key = fact.semantic_value_key();
        if let Some(&idx) = self.latest_by_key.get(&semantic_key) {
            let current = &mut self.facts[idx];
            if current.semantic_value_key() == semantic_value_key {
                if fact
                    .valid_from_date()
                    .zip(current.valid_from_date())
                    .map(|(new_date, old_date)| new_date >= old_date)
                    .unwrap_or(true)
                {
                    *current = fact;
                    current.is_latest = true;
                }
                self.latest_by_key.insert(semantic_key, idx);
                return;
            }

            if current.valid_to.is_none() {
                current.valid_to = fact
                    .valid_from
                    .clone()
                    .or_else(|| fact.chunk_timestamp.clone());
            }
            current.is_latest = false;
        }

        let idx = self.facts.len();
        self.latest_by_key.insert(semantic_key, idx);
        self.facts.push(fact);
    }

    pub fn as_of(&self, date: &str) -> Vec<&TemporalFact> {
        let Some(target) = parse_temporal_date(Some(date)) else {
            return Vec::new();
        };
        let mut facts = self
            .facts
            .iter()
            .filter(|fact| fact.active_at(target))
            .collect::<Vec<_>>();
        facts.sort_by(|a, b| {
            a.normalized_subject()
                .cmp(&b.normalized_subject())
                .then_with(|| a.normalized_predicate().cmp(&b.normalized_predicate()))
                .then_with(|| a.source_doc_id.cmp(&b.source_doc_id))
        });
        facts
    }

    pub fn timeline_events_between(&self, start: &str, end: &str) -> Vec<TimelineEvent<'_>> {
        let Some((start, end)) = normalize_range(start, end) else {
            return Vec::new();
        };
        let mut events = self
            .facts
            .iter()
            .filter_map(|fact| {
                let date = fact.valid_from_date()?;
                if date < start || date > end {
                    return None;
                }
                Some(TimelineEvent { fact, date })
            })
            .collect::<Vec<_>>();
        events.sort_by(|a, b| {
            a.date
                .cmp(&b.date)
                .then_with(|| {
                    a.fact
                        .normalized_subject()
                        .cmp(&b.fact.normalized_subject())
                })
                .then_with(|| {
                    a.fact
                        .normalized_predicate()
                        .cmp(&b.fact.normalized_predicate())
                })
                .then_with(|| a.fact.source_doc_id.cmp(&b.fact.source_doc_id))
                .then_with(|| a.fact.source_chunk_id.cmp(&b.fact.source_chunk_id))
        });
        events
    }

    pub fn timeline_window_around(
        &self,
        anchor: &str,
        before: usize,
        after: usize,
    ) -> Vec<TimelineEvent<'_>> {
        let Some(anchor_date) = parse_temporal_date(Some(anchor)) else {
            return Vec::new();
        };
        let mut events = self
            .facts
            .iter()
            .filter_map(|fact| {
                let date = fact.valid_from_date()?;
                Some(TimelineEvent { fact, date })
            })
            .collect::<Vec<_>>();
        events.sort_by(|a, b| {
            a.date
                .cmp(&b.date)
                .then_with(|| {
                    a.fact
                        .normalized_subject()
                        .cmp(&b.fact.normalized_subject())
                })
                .then_with(|| {
                    a.fact
                        .normalized_predicate()
                        .cmp(&b.fact.normalized_predicate())
                })
                .then_with(|| a.fact.source_doc_id.cmp(&b.fact.source_doc_id))
                .then_with(|| a.fact.source_chunk_id.cmp(&b.fact.source_chunk_id))
        });
        if events.is_empty() {
            return Vec::new();
        }

        let mut anchor_idx = 0usize;
        let mut min_dist = i64::MAX;
        for (idx, event) in events.iter().enumerate() {
            let dist = (event.date - anchor_date).num_days().abs();
            if dist < min_dist {
                min_dist = dist;
                anchor_idx = idx;
            }
        }

        let start_idx = anchor_idx.saturating_sub(before);
        let end_idx = usize::min(events.len().saturating_sub(1), anchor_idx + after);
        events[start_idx..=end_idx].to_vec()
    }

    pub fn adjacent_pairs_between(
        &self,
        start: &str,
        end: &str,
        max_gap_days: Option<i64>,
    ) -> Vec<TimelinePair<'_>> {
        let events = self.timeline_events_between(start, end);
        if events.len() < 2 {
            return Vec::new();
        }
        let mut pairs = Vec::new();
        for window in events.windows(2) {
            let first = window[0];
            let second = window[1];
            let gap_days = (second.date - first.date).num_days();
            if gap_days < 0 {
                continue;
            }
            if let Some(limit) = max_gap_days {
                if gap_days > limit {
                    continue;
                }
            }
            pairs.push(TimelinePair {
                first,
                second,
                gap_days,
            });
        }
        pairs
    }

    pub fn timeline(&self, subject: &str) -> Vec<&TemporalFact> {
        let needle = normalize_component(subject);
        let mut facts = self
            .facts
            .iter()
            .filter(|fact| fact.normalized_subject() == needle)
            .collect::<Vec<_>>();
        facts.sort_by(|a, b| {
            a.valid_from_date()
                .cmp(&b.valid_from_date())
                .then_with(|| a.valid_to_date().cmp(&b.valid_to_date()))
                .then_with(|| a.normalized_predicate().cmp(&b.normalized_predicate()))
        });
        facts
    }
}

fn normalize_range(start: &str, end: &str) -> Option<(NaiveDate, NaiveDate)> {
    let start = parse_temporal_date(Some(start))?;
    let end = parse_temporal_date(Some(end))?;
    if start <= end {
        Some((start, end))
    } else {
        Some((end, start))
    }
}

fn normalize_component(input: &str) -> String {
    input
        .split_whitespace()
        .map(|part| part.trim().to_lowercase())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn make_fact_id(
    source_doc_id: &str,
    source_chunk_id: &str,
    subject: &str,
    predicate: &str,
    object: &str,
    scope: Option<&str>,
) -> String {
    let scope = scope.unwrap_or_default();
    format!(
        "{}::{}::{}::{}::{}::{}",
        sanitize_id_component(source_doc_id),
        sanitize_id_component(source_chunk_id),
        sanitize_id_component(subject),
        sanitize_id_component(predicate),
        sanitize_id_component(object),
        sanitize_id_component(scope)
    )
}

fn sanitize_id_component(input: &str) -> String {
    let normalized = normalize_component(input);
    if normalized.is_empty() {
        "_".to_string()
    } else {
        normalized.replace("::", "_")
    }
}

fn best_source_chunk(
    record: &DocRecord,
    chunk_lifecycle: &HashMap<String, ChunkLifecycleMeta>,
) -> (Option<String>, u32, Option<String>) {
    let chosen = record
        .section_chunks
        .iter()
        .find(|chunk| chunk.timestamp.is_some())
        .or_else(|| record.section_chunks.first());
    let Some(chunk) = chosen else {
        return (None, 1, record.timestamp.clone());
    };
    let version = chunk_lifecycle
        .get(&chunk.chunk_id)
        .map(|meta| meta.version)
        .unwrap_or(1);
    let timestamp = chunk.timestamp.clone().or_else(|| record.timestamp.clone());
    (Some(chunk.chunk_id.clone()), version, timestamp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{Claim, Provenance, SectionChunk};
    use crate::tier1::{RankedTerm, Tier1Entity};

    fn sample_record(doc_id: &str, timestamp: Option<&str>, claims: Vec<Claim>) -> DocRecord {
        DocRecord {
            doc_id: doc_id.to_string(),
            source: format!("source://{}", doc_id),
            content: "dummy".to_string(),
            timestamp: timestamp.map(|v| v.to_string()),
            doc_length: 5,
            author_agent: None,
            group_id: Some("group-a".to_string()),
            filters: std::collections::BTreeMap::new(),
            probable_topic: Some("topic".to_string()),
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
                term: "alice".to_string(),
                score: 1.0,
                source: "yake".to_string(),
            }],
            section_chunks: vec![SectionChunk {
                chunk_id: format!("{}::chunk-0", doc_id),
                heading: "Overview".to_string(),
                content: "Alice works at Acme".to_string(),
                start_line: 1,
                end_line: 1,
                timestamp: timestamp.map(|v| v.to_string()),
                key_entities: vec!["Alice".to_string()],
                important_terms: vec!["alice".to_string()],
            }],
            embedding: None,
            top_claims: claims,
            provenance: Provenance {
                source: "source://doc".to_string(),
                timestamp: timestamp.map(|v| v.to_string()),
                ner_provider: "heuristic".to_string(),
                term_ranker: "yake".to_string(),
                index_version: "v1".to_string(),
            },
        }
    }

    #[test]
    fn store_versions_conflicting_facts_without_hardcoded_patterns() {
        let mut store = TemporalFactStore::default();
        let record1 = sample_record(
            "doc-1",
            Some("2024-01-01"),
            vec![Claim {
                subject: "Alice".to_string(),
                predicate: "works_at".to_string(),
                object: "Acme".to_string(),
                confidence: 0.9,
            }],
        );
        let record2 = sample_record(
            "doc-2",
            Some("2024-02-01"),
            vec![Claim {
                subject: "Alice".to_string(),
                predicate: "works_at".to_string(),
                object: "Beta".to_string(),
                confidence: 0.95,
            }],
        );

        let chunk_lifecycle = HashMap::new();
        let built = TemporalFactStore::from_records([&record1, &record2], &chunk_lifecycle);
        assert_eq!(built.facts.len(), 2);
        let timeline = built.timeline("alice");
        assert_eq!(timeline.len(), 2);
        assert_eq!(timeline[0].object.as_deref(), Some("Acme"));
        assert_eq!(timeline[0].is_latest, false);
        assert_eq!(timeline[0].valid_to.as_deref(), Some("2024-02-01"));
        assert_eq!(timeline[1].object.as_deref(), Some("Beta"));
        assert_eq!(timeline[1].is_latest, true);

        store.ingest_claim(
            &record1,
            &Claim {
                subject: "Alice".to_string(),
                predicate: "works_at".to_string(),
                object: "Acme".to_string(),
                confidence: 0.9,
            },
            Some("doc-1::chunk-0"),
            1,
            Some("2024-01-01"),
        );
        assert_eq!(store.facts.len(), 1);
    }

    #[test]
    fn as_of_filters_by_validity_window() {
        let record1 = sample_record(
            "doc-1",
            Some("2024-01-01"),
            vec![Claim {
                subject: "Alice".to_string(),
                predicate: "lives_in".to_string(),
                object: "Seattle".to_string(),
                confidence: 0.8,
            }],
        );
        let record2 = sample_record(
            "doc-2",
            Some("2024-03-01"),
            vec![Claim {
                subject: "Alice".to_string(),
                predicate: "lives_in".to_string(),
                object: "Portland".to_string(),
                confidence: 0.8,
            }],
        );
        let store = TemporalFactStore::from_records([&record1, &record2], &HashMap::new());
        let jan = store.as_of("2024-01-15");
        assert_eq!(jan.len(), 1);
        assert_eq!(jan[0].object.as_deref(), Some("Seattle"));
        let mar = store.as_of("2024-03-15");
        assert_eq!(mar.len(), 1);
        assert_eq!(mar[0].object.as_deref(), Some("Portland"));
    }

    #[test]
    fn timeline_window_and_adjacent_pairs_are_ordered() {
        let record1 = sample_record(
            "doc-1",
            Some("2024-01-01"),
            vec![Claim {
                subject: "Alice".to_string(),
                predicate: "attended".to_string(),
                object: "Charity Walk".to_string(),
                confidence: 0.8,
            }],
        );
        let record2 = sample_record(
            "doc-2",
            Some("2024-01-02"),
            vec![Claim {
                subject: "Alice".to_string(),
                predicate: "attended".to_string(),
                object: "Charity Bike Ride".to_string(),
                confidence: 0.8,
            }],
        );
        let record3 = sample_record(
            "doc-3",
            Some("2024-01-10"),
            vec![Claim {
                subject: "Alice".to_string(),
                predicate: "attended".to_string(),
                object: "Book Drive".to_string(),
                confidence: 0.8,
            }],
        );
        let store =
            TemporalFactStore::from_records([&record1, &record2, &record3], &HashMap::new());
        let window = store.timeline_events_between("2024-01-01", "2024-01-03");
        assert_eq!(window.len(), 2);
        assert_eq!(window[0].fact.object.as_deref(), Some("Charity Walk"));
        assert_eq!(window[1].fact.object.as_deref(), Some("Charity Bike Ride"));

        let pairs = store.adjacent_pairs_between("2024-01-01", "2024-01-10", Some(2));
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].first.fact.object.as_deref(), Some("Charity Walk"));
        assert_eq!(
            pairs[0].second.fact.object.as_deref(),
            Some("Charity Bike Ride")
        );
        assert_eq!(pairs[0].gap_days, 1);
    }
}
