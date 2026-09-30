//! False-positive (over-suppression) audit for semantic supersession.
//!
//! The supersession probes check that stale values are retired (no leakage).
//! This module checks the other direction: true facts must not be wrongly
//! retired. Each case writes documents through the public
//! [`MemoryService::add`] API and asserts on the resulting document states.
//!
//! A case passes when every protected write's document is *not* Superseded
//! (it may be Current or Conflicted — conflict is the conservative correct
//! outcome, not over-suppression) and every expected-superseded write's
//! document *is* Superseded (true-positive controls keep the harness honest).

use crate::memory_api::{AddRequest, MemoryService, Message};
use crate::semantic_relations::SemanticStatus;
use crate::stable_doc_id_from_source;

/// One write in an audit case.
pub struct AuditWrite {
    pub session: &'static str,
    pub content: &'static str,
    pub timestamp_ms: Option<i64>,
}

/// A single over-suppression scenario.
pub struct AuditCase {
    pub name: &'static str,
    pub description: &'static str,
    pub writes: Vec<AuditWrite>,
    /// Indices into `writes` whose documents must NOT end up Superseded.
    pub must_stay_current: Vec<usize>,
    /// Indices into `writes` whose documents MUST end up Superseded.
    /// True-positive controls: if the harness cannot detect real
    /// supersession, it is vacuous.
    pub must_be_superseded: Vec<usize>,
}

pub struct AuditResult {
    pub name: &'static str,
    pub passed: bool,
    pub failures: Vec<String>,
}

const T1: i64 = 1_699_939_200_000; // 2023-11-14
const T2: i64 = 1_700_025_600_000; // 2023-11-15

/// All over-suppression audit cases.
pub fn all_cases() -> Vec<AuditCase> {
    vec![
        AuditCase {
            name: "different_subjects_coexist",
            description: "Different entities share a predicate; neither may retire the other.",
            writes: vec![
                AuditWrite { session: "s1", content: "The bicycle is owned by Rossi.", timestamp_ms: Some(T1) },
                AuditWrite { session: "s2", content: "The car is owned by Rossi.", timestamp_ms: Some(T2) },
            ],
            must_stay_current: vec![0, 1],
            must_be_superseded: vec![],
        },
        AuditCase {
            name: "dateless_conflict_preserved",
            description: "Same subject, no timestamps: conflict must not silently retire either fact.",
            writes: vec![
                AuditWrite { session: "s1", content: "The bicycle is owned by Rossi.", timestamp_ms: None },
                AuditWrite { session: "s2", content: "The bicycle is owned by Bianchi.", timestamp_ms: None },
            ],
            must_stay_current: vec![0, 1],
            must_be_superseded: vec![],
        },
        AuditCase {
            name: "cross_session_config_independent",
            description: "Same config key in different sessions are different authorities.",
            writes: vec![
                AuditWrite { session: "s1", content: "timeout: 100", timestamp_ms: Some(T1) },
                AuditWrite { session: "s2", content: "timeout: 150", timestamp_ms: Some(T2) },
            ],
            must_stay_current: vec![0, 1],
            must_be_superseded: vec![],
        },
        AuditCase {
            name: "reaffirmation_keeps_both_current",
            description: "Re-asserting the same value confirms it; the earlier doc stays live.",
            writes: vec![
                AuditWrite { session: "s1", content: "The bicycle is owned by Rossi.", timestamp_ms: Some(T1) },
                AuditWrite { session: "s2", content: "The bicycle is owned by Rossi.", timestamp_ms: Some(T2) },
            ],
            must_stay_current: vec![0, 1],
            must_be_superseded: vec![],
        },
        AuditCase {
            name: "backfill_preserves_newer",
            description: "Writing an older value later must not retire the current value.",
            writes: vec![
                AuditWrite { session: "s1", content: "version: 2.0", timestamp_ms: Some(T2) },
                AuditWrite { session: "s1", content: "version: 1.5", timestamp_ms: Some(T1) },
            ],
            must_stay_current: vec![0],
            must_be_superseded: vec![1],
        },
        AuditCase {
            name: "unrelated_predicate_untouched",
            description: "A claim about one predicate must not retire a claim about another.",
            writes: vec![
                AuditWrite { session: "s1", content: "The bicycle is red.", timestamp_ms: Some(T1) },
                AuditWrite { session: "s2", content: "The bicycle is owned by Bianchi.", timestamp_ms: Some(T2) },
            ],
            must_stay_current: vec![0, 1],
            must_be_superseded: vec![],
        },
        AuditCase {
            name: "chronological_supersession_fires",
            description: "Control: same subject, strictly newer date DOES retire the old fact.",
            writes: vec![
                AuditWrite { session: "s1", content: "The bicycle is owned by Rossi.", timestamp_ms: Some(T1) },
                AuditWrite { session: "s2", content: "The bicycle is owned by Bianchi.", timestamp_ms: Some(T2) },
            ],
            must_stay_current: vec![1],
            must_be_superseded: vec![0],
        },
        AuditCase {
            name: "correction_cue_supersedes",
            description: "Control: an explicit correction cue retires the corrected fact.",
            writes: vec![
                AuditWrite { session: "s1", content: "We use Postgres for analytics.", timestamp_ms: Some(T1) },
                AuditWrite { session: "s2", content: "We use MongoDB for analytics instead of Postgres.", timestamp_ms: Some(T1) },
            ],
            must_stay_current: vec![1],
            must_be_superseded: vec![0],
        },
    ]
}

/// Runs one audit case against a fresh in-memory service.
pub fn run_case(case: &AuditCase, case_idx: usize) -> AuditResult {
    let mut service =
        MemoryService::in_memory(crate::default_production_pipeline_options());
    let mut doc_ids = Vec::with_capacity(case.writes.len());
    for (write_idx, write) in case.writes.iter().enumerate() {
        let request_id = format!("audit-{case_idx}-{write_idx}");
        service
            .add(AddRequest {
                request_id: request_id.clone(),
                messages: vec![Message {
                    role: "user".into(),
                    timestamp: write.timestamp_ms,
                    content: write.content.into(),
                    expires_at_ms: None,
                    supersedes_id: None,
                }],
                user_id: "audit-user".into(),
                session_id: write.session.into(),
            })
            .expect("audit add failed");
        doc_ids.push(stable_doc_id_from_source(&format!(
            "audit-user:{request_id}:0"
        )));
    }

    let mut failures = Vec::new();
    for &idx in &case.must_stay_current {
        let state = service.semantic_document_state(&doc_ids[idx]);
        if state.status == Some(SemanticStatus::Superseded) {
            failures.push(format!(
                "write {idx} ('{}') was wrongly superseded by {:?}",
                case.writes[idx].content, state.superseded_by,
            ));
        }
    }
    for &idx in &case.must_be_superseded {
        let state = service.semantic_document_state(&doc_ids[idx]);
        if state.status != Some(SemanticStatus::Superseded) {
            failures.push(format!(
                "write {idx} ('{}') should have been superseded but status is {:?}",
                case.writes[idx].content, state.status,
            ));
        }
    }
    AuditResult {
        name: case.name,
        passed: failures.is_empty(),
        failures,
    }
}

/// Runs all audit cases, each against a fresh in-memory service.
pub fn run_all() -> Vec<AuditResult> {
    all_cases()
        .iter()
        .enumerate()
        .map(|(idx, case)| run_case(case, idx))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_true_fact_is_wrongly_retired() {
        let results = run_all();
        let failed: Vec<&AuditResult> =
            results.iter().filter(|r| !r.passed).collect();
        for result in &failed {
            for failure in &result.failures {
                eprintln!("AUDIT FAIL [{}]: {}", result.name, failure);
            }
        }
        assert!(
            failed.is_empty(),
            "{} of {} audit cases failed",
            failed.len(),
            results.len()
        );
    }

    #[test]
    fn audit_covers_both_directions() {
        // The harness is vacuous if no case expects supersession.
        let cases = all_cases();
        assert!(cases.iter().any(|c| !c.must_be_superseded.is_empty()));
        assert!(cases.iter().any(|c| !c.must_stay_current.is_empty()));
    }
}
