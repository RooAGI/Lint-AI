//! Plain-text language cues for deterministic reasoning. The data is parsed
//! once; interpretation and evidence checks remain in typed Rust code.

use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Deserialize)]
pub(crate) struct EventFrameRule {
    pub phrase: String,
    #[serde(default)]
    pub prefer_named_event: bool,
    pub status: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ReasoningRules {
    pub event_frames: Vec<EventFrameRule>,
    pub event_nouns: Vec<String>,
    pub first_person_words: Vec<String>,
    pub acquisition_subject_markers: Vec<String>,
    pub action_prefix_words: Vec<String>,
    pub acquisition_frames: Vec<String>,
    pub acquisition_got_object_prefixes: Vec<String>,
    pub object_stop_words: Vec<String>,
    pub relative_time_numbers: Vec<String>,
    pub direct_participation_phrases: Vec<String>,
    pub reported_participation_regex: String,
    pub acquisition_query_phrases: Vec<String>,
    pub count_words: HashMap<String, usize>,
    pub temporal_order_markers: Vec<String>,
    pub query_time_units: Vec<String>,
    pub query_count_phrases: Vec<String>,
    pub query_option_connectors: Vec<String>,
    pub query_option_prefixes: Vec<String>,
    pub query_option_phrases: Vec<String>,
    pub query_temporal_diff_marker: String,
    pub query_temporal_filter_markers: Vec<String>,
    pub query_both_marker: String,
    pub query_event_marker: String,
    pub query_both_verbs: Vec<String>,
    pub query_aggregate_phrases: Vec<String>,
}

static RULES: OnceLock<ReasoningRules> = OnceLock::new();

pub(crate) fn reasoning_rules() -> &'static ReasoningRules {
    RULES.get_or_init(|| {
        let rules: ReasoningRules =
            serde_json::from_str(include_str!("../data/reasoning_rules.json"))
                .expect("bundled reasoning_rules.json must be valid");
        assert!(rules
            .event_frames
            .iter()
            .all(|rule| !rule.phrase.is_empty()));
        assert!(rules.event_frames.iter().all(|rule| {
            rule.status
                .as_deref()
                .is_none_or(|status| crate::index::ClaimStatus::parse(status).is_some())
        }));
        rules
    })
}
