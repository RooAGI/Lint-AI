//! Stateful query preparation for multi-turn search.
//!
//! When a search carries a `session_id`, the query is prepared against the
//! bounded prior state for that session before retrieval:
//!
//! 1. **Follow-up resolution.** Queries that read as continuations ("what
//!    about the budget?", "tell me more", "why?") are rewritten to carry the
//!    entities and phrasing of the recent turns, so the analyzer sees the
//!    same terms a standalone question would have used.
//! 2. **Temporal-anchor seeding.** When the new query sets no explicit
//!    temporal anchor but the session has one carried from a previous turn,
//!    the anchor is seeded into the rewritten query so temporal retrieval
//!    stays centered on the session's timeframe.
//!
//! A query that is not a follow-up, or a session with no prior state, flows
//! through unchanged: the stateless path is byte-for-byte today's behavior.

use crate::conversation_state::ConversationState;

/// Markers that strongly suggest the query continues a previous turn rather
/// than starting a new topic. Matched case-insensitively against the trimmed
/// query prefix.
const FOLLOW_UP_PREFIXES: &[&str] = &[
    "what about",
    "how about",
    "tell me more",
    "and ",
    "and then",
    "what else",
    "why",
    "how come",
    "what if",
];

/// Pronouns and demonstratives whose referent lives in the prior turns.
const COREFERENCE_TOKENS: &[&str] = &[
    " it ", " its ", " they ", " them ", " their ", " this ", " that ", " those ", " these ",
];

/// True when the query reads as a continuation of the session rather than a
/// standalone question.
pub fn is_follow_up(query: &str) -> bool {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return false;
    }
    if FOLLOW_UP_PREFIXES
        .iter()
        .any(|prefix| q.starts_with(prefix))
    {
        return true;
    }
    // A bare pronoun/demonstrative query ("tell me about it") is a follow-up;
    // a query that merely *contains* one alongside real content is not enough
    // on its own, so require the query to be short.
    if q.split_whitespace().count() <= 6 {
        let padded = format!(" {q} ");
        if COREFERENCE_TOKENS
            .iter()
            .any(|token| padded.contains(token))
        {
            return true;
        }
    }
    false
}

/// Rewrite a follow-up query against the session state, returning the query
/// text to analyze. Non-follow-ups and sessions without usable state return
/// the query unchanged.
///
/// The rewrite appends the session's resolved entities and the most recent
/// prior query as bracketed context. The analyzer then sees the same terms a
/// standalone question would have used, while the original phrasing stays
/// intact at the front so intent signals (question kind, temporal phrases)
/// are still read from the user's own words.
pub fn resolve_follow_up(query: &str, state: Option<&ConversationState>) -> String {
    let Some(state) = state else {
        return query.to_string();
    };
    if !is_follow_up(query) {
        return query.to_string();
    }
    let mut context_parts: Vec<String> = Vec::new();
    if !state.resolved_entities.is_empty() {
        context_parts.push(format!("entities: {}", state.resolved_entities.join(", ")));
    }
    if let Some(previous) = state.recent_queries.front() {
        // The most recent query is usually the turn being continued.
        context_parts.push(format!("previously asked: {previous}"));
    }
    if context_parts.is_empty() {
        return query.to_string();
    }
    // Seed the carried temporal anchor when the new query sets none of its
    // own. The analyzer resolves the explicit date into the same anchor the
    // prior turn used.
    if let Some(anchor) = state.temporal_anchor.as_deref() {
        context_parts.push(format!("temporal anchor: {anchor}"));
    }
    format!("{} [{}]", query.trim(), context_parts.join("; "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation_state::ConversationStateStore;

    fn state_with_history() -> ConversationState {
        let mut store = ConversationStateStore::new(None);
        let now = 1_700_000_000_000u64;
        store.observe(
            "user-a",
            "s1",
            "Tell me about the Quartz database",
            &["doc-1".to_string()],
            &["Quartz".to_string(), "database".to_string()],
            Some("2026-09-21".to_string()),
            now,
        );
        store.get("user-a", "s1", now).unwrap().clone()
    }

    #[test]
    fn detects_follow_up_prefixes() {
        assert!(is_follow_up("what about the budget?"));
        assert!(is_follow_up("how about next quarter"));
        assert!(is_follow_up("tell me more"));
        assert!(is_follow_up("and then what happened"));
        assert!(is_follow_up("why?"));
    }

    #[test]
    fn detects_short_coreference_queries() {
        assert!(is_follow_up("tell me about it"));
        assert!(is_follow_up("what did they decide"));
    }

    #[test]
    fn standalone_questions_are_not_follow_ups() {
        assert!(!is_follow_up("What is the Quartz database architecture?"));
        assert!(!is_follow_up(
            "How does the indexing pipeline handle updates to the document store?"
        ));
        assert!(!is_follow_up(""));
    }

    #[test]
    fn rewrite_carries_entities_and_anchor() {
        let state = state_with_history();
        let rewritten = resolve_follow_up("what are its limitations?", Some(&state));
        assert!(rewritten.starts_with("what are its limitations?"));
        assert!(rewritten.contains("Quartz"));
        assert!(rewritten.contains("2026-09-21"));
        assert!(rewritten.contains("Tell me about the Quartz database"));
    }

    #[test]
    fn non_follow_up_passes_through_unchanged() {
        let state = state_with_history();
        let query = "What is the capital of France?";
        assert_eq!(resolve_follow_up(query, Some(&state)), query);
    }

    #[test]
    fn missing_state_passes_through_unchanged() {
        let query = "tell me more";
        assert_eq!(resolve_follow_up(query, None), query);
    }
}
