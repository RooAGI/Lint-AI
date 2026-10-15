//! Canonical source and session identifiers for captured provider memories.

#[derive(Debug, Clone, Copy)]
pub(crate) struct ProviderSource {
    pub name: &'static str,
    pub source_prefixes: &'static [&'static str],
    pub session_tag: &'static str,
}

pub(crate) const PROVIDERS: &[ProviderSource] = &[
    ProviderSource {
        name: "claude",
        source_prefixes: &["claude-code://", "claude://"],
        session_tag: "claude-session:",
    },
    ProviderSource {
        name: "codex",
        source_prefixes: &["codex://"],
        session_tag: "codex-session:",
    },
    ProviderSource {
        name: "gemini-cli",
        source_prefixes: &["gemini-cli://"],
        session_tag: "gemini-cli-session:",
    },
    ProviderSource {
        name: "agy",
        source_prefixes: &["agy://"],
        session_tag: "agy-session:",
    },
    ProviderSource {
        name: "muse",
        source_prefixes: &["muse://"],
        session_tag: "muse-session:",
    },
    ProviderSource {
        name: "openclaw",
        source_prefixes: &["openclaw://"],
        session_tag: "openclaw-session:",
    },
    ProviderSource {
        name: "hermes",
        source_prefixes: &["hermes://"],
        session_tag: "hermes-session:",
    },
    ProviderSource {
        name: "rooagi_runtime",
        source_prefixes: &["roo-runtime://"],
        session_tag: "rooagi_runtime:",
    },
];

pub(crate) const INTERNAL_MEMORY_SOURCE_PREFIXES: &[&str] = &["membrick://", "lint-ai://"];

pub(crate) fn source_prefix(provider_name: &str) -> &'static str {
    PROVIDERS
        .iter()
        .find(|provider| provider.name == provider_name)
        .and_then(|provider| provider.source_prefixes.first().copied())
        .unwrap_or_else(|| panic!("provider source is not registered: {provider_name}"))
}

pub(crate) fn session_tag(provider_name: &str) -> &'static str {
    PROVIDERS
        .iter()
        .find(|provider| provider.name == provider_name)
        .map(|provider| provider.session_tag)
        .unwrap_or_else(|| panic!("provider source is not registered: {provider_name}"))
}

pub(crate) fn is_provider_source(source: &str) -> bool {
    PROVIDERS.iter().any(|provider| {
        provider
            .source_prefixes
            .iter()
            .any(|prefix| source.starts_with(prefix))
    })
}

pub(crate) fn is_provider_session(source: &str, group_id: Option<&str>) -> bool {
    let Some(group_id) = group_id else {
        return false;
    };
    PROVIDERS.iter().any(|provider| {
        provider
            .source_prefixes
            .iter()
            .any(|prefix| source.starts_with(prefix))
            && group_id.starts_with(provider.session_tag)
    })
}

pub(crate) fn is_operational_provider_source(source: &str, group_id: Option<&str>) -> bool {
    is_provider_source(source) && !is_provider_session(source, group_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_registry_has_unique_prefixes_and_session_tags() {
        let mut prefixes = Vec::new();
        let mut tags = Vec::new();
        for provider in PROVIDERS {
            assert!(!provider.name.is_empty());
            assert!(!provider.source_prefixes.is_empty());
            assert!(!provider.session_tag.is_empty());
            for prefix in provider.source_prefixes {
                assert!(
                    !prefixes.contains(prefix),
                    "duplicate source prefix: {prefix}"
                );
                prefixes.push(prefix);
            }
            assert!(
                !tags.contains(&provider.session_tag),
                "duplicate session tag"
            );
            tags.push(provider.session_tag);
        }
    }

    #[test]
    fn codex_sessions_are_distinguished_from_codex_operational_records() {
        assert!(is_provider_session(
            "codex://project/session/outcome",
            Some("codex-session:project:session")
        ));
        assert!(is_operational_provider_source("codex://tool-event", None));
        assert!(!is_operational_provider_source(
            "codex://project/session/outcome",
            Some("codex-session:project:session")
        ));
    }
}
