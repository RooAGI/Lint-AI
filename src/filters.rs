use crate::config::{normalize_list, Config};

/// Return true if the concept is a built-in stopword.
pub fn is_stopword(concept: &str) -> bool {
    const STOP: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "if", "in", "is",
        "it", "its", "of", "on", "or", "the", "to", "via", "with",
        "api", "app", "apps", "auth", "build", "config", "data", "doc", "docs", "feature",
        "features", "file", "files", "guide", "help", "id", "index", "info", "issue", "issues",
        "key", "keys", "log", "logs", "model", "models", "page", "pages", "role", "roles",
        "run", "runs", "service", "services", "setup", "status", "system", "test", "tests",
        "tool", "tools", "user", "users", "web", "cli", "sdk", "repo", "project",
    ];
    STOP.contains(&concept)
}

/// Return true if the concept should be ignored for matching.
pub fn is_noise_concept(concept: &str, cfg: &Config) -> bool {
    if concept.len() < 3 {
        return true;
    }
    if concept.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }
    if is_stopword(concept) {
        return true;
    }
    let extra = normalize_list(&cfg.stopwords);
    extra.contains(&concept.to_string())
}
