//! Plain-text language cues for deterministic reasoning. The data is parsed
//! once; interpretation and evidence checks remain in typed Rust code.

use anyhow::{Context, Result};
use notify::{RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex, OnceLock, RwLock};
use std::thread::JoinHandle;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct EventFrameRule {
    pub phrase: String,
    #[serde(default)]
    pub prefer_named_event: bool,
    pub status: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
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
    pub query_comparison_markers: Vec<String>,
    pub query_temporal_diff_marker: String,
    pub query_temporal_filter_markers: Vec<String>,
    pub query_both_marker: String,
    pub query_event_marker: String,
    pub query_both_verbs: Vec<String>,
    pub query_aggregate_phrases: Vec<String>,
    pub query_current_state_markers: Vec<String>,
    pub query_explanation_terms: Vec<String>,
}

static RULES: OnceLock<ReasoningRules> = OnceLock::new();
static RULES_ARC: OnceLock<Arc<ReasoningRules>> = OnceLock::new();

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

pub(crate) fn builtin_rules() -> Arc<ReasoningRules> {
    Arc::clone(RULES_ARC.get_or_init(|| Arc::new(reasoning_rules().clone())))
}

/// Per-project immutable rule snapshots. The watcher validates a whole
/// candidate before swapping it, so a query sees either the previous rules or
/// the complete new rules. Invalid edits leave the last good snapshot active.
pub(crate) struct ProjectReasoningRules {
    current: Arc<RwLock<Arc<ReasoningRules>>>,
    status: Arc<Mutex<RuleReloadStatus>>,
    _watcher: Option<RuleWatcher>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct RuleReloadStatus {
    pub fingerprint: String,
    pub query_fingerprint: String,
    pub extraction_fingerprint: String,
    pub path: Option<PathBuf>,
    pub last_error: Option<String>,
}

struct RuleWatcher {
    stop: mpsc::Sender<RuleWatchEvent>,
    thread: Option<JoinHandle<()>>,
}

enum RuleWatchEvent {
    Changed(Vec<PathBuf>),
    Stop,
}

impl Drop for RuleWatcher {
    fn drop(&mut self) {
        let _ = self.stop.send(RuleWatchEvent::Stop);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl ProjectReasoningRules {
    pub(crate) fn open(project_root: &Path) -> Self {
        let current = Arc::new(RwLock::new(builtin_rules()));
        let status = Arc::new(Mutex::new(RuleReloadStatus::default()));
        let project_root = project_root
            .canonicalize()
            .unwrap_or_else(|_| project_root.to_path_buf());
        let settings_path = project_root.join(".membrick").join("settings.toml");
        let (rules_path, initial_settings_error) = match resolve_rules_path(&project_root) {
            Ok(path) => (path, None),
            Err(error) => (
                project_root.join(".membrick").join("rules.json"),
                Some(error),
            ),
        };

        if let Some(error) = initial_settings_error {
            if let Ok(mut status) = status.lock() {
                status.path = Some(rules_path.clone());
                status.last_error = Some(format!("{error:#}"));
            }
        } else if let Err(error) = reload_candidate(&rules_path, &current, &status) {
            if let Ok(mut status) = status.lock() {
                status.path = Some(rules_path.clone());
                status.last_error = Some(format!("{error:#}"));
            }
        }

        let (signal_tx, signal_rx) = mpsc::channel::<RuleWatchEvent>();
        let event_tx = signal_tx.clone();
        let mut watcher =
            notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
                if let Ok(event) = event {
                    let _ = event_tx.send(RuleWatchEvent::Changed(event.paths));
                }
            })
            .ok();
        let mut watched_parents = HashSet::new();
        if let Some(watcher) = watcher.as_mut() {
            // The project settings file is always inside .membrick. Watch that
            // directory so changing the configured rules path is observable.
            let project_state = project_root.join(".membrick");
            let watch_root = if project_state.exists() {
                project_state
            } else {
                project_root.clone()
            };
            if watcher
                .watch(&watch_root, RecursiveMode::NonRecursive)
                .is_ok()
            {
                watched_parents.insert(watch_root);
            }
        }

        let stop_tx = signal_tx;
        let stop_rx = signal_rx;
        let reload_root = project_root.clone();
        let reload_settings = settings_path;
        let reload_current = Arc::clone(&current);
        let reload_status = Arc::clone(&status);
        let thread = std::thread::Builder::new()
            .name("membrick-rules-reload".into())
            .spawn(move || {
                let mut watcher = watcher;
                let mut watched_parents = watched_parents;
                let mut active_path = rules_path;
                loop {
                    let mut recovery_scan = false;
                    let mut changed_paths = Vec::new();
                    match stop_rx.recv_timeout(std::time::Duration::from_secs(30)) {
                        Ok(RuleWatchEvent::Stop) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                            break
                        }
                        Ok(RuleWatchEvent::Changed(paths)) => changed_paths.extend(paths),
                        Err(mpsc::RecvTimeoutError::Timeout) => recovery_scan = true,
                    }
                    loop {
                        match stop_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                            Ok(RuleWatchEvent::Changed(paths)) => changed_paths.extend(paths),
                            Ok(RuleWatchEvent::Stop)
                            | Err(mpsc::RecvTimeoutError::Disconnected) => return,
                            Err(mpsc::RecvTimeoutError::Timeout) => break,
                        }
                    }
                    let configured =
                        resolve_rules_path_from_settings(&reload_root, &reload_settings);
                    match configured {
                        Ok(path) => {
                            let relevant_event = changed_paths.is_empty()
                                || changed_paths.iter().any(|changed| {
                                    changed == &reload_settings
                                        || changed
                                            == reload_settings.parent().unwrap_or(&reload_root)
                                        || changed
                                            .file_name()
                                            .is_some_and(|name| name == "settings.toml")
                                        || changed == &path
                                        || changed == &active_path
                                        || changed.parent().is_some_and(|parent| {
                                            (parent
                                                == reload_settings.parent().unwrap_or(&reload_root)
                                                || Some(parent) == path.parent()
                                                || Some(parent) == active_path.parent())
                                                && !is_high_volume_runtime_path(changed)
                                        })
                                });
                            if !recovery_scan && !relevant_event {
                                continue;
                            }
                            let state_dir = reload_root.join(".membrick");
                            let mut desired_parents = HashSet::from([if state_dir.exists() {
                                state_dir
                            } else {
                                reload_root.clone()
                            }]);
                            if let Some(parent) = path.parent() {
                                desired_parents.insert(parent.to_path_buf());
                            }
                            let removed_parents = watched_parents
                                .difference(&desired_parents)
                                .cloned()
                                .collect::<Vec<_>>();
                            for old_parent in removed_parents {
                                if let Some(watcher) = watcher.as_mut() {
                                    let _ = watcher.unwatch(&old_parent);
                                }
                                watched_parents.remove(&old_parent);
                            }
                            let added_parents = desired_parents
                                .difference(&watched_parents)
                                .cloned()
                                .collect::<Vec<_>>();
                            for new_parent in added_parents {
                                if new_parent.exists()
                                    && watcher.as_mut().is_some_and(|watcher| {
                                        watcher
                                            .watch(&new_parent, RecursiveMode::NonRecursive)
                                            .is_ok()
                                    })
                                {
                                    watched_parents.insert(new_parent);
                                }
                            }
                            if path != active_path {
                                active_path = path;
                            }
                            if let Err(error) =
                                reload_candidate(&active_path, &reload_current, &reload_status)
                            {
                                if let Ok(mut status) = reload_status.lock() {
                                    status.path = Some(active_path.clone());
                                    status.last_error = Some(format!("{error:#}"));
                                }
                            }
                        }
                        Err(error) => {
                            if let Ok(mut status) = reload_status.lock() {
                                status.last_error = Some(format!("{error:#}"));
                            }
                        }
                    }
                }
            })
            .ok();

        let watcher = match thread {
            Some(thread) => Some(RuleWatcher {
                stop: stop_tx,
                thread: Some(thread),
            }),
            _ => None,
        };
        Self {
            current,
            status,
            _watcher: watcher,
        }
    }

    pub(crate) fn snapshot(&self) -> Arc<ReasoningRules> {
        self.current
            .read()
            .map(|rules| Arc::clone(&rules))
            .unwrap_or_else(|_| builtin_rules())
    }

    pub(crate) fn status(&self) -> RuleReloadStatus {
        self.status
            .lock()
            .map(|status| status.clone())
            .unwrap_or_default()
    }

    /// Validate a proposed query-only overlay without writing it. Feedback
    /// callers use this before reserving a search review so a malformed rule
    /// proposal can be corrected and resubmitted with the same search ID.
    pub(crate) fn validate_query_rule_terms(
        &self,
        additions: &BTreeMap<String, Vec<String>>,
    ) -> Result<()> {
        if additions.is_empty() {
            return Ok(());
        }
        let path = self
            .status()
            .path
            .context("workspace rules path is unavailable")?;
        let existing = match std::fs::read(&path) {
            Ok(bytes) => Some(bytes),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        let mut overlay = match existing.as_deref() {
            Some(bytes) => serde_json::from_slice::<RuleOverlay>(bytes)
                .with_context(|| format!("parsing project rules at {}", path.display()))?,
            None => RuleOverlay {
                schema_version: 1,
                additions: BTreeMap::new(),
            },
        };
        anyhow::ensure!(
            overlay.schema_version == 1,
            "unsupported rules schema_version"
        );
        for (field, values) in additions {
            anyhow::ensure!(
                field.starts_with("query_") && RELOADABLE_RULE_LISTS.contains(&field.as_str()),
                "rule list {field:?} is not a supported query-only rule"
            );
            let target = overlay.additions.entry(field.clone()).or_default();
            for value in values {
                let value = value.trim();
                anyhow::ensure!(!value.is_empty(), "rule additions cannot be empty");
                if !target
                    .iter()
                    .any(|existing| existing.eq_ignore_ascii_case(value))
                {
                    target.push(value.to_string());
                }
            }
        }
        let mut merged: serde_json::Value =
            serde_json::from_str(include_str!("../data/reasoning_rules.json"))?;
        apply_rule_overlay(&mut merged, &overlay)?;
        let candidate: ReasoningRules = serde_json::from_value(merged)?;
        validate_rules(&candidate)
    }

    /// Add reviewed query-only vocabulary to this workspace's active overlay.
    /// The complete effective ruleset is validated before the atomic replace,
    /// then activated through the same reload path used by the watcher.
    pub(crate) fn add_query_rule_terms(
        &self,
        additions: BTreeMap<String, Vec<String>>,
    ) -> Result<usize> {
        let path = self
            .status()
            .path
            .context("workspace rules path is unavailable")?;
        let previous = match std::fs::read(&path) {
            Ok(bytes) => Some(bytes),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        let mut overlay = match previous.as_deref() {
            Some(bytes) => serde_json::from_slice::<RuleOverlay>(bytes)
                .with_context(|| format!("parsing project rules at {}", path.display()))?,
            None => RuleOverlay {
                schema_version: 1,
                additions: BTreeMap::new(),
            },
        };
        anyhow::ensure!(
            overlay.schema_version == 1,
            "unsupported rules schema_version"
        );

        let mut added = 0usize;
        let mut accepted = HashSet::new();
        for (field, values) in additions {
            anyhow::ensure!(
                field.starts_with("query_") && RELOADABLE_RULE_LISTS.contains(&field.as_str()),
                "rule list {field:?} is not a supported query-only rule"
            );
            let target = overlay.additions.entry(field.clone()).or_default();
            for value in values {
                let value = value.trim();
                anyhow::ensure!(!value.is_empty(), "rule additions cannot be empty");
                accepted.insert((field.clone(), value.to_lowercase()));
                if !target
                    .iter()
                    .any(|existing| existing.eq_ignore_ascii_case(value))
                {
                    target.push(value.to_string());
                    added += 1;
                }
            }
        }
        if added == 0 {
            return Ok(accepted.len());
        }

        let mut merged: serde_json::Value =
            serde_json::from_str(include_str!("../data/reasoning_rules.json"))?;
        apply_rule_overlay(&mut merged, &overlay)?;
        let candidate: ReasoningRules = serde_json::from_value(merged)?;
        validate_rules(&candidate)?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut temp = path.clone();
        temp.set_extension(format!("rules-tmp-{}", std::process::id()));
        let mut options = std::fs::OpenOptions::new();
        options.create(true).truncate(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        use std::io::Write;
        let mut file = options.open(&temp)?;
        file.write_all(&serde_json::to_vec_pretty(&overlay)?)?;
        file.sync_all()?;
        std::fs::rename(&temp, &path)?;

        if let Err(error) = reload_candidate(&path, &self.current, &self.status) {
            match previous {
                Some(bytes) => std::fs::write(&path, bytes)?,
                None => {
                    let _ = std::fs::remove_file(&path);
                }
            }
            return Err(error);
        }
        Ok(accepted.len())
    }

    #[cfg(test)]
    fn reload_now(&self, project_root: &Path) -> Result<()> {
        reload_candidate(
            &resolve_rules_path(project_root)?,
            &self.current,
            &self.status,
        )
    }
}

fn is_high_volume_runtime_path(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(
            "mcp-call-history.jsonl"
                | "mcp-call-history.lock"
                | "query-telemetry.json"
                | "provider-telemetry"
                | "runtime"
        )
    )
}

fn resolve_rules_path(project_root: &Path) -> Result<PathBuf> {
    let settings = project_root.join(".membrick").join("settings.toml");
    resolve_rules_path_from_settings(project_root, &settings)
}

fn resolve_rules_path_from_settings(project_root: &Path, settings_path: &Path) -> Result<PathBuf> {
    let configured = if settings_path.exists() {
        let text = std::fs::read_to_string(settings_path)
            .with_context(|| format!("reading project settings at {}", settings_path.display()))?;
        let settings: toml::Value = toml::from_str(&text)
            .with_context(|| format!("parsing project settings at {}", settings_path.display()))?;
        match settings.get("rules") {
            None => None,
            Some(rules) => {
                let rules = rules
                    .as_table()
                    .context("settings [rules] must be a table")?;
                anyhow::ensure!(
                    rules.keys().all(|key| key == "path"),
                    "unsupported setting in [rules]; only `path` is supported"
                );
                match rules.get("path") {
                    None => None,
                    Some(path) => {
                        let path = path
                            .as_str()
                            .context("settings [rules].path must be a string")?
                            .trim();
                        anyhow::ensure!(!path.is_empty(), "settings [rules].path cannot be empty");
                        Some(PathBuf::from(path))
                    }
                }
            }
        }
    } else {
        None
    };
    let path = configured.unwrap_or_else(|| PathBuf::from(".membrick/rules.json"));
    Ok(if path.is_absolute() {
        path
    } else {
        project_root.join(path)
    })
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RuleOverlay {
    schema_version: u32,
    #[serde(default)]
    additions: BTreeMap<String, Vec<String>>,
}

const RELOADABLE_RULE_LISTS: &[&str] = &[
    "temporal_order_markers",
    "query_time_units",
    "query_count_phrases",
    "query_option_connectors",
    "query_option_prefixes",
    "query_option_phrases",
    "query_comparison_markers",
    "query_temporal_filter_markers",
    "query_both_verbs",
    "query_aggregate_phrases",
    "query_current_state_markers",
    "query_explanation_terms",
];

fn reload_candidate(
    path: &Path,
    current: &RwLock<Arc<ReasoningRules>>,
    status: &Mutex<RuleReloadStatus>,
) -> Result<()> {
    let mut merged: serde_json::Value =
        serde_json::from_str(include_str!("../data/reasoning_rules.json"))?;
    if path.exists() {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading project rules at {}", path.display()))?;
        let overlay: RuleOverlay = serde_json::from_str(&content)
            .with_context(|| format!("parsing project rules at {}", path.display()))?;
        anyhow::ensure!(
            overlay.schema_version == 1,
            "unsupported rules schema_version {}",
            overlay.schema_version
        );
        apply_rule_overlay(&mut merged, &overlay)?;
    }
    let candidate: ReasoningRules = serde_json::from_value(merged)?;
    validate_rules(&candidate)?;
    let serialized = serde_json::to_value(&candidate)?;
    // Serialize through serde_json::Value so object keys use its canonical
    // ordering. Serializing ReasoningRules directly includes a HashMap whose
    // iteration order can vary between reloads, falsely changing the rules
    // generation for formatting-only edits.
    let fingerprint = format!("{:x}", Sha256::digest(serde_json::to_vec(&serialized)?));
    let query_fingerprint = fingerprint_fields(&serialized, |name| {
        name == "temporal_order_markers" || name.starts_with("query_")
    })?;
    let extraction_fingerprint = fingerprint_fields(&serialized, |name| {
        name != "temporal_order_markers" && !name.starts_with("query_")
    })?;
    let mut active = current
        .write()
        .map_err(|_| anyhow::anyhow!("rules snapshot lock poisoned"))?;
    let active_fingerprint = status
        .lock()
        .map(|status| status.fingerprint.clone())
        .unwrap_or_default();
    if fingerprint != active_fingerprint {
        *active = Arc::new(candidate);
    }
    if let Ok(mut status) = status.lock() {
        status.fingerprint = fingerprint;
        status.query_fingerprint = query_fingerprint;
        status.extraction_fingerprint = extraction_fingerprint;
        status.path = Some(path.to_path_buf());
        status.last_error = None;
    }
    Ok(())
}

fn apply_rule_overlay(merged: &mut serde_json::Value, overlay: &RuleOverlay) -> Result<()> {
    anyhow::ensure!(
        overlay.schema_version == 1,
        "unsupported rules schema_version"
    );
    for (field, additions) in &overlay.additions {
        anyhow::ensure!(
            RELOADABLE_RULE_LISTS.contains(&field.as_str()),
            "rule list {field:?} is not reloadable"
        );
        let target = merged
            .get_mut(field)
            .and_then(serde_json::Value::as_array_mut)
            .with_context(|| format!("built-in rule list {field:?} is unavailable"))?;
        for addition in additions {
            let addition = addition.trim();
            anyhow::ensure!(!addition.is_empty(), "rule additions cannot be empty");
            if !target.iter().any(|value| {
                value
                    .as_str()
                    .is_some_and(|known| known.eq_ignore_ascii_case(addition))
            }) {
                target.push(serde_json::Value::String(addition.to_string()));
            }
        }
    }
    Ok(())
}

fn fingerprint_fields(rules: &serde_json::Value, include: impl Fn(&str) -> bool) -> Result<String> {
    let fields = rules
        .as_object()
        .context("serialized rules must be an object")?
        .iter()
        .filter(|(name, _)| include(name))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect::<serde_json::Map<_, _>>();
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&fields)?)
    ))
}

fn validate_rules(rules: &ReasoningRules) -> Result<()> {
    anyhow::ensure!(
        rules
            .event_frames
            .iter()
            .all(|rule| !rule.phrase.trim().is_empty()),
        "event frame phrases cannot be empty"
    );
    anyhow::ensure!(
        rules.event_frames.iter().all(|rule| rule
            .status
            .as_deref()
            .is_none_or(|status| crate::index::ClaimStatus::parse(status).is_some())),
        "invalid event frame claim status"
    );
    regex::Regex::new(&rules.reported_participation_regex)
        .context("invalid reported_participation_regex")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    struct TestRoot(PathBuf);

    impl TestRoot {
        fn new() -> Self {
            static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "membrick-project-rules-{}-{}",
                std::process::id(),
                NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir_all(root.join(".membrick")).unwrap();
            Self(root)
        }
    }

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn project_overlay_is_scoped_validated_and_keeps_last_good_rules() {
        let first = TestRoot::new();
        let second = TestRoot::new();
        let rules_path = first.0.join(".membrick/rules.json");
        std::fs::write(
            &rules_path,
            r#"{"schema_version":1,"additions":{"temporal_order_markers":["in release sequence"]}}"#,
        )
        .unwrap();

        let first_rules = ProjectReasoningRules::open(&first.0);
        let second_rules = ProjectReasoningRules::open(&second.0);
        assert!(
            first_rules
                .snapshot()
                .temporal_order_markers
                .contains(&"in release sequence".to_string()),
            "first project rules failed to load: {:?}",
            first_rules.status()
        );
        assert!(!second_rules
            .snapshot()
            .temporal_order_markers
            .contains(&"in release sequence".to_string()));

        std::fs::write(&rules_path, r#"{"schema_version":1,"unsupported":true}"#).unwrap();
        assert!(first_rules.reload_now(&first.0).is_err());
        assert!(first_rules
            .snapshot()
            .temporal_order_markers
            .contains(&"in release sequence".to_string()));
    }

    #[test]
    fn rule_reload_keeps_pinned_queries_and_other_projects_isolated() {
        let first = TestRoot::new();
        let second = TestRoot::new();
        let first_path = first.0.join(".membrick/rules.json");
        let second_path = second.0.join(".membrick/rules.json");
        std::fs::write(
            &first_path,
            r#"{"schema_version":1,"additions":{"query_explanation_terms":["why-first"]}}"#,
        )
        .unwrap();
        std::fs::write(
            &second_path,
            r#"{"schema_version":1,"additions":{"query_explanation_terms":["why-second"]}}"#,
        )
        .unwrap();

        let first_rules = ProjectReasoningRules::open(&first.0);
        let second_rules = ProjectReasoningRules::open(&second.0);
        let pinned = first_rules.snapshot();
        let pinned_query = crate::query_plan::PreparedQuery::new_with_rules(
            "Why did we change this interface?",
            Arc::clone(&pinned),
        );
        assert!(pinned.query_explanation_terms.contains(&"why-first".into()));
        assert!(pinned_query
            .analysis()
            .augmented_query
            .contains("why-first"));
        assert!(!pinned
            .query_explanation_terms
            .contains(&"why-second".into()));

        std::fs::write(
            &first_path,
            r#"{"schema_version":1,"additions":{"query_explanation_terms":["why-first-revised"]}}"#,
        )
        .unwrap();
        first_rules.reload_now(&first.0).unwrap();

        let active_first = first_rules.snapshot();
        let active_second = second_rules.snapshot();
        let next_query = crate::query_plan::PreparedQuery::new_with_rules(
            "Why did we change this interface?",
            Arc::clone(&active_first),
        );
        assert!(pinned.query_explanation_terms.contains(&"why-first".into()));
        assert!(pinned_query
            .analysis()
            .augmented_query
            .contains("why-first"));
        assert!(!pinned_query
            .analysis()
            .augmented_query
            .contains("why-first-revised"));
        assert!(next_query
            .analysis()
            .augmented_query
            .contains("why-first-revised"));
        assert!(!pinned
            .query_explanation_terms
            .contains(&"why-first-revised".into()));
        assert!(active_first
            .query_explanation_terms
            .contains(&"why-first-revised".into()));
        assert!(!active_first
            .query_explanation_terms
            .contains(&"why-second".into()));
        assert!(active_second
            .query_explanation_terms
            .contains(&"why-second".into()));
        assert!(!active_second
            .query_explanation_terms
            .contains(&"why-first-revised".into()));
    }

    #[test]
    fn project_overlay_rejects_unknown_rule_fields_and_empty_phrases() {
        let root = TestRoot::new();
        let rules_path = root.0.join(".membrick/rules.json");
        std::fs::write(
            &rules_path,
            r#"{"schema_version":1,"additions":{"query_option_phrases":[""]}}"#,
        )
        .unwrap();
        let rules = ProjectReasoningRules::open(&root.0);
        assert!(rules.status().last_error.is_some());
        assert_eq!(
            rules.snapshot().query_option_phrases,
            reasoning_rules().query_option_phrases
        );
    }

    #[test]
    fn reviewed_query_rule_additions_activate_and_reject_extraction_rules() {
        let root = TestRoot::new();
        let rules = ProjectReasoningRules::open(&root.0);
        let additions = BTreeMap::from([(
            "query_current_state_markers".to_string(),
            vec!["as of release".to_string()],
        )]);
        assert_eq!(rules.add_query_rule_terms(additions.clone()).unwrap(), 1);
        assert!(rules
            .snapshot()
            .query_current_state_markers
            .contains(&"as of release".to_string()));
        assert_eq!(rules.add_query_rule_terms(additions).unwrap(), 1);

        let before = std::fs::read(root.0.join(".membrick/rules.json")).unwrap();
        let extraction_rule =
            BTreeMap::from([("event_nouns".to_string(), vec!["deployment".to_string()])]);
        assert!(rules.add_query_rule_terms(extraction_rule).is_err());
        assert_eq!(
            std::fs::read(root.0.join(".membrick/rules.json")).unwrap(),
            before
        );
    }

    #[test]
    fn reloadable_comparison_cue_changes_typed_query_analysis() {
        let root = TestRoot::new();
        std::fs::write(
            root.0.join(".membrick/rules.json"),
            r#"{"schema_version":1,"additions":{"query_comparison_markers":["weigh"]}}"#,
        )
        .unwrap();
        let rules = ProjectReasoningRules::open(&root.0);
        let analysis = crate::query_semantics::analyze_query_with_rules(
            "Weigh the old and new command interface",
            rules.snapshot(),
        );
        assert_eq!(
            analysis.intent,
            crate::query_semantics::QueryIntent::Compare
        );
        assert_eq!(
            analysis.comparison_operands,
            vec!["old command interface", "new command interface"]
        );
        let status = rules.status();
        let builtins = serde_json::to_value(reasoning_rules()).unwrap();
        let built_in_query_fingerprint = fingerprint_fields(&builtins, |name| {
            name == "temporal_order_markers" || name.starts_with("query_")
        })
        .unwrap();
        let built_in_extraction_fingerprint = fingerprint_fields(&builtins, |name| {
            name != "temporal_order_markers" && !name.starts_with("query_")
        })
        .unwrap();
        assert_ne!(status.query_fingerprint, built_in_query_fingerprint);
        assert_eq!(
            status.extraction_fingerprint,
            built_in_extraction_fingerprint
        );
    }

    #[test]
    fn configured_rule_path_is_project_relative_and_can_be_absolute() {
        let root = TestRoot::new();
        let settings = root.0.join(".membrick/settings.toml");
        std::fs::write(&settings, "[rules]\npath = 'config/project-rules.json'\n").unwrap();
        assert_eq!(
            resolve_rules_path(&root.0).unwrap(),
            root.0.join("config/project-rules.json")
        );
        let absolute = root.0.join("external/rules.json");
        std::fs::write(
            &settings,
            format!("[rules]\npath = {:?}\n", absolute.display()),
        )
        .unwrap();
        assert_eq!(resolve_rules_path(&root.0).unwrap(), absolute);
    }

    #[test]
    fn watcher_reloads_rules_and_rebinds_when_settings_path_changes() {
        let root = TestRoot::new();
        let default_rules = root.0.join(".membrick/rules.json");
        std::fs::write(
            &default_rules,
            r#"{"schema_version":1,"additions":{"temporal_order_markers":["first cue"]}}"#,
        )
        .unwrap();
        let rules = ProjectReasoningRules::open(&root.0);
        assert!(rules
            .snapshot()
            .temporal_order_markers
            .contains(&"first cue".to_string()));

        let external_dir = root.0.join("external");
        std::fs::create_dir_all(&external_dir).unwrap();
        let external_rules = external_dir.join("rules.json");
        std::fs::write(
            &external_rules,
            r#"{"schema_version":1,"additions":{"temporal_order_markers":["second cue"]}}"#,
        )
        .unwrap();
        std::fs::write(
            root.0.join(".membrick/settings.toml"),
            "[rules]\npath = 'external/rules.json'\n",
        )
        .unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            if rules
                .snapshot()
                .temporal_order_markers
                .contains(&"second cue".to_string())
            {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        panic!("project rules watcher did not reload the configured overlay");
    }

    #[test]
    fn watcher_reloads_atomic_replacement_and_overlay_deletion() {
        let root = TestRoot::new();
        let rules_path = root.0.join(".membrick/rules.json");
        std::fs::write(
            &rules_path,
            r#"{"schema_version":1,"additions":{"query_aggregate_phrases":["first phrase"]}}"#,
        )
        .unwrap();
        let rules = ProjectReasoningRules::open(&root.0);
        let replacement = root.0.join(".membrick/rules.json.next");
        std::fs::write(
            &replacement,
            r#"{"schema_version":1,"additions":{"query_aggregate_phrases":["replacement phrase"]}}"#,
        )
        .unwrap();
        std::fs::rename(&replacement, &rules_path).unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            if rules
                .snapshot()
                .query_aggregate_phrases
                .contains(&"replacement phrase".to_string())
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        assert!(rules
            .snapshot()
            .query_aggregate_phrases
            .contains(&"replacement phrase".to_string()));

        std::fs::remove_file(&rules_path).unwrap();
        let defaults = reasoning_rules().query_aggregate_phrases.clone();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            if rules.snapshot().query_aggregate_phrases == defaults {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        panic!("project rules watcher did not restore bundled defaults after deletion");
    }

    #[test]
    fn watcher_keeps_last_good_rules_on_invalid_edit_and_ignores_formatting_only_changes() {
        let root = TestRoot::new();
        let rules_path = root.0.join(".membrick/rules.json");
        std::fs::write(
            &rules_path,
            r#"{"schema_version":1,"additions":{"query_explanation_terms":["because of"]}}"#,
        )
        .unwrap();

        let rules = ProjectReasoningRules::open(&root.0);
        let initial = rules.snapshot();
        assert!(initial
            .query_explanation_terms
            .contains(&"because of".to_string()));
        let initial_fingerprint = rules.status().fingerprint;

        // An equivalent effective ruleset must not replace the immutable
        // snapshot or change its generation just because JSON was reformatted.
        std::fs::write(
            &rules_path,
            "{\n  \"schema_version\": 1,\n  \"additions\": {\n    \"query_explanation_terms\": [\"because of\"]\n  }\n}\n",
        )
        .unwrap();
        rules.reload_now(&root.0).unwrap();
        let after_formatting = rules.snapshot();
        assert!(
            Arc::ptr_eq(&initial, &after_formatting),
            "formatting-only edit replaced effective rules: before={initial_fingerprint}, after={:?}",
            rules.status()
        );
        assert_eq!(rules.status().fingerprint, initial_fingerprint);

        // Invalid content is observable but never displaces the last accepted
        // rules. Repairing the same file should clear the error and activate
        // the new generation without restarting the workspace service.
        std::fs::write(&rules_path, r#"{"schema_version":1,"additions":false}"#).unwrap();
        let invalid_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < invalid_deadline {
            if rules.status().last_error.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        assert!(rules.status().last_error.is_some());
        assert!(Arc::ptr_eq(&initial, &rules.snapshot()));
        assert_eq!(rules.status().fingerprint, initial_fingerprint);

        std::fs::write(
            &rules_path,
            r#"{"schema_version":1,"additions":{"query_explanation_terms":["as justified by"]}}"#,
        )
        .unwrap();
        let repaired_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < repaired_deadline {
            let status = rules.status();
            if status.last_error.is_none()
                && rules
                    .snapshot()
                    .query_explanation_terms
                    .contains(&"as justified by".to_string())
            {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        panic!("valid rules did not activate after repairing an invalid overlay");
    }

    #[test]
    fn project_query_rules_change_typed_query_analysis() {
        let root = TestRoot::new();
        std::fs::write(
            root.0.join(".membrick/rules.json"),
            r#"{"schema_version":1,"additions":{"query_aggregate_phrases":["historic join"]}}"#,
        )
        .unwrap();
        let project_rules = ProjectReasoningRules::open(&root.0);
        let query = "What is the historic join of Alpha and Beta?";
        assert_ne!(
            crate::query_semantics::analyze_query(query).intent,
            crate::query_semantics::QueryIntent::Aggregate
        );
        assert_eq!(
            crate::query_semantics::analyze_query_with_rules(query, project_rules.snapshot())
                .intent,
            crate::query_semantics::QueryIntent::Aggregate
        );
        let active_status = project_rules.status();
        let builtins = serde_json::to_value(reasoning_rules()).unwrap();
        let builtin_query_fingerprint = fingerprint_fields(&builtins, |name| {
            name == "temporal_order_markers" || name.starts_with("query_")
        })
        .unwrap();
        let builtin_extraction_fingerprint = fingerprint_fields(&builtins, |name| {
            name != "temporal_order_markers" && !name.starts_with("query_")
        })
        .unwrap();
        assert_ne!(builtin_query_fingerprint, active_status.query_fingerprint);
        assert_eq!(
            builtin_extraction_fingerprint,
            active_status.extraction_fingerprint
        );
    }

    #[test]
    fn reloadable_rationale_terms_change_explanation_queries() {
        let root = TestRoot::new();
        std::fs::write(
            root.0.join(".membrick/rules.json"),
            r#"{"schema_version":1,"additions":{"query_explanation_terms":["justification"]}}"#,
        )
        .unwrap();
        let rules = ProjectReasoningRules::open(&root.0);
        let analysis = crate::query_semantics::analyze_query_with_rules(
            "Why did we change the command interface?",
            rules.snapshot(),
        );
        assert_eq!(
            analysis.intent,
            crate::query_semantics::QueryIntent::Explain
        );
        assert!(analysis.augmented_query.contains("justification"));
        assert_ne!(
            analysis.augmented_query,
            crate::query_semantics::analyze_query("Why did we change the command interface?")
                .augmented_query
        );
        let status = rules.status();
        let builtins = serde_json::to_value(reasoning_rules()).unwrap();
        let builtin_query_fingerprint = fingerprint_fields(&builtins, |name| {
            name == "temporal_order_markers" || name.starts_with("query_")
        })
        .unwrap();
        let builtin_extraction_fingerprint = fingerprint_fields(&builtins, |name| {
            name != "temporal_order_markers" && !name.starts_with("query_")
        })
        .unwrap();
        assert_ne!(status.query_fingerprint, builtin_query_fingerprint);
        assert_eq!(
            status.extraction_fingerprint,
            builtin_extraction_fingerprint
        );
    }

    #[test]
    fn reloadable_current_state_markers_classify_applicable_state_queries() {
        let root = TestRoot::new();
        let query = "What is the applicable command interface?";
        assert_eq!(
            crate::query_semantics::analyze_query(query).intent,
            crate::query_semantics::QueryIntent::Lookup
        );
        assert_eq!(
            crate::query_semantics::analyze_query("What is the latest command interface?").intent,
            crate::query_semantics::QueryIntent::CurrentState
        );
        assert_eq!(
            crate::query_semantics::analyze_query(
                "What is the order of the three events from earliest to latest in the past month?"
            )
            .intent,
            crate::query_semantics::QueryIntent::TemporalOrder
        );
        std::fs::write(
            root.0.join(".membrick/rules.json"),
            r#"{"schema_version":1,"additions":{"query_current_state_markers":["applicable"]}}"#,
        )
        .unwrap();
        let rules = ProjectReasoningRules::open(&root.0);
        let analysis = crate::query_semantics::analyze_query_with_rules(query, rules.snapshot());
        assert_eq!(
            analysis.intent,
            crate::query_semantics::QueryIntent::CurrentState
        );
        assert!(!analysis.intent.is_multi_operand());

        let status = rules.status();
        let builtins = serde_json::to_value(reasoning_rules()).unwrap();
        let builtin_extraction_fingerprint = fingerprint_fields(&builtins, |name| {
            name != "temporal_order_markers" && !name.starts_with("query_")
        })
        .unwrap();
        assert_eq!(
            status.extraction_fingerprint,
            builtin_extraction_fingerprint
        );
    }
}
