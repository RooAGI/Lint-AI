use crate::pipeline::{IndexStore, MemoryIndexLayout, PipelineOptions};
use crate::segments::SegmentRoutingStrategy;
use crate::source::SourceDocument;
use anyhow::{Context, Result};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAX_EVENT_BYTES: usize = 64 * 1024;
const MAX_STRING_BYTES: usize = 8 * 1024;
const MAX_ARRAY_ITEMS: usize = 128;
const MAX_TRANSCRIPT_BYTES: u64 = 4 * 1024 * 1024;
const REPLAY_SESSION_ENV: &str = "LINT_AI_REPLAY_SESSION_ID";
const INTERRUPTED_SESSION_STALE_SECONDS: u64 = 30 * 60;

#[derive(Debug, Clone, Copy)]
pub enum RecordingProvider {
    Claude,
    Codex,
    Gemini,
    Agy,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionImportReport {
    pub session_id: String,
    pub group_id: String,
    pub imported_document_ids: Vec<String>,
    pub skipped_events: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ReplayReport {
    pub provider: String,
    pub baseline_session_id: String,
    pub replay_session_id: String,
    pub replay_archive: String,
    pub lint_ai_enabled: bool,
    pub exit_code: Option<i32>,
    pub output_bytes: usize,
    pub prompt_count: usize,
    pub recorded_event_count: usize,
    pub recording_complete: bool,
}

/// Launch a fresh provider process from the recorded user prompts.
/// Replay recording is always enabled for the duration of the process.
pub fn replay_recorded_session(
    provider: RecordingProvider,
    project_root: &Path,
    archive_override: Option<&Path>,
    session_id: &str,
    enable_lint_ai: bool,
) -> Result<ReplayReport> {
    let replay_workspace = ReplayWorkspace::create(project_root)?;
    let archive_root = archive_override
        .map(Path::to_path_buf)
        .unwrap_or_else(|| session_root(provider, project_root));
    let baseline_dir = archive_root.join(safe_component(session_id));
    let events_path = baseline_dir.join("events.jsonl");
    let prompts = recorded_prompts(&events_path)?;
    let replay_session_id = format!(
        "replay-{}-{}",
        timestamp_compact(),
        safe_component(session_id)
    );
    let replay_dir = archive_root.join(safe_component(&replay_session_id));
    fs::create_dir_all(&replay_dir)?;
    let manifest_path = replay_dir.join("manifest.json");
    write_manifest(
        &manifest_path,
        &serde_json::json!({
            "schema_version": 1,
            "session_id": replay_session_id,
            "provider": provider.as_str(),
            "project_root": project_root,
            "started_at": timestamp(),
            "ended_at": Value::Null,
            "status": "active",
            "run_type": "replay",
            "replay_of_session_id": session_id,
            "recording_mode": "replay",
            "recording_enabled": true,
            "lint_ai_enabled": enable_lint_ai,
            "prompt_count": prompts.len(),
            "event_count": 0,
            "redaction_policy": "default-v1"
        }),
    )?;

    set_recording_state(provider, replay_workspace.path(), true)?;
    set_lint_ai_state(provider, replay_workspace.path(), enable_lint_ai)?;

    let command_result = run_provider_process(
        provider,
        replay_workspace.path(),
        &prompts,
        &replay_session_id,
    );
    sync_replay_archive(
        provider,
        replay_workspace.path(),
        archive_root.as_path(),
        &replay_session_id,
    )?;
    let execution = match command_result {
        Ok(execution) => execution,
        Err(error) => {
            let mut manifest: Value = serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;
            manifest["ended_at"] = Value::String(timestamp());
            manifest["status"] = Value::String("failed".to_string());
            manifest["recording_error"] = Value::String(error.to_string());
            write_manifest(&manifest_path, &manifest)?;
            return Err(error);
        }
    };
    let recorded_event_count = provider_event_count(&replay_dir.join("events.jsonl"))?;
    let recording_complete =
        replay_recording_complete(&replay_dir.join("events.jsonl"), prompts.len())?;

    record_event(
        provider,
        &archive_root,
        project_root,
        &replay_session_id,
        "ReplayCompleted",
        serde_json::json!({
            "replay_of_session_id": session_id,
            "prompt_count": prompts.len(),
            "exit_code": execution.exit_code,
            "stdout": execution.stdout,
            "stderr": execution.stderr,
        }),
    )?;
    let mut manifest: Value = serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;
    manifest["ended_at"] = Value::String(timestamp());
    manifest["status"] = Value::String(if execution.success && recording_complete {
        "replayed".to_string()
    } else {
        "failed".to_string()
    });
    if !recording_complete {
        manifest["recording_error"] = Value::String(
            "provider process completed without recording any provider hook events".to_string(),
        );
    }
    write_manifest(&manifest_path, &manifest)?;

    Ok(ReplayReport {
        provider: provider.as_str().to_string(),
        baseline_session_id: session_id.to_string(),
        replay_session_id,
        replay_archive: replay_dir.to_string_lossy().into_owned(),
        lint_ai_enabled: enable_lint_ai,
        exit_code: execution.exit_code,
        output_bytes: execution.stdout.len() + execution.stderr.len(),
        prompt_count: prompts.len(),
        recorded_event_count,
        recording_complete,
    })
}

struct ReplayWorkspace {
    path: PathBuf,
}

impl ReplayWorkspace {
    fn create(source: &Path) -> Result<Self> {
        let path = source
            .join(".lint-ai")
            .join("replay-workspaces")
            .join(format!(
                "replay-{}-{}",
                std::process::id(),
                timestamp_compact()
            ));
        fs::create_dir_all(&path)?;
        copy_replay_tree(source, &path)?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ReplayWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn copy_replay_tree(source: &Path, destination: &Path) -> Result<()> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let name = entry.file_name();
        if matches!(
            name.to_str(),
            Some(".git" | ".lint-ai" | "target" | "node_modules")
        ) {
            continue;
        }
        let destination_path = destination.join(&name);
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            fs::create_dir_all(&destination_path)?;
            copy_replay_tree(&source_path, &destination_path)?;
        } else if file_type.is_file() {
            fs::copy(&source_path, &destination_path)?;
        }
    }
    Ok(())
}

fn sync_replay_archive(
    provider: RecordingProvider,
    workspace_root: &Path,
    archive_root: &Path,
    session_id: &str,
) -> Result<()> {
    let source = session_root(provider, workspace_root).join(safe_component(session_id));
    let destination = archive_root.join(safe_component(session_id));
    fs::create_dir_all(&destination)?;
    for name in ["recording.json", "events.jsonl"] {
        let source_path = source.join(name);
        if source_path.is_file() {
            fs::copy(source_path, destination.join(name))?;
            restrict_path_permissions(&destination.join(name))?;
        }
    }
    Ok(())
}

/// Count provider hook events while excluding the synthetic ReplayCompleted
/// event written by the replay orchestrator itself.
fn provider_event_count(events_path: &Path) -> Result<usize> {
    if !events_path.exists() {
        return Ok(0);
    }
    let content = fs::read_to_string(events_path)?;
    Ok(content
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|event| {
            let kind = event.get("kind").and_then(Value::as_str);
            !(kind == Some("provider_event")
                && event
                    .get("payload")
                    .and_then(|payload| payload.get("replay_of_session_id"))
                    .is_some())
        })
        .count())
}

fn replay_recording_complete(events_path: &Path, prompt_count: usize) -> Result<bool> {
    if provider_event_count(events_path)? == 0 {
        return Ok(false);
    }
    if prompt_count == 0 {
        return Ok(true);
    }
    let content = fs::read_to_string(events_path)?;
    Ok(content
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .any(|event| event.get("kind").and_then(Value::as_str) == Some("user_prompt")))
}

#[derive(Debug, Default)]
struct ReplayExecution {
    success: bool,
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
}

fn recorded_prompts(events_path: &Path) -> Result<Vec<String>> {
    let content = fs::read_to_string(events_path)
        .with_context(|| format!("failed to read recorded session {}", events_path.display()))?;
    let mut prompts = Vec::new();
    for line in content.lines() {
        let Ok(event) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if event.get("kind").and_then(Value::as_str) != Some("user_prompt") {
            continue;
        }
        if let Some(prompt) = event
            .get("payload")
            .and_then(|payload| payload.get("prompt"))
            .and_then(Value::as_str)
            .filter(|prompt| !prompt.trim().is_empty())
        {
            prompts.push(prompt.to_string());
        }
    }
    if prompts.is_empty() {
        anyhow::bail!("recorded session contains no replayable user prompt")
    }
    Ok(prompts)
}

fn run_provider_process(
    provider: RecordingProvider,
    project_root: &Path,
    prompts: &[String],
    replay_session_id: &str,
) -> Result<ReplayExecution> {
    let mut execution = ReplayExecution::default();
    let mut provider_session_id = None;
    let isolated_home = project_root.join(".lint-ai-replay-home");
    fs::create_dir_all(&isolated_home)?;
    let isolated_codex_home = isolated_home.join(".codex");
    fs::create_dir_all(&isolated_codex_home)?;
    let isolated_app_data = isolated_home.join("AppData").join("Roaming");
    let isolated_local_app_data = isolated_home.join("AppData").join("Local");
    fs::create_dir_all(&isolated_app_data)?;
    fs::create_dir_all(&isolated_local_app_data)?;

    for (index, prompt) in prompts.iter().enumerate() {
        eprintln!(
            "lint-ai replay: starting turn {}/{} ({})",
            index + 1,
            prompts.len(),
            provider.as_str()
        );
        let turn_started = Instant::now();
        let mut command = match provider {
            RecordingProvider::Claude => {
                // Claude's print mode does not expose a portable resume API. Run
                // each recorded prompt as a fresh provider turn while keeping
                // all resulting hook events in the replay archive.
                let mut command = Command::new("claude");
                command.args(["-p", prompt, "--output-format", "json"]);
                command
            }
            RecordingProvider::Codex if index == 0 => {
                let mut command = Command::new("codex");
                command.args(["exec", prompt]);
                command
            }
            RecordingProvider::Codex => {
                let session_id = provider_session_id.as_deref().with_context(|| {
                    "Codex replay could not find the provider session ID from the first turn"
                })?;
                let mut command = Command::new("codex");
                command.args(["exec", "resume", session_id, prompt]);
                command
            }
            RecordingProvider::Gemini => {
                let mut command = Command::new("gemini");
                command.args(["-p", prompt, "--output-format", "stream-json"]);
                command
            }
            RecordingProvider::Agy => {
                let mut command = Command::new("agy");
                command.args(["-p", prompt, "--output-format", "stream-json"]);
                command
            }
        };
        let output = command
            .current_dir(project_root)
            .env(REPLAY_SESSION_ENV, replay_session_id)
            .env("HOME", &isolated_home)
            .env("CODEX_HOME", &isolated_codex_home)
            .env("XDG_CONFIG_HOME", isolated_home.join(".config"))
            .env("USERPROFILE", &isolated_home)
            .env("APPDATA", &isolated_app_data)
            .env("LOCALAPPDATA", &isolated_local_app_data)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("failed to launch {} replay process", provider.as_str()))?;

        if index == 0 && matches!(provider, RecordingProvider::Codex) {
            provider_session_id = extract_codex_session_id(&output.stderr);
        }
        execution.success = output.status.success();
        execution.exit_code = output.status.code();
        execution
            .stdout
            .push_str(&String::from_utf8_lossy(&output.stdout));
        execution
            .stderr
            .push_str(&String::from_utf8_lossy(&output.stderr));
        eprintln!(
            "lint-ai replay: finished turn {}/{} in {:.1}s ({})",
            index + 1,
            prompts.len(),
            turn_started.elapsed().as_secs_f64(),
            if output.status.success() {
                "ok"
            } else {
                "failed"
            }
        );
        if !output.status.success() {
            break;
        }
    }
    Ok(execution)
}

fn extract_codex_session_id(stderr: &[u8]) -> Option<String> {
    let stderr = String::from_utf8_lossy(stderr);
    stderr.lines().find_map(|line| {
        let (_, value) = line.split_once("session id:")?;
        let value = value.trim();
        (!value.is_empty()).then(|| value.to_string())
    })
}

pub fn promote_recorded_session(
    provider: RecordingProvider,
    project_root: &Path,
    archive_override: Option<&Path>,
    session_id: &str,
) -> Result<SessionImportReport> {
    let archive_root = archive_override
        .map(Path::to_path_buf)
        .unwrap_or_else(|| session_root(provider, project_root));
    let session_dir = archive_root.join(safe_component(session_id));
    let events_path = session_dir.join("events.jsonl");
    let content = fs::read_to_string(&events_path)
        .with_context(|| format!("failed to read recorded session {}", events_path.display()))?;
    let group_id = format!("{}-session:{}", provider.as_str(), session_id);
    let memory_root = project_root
        .join(".lint-ai")
        .join(format!("{}-memory", provider.as_str()));
    let options = PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 3,
            routing_strategy: SegmentRoutingStrategy::LocalDistinctiveness,
        },
        ..PipelineOptions::default()
    };
    let mut store = IndexStore::at_path(&memory_root, options)?;
    let mut imported_document_ids = Vec::new();
    let mut skipped_events = 0;

    for line in content.lines() {
        let event: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(_) => {
                skipped_events += 1;
                continue;
            }
        };
        let kind = event.get("kind").and_then(Value::as_str).unwrap_or("event");
        if matches!(kind, "session_start" | "session_end") {
            skipped_events += 1;
            continue;
        }
        let sequence = event.get("sequence").and_then(Value::as_u64).unwrap_or(0);
        let doc_id = format!(
            "session-recording:{}:{}:{}",
            provider.as_str(),
            safe_component(session_id),
            sequence
        );
        let mut filters = BTreeMap::new();
        filters.insert("provider".to_string(), provider.as_str().to_string());
        filters.insert("session_id".to_string(), session_id.to_string());
        filters.insert("event_kind".to_string(), kind.to_string());
        filters.insert("source_type".to_string(), "recorded-session".to_string());
        let document = SourceDocument {
            source: format!(
                "lint-ai://{}/session/{}/{}",
                provider.as_str(),
                session_id,
                sequence
            ),
            concept: "recorded-session".to_string(),
            headings: vec![kind.to_string()],
            links: vec![],
            timestamp: event
                .get("timestamp")
                .and_then(Value::as_str)
                .map(str::to_string),
            doc_length: event_content(&event).len(),
            author_agent: Some(provider.as_str().to_string()),
            filters,
            group_id: Some(group_id.clone()),
            doc_id: doc_id.clone(),
            content: event_content(&event),
        };
        store.upsert(document);
        imported_document_ids.push(doc_id);
    }
    store.refresh()?;
    Ok(SessionImportReport {
        session_id: session_id.to_string(),
        group_id,
        imported_document_ids,
        skipped_events,
    })
}

fn event_content(event: &Value) -> String {
    let kind = event.get("kind").and_then(Value::as_str).unwrap_or("event");
    let payload = event.get("payload").cloned().unwrap_or(Value::Null);
    format!(
        "Recorded session event ({kind}):\n{}",
        serde_json::to_string_pretty(&payload).unwrap_or_default()
    )
}

pub fn set_recording_state(
    provider: RecordingProvider,
    project_root: &Path,
    enabled: bool,
) -> Result<Value> {
    let root = session_root(provider, project_root);
    fs::create_dir_all(&root)?;
    let state = serde_json::json!({
        "schema_version": 1,
        "provider": provider.as_str(),
        "project_root": project_root,
        "enabled": enabled,
        "updated_at": timestamp(),
    });
    write_manifest(&root.join("recording.json"), &state)?;
    Ok(state)
}

pub fn recording_state(provider: RecordingProvider, project_root: &Path) -> Result<Value> {
    let path = session_root(provider, project_root).join("recording.json");
    if !path.exists() {
        return Ok(serde_json::json!({
            "provider": provider.as_str(),
            "project_root": project_root,
            "enabled": false,
        }));
    }
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

pub fn set_lint_ai_state(
    provider: RecordingProvider,
    project_root: &Path,
    enabled: bool,
) -> Result<Value> {
    let root = memory_root(provider, project_root);
    fs::create_dir_all(&root)?;
    let state = serde_json::json!({
        "schema_version": 1,
        "provider": provider.as_str(),
        "project_root": project_root,
        "enabled": enabled,
        "updated_at": timestamp(),
    });
    write_manifest(&root.join("integration.json"), &state)?;
    Ok(state)
}

pub fn lint_ai_enabled(provider: RecordingProvider, project_root: &Path) -> Result<bool> {
    let path = memory_root(provider, project_root).join("integration.json");
    if !path.exists() {
        return Ok(true);
    }
    Ok(
        serde_json::from_str::<Value>(&fs::read_to_string(path)?)?["enabled"]
            .as_bool()
            .unwrap_or(true),
    )
}

pub fn record_event_if_enabled(
    provider: RecordingProvider,
    project_root: &Path,
    session_id: &str,
    event_name: &str,
    payload: Value,
) -> Result<()> {
    let state = recording_state(provider, project_root)?;
    if !state["enabled"].as_bool().unwrap_or(false) {
        return Ok(());
    }
    let target_session_id = std::env::var(REPLAY_SESSION_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| session_id.to_string());
    let mut payload = payload;
    if target_session_id != session_id {
        if let Some(object) = payload.as_object_mut() {
            object.insert(
                "provider_session_id".to_string(),
                Value::String(session_id.to_string()),
            );
        }
    }
    record_event(
        provider,
        &session_root(provider, project_root),
        project_root,
        &target_session_id,
        event_name,
        payload,
    )
}

fn session_root(provider: RecordingProvider, project_root: &Path) -> PathBuf {
    project_root
        .join(".lint-ai")
        .join(format!("{}-sessions", provider.as_str()))
}

fn memory_root(provider: RecordingProvider, project_root: &Path) -> PathBuf {
    project_root
        .join(".lint-ai")
        .join(format!("{}-memory", provider.as_str()))
}

impl RecordingProvider {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Codex => "codex",
            Self::Gemini => "gemini-cli",
            Self::Agy => "agy",
        }
    }
}

pub fn record_event(
    provider: RecordingProvider,
    session_root: &Path,
    project_root: &Path,
    session_id: &str,
    event_name: &str,
    payload: Value,
) -> Result<()> {
    let provider_name = provider.as_str();
    let session_key = safe_component(session_id);
    let session_dir = session_root.join(session_key);
    fs::create_dir_all(&session_dir)?;
    let lock = acquire_lock(&session_dir.join(".lock"))?;

    if event_name.eq_ignore_ascii_case("SessionStart") {
        mark_interrupted_sessions(session_root, project_root, session_id)?;
    }

    let events_path = session_dir.join("events.jsonl");
    let manifest_path = session_dir.join("manifest.json");
    if !manifest_path.exists() {
        write_manifest(
            &manifest_path,
            &serde_json::json!({
                "schema_version": 1,
                "session_id": session_id,
                "provider": provider_name,
                "project_root": project_root,
                "started_at": timestamp(),
                "ended_at": Value::Null,
                "status": "active",
                "recording_mode": "capture-only",
                "event_count": 0,
                "redaction_policy": "default-v1"
            }),
        )?;
    }

    let sequence = count_complete_lines(&events_path)? + 1;
    let mut redactions = Vec::new();
    let bounded_payload = redact_and_bound(payload, &mut redactions, "payload");
    let mut usage = extract_usage(&bounded_payload);
    if let Some(source) = bounded_payload.get("source").and_then(Value::as_str) {
        if let Some(usage_object) = usage.as_mut().and_then(Value::as_object_mut) {
            usage_object.insert("source".to_string(), Value::String(source.to_string()));
        }
    }
    let mut event = serde_json::json!({
        "schema_version": 1,
        "sequence": sequence,
        "event_id": format!("{}:{}:{}", session_id, event_name, sequence),
        "timestamp": timestamp(),
        "kind": normalized_kind(event_name),
        "provider": provider_name,
        "session_id": session_id,
        "payload": bounded_payload,
        "redactions": redactions,
    });
    if let Some(usage) = usage {
        event["usage"] = usage;
    }
    if serde_json::to_vec(&event)?.len() > MAX_EVENT_BYTES {
        event["payload"] = serde_json::json!({
            "truncated": true,
            "summary": "event exceeded the maximum recording size"
        });
        event["truncated"] = Value::Bool(true);
    }

    let encoded = serde_json::to_vec(&event)?;
    if encoded.len() > MAX_EVENT_BYTES {
        return Err(anyhow::anyhow!(
            "recorded event remains too large after bounding"
        ));
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&events_path)
        .with_context(|| format!("failed to open {}", events_path.display()))?;
    restrict_file_permissions(&file)?;
    file.write_all(&encoded)?;
    file.write_all(b"\n")?;
    file.flush()?;

    let terminal = event_name.eq_ignore_ascii_case("SessionEnd");
    let mut manifest: Value = serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;
    manifest["event_count"] = Value::from(sequence);
    manifest["last_event"] = Value::String(event_name.to_string());
    manifest["last_event_at"] = Value::String(timestamp());
    if terminal {
        manifest["ended_at"] = Value::String(timestamp());
        manifest["status"] = Value::String("completed".to_string());
    }
    write_manifest(&manifest_path, &manifest)?;
    drop(lock);
    Ok(())
}

fn mark_interrupted_sessions(
    session_root: &Path,
    project_root: &Path,
    current_session_id: &str,
) -> Result<()> {
    let current_session_key = safe_component(current_session_id);
    let entries = match fs::read_dir(session_root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir()
            || path.file_name().and_then(|name| name.to_str()) == Some(current_session_key.as_str())
        {
            continue;
        }
        let manifest_path = path.join("manifest.json");
        let Ok(contents) = fs::read_to_string(&manifest_path) else {
            continue;
        };
        let Ok(mut manifest) = serde_json::from_str::<Value>(&contents) else {
            continue;
        };
        let same_project = manifest
            .get("project_root")
            .and_then(Value::as_str)
            .map(|value| value == project_root.to_string_lossy())
            .unwrap_or(false);
        let last_activity = manifest
            .get("last_event_at")
            .or_else(|| manifest.get("started_at"))
            .and_then(Value::as_str)
            .and_then(timestamp_seconds);
        let stale = last_activity
            .map(|seconds| {
                current_timestamp_seconds().saturating_sub(seconds)
                    >= INTERRUPTED_SESSION_STALE_SECONDS
            })
            .unwrap_or(false);
        if !same_project
            || manifest.get("status").and_then(Value::as_str) != Some("active")
            || !stale
        {
            continue;
        }
        let ended_at = timestamp();
        manifest["status"] = Value::String("interrupted".to_string());
        manifest["ended_at"] = Value::String(ended_at.clone());
        manifest["interrupted_at"] = Value::String(ended_at);
        manifest["status_reason"] = Value::String(
            "a later session started before this session emitted SessionEnd".to_string(),
        );
        write_manifest(&manifest_path, &manifest)?;
    }
    Ok(())
}

fn normalized_kind(event_name: &str) -> &'static str {
    match event_name {
        "SessionStart" => "session_start",
        "UserPromptSubmit" | "UserPromptExpansion" => "user_prompt",
        "PreToolUse" => "tool_call",
        "PostToolUse" => "tool_result",
        "PostToolUseFailure" => "tool_error",
        "PreCompact" | "PostCompact" => "compaction",
        "SubagentStart" => "subagent_start",
        "SubagentStop" => "subagent_stop",
        "Stop" => "assistant_message",
        "TurnUsage" => "turn_usage",
        "SessionEnd" => "session_end",
        _ => "provider_event",
    }
}

/// Read the provider transcript at a turn boundary and append the latest
/// authoritative usage object when the provider exposes one there. This is a
/// best-effort side channel: missing usage remains missing, never zero.
pub fn record_transcript_usage_if_available(
    provider: RecordingProvider,
    project_root: &Path,
    session_id: &str,
    transcript_path: Option<&Path>,
    turn_id: Option<&str>,
    source: &str,
) -> Result<bool> {
    if !recording_state(provider, project_root)?["enabled"]
        .as_bool()
        .unwrap_or(false)
    {
        return Ok(false);
    }
    let Some(path) = transcript_path else {
        return Ok(false);
    };
    if !path.is_file() {
        return Ok(false);
    }

    let mut file = fs::File::open(path)
        .with_context(|| format!("failed to open provider transcript {}", path.display()))?;
    let mut bytes = Vec::new();
    std::io::Read::by_ref(&mut file)
        .take(MAX_TRANSCRIPT_BYTES + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read provider transcript {}", path.display()))?;
    if bytes.len() as u64 > MAX_TRANSCRIPT_BYTES {
        anyhow::bail!(
            "provider transcript {} exceeds the {} byte recording limit",
            path.display(),
            MAX_TRANSCRIPT_BYTES
        );
    }
    let content = String::from_utf8_lossy(&bytes);
    let mut latest_usage = None;
    for line in content.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if let Some(usage) = extract_usage(&value) {
            latest_usage = Some(usage);
        }
    }
    let Some(mut usage) = latest_usage else {
        return Ok(false);
    };
    if let Some(object) = usage.as_object_mut() {
        object.insert("source".to_string(), Value::String(source.to_string()));
    }
    record_event_if_enabled(
        provider,
        project_root,
        session_id,
        "TurnUsage",
        serde_json::json!({
            "turn_id": turn_id,
            "source": source,
            "transcript_path": path,
            "usage": usage,
        }),
    )?;
    Ok(true)
}

fn extract_usage(value: &Value) -> Option<Value> {
    match value {
        Value::Object(object) => {
            for key in ["usage", "token_usage", "tokenUsage"] {
                if let Some(candidate) = object.get(key) {
                    if let Some(normalized) = normalize_usage(candidate, object) {
                        return Some(normalized);
                    }
                }
            }
            object.values().find_map(extract_usage)
        }
        Value::Array(values) => values.iter().find_map(extract_usage),
        _ => None,
    }
}

fn normalize_usage(value: &Value, parent: &Map<String, Value>) -> Option<Value> {
    let object = value.as_object()?;
    let number = |keys: &[&str]| {
        keys.iter()
            .find_map(|key| object.get(*key).and_then(Value::as_u64))
            .or_else(|| {
                keys.iter()
                    .find_map(|key| parent.get(*key).and_then(Value::as_u64))
            })
    };
    let input = number(&["input_tokens", "inputTokens", "prompt_tokens"]);
    let output = number(&["output_tokens", "outputTokens", "completion_tokens"]);
    let cache_creation = number(&["cache_creation_input_tokens", "cacheCreationInputTokens"]);
    let cache_read = number(&["cache_read_input_tokens", "cacheReadInputTokens"]);
    let total = number(&["total_tokens", "totalTokens"]);
    if input.is_none()
        && output.is_none()
        && cache_creation.is_none()
        && cache_read.is_none()
        && total.is_none()
    {
        return None;
    }
    Some(serde_json::json!({
        "input_tokens": input,
        "output_tokens": output,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
        "total_tokens": total.or_else(|| Some(input.unwrap_or(0) + output.unwrap_or(0))),
        "source": "hook-payload"
    }))
}

fn redact_and_bound(value: Value, redactions: &mut Vec<String>, path: &str) -> Value {
    match value {
        Value::Object(object) => {
            let mut output = Map::new();
            for (key, value) in object {
                let child_path = format!("{path}.{key}");
                if is_sensitive_key(&key) {
                    redactions.push(child_path);
                    output.insert(key, Value::String("[REDACTED]".to_string()));
                } else {
                    output.insert(key, redact_and_bound(value, redactions, &child_path));
                }
            }
            Value::Object(output)
        }
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .take(MAX_ARRAY_ITEMS)
                .enumerate()
                .map(|(index, value)| {
                    redact_and_bound(value, redactions, &format!("{path}[{index}]"))
                })
                .collect(),
        ),
        Value::String(value) => {
            if looks_sensitive(&value) {
                redactions.push(path.to_string());
                Value::String("[REDACTED]".to_string())
            } else {
                Value::String(truncate(&value, MAX_STRING_BYTES))
            }
        }
        other => other,
    }
}

fn is_sensitive_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    if matches!(
        key.as_str(),
        "input_tokens"
            | "output_tokens"
            | "total_tokens"
            | "prompt_tokens"
            | "completion_tokens"
            | "cache_creation_input_tokens"
            | "cache_read_input_tokens"
            | "inputtokens"
            | "outputtokens"
            | "totaltokens"
    ) {
        return false;
    }
    [
        "api_key",
        "apikey",
        "access_key",
        "authorization",
        "bearer",
        "cookie",
        "credential",
        "client_secret",
        "password",
        "private_key",
        "secret",
        "token",
    ]
    .iter()
    .any(|needle| key.contains(needle))
}

fn looks_sensitive(value: &str) -> bool {
    contains_credential_material(value)
}

/// Credential shapes that carry no surrounding key name, so the key-name checks
/// in the provider redactors miss them when they appear in free text.
pub(crate) fn contains_credential_material(value: &str) -> bool {
    const PREFIXES: [&str; 17] = [
        "sk-",
        "sk_live_",
        "rk_live_",
        "xox",
        "ghp_",
        "gho_",
        "ghu_",
        "ghs_",
        "ghr_",
        "github_pat_",
        "glpat-",
        "AKIA",
        "ASIA",
        "AIza",
        "ya29.",
        "hf_",
        "npm_",
    ];
    if value.contains("-----BEGIN ") || value.contains("Bearer ") {
        return true;
    }
    value
        .split(|character: char| {
            character.is_whitespace()
                || matches!(
                    character,
                    '"' | '\''
                        | ','
                        | ';'
                        | '='
                        | ':'
                        | '('
                        | ')'
                        | '['
                        | ']'
                        | '{'
                        | '}'
                        | '<'
                        | '>'
                )
        })
        .any(|token| {
            PREFIXES
                .iter()
                .any(|prefix| token.starts_with(prefix) && token.len() >= prefix.len() + 8)
                || looks_like_jwt(token)
        })
}

fn looks_like_jwt(token: &str) -> bool {
    let mut parts = token.split('.');
    let (Some(header), Some(payload), Some(signature), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    token.len() >= 40
        && [header, payload, signature].iter().all(|part| {
            !part.is_empty()
                && part
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        })
}

fn truncate(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", value[..end].trim_end())
}

fn safe_component(value: &str) -> String {
    let mut output = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if output.is_empty() {
        output = "unknown-session".to_string();
    }
    output.chars().take(120).collect()
}

fn count_complete_lines(path: &Path) -> Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    Ok(fs::read_to_string(path)?.lines().count())
}

fn write_manifest(path: &Path, value: &Value) -> Result<()> {
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    restrict_path_permissions(&temporary)?;
    fs::rename(temporary, path)?;
    restrict_path_permissions(path)?;
    Ok(())
}

fn restrict_file_permissions(file: &fs::File) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

fn restrict_path_permissions(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

impl Drop for RecordingLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

struct RecordingLock {
    path: PathBuf,
}

fn acquire_lock(path: &Path) -> Result<RecordingLock> {
    for _ in 0..500 {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(_) => {
                return Ok(RecordingLock {
                    path: path.to_path_buf(),
                })
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                thread::sleep(Duration::from_millis(2));
            }
            Err(error) => return Err(error.into()),
        }
    }
    Err(anyhow::anyhow!(
        "timed out waiting for session recording lock"
    ))
}

fn timestamp() -> String {
    format!("unix:{}", current_timestamp_seconds())
}

fn current_timestamp_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn timestamp_seconds(value: &str) -> Option<u64> {
    value.strip_prefix("unix:")?.parse().ok()
}

fn timestamp_compact() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("lint-ai-recording-{name}-{nonce}"));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn records_redacted_event_and_manifest() {
        let root = temp_root("redaction");
        record_event(
            RecordingProvider::Claude,
            &root,
            Path::new("/tmp/project"),
            "session-1",
            "PostToolUse",
            serde_json::json!({
                "tool_name": "Bash",
                "tool_response": {"token": "secret", "output": "ok"}
            }),
        )
        .unwrap();
        let events = fs::read_to_string(root.join("session-1/events.jsonl")).unwrap();
        assert!(events.contains("[REDACTED]"));
        assert!(!events.contains("secret"));
        assert!(events.contains("tool_result"));
        record_event(
            RecordingProvider::Claude,
            &root,
            Path::new("/tmp/project"),
            "session-1",
            "PostToolUse",
            serde_json::json!({
                "tool_response": {"usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}}
            }),
        )
        .unwrap();
        let events = fs::read_to_string(root.join("session-1/events.jsonl")).unwrap();
        assert!(events.contains("\"total_tokens\""));
        let manifest = fs::read_to_string(root.join("session-1/manifest.json")).unwrap();
        assert!(manifest.contains("capture-only"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn tracks_completed_and_interrupted_session_statuses() {
        let root = temp_root("session-status");
        let project = Path::new("/tmp/project");
        record_event(
            RecordingProvider::Codex,
            &root,
            project,
            "active-session",
            "SessionStart",
            serde_json::json!({}),
        )
        .unwrap();
        record_event(
            RecordingProvider::Codex,
            &root,
            project,
            "new-session",
            "SessionStart",
            serde_json::json!({}),
        )
        .unwrap();

        let still_active: Value = serde_json::from_str(
            &fs::read_to_string(root.join("active-session/manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(still_active["status"], "active");

        let stale_manifest_path = root.join("active-session/manifest.json");
        let mut stale_manifest: Value =
            serde_json::from_str(&fs::read_to_string(&stale_manifest_path).unwrap()).unwrap();
        stale_manifest["last_event_at"] = Value::String("unix:0".to_string());
        write_manifest(&stale_manifest_path, &stale_manifest).unwrap();
        record_event(
            RecordingProvider::Codex,
            &root,
            project,
            "recovery-session",
            "SessionStart",
            serde_json::json!({}),
        )
        .unwrap();

        let interrupted: Value = serde_json::from_str(
            &fs::read_to_string(root.join("active-session/manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(interrupted["status"], "interrupted");
        assert!(interrupted["interrupted_at"].is_string());

        record_event(
            RecordingProvider::Codex,
            &root,
            project,
            "new-session",
            "SessionEnd",
            serde_json::json!({}),
        )
        .unwrap();
        let completed: Value = serde_json::from_str(
            &fs::read_to_string(root.join("new-session/manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(completed["status"], "completed");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn replay_workspace_stays_under_the_project_root() {
        let root = temp_root("replay-workspace");
        let workspace = ReplayWorkspace::create(&root).unwrap();
        assert!(workspace
            .path()
            .starts_with(root.join(".lint-ai/replay-workspaces")));
        let path = workspace.path().to_path_buf();
        drop(workspace);
        assert!(!path.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sync_replay_archive_preserves_replay_manifest() {
        let workspace = temp_root("replay-sync-workspace");
        let archive = temp_root("replay-sync-archive");
        let session_id = "replay-1";
        let source = session_root(RecordingProvider::Codex, &workspace).join(session_id);
        fs::create_dir_all(&source).unwrap();
        fs::write(
            source.join("events.jsonl"),
            "{\"kind\":\"session_start\"}\n",
        )
        .unwrap();
        fs::write(source.join("recording.json"), "{\"enabled\":true}\n").unwrap();
        fs::write(source.join("manifest.json"), "{\"status\":\"completed\"}\n").unwrap();

        let destination = archive.join(session_id);
        fs::create_dir_all(&destination).unwrap();
        fs::write(
            destination.join("manifest.json"),
            "{\"run_type\":\"replay\",\"replay_of_session_id\":\"baseline\"}\n",
        )
        .unwrap();

        sync_replay_archive(RecordingProvider::Codex, &workspace, &archive, session_id).unwrap();
        let manifest: Value =
            serde_json::from_str(&fs::read_to_string(destination.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["run_type"], "replay");
        assert_eq!(manifest["replay_of_session_id"], "baseline");
        assert!(destination.join("events.jsonl").is_file());
        assert!(destination.join("recording.json").is_file());
        fs::remove_dir_all(workspace).unwrap();
        fs::remove_dir_all(archive).unwrap();
    }

    #[test]
    fn records_latest_transcript_usage_only_when_recording_is_enabled() {
        let root = temp_root("transcript-usage");
        let transcript = root.join("provider.jsonl");
        fs::write(
            &transcript,
            concat!(
                "{\"type\":\"assistant\",\"usage\":{\"input_tokens\":10,\"output_tokens\":4,\"cache_read_input_tokens\":2}}\n",
                "{\"type\":\"assistant\",\"usage\":{\"input_tokens\":22,\"output_tokens\":7,\"total_tokens\":29}}\n"
            ),
        )
        .unwrap();

        assert!(!record_transcript_usage_if_available(
            RecordingProvider::Claude,
            &root,
            "session-off",
            Some(&transcript),
            Some("turn-1"),
            "claude-transcript",
        )
        .unwrap());

        set_recording_state(RecordingProvider::Claude, &root, true).unwrap();
        assert!(record_transcript_usage_if_available(
            RecordingProvider::Claude,
            &root,
            "session-1",
            Some(&transcript),
            Some("turn-1"),
            "claude-transcript",
        )
        .unwrap());
        let events =
            fs::read_to_string(root.join(".lint-ai/claude-sessions/session-1/events.jsonl"))
                .unwrap();
        assert!(events.contains("\"kind\":\"turn_usage\""));
        assert!(events.contains("\"source\":\"claude-transcript\""));
        assert!(events.contains("\"total_tokens\":29"));
        assert!(events.contains("\"turn_id\":\"turn-1\""));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn recording_and_memory_controls_round_trip_for_each_provider() {
        for provider in [
            RecordingProvider::Claude,
            RecordingProvider::Codex,
            RecordingProvider::Gemini,
            RecordingProvider::Agy,
        ] {
            let root = temp_root(provider.as_str());
            assert!(!recording_state(provider, &root).unwrap()["enabled"]
                .as_bool()
                .unwrap());
            assert!(lint_ai_enabled(provider, &root).unwrap());

            set_recording_state(provider, &root, true).unwrap();
            set_lint_ai_state(provider, &root, false).unwrap();
            assert!(recording_state(provider, &root).unwrap()["enabled"]
                .as_bool()
                .unwrap());
            assert!(!lint_ai_enabled(provider, &root).unwrap());

            set_recording_state(provider, &root, false).unwrap();
            set_lint_ai_state(provider, &root, true).unwrap();
            assert!(!recording_state(provider, &root).unwrap()["enabled"]
                .as_bool()
                .unwrap());
            assert!(lint_ai_enabled(provider, &root).unwrap());
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn recording_is_independent_from_memory_state() {
        for provider in [
            RecordingProvider::Claude,
            RecordingProvider::Codex,
            RecordingProvider::Gemini,
            RecordingProvider::Agy,
        ] {
            let root = temp_root(provider.as_str());
            set_lint_ai_state(provider, &root, false).unwrap();
            record_event_if_enabled(
                provider,
                &root,
                "session-off-memory",
                "UserPromptSubmit",
                serde_json::json!({"prompt": "still record this"}),
            )
            .unwrap();
            let events = root
                .join(".lint-ai")
                .join(format!("{}-sessions", provider.as_str()))
                .join("session-off-memory/events.jsonl");
            assert!(!events.exists());

            set_recording_state(provider, &root, true).unwrap();
            record_event_if_enabled(
                provider,
                &root,
                "session-on-recording",
                "UserPromptSubmit",
                serde_json::json!({"prompt": "record despite memory off"}),
            )
            .unwrap();
            assert!(root
                .join(".lint-ai")
                .join(format!("{}-sessions", provider.as_str()))
                .join("session-on-recording/events.jsonl")
                .exists());
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn promotes_recorded_events_into_memory() {
        let project_root = temp_root("promotion-project");
        let archive_root = temp_root("promotion-archive");
        record_event(
            RecordingProvider::Claude,
            &archive_root,
            &project_root,
            "baseline",
            "UserPromptSubmit",
            serde_json::json!({"prompt": "remember the routing decision"}),
        )
        .unwrap();
        record_event(
            RecordingProvider::Claude,
            &archive_root,
            &project_root,
            "baseline",
            "Stop",
            serde_json::json!({"response": "Use IndexStore with segmented routing"}),
        )
        .unwrap();

        let report = promote_recorded_session(
            RecordingProvider::Claude,
            &project_root,
            Some(&archive_root),
            "baseline",
        )
        .unwrap();
        assert_eq!(report.session_id, "baseline");
        assert_eq!(report.imported_document_ids.len(), 2);
        assert!(project_root.join(".lint-ai/claude-memory").exists());
        fs::remove_dir_all(project_root).unwrap();
        fs::remove_dir_all(archive_root).unwrap();
    }

    #[test]
    fn extracts_all_user_prompts_for_replay() {
        let root = temp_root("prompt");
        let events = root.join("events.jsonl");
        fs::write(
            &events,
            concat!(
                "{\"kind\":\"session_start\",\"payload\":{}}\n",
                "{\"kind\":\"user_prompt\",\"payload\":{\"prompt\":\"Fix routing\"}}\n",
                "{\"kind\":\"assistant_message\",\"payload\":{}}\n",
                "{\"kind\":\"user_prompt\",\"payload\":{\"prompt\":\"Run the tests\"}}\n"
            ),
        )
        .unwrap();
        assert_eq!(
            recorded_prompts(&events).unwrap(),
            vec!["Fix routing".to_string(), "Run the tests".to_string()]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn extracts_codex_session_id_from_provider_output() {
        let output = b"model: gpt-5\nsession id: 123e4567-e89b-12d3-a456-426614174000\n";
        assert_eq!(
            extract_codex_session_id(output).as_deref(),
            Some("123e4567-e89b-12d3-a456-426614174000")
        );
    }

    #[test]
    fn excludes_synthetic_replay_completion_from_provider_event_count() {
        let root = temp_root("replay-events");
        let events = root.join("events.jsonl");
        fs::write(
            &events,
            concat!(
                "{\"kind\":\"provider_event\",\"payload\":{\"replay_of_session_id\":\"baseline\"}}\n",
                "{\"kind\":\"session_start\",\"payload\":{}}\n",
                "{\"kind\":\"user_prompt\",\"payload\":{\"prompt\":\"test\"}}\n"
            ),
        )
        .unwrap();
        assert_eq!(provider_event_count(&events).unwrap(), 2);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn missing_replay_events_is_not_recording_complete() {
        let root = temp_root("empty-replay-events");
        let events = root.join("events.jsonl");
        fs::write(
            &events,
            "{\"kind\":\"provider_event\",\"payload\":{\"replay_of_session_id\":\"baseline\"}}\n",
        )
        .unwrap();
        assert_eq!(provider_event_count(&events).unwrap(), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn replay_with_prompts_requires_a_replayed_user_prompt_event() {
        let root = temp_root("incomplete-prompt-replay");
        let events = root.join("events.jsonl");
        fs::write(
            &events,
            "{\"kind\":\"provider_event\",\"payload\":{\"event\":\"stop\"}}\n",
        )
        .unwrap();
        assert!(!replay_recording_complete(&events, 1).unwrap());
        fs::remove_dir_all(root).unwrap();
    }
}
