use crate::adapters::{
    apply_ignore_paths, build_project_graph, graph_to_source_documents, AdapterInput,
};
use crate::integrations::mcp_index;
use crate::integrations::mcp_tools;
use crate::integrations::mcp_transport;
use crate::integrations::mcp_transport::{
    JsonRpcError, JsonRpcRequest, JsonRpcResponse, ToolDefinition,
};
use crate::integrations::session_recording::{
    lint_ai_enabled, recording_state, set_lint_ai_state, set_recording_state, RecordingProvider,
};
use crate::pipeline::IndexStore;
#[cfg(test)]
use crate::pipeline::{MemoryIndexLayout, PipelineOptions};
use crate::query_plan::PreparedQuery;
#[cfg(test)]
use crate::segments::SegmentRoutingStrategy;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::env;
use std::fs;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub mod hooks;

const SERVER_NAME: &str = "lint-ai";
const DEFAULT_QUERY_TOP_K: usize = 5;

/// Muse hook entry marker: the validated schema uses matcher groups whose
/// inner hooks run `<exe> --muse-hook <event>` as a single shell string.
const HOOK_MARKER: &str = "--muse-hook";
/// Timeout seconds for hook commands (validated field name: `timeout`).
const HOOK_TIMEOUT_SECS: u64 = 60;

/// Muse Code requires this schema marker in `settings.json`; a file without it
/// fails every `muse` command with "malformed settings file". The installer
/// preserves an existing value and inserts `1` when the file is new or lacks it.
const SETTINGS_SCHEMA_VERSION: i64 = 1;

#[derive(Debug, Clone)]
pub struct MuseServerOptions<'a> {
    pub max_bytes: usize,
    pub max_files: usize,
    pub max_depth: usize,
    pub max_total_bytes: usize,
    pub ignore_paths: &'a [String],
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct McpServerEntry {
    transport: String,
    command: String,
    args: Vec<String>,
    enabled: bool,
}

struct MuseMcp {
    root: PathBuf,
    max_bytes: usize,
    max_files: usize,
    max_depth: usize,
    max_total_bytes: usize,
    ignore_paths: Vec<String>,
    workspace_watcher: Option<mcp_index::WorkspaceWatcher>,
    store: Mutex<Option<IndexStore>>,
}

/// Merge the `lint-ai` MCP server entry into Muse Code's `settings.json`.
///
/// Muse Code reads MCP servers only from the user/global `settings.json`
/// (`~/.config/muse/settings.json`); there is no documented project-scoped MCP
/// location. The entry pins the project root in its args so one install maps to
/// one project. Every existing key — including `schema_version`, hooks, and
/// other MCP servers — is preserved untouched.
///
/// The key is `mcpServers` (camelCase). Earlier installers wrote the legacy
/// `mcp_servers` key, which Muse silently ignores — and if both keys exist,
/// Muse drops the whole MCP member, disabling every configured server. The
/// installer therefore migrates any legacy `mcp_servers` entries into
/// `mcpServers` before writing.
pub fn install_user_config(root: &Path, config_path: Option<&Path>) -> Result<PathBuf> {
    let config_path = match config_path {
        Some(path) => path.to_path_buf(),
        None => default_muse_config_path()?,
    };
    // Still canonicalized, to reject a path that does not exist before writing.
    let _root = root
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", root.display()))?;
    let executable = env::current_exe()
        .context("failed to locate lint-ai executable; refusing PATH-based installation")?
        .to_string_lossy()
        .into_owned();
    let mut settings = read_json_object(&config_path).with_context(|| {
        format!(
            "failed to parse Muse Code settings at {}; refusing to overwrite a malformed file",
            config_path.display()
        )
    })?;

    settings
        .entry("schema_version".to_string())
        .or_insert_with(|| json!(SETTINGS_SCHEMA_VERSION));

    // Migrate the legacy snake_case key into the canonical camelCase one.
    // `mcpServers` entries win on conflict; the legacy key is always removed
    // so Muse never sees both keys at once.
    if let Some(legacy) = settings.remove("mcp_servers") {
        let legacy = legacy
            .as_object()
            .context("Muse Code settings 'mcp_servers' must be an object")?;
        let canonical = settings
            .entry("mcpServers".to_string())
            .or_insert_with(|| json!({}));
        let canonical = canonical
            .as_object_mut()
            .context("Muse Code settings 'mcpServers' must be an object")?;
        for (name, entry) in legacy {
            canonical
                .entry(name.clone())
                .or_insert_with(|| entry.clone());
        }
    }

    let mcp_servers = settings
        .entry("mcpServers".to_string())
        .or_insert_with(|| json!({}));
    let mcp_servers = mcp_servers
        .as_object_mut()
        .context("Muse Code settings 'mcpServers' must be an object")?;
    // Pin the project root explicitly so the MCP server does not depend on the
    // client's working directory (which may be the user's home directory).
    let entry = McpServerEntry {
        transport: "stdio".to_string(),
        command: executable,
        args: vec![
            "--muse-serve".to_string(),
            root.to_string_lossy().into_owned(),
        ],
        enabled: true,
    };
    mcp_servers.insert(
        "lint-ai".to_string(),
        serde_json::to_value(&entry).context("failed to serialize MCP server entry")?,
    );

    write_json_object(&config_path, &settings).context("failed to write Muse Code settings")?;
    Ok(config_path)
}

/// Install capture-only session hooks into Muse Code's `settings.json`.
///
/// Muse reads hooks from the same user/global `settings.json` as the MCP
/// servers. Each event gets a matcher group that runs
/// `lint-ai --muse-hook <event>` as a single shell command string (the only
/// command shape Muse accepts). The hooks record the session lifecycle and
/// tool use into Lint-AI's session store; they never inject memory into the
/// context and never block the session. Existing user hooks are preserved and
/// the install is idempotent.
pub fn install_hook_settings(root: &Path, config_path: Option<&Path>) -> Result<PathBuf> {
    let config_path = match config_path {
        Some(path) => path.to_path_buf(),
        None => default_muse_config_path()?,
    };
    let _root = root
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", root.display()))?;
    let executable = env::current_exe()
        .context("failed to locate lint-ai executable; refusing PATH-based installation")?
        .to_string_lossy()
        .into_owned();
    let mut settings = read_json_object(&config_path).with_context(|| {
        format!(
            "failed to parse Muse Code settings at {}; refusing to overwrite a malformed file",
            config_path.display()
        )
    })?;

    settings
        .entry("schema_version".to_string())
        .or_insert_with(|| json!(SETTINGS_SCHEMA_VERSION));

    let hooks = settings
        .entry("hooks".to_string())
        .or_insert_with(|| json!({}));
    let hooks = hooks
        .as_object_mut()
        .context("Muse Code settings 'hooks' must be an object")?;

    for kind in hooks::HOOK_EVENTS {
        let event_name = kind.event_name();
        let entries = hooks
            .entry(event_name.to_string())
            .or_insert_with(|| json!([]));
        let entries = entries
            .as_array_mut()
            .with_context(|| format!("Muse Code hook event '{event_name}' must be an array"))?;
        // Remove our own previous entries so reinstalls stay idempotent;
        // every other matcher group is preserved untouched.
        entries.retain(|entry| !contains_lint_ai_hook(entry));
        entries.push(json!({
            "matcher": "*",
            "hooks": [{
                "type": "command",
                "command": format!("{} {HOOK_MARKER} {}", shell_quote(&executable), kind.cli_name()),
                "timeout": HOOK_TIMEOUT_SECS
            }]
        }));
    }

    write_json_object(&config_path, &settings).context("failed to write Muse Code settings")?;
    Ok(config_path)
}

fn contains_lint_ai_hook(entry: &Value) -> bool {
    entry
        .get("hooks")
        .and_then(Value::as_array)
        .is_some_and(|hooks| {
            hooks.iter().any(|hook| {
                hook.get("command")
                    .and_then(Value::as_str)
                    .is_some_and(|command| command.contains(HOOK_MARKER))
            })
        })
}

/// Quote an executable path for embedding in a shell command string.
fn shell_quote(path: &str) -> String {
    if path
        .chars()
        .all(|c| c.is_alphanumeric() || matches!(c, '/' | '.' | '-' | '_' | '+'))
    {
        return path.to_string();
    }
    format!("'{}'", path.replace('\'', "'\\''"))
}

/// Delimiters so the block can be found and replaced without touching the rest of
/// a file the user also writes in.
const POLICY_START: &str = "<!-- lint-ai:memory-policy:start -->";
const POLICY_END: &str = "<!-- lint-ai:memory-policy:end -->";

/// Writes the memory policy into the project's `AGENTS.md`. Muse Code prefers
/// `AGENTS.md` over `CLAUDE.md` when both exist, so the same delimited block
/// the Codex installer writes applies here unchanged. Without it the MCP server
/// exists but the agent is never told to consult it.
///
/// `AGENTS.md` belongs to the user, so the block is delimited and replaced in
/// place; everything outside it is left exactly as it was, and an existing file
/// is never clobbered.
pub fn install_memory_policy(root: &Path) -> Result<PathBuf> {
    let root = root
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", root.display()))?;
    let path = root.join("AGENTS.md");
    let block = include_str!("agents_md.md").trim_end();

    let next = match fs::read_to_string(&path) {
        Ok(existing) => match (existing.find(POLICY_START), existing.find(POLICY_END)) {
            (Some(start), Some(end)) if end > start => {
                let mut updated = String::with_capacity(existing.len());
                updated.push_str(&existing[..start]);
                updated.push_str(block);
                updated.push_str(&existing[end + POLICY_END.len()..]);
                updated
            }
            _ => {
                let mut updated = existing.trim_end().to_string();
                if !updated.is_empty() {
                    updated.push_str("\n\n");
                }
                updated.push_str(block);
                updated.push('\n');
                updated
            }
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => format!("{block}\n"),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", path.display()))
        }
    };

    write_text_object(&path, &next)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(path)
}

pub fn run_server(root: &Path, options: MuseServerOptions<'_>) -> Result<()> {
    mcp_index::trace_event("server-start");
    let mcp = MuseMcp {
        root: root.to_path_buf(),
        max_bytes: options.max_bytes,
        max_files: options.max_files,
        max_depth: options.max_depth,
        max_total_bytes: options.max_total_bytes,
        ignore_paths: options.ignore_paths.to_vec(),
        workspace_watcher: Some(mcp_index::WorkspaceWatcher::new(
            root,
            options.ignore_paths,
        )?),
        store: Mutex::new(None),
    };
    mcp.serve()
}

impl MuseMcp {
    fn store(&self) -> Result<std::sync::MutexGuard<'_, Option<IndexStore>>> {
        let mut store = self
            .store
            .lock()
            .map_err(|_| anyhow::anyhow!("MCP index lock poisoned"))?;
        if self
            .workspace_watcher
            .as_ref()
            .is_some_and(mcp_index::WorkspaceWatcher::take_change)
        {
            *store = None;
        }
        if store.is_none() {
            let graph = build_project_graph(&AdapterInput {
                root: &self.root,
                max_bytes: self.max_bytes,
                max_files: self.max_files,
                max_depth: self.max_depth,
                max_total_bytes: self.max_total_bytes,
            })?;
            let graph = apply_ignore_paths(graph, &self.ignore_paths);
            let documents = graph_to_source_documents(&graph);
            let root = self.root.clone();
            *store = Some(mcp_index::open_workspace_memory_store(
                &root,
                mcp_index::SHARED_MEMORY_DIR,
                &self.ignore_paths,
                || Ok(documents),
            )?);
        }
        Ok(store)
    }

    fn serve(self) -> Result<()> {
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut reader = BufReader::new(stdin.lock());
        let mut writer = stdout.lock();

        while let Some((request, line_framed)) = mcp_transport::read_request(&mut reader)? {
            mcp_index::trace_event(&format!("request:{}", request.method));
            if request.id.is_none() {
                continue;
            }
            let response = self.handle_request(request)?;
            mcp_transport::write_response(&mut writer, &response, line_framed)?;
            mcp_index::trace_event("response-written");
        }
        Ok(())
    }

    fn handle_request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
        let id = request.id;
        match request.method.as_str() {
            "initialize" => Ok(JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(json!({
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                    "capabilities": {
                        "tools": {
                            "listChanged": false
                        }
                    }
                })),
                error: None,
            }),
            "tools/list" => {
                // Tool definitions are static; the store initializes lazily
                // on the first real tool call (search, list_memories, info).
                Ok(JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: Some(json!({
                        "tools": self.tools(),
                    })),
                    error: None,
                })
            }
            "tools/call" => self.handle_tool_call(id, request.params),
            _ => Ok(JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: format!("unknown method: {}", request.method),
                }),
            }),
        }
    }

    fn handle_tool_call(
        &self,
        id: Option<Value>,
        params: Option<Value>,
    ) -> Result<JsonRpcResponse> {
        let params = params.unwrap_or_else(|| json!({}));
        let tool_name = params
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));

        match tool_name {
            "search" => {
                if let Some(name) = unknown_argument(&arguments, &["query", "top_k", "provider"]) {
                    return Ok(error_response(
                        id,
                        -32602,
                        &format!("unknown search argument: {name}"),
                    ));
                }
                let query = arguments
                    .get("query")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim();
                if query.is_empty() {
                    return Ok(error_response(id, -32602, "query is required"));
                }
                let top_k = arguments
                    .get("top_k")
                    .and_then(Value::as_u64)
                    .unwrap_or(DEFAULT_QUERY_TOP_K as u64)
                    .clamp(1, 20) as usize;
                let filters = match mcp_tools::search_provider_filters(&arguments) {
                    Ok(filters) => filters,
                    Err(message) => return Ok(error_response(id, -32602, &message)),
                };
                let mut store = self.store()?;
                let store = store.as_mut().expect("MCP store initialized");
                mcp_index::sync_memory_documents(
                    &mcp_index::shared_memory_root(&self.root),
                    &mut *store,
                )?;
                let started = std::time::Instant::now();
                let results = store.query_prepared(&PreparedQuery::new(query), top_k, &filters);
                let _ = crate::telemetry::record_project_query(
                    &self.root,
                    started.elapsed().as_millis() as u64,
                    results.is_err(),
                    results.as_ref().is_ok_and(Vec::is_empty),
                );
                let results = results?;
                let payload = mcp_tools::search_results(store, results);
                Ok(JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: Some(json!({
                        "content": [
                            {
                                "type": "text",
                                "text": serde_json::to_string_pretty(&payload)?,
                            }
                        ]
                    })),
                    error: None,
                })
            }
            "list_memories" => {
                if let Some(name) = unknown_argument(&arguments, &["limit"]) {
                    return Ok(error_response(
                        id,
                        -32602,
                        &format!("unknown list_memories argument: {name}"),
                    ));
                }
                let limit = arguments
                    .get("limit")
                    .and_then(Value::as_u64)
                    .unwrap_or(20)
                    .clamp(1, 100) as usize;
                let mut store = self.store()?;
                let store = store.as_mut().expect("MCP store initialized");
                mcp_index::sync_memory_documents(
                    &mcp_index::shared_memory_root(&self.root),
                    &mut *store,
                )?;
                Ok(text_response(
                    id,
                    &serde_json::to_string_pretty(&mcp_tools::list_memories(store, limit))?,
                ))
            }
            "enable_lint_ai" => {
                if let Some(name) = unknown_argument(&arguments, &[]) {
                    return Ok(error_response(
                        id,
                        -32602,
                        &format!("unknown argument: {name}"),
                    ));
                }
                let state = set_lint_ai_state(RecordingProvider::Muse, &self.root, true)?;
                set_recording_state(RecordingProvider::Muse, &self.root, true)?;
                Ok(text_response(id, &serde_json::to_string_pretty(&state)?))
            }
            "disable_lint_ai" => {
                if let Some(name) = unknown_argument(&arguments, &[]) {
                    return Ok(error_response(
                        id,
                        -32602,
                        &format!("unknown argument: {name}"),
                    ));
                }
                let state = set_lint_ai_state(RecordingProvider::Muse, &self.root, false)?;
                Ok(text_response(id, &serde_json::to_string_pretty(&state)?))
            }
            "lint_ai_status" => {
                if let Some(name) = unknown_argument(&arguments, &[]) {
                    return Ok(error_response(
                        id,
                        -32602,
                        &format!("unknown argument: {name}"),
                    ));
                }
                let memory_on = lint_ai_enabled(RecordingProvider::Muse, &self.root)?;
                let recording_on = recording_state(RecordingProvider::Muse, &self.root)?
                    .get("enabled")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let state = json!({
                    "provider": "muse",
                    "enabled": memory_on,
                    "recording_enabled": recording_on,
                    "display": format!(
                        "Lint-AI:{} | Record:{}",
                        if memory_on { "ON" } else { "OFF" },
                        if recording_on { "ON" } else { "OFF" }
                    )
                });
                Ok(text_response(id, &serde_json::to_string_pretty(&state)?))
            }
            "record_session" => {
                if let Some(name) = unknown_argument(&arguments, &["action"]) {
                    return Ok(error_response(
                        id,
                        -32602,
                        &format!("unknown record_session argument: {name}"),
                    ));
                }
                let action = arguments
                    .get("action")
                    .and_then(Value::as_str)
                    .unwrap_or("status");
                let state = match action {
                    "start" => set_recording_state(RecordingProvider::Muse, &self.root, true)?,
                    "stop" => set_recording_state(RecordingProvider::Muse, &self.root, false)?,
                    "status" => recording_state(RecordingProvider::Muse, &self.root)?,
                    _ => {
                        return Ok(error_response(
                            id,
                            -32602,
                            "action must be start, stop, or status",
                        ))
                    }
                };
                Ok(JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: Some(json!({
                        "content": [{"type": "text", "text": serde_json::to_string_pretty(&state)?}]
                    })),
                    error: None,
                })
            }
            "info" => {
                if let Some(name) = unknown_argument(&arguments, &[]) {
                    return Ok(error_response(
                        id,
                        -32602,
                        &format!("unknown info argument: {name}"),
                    ));
                }
                let store = self.store()?;
                let store = store.as_ref().expect("MCP store initialized");
                Ok(JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: Some(json!({
                        "content": [
                            {
                                "type": "text",
                                "text": serde_json::to_string_pretty(&json!({
                                    "root": self.root,
                                    "docs_count": store.source_documents().len(),
                                }))?,
                            }
                        ]
                    })),
                    error: None,
                })
            }
            _ => Ok(error_response(id, -32602, "unknown tool")),
        }
    }

    fn tools(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "search".to_string(),
                description:
                    "Search the indexed workspace and return concise, self-contained relevant memories."
                        .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "top_k": { "type": "integer", "minimum": 1, "maximum": 20, "default": DEFAULT_QUERY_TOP_K },
                        "provider": mcp_tools::provider_argument_schema(),
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }),
            },
            ToolDefinition {
                name: "info".to_string(),
                description: "Return basic information about the indexed workspace.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
            },
            mcp_tools::list_memories_tool_definition(),
            ToolDefinition {
                name: "record_session".to_string(),
                description: "Start, stop, or inspect opt-in local session recording. Recording is capture-only and does not inject memory.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["start", "stop", "status"], "default": "status"}
                    },
                    "additionalProperties": false
                }),
            },
            ToolDefinition {
                name: "enable_lint_ai".to_string(),
                description: "Enable Lint-AI memory retrieval and capture for future hook events. This also turns on session recording by default; use record_session stop to override recording independently.".to_string(),
                input_schema: json!({"type":"object","properties":{},"additionalProperties":false}),
            },
            ToolDefinition {
                name: "disable_lint_ai".to_string(),
                description: "Disable Lint-AI memory retrieval and capture for future hook events. Session recording remains unchanged and can be controlled with record_session.".to_string(),
                input_schema: json!({"type":"object","properties":{},"additionalProperties":false}),
            },
            ToolDefinition {
                name: "lint_ai_status".to_string(),
                description: "Report whether Lint-AI memory behavior is enabled for this project.".to_string(),
                input_schema: json!({"type":"object","properties":{},"additionalProperties":false}),
            },
        ]
    }
}

fn unknown_argument<'a>(arguments: &'a Value, allowed: &[&str]) -> Option<&'a str> {
    arguments
        .as_object()?
        .keys()
        .find(|name| !allowed.contains(&name.as_str()))
        .map(String::as_str)
}

fn error_response(id: Option<Value>, code: i64, message: &str) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: None,
        error: Some(JsonRpcError {
            code,
            message: message.to_string(),
        }),
    }
}

fn text_response(id: Option<Value>, text: &str) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: Some(json!({"content": [{"type": "text", "text": text}]})),
        error: None,
    }
}

fn read_json_object(path: &Path) -> Result<Map<String, Value>> {
    if !path.exists() {
        return Ok(Map::new());
    }
    let value: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    value
        .as_object()
        .cloned()
        .context("Muse Code settings must be a JSON object")
}

fn write_json_object(path: &Path, value: &Map<String, Value>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn write_text_object(path: &Path, value: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, value)?;
    Ok(())
}

/// Muse Code's user settings live at `~/.config/muse/settings.json` (or
/// `$XDG_CONFIG_HOME/muse/settings.json` when XDG_CONFIG_HOME is set).
fn default_muse_config_path() -> Result<PathBuf> {
    let config_dir = if let Some(xdg) = env::var_os("XDG_CONFIG_HOME") {
        PathBuf::from(xdg)
    } else if let Some(home) = env::var_os("HOME") {
        PathBuf::from(home).join(".config")
    } else {
        anyhow::bail!("unable to determine Muse Code config path; set --muse-config");
    };
    Ok(config_dir.join("muse").join("settings.json"))
}

#[cfg(test)]
fn segmented_store_options() -> PipelineOptions {
    PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 3,
            routing_strategy: SegmentRoutingStrategy::LocalDistinctiveness,
        },
        ..PipelineOptions::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::SourceDocument;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should move forward")
            .as_nanos();
        env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("lint-ai-{name}-{nanos}.json"))
    }

    fn temp_dir(name: &str) -> PathBuf {
        let path = temp_path(name).with_extension("");
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn test_mcp(root: PathBuf, documents: Vec<SourceDocument>) -> MuseMcp {
        let mut store = IndexStore::new(segmented_store_options());
        for document in documents {
            store.upsert(document);
        }
        store.refresh().unwrap();
        MuseMcp {
            root,
            max_bytes: 0,
            max_files: 0,
            max_depth: 0,
            max_total_bytes: 0,
            ignore_paths: Vec::new(),
            workspace_watcher: None,
            store: Mutex::new(Some(store)),
        }
    }

    fn call_tool(mcp: &MuseMcp, name: &str, arguments: Value) -> Value {
        let response = mcp
            .handle_request(JsonRpcRequest {
                id: Some(json!(1)),
                method: "tools/call".to_string(),
                params: Some(json!({"name": name, "arguments": arguments})),
            })
            .unwrap();
        let text = response.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        serde_json::from_str(&text).unwrap()
    }

    #[test]
    fn install_user_config_writes_camel_case_mcp_servers() {
        let dir = temp_dir("muse-config-camel");
        let config_path = dir.join("settings.json");
        fs::write(&config_path, r#"{"theme": "light"}"#).unwrap();

        let root = temp_dir("muse-config-camel-root");
        install_user_config(&root, Some(&config_path)).unwrap();

        let settings: Value =
            serde_json::from_str(&fs::read_to_string(&config_path).unwrap()).unwrap();
        // Muse only reads the camelCase key; the legacy key must not exist.
        assert!(settings.get("mcp_servers").is_none());
        let entry = &settings["mcpServers"]["lint-ai"];
        assert_eq!(entry["transport"], json!("stdio"));
        assert!(entry["enabled"].as_bool().unwrap());
        assert!(entry.get("mode").is_none());
        let args = entry["args"].as_array().unwrap();
        assert_eq!(args[0], json!("--muse-serve"));
        assert_eq!(args[1].as_str().unwrap(), root.to_string_lossy());

        fs::remove_dir_all(dir).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn install_user_config_migrates_legacy_mcp_servers_key() {
        let dir = temp_dir("muse-config-migrate");
        let config_path = dir.join("settings.json");
        fs::write(
            &config_path,
            r#"{
                "schema_version": 1,
                "theme": "dark",
                "mcpServers": {"existing": {"transport": "stdio", "command": "existing", "args": []}},
                "mcp_servers": {
                    "other": {"transport": "stdio", "command": "other", "args": []},
                    "existing": {"transport": "stdio", "command": "stale", "args": []}
                }
            }"#,
        )
        .unwrap();

        let root = temp_dir("muse-config-migrate-root");
        install_user_config(&root, Some(&config_path)).unwrap();

        let settings: Value =
            serde_json::from_str(&fs::read_to_string(&config_path).unwrap()).unwrap();
        // The legacy key is gone entirely; Muse drops the whole MCP member
        // when both keys exist, so leaving it would disable every server.
        assert!(settings.get("mcp_servers").is_none());
        let servers = settings["mcpServers"].as_object().unwrap();
        // Legacy entries merge in; canonical entries win on conflict.
        assert!(servers["other"].is_object());
        assert_eq!(servers["existing"]["command"], json!("existing"));
        assert!(servers["lint-ai"].is_object());

        // Reinstalling is idempotent: exactly one lint-ai entry.
        install_user_config(&root, Some(&config_path)).unwrap();
        let settings: Value =
            serde_json::from_str(&fs::read_to_string(&config_path).unwrap()).unwrap();
        assert_eq!(
            settings["mcpServers"]
                .as_object()
                .unwrap()
                .keys()
                .filter(|key| *key == "lint-ai")
                .count(),
            1
        );

        fs::remove_dir_all(dir).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn install_user_config_preserves_existing_settings_and_schema_version() {
        let dir = temp_dir("muse-config-merge");
        let config_path = dir.join("settings.json");
        fs::write(
            &config_path,
            r#"{
                "schema_version": 1,
                "theme": "dark",
                "mcp_servers": {
                    "other": {"transport": "stdio", "command": "other", "args": []}
                }
            }"#,
        )
        .unwrap();

        let root = temp_dir("muse-config-root");
        let written = install_user_config(&root, Some(&config_path)).unwrap();
        assert_eq!(written, config_path);

        let settings: Value =
            serde_json::from_str(&fs::read_to_string(&config_path).unwrap()).unwrap();
        assert_eq!(settings["schema_version"], json!(1));
        assert_eq!(settings["theme"], json!("dark"));
        // The legacy key is migrated, not preserved in place.
        assert!(settings.get("mcp_servers").is_none());
        assert!(settings["mcpServers"]["other"].is_object());
        let entry = &settings["mcpServers"]["lint-ai"];
        assert_eq!(entry["transport"], json!("stdio"));
        assert!(entry.get("mode").is_none());
        assert!(entry["enabled"].as_bool().unwrap());
        let args = entry["args"].as_array().unwrap();
        assert_eq!(args[0], json!("--muse-serve"));
        assert_eq!(args[1].as_str().unwrap(), root.to_string_lossy());

        // Reinstalling is idempotent: exactly one lint-ai entry.
        install_user_config(&root, Some(&config_path)).unwrap();
        let settings: Value =
            serde_json::from_str(&fs::read_to_string(&config_path).unwrap()).unwrap();
        assert_eq!(
            settings["mcpServers"]
                .as_object()
                .unwrap()
                .keys()
                .filter(|key| *key == "lint-ai")
                .count(),
            1
        );

        fs::remove_dir_all(dir).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn install_user_config_inserts_schema_version_when_missing() {
        let dir = temp_dir("muse-config-schema");
        let config_path = dir.join("settings.json");
        fs::write(&config_path, r#"{"theme": "light"}"#).unwrap();

        let root = temp_dir("muse-config-schema-root");
        install_user_config(&root, Some(&config_path)).unwrap();

        let settings: Value =
            serde_json::from_str(&fs::read_to_string(&config_path).unwrap()).unwrap();
        assert_eq!(settings["schema_version"], json!(1));
        assert_eq!(settings["theme"], json!("light"));

        fs::remove_dir_all(dir).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn install_user_config_refuses_malformed_settings() {
        let dir = temp_dir("muse-config-malformed");
        let config_path = dir.join("settings.json");
        fs::write(&config_path, "{not valid json").unwrap();

        let root = temp_dir("muse-config-malformed-root");
        let result = install_user_config(&root, Some(&config_path));
        assert!(result.is_err());
        // The malformed file is left untouched.
        assert_eq!(fs::read_to_string(&config_path).unwrap(), "{not valid json");

        fs::remove_dir_all(dir).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn install_memory_policy_preserves_user_text_and_is_idempotent() {
        let dir = temp_dir("muse-policy");
        let path = dir.join("AGENTS.md");
        fs::write(&path, "# Mine\n\nMy own standing instructions.\n").unwrap();

        install_memory_policy(&dir).unwrap();
        let once = fs::read_to_string(&path).unwrap();
        assert!(once.contains("My own standing instructions."));
        assert_eq!(once.matches(POLICY_START).count(), 1);

        install_memory_policy(&dir).unwrap();
        let twice = fs::read_to_string(&path).unwrap();
        assert_eq!(once, twice);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn install_hook_settings_is_idempotent_and_preserves_user_hooks() {
        let dir = temp_dir("muse-hooks");
        let config_path = dir.join("settings.json");
        fs::write(
            &config_path,
            r#"{
                "schema_version": 1,
                "hooks": {
                    "Stop": [
                        {"matcher": "Bash", "hooks": [{"type": "command", "command": "other-tool"}]}
                    ]
                }
            }"#,
        )
        .unwrap();

        let root = temp_dir("muse-hooks-root");
        install_hook_settings(&root, Some(&config_path)).unwrap();
        install_hook_settings(&root, Some(&config_path)).unwrap();

        let settings: Value =
            serde_json::from_str(&fs::read_to_string(&config_path).unwrap()).unwrap();
        let hooks = settings["hooks"].as_object().unwrap();
        // All seven capture events are wired.
        for event in [
            "SessionStart",
            "UserPromptSubmit",
            "PreToolUse",
            "PostToolUse",
            "PostToolUseFailure",
            "Stop",
            "SessionEnd",
        ] {
            let entries = hooks[event].as_array().unwrap();
            let ours: Vec<&Value> = entries
                .iter()
                .filter(|entry| contains_lint_ai_hook(entry))
                .collect();
            assert_eq!(
                ours.len(),
                1,
                "event {event} should have exactly one lint-ai hook"
            );
            let hook = &ours[0]["hooks"][0];
            assert_eq!(hook["type"], json!("command"));
            let command = hook["command"].as_str().unwrap();
            assert!(
                command.contains("--muse-hook"),
                "hook command must use the --muse-hook marker: {command}"
            );
            assert!(
                !command.contains('[') && !command.contains('{'),
                "command must be a shell string, not an argv array: {command}"
            );
            assert_eq!(hook["timeout"], json!(60));
            assert_eq!(ours[0]["matcher"], json!("*"));
        }
        // The user's own Stop hook survives.
        let stop = hooks["Stop"].as_array().unwrap();
        assert!(stop
            .iter()
            .any(|entry| { entry["hooks"][0]["command"].as_str() == Some("other-tool") }));

        fs::remove_dir_all(dir).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unknown_tool_returns_invalid_params() {
        let root = temp_dir("muse-unknown-tool");
        let mcp = test_mcp(root.clone(), Vec::new());
        let response = mcp
            .handle_request(JsonRpcRequest {
                id: Some(json!(7)),
                method: "tools/call".to_string(),
                params: Some(json!({"name": "nope", "arguments": {}})),
            })
            .unwrap();
        let error = response.error.unwrap();
        assert_eq!(error.code, -32602);
        assert_eq!(response.id, Some(json!(7)));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn search_tool_rejects_empty_query() {
        let root = temp_dir("muse-empty-search");
        let mcp = test_mcp(root.clone(), Vec::new());
        let response = mcp
            .handle_request(JsonRpcRequest {
                id: Some(json!(1)),
                method: "tools/call".to_string(),
                params: Some(json!({"name": "search", "arguments": {"query": "  "}})),
            })
            .unwrap();
        let error = response.error.unwrap();
        assert_eq!(error.code, -32602);
        assert!(error.message.contains("query is required"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn record_session_rejects_invalid_action() {
        let root = temp_dir("muse-bad-action");
        let mcp = test_mcp(root.clone(), Vec::new());
        let response = mcp
            .handle_request(JsonRpcRequest {
                id: Some(json!(1)),
                method: "tools/call".to_string(),
                params: Some(json!({"name": "record_session", "arguments": {"action": "explode"}})),
            })
            .unwrap();
        let error = response.error.unwrap();
        assert_eq!(error.code, -32602);
        assert!(error.message.contains("start, stop, or status"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn list_memories_and_info_return_expected_shapes() {
        let root = temp_dir("muse-shapes");
        let mcp = test_mcp(
            root.clone(),
            vec![SourceDocument {
                doc_id: "doc-1".to_string(),
                source: "decisions.md".to_string(),
                content: "We dropped embeddings because the heuristic backend was faster."
                    .to_string(),
                concept: "Decisions".to_string(),
                group_id: None,
                headings: Vec::new(),
                links: Vec::new(),
                timestamp: None,
                doc_length: 61,
                author_agent: None,
                filters: std::collections::BTreeMap::new(),
            }],
        );
        let list = call_tool(&mcp, "list_memories", json!({"limit": 5}));
        assert!(list["count"].is_u64());
        assert!(list["memories"].is_array());
        let info = call_tool(&mcp, "info", json!({}));
        assert_eq!(info["docs_count"], json!(1));
        assert!(info["root"].is_string());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn search_tool_returns_ranked_hits_with_semantic_status() {
        let root = temp_dir("muse-search");
        let mcp = test_mcp(
            root.clone(),
            vec![SourceDocument {
                doc_id: "doc-1".to_string(),
                source: "decisions.md".to_string(),
                content: "We dropped embeddings because the heuristic backend was faster."
                    .to_string(),
                concept: "Decisions".to_string(),
                group_id: None,
                headings: Vec::new(),
                links: Vec::new(),
                timestamp: None,
                doc_length: 61,
                author_agent: None,
                filters: std::collections::BTreeMap::new(),
            }],
        );

        let result = call_tool(
            &mcp,
            "search",
            json!({"query": "why did we drop embeddings"}),
        );
        let hits = result["results"].as_array().unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].get("semantic_status").is_some());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn lint_ai_status_reports_provider_state() {
        let root = temp_dir("muse-status");
        let mcp = test_mcp(root.clone(), Vec::new());

        let off = call_tool(&mcp, "lint_ai_status", json!({}));
        assert_eq!(off["provider"], json!("muse"));

        call_tool(&mcp, "enable_lint_ai", json!({}));
        let on = call_tool(&mcp, "lint_ai_status", json!({}));
        assert!(on["enabled"].as_bool().unwrap());

        call_tool(&mcp, "disable_lint_ai", json!({}));
        let off_again = call_tool(&mcp, "lint_ai_status", json!({}));
        assert!(!off_again["enabled"].as_bool().unwrap());

        fs::remove_dir_all(root).unwrap();
    }
}
