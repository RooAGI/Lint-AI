pub mod document;
pub mod hooks;

use crate::config::normalize_list;
use crate::graph::{Graph, Tier0Record};
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
#[cfg(test)]
use crate::segments::SegmentRoutingStrategy;
use crate::source::SourceDocument;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;
use toml::map::Map as TomlMap;
use toml::Value as TomlValue;

const SERVER_NAME: &str = "lint-ai";
const DEFAULT_QUERY_TOP_K: usize = 5;
const MCP_STARTUP_TIMEOUT_SECONDS: i64 = 120;
const HOOK_MARKER: &str = "--codex-hook";
const HOOK_EVENTS: &[(&str, &str)] = &[
    ("SessionStart", "session-start"),
    ("UserPromptSubmit", "user-prompt-submit"),
    ("PreToolUse", "pre-tool-use"),
    ("PermissionRequest", "permission-request"),
    ("PostToolUse", "post-tool-use"),
    ("UserPromptExpansion", "user-prompt-expansion"),
    ("PreCompact", "pre-compact"),
    ("PostCompact", "post-compact"),
    ("Stop", "stop"),
    ("SessionEnd", "session-end"),
    ("SubagentStart", "subagent-start"),
    ("SubagentStop", "subagent-stop"),
];

#[derive(Debug, Clone)]
pub struct CodexServerOptions<'a> {
    pub max_bytes: usize,
    pub max_files: usize,
    pub max_depth: usize,
    pub max_total_bytes: usize,
    pub ignore_paths: &'a [String],
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct McpServerEntry {
    command: String,
    args: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CodexConfig {
    #[serde(rename = "mcpServers", default)]
    mcp_servers: HashMap<String, McpServerEntry>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

struct CodexMcp {
    root: PathBuf,
    max_bytes: usize,
    max_files: usize,
    max_depth: usize,
    max_total_bytes: usize,
    ignore_paths: Vec<String>,
    store: Mutex<Option<IndexStore>>,
}

pub fn install_user_config(root: &Path, config_path: Option<&Path>) -> Result<PathBuf> {
    let config_path = match config_path {
        Some(path) => path.to_path_buf(),
        None => default_codex_config_path()?,
    };
    let root = root
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", root.display()))?;
    let executable = env::current_exe()
        .context("failed to locate lint-ai executable; refusing PATH-based installation")?
        .to_string_lossy()
        .into_owned();
    let mut config = if config_path.exists() {
        let current = fs::read_to_string(&config_path)?;
        if current.trim().is_empty() {
            TomlValue::Table(TomlMap::new())
        } else {
            current
                .parse::<TomlValue>()
                .context("failed to parse Codex TOML config")?
        }
    } else {
        TomlValue::Table(TomlMap::new())
    };
    let table = config
        .as_table_mut()
        .context("Codex config must be a TOML table")?;
    let mcp_servers = table
        .entry("mcp_servers".to_string())
        .or_insert_with(|| TomlValue::Table(TomlMap::new()));
    let mcp_servers = mcp_servers
        .as_table_mut()
        .context("Codex config 'mcp_servers' must be a table")?;
    let mut entry = TomlMap::new();
    entry.insert("command".to_string(), TomlValue::String(executable));
    entry.insert(
        "startup_timeout_sec".to_string(),
        TomlValue::Integer(MCP_STARTUP_TIMEOUT_SECONDS),
    );
    entry.insert(
        "args".to_string(),
        TomlValue::Array(vec![
            TomlValue::String("--codex-serve".to_string()),
            TomlValue::String(root.to_string_lossy().into_owned()),
        ]),
    );
    mcp_servers.insert("lint-ai".to_string(), TomlValue::Table(entry));

    // Codex Desktop and newer Codex CLI builds gate lifecycle hooks behind
    // the stable `features.hooks` flag. Preserve all existing feature flags
    // while enabling the hook runtime required by the installed hooks.json.
    let features = table
        .entry("features".to_string())
        .or_insert_with(|| TomlValue::Table(TomlMap::new()));
    let features = features
        .as_table_mut()
        .context("Codex config 'features' must be a table")?;
    features.insert("hooks".to_string(), TomlValue::Boolean(true));

    write_text_object(&config_path, &toml::to_string_pretty(&config)?)
        .context("failed to write Codex config")?;
    Ok(config_path)
}

pub fn install_hook_settings(root: &Path, settings_path: Option<&Path>) -> Result<PathBuf> {
    let settings_path = match settings_path {
        Some(path) => path.to_path_buf(),
        None => default_codex_settings_path()?,
    };
    let _root = root
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", root.display()))?;
    let executable = env::current_exe()
        .context("failed to locate lint-ai executable; refusing PATH-based installation")?
        .to_string_lossy()
        .into_owned();
    let mut settings = read_json_object(&settings_path)?;
    let hooks = settings
        .entry("hooks".to_string())
        .or_insert_with(|| json!({}));
    let hooks = hooks
        .as_object_mut()
        .context("Codex settings 'hooks' must be an object")?;

    for (event_name, hook_name) in HOOK_EVENTS {
        let entries = hooks
            .entry((*event_name).to_string())
            .or_insert_with(|| json!([]));
        let entries = entries
            .as_array_mut()
            .with_context(|| format!("Codex hook event '{event_name}' must be an array"))?;
        entries.retain(|entry| !contains_lint_ai_hook(entry));
        entries.push(json!({
            "hooks": [{
                "type": "command",
                "command": format!("{} {HOOK_MARKER} {hook_name}", shell_quote(&executable))
            }]
        }));
    }

    write_json_object(&settings_path, &settings)?;
    Ok(settings_path)
}

pub fn run_server(root: &Path, options: CodexServerOptions<'_>) -> Result<()> {
    mcp_index::trace_event("server-start");
    let mcp = CodexMcp {
        root: root.to_path_buf(),
        max_bytes: options.max_bytes,
        max_files: options.max_files,
        max_depth: options.max_depth,
        max_total_bytes: options.max_total_bytes,
        ignore_paths: options.ignore_paths.to_vec(),
        store: Mutex::new(None),
    };
    mcp.serve()
}

fn build_graph(
    root: &Path,
    max_bytes: usize,
    max_files: usize,
    max_depth: usize,
    max_total_bytes: usize,
) -> Result<Graph> {
    Graph::build(
        &root.to_string_lossy(),
        max_bytes,
        max_files,
        max_depth,
        max_total_bytes,
    )
}

fn apply_ignores(mut graph: Graph, ignore_paths: &[String]) -> Graph {
    if ignore_paths.is_empty() {
        return graph;
    }
    let ignore = normalize_list(ignore_paths);
    graph.pages.retain(|p| {
        let rel = p.rel_path.to_lowercase();
        !ignore.iter().any(|pat| rel.contains(pat))
    });
    let retained: std::collections::HashSet<String> =
        graph.pages.iter().map(|p| p.rel_path.clone()).collect();
    graph.tier0_records.retain(|r| retained.contains(&r.source));
    graph
}

fn graph_to_source_documents(graph: &Graph) -> Vec<SourceDocument> {
    let mut tier0_by_source = HashMap::<String, &Tier0Record>::new();
    for record in &graph.tier0_records {
        if tier0_by_source
            .insert(record.source.clone(), record)
            .is_some()
        {
            debug_assert!(false, "duplicate Tier0 source: {}", record.source);
        }
    }
    let concept_to_rel: HashMap<String, String> = graph
        .pages
        .iter()
        .map(|p| (p.concept.clone(), p.rel_path.clone()))
        .collect();

    graph
        .pages
        .iter()
        .map(|p| {
            let t0 = tier0_by_source.get(&p.rel_path).copied();
            SourceDocument {
                doc_id: p.rel_path.clone(),
                source: p.rel_path.clone(),
                content: p.content.clone(),
                concept: p.raw_concept.clone(),
                group_id: None,
                filters: std::collections::BTreeMap::new(),
                headings: p.headings.clone(),
                links: p
                    .links
                    .iter()
                    .filter_map(|c| concept_to_rel.get(c).cloned())
                    .collect(),
                timestamp: t0.and_then(|r| r.timestamp.clone()),
                doc_length: t0.map(|r| r.doc_length).unwrap_or(p.content.len()),
                author_agent: t0.and_then(|r| r.author_agent.clone()),
            }
        })
        .collect()
}

impl CodexMcp {
    fn store(&self) -> Result<std::sync::MutexGuard<'_, Option<IndexStore>>> {
        let mut store = self
            .store
            .lock()
            .map_err(|_| anyhow::anyhow!("MCP index lock poisoned"))?;
        if store.is_none() {
            let graph = build_graph(
                &self.root,
                self.max_bytes,
                self.max_files,
                self.max_depth,
                self.max_total_bytes,
            )?;
            let graph = apply_ignores(graph, &self.ignore_paths);
            let documents = graph_to_source_documents(&graph);
            let root = self.root.clone();
            *store = Some(mcp_index::open_persistent_store(
                &root,
                "codex-mcp-index",
                "codex-memory",
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
                let _store = self.store()?;
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
                if let Some(name) = unknown_argument(&arguments, &["query", "top_k"]) {
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
                let started = Instant::now();
                let mut store = self.store()?;
                let store = store.as_mut().expect("MCP store initialized");
                mcp_index::sync_memory_documents(
                    &self.root.join(".lint-ai").join("codex-memory"),
                    &mut *store,
                )?;
                let results = store.query(query, top_k)?;
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                let diagnostics = store.inspection();
                let payload = json!({
                    "query": query,
                    "top_k": top_k,
                    "root": self.root,
                    "docs_count": store.source_documents().len(),
                    "results": results,
                    "timings_ms": { "total_ms": elapsed_ms },
                    "diagnostics": diagnostics,
                });
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
                    &self.root.join(".lint-ai").join("codex-memory"),
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
                let state = set_lint_ai_state(RecordingProvider::Codex, &self.root, true)?;
                set_recording_state(RecordingProvider::Codex, &self.root, true)?;
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
                let state = set_lint_ai_state(RecordingProvider::Codex, &self.root, false)?;
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
                let memory_on = lint_ai_enabled(RecordingProvider::Codex, &self.root)?;
                let recording_on = recording_state(RecordingProvider::Codex, &self.root)?
                    .get("enabled")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let state = json!({
                    "provider": "codex",
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
                    "start" => set_recording_state(RecordingProvider::Codex, &self.root, true)?,
                    "stop" => set_recording_state(RecordingProvider::Codex, &self.root, false)?,
                    "status" => recording_state(RecordingProvider::Codex, &self.root)?,
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
                    "Search the indexed workspace and return ranked results with diagnostics."
                        .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "top_k": { "type": "integer", "minimum": 1, "maximum": 20, "default": DEFAULT_QUERY_TOP_K },
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

/// Render the provider-owned status indicator for Codex integrations that can
/// execute an external status command. Codex's built-in TUI currently accepts
/// only its own fixed status item identifiers, so the installer does not put
/// this command into `tui.status_line` automatically.
pub fn run_status_line() -> Result<()> {
    let mut input = String::new();
    io::stdin().lock().read_to_string(&mut input)?;
    let payload: Value = serde_json::from_str(&input).unwrap_or_default();
    let cwd = payload
        .get("workspace")
        .and_then(|workspace| workspace.get("current_dir"))
        .and_then(Value::as_str)
        .or_else(|| payload.get("cwd").and_then(Value::as_str))
        .unwrap_or(".");
    let root = Path::new(cwd);
    let memory_on = lint_ai_enabled(RecordingProvider::Codex, root)?;
    let recording_on = recording_state(RecordingProvider::Codex, root)?
        .get("enabled")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    println!(
        "Lint-AI:{} | Record:{}",
        if memory_on { "ON" } else { "OFF" },
        if recording_on { "ON" } else { "OFF" }
    );
    Ok(())
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
        .context("Codex settings must be a JSON object")
}

fn write_json_object(path: &Path, value: &Map<String, Value>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn contains_lint_ai_hook(entry: &Value) -> bool {
    entry
        .get("hooks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|hook| hook.get("command").and_then(Value::as_str))
        .any(|command| command.contains(HOOK_MARKER))
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn write_text_object(path: &Path, value: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, value)?;
    Ok(())
}

fn default_codex_config_path() -> Result<PathBuf> {
    if let Some(home) = env::var_os("HOME") {
        return Ok(PathBuf::from(home).join(".codex").join("config.toml"));
    }
    if cfg!(windows) {
        if let Some(profile) = env::var_os("USERPROFILE") {
            return Ok(PathBuf::from(profile).join(".codex").join("config.toml"));
        }
    }
    anyhow::bail!("unable to determine Codex config path; set --codex-config");
}

fn default_codex_settings_path() -> Result<PathBuf> {
    if let Some(home) = env::var_os("HOME") {
        return Ok(PathBuf::from(home).join(".codex").join("hooks.json"));
    }
    if cfg!(windows) {
        if let Some(profile) = env::var_os("USERPROFILE") {
            return Ok(PathBuf::from(profile).join(".codex").join("hooks.json"));
        }
    }
    anyhow::bail!("unable to determine Codex hooks path")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
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

    fn test_mcp(root: PathBuf, documents: Vec<SourceDocument>) -> CodexMcp {
        let mut store = IndexStore::new(segmented_store_options());
        for document in documents {
            store.upsert(document);
        }
        store.refresh().unwrap();
        CodexMcp {
            root,
            max_bytes: 0,
            max_files: 0,
            max_depth: 0,
            max_total_bytes: 0,
            ignore_paths: Vec::new(),
            store: Mutex::new(Some(store)),
        }
    }

    fn call_tool(mcp: &CodexMcp, name: &str, arguments: Value) -> Value {
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
    fn install_user_config_merges_existing_servers() {
        let config_path = temp_path("claude-config");
        fs::write(
            &config_path,
            r#"# keep this comment
title = "keep"

[mcp_servers.existing]
command = "npx"
args = ["-y", "existing"]

[mcp_servers.lint-ai]
command = "old"
args = ["old"]
"#,
        )
        .unwrap();
        let root = env::current_dir().unwrap();

        let written = install_user_config(&root, Some(&config_path)).unwrap();
        assert_eq!(written, config_path);

        let text = fs::read_to_string(&config_path).unwrap();
        let parsed: TomlValue = text.parse().unwrap();
        assert_eq!(parsed["title"].as_str(), Some("keep"));
        assert_eq!(
            parsed["mcp_servers"]["existing"]["command"].as_str(),
            Some("npx")
        );
        assert_eq!(
            parsed["mcp_servers"]["lint-ai"]["command"].as_str(),
            Some(env::current_exe().unwrap().to_string_lossy().as_ref())
        );
        assert_eq!(
            parsed["mcp_servers"]["lint-ai"]["startup_timeout_sec"].as_integer(),
            Some(MCP_STARTUP_TIMEOUT_SECONDS)
        );
        assert_eq!(parsed["features"]["hooks"].as_bool(), Some(true));
        assert_eq!(
            parsed["mcp_servers"]["lint-ai"]["args"]
                .as_array()
                .unwrap()
                .iter()
                .map(TomlValue::as_str)
                .collect::<Option<Vec<_>>>()
                .unwrap(),
            vec!["--codex-serve", root.to_string_lossy().as_ref()]
        );
    }

    #[test]
    fn install_hook_settings_preserves_entries_and_is_idempotent() {
        let settings_path = temp_path("claude-settings");
        fs::write(
            &settings_path,
            r#"{"hooks":{"Stop":[{"matcher":"Bash","hooks":[{"type":"command","command":"other-tool"}]}]},"theme":"dark"}"#,
        )
        .unwrap();
        let root = env::current_dir().unwrap();

        install_hook_settings(&root, Some(&settings_path)).unwrap();
        install_hook_settings(&root, Some(&settings_path)).unwrap();

        let settings: Value =
            serde_json::from_str(&fs::read_to_string(&settings_path).unwrap()).unwrap();
        assert_eq!(settings["theme"], "dark");
        let stop = settings["hooks"]["Stop"].as_array().unwrap();
        assert_eq!(stop.len(), 2);
        assert_eq!(stop[0]["hooks"][0]["command"], "other-tool");
        assert!(stop[1]["hooks"][0]["command"]
            .as_str()
            .unwrap()
            .contains("--codex-hook stop"));
        assert_eq!(
            settings["hooks"]["UserPromptExpansion"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        let _ = fs::remove_file(settings_path);
    }

    #[test]
    fn tools_list_exposes_search_info_and_recording() {
        let root = temp_dir("mcp-tools");
        let mcp = test_mcp(root.clone(), vec![]);
        let response = mcp
            .handle_request(JsonRpcRequest {
                id: Some(json!(1)),
                method: "tools/list".to_string(),
                params: None,
            })
            .unwrap();
        let result = response.result.unwrap();
        let names = result["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["name"].as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "search".to_string(),
                "info".to_string(),
                "list_memories".to_string(),
                "record_session".to_string(),
                "enable_lint_ai".to_string(),
                "disable_lint_ai".to_string(),
                "lint_ai_status".to_string()
            ]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn control_tools_cover_recording_and_memory_matrix() {
        let root = temp_dir("control-matrix");
        let mcp = test_mcp(root.clone(), vec![]);

        assert_eq!(
            call_tool(&mcp, "record_session", json!({"action": "status"}))["enabled"],
            false
        );
        assert_eq!(
            call_tool(&mcp, "enable_lint_ai", json!({}))["enabled"],
            true
        );
        let status = call_tool(&mcp, "lint_ai_status", json!({}));
        assert_eq!(status["enabled"], true);
        assert_eq!(status["recording_enabled"], true);
        assert_eq!(status["display"], "Lint-AI:ON | Record:ON");
        assert_eq!(
            call_tool(&mcp, "record_session", json!({"action": "stop"}))["enabled"],
            false
        );
        let status = call_tool(&mcp, "lint_ai_status", json!({}));
        assert_eq!(status["enabled"], true);
        assert_eq!(status["recording_enabled"], false);
        assert_eq!(status["display"], "Lint-AI:ON | Record:OFF");
        assert_eq!(
            call_tool(&mcp, "disable_lint_ai", json!({}))["enabled"],
            false
        );
        assert_eq!(
            call_tool(&mcp, "record_session", json!({"action": "start"}))["enabled"],
            true
        );
        let status = call_tool(&mcp, "lint_ai_status", json!({}));
        assert_eq!(status["enabled"], false);
        assert_eq!(status["recording_enabled"], true);
        assert_eq!(status["display"], "Lint-AI:OFF | Record:ON");
        assert_eq!(
            call_tool(&mcp, "enable_lint_ai", json!({}))["enabled"],
            true
        );
        assert_eq!(
            call_tool(&mcp, "record_session", json!({"action": "stop"}))["enabled"],
            false
        );
        let status = call_tool(&mcp, "lint_ai_status", json!({}));
        assert_eq!(status["display"], "Lint-AI:ON | Record:OFF");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn tools_list_requires_store_initialization() {
        let root = temp_dir("mcp-lazy-store");
        let mcp = CodexMcp {
            root: root.clone(),
            max_bytes: 1_000_000,
            max_files: 100,
            max_depth: 5,
            max_total_bytes: 2_000_000,
            ignore_paths: Vec::new(),
            store: Mutex::new(None),
        };

        assert!(mcp.store.lock().unwrap().is_none());
        let response = mcp
            .handle_request(JsonRpcRequest {
                id: Some(json!(1)),
                method: "tools/list".to_string(),
                params: None,
            })
            .unwrap();

        assert!(response.error.is_none());
        assert!(mcp.store.lock().unwrap().is_some());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn search_tool_returns_ranked_results() {
        let root = temp_dir("mcp-search");
        let document = SourceDocument {
            doc_id: "doc-1".to_string(),
            source: "docs/install.md".to_string(),
            content: "docker install guide".to_string(),
            concept: "Install".to_string(),
            group_id: None,
            filters: std::collections::BTreeMap::new(),
            headings: vec!["Overview".to_string()],
            links: vec![],
            timestamp: None,
            doc_length: "docker install guide".len(),
            author_agent: None,
        };
        let mcp = test_mcp(root.clone(), vec![document]);
        let response = mcp
            .handle_request(JsonRpcRequest {
                id: Some(json!(2)),
                method: "tools/call".to_string(),
                params: Some(json!({
                    "name": "search",
                    "arguments": {
                        "query": "docker",
                        "top_k": 5
                    }
                })),
            })
            .unwrap();
        let result = response.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["docs_count"].as_u64(), Some(1));
        assert_eq!(parsed["results"].as_array().unwrap().len(), 1);
        assert_eq!(parsed["results"][0]["doc_id"].as_str(), Some("doc-1"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn search_tool_rejects_unknown_arguments() {
        let root = temp_dir("mcp-arguments");
        let mcp = test_mcp(root.clone(), vec![]);
        let response = mcp
            .handle_request(JsonRpcRequest {
                id: Some(json!(3)),
                method: "tools/call".to_string(),
                params: Some(json!({
                    "name": "search",
                    "arguments": { "query": "docker", "unexpected": true }
                })),
            })
            .unwrap();
        assert_eq!(response.error.unwrap().code, -32602);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn search_tool_synchronizes_memory_captured_after_startup() {
        let root = temp_dir("mcp-live-memory");
        let mcp = test_mcp(root.clone(), vec![]);
        let memory_root = root.join(".lint-ai").join("codex-memory");
        let mut memory = IndexStore::at_path(&memory_root, segmented_store_options()).unwrap();
        memory.upsert(SourceDocument {
            doc_id: "memory-1".to_string(),
            source: "codex://project/session/outcome".to_string(),
            content: "The durable routing codename is cobalt".to_string(),
            concept: "Codex Code outcome".to_string(),
            group_id: Some("codex-session:session".to_string()),
            filters: std::collections::BTreeMap::new(),
            headings: vec![],
            links: vec![],
            timestamp: None,
            doc_length: 38,
            author_agent: Some("codex".to_string()),
        });
        memory.refresh().unwrap();
        drop(memory);

        let response = mcp
            .handle_request(JsonRpcRequest {
                id: Some(json!(4)),
                method: "tools/call".to_string(),
                params: Some(json!({
                    "name": "search",
                    "arguments": { "query": "cobalt routing" }
                })),
            })
            .unwrap();
        let text = response.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        let payload: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(payload["results"][0]["doc_id"], "memory-1");
        fs::remove_dir_all(root).unwrap();
    }
}
