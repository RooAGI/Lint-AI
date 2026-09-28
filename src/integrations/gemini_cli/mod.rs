//! Gemini CLI integration.
//!
//! Gemini CLI hooks use JSON on stdin/stdout.  This adapter deliberately keeps
//! stdout protocol-clean and fails open: a Lint-AI problem must not interrupt
//! the user's Gemini session.

pub mod hooks;

use crate::adapters::{
    apply_ignore_paths, build_project_graph, graph_to_source_documents, AdapterInput,
};
use crate::integrations::mcp_index;
use crate::integrations::mcp_tools;
use crate::integrations::mcp_transport::{
    self, JsonRpcError, JsonRpcRequest, JsonRpcResponse, ToolDefinition,
};
use crate::integrations::session_recording::{
    lint_ai_enabled, recording_state, set_lint_ai_state, set_recording_state, RecordingProvider,
};
#[cfg(test)]
use crate::pipeline::IndexStore;
use anyhow::{Context, Result};
use serde_json::{json, Map, Value};
use std::env;
use std::fs;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const SERVER_NAME: &str = "lint-ai";
const HOOK_MARKER: &str = "--gemini-cli-hook";
const HOOK_EVENTS: &[(&str, &str)] = &[
    ("SessionStart", "session-start"),
    ("BeforeAgent", "before-agent"),
    ("AfterAgent", "after-agent"),
    ("BeforeModel", "before-model"),
    ("BeforeToolSelection", "before-tool-selection"),
    ("BeforeTool", "before-tool"),
    ("AfterTool", "after-tool"),
    ("PreCompress", "pre-compress"),
    ("SessionEnd", "session-end"),
];

#[derive(Debug, Clone)]
pub struct GeminiCliServerOptions<'a> {
    pub max_bytes: usize,
    pub max_files: usize,
    pub max_depth: usize,
    pub max_total_bytes: usize,
    pub ignore_paths: &'a [String],
}

struct GeminiMcp {
    root: PathBuf,
    store: Mutex<Option<crate::memory_api::MemoryService>>,
    provider: RecordingProvider,
    provider_label: &'static str,
    /// Display name used in the user-visible MCP tool descriptions.
    /// Passed per adapter ("Gemini" for gemini-cli/agy, "Hermes" for hermes)
    /// so shared scaffolding never leaks one adapter's name into another's.
    provider_display_name: &'static str,
    max_bytes: usize,
    max_files: usize,
    max_depth: usize,
    max_total_bytes: usize,
    ignore_paths: Vec<String>,
    workspace_watcher: Option<mcp_index::WorkspaceWatcher>,
}

pub fn install_user_config(root: &Path, config_path: Option<&Path>) -> Result<PathBuf> {
    let path = config_path
        .map(Path::to_path_buf)
        .unwrap_or(default_settings_path()?);
    let root = root
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", root.display()))?;
    let executable = env::current_exe().context("failed to locate lint-ai executable")?;
    let mut settings = read_json_object(&path)?;
    let servers = settings.entry("mcpServers").or_insert_with(|| json!({}));
    let servers = servers
        .as_object_mut()
        .context("Gemini mcpServers must be an object")?;
    servers.insert(
        SERVER_NAME.into(),
        json!({
            "command": executable,
            "args": ["--gemini-cli-serve", root.to_string_lossy()]
        }),
    );
    write_json_object(&path, &settings)?;
    Ok(path)
}

/// Install or update the Gemini CLI hook commands while preserving other hooks.
pub fn install_hook_settings(root: &Path, settings_path: Option<&Path>) -> Result<PathBuf> {
    let path = settings_path
        .map(Path::to_path_buf)
        .unwrap_or(default_settings_path()?);
    let _ = root
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", root.display()))?;
    let executable = env::current_exe().context("failed to locate lint-ai executable")?;
    let mut settings = read_json_object(&path)?;
    let hooks = settings.entry("hooks").or_insert_with(|| json!({}));
    let hooks = hooks
        .as_object_mut()
        .context("Gemini hooks must be an object")?;
    for (event, name) in HOOK_EVENTS {
        let entries = hooks.entry(*event).or_insert_with(|| json!([]));
        let entries = entries
            .as_array_mut()
            .with_context(|| format!("Gemini hook {event} must be an array"))?;
        entries.retain(|entry| !contains_lint_ai_hook(entry));
        let command = json!({
            "type": "command",
            "command": format!("{} {HOOK_MARKER} {name}", shell_quote(&executable.to_string_lossy()))
        });
        if matches!(*event, "BeforeTool" | "AfterTool") {
            entries.push(json!({"matcher": ".*", "hooks": [command]}));
        } else {
            entries.push(json!({"hooks": [command]}));
        }
    }
    write_json_object(&path, &settings)?;
    Ok(path)
}

pub fn run_server(root: &Path, options: GeminiCliServerOptions<'_>) -> Result<()> {
    run_server_for(
        root,
        RecordingProvider::Gemini,
        "gemini-cli",
        "Gemini",
        options,
    )
}

pub fn run_server_for(
    root: &Path,
    provider: RecordingProvider,
    provider_label: &'static str,
    provider_display_name: &'static str,
    options: GeminiCliServerOptions<'_>,
) -> Result<()> {
    mcp_index::trace_event("gemini-server-start");
    let server = GeminiMcp {
        root: root.to_path_buf(),
        store: Mutex::new(None),
        provider,
        provider_label,
        provider_display_name,
        max_bytes: options.max_bytes,
        max_files: options.max_files,
        max_depth: options.max_depth,
        max_total_bytes: options.max_total_bytes,
        ignore_paths: options.ignore_paths.to_vec(),
        workspace_watcher: Some(mcp_index::WorkspaceWatcher::new(
            root,
            options.ignore_paths,
        )?),
    };
    server.serve()
}

impl GeminiMcp {
    fn store(&self) -> Result<std::sync::MutexGuard<'_, Option<crate::memory_api::MemoryService>>> {
        let mut store = self
            .store
            .lock()
            .map_err(|_| anyhow::anyhow!("Gemini MCP store lock poisoned"))?;
        if self
            .workspace_watcher
            .as_ref()
            .is_some_and(mcp_index::WorkspaceWatcher::take_change)
        {
            *store = None;
        }
        if store.is_none() {
            let input = AdapterInput {
                root: &self.root,
                max_bytes: self.max_bytes,
                max_files: self.max_files,
                max_depth: self.max_depth,
                max_total_bytes: self.max_total_bytes,
            };
            let ignores = self.ignore_paths.clone();
            *store = Some(mcp_index::open_workspace_memory_store(
                &self.root,
                mcp_index::SHARED_MEMORY_DIR,
                &ignores,
                || {
                    let graph = build_project_graph(&input)?;
                    let graph = apply_ignore_paths(graph, &ignores);
                    Ok(graph_to_source_documents(&graph))
                },
            )?);
        }
        Ok(store)
    }

    fn serve(&self) -> Result<()> {
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut reader = BufReader::new(stdin.lock());
        let mut writer = stdout.lock();
        while let Some((request, line_framed)) = mcp_transport::read_request(&mut reader)? {
            if request.id.is_none() {
                continue;
            }
            let response = self.handle_request(request)?;
            mcp_transport::write_response(&mut writer, &response, line_framed)?;
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
                    "serverInfo": {"name": SERVER_NAME, "version": env!("CARGO_PKG_VERSION")},
                    "capabilities": {"tools": {"listChanged": false}}
                })),
                error: None,
            }),
            "notifications/initialized" => Ok(empty_response(id)),
            "tools/list" => Ok(JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(json!({"tools": tool_definitions(self.provider_display_name)})),
                error: None,
            }),
            "tools/call" => self.call_tool(id, request.params.unwrap_or_else(|| json!({}))),
            _ => Ok(error_response(id, -32601, "method not found")),
        }
    }

    fn call_tool(&self, id: Option<Value>, params: Value) -> Result<JsonRpcResponse> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let args = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));
        match name {
            "search" => {
                let query = args
                    .get("query")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim();
                if query.is_empty() {
                    return Ok(error_response(id, -32602, "query is required"));
                }
                let top_k = args
                    .get("top_k")
                    .and_then(Value::as_u64)
                    .unwrap_or(5)
                    .clamp(1, 20) as usize;
                let filters = match mcp_tools::search_provider_filters(&args) {
                    Ok(filters) => filters,
                    Err(message) => return Ok(error_response(id, -32602, &message)),
                };
                let mut service = self.store()?;
                let service = service.as_mut().expect("initialized");
                service.sync_shared_memory(&mcp_index::shared_memory_root(&self.root))?;
                service.refresh_index()?;
                // Stateful search: an explicit session id scopes follow-up
                // resolution and temporal-anchor carry to this provider's
                // conversation; when omitted, the search inherits the session
                // most recently seen active in this workspace (hooks keep that
                // pointer current in the service's conversation-state store).
                // Absent means stateless.
                let session_id = match mcp_tools::resolve_search_session_id(
                    &args,
                    &*service,
                    self.provider.as_str(),
                ) {
                    Ok(session_id) => session_id,
                    Err(message) => return Ok(error_response(id, -32602, &message)),
                };
                let started = std::time::Instant::now();
                let results = service.search_with_filters(
                    query,
                    self.provider.as_str(),
                    session_id.as_deref(),
                    top_k,
                    &filters,
                );
                let _ = crate::telemetry::record_project_query(
                    &self.root,
                    started.elapsed().as_millis() as u64,
                    results.is_err(),
                    results.as_ref().is_ok_and(Vec::is_empty),
                );
                let results = results?;
                Ok(text_response(
                    id,
                    &serde_json::to_string_pretty(
                        &json!({"query": query, "results": results, "provider": self.provider_label}),
                    )?,
                ))
            }
            "info" => {
                let service = self.store()?;
                Ok(text_response(
                    id,
                    &serde_json::to_string_pretty(
                        &json!({"provider":self.provider_label, "root":self.root, "docs_count":service.as_ref().map(|s| s.docs_count()).unwrap_or(0)}),
                    )?,
                ))
            }
            "list_memories" => {
                let limit = match mcp_tools::parse_list_memories_limit(&args) {
                    Ok(limit) => limit,
                    Err(_) => {
                        return Ok(error_response(id, -32602, "unknown list_memories argument"))
                    }
                };
                let mut service = self.store()?;
                let service = service.as_mut().expect("initialized");
                service.sync_shared_memory(&mcp_index::shared_memory_root(&self.root))?;
                service.refresh_index()?;
                Ok(text_response(
                    id,
                    &serde_json::to_string_pretty(&service.list_memories_payload(limit))?,
                ))
            }
            "enable_lint_ai" => {
                let state = set_lint_ai_state(self.provider, &self.root, true)?;
                set_recording_state(self.provider, &self.root, true)?;
                Ok(text_response(id, &serde_json::to_string_pretty(&state)?))
            }
            "disable_lint_ai" => {
                let state = set_lint_ai_state(self.provider, &self.root, false)?;
                Ok(text_response(id, &serde_json::to_string_pretty(&state)?))
            }
            "lint_ai_status" => {
                let state = json!({"provider":self.provider_label, "enabled":lint_ai_enabled(self.provider, &self.root)?, "recording_enabled":recording_state(self.provider, &self.root)?["enabled"]});
                Ok(text_response(id, &serde_json::to_string_pretty(&state)?))
            }
            "record_session" => {
                let action = args
                    .get("action")
                    .and_then(Value::as_str)
                    .unwrap_or("status");
                let state = match action {
                    "start" => set_recording_state(self.provider, &self.root, true)?,
                    "stop" => set_recording_state(self.provider, &self.root, false)?,
                    "status" => recording_state(self.provider, &self.root)?,
                    _ => {
                        return Ok(error_response(
                            id,
                            -32602,
                            "action must be start, stop, or status",
                        ))
                    }
                };
                Ok(text_response(id, &serde_json::to_string_pretty(&state)?))
            }
            "board_open" | "board_list" | "board_info" | "board_post" | "board_read"
            | "board_get" | "board_search" | "add_memory" | "get_memory" => {
                let mut service = self.store()?;
                let service = service.as_mut().expect("initialized");
                mcp_tools::call_board_or_memory_tool(
                    name,
                    id,
                    &args,
                    service,
                    &self.root,
                    self.provider.as_str(),
                )
            }
            _ => Ok(error_response(id, -32602, "unknown tool")),
        }
    }
}

fn tool_definitions(provider_display_name: &str) -> Vec<ToolDefinition> {
    let schema = |properties: Value, required: Vec<&str>| json!({"type":"object", "properties":properties, "required":required});
    let name = provider_display_name;
    let tools = vec![
        ToolDefinition {
            name: "search".into(),
            description: format!("Search {name} project memory."),
            input_schema: schema(
                json!({"query":{"type":"string"},"top_k":{"type":"integer"},"provider": mcp_tools::provider_argument_schema(),"session_id": {"type": "string", "description": "Optional conversation session id for follow-up resolution against prior session state. When omitted, the search inherits the session most recently seen active in this workspace (tracked by Lint-AI hooks); omit entirely only for stateless search."}}),
                vec!["query"],
            ),
        },
        ToolDefinition {
            name: "info".into(),
            description: format!("Show {name} Lint-AI memory status."),
            input_schema: schema(json!({}), vec![]),
        },
        mcp_tools::list_memories_tool_definition(),
        ToolDefinition {
            name: "record_session".into(),
            description: "Start, stop, or inspect session recording.".into(),
            input_schema: schema(
                json!({"action":{"type":"string","enum":["start","stop","status"]}}),
                vec![],
            ),
        },
        ToolDefinition {
            name: "enable_lint_ai".into(),
            description: format!("Enable {name} Lint-AI memory and recording."),
            input_schema: schema(json!({}), vec![]),
        },
        ToolDefinition {
            name: "disable_lint_ai".into(),
            description: format!("Disable {name} Lint-AI memory injection."),
            input_schema: schema(json!({}), vec![]),
        },
        ToolDefinition {
            name: "lint_ai_status".into(),
            description: format!("Show {name} Lint-AI and recording state."),
            input_schema: schema(json!({}), vec![]),
        },
    ];
    let mut tools = tools;
    tools.extend(mcp_tools::board_tool_definitions());
    tools.extend(mcp_tools::memory_tool_definitions());
    tools
}

fn text_response(id: Option<Value>, text: &str) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: Some(json!({"content":[{"type":"text","text":text}]})),
        error: None,
    }
}
fn empty_response(id: Option<Value>) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: Some(json!({})),
        error: None,
    }
}
fn error_response(id: Option<Value>, code: i64, message: &str) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: None,
        error: Some(JsonRpcError {
            code,
            message: message.into(),
        }),
    }
}

fn read_json_object(path: &Path) -> Result<Map<String, Value>> {
    if !path.exists() {
        return Ok(Map::new());
    }
    let contents = fs::read_to_string(path)?;
    if contents.trim().is_empty() {
        return Ok(Map::new());
    }
    let value: Value = serde_json::from_str(&contents)?;
    value
        .as_object()
        .cloned()
        .context("Gemini settings must contain a JSON object")
}

fn write_json_object(path: &Path, value: &Map<String, Value>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(value)? + "\n")?;
    Ok(())
}

fn contains_lint_ai_hook(value: &Value) -> bool {
    serde_json::to_string(value)
        .map(|s| s.contains(HOOK_MARKER))
        .unwrap_or(false)
}

fn shell_quote(value: &str) -> String {
    if value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b"/_-.".contains(&b))
    {
        value.into()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn home_dir() -> Result<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .context("HOME or USERPROFILE is not set")
}

fn default_settings_path() -> Result<PathBuf> {
    Ok(home_dir()?.join(".gemini").join("settings.json"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::PipelineOptions;
    use crate::source::SourceDocument;
    use std::collections::BTreeMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("lint-ai-gemini-mcp-{name}-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn call_tool(mcp: &GeminiMcp, name: &str, arguments: Value) -> Value {
        let response = mcp
            .handle_request(JsonRpcRequest {
                id: Some(json!(1)),
                method: "tools/call".to_string(),
                params: Some(json!({"name": name, "arguments": arguments})),
            })
            .unwrap();
        assert!(response.error.is_none(), "{name} should succeed");
        let text = response.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        serde_json::from_str(&text).unwrap()
    }

    #[test]
    fn adapter_search_returns_hits_for_all_providers() {
        // Hit-parity gate: the shared adapter search path (gemini/agy/openclaw)
        // must return hits for user-ID-less local docs. A 0-hits outcome here
        // would reproduce the Hermes finding (unconditional ownership filter).
        for (provider, label, display_name) in [
            (RecordingProvider::Gemini, "gemini-cli", "Gemini"),
            (RecordingProvider::Agy, "agy", "Gemini"),
            (RecordingProvider::OpenClaw, "openclaw", "OpenClaw"),
        ] {
            let root = temp_root(label);
            // Seed through the shared memory root, like hook captures do:
            // no memory_user_id filter on the documents.
            let memory_root = mcp_index::shared_memory_root(&root);
            let mut memory = IndexStore::at_path(&memory_root, PipelineOptions::default()).unwrap();
            let doc_id = format!("{label}-search-doc");
            let mut filters = BTreeMap::new();
            filters.insert("provider".to_string(), label.to_string());
            memory.upsert(SourceDocument {
                doc_id: doc_id.clone(),
                source: format!("{}://session-1/outcome", provider.as_str()),
                content: format!(
                    "{label} deployment runbook: rotate the staging API key every Friday"
                ),
                concept: "outcome".to_string(),
                group_id: Some(format!("{}-session:session-1", provider.as_str())),
                filters,
                headings: vec![],
                links: vec![],
                timestamp: None,
                doc_length: 64,
                author_agent: Some(provider.as_str().to_string()),
                key_phrases: Vec::new(),
                key_phrase_extraction_hash: String::new(),
            });
            memory.refresh().unwrap();
            drop(memory);

            let mcp = GeminiMcp {
                root: root.clone(),
                store: Mutex::new(None),
                provider,
                provider_label: label,
                provider_display_name: display_name,
                max_bytes: 5_000_000,
                max_files: 50_000,
                max_depth: 20,
                max_total_bytes: 100_000_000,
                ignore_paths: vec![],
                workspace_watcher: None,
            };
            // Unfiltered search over the shared pool.
            let res = call_tool(
                &mcp,
                "search",
                json!({"query": "staging API key rotation runbook"}),
            );
            let results = res["results"].as_array().cloned().unwrap_or_default();
            assert!(
                !results.is_empty(),
                "{label}: adapter search returned 0 hits (0-hits regression)"
            );
            assert!(
                results
                    .iter()
                    .any(|hit| hit["doc_id"].as_str() == Some(doc_id.as_str())),
                "{label}: seeded doc missing from search results: {res}"
            );
            // Provider-scoped search keeps this provider's doc...
            let res = call_tool(
                &mcp,
                "search",
                json!({"query": "staging API key rotation runbook", "provider": label}),
            );
            let results = res["results"].as_array().cloned().unwrap_or_default();
            assert!(
                results
                    .iter()
                    .any(|hit| hit["doc_id"].as_str() == Some(doc_id.as_str())),
                "{label}: provider filter dropped its own doc: {res}"
            );
            // ...and rejects an unknown provider instead of returning empty.
            let err = mcp
                .handle_request(JsonRpcRequest {
                    id: Some(json!(1)),
                    method: "tools/call".to_string(),
                    params: Some(
                        json!({"name": "search", "arguments": {"query": "key", "provider": "nope"}}),
                    ),
                })
                .unwrap();
            assert!(
                err.error
                    .as_ref()
                    .is_some_and(|e| e.message.contains("unknown provider")),
                "{label}: unknown provider should be an error: {err:?}"
            );
            drop(mcp);
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn installs_hooks_idempotently_and_preserves_settings() {
        let root = env::temp_dir().join(format!(
            "lint-ai-gemini-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let settings = root.join("settings.json");
        fs::write(&settings, r#"{"theme":"dark","hooks":{"BeforeAgent":[{"hooks":[{"type":"command","command":"user-hook"}]}]}}"#).unwrap();
        install_hook_settings(&root, Some(&settings)).unwrap();
        install_hook_settings(&root, Some(&settings)).unwrap();
        let value: Value = serde_json::from_str(&fs::read_to_string(&settings).unwrap()).unwrap();
        assert_eq!(value["theme"], "dark");
        assert_eq!(value["hooks"]["SessionStart"].as_array().unwrap().len(), 1);
        assert_eq!(value["hooks"]["BeforeAgent"].as_array().unwrap().len(), 2);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn gemini_compatible_mcp_contract_applies_to_all_adapters() {
        for (provider, label, display_name) in [
            (RecordingProvider::Gemini, "gemini-cli", "Gemini"),
            (RecordingProvider::Agy, "agy", "Gemini"),
            (RecordingProvider::OpenClaw, "openclaw", "OpenClaw"),
            (RecordingProvider::Hermes, "hermes", "Hermes"),
        ] {
            let root = temp_root(label);
            let memory_root = mcp_index::shared_memory_root(&root);
            let mut memory = IndexStore::at_path(&memory_root, PipelineOptions::default()).unwrap();
            memory.upsert(SourceDocument {
                doc_id: "memory-1".to_string(),
                source: format!("{}://session-1/outcome", provider.as_str()),
                content: format!("{label} durable routing decision"),
                concept: "outcome".to_string(),
                group_id: Some(format!("{}-session:session-1", provider.as_str())),
                filters: BTreeMap::new(),
                headings: vec![],
                links: vec![],
                timestamp: None,
                doc_length: 32,
                author_agent: Some(provider.as_str().to_string()),
                key_phrases: Vec::new(),
                key_phrase_extraction_hash: String::new(),
            });
            memory.refresh().unwrap();
            drop(memory);

            let mcp = GeminiMcp {
                root: root.clone(),
                store: Mutex::new(None),
                provider,
                provider_label: label,
                provider_display_name: display_name,
                max_bytes: 5_000_000,
                max_files: 50_000,
                max_depth: 20,
                max_total_bytes: 100_000_000,
                ignore_paths: vec![],
                workspace_watcher: None,
            };
            let tools = mcp
                .handle_request(JsonRpcRequest {
                    id: Some(json!(1)),
                    method: "tools/list".to_string(),
                    params: None,
                })
                .unwrap()
                .result
                .unwrap();
            let names = tools["tools"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|tool| tool["name"].as_str())
                .collect::<Vec<_>>();
            for required in [
                "search",
                "list_memories",
                "record_session",
                "enable_lint_ai",
                "disable_lint_ai",
                "lint_ai_status",
                // Bulletin-board tools and add_memory / get_memory are now
                // opted in for all Gemini-compatible providers.
                "board_open",
                "board_list",
                "board_info",
                "board_post",
                "board_read",
                "board_get",
                "board_search",
                "add_memory",
                "get_memory",
            ] {
                assert!(names.contains(&required), "{label} must expose {required}");
            }
            // Tool descriptions carry the adapter's own display name, never
            // another adapter's.
            for tool in tools["tools"].as_array().unwrap() {
                let description = tool["description"].as_str().unwrap_or("");
                assert!(
                    !description.contains("Gemini") || display_name == "Gemini",
                    "{label} tool {} leaks a Gemini description: {description}",
                    tool["name"].as_str().unwrap_or("?")
                );
                if tool["name"] == "search" {
                    assert_eq!(
                        description,
                        format!("Search {display_name} project memory.")
                    );
                }
            }

            let memories = call_tool(&mcp, "list_memories", json!({"limit": 20}));
            assert_eq!(memories["count"], 1);
            assert!(memories["memories"][0]["content"]
                .as_str()
                .unwrap()
                .contains(label));
            assert_eq!(
                call_tool(&mcp, "enable_lint_ai", json!({}))["enabled"],
                true
            );
            assert_eq!(
                call_tool(&mcp, "record_session", json!({"action": "stop"}))["enabled"],
                false
            );
            let status = call_tool(&mcp, "lint_ai_status", json!({}));
            assert_eq!(status["enabled"], true);
            assert_eq!(status["recording_enabled"], false);
            drop(mcp);
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn board_and_memory_tools_round_trip_for_gemini_compatible_providers() {
        let providers = [
            (RecordingProvider::Gemini, "gemini-cli", "Gemini"),
            (RecordingProvider::Agy, "agy", "Gemini"),
            (RecordingProvider::OpenClaw, "openclaw", "OpenClaw"),
        ];
        for (provider, label, display_name) in providers {
            let root = temp_root(label);
            let mcp = GeminiMcp {
                root: root.clone(),
                store: Mutex::new(None),
                provider,
                provider_label: label,
                provider_display_name: display_name,
                max_bytes: 5_000_000,
                max_files: 50_000,
                max_depth: 20,
                max_total_bytes: 100_000_000,
                ignore_paths: vec![],
                workspace_watcher: None,
            };
            let session = format!("gemini-board-{label}");

            // board_post -> board_read: same session's default board.
            let request_id = format!("post-{label}");
            let posted = call_tool(
                &mcp,
                "board_post",
                json!({
                    "content": "OpenClaw integration checkpoint reached.",
                    "request_id": request_id,
                    "session_id": session,
                }),
            );
            assert!(
                posted["content"]
                    .as_str()
                    .unwrap()
                    .contains("OpenClaw integration checkpoint"),
                "{provider:?}: board_post did not echo content: {posted}"
            );
            let read = call_tool(&mcp, "board_read", json!({"session_id": session}));
            let posts = read["posts"].as_array().cloned().unwrap_or_default();
            assert!(
                posts.iter().any(|post| post["content"]
                    .as_str()
                    .is_some_and(|content| content.contains("OpenClaw integration checkpoint"))),
                "{provider:?}: board_read missed the post: {read}"
            );

            // add_memory -> get_memory: the memory ID is deterministic.
            let mem_request = format!("mem-{label}");
            let added = call_tool(
                &mcp,
                "add_memory",
                json!({
                    "content": "The staging API key rotates every Friday.",
                    "request_id": mem_request,
                    "session_id": session,
                }),
            );
            assert_eq!(
                added["request_id"].as_str().unwrap(),
                mem_request,
                "{provider:?}: add_memory response missing request id: {added}"
            );
            let memory_id = crate::stable_doc_id_from_source(&format!("mcp:{mem_request}:0"));
            let fetched = call_tool(&mcp, "get_memory", json!({"memory_id": memory_id}));
            assert!(
                fetched["content"]
                    .as_str()
                    .unwrap()
                    .contains("rotates every Friday"),
                "{provider:?}: get_memory missed the content: {fetched}"
            );
            drop(mcp);
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn shared_search_scopes_conversation_state_to_adapter_provider() {
        // The shared Gemini-compatible search must scope conversation state
        // (session pointers, follow-up resolution) to the adapter's own
        // provider, never hardcoded "gemini".
        let root = temp_root("hermes-scope");
        let memory_root = mcp_index::shared_memory_root(&root);
        fs::create_dir_all(&memory_root).unwrap();
        fs::write(
            memory_root.join("tea.md"),
            "# Tea\nLuyi prefers oolong tea from Alishan, Taiwan.",
        )
        .unwrap();
        let make_mcp = |provider, label: &'static str, display_name: &'static str| GeminiMcp {
            root: root.clone(),
            store: Mutex::new(None),
            provider,
            provider_label: label,
            provider_display_name: display_name,
            max_bytes: 5_000_000,
            max_files: 50_000,
            max_depth: 20,
            max_total_bytes: 100_000_000,
            ignore_paths: vec![],
            workspace_watcher: None,
        };

        // Gemini turn in session s1: seeds ("gemini", "s1") conversation
        // state about oolong tea.
        let gemini_mcp = make_mcp(RecordingProvider::Gemini, "gemini-cli", "Gemini");
        let seed = call_tool(
            &gemini_mcp,
            "search",
            json!({"query": "oolong tea", "top_k": 3, "session_id": "s1"}),
        );
        assert!(
            !seed["results"].as_array().unwrap().is_empty(),
            "seed turn should find the tea doc"
        );
        drop(gemini_mcp);

        let hermes_mcp = make_mcp(RecordingProvider::Hermes, "hermes", "Hermes");
        // Hermes follow-up in the same session id: with the adapter's own
        // scope there is no hermes turn to resolve against, so the bare
        // follow-up finds nothing. A hardcoded "gemini" scope leaks
        // Gemini's turn in and returns the tea doc.
        let follow_up = call_tool(
            &hermes_mcp,
            "search",
            json!({"query": "tell me about it", "top_k": 3, "session_id": "s1"}),
        );
        assert!(
            follow_up["results"].as_array().unwrap().is_empty(),
            "hermes follow-up must not resolve against gemini's conversation state: {}",
            serde_json::to_string_pretty(&follow_up["results"]).unwrap()
        );

        // Session-pointer wiring: an explicit session id is noted under the
        // adapter's own provider, not gemini's.
        call_tool(
            &hermes_mcp,
            "search",
            json!({"query": "oolong tea", "top_k": 1, "session_id": "s2"}),
        );
        {
            let mut guard = hermes_mcp.store().unwrap();
            let service = guard.as_mut().expect("initialized");
            assert_eq!(
                service.current_session_id("hermes").as_deref(),
                Some("s2"),
                "hermes adapter must note the session under its own provider"
            );
        }
        drop(hermes_mcp);
        fs::remove_dir_all(root).unwrap();
    }
}
