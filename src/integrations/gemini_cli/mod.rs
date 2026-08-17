//! Gemini CLI integration.
//!
//! Gemini CLI hooks use JSON on stdin/stdout.  This adapter deliberately keeps
//! stdout protocol-clean and fails open: a Lint-AI problem must not interrupt
//! the user's Gemini session.

pub mod hooks;

use crate::integrations::mcp_index;
use crate::integrations::mcp_tools;
use crate::integrations::mcp_transport::{
    self, JsonRpcError, JsonRpcRequest, JsonRpcResponse, ToolDefinition,
};
use crate::integrations::session_recording::{
    lint_ai_enabled, recording_state, set_lint_ai_state, set_recording_state, RecordingProvider,
};
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
    store: Mutex<Option<IndexStore>>,
    provider: RecordingProvider,
    provider_label: &'static str,
    memory_dir: &'static str,
    index_name: &'static str,
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

pub fn run_server(root: &Path, _options: GeminiCliServerOptions<'_>) -> Result<()> {
    run_server_for(
        root,
        RecordingProvider::Gemini,
        "gemini-cli",
        "gemini-cli-memory",
        "gemini-mcp-index",
    )
}

pub fn run_server_for(
    root: &Path,
    provider: RecordingProvider,
    provider_label: &'static str,
    memory_dir: &'static str,
    index_name: &'static str,
) -> Result<()> {
    mcp_index::trace_event("gemini-server-start");
    let server = GeminiMcp {
        root: root.to_path_buf(),
        store: Mutex::new(None),
        provider,
        provider_label,
        memory_dir,
        index_name,
    };
    server.serve()
}

impl GeminiMcp {
    fn store(&self) -> Result<std::sync::MutexGuard<'_, Option<IndexStore>>> {
        let mut store = self
            .store
            .lock()
            .map_err(|_| anyhow::anyhow!("Gemini MCP store lock poisoned"))?;
        if store.is_none() {
            *store = Some(mcp_index::open_persistent_store(
                &self.root,
                self.index_name,
                self.memory_dir,
                || Ok(Vec::new()),
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
                result: Some(json!({"tools": tool_definitions()})),
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
                let mut store = self.store()?;
                let store = store.as_mut().expect("initialized");
                mcp_index::sync_memory_documents(
                    &self.root.join(".lint-ai").join(self.memory_dir),
                    store,
                )?;
                store.refresh()?;
                let results = store.query(query, top_k)?;
                Ok(text_response(
                    id,
                    &serde_json::to_string_pretty(
                        &json!({"query": query, "results": results, "provider": self.provider_label}),
                    )?,
                ))
            }
            "info" => {
                let store = self.store()?;
                Ok(text_response(
                    id,
                    &serde_json::to_string_pretty(
                        &json!({"provider":self.provider_label, "root":self.root, "docs_count":store.as_ref().map(|s| s.source_documents().len()).unwrap_or(0)}),
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
                let mut store = self.store()?;
                let store = store.as_mut().expect("initialized");
                mcp_index::sync_memory_documents(
                    &self.root.join(".lint-ai").join(self.memory_dir),
                    store,
                )?;
                store.refresh()?;
                Ok(text_response(
                    id,
                    &serde_json::to_string_pretty(&mcp_tools::list_memories(store, limit))?,
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
            _ => Ok(error_response(id, -32602, "unknown tool")),
        }
    }
}

fn tool_definitions() -> Vec<ToolDefinition> {
    let schema = |properties: Value, required: Vec<&str>| json!({"type":"object", "properties":properties, "required":required});
    vec![
        ToolDefinition {
            name: "search".into(),
            description: "Search Gemini project memory.".into(),
            input_schema: schema(
                json!({"query":{"type":"string"},"top_k":{"type":"integer"}}),
                vec!["query"],
            ),
        },
        ToolDefinition {
            name: "info".into(),
            description: "Show Gemini Lint-AI memory status.".into(),
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
            description: "Enable Gemini Lint-AI memory and recording.".into(),
            input_schema: schema(json!({}), vec![]),
        },
        ToolDefinition {
            name: "disable_lint_ai".into(),
            description: "Disable Gemini Lint-AI memory injection.".into(),
            input_schema: schema(json!({}), vec![]),
        },
        ToolDefinition {
            name: "lint_ai_status".into(),
            description: "Show Gemini Lint-AI and recording state.".into(),
            input_schema: schema(json!({}), vec![]),
        },
    ]
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
    fn gemini_compatible_mcp_contract_applies_to_gemini_and_agy() {
        for (provider, label, memory_dir, index_name) in [
            (
                RecordingProvider::Gemini,
                "gemini-cli",
                "gemini-cli-memory",
                "gemini-mcp-index",
            ),
            (RecordingProvider::Agy, "agy", "agy-memory", "agy-mcp-index"),
        ] {
            let root = temp_root(label);
            let memory_root = root.join(".lint-ai").join(memory_dir);
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
            });
            memory.refresh().unwrap();
            drop(memory);

            let mcp = GeminiMcp {
                root: root.clone(),
                store: Mutex::new(None),
                provider,
                provider_label: label,
                memory_dir,
                index_name,
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
            ] {
                assert!(names.contains(&required), "{label} must expose {required}");
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
}
