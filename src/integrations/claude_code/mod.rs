pub mod document;
pub mod hooks;

use crate::config::normalize_list;
use crate::graph::{Graph, Tier0Record};
use crate::pipeline::{IndexStore, MemoryIndexLayout, PipelineOptions};
use crate::segments::SegmentRoutingStrategy;
use crate::source::SourceDocument;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

const SERVER_NAME: &str = "lint-ai";
const DEFAULT_QUERY_TOP_K: usize = 5;
const MAX_REQUEST_BYTES: usize = 1024 * 1024;
const HOOK_MARKER: &str = "--claude-code-hook";
const HOOK_EVENTS: &[(&str, &str)] = &[
    ("SessionStart", "session-start"),
    ("UserPromptSubmit", "user-prompt-submit"),
    ("UserPromptExpansion", "user-prompt-expansion"),
    ("PreCompact", "pre-compact"),
    ("Stop", "stop"),
    ("SessionEnd", "session-end"),
];

#[derive(Debug, Clone)]
pub struct ClaudeCodeServerOptions<'a> {
    pub max_bytes: usize,
    pub max_files: usize,
    pub max_depth: usize,
    pub max_total_bytes: usize,
    pub ignore_paths: &'a [String],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct McpServerEntry {
    command: String,
    args: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ClaudeConfig {
    #[serde(rename = "mcpServers", default)]
    mcp_servers: HashMap<String, McpServerEntry>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct JsonRpcRequest {
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
struct ToolDefinition {
    name: String,
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

struct ClaudeMcp {
    root: PathBuf,
    store: Mutex<IndexStore>,
}

pub fn install_user_config(root: &Path, config_path: Option<&Path>) -> Result<PathBuf> {
    let config_path = match config_path {
        Some(path) => path.to_path_buf(),
        None => default_claude_config_path()?,
    };
    let root = root
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", root.display()))?;
    let mut config = read_claude_config(&config_path)?;
    config.mcp_servers.insert(
        SERVER_NAME.to_string(),
        McpServerEntry {
            command: "lint-ai".to_string(),
            args: vec![
                "--claude-code-serve".to_string(),
                root.to_string_lossy().into_owned(),
            ],
        },
    );
    write_claude_config(&config_path, &config)?;
    Ok(config_path)
}

pub fn install_hook_settings(root: &Path, settings_path: Option<&Path>) -> Result<PathBuf> {
    let settings_path = match settings_path {
        Some(path) => path.to_path_buf(),
        None => default_claude_settings_path()?,
    };
    let _root = root
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", root.display()))?;
    let executable = env::current_exe()
        .ok()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|| "lint-ai".to_string());
    let mut settings = read_json_object(&settings_path)?;
    let hooks = settings
        .entry("hooks".to_string())
        .or_insert_with(|| json!({}));
    let hooks = hooks
        .as_object_mut()
        .context("Claude settings 'hooks' must be an object")?;

    for (event_name, hook_name) in HOOK_EVENTS {
        let entries = hooks
            .entry((*event_name).to_string())
            .or_insert_with(|| json!([]));
        let entries = entries
            .as_array_mut()
            .with_context(|| format!("Claude hook event '{event_name}' must be an array"))?;
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

pub fn run_server(root: &Path, options: ClaudeCodeServerOptions<'_>) -> Result<()> {
    let graph = build_graph(
        root,
        options.max_bytes,
        options.max_files,
        options.max_depth,
        options.max_total_bytes,
    )?;
    let graph = apply_ignores(graph, options.ignore_paths);
    let source_docs = graph_to_source_documents(&graph);
    let mut store = IndexStore::new(segmented_store_options());
    for document in source_docs {
        store.upsert(document);
    }
    sync_memory_documents(root, &mut store)?;
    store.refresh()?;

    let mcp = ClaudeMcp {
        root: root.to_path_buf(),
        store: Mutex::new(store),
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

impl ClaudeMcp {
    fn serve(self) -> Result<()> {
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut reader = BufReader::new(stdin.lock());
        let mut writer = stdout.lock();

        while let Some(request) = read_request(&mut reader)? {
            if request.id.is_none() {
                continue;
            }
            let response = self.handle_request(request)?;
            write_response(&mut writer, &response)?;
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
            "tools/list" => Ok(JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(json!({
                    "tools": self.tools(),
                })),
                error: None,
            }),
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
                let mut store = self
                    .store
                    .lock()
                    .map_err(|_| anyhow::anyhow!("MCP index lock poisoned"))?;
                sync_memory_documents(&self.root, &mut store)?;
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
            "info" => {
                if let Some(name) = unknown_argument(&arguments, &[]) {
                    return Ok(error_response(
                        id,
                        -32602,
                        &format!("unknown info argument: {name}"),
                    ));
                }
                let store = self
                    .store
                    .lock()
                    .map_err(|_| anyhow::anyhow!("MCP index lock poisoned"))?;
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
        ]
    }
}

fn segmented_store_options() -> PipelineOptions {
    PipelineOptions {
        memory_index_layout: MemoryIndexLayout::Segmented {
            query_top_n: 3,
            routing_strategy: SegmentRoutingStrategy::LocalDistinctiveness,
        },
        ..PipelineOptions::default()
    }
}

fn sync_memory_documents(root: &Path, target: &mut IndexStore) -> Result<()> {
    let memory_root = root.join(".lint-ai").join("claude-memory");
    if !memory_root.exists() {
        return Ok(());
    }
    let memory = IndexStore::at_path(&memory_root, segmented_store_options())?;
    for document in memory.source_documents() {
        let unchanged = target
            .source_document_by_id(&document.doc_id)
            .map(|current| {
                current.source == document.source
                    && current.content == document.content
                    && current.group_id == document.group_id
                    && current.timestamp == document.timestamp
                    && current.filters == document.filters
                    && current.headings == document.headings
                    && current.links == document.links
            })
            .unwrap_or(false);
        if !unchanged {
            target.upsert(document.clone());
        }
    }
    Ok(())
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

fn read_request(reader: &mut impl BufRead) -> Result<Option<JsonRpcRequest>> {
    let mut content_length = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Ok(None);
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
            content_length = Some(
                rest.trim()
                    .parse::<usize>()
                    .context("invalid Content-Length header")?,
            );
        }
    }

    let len = content_length.context("missing Content-Length header")?;
    if len > MAX_REQUEST_BYTES {
        anyhow::bail!("MCP request body exceeds {MAX_REQUEST_BYTES} byte limit: {len} bytes");
    }
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body)?;
    let request: JsonRpcRequest = serde_json::from_slice(&body)?;
    Ok(Some(request))
}

fn write_response(writer: &mut impl Write, response: &JsonRpcResponse) -> Result<()> {
    let body = serde_json::to_vec(response)?;
    write!(writer, "Content-Length: {}\r\n\r\n", body.len())?;
    writer.write_all(&body)?;
    writer.flush()?;
    Ok(())
}

fn read_claude_config(path: &Path) -> Result<ClaudeConfig> {
    if !path.exists() {
        return Ok(ClaudeConfig::default());
    }
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

fn write_claude_config(path: &Path, config: &ClaudeConfig) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(config)?;
    fs::write(path, text)?;
    Ok(())
}

fn read_json_object(path: &Path) -> Result<Map<String, Value>> {
    if !path.exists() {
        return Ok(Map::new());
    }
    let value: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    value
        .as_object()
        .cloned()
        .context("Claude settings must be a JSON object")
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

fn default_claude_config_path() -> Result<PathBuf> {
    if let Some(home) = env::var_os("HOME") {
        return Ok(PathBuf::from(home).join(".claude.json"));
    }
    if cfg!(windows) {
        if let Some(profile) = env::var_os("USERPROFILE") {
            return Ok(PathBuf::from(profile).join(".claude.json"));
        }
    }
    anyhow::bail!("unable to determine Claude Code config path; set --claude-code-config");
}

fn default_claude_settings_path() -> Result<PathBuf> {
    if let Some(home) = env::var_os("HOME") {
        return Ok(PathBuf::from(home).join(".claude").join("settings.json"));
    }
    if cfg!(windows) {
        if let Some(profile) = env::var_os("USERPROFILE") {
            return Ok(PathBuf::from(profile).join(".claude").join("settings.json"));
        }
    }
    anyhow::bail!("unable to determine Claude Code settings path")
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
        env::temp_dir().join(format!("lint-ai-{name}-{nanos}.json"))
    }

    fn temp_dir(name: &str) -> PathBuf {
        let path = temp_path(name).with_extension("");
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn test_mcp(root: PathBuf, documents: Vec<SourceDocument>) -> ClaudeMcp {
        let mut store = IndexStore::new(segmented_store_options());
        for document in documents {
            store.upsert(document);
        }
        store.refresh().unwrap();
        ClaudeMcp {
            root,
            store: Mutex::new(store),
        }
    }

    #[test]
    fn install_user_config_merges_existing_servers() {
        let config_path = temp_path("claude-config");
        fs::write(
            &config_path,
            r#"{"mcpServers":{"existing":{"command":"npx","args":["-y","existing"]}},"x":1}"#,
        )
        .unwrap();
        let root = env::current_dir().unwrap();

        let written = install_user_config(&root, Some(&config_path)).unwrap();
        assert_eq!(written, config_path);

        let parsed: ClaudeConfig =
            serde_json::from_str(&fs::read_to_string(&config_path).unwrap()).unwrap();
        assert!(parsed.mcp_servers.contains_key("existing"));
        let entry = parsed.mcp_servers.get(SERVER_NAME).unwrap();
        assert_eq!(entry.command, "lint-ai");
        assert_eq!(entry.args[0], "--claude-code-serve");
        assert_eq!(entry.args[1], root.to_string_lossy());
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
            .contains("--claude-code-hook stop"));
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
    fn tools_list_exposes_search_and_info() {
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
        assert_eq!(names, vec!["search".to_string(), "info".to_string()]);
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
        let memory_root = root.join(".lint-ai").join("claude-memory");
        let mut memory = IndexStore::at_path(&memory_root, segmented_store_options()).unwrap();
        memory.upsert(SourceDocument {
            doc_id: "memory-1".to_string(),
            source: "claude-code://project/session/outcome".to_string(),
            content: "The durable routing codename is cobalt".to_string(),
            concept: "Claude Code outcome".to_string(),
            group_id: Some("claude-session:session".to_string()),
            filters: std::collections::BTreeMap::new(),
            headings: vec![],
            links: vec![],
            timestamp: None,
            doc_length: 38,
            author_agent: Some("claude-code".to_string()),
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

    #[test]
    fn read_request_rejects_oversized_body() {
        let request = format!("Content-Length: {}\r\n\r\n", MAX_REQUEST_BYTES + 1);
        let error = read_request(&mut std::io::Cursor::new(request)).unwrap_err();
        assert!(error.to_string().contains("exceeds"));
    }
}
