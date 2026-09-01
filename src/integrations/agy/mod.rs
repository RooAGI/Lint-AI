//! Antigravity CLI integration. MCP transport is shared with Gemini CLI, while
//! lifecycle hooks use AGY's independent hooks.json protocol.

pub mod hooks;

use crate::integrations::gemini_cli::{self, GeminiCliServerOptions};
use crate::integrations::session_recording::RecordingProvider;
use anyhow::Result;
use serde_json::{json, Map, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const SERVER_NAME: &str = "lint-ai";
const HOOK_MARKER: &str = "--agy-hook";

pub type AgyServerOptions<'a> = GeminiCliServerOptions<'a>;

pub fn install_memory_skill(root: &Path, force: bool) -> Result<PathBuf> {
    let root = root.canonicalize()?;
    let skill_path = root.join(".agents/skills/lint-ai-memory/SKILL.md");
    if let Some(parent) = skill_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let generated = include_str!("skill.md");
    if let Ok(existing) = fs::read_to_string(&skill_path) {
        if !force && existing != generated {
            anyhow::bail!(
                "refusing to overwrite existing AGY skill {}; it appears user-modified; use --agy-force-skill to replace it",
                skill_path.display()
            );
        }
        if existing == generated {
            return Ok(skill_path);
        }
    }
    fs::write(&skill_path, generated)?;
    Ok(skill_path)
}

pub fn install_user_config(root: &Path, config_path: Option<&Path>) -> Result<PathBuf> {
    let path = config_path
        .map(Path::to_path_buf)
        .unwrap_or(default_config_path()?);
    let root = root.canonicalize()?;
    let executable = env::current_exe()?;
    let mut settings = read_json_object(&path)?;
    let servers = settings.entry("mcpServers").or_insert_with(|| json!({}));
    let servers = servers
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("AGY mcpServers must be an object"))?;
    servers.insert(
        SERVER_NAME.into(),
        json!({
            "command": executable,
            "args": ["--agy-serve", root.to_string_lossy()]
        }),
    );
    write_json_object(&path, &settings)?;
    Ok(path)
}

pub fn install_hook_settings(root: &Path, settings_path: Option<&Path>) -> Result<PathBuf> {
    let path = settings_path
        .map(Path::to_path_buf)
        .unwrap_or(default_settings_path()?);
    let root = root.canonicalize()?;
    let executable = env::current_exe()?;
    let mut settings = read_json_object(&path)?;
    let hooks = settings.entry("lint-ai").or_insert_with(|| json!({}));
    let hooks = hooks
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("AGY hooks must be an object"))?;
    for (event, name) in [
        ("PreToolUse", "pre-tool-use"),
        ("PostToolUse", "post-tool-use"),
        ("PreInvocation", "pre-invocation"),
        ("PostInvocation", "post-invocation"),
        ("Stop", "stop"),
    ] {
        let entries = hooks.entry(event).or_insert_with(|| json!([]));
        let entries = entries
            .as_array_mut()
            .ok_or_else(|| anyhow::anyhow!("AGY hook {event} must be an array"))?;
        entries.retain(|entry| {
            !entry.as_object().is_some_and(Map::is_empty)
                && !serde_json::to_string(entry)
                    .unwrap_or_default()
                    .contains(HOOK_MARKER)
        });
        let command = json!({"type":"command", "command": format!("{} {HOOK_MARKER} {name} {}", shell_quote(&executable.to_string_lossy()), shell_quote(&root.to_string_lossy()))});
        if matches!(event, "PreToolUse" | "PostToolUse") {
            entries.push(json!({"matcher":".*", "hooks":[command]}));
        } else {
            entries.push(command);
        }
    }
    write_json_object(&path, &settings)?;
    Ok(path)
}

pub fn run_server(root: &Path, options: AgyServerOptions<'_>) -> Result<()> {
    let _ = options;
    gemini_cli::run_server_for(
        root,
        RecordingProvider::Agy,
        "agy",
        "agy-memory",
        "agy-mcp-index",
    )
}

fn home_dir() -> Result<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("HOME or USERPROFILE is not set"))
}
fn default_config_path() -> Result<PathBuf> {
    Ok(home_dir()?.join(".gemini/config/mcp_config.json"))
}
fn default_settings_path() -> Result<PathBuf> {
    Ok(home_dir()?.join(".gemini/config/hooks.json"))
}
fn read_json_object(path: &Path) -> Result<Map<String, Value>> {
    if !path.exists() {
        return Ok(Map::new());
    }
    let contents = fs::read_to_string(path)?;
    if contents.trim().is_empty() {
        return Ok(Map::new());
    }
    Ok(serde_json::from_str(&contents)?)
}
fn write_json_object(path: &Path, value: &Map<String, Value>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(value)? + "\n")?;
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root() -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("lint-ai-agy-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn uses_agy_configuration_paths() {
        let root = PathBuf::from("/tmp/project");
        assert!(default_config_path()
            .unwrap()
            .to_string_lossy()
            .contains("mcp_config.json"));
        assert_eq!(HOOK_MARKER, "--agy-hook");
        let _ = root;
    }

    #[test]
    fn memory_skill_preserves_user_edits_without_force() {
        let root = temp_root();
        let path = root.join(".agents/skills/lint-ai-memory/SKILL.md");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "custom").unwrap();

        let error = install_memory_skill(&root, false).unwrap_err().to_string();
        assert!(error.contains("--agy-force-skill"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "custom");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn memory_skill_force_replaces_user_edits_and_is_idempotent() {
        let root = temp_root();

        install_memory_skill(&root, true).unwrap();
        let path = root.join(".agents/skills/lint-ai-memory/SKILL.md");
        let installed = fs::read_to_string(&path).unwrap();
        assert!(installed.contains("<!-- lint-ai-managed-skill -->"));
        install_memory_skill(&root, false).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), installed);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn installation_is_idempotent_and_preserves_existing_configuration() {
        let root = temp_root();
        let config = root.join("mcp-config.json");
        let settings = root.join("settings.json");
        fs::write(&config, r#"{"mcpServers":{"other":{"command":"other"}}}"#).unwrap();
        fs::write(
            &settings,
            r#"{"theme":"dark","hooks":{"BeforeAgent":[{}, {"hooks":[{"type":"command","command":"user-hook"}]}]}}"#,
        )
        .unwrap();

        install_user_config(&root, Some(&config)).unwrap();
        install_user_config(&root, Some(&config)).unwrap();
        install_hook_settings(&root, Some(&settings)).unwrap();
        install_hook_settings(&root, Some(&settings)).unwrap();

        let config: Value = serde_json::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert_eq!(config["mcpServers"]["other"]["command"], "other");
        assert_eq!(config["mcpServers"]["lint-ai"]["args"][0], "--agy-serve");
        assert_eq!(
            config["mcpServers"]["lint-ai"]["args"][1],
            root.to_string_lossy().as_ref()
        );
        let settings: Value =
            serde_json::from_str(&fs::read_to_string(&settings).unwrap()).unwrap();
        assert_eq!(settings["theme"], "dark");
        assert_eq!(settings["lint-ai"]["Stop"].as_array().unwrap().len(), 1);
        assert_eq!(
            settings["lint-ai"]["PreToolUse"].as_array().unwrap().len(),
            1
        );
        fs::remove_dir_all(root).unwrap();
    }
}
