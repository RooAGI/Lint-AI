//! OpenClaw integration. The MCP transport is the shared stdio server (same
//! contract as the Gemini CLI / Agy adapters): OpenClaw consumes lint-ai as a
//! stdio MCP server registered under `mcp.servers` in `~/.openclaw/openclaw.json`.
//!
//! Hooks are deliberately out of scope for v1: per project policy, lifecycle
//! hook entry schemas are probed against a live host binary before shipping.

use crate::integrations::gemini_cli::{self, GeminiCliServerOptions};
use crate::integrations::session_recording::RecordingProvider;
use anyhow::{Context, Result};
use serde_json::{json, Map, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const SERVER_NAME: &str = "lint-ai";

pub type OpenClawServerOptions<'a> = GeminiCliServerOptions<'a>;

/// Install the lint-ai memory skill for OpenClaw.
///
/// OpenClaw loads skills from project `skills/` directories first, then from
/// `~/.openclaw/skills/`. The skill is host-global (unlike the per-project MCP
/// server entry), so it defaults to the user-global skills directory. Pass
/// `skill_dir` to override (tests, unusual layouts).
pub fn install_memory_skill(skill_dir: Option<&Path>, force: bool) -> Result<PathBuf> {
    let skill_path = skill_dir
        .map(Path::to_path_buf)
        .unwrap_or(default_skill_dir()?)
        .join("lint-ai-memory")
        .join("SKILL.md");
    if let Some(parent) = skill_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let generated = include_str!("skill.md");
    if let Ok(existing) = fs::read_to_string(&skill_path) {
        if !force && existing != generated {
            anyhow::bail!(
                "refusing to overwrite existing OpenClaw skill {}; it appears user-modified; use --openclaw-force-skill to replace it",
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

/// Merge the lint-ai stdio MCP server entry into the OpenClaw config.
///
/// Only the `command`/`args` keys verified against OpenClaw's MCP docs are
/// written; OpenClaw validates its config strictly, so no speculative keys.
pub fn install_user_config(root: &Path, config_path: Option<&Path>) -> Result<PathBuf> {
    let path = config_path
        .map(Path::to_path_buf)
        .unwrap_or(default_config_path()?);
    let root = root.canonicalize()?;
    let executable = env::current_exe()?;
    let mut settings = read_json_object(&path)?;
    let mcp = settings.entry("mcp").or_insert_with(|| json!({}));
    let mcp = mcp
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("OpenClaw mcp must be an object"))?;
    let servers = mcp.entry("servers").or_insert_with(|| json!({}));
    let servers = servers
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("OpenClaw mcp.servers must be an object"))?;
    servers.insert(
        SERVER_NAME.into(),
        json!({
            "command": executable,
            "args": ["--openclaw-serve", root.to_string_lossy()]
        }),
    );
    write_json_object(&path, &settings)?;
    Ok(path)
}

pub fn run_server(root: &Path, options: OpenClawServerOptions<'_>) -> Result<()> {
    gemini_cli::run_server_for(root, RecordingProvider::OpenClaw, "openclaw", options)
}

fn home_dir() -> Result<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("HOME or USERPROFILE is not set"))
}
fn default_config_path() -> Result<PathBuf> {
    Ok(home_dir()?.join(".openclaw/openclaw.json"))
}
fn default_skill_dir() -> Result<PathBuf> {
    Ok(home_dir()?.join(".openclaw/skills"))
}
fn read_json_object(path: &Path) -> Result<Map<String, Value>> {
    if !path.exists() {
        return Ok(Map::new());
    }
    let contents = fs::read_to_string(path)?;
    if contents.trim().is_empty() {
        return Ok(Map::new());
    }
    // openclaw.json is JSON5 (comments, trailing commas, unquoted keys are
    // common). lint-ai only speaks strict JSON, so instead of corrupting a
    // JSON5 file by rewriting it as strict JSON, fail with the manual-install
    // recipe taken from OpenClaw's own MCP docs.
    serde_json::from_str(&contents).with_context(|| {
        format!(
            "could not parse {} as JSON (it may use JSON5 features like comments); add the server manually instead:\n  \
             openclaw mcp add lint-ai --command <lint-ai-binary> --arg --openclaw-serve --arg <project-root>\n  \
             openclaw mcp doctor lint-ai --probe",
            path.display()
        )
    })
}
fn write_json_object(path: &Path, value: &Map<String, Value>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(value)? + "\n")?;
    Ok(())
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
            .join(format!("lint-ai-openclaw-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn uses_openclaw_configuration_paths() {
        assert!(default_config_path()
            .unwrap()
            .to_string_lossy()
            .ends_with(".openclaw/openclaw.json"));
        assert!(default_skill_dir()
            .unwrap()
            .to_string_lossy()
            .ends_with(".openclaw/skills"));
    }

    #[test]
    fn memory_skill_preserves_user_edits_without_force() {
        let root = temp_root();
        let skill_dir = root.join("skills");
        let path = skill_dir.join("lint-ai-memory/SKILL.md");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "custom").unwrap();

        let error = install_memory_skill(Some(&skill_dir), false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("--openclaw-force-skill"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "custom");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn memory_skill_force_replaces_user_edits_and_is_idempotent() {
        let root = temp_root();
        let skill_dir = root.join("skills");

        install_memory_skill(Some(&skill_dir), true).unwrap();
        let path = skill_dir.join("lint-ai-memory/SKILL.md");
        let installed = fs::read_to_string(&path).unwrap();
        assert!(installed.contains("<!-- lint-ai-managed-skill -->"));
        install_memory_skill(Some(&skill_dir), false).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), installed);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn installation_is_idempotent_and_preserves_existing_configuration() {
        let root = temp_root();
        let config = root.join("openclaw.json");
        fs::write(
            &config,
            r#"{"mcp":{"servers":{"other":{"command":"other"}}}}"#,
        )
        .unwrap();

        install_user_config(&root, Some(&config)).unwrap();
        install_user_config(&root, Some(&config)).unwrap();

        let config: Value = serde_json::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert_eq!(config["mcp"]["servers"]["other"]["command"], "other");
        assert_eq!(config["mcp"]["servers"]["lint-ai"]["args"][0], "--openclaw-serve");
        assert_eq!(
            config["mcp"]["servers"]["lint-ai"]["args"][1],
            root.to_string_lossy().as_ref()
        );
        // Only the verified command/args keys are written; OpenClaw validates
        // its config strictly, so no speculative keys.
        let entry = config["mcp"]["servers"]["lint-ai"].as_object().unwrap();
        assert!(entry.len() == 2 && entry.contains_key("command") && entry.contains_key("args"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn json5_config_is_not_rewritten_and_points_at_manual_install() {
        let root = temp_root();
        let config = root.join("openclaw.json");
        let original = "// comment\n{ mcp: { servers: {}, }, }\n";
        fs::write(&config, original).unwrap();

        let error = install_user_config(&root, Some(&config))
            .unwrap_err()
            .to_string();
        assert!(error.contains("openclaw mcp add lint-ai"));
        assert!(error.contains("openclaw mcp doctor lint-ai --probe"));
        assert_eq!(fs::read_to_string(&config).unwrap(), original);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn provider_strings_match_storage_layout() {
        assert_eq!(RecordingProvider::OpenClaw.as_str(), "openclaw");
    }
}
