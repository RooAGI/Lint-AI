//! OpenClaw integration. The MCP transport is the shared stdio server (same
//! contract as the Gemini CLI / Agy adapters): OpenClaw consumes lint-ai as a
//! stdio MCP server registered under `mcp.servers` in `~/.openclaw/openclaw.json`.
//!
//! Lifecycle hooks are a second, complementary surface (verified against a
//! live 2026.9.6 host):
//!   - internal hooks (`agent:bootstrap` recall/injection) live under
//!     `<stateDir>/hooks/lint-ai/` (`OPENCLAW_STATE_DIR`, else `~/.openclaw`),
//!   - the typed plugin (`agent_end`/`before_reset`/`session_start`/
//!     `session_end`/shutdown capture) lives under
//!     `<configDir>/extensions/lint-ai/` (`~/.openclaw/extensions`).
//! Both are thin JS shims around `lint-ai --openclaw-hook <kind>`; all
//! lifecycle logic is in [`hooks`].

pub mod document;
pub mod hooks;

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
    gemini_cli::run_server_for(root, RecordingProvider::OpenClaw, "openclaw", "OpenClaw", options)
}

/// Install the internal-hook wrapper for recall/injection.
///
/// Internal hooks live under `<stateDir>/hooks/<name>/`, where stateDir is
/// `OPENCLAW_STATE_DIR` when set, else `~/.openclaw` (verified against the
/// 2026.9.6 runtime). The installed `handler.js` is a thin wrapper around
/// `lint-ai --openclaw-hook bootstrap`; `__LINT_AI_BIN__` is replaced with the
/// current binary path and `__LINT_AI_ROOT__` with the canonicalized project
/// root at install time.
pub fn install_hooks(hooks_dir: Option<&Path>, root: &Path, force: bool) -> Result<PathBuf> {
    let dir = hooks_dir
        .map(Path::to_path_buf)
        .unwrap_or(default_hooks_dir()?)
        .join("lint-ai");
    let bin = env::current_exe()?.to_string_lossy().into_owned();
    let root = root.canonicalize().with_context(|| {
        format!(
            "failed to canonicalize OpenClaw install root {}",
            root.display()
        )
    })?;
    let root = root.to_string_lossy().into_owned();
    write_install_asset(&dir.join("HOOK.md"), include_str!("hooks/HOOK.md"), force)?;
    write_install_asset(
        &dir.join("handler.js"),
        &include_str!("hooks/handler.js")
            .replace("__LINT_AI_BIN__", &bin)
            .replace("__LINT_AI_ROOT__", &root),
        force,
    )?;
    Ok(dir)
}

/// Install the typed plugin for lifecycle capture (`agent_end` → Outcome,
/// `before_reset` → SessionSummary).
///
/// Plugins are discovered under `<configDir>/extensions/` (`~/.openclaw`
/// by default; `OPENCLAW_STATE_DIR` overrides it the same way it overrides
/// the state dir). The plugin files are static — they ship as-is on ClawHub
/// and resolve the binary path and project root at runtime — so installation
/// is a verbatim copy; per-install configuration lives in the plugin config
/// entry written by [`install_plugin_config`].
pub fn install_plugin(plugin_dir: Option<&Path>, force: bool) -> Result<PathBuf> {
    let dir = plugin_dir
        .map(Path::to_path_buf)
        .unwrap_or(default_plugin_dir()?)
        .join("lint-ai");
    write_install_asset(
        &dir.join("openclaw.plugin.json"),
        include_str!("plugin/openclaw.plugin.json"),
        force,
    )?;
    write_install_asset(
        &dir.join("package.json"),
        include_str!("plugin/package.json"),
        force,
    )?;
    write_install_asset(
        &dir.join("index.js"),
        include_str!("plugin/index.js"),
        force,
    )?;
    Ok(dir)
}

/// Write the typed plugin's `projectRoot` into OpenClaw's plugin config
/// (`plugins.entries.lint-ai.config` in `openclaw.json`), so a local
/// `--openclaw-install` keeps working exactly as before without baked-in
/// paths. The plugin resolves the binary at runtime
/// (`binaryPath` → `LINT_AI_BIN` → `lint-ai` on `PATH`); only the project
/// root needs recording here, and the event `workspaceDir` still takes
/// precedence when present. Existing entry keys (e.g. `enabled`) are
/// preserved; only `config.projectRoot` is set.
pub fn install_plugin_config(root: &Path, config_path: Option<&Path>) -> Result<PathBuf> {
    let path = config_path
        .map(Path::to_path_buf)
        .unwrap_or(default_config_path()?);
    let root = root.canonicalize().with_context(|| {
        format!(
            "failed to canonicalize OpenClaw install root {}",
            root.display()
        )
    })?;
    let mut settings = read_json_object(&path)?;
    let plugins = settings.entry("plugins").or_insert_with(|| json!({}));
    let plugins = plugins
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("OpenClaw plugins must be an object"))?;
    let entries = plugins.entry("entries").or_insert_with(|| json!({}));
    let entries = entries
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("OpenClaw plugins.entries must be an object"))?;
    let entry = entries.entry(SERVER_NAME).or_insert_with(|| json!({}));
    let entry = entry.as_object_mut().ok_or_else(|| {
        anyhow::anyhow!("OpenClaw plugins.entries.lint-ai must be an object")
    })?;
    let config = entry.entry("config").or_insert_with(|| json!({}));
    let config = config.as_object_mut().ok_or_else(|| {
        anyhow::anyhow!("OpenClaw plugins.entries.lint-ai.config must be an object")
    })?;
    config.insert(
        "projectRoot".to_string(),
        json!(root.to_string_lossy()),
    );
    write_json_object(&path, &settings)?;
    Ok(path)
}

/// Write an install asset. Reinstalling is idempotent; an existing
/// file that is not byte-identical is only replaced with `force` or when it
/// is clearly ours (installed by a previous `--openclaw-install`).
fn write_install_asset(path: &Path, contents: &str, force: bool) -> Result<()> {
    if path.is_file() {
        let existing = fs::read_to_string(path).unwrap_or_default();
        if existing == contents {
            return Ok(());
        }
        if !force && !existing.contains("lint-ai --openclaw-install") {
            anyhow::bail!(
                "refusing to overwrite existing {}; it does not look like a lint-ai install; rerun with --openclaw-force-skill to replace it",
                path.display()
            );
        }
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, contents)?;
    Ok(())
}

/// OpenClaw's home: `OPENCLAW_STATE_DIR` when set, else `~/.openclaw`.
/// (The runtime resolves both the state dir and the config dir from the same
/// default; the plugin `extensions/` directory lives under the config dir.)
fn openclaw_home() -> Result<PathBuf> {
    if let Some(dir) = env::var_os("OPENCLAW_STATE_DIR").filter(|d| !d.is_empty()) {
        return Ok(PathBuf::from(dir));
    }
    home_dir().map(|home| home.join(".openclaw"))
}
fn default_hooks_dir() -> Result<PathBuf> {
    Ok(openclaw_home()?.join("hooks"))
}
fn default_plugin_dir() -> Result<PathBuf> {
    Ok(openclaw_home()?.join("extensions"))
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
        assert_eq!(
            config["mcp"]["servers"]["lint-ai"]["args"][0],
            "--openclaw-serve"
        );
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

    #[test]
    fn plugin_install_is_static_and_idempotent() {
        let root = temp_root();
        let plugin_dir = root.join("extensions");

        install_plugin(Some(&plugin_dir), false).unwrap();
        let dir = plugin_dir.join("lint-ai");
        // The published files ship verbatim: no install-time placeholders.
        let index = fs::read_to_string(dir.join("index.js")).unwrap();
        assert_eq!(index, include_str!("plugin/index.js"));
        assert!(!index.contains("__LINT_AI_BIN__"));
        assert!(!index.contains("__LINT_AI_ROOT__"));
        // Manifest carries the binary release version and the new config keys.
        let manifest: Value =
            serde_json::from_str(&fs::read_to_string(dir.join("openclaw.plugin.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["id"], "lint-ai");
        assert_eq!(manifest["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(
            manifest["configSchema"]["properties"]["projectRoot"]["type"],
            "string"
        );
        assert_eq!(
            manifest["configSchema"]["properties"]["binaryPath"]["type"],
            "string"
        );
        // Reinstall is a no-op.
        install_plugin(Some(&plugin_dir), false).unwrap();
        assert_eq!(
            fs::read_to_string(dir.join("index.js")).unwrap(),
            index
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn plugin_config_writes_project_root_and_preserves_entries() {
        let root = temp_root();
        let config = root.join("openclaw.json");
        fs::write(
            &config,
            r#"{"plugins":{"entries":{"lint-ai":{"enabled":true},"other":{}}}}"#,
        )
        .unwrap();

        install_plugin_config(&root, Some(&config)).unwrap();
        install_plugin_config(&root, Some(&config)).unwrap();

        let config: Value =
            serde_json::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        let entry = &config["plugins"]["entries"]["lint-ai"];
        // Existing entry keys survive; only config.projectRoot is set.
        assert_eq!(entry["enabled"], true);
        assert_eq!(
            entry["config"]["projectRoot"].as_str().unwrap(),
            root.canonicalize()
                .unwrap()
                .to_string_lossy()
                .as_ref()
        );
        assert!(config["plugins"]["entries"]["other"].is_object());
        fs::remove_dir_all(root).unwrap();
    }
}
