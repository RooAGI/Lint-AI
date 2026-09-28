//! Hermes Agent integration. The MCP transport is the shared stdio server (same
//! contract as the Gemini CLI / Agy / OpenClaw adapters): Hermes consumes
//! lint-ai as a stdio MCP server registered under `mcp_servers` in
//! `~/.hermes/config.yaml`. Tools surface to the Hermes agent as
//! `mcp__lint-ai__*`.
//!
//! Hooks are deliberately out of scope for v1: per project policy, lifecycle
//! hook entry schemas are probed against a live host binary before shipping.
//! (Hermes does expose plugin hooks and a pluggable memory-provider
//! interface; those are future integration levels, not this adapter.)

use crate::integrations::gemini_cli::{self, GeminiCliServerOptions};
use crate::integrations::session_recording::RecordingProvider;
use anyhow::{Context, Result};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const SERVER_NAME: &str = "lint-ai";

pub type HermesServerOptions<'a> = GeminiCliServerOptions<'a>;

/// Install the lint-ai memory skill for Hermes.
///
/// Hermes loads skills from project skill directories first, then from the
/// active skills directory (`~/.hermes/skills/`). The skill is host-global
/// (unlike the per-project MCP server entry), so it defaults to the
/// user-global skills directory. Pass `skill_dir` to override (tests,
/// unusual layouts).
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
                "refusing to overwrite existing Hermes skill {}; it appears user-modified; use --hermes-force-skill to replace it",
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

/// Merge the lint-ai stdio MCP server entry into the Hermes config.
///
/// Hermes' config is YAML (`~/.hermes/config.yaml`). lint-ai ships no YAML
/// parser, so the merge is line-based and conservative: it only manages the
/// top-level `mcp_servers:` mapping and the `lint-ai:` entry inside it, in
/// the canonical two-space block style Hermes itself emits (entry indent is
/// detected from existing servers when present). Anything shaped differently
/// fails with the manual-install recipe instead of corrupting the file.
pub fn install_user_config(root: &Path, config_path: Option<&Path>) -> Result<PathBuf> {
    let path = config_path
        .map(Path::to_path_buf)
        .unwrap_or(default_config_path()?);
    let root = root.canonicalize()?;
    let executable = env::current_exe()?;
    let existing = read_optional_text(&path)?;
    let merged = merge_mcp_servers_yaml(existing.as_deref(), &root, &executable).with_context(|| {
        format!(
            "could not merge into {} (unexpected YAML shape); add the server manually instead:\n  \
             hermes mcp add lint-ai --command {} --args --hermes-serve --args {}\n  \
             hermes mcp test lint-ai",
            path.display(),
            executable.display(),
            root.display()
        )
    })?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, merged)?;
    Ok(path)
}

pub fn run_server(root: &Path, options: HermesServerOptions<'_>) -> Result<()> {
    gemini_cli::run_server_for(root, RecordingProvider::Hermes, "hermes", "Hermes", options)
}

fn home_dir() -> Result<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("HOME or USERPROFILE is not set"))
}
fn default_config_path() -> Result<PathBuf> {
    Ok(home_dir()?.join(".hermes/config.yaml"))
}
fn default_skill_dir() -> Result<PathBuf> {
    Ok(home_dir()?.join(".hermes/skills"))
}
fn read_optional_text(path: &Path) -> Result<Option<String>> {
    if !path.exists() {
        return Ok(None);
    }
    let contents = fs::read_to_string(path)?;
    if contents.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some(contents))
}

/// Quote a string as a YAML double-quoted scalar.
fn yaml_string(value: &str) -> String {
    format!(
        "\"{}\"",
        value.replace('\\', "\\\\").replace('"', "\\\"")
    )
}

/// Render the `lint-ai:` server block at the given entry indent.
fn server_block(name: &str, indent: &str, executable: &Path, root: &Path) -> String {
    format!(
        "{indent}{name}:\n\
         {indent}  command: {command}\n\
         {indent}  args: [\"--hermes-serve\", {root}]\n",
        indent = indent,
        name = name,
        command = yaml_string(&executable.to_string_lossy()),
        root = yaml_string(&root.to_string_lossy()),
    )
}

/// True for lines that continue a YAML block mapping entry: blank lines or
/// lines indented deeper than `indent_width`.
fn is_continuation(line: &str, indent_width: usize) -> bool {
    if line.trim().is_empty() {
        return true;
    }
    line.len() - line.trim_start().len() > indent_width
}

fn merge_mcp_servers_yaml(
    existing: Option<&str>,
    root: &Path,
    executable: &Path,
) -> Result<String> {
    let block_for = |indent: &str| server_block(SERVER_NAME, indent, executable, root);
    let Some(text) = existing else {
        return Ok(format!("mcp_servers:\n{}", block_for("  ")));
    };
    if text.contains('\t') {
        anyhow::bail!("config contains tab indentation");
    }
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    // Top-level `mcp_servers:` key only. An empty inline value (`{}`, `null`)
    // is normalized to a block mapping; a non-empty inline value is not a
    // mapping we can merge into and fails with the manual recipe.
    let key_idx = lines.iter().position(|line| {
        line.starts_with("mcp_servers:") && {
            let rest = line["mcp_servers:".len()..].trim();
            rest.is_empty()
                || rest.starts_with('#')
                || rest == "{}"
                || rest == "null"
                || rest == "~"
        }
    });
    if key_idx.is_none()
        && lines.iter().any(|line| {
            line.starts_with("mcp_servers:")
                && !line["mcp_servers:".len()..].trim().is_empty()
                && !line["mcp_servers:".len()..].trim().starts_with('#')
        })
    {
        anyhow::bail!("mcp_servers has an inline value that is not an empty mapping");
    }
    let Some(key_idx) = key_idx else {
        // No mcp_servers section: append one at the end of the file.
        let mut out = text.to_string();
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("mcp_servers:\n");
        out.push_str(&block_for("  "));
        return Ok(out);
    };
    // Normalize an empty inline value (`mcp_servers: {}`) to a block mapping
    // before merging.
    if lines[key_idx] != "mcp_servers:" {
        lines[key_idx] = "mcp_servers:".to_string();
    }
    // Section = indented lines following the key, up to the next
    // top-level (unindented, non-blank) line or EOF.
    let mut section_end = key_idx + 1;
    while section_end < lines.len() && is_continuation(&lines[section_end], 0) {
        section_end += 1;
    }
    // Detect the entry indent from the first server entry in the section;
    // fall back to the canonical two spaces.
    let mut entry_indent = "  ".to_string();
    for line in &lines[key_idx + 1..section_end] {
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.contains(':') {
            entry_indent = line[..line.len() - trimmed.len()].to_string();
            break;
        }
    }
    let entry_indent_width = entry_indent.len();
    // Find an existing `lint-ai:` entry in the section.
    let entry_prefix = format!("{entry_indent}{SERVER_NAME}:");
    let mut entry_idx = None;
    let mut entry_inline = false;
    for (i, line) in lines
        .iter()
        .enumerate()
        .take(section_end)
        .skip(key_idx + 1)
    {
        if line.starts_with(&entry_prefix) {
            let rest = line[entry_prefix.len()..].trim();
            entry_idx = Some(i);
            entry_inline = !(rest.is_empty() || rest.starts_with('#'));
            break;
        }
    }
    let replacement: Vec<String> = block_for(&entry_indent)
        .lines()
        .map(str::to_string)
        .collect();
    match entry_idx {
        None => {
            // Insert right after the key (before any existing servers).
            lines.splice(key_idx + 1..key_idx + 1, replacement);
        }
        Some(i) => {
            let mut block_end = i + 1;
            if !entry_inline {
                while block_end < section_end
                    && is_continuation(&lines[block_end], entry_indent_width)
                {
                    block_end += 1;
                }
            }
            lines.splice(i..block_end, replacement);
        }
    }
    let mut out = lines.join("\n");
    out.push('\n');
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("lint-ai-hermes-{label}-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn uses_hermes_configuration_paths() {
        assert!(default_config_path()
            .unwrap()
            .to_string_lossy()
            .ends_with(".hermes/config.yaml"));
        assert!(default_skill_dir()
            .unwrap()
            .to_string_lossy()
            .ends_with(".hermes/skills"));
    }

    #[test]
    fn memory_skill_preserves_user_edits_without_force() {
        let root = temp_root("skill");
        let skill_dir = root.join("skills");
        let path = skill_dir.join("lint-ai-memory/SKILL.md");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "custom").unwrap();

        let error = install_memory_skill(Some(&skill_dir), false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("--hermes-force-skill"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "custom");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn memory_skill_force_replaces_user_edits_and_is_idempotent() {
        let root = temp_root("skill-force");
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
        let root = temp_root("config");
        let config = root.join("config.yaml");
        fs::write(
            &config,
            "theme: dark\nmcp_servers:\n  other:\n    command: other\n",
        )
        .unwrap();

        install_user_config(&root, Some(&config)).unwrap();
        install_user_config(&root, Some(&config)).unwrap();

        let merged = fs::read_to_string(&config).unwrap();
        assert!(merged.contains("theme: dark"));
        assert!(merged.contains("  other:\n    command: other"));
        // Only the verified command/args keys are written.
        assert!(merged.contains("  lint-ai:\n    command: "));
        assert!(merged.contains("\"--hermes-serve\""));
        assert!(merged.contains(&yaml_string(&root.to_string_lossy())));
        assert_eq!(merged.matches("  lint-ai:").count(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn creates_config_when_missing() {
        let root = temp_root("config-new");
        let config = root.join("sub").join("config.yaml");

        install_user_config(&root, Some(&config)).unwrap();

        let merged = fs::read_to_string(&config).unwrap();
        assert!(merged.starts_with("mcp_servers:\n"));
        assert!(merged.contains("\"--hermes-serve\""));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn appends_mcp_servers_section_when_absent() {
        let root = temp_root("config-append");
        let config = root.join("config.yaml");
        fs::write(&config, "memory:\n  provider: mem0\n").unwrap();

        install_user_config(&root, Some(&config)).unwrap();

        let merged = fs::read_to_string(&config).unwrap();
        assert!(merged.contains("memory:\n  provider: mem0"));
        assert!(merged.contains("\nmcp_servers:\n  lint-ai:\n"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn empty_inline_mcp_servers_value_is_normalized_and_merged() {
        let root = temp_root("config-inline-empty");
        let config = root.join("config.yaml");
        fs::write(&config, "mcp_servers: {}\n").unwrap();

        install_user_config(&root, Some(&config)).unwrap();

        let merged = fs::read_to_string(&config).unwrap();
        assert!(merged.contains("mcp_servers:\n  lint-ai:\n"));
        assert!(merged.contains("\"--hermes-serve\""));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn inline_mcp_servers_value_fails_with_manual_install_recipe() {
        let root = temp_root("config-inline");
        let config = root.join("config.yaml");
        let original = "mcp_servers: {other: {command: other}}\n";
        fs::write(&config, original).unwrap();

        let error = install_user_config(&root, Some(&config))
            .unwrap_err()
            .to_string();
        assert!(error.contains("hermes mcp add lint-ai"));
        assert!(error.contains("hermes mcp test lint-ai"));
        assert_eq!(fs::read_to_string(&config).unwrap(), original);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn provider_strings_match_storage_layout() {
        assert_eq!(RecordingProvider::Hermes.as_str(), "hermes");
    }
}
