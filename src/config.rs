use serde::Deserialize;
use serde_json;
use std::fs;
use std::path::Path;

#[derive(Debug, Default, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub stopwords: Vec<String>,
    #[serde(default)]
    pub ignore_sections: Vec<String>,
    #[serde(default)]
    pub ignore_crossref_sections: Vec<String>,
    #[serde(default)]
    pub ignore_paths: Vec<String>,
    #[serde(default)]
    pub allowlist_concepts: Vec<String>,
    #[serde(default)]
    pub scope_prefix: Option<String>,
}

fn load_from_path(path: &Path, max_bytes: u64) -> Result<Config, String> {
    let meta = fs::metadata(path).map_err(|e| e.to_string())?;
    if meta.len() > max_bytes {
        return Err("config too large".to_string());
    }
    let raw = fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&raw).map_err(|e| e.to_string())
}

pub fn load_config(
    config_path: Option<&str>,
    target_path: &str,
    strict: bool,
    max_bytes: u64,
) -> Result<Config, String> {
    if let Some(path) = config_path {
        match load_from_path(Path::new(path), max_bytes) {
            Ok(cfg) => return Ok(cfg),
            Err(err) => {
                if strict {
                    return Err(format!("Failed to load config at {}: {}", path, err));
                }
                return Err(format!(
                    "Failed to load config at {}: {} (use --strict-config to fail or fix the file)",
                    path, err
                ));
            }
        }
    }

    let target = Path::new(target_path);
    let root = if target.is_file() {
        target.parent().unwrap_or(target)
    } else {
        target
    };
    let candidate = root.join("lint-ai.json");
    match load_from_path(&candidate, max_bytes) {
        Ok(cfg) => return Ok(cfg),
        Err(err) => {
            if candidate.exists() {
                if strict {
                    return Err(format!(
                        "Failed to load config at {}: {}",
                        candidate.display(),
                        err
                    ));
                }
                return Err(format!(
                    "Failed to load config at {}: {} (use --strict-config to fail or fix the file)",
                    candidate.display(),
                    err
                ));
            }
        }
    }

    Ok(Config::default())
}

pub fn normalize_list(values: &[String]) -> Vec<String> {
    values
        .iter()
        .map(|v| v.trim().to_lowercase())
        .filter(|v| !v.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_list_trims_and_lowercases() {
        let values = vec![" Foo ".to_string(), "BAR".to_string(), "".to_string()];
        let out = normalize_list(&values);
        assert_eq!(out, vec!["foo".to_string(), "bar".to_string()]);
    }
}
