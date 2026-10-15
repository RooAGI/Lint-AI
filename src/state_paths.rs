use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const RESIDENT_HTTP_PORT_START: u16 = 8465;
const RESIDENT_HTTP_PORT_END: u16 = 9999;

/// Initialize project-local Membrick state and install default server settings
/// without replacing settings the user has already created.
pub(crate) fn initialize_project_state(workspace: &Path) -> anyhow::Result<PathBuf> {
    let state = migrate_legacy_state_dir(workspace)?;
    fs::create_dir_all(&state)?;
    let settings = state.join("server.toml");
    if settings.exists() {
        return Ok(settings);
    }

    let port = available_resident_http_port(workspace)?;
    let defaults = format!(
        "# Membrick local server settings. Existing files are preserved by install.\n\
bind = \"127.0.0.1:8080\"\n\
resident_http_bind = \"127.0.0.1:{port}\"\n\
resident_http_auto_port = true\n\
dashboard_url = \"http://127.0.0.1:{port}/dashboard\"\n"
    );
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let temporary = state.join(format!(".server.toml.{}.{}.tmp", std::process::id(), nonce));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    if let Err(error) = std::io::Write::write_all(&mut file, defaults.as_bytes()) {
        drop(file);
        let _ = fs::remove_file(&temporary);
        return Err(error.into());
    }
    drop(file);
    match fs::hard_link(&temporary, &settings) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            return Err(error.into());
        }
    }
    fs::remove_file(temporary)?;
    Ok(settings)
}

fn preferred_resident_http_port(workspace: &Path) -> u16 {
    let canonical = workspace
        .canonicalize()
        .unwrap_or_else(|_| workspace.to_path_buf());
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in canonical.to_string_lossy().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let count = u32::from(RESIDENT_HTTP_PORT_END - RESIDENT_HTTP_PORT_START) + 1;
    RESIDENT_HTTP_PORT_START + (hash % u64::from(count)) as u16
}

fn available_resident_http_port(workspace: &Path) -> anyhow::Result<u16> {
    let first = preferred_resident_http_port(workspace);
    let count = u32::from(RESIDENT_HTTP_PORT_END - RESIDENT_HTTP_PORT_START) + 1;
    for offset in 0..count {
        let port = RESIDENT_HTTP_PORT_START
            + ((u32::from(first - RESIDENT_HTTP_PORT_START) + offset) % count) as u16;
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return Ok(port);
        }
    }
    anyhow::bail!(
        "no available loopback resident HTTP port in {}..={}",
        RESIDENT_HTTP_PORT_START,
        RESIDENT_HTTP_PORT_END
    )
}

/// Move the previous product's workspace state directory to its current name.
/// If both names exist, merge non-conflicting top-level entries and archive
/// colliding legacy entries without overwriting current Membrick data.
pub(crate) fn migrate_legacy_state_dir(workspace: &Path) -> anyhow::Result<PathBuf> {
    let current = workspace.join(".membrick");
    let legacy = workspace.join(".lint-ai");

    match (current.exists(), legacy.exists()) {
        (false, true) => fs::rename(&legacy, &current).map_err(|error| {
            anyhow::anyhow!(
                "could not migrate legacy Membrick state from {} to {}: {error}",
                legacy.display(),
                current.display()
            )
        })?,
        (true, true) => merge_legacy_state(&current, &legacy)?,
        _ => {}
    }

    Ok(current)
}

fn merge_legacy_state(current: &Path, legacy: &Path) -> anyhow::Result<()> {
    let entries = fs::read_dir(legacy)
        .map_err(|error| {
            anyhow::anyhow!("could not read legacy state {}: {error}", legacy.display())
        })?
        .collect::<Result<Vec<_>, _>>()?;
    let conflicts = entries
        .iter()
        .filter(|entry| fs::symlink_metadata(current.join(entry.file_name())).is_ok())
        .map(|entry| entry.file_name())
        .collect::<Vec<_>>();
    let archive = if conflicts.is_empty() {
        None
    } else {
        Some(create_conflict_archive(current)?)
    };

    for entry in entries {
        let name = entry.file_name();
        let target = if conflicts.contains(&name) {
            archive
                .as_ref()
                .expect("conflicts have an archive")
                .join(&name)
        } else {
            current.join(&name)
        };
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::rename(entry.path(), &target).map_err(|error| {
            anyhow::anyhow!(
                "could not preserve legacy state {} at {}: {error}",
                entry.path().display(),
                target.display()
            )
        })?;
    }
    fs::remove_dir(legacy).map_err(|error| {
        anyhow::anyhow!(
            "could not remove emptied legacy state {}: {error}",
            legacy.display()
        )
    })?;
    Ok(())
}

fn create_conflict_archive(current: &Path) -> anyhow::Result<PathBuf> {
    let archive_root = current.join("migration-archive");
    fs::create_dir_all(&archive_root)?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    for suffix in 0..1000u16 {
        let name = if suffix == 0 {
            format!("lint-ai-{timestamp}")
        } else {
            format!("lint-ai-{timestamp}-{suffix}")
        };
        let path = archive_root.join(name);
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    anyhow::bail!(
        "could not allocate a unique legacy-state archive under {}",
        archive_root.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    fn temp_workspace() -> PathBuf {
        let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("membrick-state-paths-{}-{id}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn merges_non_conflicting_legacy_state_and_archives_conflicts() {
        let workspace = temp_workspace();
        let current = workspace.join(".membrick");
        let legacy = workspace.join(".lint-ai");
        fs::create_dir_all(current.join("provider-telemetry")).unwrap();
        fs::create_dir_all(legacy.join("provider-telemetry")).unwrap();
        fs::create_dir_all(legacy.join("memory")).unwrap();
        fs::write(current.join("provider-telemetry/claude.json"), "current").unwrap();
        fs::write(legacy.join("provider-telemetry/claude.json"), "legacy").unwrap();
        fs::write(legacy.join("memory/decision.md"), "decision").unwrap();

        assert_eq!(migrate_legacy_state_dir(&workspace).unwrap(), current);
        assert_eq!(
            fs::read_to_string(current.join("provider-telemetry/claude.json")).unwrap(),
            "current"
        );
        assert_eq!(
            fs::read_to_string(current.join("memory/decision.md")).unwrap(),
            "decision"
        );
        let archived = current.join("migration-archive");
        let archived_legacy = fs::read_dir(archived)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path()
            .join("provider-telemetry/claude.json");
        assert_eq!(fs::read_to_string(archived_legacy).unwrap(), "legacy");
        assert!(!legacy.exists());
        let _ = fs::remove_dir_all(workspace);
    }

    #[test]
    fn concurrent_installers_create_one_complete_settings_file() {
        let workspace = temp_workspace();
        let first_root = workspace.clone();
        let second_root = workspace.clone();
        let first = std::thread::spawn(move || initialize_project_state(&first_root).unwrap());
        let second = std::thread::spawn(move || initialize_project_state(&second_root).unwrap());
        let first_settings = first.join().unwrap();
        let second_settings = second.join().unwrap();
        assert_eq!(first_settings, second_settings);
        let parsed: toml::Value = fs::read_to_string(&first_settings)
            .unwrap()
            .parse()
            .unwrap();
        assert!(parsed["resident_http_bind"].as_str().is_some());
        assert!(parsed["dashboard_url"].as_str().is_some());
        fs::remove_dir_all(workspace).unwrap();
    }

    #[test]
    fn project_http_port_candidates_are_stable_and_project_specific() {
        let first = temp_workspace();
        let second = temp_workspace();
        let first_port = preferred_resident_http_port(&first);
        assert_eq!(first_port, preferred_resident_http_port(&first));
        assert_ne!(first_port, preferred_resident_http_port(&second));
        let _ = fs::remove_dir_all(first);
        let _ = fs::remove_dir_all(second);
    }

    #[test]
    fn initializes_default_server_settings_once_and_preserves_custom_settings() {
        let workspace = temp_workspace();
        let settings = initialize_project_state(&workspace).unwrap();
        let raw = fs::read_to_string(&settings).unwrap();
        let parsed: toml::Value = toml::from_str(&raw).unwrap();
        assert_eq!(parsed["bind"].as_str(), Some("127.0.0.1:8080"));
        let bind = parsed["resident_http_bind"].as_str().unwrap();
        let port = bind.strip_prefix("127.0.0.1:").unwrap();
        let port = port.parse::<u16>().unwrap();
        assert!((RESIDENT_HTTP_PORT_START..=RESIDENT_HTTP_PORT_END).contains(&port));
        assert!(parsed["resident_http_auto_port"].as_bool().unwrap());
        assert_eq!(
            parsed["dashboard_url"].as_str(),
            Some(format!("http://127.0.0.1:{port}/dashboard").as_str())
        );

        let custom = r#"resident_http_bind = "127.0.0.1:9999"
"#;
        fs::write(&settings, custom).unwrap();
        initialize_project_state(&workspace).unwrap();
        assert_eq!(fs::read_to_string(&settings).unwrap(), custom);
        fs::remove_dir_all(workspace).unwrap();
    }
}
