use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

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
}
