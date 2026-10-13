use std::fs;
use std::path::{Path, PathBuf};

/// Move the previous product's workspace state directory to its current name.
/// Existing files and index contents are preserved with a filesystem rename.
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
        (true, true) => anyhow::bail!(
            "both {} and {} exist; move or merge the legacy state before starting Membrick",
            current.display(),
            legacy.display()
        ),
        _ => {}
    }

    Ok(current)
}
