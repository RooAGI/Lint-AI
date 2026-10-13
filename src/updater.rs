use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const CHECK_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
const MANAGED_MARKER: &str = ".membrick-managed";
const LAST_CHECK: &str = ".membrick-last-update-check";

/// Check for a release only for an official installer-managed binary in an
/// interactive terminal. The updater prompts before replacing the executable.
pub(crate) fn check_for_update() {
    if !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
        return;
    }

    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| {
        matches!(
            arg.as_str(),
            "--help" | "-h" | "--version" | "-V" | "--no-update-check"
        ) || arg.contains("-serve")
            || arg.contains("-hook")
            || arg.contains("-install")
            || arg.contains("-verify-mcp")
    }) {
        return;
    }

    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let Some(parent) = exe.parent() else {
        return;
    };
    let Some(file_name) = exe.file_name().and_then(|name| name.to_str()) else {
        return;
    };
    if !matches!(file_name, "membrick" | "membrick.exe") || !parent.join(MANAGED_MARKER).is_file() {
        return;
    }

    let checked_at = parent.join(LAST_CHECK);
    if !check_is_due(&checked_at) {
        return;
    }
    // Save before making the network request so an unavailable service does
    // not cause another request on every CLI invocation.
    let _ = std::fs::write(&checked_at, now_seconds().to_string());

    if let Err(error) = update_installed_binary(&exe) {
        eprintln!("Membrick update check failed: {error}");
    }
}

fn check_is_due(path: &Path) -> bool {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return true;
    };
    let Ok(previous) = contents.trim().parse::<u64>() else {
        return true;
    };
    now_seconds().saturating_sub(previous) >= CHECK_INTERVAL.as_secs()
}

fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn update_installed_binary(exe: &Path) -> anyhow::Result<()> {
    let asset_identifier = platform_assets()?;
    let mut update = self_update::backends::github::Update::configure();
    update
        .repo_owner("RooAGI")
        .repo_name("Membrick")
        .bin_name("membrick")
        .bin_install_path(PathBuf::from(exe))
        .current_version(env!("CARGO_PKG_VERSION"))
        .asset_identifier(asset_identifier)
        .checksum_from_asset("SHA256SUMS")
        .show_download_progress(true)
        .show_output(true)
        .no_confirm(false)
        .timeout(Duration::from_secs(20));

    let status = update.build()?.update()?;
    if status.version() != env!("CARGO_PKG_VERSION") {
        eprintln!(
            "Membrick updated to {}. Restart Membrick to use the new version.",
            status.version()
        );
    }
    Ok(())
}

fn platform_assets() -> anyhow::Result<&'static str> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        return Ok("linux-x86_64");
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        return Ok("macos-x86_64");
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        return Ok("macos-aarch64");
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        return Ok("windows-x86_64");
    }
    #[allow(unreachable_code)]
    Err(anyhow::anyhow!(
        "automatic updates are unavailable for this platform"
    ))
}
