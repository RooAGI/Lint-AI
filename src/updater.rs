use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const CHECK_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
const CHECK_TIMEOUT: Duration = Duration::from_secs(10);
const MANAGED_MARKER: &str = ".membrick-managed";
const INSTALL_METHOD_MARKER: &str = "membrick-install-method";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct UpdateState {
    checked_at: u64,
    checked_current_version: String,
    latest_version: Option<String>,
    install_method: String,
    notified_version: Option<String>,
}

/// Start a best-effort, non-blocking release check for an MCP process.
/// The result is written to a per-installation user cache and surfaced by a
/// provider lifecycle hook on a later interaction.
pub(crate) fn schedule_daily_check() {
    if cfg!(debug_assertions) {
        return;
    }
    let Ok(executable) = std::env::current_exe() else {
        return;
    };
    let Some(method) = install_method(&executable) else {
        return;
    };
    let Some(cache_file) = state_path(&executable) else {
        return;
    };
    let current_version = env!("CARGO_PKG_VERSION").to_string();
    if !check_is_due(&cache_file, &current_version) {
        return;
    }
    let Some(lock) = try_lock(&cache_file) else {
        return;
    };
    let cache_file = cache_file.clone();
    std::thread::spawn(move || {
        let _lock = lock;
        let mut state = read_state(&cache_file).unwrap_or_default();
        match fetch_available_version(&current_version) {
            Ok(version) => {
                state.checked_at = now_seconds();
                state.checked_current_version = current_version;
                state.install_method = method;
                state.latest_version = version;
                if let Err(error) = write_state(&cache_file, &state) {
                    eprintln!("Membrick update state could not be saved: {error:#}");
                }
            }
            Err(error) => eprintln!("Membrick update check unavailable: {error:#}"),
        }
    });
}

/// Return and mark the per-installation update notice as shown. Hooks call
/// this only on SessionStart or UserPromptSubmit, so no network work occurs
/// in the host's synchronous hook path.
pub(crate) fn take_update_notice() -> Option<String> {
    let executable = std::env::current_exe().ok()?;
    let cache_file = state_path(&executable)?;
    let _lock = try_lock(&cache_file)?;
    let mut state = read_state(&cache_file)?;
    let current = env!("CARGO_PKG_VERSION");
    if state.checked_current_version != current {
        return None;
    }
    let latest = state.latest_version.as_deref()?;
    if latest == current || state.notified_version.as_deref() == Some(latest) {
        return None;
    }
    let method = if state.install_method.is_empty() {
        install_method(&executable).unwrap_or_else(|| "unknown".to_string())
    } else {
        state.install_method.clone()
    };
    let message = update_message(latest, current, &method);
    state.notified_version = Some(latest.to_string());
    write_state(&cache_file, &state).ok()?;
    Some(message)
}

fn fetch_available_version(current_version: &str) -> anyhow::Result<Option<String>> {
    let mut update = self_update::backends::github::Update::configure();
    update
        .repo_owner("RooAGI")
        .repo_name("Membrick")
        .bin_name("membrick")
        .current_version(current_version)
        .timeout(CHECK_TIMEOUT);
    let releases = update.build()?.get_latest_release()?;
    if releases.is_update_available()? {
        Ok(releases
            .latest()
            .map(|release| release.version().to_string()))
    } else {
        Ok(None)
    }
}

fn update_message(latest: &str, current: &str, method: &str) -> String {
    let command = match method {
        "npm" => "for a global install, run `npm update -g membrick`; for a local install, run `npm update membrick` in the project",
        "pipx" => "run `pipx upgrade membrick`",
        "pip" => "run `python -m pip install --upgrade membrick`",
        "cargo" => "run `cargo install membrick --locked --force`",
        "standalone" => "rerun the official Membrick installer for your platform",
        _ => "use the package manager you used to install Membrick",
    };
    format!(
        "Membrick {latest} is available (installed: {current}, source: {method}). Update guidance: {command}. Release notes: https://github.com/RooAGI/Membrick/releases/tag/v{latest}",
    )
}

fn install_method(executable: &Path) -> Option<String> {
    let parent = executable.parent()?;
    let path = executable.to_string_lossy();
    if path.contains("/pipx/venvs/") || path.contains("\\pipx\\venvs\\") {
        return Some("pipx".to_string());
    }
    let marker = parent.join(INSTALL_METHOD_MARKER);
    if let Ok(value) = fs::read_to_string(marker) {
        let method = value.trim();
        if matches!(method, "npm" | "pip" | "pipx" | "standalone" | "cargo") {
            return Some(method.to_string());
        }
    }
    if parent.join(MANAGED_MARKER).is_file() {
        return Some("standalone".to_string());
    }
    if path.contains("/.cargo/bin/") || path.contains("\\.cargo\\bin\\") {
        return Some("cargo".to_string());
    }
    None
}

fn check_is_due(cache_file: &Path, current_version: &str) -> bool {
    let Some(state) = read_state(cache_file) else {
        return true;
    };
    state.checked_current_version != current_version
        || now_seconds().saturating_sub(state.checked_at) >= CHECK_INTERVAL.as_secs()
}

fn cache_root() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME").filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(path).join("membrick"));
    }
    #[cfg(target_os = "macos")]
    if let Some(home) = std::env::var_os("HOME") {
        return Some(PathBuf::from(home).join("Library/Caches/Membrick"));
    }
    #[cfg(target_os = "windows")]
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        return Some(PathBuf::from(local).join("Membrick/cache"));
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/membrick"))
}

fn state_path(executable: &Path) -> Option<PathBuf> {
    use std::hash::{Hash, Hasher};
    let root = cache_root()?;
    let executable = executable
        .canonicalize()
        .unwrap_or_else(|_| executable.to_path_buf());
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    executable.hash(&mut hasher);
    Some(root.join(format!("update-{:016x}.json", hasher.finish())))
}

struct FileLock(PathBuf);

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn try_lock(cache_file: &Path) -> Option<FileLock> {
    let parent = cache_file.parent()?;
    fs::create_dir_all(parent).ok()?;
    let lock_path = cache_file.with_extension("lock");
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)
    {
        Ok(_) => Some(FileLock(lock_path)),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let stale = fs::metadata(&lock_path)
                .and_then(|metadata| metadata.modified())
                .ok()
                .and_then(|modified| modified.elapsed().ok())
                .is_some_and(|age| age >= Duration::from_secs(5 * 60));
            if stale {
                let _ = fs::remove_file(&lock_path);
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&lock_path)
                    .ok()
                    .map(|_| FileLock(lock_path))
            } else {
                None
            }
        }
        Err(_) => None,
    }
}

fn read_state(path: &Path) -> Option<UpdateState> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

fn write_state(path: &Path, state: &UpdateState) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("update cache path has no parent"))?;
    fs::create_dir_all(parent)?;
    let temp = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temp)?;
    serde_json::to_writer(&mut file, state)?;
    file.flush()?;
    file.sync_all()?;
    #[cfg(windows)]
    if path.exists() {
        fs::remove_file(path)?;
    }
    fs::rename(temp, path)?;
    Ok(())
}

fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Preserve the interactive standalone-binary updater used by direct CLI
/// invocations. MCP update discovery never replaces a binary.
pub(crate) fn check_for_update() {
    use std::io::IsTerminal;

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
    if !matches!(
        exe.file_name().and_then(|name| name.to_str()),
        Some("membrick" | "membrick.exe")
    ) || !parent.join(MANAGED_MARKER).is_file()
    {
        return;
    }
    let checked_at = parent.join(".membrick-last-update-check");
    let previous = fs::read_to_string(&checked_at)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or_default();
    if now_seconds().saturating_sub(previous) < CHECK_INTERVAL.as_secs() {
        return;
    }
    let _ = fs::write(&checked_at, now_seconds().to_string());
    if let Err(error) = update_installed_binary(&exe) {
        eprintln!("Membrick update check failed: {error:#}");
    }
}

fn update_installed_binary(exe: &Path) -> anyhow::Result<()> {
    let asset_identifier = platform_assets()?;
    let mut update = self_update::backends::github::Update::configure();
    update
        .repo_owner("RooAGI")
        .repo_name("Membrick")
        .bin_name("membrick")
        .bin_install_path(exe.to_path_buf())
        .current_version(env!("CARGO_PKG_VERSION"))
        .asset_identifier(asset_identifier)
        .checksum_from_asset("SHA256SUMS")
        .show_download_progress(true)
        .show_output(true)
        .no_confirm(false)
        .timeout(CHECK_TIMEOUT);
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
