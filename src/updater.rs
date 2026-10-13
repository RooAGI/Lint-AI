use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const CHECK_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
const CHECK_TIMEOUT: Duration = Duration::from_secs(10);
const MANAGED_MARKER: &str = ".membrick-managed";
const INSTALL_METHOD_MARKER: &str = "membrick-install-method";
#[cfg(windows)]
const PYTHON_COMMAND: &str = "python";
#[cfg(not(windows))]
const PYTHON_COMMAND: &str = "python3";

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
        if let Err(error) = run_release_check(
            &cache_file,
            &current_version,
            &method,
            now_seconds(),
            fetch_available_version,
        ) {
            eprintln!("Membrick update check unavailable: {error:#}");
        }
    });
}

/// Return and mark the per-installation update notice as shown. Hooks call
/// this only on SessionStart or UserPromptSubmit, so no network work occurs
/// in the host's synchronous hook path.
pub(crate) fn take_update_notice() -> Option<String> {
    let executable = std::env::current_exe().ok()?;
    let cache_file = state_path(&executable)?;
    take_update_notice_from(&cache_file, env!("CARGO_PKG_VERSION"), &executable)
}

fn take_update_notice_from(
    cache_file: &Path,
    current_version: &str,
    executable: &Path,
) -> Option<String> {
    let _lock = try_lock(&cache_file)?;
    let mut state = read_state(&cache_file)?;
    if state.checked_current_version != current_version {
        return None;
    }
    let latest = state.latest_version.as_deref()?;
    if latest == current_version || state.notified_version.as_deref() == Some(latest) {
        return None;
    }
    let method = if state.install_method.is_empty() {
        install_method(&executable).unwrap_or_else(|| "unknown".to_string())
    } else {
        state.install_method.clone()
    };
    let message = update_message(latest, current_version, &method);
    state.notified_version = Some(latest.to_string());
    write_state(&cache_file, &state).ok()?;
    Some(message)
}

fn run_release_check(
    cache_file: &Path,
    current_version: &str,
    method: &str,
    checked_at: u64,
    fetch: impl FnOnce(&str) -> anyhow::Result<Option<String>>,
) -> anyhow::Result<()> {
    let latest_version = fetch(current_version)?;
    let mut state = read_state(cache_file).unwrap_or_default();
    state.checked_at = checked_at;
    state.checked_current_version = current_version.to_string();
    state.install_method = method.to_string();
    state.latest_version = latest_version;
    write_state(cache_file, &state)
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
        "Membrick {latest} is available (installed: {current}, source: {method}). Tell the user and offer the `update_membrick` MCP action for version {latest}; wait for their approval before calling it. If this client cannot call MCP tools, the manual update command is: {command}. Release notes: https://github.com/RooAGI/Membrick/releases/tag/v{latest}",
    )
}

/// Install the cached release after the host client has obtained user approval
/// for the `update_membrick` MCP tool. Commands use fixed executable names and
/// argument lists; no shell command is constructed from cached or user data.
pub(crate) fn install_cached_update(expected_version: &str) -> anyhow::Result<String> {
    let executable = std::env::current_exe()?;
    let cache_file = state_path(&executable)
        .ok_or_else(|| anyhow::anyhow!("update cache location is unavailable"))?;
    let _lock = try_lock(&cache_file)
        .ok_or_else(|| anyhow::anyhow!("update state is busy; retry in a moment"))?;
    let current = env!("CARGO_PKG_VERSION");
    let state = read_state(&cache_file).ok_or_else(|| {
        anyhow::anyhow!("no update check is cached yet; retry after the next session start")
    })?;
    anyhow::ensure!(
        state.checked_current_version == current,
        "cached update check belongs to a different installed version; restart the MCP client and check again"
    );
    let latest = state
        .latest_version
        .as_deref()
        .filter(|latest| *latest != current)
        .ok_or_else(|| anyhow::anyhow!("no newer Membrick release is currently available"))?;
    anyhow::ensure!(
        latest == expected_version,
        "cached update changed; refresh the notice before installing"
    );

    match state.install_method.as_str() {
        "npm" => {
            let global_root = Command::new("npm")
                .args(["root", "--global"])
                .stdin(Stdio::null())
                .stderr(Stdio::null())
                .output()
                .map_err(|error| anyhow::anyhow!("could not inspect npm global install: {error}"))?;
            anyhow::ensure!(global_root.status.success(), "could not inspect npm global install");
            let global_root = String::from_utf8_lossy(&global_root.stdout).trim().to_string();
            let installed_path = executable.canonicalize().unwrap_or(executable.clone());
            let global_root = PathBuf::from(global_root);
            let global_root = global_root.canonicalize().unwrap_or(global_root);
            anyhow::ensure!(
                installed_path.starts_with(&global_root),
                "this appears to be a local npm install. To avoid changing the project or updating a different installation, run `npm update membrick` in the project that installed it."
            );
            run_update_command("npm", &["update", "--global", "membrick"])?;
        }
        "pipx" => run_update_command("pipx", &["upgrade", "membrick"])?,
        "pip" => run_update_command(
            PYTHON_COMMAND,
            &["-m", "pip", "install", "--upgrade", "membrick"],
        )?,
        "cargo" => run_update_command("cargo", &["install", "membrick", "--locked", "--force"] )?,
        "standalone" => update_managed_binary(&executable)?,
        other => anyhow::bail!(
            "automatic update is not available for install source '{other}'. See https://github.com/RooAGI/Membrick/releases/latest"
        ),
    }

    Ok(format!(
        "Updated Membrick from {current} to {latest}. Restart the agent client to reconnect using the new version."
    ))
}

fn run_update_command(program: &str, args: &[&str]) -> anyhow::Result<()> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| anyhow::anyhow!("could not start {program}: {error}"))?;
    let deadline = std::time::Instant::now() + Duration::from_secs(180);
    loop {
        if let Some(status) = child.try_wait()? {
            anyhow::ensure!(
                status.success(),
                "{program} update command failed with {status}"
            );
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("{program} update command timed out after 180 seconds");
        }
        std::thread::sleep(Duration::from_millis(200));
    }
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
    check_is_due_at(cache_file, current_version, now_seconds())
}

fn check_is_due_at(cache_file: &Path, current_version: &str, now: u64) -> bool {
    let Some(state) = read_state(cache_file) else {
        return true;
    };
    state.checked_current_version != current_version
        || now.saturating_sub(state.checked_at) >= CHECK_INTERVAL.as_secs()
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

fn update_managed_binary(exe: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        exe.parent()
            .is_some_and(|parent| parent.join(MANAGED_MARKER).is_file()),
        "the standalone installer marker is missing; refusing to replace this executable"
    );
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
        .show_download_progress(false)
        .show_output(false)
        .no_confirm(true)
        .timeout(CHECK_TIMEOUT);
    let status = update.build()?.update()?;
    anyhow::ensure!(
        status.version() != env!("CARGO_PKG_VERSION"),
        "the standalone updater found no newer version to install"
    );
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("membrick-updater-test-{}-{id}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn cache_file(temp: &TempDir) -> PathBuf {
        temp.path().join("cache/update.json")
    }

    #[test]
    fn successful_release_check_persists_result_and_respects_daily_interval() {
        let temp = TempDir::new();
        let cache = cache_file(&temp);
        let mut requested_version = None;

        run_release_check(&cache, "0.4.0", "cargo", 1_000, |current| {
            requested_version = Some(current.to_string());
            Ok(Some("0.4.1".to_string()))
        })
        .unwrap();

        let state = read_state(&cache).unwrap();
        assert_eq!(requested_version.as_deref(), Some("0.4.0"));
        assert_eq!(state.checked_at, 1_000);
        assert_eq!(state.checked_current_version, "0.4.0");
        assert_eq!(state.latest_version.as_deref(), Some("0.4.1"));
        assert_eq!(state.install_method, "cargo");
        assert!(!check_is_due_at(
            &cache,
            "0.4.0",
            1_000 + CHECK_INTERVAL.as_secs() - 1
        ));
        assert!(check_is_due_at(
            &cache,
            "0.4.0",
            1_000 + CHECK_INTERVAL.as_secs()
        ));
        assert!(check_is_due_at(&cache, "0.4.1", 1_001));
    }

    #[test]
    fn failed_release_check_does_not_write_or_extend_the_cache() {
        let temp = TempDir::new();
        let cache = cache_file(&temp);

        run_release_check(&cache, "0.4.0", "cargo", 1_000, |_| {
            Ok(Some("0.4.1".to_string()))
        })
        .unwrap();
        let before = read_state(&cache).unwrap();

        let result = run_release_check(&cache, "0.4.0", "npm", 2_000, |_| {
            anyhow::bail!("simulated network failure")
        });

        assert!(result.is_err());
        assert_eq!(read_state(&cache).unwrap().checked_at, before.checked_at);
        assert_eq!(read_state(&cache).unwrap().install_method, "cargo");
        assert!(!check_is_due_at(&cache, "0.4.0", 1_001));
    }

    #[test]
    fn successful_check_without_newer_release_clears_previous_available_version() {
        let temp = TempDir::new();
        let cache = cache_file(&temp);
        run_release_check(&cache, "0.4.0", "cargo", 1_000, |_| {
            Ok(Some("0.4.1".to_string()))
        })
        .unwrap();

        run_release_check(&cache, "0.4.0", "cargo", 2_000, |_| Ok(None)).unwrap();

        let state = read_state(&cache).unwrap();
        assert_eq!(state.checked_at, 2_000);
        assert_eq!(state.latest_version, None);
        assert!(
            take_update_notice_from(&cache, "0.4.0", &temp.path().join("bin/membrick")).is_none()
        );
    }

    #[test]
    fn hook_notice_is_emitted_once_per_release_and_can_notify_for_a_later_release() {
        let temp = TempDir::new();
        let cache = cache_file(&temp);
        let executable = temp.path().join("bin/membrick");

        run_release_check(&cache, "0.4.0", "pipx", 1_000, |_| {
            Ok(Some("0.4.1".to_string()))
        })
        .unwrap();
        let notice = take_update_notice_from(&cache, "0.4.0", &executable).unwrap();
        assert!(notice.contains("Membrick 0.4.1 is available"));
        assert!(notice.contains("pipx upgrade membrick"));
        assert!(notice.contains("/releases/tag/v0.4.1"));
        assert!(take_update_notice_from(&cache, "0.4.0", &executable).is_none());

        run_release_check(&cache, "0.4.0", "pipx", 2_000, |_| {
            Ok(Some("0.4.2".to_string()))
        })
        .unwrap();
        let next_notice = take_update_notice_from(&cache, "0.4.0", &executable).unwrap();
        assert!(next_notice.contains("Membrick 0.4.2 is available"));
    }

    #[test]
    fn no_notice_is_emitted_for_current_or_mismatched_install_version() {
        let temp = TempDir::new();
        let cache = cache_file(&temp);
        let executable = temp.path().join("bin/membrick");
        run_release_check(&cache, "0.4.0", "npm", 1_000, |_| {
            Ok(Some("0.4.0".to_string()))
        })
        .unwrap();
        assert!(take_update_notice_from(&cache, "0.4.0", &executable).is_none());

        run_release_check(&cache, "0.4.0", "npm", 2_000, |_| {
            Ok(Some("0.4.1".to_string()))
        })
        .unwrap();
        assert!(take_update_notice_from(&cache, "0.4.1", &executable).is_none());
    }

    #[test]
    fn install_method_detection_covers_package_markers_and_known_paths() {
        let temp = TempDir::new();
        let npm_dir = temp.path().join("npm/bin");
        fs::create_dir_all(&npm_dir).unwrap();
        fs::write(npm_dir.join(INSTALL_METHOD_MARKER), "npm\n").unwrap();
        assert_eq!(
            install_method(&npm_dir.join("membrick")),
            Some("npm".to_string())
        );

        let standalone_dir = temp.path().join("standalone/bin");
        fs::create_dir_all(&standalone_dir).unwrap();
        fs::write(standalone_dir.join(MANAGED_MARKER), "").unwrap();
        assert_eq!(
            install_method(&standalone_dir.join("membrick")),
            Some("standalone".to_string())
        );

        assert_eq!(
            install_method(Path::new("/Users/test/.cargo/bin/membrick")),
            Some("cargo".to_string())
        );
        assert_eq!(
            install_method(Path::new("/tmp/pipx/venvs/membrick/bin/membrick")),
            Some("pipx".to_string())
        );
        assert_eq!(install_method(Path::new("/tmp/custom/membrick")), None);
    }

    #[test]
    fn cache_lock_excludes_concurrent_readers_and_releases_on_drop() {
        let temp = TempDir::new();
        let cache = cache_file(&temp);
        let lock = try_lock(&cache).unwrap();
        assert!(try_lock(&cache).is_none());
        drop(lock);
        assert!(try_lock(&cache).is_some());
    }
}
