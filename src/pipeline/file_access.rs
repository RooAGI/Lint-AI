//! Descriptor-based reads for untrusted workspace paths.
use std::fs::File;
use std::io;
use std::path::{Component, Path};

/// Open a regular file without following a symlink in any path component.
/// Each directory descriptor pins the next lookup, so a directory rename or
/// symlink replacement cannot redirect a later lookup to another directory.
#[cfg(unix)]
pub(crate) fn open_regular_file(path: &Path) -> io::Result<File> {
    use std::ffi::CString;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;

    let mut directory = File::open(if path.is_absolute() { "/" } else { "." })?;
    let components = path
        .components()
        .filter(|c| !matches!(c, Component::RootDir | Component::CurDir))
        .collect::<Vec<_>>();
    if components.is_empty() {
        return Err(io::ErrorKind::InvalidInput.into());
    }
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(name) = component else {
            return Err(io::ErrorKind::InvalidInput.into());
        };
        let name = CString::new(name.as_bytes()).map_err(|_| io::ErrorKind::InvalidInput)?;
        let last = index + 1 == components.len();
        let flags = libc::O_RDONLY
            | libc::O_CLOEXEC
            | libc::O_NOFOLLOW
            | if last {
                libc::O_NONBLOCK
            } else {
                libc::O_DIRECTORY
            };
        // SAFETY: the directory descriptor is owned and live; name is a
        // nul-terminated CString. No create flag is used, so no mode is needed.
        let descriptor = unsafe { libc::openat(directory.as_raw_fd(), name.as_ptr(), flags) };
        if descriptor < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: openat returned a new descriptor, which this File owns once.
        let file = unsafe { File::from_raw_fd(descriptor) };
        if last {
            return if file.metadata()?.is_file() {
                Ok(file)
            } else {
                Err(io::ErrorKind::InvalidInput.into())
            };
        }
        directory = file;
    }
    Err(io::ErrorKind::InvalidInput.into())
}

/// Until a descriptor-relative implementation is available on other targets,
/// skip automatic content capture rather than follow a mutable path.
#[cfg(not(unix))]
pub(crate) fn open_regular_file(_path: &Path) -> io::Result<File> {
    Err(io::ErrorKind::Unsupported.into())
}
