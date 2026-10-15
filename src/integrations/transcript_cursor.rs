//! Per-session transcript position for capture hooks.
//!
//! Claude Code and Codex append to their JSONL transcripts. Each capture hook
//! stores only the lines written since the previous capture of the same
//! session, so capture cost follows the size of the turn rather than the
//! size of the whole session.

use anyhow::{Context, Result};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Transcript text appended since the previous capture.
pub(crate) struct TranscriptSlice {
    /// Complete JSONL lines starting at `start`.
    pub text: String,
    /// Byte offset where this slice starts.
    pub start: u64,
    /// Byte offset just past the last complete line in this slice.
    pub end: u64,
}

/// Read the complete lines written after the stored position. A trailing line
/// without a newline is still being written and is left for the next capture.
pub(crate) fn read_since_last_capture(
    state_dir: &Path,
    session_id: &str,
    transcript: &Path,
) -> Result<TranscriptSlice> {
    let stored = load_offset(state_dir, session_id);
    let mut file = fs::File::open(transcript)
        .with_context(|| format!("failed to open transcript {}", transcript.display()))?;
    let length = file.metadata()?.len();
    // A transcript shorter than the stored position was rewritten; read it
    // again from the beginning.
    let start = if stored > length { 0 } else { stored };
    file.seek(SeekFrom::Start(start))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .with_context(|| format!("failed to read transcript {}", transcript.display()))?;
    let complete = bytes
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |index| index + 1);
    bytes.truncate(complete);
    Ok(TranscriptSlice {
        text: String::from_utf8_lossy(&bytes).into_owned(),
        start,
        end: start + complete as u64,
    })
}

/// Record that everything before `end` has been captured. Call only after
/// the slice was stored, so a failed capture is retried on the next hook.
pub(crate) fn mark_captured(state_dir: &Path, session_id: &str, end: u64) -> Result<()> {
    let path = cursor_path(state_dir, session_id);
    let parent = path.parent().expect("cursor path has a parent");
    fs::create_dir_all(parent)?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, serde_json::json!({ "offset": end }).to_string())?;
    fs::rename(&temporary, &path)?;
    Ok(())
}

fn load_offset(state_dir: &Path, session_id: &str) -> u64 {
    fs::read_to_string(cursor_path(state_dir, session_id))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|value| value.get("offset")?.as_u64())
        .unwrap_or(0)
}

fn cursor_path(state_dir: &Path, session_id: &str) -> PathBuf {
    // Session IDs come from the host; hash them so they never form a path.
    let name = crate::ids::stable_doc_id_from_source(session_id)
        .trim_start_matches("doc:")
        .to_string();
    state_dir
        .join("capture-cursors")
        .join(format!("{name}.json"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_dir(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "membrick-cursor-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn reads_only_lines_appended_since_the_last_capture() {
        let root = temp_dir("append");
        let transcript = root.join("t.jsonl");
        fs::write(&transcript, "one\ntwo\n").unwrap();

        let first = read_since_last_capture(&root, "s", &transcript).unwrap();
        assert_eq!(first.text, "one\ntwo\n");
        assert_eq!(first.start, 0);
        mark_captured(&root, "s", first.end).unwrap();

        fs::OpenOptions::new()
            .append(true)
            .open(&transcript)
            .unwrap()
            .write_all(b"three\n")
            .unwrap();
        let second = read_since_last_capture(&root, "s", &transcript).unwrap();
        assert_eq!(second.text, "three\n");
        assert_eq!(second.start, first.end);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn leaves_a_partial_trailing_line_for_the_next_capture() {
        let root = temp_dir("partial");
        let transcript = root.join("t.jsonl");
        fs::write(&transcript, "one\ntw").unwrap();
        let slice = read_since_last_capture(&root, "s", &transcript).unwrap();
        assert_eq!(slice.text, "one\n");
        assert_eq!(slice.end, 4);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn restarts_from_the_beginning_when_the_transcript_shrinks() {
        let root = temp_dir("shrink");
        let transcript = root.join("t.jsonl");
        fs::write(&transcript, "one\ntwo\nthree\n").unwrap();
        mark_captured(&root, "s", 14).unwrap();
        fs::write(&transcript, "new\n").unwrap();
        let slice = read_since_last_capture(&root, "s", &transcript).unwrap();
        assert_eq!(slice.start, 0);
        assert_eq!(slice.text, "new\n");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sessions_keep_separate_positions() {
        let root = temp_dir("sessions");
        let transcript = root.join("t.jsonl");
        fs::write(&transcript, "one\n").unwrap();
        mark_captured(&root, "a", 4).unwrap();
        let other = read_since_last_capture(&root, "b", &transcript).unwrap();
        assert_eq!(other.text, "one\n");
        fs::remove_dir_all(root).unwrap();
    }
}
