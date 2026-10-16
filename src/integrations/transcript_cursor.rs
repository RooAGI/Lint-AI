//! Per-session transcript position for capture hooks.
//!
//! Claude Code and Codex append to their JSONL transcripts. Each capture hook
//! stores only the lines written since the previous capture of the same
//! session, so capture cost follows the size of the turn rather than the
//! size of the whole session.

use anyhow::{Context, Result};
use std::fs;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAX_TRANSCRIPT_BATCH_BYTES: usize = 1024 * 1024;

/// Keep the provider turn available for lookup and include it in the existing
/// document identity. The original event identity distinguishes messages and
/// capture fragments within the same turn, so partial captures cannot overwrite
/// each other's text.
pub(crate) fn link_provider_turn(document: &mut crate::SourceDocument, turn_id: Option<&str>) {
    let Some(turn_id) = turn_id.filter(|value| !value.trim().is_empty()) else {
        return;
    };
    // Hex preserves case and separators despite source ID normalization.
    let encoded = turn_id
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    document.doc_id = crate::ids::stable_doc_id_from_source(&format!(
        "{}::provider-turn:{encoded}",
        document.doc_id
    ));
    document.filters.insert("turn_id".into(), turn_id.into());
}

pub(crate) fn transcript_turn_id(value: &serde_json::Value) -> Option<&str> {
    value
        .get("turn_id")
        .or_else(|| value.get("turnId"))
        .or_else(|| {
            value
                .get("message")
                .and_then(|message| message.get("turn_id"))
        })
        .or_else(|| {
            value
                .get("payload")
                .and_then(|payload| payload.get("turn_id"))
        })
        .and_then(serde_json::Value::as_str)
        .filter(|id| !id.trim().is_empty())
}

/// Transcript text appended since the previous capture.
pub(crate) struct TranscriptSlice {
    /// Complete JSONL lines starting at `start`.
    pub text: String,
    /// Byte offset where this slice starts.
    pub start: u64,
    /// Byte offset just past the last complete line in this slice.
    pub end: u64,
    /// Invalid cursor metadata was reset before this replay.
    pub restarted: bool,
    checkpoint: serde_json::Value,
}

/// Read one bounded batch of complete lines after the stored position. A
/// trailing line without a newline is left for the next capture. The first
/// cursor is written at offset zero before transcript data is read.
pub(crate) fn read_since_last_capture(
    state_dir: &Path,
    session_id: &str,
    transcript: &Path,
) -> Result<TranscriptSlice> {
    read_next_batch(
        state_dir,
        session_id,
        transcript,
        MAX_TRANSCRIPT_BATCH_BYTES,
    )
}

fn read_next_batch(
    state_dir: &Path,
    session_id: &str,
    transcript: &Path,
    max_batch_bytes: usize,
) -> Result<TranscriptSlice> {
    read_next_batch_until(state_dir, session_id, transcript, max_batch_bytes, None)
}

pub(crate) fn read_capture_batch(
    state_dir: &Path,
    session_id: &str,
    transcript: &Path,
    target_bytes: u64,
) -> Result<TranscriptSlice> {
    read_next_batch_until(
        state_dir,
        session_id,
        transcript,
        MAX_TRANSCRIPT_BATCH_BYTES,
        Some(target_bytes),
    )
}

fn read_next_batch_until(
    state_dir: &Path,
    session_id: &str,
    transcript: &Path,
    max_batch_bytes: usize,
    target_bytes: Option<u64>,
) -> Result<TranscriptSlice> {
    initialize_cursor(state_dir, session_id)?;
    let mut file = fs::File::open(transcript)
        .with_context(|| format!("failed to open transcript {}", transcript.display()))?;
    let length = file.metadata()?.len();
    let path = cursor_path(state_dir, session_id);
    // Read errors are not corruption. Preserve permission/I/O failures for the
    // worker's retry path rather than silently replaying from a false zero.
    let raw = fs::read(&path)
        .with_context(|| format!("failed to read transcript cursor {}", path.display()))?;
    let saved = serde_json::from_slice::<serde_json::Value>(&raw).ok();
    let stored = saved
        .as_ref()
        .and_then(|v| v.get("offset"))
        .and_then(|v| v.as_u64());
    let mut reason = match stored {
        None => Some("malformed cursor or invalid offset"),
        Some(offset) if offset > length => Some("cursor beyond transcript EOF"),
        Some(offset) if offset > 0 => {
            file.seek(SeekFrom::Start(offset - 1))?;
            let mut boundary = [0u8; 1];
            file.read_exact(&mut boundary)?;
            (boundary[0] != b'\n').then_some("cursor inside a transcript line")
        }
        _ => None,
    };
    if reason.is_none() {
        if let Some(anchor) = saved.as_ref().and_then(|v| v.get("transcript_anchor")) {
            if *anchor != transcript_anchor(&mut file, stored.unwrap())? {
                reason = Some("transcript identity or checkpoint bytes changed");
            }
        }
    }
    let restarted = reason.is_some();
    let start = if let Some(reason) = reason {
        // Preserve the exact original bytes and reason before replacing the
        // unusable checkpoint. No transcript or memory document is deleted.
        // Content-address the archive so repeated failures retain evidence.
        use sha2::{Digest, Sha256};
        let archive = path.with_extension("json.recovery");
        let original_hex: String = raw.iter().map(|b| format!("{b:02x}")).collect();
        let recovery = serde_json::json!({"reason":reason,"original_hex":original_hex});
        let versioned = path.with_extension(format!("json.recovery-{:x}", Sha256::digest(&raw)));
        crate::pipeline::persistence::write_private_text_file_atomic(
            &versioned,
            &recovery.to_string(),
        )?;
        crate::pipeline::persistence::write_private_text_file_atomic(
            &archive,
            &recovery.to_string(),
        )?;
        mark_captured(state_dir, session_id, 0)?;
        eprintln!(
            "Recovered transcript cursor {}: {reason}; replaying from zero",
            path.display()
        );
        0
    } else {
        stored.expect("validated cursor offset")
    };
    file.seek(SeekFrom::Start(start))?;
    let end = target_bytes.unwrap_or(length).min(length);
    let mut reader = BufReader::new(file.take(end.saturating_sub(start)));
    let mut bytes = Vec::with_capacity(max_batch_bytes.min(64 * 1024));
    loop {
        let mut line = Vec::new();
        let read = reader
            .read_until(b'\n', &mut line)
            .with_context(|| format!("failed to read transcript {}", transcript.display()))?;
        if read == 0 || line.last() != Some(&b'\n') {
            break;
        }
        if !bytes.is_empty() && bytes.len().saturating_add(line.len()) > max_batch_bytes {
            break;
        }
        bytes.extend_from_slice(&line);
        if bytes.len() >= max_batch_bytes {
            break;
        }
    }
    let complete = bytes.len();
    let mut file = reader.into_inner().into_inner();
    let checkpoint = serde_json::json!({
        "offset": start + complete as u64,
        "transcript_anchor": transcript_anchor(&mut file, start + complete as u64)?,
    });
    Ok(TranscriptSlice {
        text: String::from_utf8_lossy(&bytes).into_owned(),
        start,
        end: start + complete as u64,
        restarted,
        checkpoint,
    })
}

/// Process transcript batches in order. The cursor advances only after the
/// callback confirms that a batch is durable. A failed batch is retried on the
/// next call, while prior batches remain checkpointed.
pub(crate) fn replay_batches(
    state_dir: &Path,
    session_id: &str,
    transcript: &Path,
    persist: impl FnMut(&TranscriptSlice) -> Result<()>,
) -> Result<usize> {
    replay_batches_with_limit(
        state_dir,
        session_id,
        transcript,
        MAX_TRANSCRIPT_BATCH_BYTES,
        persist,
    )
}

pub(crate) fn replay_batches_with_limit(
    state_dir: &Path,
    session_id: &str,
    transcript: &Path,
    max_batch_bytes: usize,
    mut persist: impl FnMut(&TranscriptSlice) -> Result<()>,
) -> Result<usize> {
    initialize_cursor(state_dir, session_id)?;
    let mut batches = 0;
    loop {
        let slice = read_next_batch(state_dir, session_id, transcript, max_batch_bytes)?;
        if slice.end == slice.start {
            break;
        }
        persist(&slice)?;
        commit_capture(state_dir, session_id, transcript, &slice)?;
        batches += 1;
    }
    Ok(batches)
}

pub(crate) fn initialize_cursor(state_dir: &Path, session_id: &str) -> Result<()> {
    let path = cursor_path(state_dir, session_id);
    let parent = path.parent().expect("cursor path has a parent");
    fs::create_dir_all(parent)?;
    match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(mut file) => file
            .write_all(b"{\"offset\":0}")
            .with_context(|| format!("failed to initialize transcript cursor {}", path.display())),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(error)
            .with_context(|| format!("failed to initialize transcript cursor {}", path.display())),
    }
}

/// Record that everything before `end` has been captured. Call only after
/// the slice was stored, so a failed capture is retried on the next hook.
pub(crate) fn mark_captured(state_dir: &Path, session_id: &str, end: u64) -> Result<()> {
    let path = cursor_path(state_dir, session_id);
    let parent = path.parent().expect("cursor path has a parent");
    fs::create_dir_all(parent)?;
    crate::pipeline::persistence::write_private_text_file_atomic(
        &path,
        &serde_json::json!({ "offset": end }).to_string(),
    )
}

// Fingerprints are bounded to the first and preceding 64 bytes of the
// captured prefix. Appends do not change them. Unix identity also detects
// replacement with identical text. This is not a whole-file integrity hash.
fn transcript_anchor(file: &mut fs::File, offset: u64) -> Result<serde_json::Value> {
    use sha2::{Digest, Sha256};
    let length = file.metadata()?.len();
    anyhow::ensure!(offset <= length, "transcript shortened during capture");
    let mut prefix = vec![0; offset.min(64) as usize];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut prefix)?;
    let mut boundary = vec![0; offset.min(64) as usize];
    file.seek(SeekFrom::Start(offset.saturating_sub(64)))?;
    file.read_exact(&mut boundary)?;
    let mut anchor = serde_json::json!({
        "prefix_sha256": format!("{:x}", Sha256::digest(prefix)),
        "boundary_sha256": format!("{:x}", Sha256::digest(boundary)),
    });
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let stat = file.metadata()?;
        anchor["device"] = stat.dev().into();
        anchor["inode"] = stat.ino().into();
    }
    Ok(anchor)
}

/// Publish the read snapshot only after MemoryService acknowledges persistence.
/// If the transcript was replaced during the write, leave the old checkpoint
/// intact. The worker can safely retry and recover on its next pass.
pub(crate) fn commit_capture(
    state_dir: &Path,
    session_id: &str,
    transcript: &Path,
    slice: &TranscriptSlice,
) -> Result<()> {
    let mut file = fs::File::open(transcript)?;
    anyhow::ensure!(
        slice.checkpoint["transcript_anchor"] == transcript_anchor(&mut file, slice.end)?,
        "transcript changed during capture; cursor was not advanced"
    );
    crate::pipeline::persistence::write_private_text_file_atomic(
        &cursor_path(state_dir, session_id),
        &slice.checkpoint.to_string(),
    )
}

pub(crate) fn load_offset(state_dir: &Path, session_id: &str) -> u64 {
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
        path.canonicalize().unwrap()
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
    fn first_read_initializes_cursor_before_capture_can_fail() {
        let root = temp_dir("initial-cursor");
        let transcript = root.join("t.jsonl");
        fs::write(&transcript, "first turn\n").unwrap();

        let slice = read_since_last_capture(&root, "s", &transcript).unwrap();
        let path = cursor_path(&root, "s");

        assert!(
            path.exists(),
            "first capture must durably establish offset zero"
        );
        assert_eq!(load_offset(&root, "s"), 0);
        assert_eq!(slice.start, 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn transcript_reads_are_bounded_to_complete_line_batches() {
        let root = temp_dir("bounded-batches");
        let transcript = root.join("t.jsonl");
        let line = format!("{}\n", "x".repeat(MAX_TRANSCRIPT_BATCH_BYTES / 3));
        fs::write(&transcript, format!("{line}{line}{line}{line}")).unwrap();

        let first = read_since_last_capture(&root, "s", &transcript).unwrap();
        assert!(
            first.text.len() <= MAX_TRANSCRIPT_BATCH_BYTES,
            "one read must stay within the configured batch size"
        );
        assert!(!first.text.is_empty());
        assert!(first.text.ends_with('\n'));
        assert_eq!(first.end - first.start, first.text.len() as u64);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_batch_replay_keeps_only_durable_batches_checkpointed() {
        let root = temp_dir("batch-retry");
        let transcript = root.join("t.jsonl");
        fs::write(&transcript, "one\ntwo\nthree\n").unwrap();
        let mut durable = Vec::new();

        let result = replay_batches_with_limit(&root, "s", &transcript, 4, |slice| {
            if slice.start == 4 {
                anyhow::bail!("simulated MemoryService write failure");
            }
            durable.push(slice.text.clone());
            Ok(())
        });
        assert!(result.is_err());
        assert_eq!(load_offset(&root, "s"), 4);
        assert_eq!(durable, ["one\n"]);

        let mut retried = Vec::new();
        let count = replay_batches_with_limit(&root, "s", &transcript, 4, |slice| {
            retried.push(slice.text.clone());
            Ok(())
        })
        .unwrap();
        assert_eq!(count, 2);
        assert_eq!(retried, ["two\n", "three\n"]);
        assert_eq!(load_offset(&root, "s"), 14);
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
    fn recovery_replays_whole_lines_instead_of_midline_fragments() {
        let root = temp_dir("recovery-midline");
        let transcript = root.join("t.jsonl");
        fs::write(&transcript, "one\ntwo\n").unwrap();
        mark_captured(&root, "s", 2).unwrap();
        let slice = read_since_last_capture(&root, "s", &transcript).unwrap();
        assert_eq!(
            slice.text, "one\ntwo\n",
            "must never skip a damaged line prefix"
        );
        assert_eq!(load_offset(&root, "s"), 0);
        assert!(slice.restarted);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn recovery_repairs_corrupt_cursor_before_a_failed_write() {
        let root = temp_dir("recovery-corrupt");
        let transcript = root.join("t.jsonl");
        fs::write(&transcript, "one\n").unwrap();
        initialize_cursor(&root, "s").unwrap();
        fs::write(cursor_path(&root, "s"), "{broken").unwrap();
        let result = replay_batches_with_limit(&root, "s", &transcript, 4, |_| {
            anyhow::bail!("write rejected")
        });
        assert!(result.is_err());
        let value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(cursor_path(&root, "s")).unwrap()).unwrap();
        assert_eq!(value["offset"], 0);
        assert!(cursor_path(&root, "s")
            .with_extension("json.recovery")
            .exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn recovery_resets_out_of_range_cursor_even_for_empty_transcript() {
        let root = temp_dir("recovery-empty");
        let transcript = root.join("t.jsonl");
        fs::write(&transcript, "").unwrap();
        mark_captured(&root, "s", 100).unwrap();
        let slice = read_since_last_capture(&root, "s", &transcript).unwrap();
        assert_eq!(slice.start, 0);
        assert_eq!(
            load_offset(&root, "s"),
            0,
            "empty replay must repair saved position"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn recovery_replays_same_length_rewritten_transcript() {
        let root = temp_dir("recovery-rewrite");
        let transcript = root.join("t.jsonl");
        fs::write(&transcript, "old\n").unwrap();
        replay_batches(&root, "s", &transcript, |_| Ok(())).unwrap();
        fs::write(&transcript, "new\n").unwrap();
        let mut stored = String::new();
        replay_batches(&root, "s", &transcript, |slice| {
            stored.push_str(&slice.text);
            Ok(())
        })
        .unwrap();
        assert_eq!(
            stored, "new\n",
            "same-size rewrite must not be treated as captured"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn transcript_rewrite_during_persist_does_not_advance_cursor() {
        let root = temp_dir("recovery-in-flight-rewrite");
        let transcript = root.join("t.jsonl");
        fs::write(&transcript, "old\n").unwrap();
        let result = replay_batches(&root, "s", &transcript, |_| {
            fs::write(&transcript, "new\n")?;
            Ok(())
        });
        assert!(result.is_err());
        assert_eq!(load_offset(&root, "s"), 0);
        let mut retried = String::new();
        replay_batches(&root, "s", &transcript, |slice| {
            retried.push_str(&slice.text);
            Ok(())
        })
        .unwrap();
        assert_eq!(retried, "new\n");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unavailable_transcript_does_not_reset_or_archive_cursor() {
        let root = temp_dir("recovery-unavailable");
        mark_captured(&root, "s", 123).unwrap();
        assert!(read_since_last_capture(&root, "s", &root.join("missing")).is_err());
        assert_eq!(load_offset(&root, "s"), 123);
        assert!(!cursor_path(&root, "s")
            .with_extension("json.recovery")
            .exists());
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
