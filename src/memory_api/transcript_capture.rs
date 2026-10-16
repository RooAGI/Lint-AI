//! Durable transcript capture owned by the resident, independent of hook lifetimes.
use super::MemoryService;
use crate::integrations::{
    session_recording::{provider_state_dir, RecordingProvider},
    transcript_cursor,
};
use crate::SourceDocument;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Condvar, Mutex,
    },
    thread::JoinHandle,
    time::Duration,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct CaptureRequest {
    pub provider: String,
    pub document_type: String,
    pub input: Value,
}

impl CaptureRequest {
    fn provider(&self) -> Result<RecordingProvider> {
        match self.provider.as_str() {
            #[cfg(feature = "claude-code")]
            "claude" => Ok(RecordingProvider::Claude),
            #[cfg(feature = "codex")]
            "codex" => Ok(RecordingProvider::Codex),
            _ => anyhow::bail!(
                "transcript capture is unavailable for provider {}",
                self.provider
            ),
        }
    }
    fn session_id(&self) -> Result<&str> {
        self.input
            .get("session_id")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .context("session_id is required")
    }
    fn transcript(&self) -> Result<PathBuf> {
        self.input
            .get("transcript_path")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .context("transcript_path is required")
    }
    pub(crate) fn validate(&self, root: &Path) -> Result<()> {
        self.provider()?;
        self.session_id()?;
        anyhow::ensure!(
            matches!(
                self.document_type.as_str(),
                "checkpoint" | "outcome" | "session-summary"
            ),
            "invalid capture document_type"
        );
        anyhow::ensure!(
            self.transcript()?.is_absolute(),
            "transcript_path must be absolute"
        );
        anyhow::ensure!(
            std::fs::metadata(self.transcript()?)?.is_file(),
            "transcript_path must be a regular file"
        );
        let cwd = self
            .input
            .get("cwd")
            .and_then(Value::as_str)
            .context("cwd is required")?;
        anyhow::ensure!(
            std::fs::canonicalize(cwd)? == root,
            "capture cwd does not match workspace"
        );
        match self.provider.as_str() {
            #[cfg(feature = "claude-code")]
            "claude" => {
                let _: crate::integrations::claude_code::hooks::protocol::ClaudeHookInput =
                    serde_json::from_value(self.input.clone())?;
            }
            #[cfg(feature = "codex")]
            "codex" => {
                let _: crate::integrations::codex::hooks::protocol::CodexHookInput =
                    serde_json::from_value(self.input.clone())?;
            }
            _ => unreachable!(),
        }
        Ok(())
    }
    fn state_dir(&self, root: &Path) -> Result<PathBuf> {
        Ok(provider_state_dir(self.provider()?, root))
    }
    fn key(&self) -> Result<String> {
        Ok(format!(
            "{}-{}",
            self.provider,
            crate::ids::stable_doc_id_from_source(self.session_id()?).trim_start_matches("doc:")
        ))
    }
    fn documents(
        &self,
        root: &Path,
        slice: &transcript_cursor::TranscriptSlice,
    ) -> Result<Vec<SourceDocument>> {
        match self.provider.as_str() {
            #[cfg(feature = "claude-code")]
            "claude" => crate::integrations::claude_code::hooks::capture_documents_from_request(
                root,
                &self.input,
                &self.document_type,
                slice,
            ),
            #[cfg(feature = "codex")]
            "codex" => crate::integrations::codex::hooks::capture_documents_from_request(
                root,
                &self.input,
                &self.document_type,
                slice,
            ),
            _ => anyhow::bail!("unsupported capture provider"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PendingCapture {
    request: CaptureRequest,
    target_bytes: u64,
    #[serde(default)]
    last_progress_ms: Option<u64>,
    #[serde(default)]
    last_error: Option<String>,
    #[serde(default)]
    failures: u32,
    #[serde(default)]
    retry_after_ms: u64,
    #[serde(skip)]
    processing: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct CaptureStatus {
    pub provider: String,
    pub session_id: String,
    pub captured_bytes: u64,
    pub target_bytes: u64,
    pub remaining_bytes: u64,
    pub status: &'static str,
    pub last_progress_ms: Option<u64>,
    pub last_error: Option<String>,
}

impl PendingCapture {
    fn offset(&self, root: &Path) -> Result<u64> {
        Ok(transcript_cursor::load_offset(
            &self.request.state_dir(root)?,
            self.request.session_id()?,
        ))
    }
    fn status(&self, root: &Path) -> Result<CaptureStatus> {
        let offset = self.offset(root)?;
        Ok(CaptureStatus {
            provider: self.request.provider.clone(),
            session_id: self.request.session_id()?.into(),
            captured_bytes: offset,
            target_bytes: self.target_bytes,
            remaining_bytes: if offset > self.target_bytes {
                self.target_bytes
            } else {
                self.target_bytes - offset
            },
            status: if self.processing {
                "processing"
            } else if self.last_error.is_some() {
                "retrying"
            } else if offset == self.target_bytes {
                "caught-up"
            } else {
                "pending"
            },
            last_progress_ms: self.last_progress_ms,
            last_error: self.last_error.clone(),
        })
    }
}

type Persist = dyn Fn(&CaptureRequest, Vec<SourceDocument>) -> Result<()> + Send + Sync;
struct Queue {
    root: PathBuf,
    records: Mutex<BTreeMap<String, PendingCapture>>,
    load_errors: Vec<CaptureStatus>,
    wake: Condvar,
    stopping: AtomicBool,
}

/// A MemoryService scheduling handle. The resident owns one per project and
/// retains the workspace owner lock until its capture thread has stopped.
pub(crate) struct TranscriptCaptures {
    queue: Arc<Queue>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl MemoryService {
    pub(crate) fn start_transcript_captures(
        root: &Path,
        persist: impl Fn(&CaptureRequest, Vec<SourceDocument>) -> Result<()> + Send + Sync + 'static,
    ) -> Result<TranscriptCaptures> {
        let root = root.canonicalize()?;
        let directory = root.join(".membrick/runtime/transcript-captures");
        std::fs::create_dir_all(&directory)?;
        let mut records = BTreeMap::new();
        let mut load_errors = Vec::new();
        for entry in std::fs::read_dir(&directory)? {
            let path = entry?.path();
            if path.extension().is_none_or(|ext| ext != "json") {
                continue;
            }
            let loaded = (|| -> Result<PendingCapture> {
                let record: PendingCapture = serde_json::from_slice(&std::fs::read(&path)?)?;
                record.request.provider()?;
                record.request.key()?;
                Ok(record)
            })();
            match loaded {
                Ok(mut record) => {
                    record.retry_after_ms = 0;
                    records.insert(record.request.key()?, record);
                }
                Err(error) => {
                    let error = format!("invalid pending capture {}: {error:#}", path.display());
                    eprintln!("{error}");
                    load_errors.push(CaptureStatus {
                        provider: "unknown".into(),
                        session_id: path
                            .file_stem()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned(),
                        captured_bytes: 0,
                        target_bytes: 0,
                        remaining_bytes: 0,
                        status: "invalid",
                        last_progress_ms: None,
                        last_error: Some(error),
                    });
                }
            }
        }
        let queue = Arc::new(Queue {
            root,
            records: Mutex::new(records),
            load_errors,
            wake: Condvar::new(),
            stopping: AtomicBool::new(false),
        });
        let worker_queue = queue.clone();
        let worker = std::thread::Builder::new()
            .name("membrick-transcript-capture".into())
            .spawn(move || worker_loop(worker_queue, &persist))?;
        Ok(TranscriptCaptures {
            queue,
            worker: Mutex::new(Some(worker)),
        })
    }
}

impl TranscriptCaptures {
    /// Acknowledges only after a pending job is atomically saved to disk.
    /// No index lock or transcript parsing is performed on this path.
    pub(crate) fn submit(&self, mut request: CaptureRequest) -> Result<CaptureStatus> {
        request.validate(&self.queue.root)?;
        request.input["transcript_path"] =
            serde_json::to_value(request.transcript()?.canonicalize()?)?;
        let key = request.key()?;
        let length = std::fs::metadata(request.transcript()?)?.len();
        let mut records = self
            .queue
            .records
            .lock()
            .map_err(|_| anyhow::anyhow!("capture queue poisoned"))?;
        anyhow::ensure!(
            !self.queue.stopping.load(Ordering::Acquire),
            "capture worker is shutting down"
        );
        let mut record = match records.get(&key) {
            Some(previous) => {
                anyhow::ensure!(
                    previous.request.transcript()? == request.transcript()?,
                    "session transcript path changed"
                );
                let mut record = previous.clone();
                // Freeze the parsing context while a batch may be in flight,
                // so duplicate hooks cannot change document IDs on retries.
                if previous.offset(&self.queue.root)? == previous.target_bytes {
                    record.request = request;
                }
                record.target_bytes = length;
                record
            }
            None => PendingCapture {
                request,
                target_bytes: length,
                last_progress_ms: None,
                last_error: None,
                failures: 0,
                retry_after_ms: 0,
                processing: false,
            },
        };
        record.retry_after_ms = 0;
        transcript_cursor::initialize_cursor(
            &record.request.state_dir(&self.queue.root)?,
            record.request.session_id()?,
        )?;
        save_record(&self.queue.root, &key, &record)?;
        let status = record.status(&self.queue.root)?;
        records.insert(key, record);
        self.queue.wake.notify_one();
        Ok(status)
    }
    pub(crate) fn status(&self) -> Result<Vec<CaptureStatus>> {
        let records = self
            .queue
            .records
            .lock()
            .map_err(|_| anyhow::anyhow!("capture queue poisoned"))?;
        let mut statuses = records
            .values()
            .map(|r| r.status(&self.queue.root))
            .collect::<Result<Vec<_>>>()?;
        statuses.extend(
            self.queue
                .load_errors
                .iter()
                .filter(|error| !records.contains_key(&error.session_id))
                .cloned(),
        );
        Ok(statuses)
    }
    /// Finish only the in-flight batch; pending jobs are resumed on startup.
    pub(crate) fn shutdown(&self) {
        self.queue.stopping.store(true, Ordering::Release);
        self.queue.wake.notify_all();
        if let Ok(mut worker) = self.worker.lock() {
            if let Some(worker) = worker.take() {
                let _ = worker.join();
            }
        }
    }
}
impl Drop for TranscriptCaptures {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn save_record(root: &Path, key: &str, record: &PendingCapture) -> Result<()> {
    crate::pipeline::persistence::write_private_text_file_atomic(
        &root
            .join(".membrick/runtime/transcript-captures")
            .join(format!("{key}.json")),
        &serde_json::to_string(record)?,
    )
}
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn process_batch(queue: &Queue, record: &PendingCapture, persist: &Persist) -> Result<bool> {
    let request = &record.request;
    let state_dir = request.state_dir(&queue.root)?;
    let slice = transcript_cursor::read_capture_batch(
        &state_dir,
        request.session_id()?,
        &request.transcript()?,
        record.target_bytes,
    )?;
    if slice.start == slice.end {
        return Ok(false);
    }
    let documents = request.documents(&queue.root, &slice)?;
    if !documents.is_empty() {
        persist(request, documents)?;
    }
    transcript_cursor::commit_capture(
        &state_dir,
        request.session_id()?,
        &request.transcript()?,
        &slice,
    )?;
    Ok(true)
}

fn worker_loop(queue: Arc<Queue>, persist: &Persist) {
    while !queue.stopping.load(Ordering::Acquire) {
        let jobs = {
            let records = queue.records.lock().expect("capture queue poisoned");
            records
                .iter()
                .filter(|(_, r)| {
                    // Recheck even a caught-up session: transcripts can grow
                    // without another hook and cursor files can disappear or
                    // become invalid after the last successful batch.
                    r.retry_after_ms <= now_ms()
                })
                .map(|(k, r)| (k.clone(), r.clone()))
                .collect::<Vec<_>>()
        };
        if jobs.is_empty() {
            let records = queue.records.lock().expect("capture queue poisoned");
            if !queue.stopping.load(Ordering::Acquire) {
                let _ = queue.wake.wait_timeout(records, Duration::from_millis(250));
            }
            continue;
        }
        // One batch per session per pass, so a large backlog cannot monopolize
        // capture. The existing writer still serializes all document mutations.
        for (key, snapshot) in jobs {
            if queue.stopping.load(Ordering::Acquire) {
                break;
            }
            queue
                .records
                .lock()
                .expect("capture queue poisoned")
                .get_mut(&key)
                .expect("registered capture")
                .processing = true;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                process_batch(&queue, &snapshot, persist)
            }))
            .unwrap_or_else(|_| Err(anyhow::anyhow!("capture batch panicked")));
            let needs_save = !matches!(&result, Ok(false)) || snapshot.last_error.is_some();
            let mut records = queue.records.lock().expect("capture queue poisoned");
            let record = records.get_mut(&key).expect("registered capture");
            record.processing = false;
            match result {
                Ok(progress) => {
                    // Observe growth (including completion of a partial JSONL
                    // line) without depending on another hook notification.
                    if let Ok(length) = record
                        .request
                        .transcript()
                        .and_then(|p| Ok(std::fs::metadata(p)?.len()))
                    {
                        record.target_bytes = length;
                    }
                    record.failures = 0;
                    record.last_error = None;
                    record.retry_after_ms = if progress { 0 } else { now_ms() + 1000 };
                    if progress {
                        record.last_progress_ms = Some(now_ms());
                    }
                }
                Err(error) => {
                    record.failures = record.failures.saturating_add(1);
                    record.last_error = Some(format!("{error:#}"));
                    record.retry_after_ms =
                        now_ms() + (250u64 << record.failures.min(7)).min(30_000);
                }
            }
            // Idle checks need no durable write; persist changed targets and
            // progress/errors only. Keep the one-second retry deadline in RAM.
            if needs_save || record.target_bytes != snapshot.target_bytes {
                if let Err(error) = save_record(&queue.root, &key, record) {
                    record.last_error = Some(format!("failed to save capture progress: {error:#}"));
                    record.retry_after_ms = now_ms() + 1000;
                    eprintln!("{}", record.last_error.as_deref().unwrap());
                }
            }
            drop(records);
            std::thread::yield_now();
        }
    }
}

#[cfg(all(test, feature = "codex"))]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    };
    use std::time::Instant;
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "membrick-capture-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&root).unwrap();
            Self(root.canonicalize().unwrap())
        }
        fn request(&self, session: &str, backlog: bool) -> CaptureRequest {
            let path = self.0.join(format!("{session}.jsonl"));
            let mut text = message("user", "first", "Capture the amber lighthouse.");
            if backlog {
                for _ in 0..5 {
                    text.push_str(
                        &(json!({"type":"ignored", "padding":"x".repeat(600_000)}).to_string()
                            + "\n"),
                    );
                }
            }
            text.push_str(&message(
                "assistant",
                "last",
                "The amber lighthouse is ready.",
            ));
            std::fs::write(&path, text).unwrap();
            CaptureRequest {
                provider: "codex".into(),
                document_type: "outcome".into(),
                input: json!({"session_id":session,"transcript_path":path,"cwd":self.0,"hook_event_name":"Stop","turn_id":"stable-turn"}),
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn message(role: &str, id: &str, text: &str) -> String {
        json!({"type":"response_item", "payload":{"type":"message", "role":role, "id":id, "content":[{"type":"input_text","text":text}]}}).to_string() + "\n"
    }
    fn until(mut condition: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !condition() {
            assert!(Instant::now() < deadline, "capture did not make progress");
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    fn caught_up(worker: &TranscriptCaptures) -> bool {
        worker
            .status()
            .unwrap()
            .iter()
            .all(|s| s.status == "caught-up")
    }

    #[test]
    fn caught_up_worker_recovers_deleted_and_corrupt_cursors_without_new_hooks() {
        let f = Fixture::new();
        let request = f.request("cursor-recovery", false);
        let writes = Arc::new(AtomicUsize::new(0));
        let seen = writes.clone();
        let worker = MemoryService::start_transcript_captures(&f.0, move |_, _| {
            seen.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();
        worker.submit(request.clone()).unwrap();
        until(|| caught_up(&worker));
        let state = request.state_dir(&f.0).unwrap();
        let cursor = std::fs::read_dir(state.join("capture-cursors"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let initial = writes.load(Ordering::SeqCst);
        std::fs::remove_file(&cursor).unwrap();
        until(|| writes.load(Ordering::SeqCst) > initial && caught_up(&worker));
        let replayed = writes.load(Ordering::SeqCst);
        std::fs::write(&cursor, "{broken").unwrap();
        until(|| writes.load(Ordering::SeqCst) > replayed && caught_up(&worker));
        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&cursor).unwrap()).unwrap();
        assert_eq!(
            value["offset"],
            std::fs::metadata(request.transcript().unwrap())
                .unwrap()
                .len()
        );
        worker.shutdown();
    }

    #[test]
    fn caught_up_worker_captures_new_appends_without_another_hook() {
        use std::io::Write;
        let f = Fixture::new();
        let request = f.request("cursor-growth", false);
        let worker = MemoryService::start_transcript_captures(&f.0, |_, _| Ok(())).unwrap();
        worker.submit(request.clone()).unwrap();
        until(|| caught_up(&worker));
        let path = request.transcript().unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(message("user", "new-message", "A later turn.").as_bytes())
            .unwrap();
        let length = std::fs::metadata(&path).unwrap().len();
        until(|| worker.status().unwrap()[0].captured_bytes == length);
        worker.shutdown();
    }

    #[test]
    fn capture_survives_submitter_exit_and_duplicate_requests_during_write() {
        let f = Fixture::new();
        let request = f.request("duplicate", true);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let first = AtomicBool::new(true);
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let docs = Arc::new(Mutex::new(BTreeMap::new()));
        let (a, m, d) = (active.clone(), maximum.clone(), docs.clone());
        let worker = Arc::new(
            MemoryService::start_transcript_captures(&f.0, move |_, documents| {
                let current = a.fetch_add(1, Ordering::SeqCst) + 1;
                m.fetch_max(current, Ordering::SeqCst);
                if first.swap(false, Ordering::SeqCst) {
                    entered_tx.send(()).unwrap();
                    release_rx.lock().unwrap().recv().unwrap();
                }
                for document in documents {
                    d.lock().unwrap().insert(document.doc_id.clone(), document);
                }
                a.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            })
            .unwrap(),
        );
        let submitter = worker.clone();
        let r = request.clone();
        std::thread::spawn(move || submitter.submit(r).unwrap())
            .join()
            .unwrap();
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let worker = &worker;
                let r = request.clone();
                scope.spawn(move || worker.submit(r).unwrap());
            }
        });
        assert_eq!(
            worker.status().unwrap()[0].captured_bytes,
            0,
            "cursor advanced before durable write"
        );
        // A new hook can extend the target while an older batch is blocked.
        use std::io::Write;
        let appended = message(
            "assistant",
            "appended",
            "The turquoise beacon is now ready.",
        );
        std::fs::OpenOptions::new()
            .append(true)
            .open(request.transcript().unwrap())
            .unwrap()
            .write_all(appended.as_bytes())
            .unwrap();
        worker.submit(request.clone()).unwrap();
        release_tx.send(()).unwrap();
        until(|| caught_up(&worker));
        assert_eq!(maximum.load(Ordering::SeqCst), 1);
        assert!(docs
            .lock()
            .unwrap()
            .values()
            .any(|d| d.content.contains("turquoise beacon")));
        assert!(docs
            .lock()
            .unwrap()
            .values()
            .any(|d| d.content.contains("lighthouse is ready")));
        worker.shutdown();
    }

    #[test]
    fn pending_backlog_resumes_after_shutdown_without_another_hook() {
        let f = Fixture::new();
        let request = f.request("restart", true);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let worker = Arc::new(
            MemoryService::start_transcript_captures(&f.0, move |_, _| {
                entered_tx.send(()).unwrap();
                release_rx.lock().unwrap().recv().unwrap();
                Ok(())
            })
            .unwrap(),
        );
        worker.submit(request.clone()).unwrap();
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let shutdown_worker = worker.clone();
        let shutdown = std::thread::spawn(move || shutdown_worker.shutdown());
        until(|| worker.queue.stopping.load(Ordering::Acquire));
        release_tx.send(()).unwrap();
        shutdown.join().unwrap();
        let before = worker.status().unwrap()[0].captured_bytes;
        assert!(before > 0 && before < worker.status().unwrap()[0].target_bytes);
        drop(worker);
        let docs = Arc::new(Mutex::new(Vec::new()));
        let collected = docs.clone();
        let recovered = MemoryService::start_transcript_captures(&f.0, move |_, documents| {
            collected.lock().unwrap().extend(documents);
            Ok(())
        })
        .unwrap();
        until(|| caught_up(&recovered));
        assert!(docs
            .lock()
            .unwrap()
            .iter()
            .any(|d| d.content.contains("lighthouse is ready")));
        recovered.shutdown();
    }

    #[test]
    fn uncertain_write_retries_same_document_ids_without_advancing_cursor() {
        let f = Fixture::new();
        let request = f.request("retry", false);
        let attempts = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
        let recorded = attempts.clone();
        let worker = MemoryService::start_transcript_captures(&f.0, move |_, documents| {
            let mut attempts = recorded.lock().unwrap();
            attempts.push(documents.iter().map(|d| d.doc_id.clone()).collect());
            if attempts.len() == 1 {
                anyhow::bail!("injected uncertain acknowledgement after durable write");
            }
            Ok(())
        })
        .unwrap();
        worker.submit(request).unwrap();
        until(|| worker.status().unwrap()[0].last_error.is_some());
        let status = worker.status().unwrap().remove(0);
        assert_eq!(status.captured_bytes, 0);
        assert!(status.last_error.unwrap().contains("injected uncertain"));
        until(|| caught_up(&worker));
        {
            let attempts = attempts.lock().unwrap();
            assert_eq!(attempts[0], attempts[1]);
        }
        worker.shutdown();
    }

    #[test]
    fn partial_line_completes_without_another_capture_request() {
        use std::io::Write;
        let f = Fixture::new();
        let request = f.request("partial", false);
        let path = request.transcript().unwrap();
        let text = message("assistant", "partial-tail", "The jade lighthouse is ready.");
        std::fs::write(&path, &text[..text.len() - 1]).unwrap();
        let worker = MemoryService::start_transcript_captures(&f.0, |_, _| Ok(())).unwrap();
        worker.submit(request).unwrap();
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(worker.status().unwrap()[0].captured_bytes, 0);
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"\n")
            .unwrap();
        until(|| caught_up(&worker));
        worker.shutdown();
    }

    #[test]
    fn failed_session_does_not_block_another_session() {
        let f = Fixture::new();
        let failed = f.request("failed", false);
        let successful = f.request("successful", false);
        let worker = MemoryService::start_transcript_captures(&f.0, |r, _| {
            if r.session_id()? == "failed" {
                anyhow::bail!("injected writer failure");
            }
            Ok(())
        })
        .unwrap();
        worker.submit(failed).unwrap();
        worker.submit(successful).unwrap();
        until(|| {
            let s = worker.status().unwrap();
            s.iter()
                .any(|s| s.session_id == "successful" && s.status == "caught-up")
                && s.iter().any(|s| s.last_error.is_some())
        });
        worker.shutdown();
    }

    #[test]
    fn cursor_rewinds_after_truncation_and_empty_transcript_is_initialized() {
        let f = Fixture::new();
        let request = f.request("truncated", false);
        std::fs::write(request.transcript().unwrap(), "").unwrap();
        let worker = MemoryService::start_transcript_captures(&f.0, |_, _| Ok(())).unwrap();
        worker.submit(request.clone()).unwrap();
        let dir = request.state_dir(&f.0).unwrap();
        let files = std::fs::read_dir(dir.join("capture-cursors"))
            .unwrap()
            .count();
        assert_eq!(
            files, 1,
            "initial cursor was not saved for an empty transcript"
        );
        std::fs::write(
            request.transcript().unwrap(),
            message("user", "long", &"z".repeat(1024)),
        )
        .unwrap();
        worker.submit(request.clone()).unwrap();
        until(|| caught_up(&worker));
        std::fs::write(
            request.transcript().unwrap(),
            message("user", "short", "New transcript."),
        )
        .unwrap();
        let expected = std::fs::metadata(request.transcript().unwrap())
            .unwrap()
            .len();
        worker.submit(request).unwrap();
        until(|| caught_up(&worker));
        assert_eq!(worker.status().unwrap()[0].captured_bytes, expected);
        worker.shutdown();
    }

    #[test]
    fn corrupt_pending_record_is_reported_without_disabling_capture() {
        let f = Fixture::new();
        let request = f.request("valid", false);
        let directory = f.0.join(".membrick/runtime/transcript-captures");
        std::fs::create_dir_all(&directory).unwrap();
        let broken = directory.join("broken.json");
        std::fs::write(&broken, "{").unwrap();
        let worker = MemoryService::start_transcript_captures(&f.0, |_, _| Ok(())).unwrap();
        worker.submit(request).unwrap();
        until(|| {
            worker
                .status()
                .unwrap()
                .iter()
                .any(|s| s.session_id == "valid" && s.status == "caught-up")
        });
        let statuses = worker.status().unwrap();
        assert!(statuses.iter().any(|s| s.status == "invalid"
            && s.last_error
                .as_ref()
                .is_some_and(|e| e.contains("broken.json"))));
        assert!(broken.exists());
        worker.shutdown();
    }

    #[cfg(feature = "claude-code")]
    #[test]
    fn claude_uses_same_queue_and_preserves_provider_and_turn() {
        let f = Fixture::new();
        let mut request = f.request("shared-id", false);
        request.provider = "claude".into();
        std::fs::write(request.transcript().unwrap(), json!({"type":"user", "uuid":"claude-message", "turn_id":"claude-turn", "message":{"role":"user","content":"Retain the coral lighthouse."}}).to_string()+"\n").unwrap();
        let docs = Arc::new(Mutex::new(Vec::new()));
        let collected = docs.clone();
        let worker = MemoryService::start_transcript_captures(&f.0, move |_, documents| {
            collected.lock().unwrap().extend(documents);
            Ok(())
        })
        .unwrap();
        let mut codex_request = request.clone();
        codex_request.provider = "codex".into();
        let codex_path = f.0.join("codex-transcript.jsonl");
        std::fs::write(
            &codex_path,
            message(
                "user",
                "codex-message",
                "The pearl lighthouse belongs to Codex.",
            ),
        )
        .unwrap();
        codex_request.input["transcript_path"] = json!(codex_path);
        worker.submit(codex_request).unwrap();
        worker.submit(request).unwrap();
        until(|| caught_up(&worker));
        assert!(docs.lock().unwrap().iter().any(|d| d
            .filters
            .get("provider")
            .is_some_and(|p| p == "claude")
            && d.filters.get("turn_id").is_some_and(|t| t == "claude-turn")));
        let statuses = worker.status().unwrap();
        assert_eq!(statuses.len(), 2);
        assert!(statuses
            .iter()
            .all(|s| s.session_id == "shared-id" && s.captured_bytes == s.target_bytes));
        assert!(statuses.iter().any(|s| s.provider == "codex"));
        assert!(statuses.iter().any(|s| s.provider == "claude"));
        worker.shutdown();
    }
}
