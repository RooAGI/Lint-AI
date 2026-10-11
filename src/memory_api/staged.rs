//! Durable staging for hosts that publish accumulated writes explicitly.
use super::*;
use std::io::Write;

pub(super) struct PendingPublication {
    receiver: Mutex<std::sync::mpsc::Receiver<anyhow::Result<PublicationResult>>>,
    changed: HashSet<String>,
}

struct PublicationResult {
    view: Box<MemoryService>,
    receipts: HashMap<(String, String), WriteAdjudicationReceipt>,
    checkpoint_prefix: Option<u64>,
    checkpoint: Option<crate::pipeline::PreparedCheckpoint>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StagedAddResponse {
    pub success: bool,
    pub request_id: String,
    pub user_id: String,
    pub session_id: String,
    /// Final adjudication is available after flush, through an idempotent add.
    pub published: bool,
}

impl MemoryService {
    pub(crate) fn published_read_view(&self) -> Self {
        self.read_view_with_store(self.store.published_read_view())
    }

    fn read_view_with_store(&self, store: IndexStore) -> Self {
        Self {
            store,
            superseded_ids: self.superseded_ids.clone(),
            request_fingerprints: HashMap::new(),
            request_receipts: HashMap::new(),
            receipts_path: None,
            pending_adds: HashMap::new(),
            publication: None,
            conversation_states: Arc::clone(&self.conversation_states),
            relations_cache: Mutex::new(RelationsCache::default()),
            enrichment: Arc::new(Mutex::new(KeyPhraseEnrichment::default())),
            board_state: Mutex::new(BoardState::default()),
        }
    }

    pub(crate) fn publication_in_flight(&self) -> bool {
        self.publication.is_some()
    }

    /// Begin one owned build. Pending adds remain counted until completion.
    pub(crate) fn begin_publication(&mut self) -> anyhow::Result<()> {
        self.begin_publication_build(|| {})
    }

    pub(crate) fn begin_checkpoint_publication(&mut self) -> anyhow::Result<()> {
        self.begin_publication_job(true, || {})
    }

    pub(crate) fn publication_revisions(&self) -> (u64, u64, u64) {
        (
            self.store.store_revision(),
            self.store.snapshot_revision(),
            self.store.checkpoint_revision(),
        )
    }

    pub(crate) fn completed_add_responses(
        &self,
        requests: &[AddRequest],
    ) -> anyhow::Result<Vec<AddResponse>> {
        requests
            .iter()
            .map(|request| {
                let key = (request.user_id.clone(), request.request_id.clone());
                let receipt = self
                    .request_receipts
                    .get(&key)
                    .cloned()
                    .or_else(|| self.rebuild_receipt(&key));
                anyhow::ensure!(
                    !self.pending_adds.contains_key(&key),
                    "request has not been published"
                );
                Ok(AddResponse {
                    success: true,
                    request_id: request.request_id.clone(),
                    user_id: request.user_id.clone(),
                    session_id: request.session_id.clone(),
                    adjudication: receipt,
                })
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn begin_publication_paused(
        &mut self,
        gate: std::sync::mpsc::Receiver<()>,
    ) -> anyhow::Result<()> {
        self.begin_publication_build(move || gate.recv().unwrap())
    }

    #[cfg(test)]
    pub(crate) fn begin_checkpoint_publication_paused(
        &mut self,
        gate: std::sync::mpsc::Receiver<()>,
    ) -> anyhow::Result<()> {
        self.begin_publication_job(true, move || gate.recv().unwrap())
    }

    fn begin_publication_build(
        &mut self,
        before_build: impl FnOnce() + Send + 'static,
    ) -> anyhow::Result<()> {
        self.begin_publication_job(false, before_build)
    }

    fn begin_publication_job(
        &mut self,
        checkpoint: bool,
        before_build: impl FnOnce() + Send + 'static,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(self.publication.is_none(), "publication already in flight");
        self.drain_enrichment_inbox();
        let store = self.store.publication_build_view()?;
        let changed = self.store.pending_snapshot_changes();
        let mut view = self.read_view_with_store(store);
        let pending = self.pending_adds.clone();
        let prepare_checkpoint = checkpoint && self.store.store_paths.semantic_dir.is_some();
        // Captured under the mutable owner, after the corresponding record
        // state. Later journal appends are not covered by this checkpoint.
        let checkpoint_prefix = if checkpoint {
            Some(match self.staged_journal_path() {
                Some(path) if path.exists() => {
                    crate::pipeline::persistence::ensure_safe_output_path(&path)?;
                    std::fs::metadata(path)?.len()
                }
                _ => 0,
            })
        } else {
            None
        };
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("memory-publisher".into())
            .spawn(move || {
                before_build();
                let result = (|| -> anyhow::Result<PublicationResult> {
                    view.store.build_prepared_snapshot()?;
                    let prepared_checkpoint = if prepare_checkpoint {
                        Some(view.store.prepare_checkpoint()?)
                    } else {
                        None
                    };
                    let receipts = pending
                        .into_iter()
                        .map(|(key, documents)| {
                            let receipt =
                                view.build_adjudication_receipt(key.1.clone(), &documents);
                            (key, receipt)
                        })
                        .collect();
                    Ok(PublicationResult {
                        view: Box::new(view),
                        receipts,
                        checkpoint_prefix,
                        checkpoint: prepared_checkpoint,
                    })
                })();
                let _ = sender.send(result);
            })?;
        self.publication = Some(PendingPublication {
            receiver: Mutex::new(receiver),
            changed,
        });
        Ok(())
    }

    pub(crate) fn poll_publication(&mut self) -> anyhow::Result<Option<Self>> {
        let Some(pending) = self.publication.as_ref() else {
            return Ok(None);
        };
        let result = match pending
            .receiver
            .lock()
            .map_err(|_| anyhow::anyhow!("publication receiver poisoned"))?
            .try_recv()
        {
            Ok(result) => result,
            Err(std::sync::mpsc::TryRecvError::Empty) => return Ok(None),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                Err(anyhow::anyhow!("publication builder disconnected"))
            }
        };
        let pending = self.publication.take().expect("publication present");
        self.complete_publication(result?, &pending.changed)
    }

    fn wait_publication(&mut self) -> anyhow::Result<()> {
        let Some(pending) = self.publication.take() else {
            return Ok(());
        };
        let result = pending
            .receiver
            .into_inner()
            .map_err(|_| anyhow::anyhow!("publication receiver poisoned"))?
            .recv()
            .map_err(|_| anyhow::anyhow!("publication builder disconnected"))??;
        self.complete_publication(result, &pending.changed)?;
        Ok(())
    }

    fn complete_publication(
        &mut self,
        result: PublicationResult,
        changed: &HashSet<String>,
    ) -> anyhow::Result<Option<Self>> {
        if let Some(prefix) = result.checkpoint_prefix {
            self.store
                .checkpoint_publication_view(&result.view.store, result.checkpoint.as_ref())?;
            self.reclaim_journal_prefix(prefix)?;
            self.store
                .complete_checkpoint(result.view.store.snapshot_revision());
        }
        let accepted = self
            .store
            .accept_publication_view(&result.view.store, changed);
        if !accepted && result.view.store.snapshot_revision() < self.store.snapshot_revision() {
            return Ok(None);
        }
        let keys = result.receipts.keys().cloned().collect::<Vec<_>>();
        self.request_receipts.extend(result.receipts);
        self.persist_receipts(&keys);
        for key in keys {
            self.pending_adds.remove(&key);
        }
        Ok(Some(*result.view))
    }

    fn reclaim_journal_prefix(&self, prefix: u64) -> anyhow::Result<()> {
        let Some(path) = self.staged_journal_path() else {
            return Ok(());
        };
        if !path.exists() {
            anyhow::ensure!(prefix == 0, "checkpoint journal disappeared");
            return Ok(());
        }
        crate::pipeline::persistence::ensure_safe_output_path(&path)?;
        let bytes = std::fs::read(&path)?;
        let offset = usize::try_from(prefix)?;
        anyhow::ensure!(
            offset <= bytes.len(),
            "checkpoint journal prefix exceeds file length"
        );
        anyhow::ensure!(
            offset == 0 || bytes[offset - 1] == b'\n',
            "checkpoint prefix is not a transaction boundary"
        );
        let remaining = std::str::from_utf8(&bytes[offset..])?;
        crate::pipeline::persistence::write_private_text_file_atomic(&path, remaining)
    }
    fn staged_journal_path(&self) -> Option<std::path::PathBuf> {
        self.receipts_path
            .as_ref()
            .map(|p| p.with_file_name("pending-adds.jsonl"))
    }

    /// Persist a validated batch before changing mutable state. No index refresh.
    /// A single owner must serialize mutations to a disk-backed service.
    pub fn stage_add_batch(
        &mut self,
        requests: Vec<AddRequest>,
    ) -> anyhow::Result<Vec<StagedAddResponse>> {
        anyhow::ensure!(!requests.is_empty(), "requests must not be empty");
        let mut fingerprints = HashMap::new();
        let mut new_requests = Vec::new();
        for request in &requests {
            let (fingerprint, documents) = self.prepare_add(request)?;
            let key = (request.user_id.clone(), request.request_id.clone());
            if let Some(previous) = fingerprints.insert(key, fingerprint.clone()) {
                anyhow::ensure!(
                    previous == fingerprint,
                    "request_id was already used with different content"
                );
            } else if !documents.is_empty() {
                new_requests.push(request.clone());
            }
        }
        if !new_requests.is_empty() {
            if let Some(path) = self.staged_journal_path() {
                crate::pipeline::persistence::ensure_safe_output_path(&path)?;
                let created = !path.exists();
                let mut options = std::fs::OpenOptions::new();
                options.create(true).append(true);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::OpenOptionsExt;
                    options.custom_flags(libc::O_NOFOLLOW).mode(0o600);
                }
                let mut file = options.open(&path)?;
                let original_len = file.metadata()?.len();
                let mut bytes = serde_json::to_vec(&new_requests)?;
                bytes.push(b'\n');
                let persisted = file.write_all(&bytes).and_then(|_| file.sync_all());
                if let Err(error) = persisted {
                    file.set_len(original_len)?;
                    file.sync_all()?;
                    return Err(error.into());
                }
                if created {
                    std::fs::File::open(path.parent().expect("journal parent"))?.sync_all()?;
                }
            }
        }
        let mut responses = Vec::with_capacity(requests.len());
        for request in requests {
            let key = (request.user_id.clone(), request.request_id.clone());
            let (response, documents) = self.add_unpublished(request)?;
            if !documents.is_empty() {
                self.pending_adds.insert(key.clone(), documents);
            }
            responses.push(StagedAddResponse {
                success: true,
                request_id: response.request_id,
                user_id: response.user_id,
                session_id: response.session_id,
                published: !self.pending_adds.contains_key(&key),
            });
        }
        Ok(responses)
    }

    pub fn stage_add(&mut self, request: AddRequest) -> anyhow::Result<StagedAddResponse> {
        Ok(self.stage_add_batch(vec![request])?.remove(0))
    }

    pub(crate) fn enrichment_ready(&self) -> bool {
        !self
            .enrichment
            .lock()
            .expect("enrichment lock poisoned")
            .inbox
            .is_empty()
    }

    pub(crate) fn pending_document_count(&self) -> usize {
        self.pending_adds.values().map(Vec::len).sum()
    }

    pub fn pending_add_count(&self) -> usize {
        self.pending_adds.len()
    }

    /// Publish all staged mutations and complete their receipts, then reclaim
    /// the journal only after the canonical state has persisted successfully.
    pub fn flush(&mut self) -> anyhow::Result<()> {
        self.finish_publication(true)
    }

    pub(crate) fn publish_staged(&mut self) -> anyhow::Result<()> {
        self.finish_publication(false)
    }

    fn finish_publication(&mut self, checkpoint: bool) -> anyhow::Result<()> {
        // Synchronous mutations and flush barriers wait for the one builder,
        // then include any later writes. Never build two generations at once.
        self.wait_publication()?;
        self.drain_enrichment_inbox();
        if checkpoint {
            self.store.refresh()?;
        } else {
            self.store.refresh_without_checkpoint()?;
        }
        let keys = self.pending_adds.keys().cloned().collect::<Vec<_>>();
        for key in &keys {
            let documents = &self.pending_adds[key];
            let receipt = self.build_adjudication_receipt(key.1.clone(), documents);
            self.request_receipts.insert(key.clone(), receipt);
        }
        self.persist_receipts(&keys);
        if let Some(path) = self.staged_journal_path().filter(|_| checkpoint) {
            if path.exists() {
                crate::pipeline::persistence::ensure_safe_output_path(&path)?;
                let mut options = std::fs::OpenOptions::new();
                options.write(true);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::OpenOptionsExt;
                    options.custom_flags(libc::O_NOFOLLOW);
                }
                let file = options.open(path)?;
                file.set_len(0)?;
                file.sync_all()?;
            }
        }
        self.pending_adds.clear();
        if checkpoint {
            self.store
                .complete_checkpoint(self.store.snapshot_revision());
        }
        Ok(())
    }

    pub(super) fn recover_staged_adds(&mut self) -> anyhow::Result<()> {
        let Some(path) = self.staged_journal_path() else {
            return Ok(());
        };
        if !path.exists() {
            return Ok(());
        }
        crate::pipeline::persistence::ensure_safe_output_path(&path)?;
        let bytes = std::fs::read(&path)?;
        let mut valid_len = 0;
        for line in bytes.split_inclusive(|b| *b == b'\n') {
            // An interrupted final append was never acknowledged. Ignore only
            // that incomplete tail; corruption in a committed entry is fatal.
            if line.last() != Some(&b'\n') {
                break;
            }
            let requests: Vec<AddRequest> = serde_json::from_slice(line)?;
            for request in requests {
                let key = (request.user_id.clone(), request.request_id.clone());
                let (_, documents) = self.add_unpublished(request)?;
                if !documents.is_empty() {
                    self.pending_adds.insert(key, documents);
                }
            }
            valid_len += line.len();
        }
        if valid_len != bytes.len() {
            let mut options = std::fs::OpenOptions::new();
            options.write(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.custom_flags(libc::O_NOFOLLOW);
            }
            let file = options.open(&path)?;
            file.set_len(valid_len as u64)?;
            file.sync_all()?;
        }
        if !bytes.is_empty() {
            // A crash can interrupt a checkpoint between records and the
            // binary core. Rebuild even if every journal request is already
            // present in records, so an older core cannot hide acknowledged adds.
            self.store.invalidate_snapshot();
            self.flush()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(id: &str) -> AddRequest {
        AddRequest {
            request_id: id.into(),
            user_id: "alice".into(),
            session_id: "session".into(),
            messages: vec![Message {
                role: "user".into(),
                content: "quartz durable staged memory".into(),
                timestamp: None,
                expires_at_ms: None,
                supersedes_id: None,
            }],
        }
    }
    fn options() -> crate::PipelineOptions {
        crate::PipelineOptions {
            key_phrase_enrichment: false,
            ..Default::default()
        }
    }
    fn root() -> std::path::PathBuf {
        let base = std::env::temp_dir().canonicalize().unwrap();
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = base.join(format!(
            "lint-ai-staged-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        path
    }
    fn query() -> SearchRequest {
        SearchRequest {
            query: "quartz".into(),
            user_id: "alice".into(),
            top_k: 10,
            options: None,
            session_id: None,
            scope: None,
            filters: None,
            reference_date: None,
        }
    }
    #[test]
    fn background_build_accepts_later_writes_and_completes_only_covered_receipts() {
        for layout in [
            crate::MemoryIndexLayout::Single,
            crate::MemoryIndexLayout::Segmented {
                query_top_n: 8,
                routing_strategy: crate::segments::SegmentRoutingStrategy::SparseOverlap,
            },
        ] {
            let mut service = MemoryService::in_memory(crate::PipelineOptions {
                memory_index_layout: layout,
                ..options()
            });
            service.stage_add(request("r1")).unwrap();
            let (release, gate) = std::sync::mpsc::channel();
            service
                .begin_publication_build(move || gate.recv().unwrap())
                .unwrap();
            assert!(service.publication_in_flight());
            assert!(service.begin_publication().is_err());
            // The builder is explicitly blocked. Acceptance must still work.
            service.stage_add(request("r2")).unwrap();
            assert_eq!(service.pending_document_count(), 2);
            assert!(!service.stage_add(request("r1")).unwrap().published);
            release.send(()).unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            let view = loop {
                if let Some(view) = service.poll_publication().unwrap() {
                    break view;
                }
                assert!(std::time::Instant::now() < deadline);
                std::thread::sleep(std::time::Duration::from_millis(5));
            };
            assert_eq!(view.search_cached(query()).unwrap().data.len(), 1);
            assert_eq!(service.pending_document_count(), 1);
            assert!(service.stage_add(request("r1")).unwrap().published);
            assert!(!service.stage_add(request("r2")).unwrap().published);
            assert!(service
                .request_receipts
                .contains_key(&("alice".into(), "r1".into())));
            assert!(!service
                .request_receipts
                .contains_key(&("alice".into(), "r2".into())));
            service.flush().unwrap();
            assert_eq!(service.search_cached(query()).unwrap().data.len(), 2);
            assert_eq!(view.search_cached(query()).unwrap().data.len(), 1);
            assert!(service.stage_add(request("r2")).unwrap().published);
        }
    }

    #[test]
    fn checkpoint_keeps_journal_suffix_for_writes_accepted_during_build() {
        let path = root();
        let mut service = MemoryService::at_path(&path, options()).unwrap();
        service.stage_add(request("r1")).unwrap();
        let target = service.publication_revisions().0;
        let (release, gate) = std::sync::mpsc::channel();
        service.begin_checkpoint_publication_paused(gate).unwrap();
        service.stage_add(request("r2")).unwrap();
        assert!(service.publication_revisions().2 < target);
        release.send(()).unwrap();
        service.wait_publication().unwrap();
        assert_eq!(service.publication_revisions().2, target);
        assert_eq!(service.pending_document_count(), 1);
        let journal = std::fs::read_to_string(path.join("pending-adds.jsonl")).unwrap();
        let transactions = journal
            .lines()
            .map(|line| serde_json::from_str::<Vec<AddRequest>>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(transactions.len(), 1);
        assert_eq!(transactions[0][0].request_id, "r2");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(path.join("pending-adds.jsonl"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
        drop(service);
        let recovered = MemoryService::at_path(&path, options()).unwrap();
        assert_eq!(recovered.search_cached(query()).unwrap().data.len(), 2);
        drop(recovered);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn failed_checkpoint_does_not_advance_barrier_or_reclaim_journal() {
        let path = root();
        let mut service = MemoryService::at_path(&path, options()).unwrap();
        service.stage_add(request("r1")).unwrap();
        let before = std::fs::read(path.join("pending-adds.jsonl")).unwrap();
        std::fs::create_dir_all(path.join("semantic/core.bin")).unwrap();
        service.begin_checkpoint_publication().unwrap();
        assert!(service.wait_publication().is_err());
        assert_eq!(service.publication_revisions().2, 0);
        assert_eq!(service.pending_document_count(), 1);
        assert_eq!(
            std::fs::read(path.join("pending-adds.jsonl")).unwrap(),
            before
        );
        std::fs::remove_dir(path.join("semantic/core.bin")).unwrap();
        service.flush().unwrap();
        assert_eq!(service.search_cached(query()).unwrap().data.len(), 1);
        drop(service);
        let recovered = MemoryService::at_path(&path, options()).unwrap();
        assert_eq!(recovered.search_cached(query()).unwrap().data.len(), 1);
        drop(recovered);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn disconnected_background_builder_retains_work_for_retry() {
        let mut service = MemoryService::in_memory(options());
        service.stage_add(request("r1")).unwrap();
        service
            .begin_publication_build(|| panic!("injected builder failure"))
            .unwrap();
        assert!(service.wait_publication().is_err());
        assert!(!service.publication_in_flight());
        assert_eq!(service.pending_document_count(), 1);
        service.flush().unwrap();
        assert_eq!(service.search_cached(query()).unwrap().data.len(), 1);
        assert!(service.stage_add(request("r1")).unwrap().published);
    }

    #[test]
    fn restart_recovers_writes_accepted_during_an_unfinished_background_build() {
        let path = root();
        let mut service = MemoryService::at_path(&path, options()).unwrap();
        service.stage_add(request("r1")).unwrap();
        let (release, gate) = std::sync::mpsc::channel();
        service
            .begin_publication_build(move || gate.recv().unwrap())
            .unwrap();
        service.stage_add(request("r2")).unwrap();
        drop(service);
        release.send(()).unwrap();
        let recovered = MemoryService::at_path(&path, options()).unwrap();
        assert_eq!(recovered.search_cached(query()).unwrap().data.len(), 2);
        drop(recovered);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn staged_add_is_durable_without_publication_and_flush_completes_receipt() {
        let path = root();
        let mut service = MemoryService::at_path(&path, options()).unwrap();
        assert!(!service.stage_add(request("r1")).unwrap().published);
        assert!(service.search_cached(query()).unwrap().data.is_empty());
        assert_eq!(service.pending_add_count(), 1);
        service.flush().unwrap();
        assert_eq!(service.search_cached(query()).unwrap().data.len(), 1);
        assert!(service.add(request("r1")).unwrap().adjudication.is_some());
        assert_eq!(service.pending_add_count(), 0);
        assert_eq!(
            std::fs::metadata(path.join("pending-adds.jsonl"))
                .unwrap()
                .len(),
            0
        );
        drop(service);
        std::fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn restart_replays_acknowledged_add_and_ignores_interrupted_tail() {
        let path = root();
        {
            let mut service = MemoryService::at_path(&path, options()).unwrap();
            service.stage_add(request("r1")).unwrap();
        }
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(path.join("pending-adds.jsonl"))
            .unwrap();
        file.write_all(b"[{\"request_id\":").unwrap();
        drop(file);
        let mut recovered = MemoryService::at_path(&path, options()).unwrap();
        assert_eq!(recovered.search_cached(query()).unwrap().data.len(), 1);
        let receipt = recovered.add(request("r1")).unwrap().adjudication.unwrap();
        assert_eq!(receipt.doc_ids.len(), 1);
        drop(recovered);
        std::fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn invalid_or_conflicting_batch_has_no_side_effects() {
        let mut service = MemoryService::in_memory(options());
        let mut bad = request("r2");
        bad.messages.clear();
        assert!(service.stage_add_batch(vec![request("r1"), bad]).is_err());
        assert_eq!(service.store.len(), 0);
        let mut conflict = request("r1");
        conflict.messages[0].content = "different".into();
        assert!(service
            .stage_add_batch(vec![request("r1"), conflict])
            .is_err());
        assert_eq!(service.store.len(), 0);
    }
    #[test]
    fn journal_failure_does_not_mutate_store() {
        let path = root();
        let mut service = MemoryService::at_path(&path, options()).unwrap();
        std::fs::create_dir(path.join("pending-adds.jsonl")).unwrap();
        assert!(service.stage_add(request("r1")).is_err());
        assert_eq!(service.store.len(), 0);
        assert_eq!(service.pending_add_count(), 0);
        drop(service);
        std::fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn committed_journal_corruption_is_reported() {
        let path = root();
        let service = MemoryService::at_path(&path, options()).unwrap();
        drop(service);
        std::fs::write(path.join("pending-adds.jsonl"), b"invalid\n").unwrap();
        assert!(MemoryService::at_path(&path, options()).is_err());
        std::fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn publication_without_checkpoint_keeps_recovery_journal() {
        let path = root();
        {
            let mut service = MemoryService::at_path(&path, options()).unwrap();
            service.stage_add(request("r1")).unwrap();
            service.publish_staged().unwrap();
            assert_eq!(service.search_cached(query()).unwrap().data.len(), 1);
            assert!(
                std::fs::metadata(path.join("pending-adds.jsonl"))
                    .unwrap()
                    .len()
                    > 0
            );
        }
        let recovered = MemoryService::at_path(&path, options()).unwrap();
        assert_eq!(recovered.search_cached(query()).unwrap().data.len(), 1);
        drop(recovered);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn interrupted_checkpoint_rebuilds_old_core_even_when_records_have_the_add() {
        let path = root();
        let mut service = MemoryService::at_path(&path, options()).unwrap();
        service.add(request("old")).unwrap();
        let core_path = path.join("semantic/core.bin");
        let old_core = std::fs::read(&core_path).unwrap();
        service.stage_add(request("new")).unwrap();
        let journal = std::fs::read(path.join("pending-adds.jsonl")).unwrap();
        service.flush().unwrap();
        drop(service);
        // The records rename completed, but the core rename and WAL cleanup did not.
        std::fs::write(&core_path, old_core).unwrap();
        std::fs::write(path.join("pending-adds.jsonl"), journal).unwrap();
        let recovered = MemoryService::at_path(&path, options()).unwrap();
        assert_eq!(recovered.inspection().source_document_count, 2);
        assert_eq!(recovered.search_cached(query()).unwrap().data.len(), 2);
        drop(recovered);
        std::fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn synchronous_delete_checkpoints_staged_add_without_resurrection() {
        let path = root();
        let mut service = MemoryService::at_path(&path, options()).unwrap();
        service.stage_add(request("new")).unwrap();
        let id = service.store.source_documents()[0].doc_id.clone();
        assert!(service.delete("alice", &id).unwrap());
        drop(service);
        let recovered = MemoryService::at_path(&path, options()).unwrap();
        assert!(recovered.search_cached(query()).unwrap().data.is_empty());
        assert_eq!(recovered.inspection().source_document_count, 0);
        drop(recovered);
        std::fs::remove_dir_all(path).unwrap();
    }
    #[cfg(unix)]
    #[test]
    fn journal_symlink_cannot_modify_another_file() {
        let path = root();
        let mut service = MemoryService::at_path(&path, options()).unwrap();
        let outside = path.join("outside");
        std::fs::write(&outside, b"unchanged").unwrap();
        std::os::unix::fs::symlink(&outside, path.join("pending-adds.jsonl")).unwrap();
        assert!(service.stage_add(request("new")).is_err());
        assert_eq!(std::fs::read(&outside).unwrap(), b"unchanged");
        assert_eq!(service.inspection().source_document_count, 0);
        drop(service);
        std::fs::remove_dir_all(path).unwrap();
    }
}
