//! Durable staging for hosts that publish accumulated writes explicitly.
use super::*;
use std::io::Write;

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
        Self {
            store: self.store.published_read_view(),
            superseded_ids: self.superseded_ids.clone(),
            request_fingerprints: HashMap::new(),
            request_receipts: HashMap::new(),
            receipts_path: None,
            pending_adds: HashMap::new(),
            conversation_states: Arc::clone(&self.conversation_states),
            relations_cache: Mutex::new(RelationsCache::default()),
            enrichment: Arc::new(Mutex::new(KeyPhraseEnrichment::default())),
            board_state: Mutex::new(BoardState::default()),
        }
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
        }
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
