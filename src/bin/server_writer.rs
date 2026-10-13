//! One bounded queue and one resident MemoryService own all workspace writes.
use super::*;
use crate::memory_api::MemoryStore;
use serde::{de::DeserializeOwned, Serialize};
use std::sync::mpsc::{self, RecvTimeoutError, SyncSender};
use tokio::sync::oneshot;

const QUEUE_CAPACITY: usize = 32;
const MAX_PENDING_DOCUMENTS: usize = 32768;
#[derive(Clone, Copy)]
pub(super) struct WriteSchedule {
    pub refresh_interval: Duration,
    pub batch_size: usize,
    pub checkpoint_interval: Duration,
}
#[derive(Debug)]
struct PendingWriteLimit;
impl std::fmt::Display for PendingWriteLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("pending write limit reached")
    }
}
impl std::error::Error for PendingWriteLimit {}

#[derive(Debug)]
struct WriterStopped;
impl std::fmt::Display for WriterStopped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("resident memory writer stopped")
    }
}
impl std::error::Error for WriterStopped {}

pub(super) fn mutation_status(error: &anyhow::Error, fallback: StatusCode) -> StatusCode {
    if error.is::<PendingWriteLimit>() {
        StatusCode::TOO_MANY_REQUESTS
    } else if error.is::<WriterStopped>() {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        fallback
    }
}
fn admission_error(error: mpsc::TrySendError<(MemoryStore, Command)>) -> anyhow::Error {
    match error {
        mpsc::TrySendError::Full(_) => PendingWriteLimit.into(),
        mpsc::TrySendError::Disconnected(_) => WriterStopped.into(),
    }
}

type Reply = oneshot::Sender<anyhow::Result<Value>>;
type Mutation = Box<dyn FnOnce(&mut MemoryService) -> anyhow::Result<Value> + Send>;
enum Command {
    Add(Vec<AddRequest>, bool, Reply),
    Upsert(Vec<crate::SourceDocument>, Reply),
    ReplaceGroup(String, String, Vec<crate::SourceDocument>, Reply),
    ReplaceWorkspace(Vec<crate::SourceDocument>, Reply),
    Mutate(Mutation, Reply),
    Flush(Reply),
}

// A target is a facade over the same queue, never a second mutable owner or
// writer thread. Only MemoryService selects the backing store.
#[derive(Clone)]
struct WriterSender {
    sender: SyncSender<(MemoryStore, Command)>,
    target: MemoryStore,
}
impl WriterSender {
    fn try_send(&self, command: Command) -> Result<(), mpsc::TrySendError<(MemoryStore, Command)>> {
        self.sender.try_send((self.target, command))
    }
}

enum BarrierWork {
    Add(MemoryStore, Vec<AddRequest>),
    None,
}
struct BarrierReply {
    revisions: [u64; 2],
    work: BarrierWork,
    reply: Reply,
}
impl BarrierReply {
    fn for_store(target: MemoryStore, revision: u64, work: BarrierWork, reply: Reply) -> Self {
        let mut revisions = [0; 2];
        revisions[target.slot()] = revision;
        Self {
            revisions,
            work,
            reply,
        }
    }
    fn response(self, owner: &MemoryService) {
        let response = match self.work {
            BarrierWork::Add(target, requests) => owner
                .memory_store(target)
                .completed_add_responses(&requests)
                .and_then(|responses| Ok(serde_json::to_value(responses)?)),
            BarrierWork::None => Ok(Value::Null),
        };
        let _ = self.reply.send(response);
    }
}
fn fail_barriers(barriers: &mut Vec<BarrierReply>, message: &str) {
    for barrier in barriers.drain(..) {
        let _ = barrier.reply.send(Err(anyhow::anyhow!("{message}")));
    }
}
struct StoreSchedule {
    first_pending: Option<Instant>,
    last_checkpoint: Instant,
    checkpoint_build: bool,
    retry_after: Option<Instant>,
    visible_revision: u64,
}
impl StoreSchedule {
    fn new(visible_revision: u64) -> Self {
        Self {
            first_pending: None,
            last_checkpoint: Instant::now(),
            checkpoint_build: false,
            retry_after: None,
            visible_revision,
        }
    }
}

fn publish_owner(
    owner: &MemoryService,
    published: &RwLock<Arc<MemoryService>>,
    generation: &mut u64,
    schedules: &mut [StoreSchedule; 2],
    targets: &[MemoryStore],
) -> anyhow::Result<()> {
    let mut view = owner.published_read_view()?;
    *generation = generation.saturating_add(1);
    view.set_published_generation(*generation);
    publish_view(published, view)?;
    for &target in targets {
        schedules[target.slot()].visible_revision =
            owner.memory_store(target).publication_revisions().1;
    }
    Ok(())
}

fn flush_owner(owner: &mut MemoryService, targets: &[MemoryStore]) -> anyhow::Result<()> {
    // Finish any frozen build before synchronously rebuilding another store.
    for &target in targets {
        owner.memory_store_mut(target).wait_publication()?;
    }
    for &target in targets {
        let store = owner.memory_store_mut(target);
        let (accepted, published, durable) = store.publication_revisions();
        if accepted > published
            || accepted > durable
            || store.publication_in_flight()
            || store.enrichment_ready()
        {
            store.flush()?;
        }
    }
    Ok(())
}

fn run_writer(
    service: Arc<RwLock<MemoryService>>,
    published: Arc<RwLock<Arc<MemoryService>>>,
    receiver: mpsc::Receiver<(MemoryStore, Command)>,
    schedule: WriteSchedule,
) {
    let targets = if service.read().is_ok_and(|s| s.has_workspace_store()) {
        vec![MemoryStore::Captured, MemoryStore::Workspace]
    } else {
        vec![MemoryStore::Captured]
    };
    let mut schedules = [StoreSchedule::new(0), StoreSchedule::new(0)];
    let mut generation = published
        .read()
        .map(|s| s.inspection().snapshot_revision)
        .unwrap_or(0);
    if let Ok(owner) = service.read() {
        for &target in &targets {
            schedules[target.slot()].visible_revision =
                owner.memory_store(target).publication_revisions().1;
        }
    }
    let mut barriers = Vec::<BarrierReply>::new();
    let mut next_store = 0;
    loop {
        let building = service.read().is_ok_and(|owner| {
            targets
                .iter()
                .any(|&target| owner.memory_store(target).publication_in_flight())
        });
        let timeout = if building || !barriers.is_empty() {
            Duration::from_millis(10)
        } else {
            schedules
                .iter()
                .filter_map(|s| s.first_pending)
                .map(|first| schedule.refresh_interval.saturating_sub(first.elapsed()))
                .min()
                .unwrap_or(schedule.refresh_interval)
        };
        match receiver.recv_timeout(timeout) {
            Ok((requested_target, command)) => {
                let target = if requested_target == MemoryStore::Workspace && targets.len() == 1 {
                    MemoryStore::Captured
                } else {
                    requested_target
                };
                let slot = target.slot();
                match command {
                    Command::Add(requests, wait, reply) => {
                        let result = (|| -> anyhow::Result<(Value, u64)> {
                            anyhow::ensure!(
                                !wait || barriers.len() < QUEUE_CAPACITY,
                                PendingWriteLimit
                            );
                            let mut owner = service
                                .write()
                                .map_err(|_| anyhow::anyhow!("memory writer poisoned"))?;
                            let pending: usize = targets
                                .iter()
                                .map(|&t| owner.memory_store(t).pending_document_count())
                                .sum();
                            anyhow::ensure!(
                                pending + requests.iter().map(|r| r.messages.len()).sum::<usize>()
                                    <= MAX_PENDING_DOCUMENTS,
                                PendingWriteLimit
                            );
                            let store = owner.memory_store_mut(target);
                            let responses = store.stage_add_batch(requests.clone())?;
                            if store.pending_document_count() > 0
                                && schedules[slot].first_pending.is_none()
                            {
                                schedules[slot].first_pending = Some(Instant::now());
                            }
                            Ok((
                                serde_json::to_value(responses)?,
                                store.publication_revisions().0,
                            ))
                        })();
                        match result {
                            Ok((_, revision)) if wait => barriers.push(BarrierReply::for_store(
                                target,
                                revision,
                                BarrierWork::Add(target, requests),
                                reply,
                            )),
                            Ok((value, _)) => {
                                let _ = reply.send(Ok(value));
                            }
                            Err(error) => {
                                let _ = reply.send(Err(error));
                            }
                        }
                    }
                    command @ (Command::Upsert(..) | Command::ReplaceWorkspace(..)) => {
                        let (documents, reply, reconcile) = match command {
                            Command::Upsert(documents, reply) => (documents, reply, false),
                            Command::ReplaceWorkspace(documents, reply) => (documents, reply, true),
                            _ => unreachable!(),
                        };
                        let result = (|| -> anyhow::Result<Option<u64>> {
                            anyhow::ensure!(barriers.len() < QUEUE_CAPACITY, PendingWriteLimit);
                            let mut owner = service
                                .write()
                                .map_err(|_| anyhow::anyhow!("memory writer poisoned"))?;
                            let pending: usize = targets
                                .iter()
                                .map(|&t| owner.memory_store(t).pending_document_count())
                                .sum();
                            anyhow::ensure!(
                                pending + documents.len() <= MAX_PENDING_DOCUMENTS,
                                PendingWriteLimit
                            );
                            let store = owner.memory_store_mut(target);
                            let changed = if reconcile {
                                store.replace_workspace_documents(documents)?
                            } else {
                                for document in documents {
                                    store.upsert(document);
                                }
                                true
                            };
                            if changed {
                                schedules[slot]
                                    .first_pending
                                    .get_or_insert_with(Instant::now);
                                Ok(Some(store.publication_revisions().0))
                            } else {
                                Ok(None)
                            }
                        })();
                        match result {
                            Ok(Some(revision)) => barriers.push(BarrierReply::for_store(
                                target,
                                revision,
                                BarrierWork::None,
                                reply,
                            )),
                            Ok(None) => {
                                let _ = reply.send(Ok(Value::Null));
                            }
                            Err(error) => {
                                let _ = reply.send(Err(error));
                            }
                        }
                    }
                    Command::ReplaceGroup(key, value, documents, reply) => {
                        let result = (|| -> anyhow::Result<u64> {
                            anyhow::ensure!(barriers.len() < QUEUE_CAPACITY, PendingWriteLimit);
                            let mut owner = service
                                .write()
                                .map_err(|_| anyhow::anyhow!("memory writer poisoned"))?;
                            let pending: usize = targets
                                .iter()
                                .map(|&t| owner.memory_store(t).pending_document_count())
                                .sum();
                            anyhow::ensure!(
                                pending + documents.len() <= MAX_PENDING_DOCUMENTS,
                                PendingWriteLimit
                            );
                            let store = owner.memory_store_mut(target);
                            store.replace_source_group(&key, &value, documents)?;
                            schedules[slot]
                                .first_pending
                                .get_or_insert_with(Instant::now);
                            Ok(store.publication_revisions().0)
                        })();
                        match result {
                            Ok(revision) => barriers.push(BarrierReply::for_store(
                                target,
                                revision,
                                BarrierWork::None,
                                reply,
                            )),
                            Err(error) => {
                                let _ = reply.send(Err(error));
                            }
                        }
                    }
                    Command::Mutate(mutation, reply) => {
                        // Lifecycle and board operations use the same admission queue.
                        // Drain older staged writes before synchronous deletion/replacement
                        // so recovery cannot resurrect an older accepted document.
                        let result = (|| -> anyhow::Result<Value> {
                            let mut owner = service
                                .write()
                                .map_err(|_| anyhow::anyhow!("memory writer poisoned"))?;
                            flush_owner(&mut owner, &targets)?;
                            let result = mutation(owner.memory_store_mut(target));
                            flush_owner(&mut owner, &targets)?;
                            publish_owner(
                                &owner,
                                &published,
                                &mut generation,
                                &mut schedules,
                                &targets,
                            )?;
                            for s in &mut schedules {
                                s.first_pending = None;
                                s.checkpoint_build = false;
                                s.last_checkpoint = Instant::now();
                            }
                            result
                        })();
                        let _ = reply.send(result);
                    }
                    Command::Flush(reply) => {
                        if barriers.len() >= QUEUE_CAPACITY {
                            let _ = reply.send(Err(PendingWriteLimit.into()));
                        } else if let Ok(owner) = service.read() {
                            let mut revisions = [0; 2];
                            for &target in &targets {
                                revisions[target.slot()] =
                                    owner.memory_store(target).publication_revisions().0;
                            }
                            barriers.push(BarrierReply {
                                revisions,
                                work: BarrierWork::None,
                                reply,
                            });
                        } else {
                            let _ = reply.send(Err(anyhow::anyhow!("memory writer poisoned")));
                        }
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                let result = (|| -> anyhow::Result<()> {
                    let mut owner = service
                        .write()
                        .map_err(|_| anyhow::anyhow!("memory writer poisoned"))?;
                    flush_owner(&mut owner, &targets)?;
                    publish_owner(
                        &owner,
                        &published,
                        &mut generation,
                        &mut schedules,
                        &targets,
                    )?;
                    for barrier in barriers.drain(..) {
                        barrier.response(&owner);
                    }
                    Ok(())
                })();
                if let Err(error) = result {
                    eprintln!("memory writer shutdown checkpoint failed: {error:#}");
                    fail_barriers(&mut barriers, &format!("{error:#}"));
                }
                break;
            }
        }

        for &target in &targets {
            let slot = target.slot();
            let completion = service
                .write()
                .map_err(|_| anyhow::anyhow!("memory writer poisoned"))
                .and_then(|mut owner| {
                    if owner.memory_store_mut(target).poll_publication()?.is_some() {
                        publish_owner(
                            &owner,
                            &published,
                            &mut generation,
                            &mut schedules,
                            &targets,
                        )?;
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                });
            match completion {
                Ok(true) => {
                    if schedules[slot].checkpoint_build {
                        schedules[slot].last_checkpoint = Instant::now();
                    }
                    schedules[slot].checkpoint_build = false;
                    if service
                        .read()
                        .is_ok_and(|owner| owner.memory_store(target).pending_document_count() > 0)
                    {
                        schedules[slot]
                            .first_pending
                            .get_or_insert_with(Instant::now);
                    } else {
                        schedules[slot].first_pending = None;
                    }
                }
                Ok(false) => {}
                Err(error) => {
                    eprintln!("memory publication failed; durable journal retained: {error:#}");
                    fail_barriers(&mut barriers, &format!("{error:#}"));
                    schedules[slot].checkpoint_build = false;
                    schedules[slot].retry_after = Some(Instant::now() + schedule.refresh_interval);
                    schedules[slot].first_pending = Some(Instant::now());
                }
            }
        }
        if let Ok(owner) = service.read() {
            let mut index = 0;
            while index < barriers.len() {
                let ready = targets.iter().all(|&target| {
                    let (_, published_revision, durable) =
                        owner.memory_store(target).publication_revisions();
                    let revision = barriers[index].revisions[target.slot()];
                    revision <= durable
                        && revision <= published_revision
                        && revision <= schedules[target.slot()].visible_revision
                });
                if ready {
                    barriers.remove(index).response(&owner);
                } else {
                    index += 1;
                }
            }
        }
        // At most one frozen generation builds at a time across both stores.
        // Rotate eligible stores so workspace changes cannot starve captures.
        if service.read().is_ok_and(|owner| {
            targets
                .iter()
                .any(|&t| owner.memory_store(t).publication_in_flight())
        }) {
            continue;
        }
        for offset in 0..targets.len() {
            let target_index = (next_store + offset) % targets.len();
            let target = targets[target_index];
            let slot = target.slot();
            let (pending, enrichment_ready) = service
                .read()
                .map(|owner| {
                    let store = owner.memory_store(target);
                    (store.pending_document_count(), store.enrichment_ready())
                })
                .unwrap_or((MAX_PENDING_DOCUMENTS, false));
            let due = schedules[slot]
                .first_pending
                .is_some_and(|first| first.elapsed() >= schedule.refresh_interval);
            let checkpoint = barriers
                .iter()
                .any(|b| b.revisions[slot] > schedules[slot].visible_revision)
                || barriers.iter().any(|b| {
                    service.read().is_ok_and(|owner| {
                        b.revisions[slot] > owner.memory_store(target).publication_revisions().2
                    })
                })
                || (schedules[slot].last_checkpoint.elapsed() >= schedule.checkpoint_interval
                    && service.read().is_ok_and(|owner| {
                        let (accepted, _, durable) =
                            owner.memory_store(target).publication_revisions();
                        accepted > durable
                    }));
            let can_retry = schedules[slot]
                .retry_after
                .is_none_or(|deadline| Instant::now() >= deadline);
            if can_retry
                && ((pending > 0 && (due || pending >= schedule.batch_size))
                    || checkpoint
                    || enrichment_ready)
            {
                let result = service
                    .write()
                    .map_err(|_| anyhow::anyhow!("memory writer poisoned"))
                    .and_then(|mut owner| {
                        let store = owner.memory_store_mut(target);
                        if checkpoint {
                            store.begin_checkpoint_publication()
                        } else {
                            store.begin_publication()
                        }
                    });
                match result {
                    Ok(()) => {
                        schedules[slot].first_pending = None;
                        schedules[slot].checkpoint_build = checkpoint;
                        schedules[slot].retry_after = None;
                    }
                    Err(error) => {
                        eprintln!("memory publication capture failed; durable journal retained: {error:#}");
                        fail_barriers(&mut barriers, &format!("{error:#}"));
                        schedules[slot].retry_after =
                            Some(Instant::now() + schedule.refresh_interval);
                        schedules[slot].first_pending = Some(Instant::now());
                    }
                }
                next_store = (target_index + 1) % targets.len();
                break;
            }
        }
    }
}

#[derive(Clone)]
pub(super) struct StagedWriter {
    sender: WriterSender,
    published: Arc<RwLock<Arc<MemoryService>>>,
}
impl StagedWriter {
    #[cfg(test)]
    pub(super) fn start(service: Arc<RwLock<MemoryService>>) -> anyhow::Result<Self> {
        Self::start_with_schedule(
            service,
            WriteSchedule {
                refresh_interval: Duration::from_millis(250),
                batch_size: 512,
                checkpoint_interval: Duration::from_secs(30),
            },
        )
    }
    pub(super) fn start_with_schedule(
        service: Arc<RwLock<MemoryService>>,
        schedule: WriteSchedule,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !schedule.refresh_interval.is_zero()
                && schedule.batch_size > 0
                && !schedule.checkpoint_interval.is_zero(),
            "write schedule values must be positive"
        );
        let view = service
            .read()
            .map_err(|_| anyhow::anyhow!("memory writer poisoned"))?
            .published_read_view()?;
        let published = Arc::new(RwLock::new(Arc::new(view)));
        service
            .write()
            .map_err(|_| anyhow::anyhow!("memory writer poisoned"))?
            .retain_published_search_view(published.clone());
        let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
        let worker = Self {
            sender: WriterSender {
                sender,
                target: MemoryStore::Captured,
            },
            published: published.clone(),
        };
        std::thread::Builder::new()
            .name("memory-writer".into())
            .spawn(move || run_writer(service, published, receiver, schedule))?;
        Ok(worker)
    }
    pub(super) fn for_workspace(&self) -> Self {
        let mut writer = self.clone();
        writer.sender.target = MemoryStore::Workspace;
        writer
    }
    pub(super) fn read_view(&self) -> anyhow::Result<Arc<MemoryService>> {
        Ok(self
            .published
            .read()
            .map_err(|_| anyhow::anyhow!("published view poisoned"))?
            .clone())
    }
    fn send(&self, command: Command) -> Result<(), StatusCode> {
        self.sender.try_send(command).map_err(|error| match error {
            mpsc::TrySendError::Full(_) => StatusCode::TOO_MANY_REQUESTS,
            mpsc::TrySendError::Disconnected(_) => StatusCode::SERVICE_UNAVAILABLE,
        })
    }
    fn status(error: anyhow::Error) -> StatusCode {
        if error.is::<PendingWriteLimit>() {
            StatusCode::TOO_MANY_REQUESTS
        } else {
            eprintln!("staged mutation failed: {error:#}");
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
    pub(super) async fn add(
        &self,
        requests: Vec<AddRequest>,
        wait: bool,
    ) -> Result<Value, StatusCode> {
        let (reply, receive) = oneshot::channel();
        self.send(Command::Add(requests, wait, reply))?;
        receive
            .await
            .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?
            .map_err(Self::status)
    }
    pub(super) async fn upsert_documents(
        &self,
        documents: Vec<crate::SourceDocument>,
    ) -> Result<(), StatusCode> {
        let (reply, receive) = oneshot::channel();
        self.send(Command::Upsert(documents, reply))?;
        receive
            .await
            .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?
            .map_err(Self::status)?;
        Ok(())
    }
    pub(super) async fn replace_group(
        &self,
        key: String,
        value: String,
        documents: Vec<crate::SourceDocument>,
    ) -> Result<(), StatusCode> {
        let (reply, receive) = oneshot::channel();
        self.send(Command::ReplaceGroup(key, value, documents, reply))?;
        receive
            .await
            .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?
            .map_err(Self::status)?;
        Ok(())
    }
    pub(super) fn upsert_documents_blocking(
        &self,
        documents: Vec<crate::SourceDocument>,
    ) -> anyhow::Result<()> {
        let (reply, receive) = oneshot::channel();
        self.sender
            .try_send(Command::Upsert(documents, reply))
            .map_err(admission_error)?;
        receive.blocking_recv().map_err(|_| WriterStopped)??;
        Ok(())
    }
    pub(super) fn mutate_blocking<T: Serialize + DeserializeOwned + 'static>(
        &self,
        mutation: impl FnOnce(&mut MemoryService) -> anyhow::Result<T> + Send + 'static,
    ) -> anyhow::Result<T> {
        let (reply, receive) = oneshot::channel();
        self.sender
            .try_send(Command::Mutate(
                Box::new(move |owner| Ok(serde_json::to_value(mutation(owner)?)?)),
                reply,
            ))
            .map_err(admission_error)?;
        let value = receive.blocking_recv().map_err(|_| WriterStopped)??;
        Ok(serde_json::from_value(value)?)
    }
    pub(super) async fn mutate(
        &self,
        mutation: impl FnOnce(&mut MemoryService) -> anyhow::Result<Value> + Send + 'static,
    ) -> anyhow::Result<Value> {
        let (reply, receive) = oneshot::channel();
        self.sender
            .try_send(Command::Mutate(Box::new(mutation), reply))
            .map_err(admission_error)?;
        receive.await.map_err(|_| WriterStopped)?
    }
    pub(super) async fn flush(&self) -> Result<(), StatusCode> {
        let (reply, receive) = oneshot::channel();
        self.send(Command::Flush(reply))?;
        receive
            .await
            .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?
            .map_err(Self::status)?;
        Ok(())
    }
    pub(super) async fn replace_workspace_documents(
        &self,
        documents: Vec<crate::SourceDocument>,
    ) -> Result<(), StatusCode> {
        let (reply, receive) = oneshot::channel();
        self.send(Command::ReplaceWorkspace(documents, reply))?;
        receive
            .await
            .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?
            .map_err(Self::status)?;
        Ok(())
    }
}
fn publish_view(published: &RwLock<Arc<MemoryService>>, view: MemoryService) -> anyhow::Result<()> {
    let next = Arc::new(view);
    let mut current = published
        .write()
        .map_err(|_| anyhow::anyhow!("published view poisoned"))?;
    if next.inspection().snapshot_revision > current.inspection().snapshot_revision {
        *current = next;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_api::Message;
    fn request(i: usize) -> AddRequest {
        AddRequest {
            request_id: format!("r{i}"),
            user_id: "alice".into(),
            session_id: "session".into(),
            messages: vec![Message {
                role: "user".into(),
                content: format!("quartz memory {i}"),
                timestamp: None,
                expires_at_ms: None,
                supersedes_id: None,
            }],
        }
    }
    fn query() -> SearchRequest {
        SearchRequest {
            query: "quartz".into(),
            user_id: "alice".into(),
            top_k: 100,
            options: None,
            session_id: None,
            scope: None,
            filters: None,
            reference_date: None,
        }
    }
    fn owner() -> Arc<RwLock<MemoryService>> {
        Arc::new(RwLock::new(MemoryService::in_memory(PipelineOptions {
            key_phrase_enrichment: false,
            ..Default::default()
        })))
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_adds_share_publication_and_retained_views_stay_stable() {
        let owner = owner();
        let worker = StagedWriter::start(owner.clone()).unwrap();
        let before = worker.read_view().unwrap();
        let mut tasks = Vec::new();
        for i in 0..24 {
            let worker = worker.clone();
            tasks.push(tokio::spawn(async move {
                worker.add(vec![request(i)], false).await.unwrap()
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        worker.flush().await.unwrap();
        let view = worker.read_view().unwrap();
        assert_eq!(view.inspection().source_document_count, 24);

        assert!(!view.search_cached(query()).unwrap().data.is_empty());
        assert!(before.search_cached(query()).unwrap().data.is_empty());
        assert_eq!(owner.read().unwrap().pending_add_count(), 0);
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn search_does_not_wait_for_mutable_owner_lock() {
        let owner = owner();
        owner.write().unwrap().add(request(0)).unwrap();
        let worker = StagedWriter::start(owner.clone()).unwrap();
        {
            let _blocked_writer = owner.write().unwrap();
            let reader = worker.read_view().unwrap();
            assert_eq!(reader.search_cached(query()).unwrap().data.len(), 1);
        }
        worker.flush().await.unwrap();
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deferred_visibility_barriers_are_bounded_before_accepting_extra_write() {
        let owner = owner();
        let worker = StagedWriter::start(owner.clone()).unwrap();
        worker.add(vec![request(0)], false).await.unwrap();
        let (release, gate) = mpsc::channel();
        owner
            .write()
            .unwrap()
            .begin_publication_paused(gate)
            .unwrap();
        let mut tasks = Vec::new();
        for id in 1..=QUEUE_CAPACITY {
            let writer = worker.clone();
            tasks.push(tokio::spawn(async move {
                writer.add(vec![request(id)], true).await
            }));
            // Wait for acceptance, so the command queue itself does not fill.
            tokio::time::timeout(Duration::from_secs(2), async {
                loop {
                    if owner.read().unwrap().pending_document_count() == id + 1 {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .unwrap();
        }
        assert_eq!(
            worker.add(vec![request(1000)], true).await.unwrap_err(),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            owner.read().unwrap().pending_document_count(),
            QUEUE_CAPACITY + 1
        );
        release.send(()).unwrap();
        for task in tasks {
            tokio::time::timeout(Duration::from_secs(10), task)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
        }
        worker.flush().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flush_barrier_allows_later_acceptance_and_only_waits_for_target() {
        let owner = owner();
        let worker = StagedWriter::start_with_schedule(
            owner.clone(),
            WriteSchedule {
                refresh_interval: Duration::from_secs(30),
                batch_size: 10000,
                checkpoint_interval: Duration::from_secs(60),
            },
        )
        .unwrap();
        worker.add(vec![request(1)], false).await.unwrap();
        let target = owner.read().unwrap().publication_revisions().0;
        let (release, gate) = mpsc::channel();
        owner
            .write()
            .unwrap()
            .begin_checkpoint_publication_paused(gate)
            .unwrap();
        let (reply, mut receive) = oneshot::channel();
        worker.sender.try_send(Command::Flush(reply)).unwrap();
        tokio::time::timeout(Duration::from_secs(2), worker.add(vec![request(2)], false))
            .await
            .unwrap()
            .unwrap();
        assert!(receive.try_recv().is_err());
        assert_eq!(owner.read().unwrap().pending_document_count(), 2);
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(10), receive)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(owner.read().unwrap().publication_revisions().2, target);
        assert_eq!(owner.read().unwrap().pending_document_count(), 1);
        assert_eq!(
            worker
                .read_view()
                .unwrap()
                .search_cached(query())
                .unwrap()
                .data
                .len(),
            1
        );
        worker.flush().await.unwrap();
        assert_eq!(
            worker
                .read_view()
                .unwrap()
                .search_cached(query())
                .unwrap()
                .data
                .len(),
            2
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn completed_old_view_cannot_replace_a_newer_synchronous_publication() {
        let owner = owner();
        let worker = StagedWriter::start(owner.clone()).unwrap();
        worker.add(vec![request(1)], true).await.unwrap();
        let old = owner.read().unwrap().published_read_view().unwrap();
        worker.add(vec![request(2)], true).await.unwrap();
        let revision = worker.read_view().unwrap().inspection().snapshot_revision;
        publish_view(&worker.published, old).unwrap();
        assert_eq!(
            worker.read_view().unwrap().inspection().snapshot_revision,
            revision
        );
        assert_eq!(
            worker
                .read_view()
                .unwrap()
                .search_cached(query())
                .unwrap()
                .data
                .len(),
            2
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn acknowledgements_continue_while_publisher_is_paused() {
        let owner = owner();
        let worker = StagedWriter::start_with_schedule(
            owner.clone(),
            WriteSchedule {
                refresh_interval: Duration::from_secs(30),
                batch_size: 10000,
                checkpoint_interval: Duration::from_secs(60),
            },
        )
        .unwrap();
        worker.add(vec![request(1)], false).await.unwrap();
        let (release, gate) = mpsc::channel();
        owner
            .write()
            .unwrap()
            .begin_publication_paused(gate)
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), worker.add(vec![request(2)], false))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(owner.read().unwrap().pending_document_count(), 2);
        assert!(owner.read().unwrap().publication_in_flight());
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if owner.read().unwrap().pending_document_count() == 1 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(
            worker
                .read_view()
                .unwrap()
                .search_cached(query())
                .unwrap()
                .data
                .len(),
            1
        );
        worker.flush().await.unwrap();
        assert_eq!(
            worker
                .read_view()
                .unwrap()
                .search_cached(query())
                .unwrap()
                .data
                .len(),
            2
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn continuous_adds_cannot_postpone_interval_publication() {
        let owner = owner();
        let worker = StagedWriter::start_with_schedule(
            owner,
            WriteSchedule {
                refresh_interval: Duration::from_millis(100),
                batch_size: 10000,
                checkpoint_interval: Duration::from_secs(30),
            },
        )
        .unwrap();
        let producer = tokio::spawn({
            let worker = worker.clone();
            async move {
                for i in 0..200 {
                    worker.add(vec![request(i)], false).await.unwrap();
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        });
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if !worker
                    .read_view()
                    .unwrap()
                    .search_cached(query())
                    .unwrap()
                    .data
                    .is_empty()
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();
        assert!(
            !producer.is_finished(),
            "publication waited for writes to stop"
        );
        producer.abort();
        let _ = producer.await;
        worker.flush().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelling_reply_does_not_discard_accepted_command() {
        let owner = owner();
        let worker = StagedWriter::start(owner.clone()).unwrap();
        let guard = owner.write().unwrap();
        let (reply, receive) = oneshot::channel();
        worker
            .sender
            .try_send(Command::Add(vec![request(0)], false, reply))
            .unwrap();
        drop(receive);
        drop(guard);
        worker.flush().await.unwrap();
        assert_eq!(
            worker
                .read_view()
                .unwrap()
                .inspection()
                .source_document_count,
            1
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bounded_queue_rejects_before_mutating() {
        let owner = owner();
        let worker = StagedWriter::start(owner.clone()).unwrap();
        let guard = owner.write().unwrap();
        let mut sent = 0;
        for i in 0..QUEUE_CAPACITY + 2 {
            let (reply, _) = oneshot::channel();
            if worker
                .sender
                .try_send(Command::Add(vec![request(i)], false, reply))
                .is_ok()
            {
                sent += 1;
            } else {
                break;
            }
        }
        assert!(sent <= QUEUE_CAPACITY + 1);
        assert_eq!(guard.inspection().source_document_count, 0);
        drop(guard);
        // Once admission reopens, a flush follows every accepted command.
        loop {
            match worker.flush().await {
                Ok(()) => break,
                Err(StatusCode::TOO_MANY_REQUESTS) => {
                    tokio::time::sleep(Duration::from_millis(10)).await
                }
                Err(error) => panic!("flush failed: {error}"),
            }
        }
        assert_eq!(
            worker
                .read_view()
                .unwrap()
                .inspection()
                .source_document_count,
            sent
        );
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn workspace_and_capture_admission_share_one_bounded_queue() {
        let options = PipelineOptions {
            key_phrase_enrichment: false,
            ..crate::memory_api::workspace::segmented_store_options()
        };
        let owner = Arc::new(RwLock::new(MemoryService::retain_workspace_store(
            MemoryService::in_memory(options.clone()),
            MemoryService::in_memory(options),
        )));
        let captured = StagedWriter::start(owner.clone()).unwrap();
        let workspace = captured.for_workspace();
        let before = captured.read_view().unwrap();
        let guard = owner.write().unwrap();
        let mut sent = 0;
        for id in 0..QUEUE_CAPACITY * 2 + 4 {
            let writer = if id % 2 == 0 { &captured } else { &workspace };
            let (reply, _) = oneshot::channel();
            if writer
                .sender
                .try_send(Command::Add(vec![request(id)], false, reply))
                .is_err()
            {
                break;
            }
            sent += 1;
        }
        assert!(
            sent <= QUEUE_CAPACITY + 1,
            "the two targets admitted more than one queue's capacity"
        );
        drop(guard);
        loop {
            match captured.flush().await {
                Ok(()) => break,
                Err(StatusCode::TOO_MANY_REQUESTS) => {
                    tokio::time::sleep(Duration::from_millis(5)).await
                }
                Err(error) => panic!("global flush failed: {error}"),
            }
        }
        assert_eq!(
            captured
                .read_view()
                .unwrap()
                .inspection()
                .source_document_count,
            sent
        );
        assert_eq!(before.inspection().source_document_count, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn global_flush_and_queued_delete_recover_both_stores_without_resurrection() {
        let root = super::super::tests::security_test_root();
        let captured_root = root.join(".membrick/memory");
        let workspace_root = root.join(".membrick/workspace-memory");
        let options = PipelineOptions {
            key_phrase_enrichment: false,
            ..crate::memory_api::workspace::segmented_store_options()
        };
        let owner = MemoryService::retain_workspace_store(
            MemoryService::at_path(&captured_root, options.clone()).unwrap(),
            MemoryService::at_path(&workspace_root, options.clone()).unwrap(),
        );
        let owner = Arc::new(RwLock::new(owner));
        let captured = StagedWriter::start_with_schedule(
            owner.clone(),
            WriteSchedule {
                refresh_interval: Duration::from_secs(60),
                batch_size: 10000,
                checkpoint_interval: Duration::from_secs(60),
            },
        )
        .unwrap();
        let workspace = captured.for_workspace();
        captured.add(vec![request(0)], false).await.unwrap();
        workspace.add(vec![request(1)], false).await.unwrap();
        captured.flush().await.unwrap();
        assert_eq!(
            captured
                .read_view()
                .unwrap()
                .search_cached(query())
                .unwrap()
                .data
                .len(),
            2
        );
        for path in [&captured_root, &workspace_root] {
            assert_eq!(
                MemoryService::at_path(path, options.clone())
                    .unwrap()
                    .source_documents()
                    .len(),
                1
            );
        }
        // Delete an acknowledged staged add before its scheduled publication.
        workspace.add(vec![request(2)], false).await.unwrap();
        let doc_id = owner
            .read()
            .unwrap()
            .memory_store(MemoryStore::Workspace)
            .source_documents()
            .iter()
            .find(|doc| doc.content.contains("quartz memory 2"))
            .unwrap()
            .doc_id
            .clone();
        workspace
            .mutate(move |service| Ok(json!(service.delete("alice", &doc_id)?)))
            .await
            .unwrap();
        assert_eq!(
            captured
                .read_view()
                .unwrap()
                .search_cached(query())
                .unwrap()
                .data
                .len(),
            2
        );
        let recovered = MemoryService::at_path(&workspace_root, options).unwrap();
        assert_eq!(recovered.source_documents().len(), 1);
        assert!(!recovered
            .source_documents()
            .iter()
            .any(|doc| doc.content.contains("quartz memory 2")));
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn global_visibility_barrier_waits_for_both_stores_while_acceptance_continues() {
        let options = PipelineOptions {
            key_phrase_enrichment: false,
            ..crate::memory_api::workspace::segmented_store_options()
        };
        let owner = Arc::new(RwLock::new(MemoryService::retain_workspace_store(
            MemoryService::in_memory(options.clone()),
            MemoryService::in_memory(options),
        )));
        let captured = StagedWriter::start_with_schedule(
            owner.clone(),
            WriteSchedule {
                refresh_interval: Duration::from_secs(60),
                batch_size: 10000,
                checkpoint_interval: Duration::from_secs(60),
            },
        )
        .unwrap();
        let workspace = captured.for_workspace();
        workspace.add(vec![request(1)], false).await.unwrap();
        let (release, gate) = mpsc::channel();
        owner
            .write()
            .unwrap()
            .memory_store_mut(MemoryStore::Workspace)
            .begin_checkpoint_publication_paused(gate)
            .unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            captured.add(vec![request(0)], false),
        )
        .await
        .unwrap()
        .unwrap();
        let retained = captured.read_view().unwrap();
        assert_eq!(retained.inspection().source_document_count, 0);
        let flush = tokio::spawn({
            let captured = captured.clone();
            async move { captured.flush().await }
        });
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(
            !flush.is_finished(),
            "global barrier ignored the paused workspace generation"
        );
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(10), flush)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            captured
                .read_view()
                .unwrap()
                .search_cached(query())
                .unwrap()
                .data
                .len(),
            2
        );
        assert_eq!(retained.inspection().source_document_count, 0);
        let owner = owner.read().unwrap();
        for target in [MemoryStore::Captured, MemoryStore::Workspace] {
            let (accepted, published, checkpointed) =
                owner.memory_store(target).publication_revisions();
            assert_eq!(accepted, published);
            assert_eq!(accepted, checkpointed);
        }
    }
}
