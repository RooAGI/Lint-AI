//! One bounded write queue, one mutable owner, and a detached published view.
use super::*;
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

type Reply = oneshot::Sender<anyhow::Result<Value>>;
enum Command {
    Add(Vec<AddRequest>, bool, Reply),
    Flush(Reply),
}

struct BarrierReply {
    target: u64,
    requests: Option<Vec<AddRequest>>,
    reply: Reply,
}

fn fail_barriers(barriers: &mut Vec<BarrierReply>, message: &str) {
    for barrier in barriers.drain(..) {
        let _ = barrier.reply.send(Err(anyhow::anyhow!("{message}")));
    }
}

fn run_writer(
    service: Arc<RwLock<MemoryService>>,
    published: Arc<RwLock<Arc<MemoryService>>>,
    receiver: mpsc::Receiver<Command>,
    schedule: WriteSchedule,
) {
    let mut first_pending: Option<Instant> = None;
    let mut last_checkpoint = Instant::now();
    let mut checkpoint_build = false;
    let mut retry_after: Option<Instant> = None;
    let mut barriers = Vec::<BarrierReply>::new();
    loop {
        let building = service
            .read()
            .map(|s| s.publication_in_flight())
            .unwrap_or(false);
        let timeout = if building || !barriers.is_empty() {
            Duration::from_millis(10)
        } else {
            first_pending
                .map(|first| schedule.refresh_interval.saturating_sub(first.elapsed()))
                .unwrap_or(schedule.refresh_interval)
        };
        match receiver.recv_timeout(timeout) {
            Ok(Command::Add(requests, wait, reply)) => {
                let result = (|| -> anyhow::Result<(Value, u64)> {
                    anyhow::ensure!(!wait || barriers.len() < QUEUE_CAPACITY, PendingWriteLimit);
                    let mut owner = service
                        .write()
                        .map_err(|_| anyhow::anyhow!("memory writer poisoned"))?;
                    if owner.pending_document_count()
                        + requests.iter().map(|r| r.messages.len()).sum::<usize>()
                        > MAX_PENDING_DOCUMENTS
                    {
                        return Err(PendingWriteLimit.into());
                    }
                    let responses = owner.stage_add_batch(requests.clone())?;
                    if owner.pending_add_count() > 0 && first_pending.is_none() {
                        first_pending = Some(Instant::now());
                    }
                    Ok((
                        serde_json::to_value(responses)?,
                        owner.publication_revisions().0,
                    ))
                })();
                match result {
                    Ok((_, target)) if wait => barriers.push(BarrierReply {
                        target,
                        requests: Some(requests),
                        reply,
                    }),
                    Ok((value, _)) => {
                        let _ = reply.send(Ok(value));
                    }
                    Err(error) => {
                        let _ = reply.send(Err(error));
                    }
                }
            }
            Ok(Command::Flush(reply)) => {
                if barriers.len() >= QUEUE_CAPACITY {
                    let _ = reply.send(Err(PendingWriteLimit.into()));
                } else {
                    match service.read() {
                        Ok(owner) => barriers.push(BarrierReply {
                            target: owner.publication_revisions().0,
                            requests: None,
                            reply,
                        }),
                        Err(_) => {
                            let _ = reply.send(Err(anyhow::anyhow!("memory writer poisoned")));
                        }
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                let result = flush(&service, &published, true);
                match result {
                    Ok(()) => {
                        if let Ok(owner) = service.read() {
                            for barrier in barriers.drain(..) {
                                let response = match barrier.requests {
                                    Some(requests) => owner
                                        .completed_add_responses(&requests)
                                        .and_then(|responses| Ok(serde_json::to_value(responses)?)),
                                    None => Ok(Value::Null),
                                };
                                let _ = barrier.reply.send(response);
                            }
                        }
                    }
                    Err(error) => {
                        eprintln!("memory writer shutdown checkpoint failed: {error:#}");
                        fail_barriers(&mut barriers, &format!("{error:#}"));
                    }
                }
                break;
            }
        }
        let completion = service
            .write()
            .map_err(|_| anyhow::anyhow!("memory writer poisoned"))
            .and_then(|mut owner| owner.poll_publication());
        match completion {
            Ok(Some(view)) => {
                if checkpoint_build {
                    last_checkpoint = Instant::now();
                }
                checkpoint_build = false;
                if let Err(error) = publish_view(&published, view) {
                    fail_barriers(&mut barriers, &format!("{error:#}"));
                }
                first_pending = if service.read().is_ok_and(|s| s.pending_add_count() > 0) {
                    first_pending.or_else(|| Some(Instant::now()))
                } else {
                    None
                };
            }
            Ok(None) => {}
            Err(error) => {
                eprintln!("memory publication failed; durable journal retained: {error:#}");
                fail_barriers(&mut barriers, &format!("{error:#}"));
                checkpoint_build = false;
                retry_after = Some(Instant::now() + schedule.refresh_interval);
                first_pending = Some(Instant::now());
            }
        }
        if let Ok(owner) = service.read() {
            let (_, revision, durable) = owner.publication_revisions();
            let visible = published
                .read()
                .map(|view| view.inspection().snapshot_revision)
                .unwrap_or(0);
            let mut index = 0;
            while index < barriers.len() {
                if barriers[index].target <= durable
                    && barriers[index].target <= revision
                    && barriers[index].target <= visible
                {
                    let barrier = barriers.remove(index);
                    let response = match barrier.requests {
                        Some(requests) => owner
                            .completed_add_responses(&requests)
                            .and_then(|responses| Ok(serde_json::to_value(responses)?)),
                        None => Ok(Value::Null),
                    };
                    let _ = barrier.reply.send(response);
                } else {
                    index += 1;
                }
            }
        }
        let (pending, enrichment_ready, building) = service
            .read()
            .map(|s| {
                (
                    s.pending_document_count(),
                    s.enrichment_ready(),
                    s.publication_in_flight(),
                )
            })
            .unwrap_or((MAX_PENDING_DOCUMENTS, false, false));
        let due = first_pending.is_some_and(|first| first.elapsed() >= schedule.refresh_interval);
        let checkpoint =
            !barriers.is_empty() || last_checkpoint.elapsed() >= schedule.checkpoint_interval;
        let can_retry = retry_after.is_none_or(|deadline| Instant::now() >= deadline);
        if !building
            && can_retry
            && ((pending > 0 && (due || pending >= schedule.batch_size))
                || checkpoint
                || enrichment_ready)
        {
            let result = service
                .write()
                .map_err(|_| anyhow::anyhow!("memory writer poisoned"))
                .and_then(|mut owner| {
                    if checkpoint {
                        owner.begin_checkpoint_publication()
                    } else {
                        owner.begin_publication()
                    }
                });
            match result {
                Ok(()) => {
                    first_pending = None;
                    checkpoint_build = checkpoint;
                    retry_after = None;
                }
                Err(error) => {
                    eprintln!(
                        "memory publication capture failed; durable journal retained: {error:#}"
                    );
                    fail_barriers(&mut barriers, &format!("{error:#}"));
                    retry_after = Some(Instant::now() + schedule.refresh_interval);
                    first_pending = Some(Instant::now());
                }
            }
        }
    }
}

#[derive(Clone)]
pub(super) struct StagedWriter {
    sender: SyncSender<Command>,
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
            .published_read_view();
        let published = Arc::new(RwLock::new(Arc::new(view)));
        let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
        let worker = Self {
            sender,
            published: published.clone(),
        };
        std::thread::Builder::new()
            .name("memory-writer".into())
            .spawn(move || run_writer(service, published, receiver, schedule))?;
        Ok(worker)
    }

    pub(super) fn read_view(&self) -> anyhow::Result<Arc<MemoryService>> {
        Ok(self
            .published
            .read()
            .map_err(|_| anyhow::anyhow!("published view poisoned"))?
            .clone())
    }

    pub(super) fn publish_owner(&self, owner: &MemoryService) -> anyhow::Result<()> {
        publish(&self.published, owner)
    }

    pub(super) async fn add(
        &self,
        requests: Vec<AddRequest>,
        wait: bool,
    ) -> Result<Value, StatusCode> {
        let (reply, receive) = oneshot::channel();
        self.sender
            .try_send(Command::Add(requests, wait, reply))
            .map_err(|error| match error {
                mpsc::TrySendError::Full(_) => StatusCode::TOO_MANY_REQUESTS,
                mpsc::TrySendError::Disconnected(_) => StatusCode::SERVICE_UNAVAILABLE,
            })?;
        receive
            .await
            .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?
            .map_err(|error| {
                if error.is::<PendingWriteLimit>() {
                    StatusCode::TOO_MANY_REQUESTS
                } else {
                    eprintln!("staged mutation failed: {error:#}");
                    StatusCode::INTERNAL_SERVER_ERROR
                }
            })
    }

    pub(super) async fn flush(&self) -> Result<(), StatusCode> {
        let (reply, receive) = oneshot::channel();
        self.sender
            .try_send(Command::Flush(reply))
            .map_err(|error| match error {
                mpsc::TrySendError::Full(_) => StatusCode::TOO_MANY_REQUESTS,
                mpsc::TrySendError::Disconnected(_) => StatusCode::SERVICE_UNAVAILABLE,
            })?;
        receive
            .await
            .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?
            .map_err(|error| {
                if error.is::<PendingWriteLimit>() {
                    StatusCode::TOO_MANY_REQUESTS
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                }
            })?;
        Ok(())
    }
}

fn publish(published: &RwLock<Arc<MemoryService>>, owner: &MemoryService) -> anyhow::Result<()> {
    let revision = owner.inspection().snapshot_revision;
    if published
        .read()
        .map_err(|_| anyhow::anyhow!("published view poisoned"))?
        .inspection()
        .snapshot_revision
        == revision
    {
        return Ok(());
    }
    let next = Arc::new(owner.published_read_view());
    // Building the view happens before acquiring the publication lock.
    *published
        .write()
        .map_err(|_| anyhow::anyhow!("published view poisoned"))? = next;
    Ok(())
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

fn flush(
    service: &RwLock<MemoryService>,
    published: &RwLock<Arc<MemoryService>>,
    checkpoint: bool,
) -> anyhow::Result<()> {
    let mut owner = service
        .write()
        .map_err(|_| anyhow::anyhow!("memory writer poisoned"))?;
    if checkpoint {
        owner.flush()?;
    } else {
        owner.publish_staged()?;
    }
    publish(published, &owner)
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
        let old = owner.read().unwrap().published_read_view();
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
}
