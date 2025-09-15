use crate::config;
use crate::grpc::{self, pb};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc as tokio_mpsc;
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tonic::Status;

/// Single inference item sent to the micro-batcher.
/// - `id` is a stable, caller-supplied identifier used for routing
/// - `game_id` is used for fairness/token accounting (not enforced yet)
/// - `board` is a 4x4 grid encoded as 16 exponents (0=empty, 1=2, 2=4, ...)
#[derive(Debug, Clone)]
pub struct InferenceItem {
    pub id: u64,
    pub game_id: u32,
    pub board: [u8; 16],
}

/// Per-item output: 4 heads, each with probabilities over bins.
/// Per-item output: 4 heads × n_bins probabilities (float32).
/// The outer dimension is heads in the fixed order [Up, Down, Left, Right].
pub type Bins = Vec<Vec<f32>>; // heads x n_bins

/// Micro-batching feeder. Builds batches from a bounded queue, submits RPCs,
/// and routes responses to per-item oneshots. Backpressure is enforced via a
/// sliding window of in-flight items.
pub struct Feeder {
    batch_cfg: config::Batch,
    req_rx: mpsc::Receiver<InferenceItem>,
    completion_rx: mpsc::Receiver<usize>,
    completion_tx: mpsc::Sender<usize>,
    inflight_items: usize,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Bins, Status>>>>>,
    metrics: Option<Arc<Metrics>>, // client-side batch metrics
    emb_tx: Option<tokio_mpsc::Sender<EmbeddingRow>>, // optional inline embeddings
    cancel: Option<CancellationToken>,
}

impl Feeder {
    /// Create a feeder and its caller-facing handle.
    ///
    /// Usage: let (feeder, handle) = Feeder::new(cfg.orchestrator.batch.clone());
    ///        let _task = feeder.spawn(client);
    pub fn new(batch_cfg: config::Batch) -> (Self, FeederHandle) {
        let (req_tx, req_rx) = mpsc::channel(batch_cfg.queue_cap);
        let (completion_tx, completion_rx) = mpsc::channel::<usize>(batch_cfg.inflight_batches * 2);
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let metrics = batch_cfg
            .metrics_file
            .as_ref()
            .map(|_| Arc::new(Metrics::new()));
        let feeder = Self {
            batch_cfg,
            req_rx,
            completion_rx,
            completion_tx,
            inflight_items: 0,
            pending: pending.clone(),
            metrics: metrics.clone(),
            emb_tx: None,
            cancel: None,
        };
        (
            feeder,
            FeederHandle {
                tx: req_tx,
                pending,
            },
        )
    }

    /// Sender used internally to return credits when a batch completes.
    pub fn completion_sender(&self) -> mpsc::Sender<usize> {
        self.completion_tx.clone()
    }

    /// Start the feeder loop on a Tokio task. Non-blocking; returns a JoinHandle.
    ///
    /// The caller should hold onto the returned handle (or detach) for lifecycle
    /// management. Items are submitted via `FeederHandle::submit`.
    pub fn spawn(mut self, client: grpc::Client) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            // Start metrics reporter if configured
            if let (Some(m), Some(path)) = (&self.metrics, self.batch_cfg.metrics_file.as_ref()) {
                let path = path.clone();
                let interval = self.batch_cfg.metrics_interval_s;
                let m = m.clone();
                tokio::spawn(async move { m.run_reporter(path, interval).await });
            }

            let target = self.batch_cfg.target_batch;
            let max_items = self.batch_cfg.inflight_batches * self.batch_cfg.max_batch;
            let flush = Duration::from_micros(self.batch_cfg.flush_us);

            loop {
                // Build a batch up to `target`, waiting up to `flush` from the first item
                let mut buf: Vec<InferenceItem> = Vec::with_capacity(target);

                // Block for the first item (or exit if channel closed)
                if let Some(tok) = &self.cancel {
                    // Allow cancel to short-circuit waiting for the first item
                    tokio::select! {
                        biased;
                        _ = tok.cancelled() => { return; }
                        maybe = self.req_rx.recv() => {
                            match maybe { Some(item) => buf.push(item), None => return }
                        }
                    }
                } else {
                    match self.req_rx.recv().await {
                        Some(item) => buf.push(item),
                        None => return,
                    }
                }

                let deadline = Instant::now() + flush;
                while buf.len() < target {
                    let now = Instant::now();
                    if now >= deadline {
                        break;
                    }
                    let remaining = deadline.saturating_duration_since(now);
                    if let Some(tok) = &self.cancel {
                        tokio::select! {
                            biased;
                            _ = tok.cancelled() => break, // stop batching immediately
                            r = tokio::time::timeout(remaining, self.req_rx.recv()) => {
                                match r { Ok(Some(item)) => buf.push(item), Ok(None) => return, Err(_elapsed) => break }
                            }
                        }
                    } else {
                        match tokio::time::timeout(remaining, self.req_rx.recv()).await {
                            Ok(Some(item)) => buf.push(item),
                            Ok(None) => return,
                            Err(_elapsed) => break, // flush window expired
                        }
                    }
                }

                while self.inflight_items + buf.len() > max_items {
                    if let Some(done) = self.completion_rx.recv().await {
                        self.inflight_items = self.inflight_items.saturating_sub(done);
                    }
                }

                let batch_len = buf.len();
                self.inflight_items += batch_len;

                let ids: Vec<u64> = buf.iter().map(|it| it.id).collect();
                let items_pb: Vec<pb::Item> = buf
                    .into_iter()
                    .map(|it| pb::Item {
                        id: it.id,
                        board: it.board.to_vec(),
                    })
                    .collect();

                let req = pb::InferRequest {
                    model_id: String::new(),
                    items: items_pb,
                    batch_id: 0,
                    return_embedding: self.emb_tx.is_some(),
                };
                let mut client_clone = client.clone();
                let completion = self.completion_tx.clone();
                let pending = self.pending.clone();
                let metrics = self.metrics.clone();
                let emb_tx = self.emb_tx.clone();

                tokio::spawn(async move {
                    let res = client_clone.infer(req).await;
                    match res {
                        Ok(resp) => {
                            let resp = resp.into_inner();
                            let embed_dim = resp.embed_dim as usize;
                            // Route per item
                            let mut map = pending.lock().await;
                            // If embeddings are requested, prefer concatenated buffer when present
                            let mut emitted_from_concat = false;
                            if let (Some(ch), true) =
                                (emb_tx.as_ref(), !resp.embeddings_concat.is_empty())
                            {
                                if embed_dim > 0
                                    && resp.embed_dtype
                                        == pb::infer_response::EmbedDType::Fp32 as i32
                                {
                                    // Safe cast: u8 bytes to f32 words
                                    let floats: &[f32] =
                                        bytemuck::try_cast_slice(&resp.embeddings_concat)
                                            .unwrap_or(&[]);
                                    let n_items = resp.item_ids.len();
                                    if floats.len() == n_items * embed_dim {
                                        for (i, item_id) in resp.item_ids.iter().enumerate() {
                                            let start = i * embed_dim;
                                            let end = start + embed_dim;
                                            let v: Vec<f32> = floats[start..end].to_vec();
                                            let _ = ch.try_send(EmbeddingRow {
                                                id: *item_id,
                                                dim: embed_dim,
                                                values: v,
                                            });
                                        }
                                        emitted_from_concat = true;
                                    }
                                }
                            }

                            for (i, item_id) in resp.item_ids.iter().enumerate() {
                                if let Some(tx) = map.remove(item_id) {
                                    let bins: Result<Bins, Status> = resp
                                        .outputs
                                        .get(i)
                                        .map(|o| o.heads.iter().map(|h| h.probs.clone()).collect())
                                        .ok_or_else(|| Status::internal("missing output for item"));
                                    let _ = tx.send(bins);
                                }
                                // If not emitted via concatenated buffer, fall back to per-item field
                                if !emitted_from_concat {
                                    if let (Some(ch), Some(out)) =
                                        (emb_tx.as_ref(), resp.outputs.get(i))
                                    {
                                        if !out.embedding.is_empty()
                                            && embed_dim > 0
                                            && resp.embed_dtype
                                                == pb::infer_response::EmbedDType::Fp32 as i32
                                        {
                                            if out.embedding.len() == embed_dim * 4 {
                                                let v: Vec<f32> =
                                                    bytemuck::cast_slice(&out.embedding).to_vec();
                                                let _ = ch.try_send(EmbeddingRow {
                                                    id: *item_id,
                                                    dim: embed_dim,
                                                    values: v,
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(status) => {
                            let mut map = pending.lock().await;
                            for id in ids {
                                if let Some(tx) = map.remove(&id) {
                                    let _ = tx.send(Err(status.clone()));
                                }
                            }
                        }
                    }
                    let _ = completion.send(batch_len).await;
                    if let Some(m) = metrics.as_ref() {
                        m.record_complete(batch_len).await;
                    }
                });
            }
        })
    }
}

/// Lightweight, cloneable handle to submit items and await their results.
#[derive(Clone)]
pub struct FeederHandle {
    tx: mpsc::Sender<InferenceItem>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Bins, Status>>>>>,
}

impl FeederHandle {
    /// Submit an inference item and receive a oneshot for its per-head probabilities.
    /// The returned Receiver yields `Bins` on success or a gRPC `Status` on failure.
    pub async fn submit(
        &self,
        id: u64,
        game_id: u32,
        board: [u8; 16],
    ) -> oneshot::Receiver<Result<Bins, Status>> {
        let (tx_once, rx) = oneshot::channel();
        {
            let mut map = self.pending.lock().await;
            map.insert(id, tx_once);
        }
        let _ = self.tx.send(InferenceItem { id, game_id, board }).await;
        rx
    }
}

impl Feeder {
    /// Enable inline embeddings emission by providing a channel to receive them.
    pub fn set_embeddings_channel(&mut self, ch: tokio_mpsc::Sender<EmbeddingRow>) {
        self.emb_tx = Some(ch);
    }

    /// Install a cancellation token to request graceful shutdown.
    pub fn set_cancel_token(&mut self, token: CancellationToken) {
        self.cancel = Some(token);
    }
}

/// Inline embedding record for a single item
#[derive(Clone, Debug)]
pub struct EmbeddingRow {
    pub id: u64,
    pub dim: usize,
    pub values: Vec<f32>,
}

// ---------------- Metrics -----------------

#[derive(Default)]
struct MetricsInner {
    counts: HashMap<usize, u64>,
    total_items: u64,
    total_batches: u64,
}

#[derive(Clone)]
struct Metrics {
    inner: Arc<Mutex<MetricsInner>>,
}

impl Metrics {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(MetricsInner::default())),
        }
    }

    async fn record_complete(&self, batch_len: usize) {
        let mut g = self.inner.lock().await;
        *g.counts.entry(batch_len).or_insert(0) += 1;
        g.total_items += batch_len as u64;
        g.total_batches += 1;
    }

    async fn snapshot(&self) -> MetricsInner {
        let g = self.inner.lock().await;
        MetricsInner {
            counts: g.counts.clone(),
            total_items: g.total_items,
            total_batches: g.total_batches,
        }
    }

    async fn run_reporter(&self, path: std::path::PathBuf, interval_s: f64) {
        let mut file = match tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
        {
            Ok(f) => f,
            Err(_) => return,
        };
        let mut prev_items = 0u64;
        let mut prev_batches = 0u64;
        let mut t_prev = Instant::now();
        loop {
            tokio::time::sleep(Duration::from_secs_f64(interval_s)).await;
            let snap = self.snapshot().await;
            let dt = t_prev.elapsed().as_secs_f64();
            t_prev = Instant::now();
            let delta_items = snap.total_items.saturating_sub(prev_items);
            let delta_batches = snap.total_batches.saturating_sub(prev_batches);
            prev_items = snap.total_items;
            prev_batches = snap.total_batches;

            // top 10 by batch frequency
            let mut top: Vec<(usize, u64)> = snap.counts.iter().map(|(k, v)| (*k, *v)).collect();
            top.sort_by(|a, b| b.1.cmp(&a.1));
            if top.len() > 10 {
                top.truncate(10);
            }
            let covered_batches: u64 = top.iter().map(|(_, c)| *c).sum();
            let missed_batches = snap.total_batches.saturating_sub(covered_batches);

            // top 10 by items contributed (size * count)
            let mut top_items: Vec<(usize, u64)> = snap
                .counts
                .iter()
                .map(|(bs, c)| (*bs, (*bs as u64) * (*c)))
                .collect();
            top_items.sort_by(|a, b| b.1.cmp(&a.1));
            if top_items.len() > 10 {
                top_items.truncate(10);
            }
            let covered_items: u64 = top_items.iter().map(|(_, c)| *c).sum();
            let missed_items = snap.total_items.saturating_sub(covered_items);

            let ts = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                Ok(d) => d.as_secs_f64(),
                Err(_) => 0.0,
            };

            #[derive(serde::Serialize)]
            struct Rec<'a> {
                ts: f64,
                items_per_s: f64,
                batches_per_s: f64,
                total_items: u64,
                total_batches: u64,
                top_batch_sizes: &'a [(usize, u64)],
                missed_batches: u64,
                top_item_sizes: &'a [(usize, u64)],
                missed_items: u64,
            }

            let rec = Rec {
                ts,
                items_per_s: (delta_items as f64) / dt.max(1e-9),
                batches_per_s: (delta_batches as f64) / dt.max(1e-9),
                total_items: snap.total_items,
                total_batches: snap.total_batches,
                top_batch_sizes: &top,
                missed_batches,
                top_item_sizes: &top_items,
                missed_items,
            };
            let line = match serde_json::to_string(&rec) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let _ = file.write_all(line.as_bytes()).await;
            let _ = file.write_all(b"\n").await;
            let _ = file.flush().await;
        }
    }
}
