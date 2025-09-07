use crate::config;
use crate::grpc::{self, pb};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, Mutex};
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
        let feeder = Self {
            batch_cfg,
            req_rx,
            completion_rx,
            completion_tx,
            inflight_items: 0,
            pending: pending.clone(),
        };
        (feeder, FeederHandle { tx: req_tx, pending })
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
            let target = self.batch_cfg.target_batch;
            let max_items = self.batch_cfg.inflight_batches * self.batch_cfg.max_batch;
            let flush = Duration::from_micros(self.batch_cfg.flush_us);

            loop {
                let mut buf: Vec<InferenceItem> = Vec::with_capacity(target);
                while buf.len() < target {
                    match self.req_rx.try_recv() {
                        Ok(item) => buf.push(item),
                        Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                        Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => return,
                    }
                }

                if buf.is_empty() {
                    tokio::time::sleep(flush).await;
                    continue;
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
                };
                let mut client_clone = client.clone();
                let completion = self.completion_tx.clone();
                let pending = self.pending.clone();

                tokio::spawn(async move {
                    let res = client_clone.infer(req).await;
                    match res {
                        Ok(resp) => {
                            let resp = resp.into_inner();
                            // Route per item
                            let mut map = pending.lock().await;
                            for (i, item_id) in resp.item_ids.iter().enumerate() {
                                if let Some(tx) = map.remove(item_id) {
                                    let bins: Result<Bins, Status> = resp
                                        .outputs
                                        .get(i)
                                        .map(|o| o.heads.iter().map(|h| h.probs.clone()).collect())
                                        .ok_or_else(|| Status::internal("missing output for item"));
                                    let _ = tx.send(bins);
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
