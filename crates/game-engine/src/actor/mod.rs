use crate::config;
use crate::ds_writer::StepRow as DsStepRow;
use crate::feeder::{FeederHandle, InferenceOutput};
use ai_2048::engine as GameEngine;
use ai_2048::engine::{Board, Move};
use rand::SeedableRng;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use tokio::sync::mpsc as tokio_mpsc;
use tokio_util::sync::CancellationToken;

pub mod strategies;

/// Per-game actor that drives a single board to completion by
/// querying the model via the Feeder and applying selected moves.
pub struct GameActor {
    pub game_id: u32,
    pub handle: FeederHandle,
    pub board: Board,
    pub seed: u64,
    pub sampling: config::SamplingStrategy,
    pub head_order: config::HeadOrder,
    pub board_map: config::BoardMapping,
    pub step_tx: Option<tokio_mpsc::Sender<DsStepRow>>,
    pub cancel: CancellationToken,
    pub step_budget: Option<StepBudget>,
    pub record_logprobs: bool,
}

#[derive(Clone)]
pub struct StepBudget {
    max: u64,
    used: Arc<AtomicU64>,
}

impl StepBudget {
    pub fn new(max: u64) -> Self {
        Self {
            max,
            used: Arc::new(AtomicU64::new(0)),
        }
    }
    /// Try to consume exactly 1 step budget. Returns false if exhausted.
    pub fn try_take(&self) -> bool {
        let mut cur = self.used.load(Ordering::Relaxed);
        loop {
            if cur >= self.max {
                return false;
            }
            match self.used.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(next) => cur = next,
            }
        }
    }
    /// Read the number of steps consumed so far.
    #[allow(dead_code)]
    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }
}

pub struct GameResult {
    pub game_id: u32,
    pub seed: u64,
    pub steps: u64,
    pub score: u64,
    pub highest_tile: u32,
}

impl GameActor {
    pub fn new(
        game_id: u32,
        handle: FeederHandle,
        seed: u64,
        sampling: config::SamplingStrategy,
        head_order: config::HeadOrder,
        board_map: config::BoardMapping,
        step_tx: Option<tokio_mpsc::Sender<DsStepRow>>,
        cancel: CancellationToken,
        step_budget: Option<StepBudget>,
        record_logprobs: bool,
    ) -> Self {
        // Initialize a fresh board with two random tiles
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut board: Board = Board::EMPTY;
        board = board.with_random_tile(&mut rng);
        board = board.with_random_tile(&mut rng);
        Self {
            game_id,
            handle,
            board,
            seed,
            sampling,
            head_order,
            board_map,
            step_tx,
            cancel,
            step_budget,
            record_logprobs,
        }
    }

    /// Run the actor loop to completion and return the result.
    pub async fn run(mut self) -> GameResult {
        let mut steps: u64 = 0;
        let mut seq: u64 = 0;
        let mut rng = rand::rngs::StdRng::from_entropy();

        while !self.board.is_game_over() {
            if self.cancel.is_cancelled() {
                break;
            }
            if let Some(b) = &self.step_budget {
                if !b.try_take() {
                    break;
                }
            }
            let id = ((self.game_id as u64) << 32) | seq;
            let step_idx = steps as u32;
            let board_bytes = board_to_exponents(self.board, self.board_map.clone());
            let rx = self.handle.submit(id, self.game_id, board_bytes).await;
            let inference = tokio::select! {
                biased;
                _ = self.cancel.cancelled() => { break; }
                res = rx => {
                    match res {
                        Ok(Ok(out)) => out,
                        Ok(Err(_status)) => { break; },
                        Err(_canceled) => { break; },
                    }
                }
            };
            // Compute legal mask in the configured head order and select move
            let order = self.head_order.clone();
            let legal = legal_mask(self.board, order.clone());
            let selection = match inference {
                InferenceOutput::Bins(bins) => {
                    // Gate non-argmax sampling by steps: before start_gate or at/after stop_gate -> argmax
                    let start_gate = self.sampling.start_gate_or_default();
                    let stop_gate = self.sampling.stop_gate();
                    let outside_window =
                        (steps < start_gate) || (stop_gate.map(|s| steps >= s).unwrap_or(false));
                    if matches!(self.sampling.kind, config::SamplingStrategyKind::Argmax) || outside_window {
                        strategies::select_move_max_p1(
                            &bins,
                            &legal,
                            order.clone(),
                            self.record_logprobs,
                        )
                    } else {
                        strategies::select_move_with_details(
                            &bins,
                            &legal,
                            &self.sampling,
                            &mut rng,
                            order.clone(),
                            self.record_logprobs,
                        )
                    }
                }
                InferenceOutput::Argmax { head, .. } => {
                    // Map head index according to configured order
                    let dirs = [Move::Up, Move::Down, Move::Left, Move::Right];
                    let idx = head as usize;
                    let mv = if let Some(&choice) = dirs.get(idx) {
                        if legal.get(idx).copied().unwrap_or(false) {
                            Some(choice)
                        } else {
                            dirs.iter()
                                .enumerate()
                                .find(|(i, _)| legal.get(*i).copied().unwrap_or(false))
                                .map(|(_, &m)| m)
                        }
                    } else {
                        None
                    };
                    let log_probs = if self.record_logprobs {
                        let mut arr = [f32::NEG_INFINITY; 4];
                        if let Some(choice) = mv {
                            let idx = move_to_idx(choice);
                            if idx < 4 {
                                arr[idx] = 0.0;
                            }
                        }
                        Some(arr)
                    } else {
                        None
                    };
                    strategies::Selection { mv, log_probs }
                }
            };

            let Some(m) = selection.mv else {
                break;
            };

            let move_idx = move_to_idx(m);
            let logp = if self.record_logprobs {
                selection.log_probs.unwrap_or_else(|| {
                    let mut arr = [f32::NEG_INFINITY; 4];
                    if move_idx < 4 {
                        arr[move_idx] = 0.0;
                    }
                    arr
                })
            } else {
                [f32::NAN; 4]
            };

            if let Some(tx) = &self.step_tx {
                let _ = tx.try_send(DsStepRow {
                    run_id: self.game_id as u64,
                    step_idx,
                    exps: board_bytes,
                    move_dir: move_idx as u8,
                    logp,
                });
            }

            // Apply move and spawn new tile using the Board API
            self.board = self.board.make_move(m, &mut rng);
            steps += 1;
            seq += 1;
        }

        GameResult {
            game_id: self.game_id,
            seed: self.seed,
            steps,
            score: self.board.score(),
            highest_tile: self.board.highest_tile() as u32,
        }
    }
}

fn board_to_exponents(b: Board, _map: config::BoardMapping) -> [u8; 16] {
    // Canonical MSB-first mapping: cell i reads nibble (15 - i)
    let raw = b.raw();
    let mut out = [0u8; 16];
    for idx in 0..16 {
        let nib = 15 - idx;
        out[idx] = ((raw >> (nib * 4)) & 0xF) as u8;
    }
    out
}

fn legal_mask(board: Board, _order: config::HeadOrder) -> [bool; 4] {
    // Produce mask in UDLR order
    let dirs = [Move::Up, Move::Down, Move::Left, Move::Right];
    let mut mask = [false; 4];
    for (i, &m) in dirs.iter().enumerate() {
        let after = GameEngine::make_move(board, m);
        mask[i] = after.raw() != board.raw();
    }
    mask
}

fn move_to_idx(m: Move) -> usize {
    match m {
        Move::Up => 0,
        Move::Down => 1,
        Move::Left => 2,
        Move::Right => 3,
    }
}
