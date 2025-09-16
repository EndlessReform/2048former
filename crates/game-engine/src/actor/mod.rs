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
    pub step_tx: Option<tokio_mpsc::Sender<DsStepRow>>,
    pub cancel: CancellationToken,
    pub step_budget: Option<StepBudget>,
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
        step_tx: Option<tokio_mpsc::Sender<DsStepRow>>,
        cancel: CancellationToken,
        step_budget: Option<StepBudget>,
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
            step_tx,
            cancel,
            step_budget,
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
            let board_bytes = board_to_exponents(self.board);
            // Record step row (pre-move state) for dataset
            if let Some(tx) = &self.step_tx {
                let _ = tx.try_send(DsStepRow {
                    run_id: self.game_id as u64,
                    step_idx: steps as u32,
                    exps: board_bytes,
                });
            }
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
            // Compute legal mask and select move according to configured sampling strategy
            let legal = legal_mask(self.board);
            let mv = match inference {
                InferenceOutput::Bins(bins) => {
                    // Gate non-argmax sampling by steps: before start_gate or at/after stop_gate -> argmax
                    let start_gate = self.sampling.start_gate_or_default();
                    let stop_gate = self.sampling.stop_gate();
                    let outside_window =
                        (steps < start_gate) || (stop_gate.map(|s| steps >= s).unwrap_or(false));
                    if matches!(self.sampling.kind, config::SamplingStrategyKind::Argmax) {
                        strategies::select_move(&bins, &legal, &self.sampling, &mut rng)
                    } else if outside_window {
                        strategies::select_move_max_p1(&bins, &legal)
                    } else {
                        strategies::select_move(&bins, &legal, &self.sampling, &mut rng)
                    }
                }
                InferenceOutput::Argmax { head, .. } => {
                    let dirs = [Move::Up, Move::Down, Move::Left, Move::Right];
                    let idx = head as usize;
                    if let Some(&choice) = dirs.get(idx) {
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
                    }
                }
            };

            if let Some(m) = mv {
                // Apply move and spawn new tile using the Board API
                self.board = self.board.make_move(m, &mut rng);
                steps += 1;
            } else {
                break;
            }
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

fn board_to_exponents(b: Board) -> [u8; 16] {
    // ai_2048 packs 16 nibbles in a u64. To match the Python `to_exponents()`
    // row-major ordering expected by the model, map MSB-first to (row, col).
    // That is, nibble 15 -> index 0 (top-left), ..., nibble 0 -> index 15 (bottom-right).
    let raw = b.raw();
    let mut out = [0u8; 16];
    for idx in 0..16 {
        let nib = 15 - idx; // MSB-first
        out[idx] = ((raw >> (nib * 4)) & 0xF) as u8;
    }
    out
}

fn legal_mask(board: Board) -> [bool; 4] {
    let dirs = [Move::Up, Move::Down, Move::Left, Move::Right];
    let mut mask = [false; 4];
    for (i, &m) in dirs.iter().enumerate() {
        let after = GameEngine::make_move(board, m);
        mask[i] = after.raw() != board.raw();
    }
    mask
}
