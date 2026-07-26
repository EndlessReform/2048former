use crate::config;
use crate::ds_writer::StepRow as DsStepRow;
use crate::feeder::{FeederHandle, InferenceOutput};
use rand::{Rng, SeedableRng};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use tokio::sync::mpsc as tokio_mpsc;
use tokio_util::sync::CancellationToken;
use twenty48_utils::engine::Move;
use twenty48_utils::engine::merge_reward_exps;

pub mod strategies;

/// Per-game actor that drives a single board to completion by
/// querying the model via the Feeder and applying selected moves.
pub struct GameActor {
    pub game_id: u32,
    pub handle: FeederHandle,
    pub board: PackedBoard,
    pub seed: u64,
    rng: rand::rngs::StdRng,
    pub sampling: config::SamplingStrategy,
    pub head_order: config::HeadOrder,
    pub board_map: config::BoardMapping,
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

#[derive(Clone, Copy, Debug, Default)]
pub struct PackedBoard {
    packed: u64,
    tile_65536_mask: u16,
}

impl PackedBoard {
    pub fn empty() -> Self {
        Self {
            packed: 0,
            tile_65536_mask: 0,
        }
    }

    pub fn from_exps(exps: &[u8; 16]) -> Self {
        let mut packed = 0u64;
        let mut mask = 0u16;
        for (i, &exp) in exps.iter().enumerate() {
            let mut nib = exp;
            if nib >= 16 {
                mask |= 1 << i;
                nib = 15;
            }
            let shift = (15 - i) * 4;
            packed |= (nib as u64 & 0xF) << shift;
        }
        Self {
            packed,
            tile_65536_mask: mask,
        }
    }

    pub fn to_exps(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        for idx in 0..16 {
            let shift = (15 - idx) * 4;
            let nib = ((self.packed >> shift) & 0xF) as u8;
            out[idx] = if (self.tile_65536_mask >> idx) & 1 == 1 {
                16
            } else {
                nib
            };
        }
        out
    }
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
    ) -> Self {
        // Initialize a fresh board with two random tiles
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut exps = [0u8; 16];
        insert_random_tile(&mut exps, &mut rng);
        insert_random_tile(&mut exps, &mut rng);
        let board = PackedBoard::from_exps(&exps);
        Self {
            game_id,
            handle,
            board,
            seed,
            rng,
            sampling,
            head_order,
            board_map,
            step_tx,
            cancel,
            step_budget,
        }
    }

    /// Run the actor loop to completion and return the result.
    pub async fn run(mut self) -> GameResult {
        let mut steps: u64 = 0;
        let mut seq: u64 = 0;
        let mut score: u64 = 0;
        while !is_game_over(self.board) {
            if self.cancel.is_cancelled() {
                break;
            }
            if let Some(b) = &self.step_budget {
                if !b.try_take() {
                    break;
                }
            }
            let id = ((self.game_id as u64) << 32) | seq;
            let board_bytes = board_to_exponents(self.board, self.board_map.clone());
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
            // Compute legal mask in the configured head order and select move
            let order = self.head_order.clone();
            let legal = legal_mask(self.board, order.clone());
            let mv = match inference {
                InferenceOutput::Bins(bins) => {
                    // Gate non-argmax sampling by steps: before start_gate or at/after stop_gate -> argmax
                    let start_gate = self.sampling.start_gate_or_default();
                    let stop_gate = self.sampling.stop_gate();
                    let outside_window =
                        (steps < start_gate) || (stop_gate.map(|s| steps >= s).unwrap_or(false));
                    if matches!(self.sampling.kind, config::SamplingStrategyKind::Argmax) {
                        strategies::select_move(&bins, &legal, &self.sampling, &mut self.rng, order)
                    } else if outside_window {
                        strategies::select_move_max_p1(&bins, &legal, order)
                    } else {
                        strategies::select_move(&bins, &legal, &self.sampling, &mut self.rng, order)
                    }
                }
                InferenceOutput::Argmax { head, .. } => {
                    // Map head index according to configured order
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
                let before = self.board.to_exps();
                let after = shift_exps(&before, m);
                if after != before {
                    score = score.saturating_add(merge_reward_exps(&before, m));
                    let mut next = after;
                    insert_random_tile(&mut next, &mut self.rng);
                    self.board = PackedBoard::from_exps(&next);
                    steps += 1;
                } else {
                    break;
                }
            } else {
                break;
            }
            seq += 1;
        }

        GameResult {
            game_id: self.game_id,
            seed: self.seed,
            steps,
            score,
            highest_tile: highest_tile(self.board),
        }
    }
}

fn board_to_exponents(b: PackedBoard, _map: config::BoardMapping) -> [u8; 16] {
    b.to_exps()
}

fn legal_mask(board: PackedBoard, _order: config::HeadOrder) -> [bool; 4] {
    // Produce mask in UDLR order
    let dirs = [Move::Up, Move::Down, Move::Left, Move::Right];
    let mut mask = [false; 4];
    for (i, &m) in dirs.iter().enumerate() {
        let before = board.to_exps();
        let after = shift_exps(&before, m);
        mask[i] = after != before;
    }
    mask
}

fn is_game_over(board: PackedBoard) -> bool {
    let exps = board.to_exps();
    for dir in [Move::Up, Move::Down, Move::Left, Move::Right] {
        if shift_exps(&exps, dir) != exps {
            return false;
        }
    }
    true
}

fn highest_tile(board: PackedBoard) -> u32 {
    let max_exp = board.to_exps().iter().copied().max().unwrap_or(0);
    if max_exp == 0 {
        return 0;
    }
    let val = if max_exp < 64 {
        1u64 << max_exp
    } else {
        u64::MAX
    };
    val.min(u32::MAX as u64) as u32
}

fn insert_random_tile<R: Rng + ?Sized>(board: &mut [u8; 16], rng: &mut R) {
    let empty = board.iter().filter(|&&v| v == 0).count();
    if empty == 0 {
        return;
    }
    let mut index = rng.gen_range(0..empty);
    for cell in board.iter_mut() {
        if *cell != 0 {
            continue;
        }
        if index == 0 {
            *cell = if rng.gen_range(0..10) < 9 { 1 } else { 2 };
            return;
        }
        index -= 1;
    }
}

fn shift_exps(board: &[u8; 16], direction: Move) -> [u8; 16] {
    let mut out = [0u8; 16];
    match direction {
        Move::Left | Move::Right => {
            for row in 0..4 {
                let base = row * 4;
                let line = [
                    board[base],
                    board[base + 1],
                    board[base + 2],
                    board[base + 3],
                ];
                let shifted = shift_line(line, direction);
                out[base..base + 4].copy_from_slice(&shifted);
            }
        }
        Move::Up | Move::Down => {
            for col in 0..4 {
                let line = [board[col], board[col + 4], board[col + 8], board[col + 12]];
                let shifted = shift_line(line, direction);
                out[col] = shifted[0];
                out[col + 4] = shifted[1];
                out[col + 8] = shifted[2];
                out[col + 12] = shifted[3];
            }
        }
    }
    out
}

fn shift_line(line: [u8; 4], direction: Move) -> [u8; 4] {
    let mut work = line;
    if matches!(direction, Move::Right | Move::Down) {
        work.reverse();
    }
    let shifted = shift_line_left(work);
    if matches!(direction, Move::Right | Move::Down) {
        let mut rev = shifted;
        rev.reverse();
        rev
    } else {
        shifted
    }
}

fn shift_line_left(line: [u8; 4]) -> [u8; 4] {
    let mut tiles: Vec<u8> = line.into_iter().filter(|&v| v != 0).collect();
    let mut idx = 0usize;
    while idx + 1 < tiles.len() {
        if tiles[idx] == tiles[idx + 1] {
            tiles[idx] = tiles[idx].saturating_add(1);
            tiles.remove(idx + 1);
        }
        idx += 1;
    }
    let mut out = [0u8; 4];
    for (i, v) in tiles.into_iter().enumerate() {
        out[i] = v;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_board_tracks_65536_mask() {
        let mut exps = [0u8; 16];
        exps[5] = 16;
        let packed = PackedBoard::from_exps(&exps);
        assert_eq!(packed.tile_65536_mask, 1 << 5);
        let round = packed.to_exps();
        assert_eq!(round[5], 16);
    }

    #[test]
    fn shift_exps_merges_to_65536() {
        let mut exps = [0u8; 16];
        exps[0] = 15;
        exps[1] = 15;
        let shifted = shift_exps(&exps, Move::Left);
        assert_eq!(shifted[0], 16);
        assert_eq!(merge_reward_exps(&exps, Move::Left), 1u64 << 16);
    }

    #[test]
    fn seeded_tile_sequence_is_reproducible() {
        fn sequence(seed: u64) -> Vec<[u8; 16]> {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut board = [0u8; 16];
            let mut states = Vec::new();
            for _ in 0..8 {
                insert_random_tile(&mut board, &mut rng);
                states.push(board);
            }
            states
        }

        assert_eq!(sequence(42), sequence(42));
        assert_ne!(sequence(42), sequence(43));
    }
}
