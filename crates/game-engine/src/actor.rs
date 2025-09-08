use crate::feeder::{Bins, FeederHandle};
use crate::config;
use ai_2048::engine as GameEngine;
use ai_2048::engine::{Board, Move};
use rand::SeedableRng;
use rand::distributions::Distribution;
use tokio::task::JoinHandle;

/// Per-game actor that drives a single board to completion by
/// querying the model via the Feeder and applying selected moves.
pub struct GameActor {
    pub game_id: u32,
    pub handle: FeederHandle,
    pub board: Board,
    pub seed: u64,
    pub sampling: config::SamplingStrategy,
}

pub struct GameResult {
    pub game_id: u32,
    pub seed: u64,
    pub steps: u64,
    pub score: u64,
    pub highest_tile: u32,
}

impl GameActor {
    pub fn new(game_id: u32, handle: FeederHandle, seed: u64, sampling: config::SamplingStrategy) -> Self {
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
        }
    }

    /// Run the actor loop to completion and return the result.
    pub async fn run(mut self) -> GameResult {
        let mut steps: u64 = 0;
        let mut seq: u64 = 0;
        let mut rng = rand::rngs::StdRng::from_entropy();

        while !self.board.is_game_over() {
            let id = ((self.game_id as u64) << 32) | seq;
            let board_bytes = board_to_exponents(self.board);
            let rx = self.handle.submit(id, self.game_id, board_bytes).await;
            let bins = match rx.await {
                Ok(Ok(b)) => b,
                Ok(Err(_status)) => {
                    // On RPC failure, end the game; outer orchestrator may decide to retry/restart.
                    break;
                }
                Err(_canceled) => {
                    break;
                }
            };
            // Compute legal mask and select move according to configured sampling strategy
            let legal = legal_mask(self.board);
            let mv = select_move(&bins, &legal, &self.sampling, &mut rng);
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

fn select_move_max_p1(bins: &Bins, legal: &[bool; 4]) -> Option<Move> {
    if bins.len() != 4 {
        return None;
    }
    let n_bins = bins[0].len();
    if n_bins == 0 {
        return None;
    }
    let one_idx = n_bins - 1; // default: dedicated '1' bin at the end
    let mut best_i: Option<usize> = None;
    let mut best_v: f32 = f32::NEG_INFINITY;
    for (i, head) in bins.iter().enumerate() {
        if !legal[i] {
            continue;
        }
        let p1 = *head.get(one_idx).unwrap_or(&0.0);
        if p1 > best_v {
            best_v = p1;
            best_i = Some(i);
        }
    }
    // If none legal (should imply game over), fallback to global argmax
    let idx = best_i.or_else(|| {
        let mut bi: usize = 0;
        let mut bv = f32::NEG_INFINITY;
        for (i, head) in bins.iter().enumerate() {
            let p1 = *head.get(one_idx).unwrap_or(&0.0);
            if p1 > bv {
                bv = p1;
                bi = i;
            }
        }
        Some(bi)
    })?;
    Some(match idx {
        0 => Move::Up,
        1 => Move::Down,
        2 => Move::Left,
        _ => Move::Right,
    })
}

fn select_move_softmax(bins: &Bins, legal: &[bool; 4], temperature: f32, rng: &mut rand::rngs::StdRng) -> Option<Move> {
    if bins.len() != 4 { return None; }
    let n_bins = bins[0].len();
    if n_bins == 0 { return None; }
    let one_idx = n_bins - 1; // dedicated '1' bin at the end

    // Collect p1 for legal moves only
    let mut p1 = [0.0f32; 4];
    let mut any_legal = false;
    for (i, head) in bins.iter().enumerate() {
        if !legal[i] { continue; }
        any_legal = true;
        p1[i] = *head.get(one_idx).unwrap_or(&0.0);
    }

    if !any_legal { return select_move_max_p1(bins, legal); }

    let t = if temperature.is_finite() && temperature > 0.0 { temperature } else { 1.0 };

    // Treat provided values as probabilities and apply temperature via log-probs:
    // weights ∝ exp((ln p) / t) == p^(1/t). Use max ln for stability.
    let mut max_ln = f32::NEG_INFINITY;
    for &p in &p1 {
        if p > 0.0 && p.is_finite() { let ln = p.ln(); if ln > max_ln { max_ln = ln; } }
    }
    let mut weights = [0.0f64; 4];
    for (i, &p) in p1.iter().enumerate() {
        if !legal[i] || !(p > 0.0) { continue; }
        let ln = p.ln();
        let z = ((ln - max_ln) / t) as f64;
        weights[i] = z.exp();
    }

    // If all weights are zero (numeric underflow), fallback to argmax
    if weights.iter().all(|&w| w == 0.0) {
        return select_move_max_p1(bins, legal);
    }

    // Sample index according to weights
    let dist = match rand::distributions::WeightedIndex::new(&weights) {
        Ok(d) => d,
        Err(_) => return select_move_max_p1(bins, legal),
    };
    let idx = dist.sample(rng);
    Some(match idx { 0 => Move::Up, 1 => Move::Down, 2 => Move::Left, _ => Move::Right })
}

fn select_move(bins: &Bins, legal: &[bool; 4], sampling: &config::SamplingStrategy, rng: &mut rand::rngs::StdRng) -> Option<Move> {
    match sampling.kind {
        config::SamplingStrategyKind::Argmax => select_move_max_p1(bins, legal),
        config::SamplingStrategyKind::Softmax => {
            let t = sampling.temperature_or_default() as f32;
            select_move_softmax(bins, legal, t, rng)
        }
    }
}
