use crate::feeder::{Bins, FeederHandle};
use ai_2048::engine as GameEngine;
use ai_2048::engine::{Board, Move};
use rand::SeedableRng;
use tokio::task::JoinHandle;

/// Per-game actor that drives a single board to completion by
/// querying the model via the Feeder and applying selected moves.
pub struct GameActor {
    pub game_id: u32,
    pub handle: FeederHandle,
    pub board: Board,
}

pub struct GameResult {
    pub game_id: u32,
    pub steps: u64,
    pub score: u64,
    pub highest_tile: u32,
}

impl GameActor {
    pub fn new(game_id: u32, handle: FeederHandle, seed: u64) -> Self {
        // Initialize a fresh board with two random tiles
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut board: Board = Board::EMPTY;
        board = board.with_random_tile(&mut rng);
        board = board.with_random_tile(&mut rng);
        Self {
            game_id,
            handle,
            board,
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
            // Compute legal mask and select move according to model semantics (max-p1)
            let legal = legal_mask(self.board);
            let mv = select_move_max_p1(&bins, &legal);
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
            steps,
            score: self.board.score(),
            highest_tile: self.board.highest_tile() as u32,
        }
    }
}

fn board_to_exponents(b: Board) -> [u8; 16] {
    // ai_2048 encodes 16 exponents in a u64 (4 bits per cell), row-major
    let raw = b.raw();
    let mut out = [0u8; 16];
    for i in 0..16 {
        out[i] = ((raw >> (i * 4)) & 0xF) as u8;
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
