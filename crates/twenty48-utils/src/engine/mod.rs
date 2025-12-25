//! Engine module: compact 2048 board, fast shift/score ops, and
//! precomputed lookup tables. Public API stays small and ergonomic.
//!
//! - `Board` is the packed 4x4 state with useful methods.
//! - Free functions mirror the methods when convenient (e.g., `shift`).
//! - Internals (tables and hot ops) live in submodules to keep things tidy.

mod ops;
pub mod state;
mod tables;

pub use state::{Board, Move};

pub use ops::{
    count_empty, get_highest_tile_val, get_score, get_tile_val, insert_random_tile, is_game_over,
    line_to_vec, make_move, shift,
};

/// Initialize internal precomputed tables on first use.
/// Safe to call multiple times.
pub fn new() {
    tables::init();
}
