//! Macroxue heuristics and tokenization utilities shared across crates.

pub mod board_eval;
pub mod tokenizer;

pub use board_eval::evaluate as evaluate_board;
