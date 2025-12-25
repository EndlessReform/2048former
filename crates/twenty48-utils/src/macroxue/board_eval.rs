//! Macroxue board heuristic replicated from `packages/train_2048/tokenization/macroxue/board_eval.py`.
//!
//! This module mirrors the logic of the original C++ implementation by
//! generating lookup tables for every possible row/column configuration and
//! exposing helpers to evaluate full boards.

use std::sync::OnceLock;

use anyhow::{Result, ensure};

const N: usize = 4;
const MAP_SIZE: usize = 1 << (5 * N);

static SCORE_MAPS: OnceLock<(Vec<i64>, Vec<i64>)> = OnceLock::new();

fn tile_score(rank: u32) -> i64 {
    (rank as i64) << rank
}

fn build_score_maps() -> (Vec<i64>, Vec<i64>) {
    let mut descending = vec![0i64; MAP_SIZE];
    let mut ascending = vec![0i64; MAP_SIZE];

    for idx in 0..MAP_SIZE {
        let mut line = [0u32; N];
        for j in 0..N {
            line[j] = ((idx >> (j * 5)) & 0x1f) as u32;
        }

        let mut score = tile_score(line[0]);
        for x in 0..(N - 1) {
            let a = tile_score(line[x]);
            let b = tile_score(line[x + 1]);
            if a >= b {
                score += a + b;
            } else {
                score += (a - b) * 12;
            }
            if a == b {
                score += a;
            }
        }

        let mut key_desc = 0usize;
        for &rank in &line {
            key_desc = key_desc * 32 + rank as usize;
        }

        descending[key_desc] = score;
        ascending[idx] = score;
    }

    (descending, ascending)
}

fn score_maps() -> &'static (Vec<i64>, Vec<i64>) {
    SCORE_MAPS.get_or_init(build_score_maps)
}

fn normalize_board(board: &[u8]) -> Result<[u8; N * N]> {
    ensure!(
        board.len() == N * N,
        "expected board encoded as 16 ranks; got {}",
        board.len()
    );
    let mut ranks = [0u8; N * N];
    ranks.copy_from_slice(board);
    Ok(ranks)
}

fn encode_rows(board: &[u8; N * N]) -> [usize; N] {
    let mut keys = [0usize; N];
    for y in 0..N {
        let mut key = 0usize;
        for x in 0..N {
            key = key * 32 + board[y * N + x] as usize;
        }
        keys[y] = key;
    }
    keys
}

fn encode_cols(board: &[u8; N * N]) -> [usize; N] {
    let mut keys = [0usize; N];
    for x in 0..N {
        let mut key = 0usize;
        for y in 0..N {
            key = key * 32 + board[y * N + x] as usize;
        }
        keys[x] = key;
    }
    keys
}

/// Evaluate a board using the Macroxue heuristic.
///
/// `board` must contain 16 ranks in row-major order (0 = empty, 1 = tile 2,
/// ...). Set `interactive` to true to mirror the behaviour of the engine's
/// interactive evaluation mode.
pub fn evaluate(board: &[u8], interactive: bool) -> Result<i64> {
    let ranks = normalize_board(board)?;
    let rows = encode_rows(&ranks);
    let cols = encode_cols(&ranks);

    let maps = score_maps();
    let desc = maps.0.as_slice();
    let asc = maps.1.as_slice();

    if interactive {
        let score_left: i64 = rows.iter().map(|&k| desc[k]).sum();
        let score_right: i64 = rows.iter().map(|&k| asc[k]).sum();
        let score_up: i64 = cols.iter().map(|&k| desc[k]).sum();
        let score_down: i64 = cols.iter().map(|&k| asc[k]).sum();
        Ok(score_left.max(score_right) + score_up.max(score_down))
    } else {
        let mut score = 0i64;
        score += rows.iter().map(|&k| desc[k]).sum::<i64>();
        score += cols.iter().map(|&k| desc[k]).sum::<i64>();
        Ok(score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_empty_board() {
        let board = [0u8; 16];
        assert_eq!(evaluate(&board, false).unwrap(), 0);
        assert_eq!(evaluate(&board, true).unwrap(), 0);
    }

    #[test]
    fn evaluate_monotonic_board_matches_reference() {
        let board = [
            15, 14, 13, 12, //
            11, 10, 9, 8, //
            7, 6, 5, 4, //
            3, 2, 1, 0,
        ];
        let non_interactive = evaluate(&board, false).unwrap();
        let interactive = evaluate(&board, true).unwrap();

        assert_eq!(non_interactive, 3_618_726);
        assert_eq!(interactive, 3_618_726);
    }

    #[test]
    fn interactive_mode_prefers_best_orientation() {
        let board = [
            0, 1, 2, 3, //
            4, 5, 6, 7, //
            8, 9, 10, 11, //
            12, 13, 14, 15,
        ];
        let non_interactive = evaluate(&board, false).unwrap();
        let interactive = evaluate(&board, true).unwrap();

        assert_eq!(non_interactive, -16_031_270);
        assert_eq!(interactive, 3_618_726);
    }

    #[test]
    fn mixed_board_matches_python_reference() {
        let board = [
            2, 2, 1, 1, //
            0, 0, 3, 3, //
            4, 4, 5, 5, //
            6, 6, 7, 7,
        ];
        let non_interactive = evaluate(&board, false).unwrap();
        let interactive = evaluate(&board, true).unwrap();

        assert_eq!(non_interactive, -33_140);
        assert_eq!(interactive, 13_076);
    }
}
