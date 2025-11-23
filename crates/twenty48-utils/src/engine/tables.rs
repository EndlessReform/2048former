use std::sync::OnceLock;

use super::state::Score;

/// Precomputed lookup tables for all possible 4-tile lines (16-bit packed).
///
/// Why: shifting/merging a row or column depends only on its 4 nibbles.
/// There are 2^16 possible 16-bit values. We precompute the result of
/// shifting left/right/up/down and the score contribution for each line.
/// This keeps moves branch-light and fast at runtime.
///
/// Layout:
/// - `shift_left/right/up/down[i]`: replacement 16-bit line after applying the move.
/// - `score[i]`: sum of tile merge scores for the line.
///
/// Access is via `stores()` which lazily initializes a single global `Stores`
/// on first use. The public `engine::new()` simply forces init early.
pub(crate) struct Stores {
    pub(crate) shift_left: Box<[u64]>,
    pub(crate) shift_right: Box<[u64]>,
    pub(crate) shift_up: Box<[u64]>,
    pub(crate) shift_down: Box<[u64]>,
    pub(crate) score: Box<[Score]>,
}

const LINE_TABLE_SIZE: usize = 0x1_0000; // 65,536 possible 16-bit lines

static STORES: OnceLock<Stores> = OnceLock::new();

/// Ensure lookup tables are initialized.
pub fn init() {
    let _ = STORES.get_or_init(create_stores);
}

#[inline(always)]
pub(crate) fn stores() -> &'static Stores {
    STORES
        .get()
        .expect("Engine stores not initialized; call engine::new() first")
}

fn create_stores() -> Stores {
    // Allocate on the heap to keep stack frames small during init.
    let mut shift_left = vec![0u64; LINE_TABLE_SIZE];
    let mut shift_right = vec![0u64; LINE_TABLE_SIZE];
    let mut shift_up = vec![0u64; LINE_TABLE_SIZE];
    let mut shift_down = vec![0u64; LINE_TABLE_SIZE];
    let mut score = vec![0u64; LINE_TABLE_SIZE];

    let mut val: usize = 0;
    while val < LINE_TABLE_SIZE {
        let line = val as u64;
        shift_left[val] = super::ops::shift_line(line, super::state::Move::Left);
        shift_right[val] = super::ops::shift_line(line, super::state::Move::Right);
        shift_up[val] = super::ops::shift_line(line, super::state::Move::Up);
        shift_down[val] = super::ops::shift_line(line, super::state::Move::Down);
        score[val] = super::ops::calc_score(line);
        val += 1;
    }

    Stores {
        shift_left: shift_left.into_boxed_slice(),
        shift_right: shift_right.into_boxed_slice(),
        shift_up: shift_up.into_boxed_slice(),
        shift_down: shift_down.into_boxed_slice(),
        score: score.into_boxed_slice(),
    }
}

#[inline(always)]
pub(crate) fn get_line_entry(table: &[u64], idx: u16) -> u64 {
    debug_assert!((idx as usize) < LINE_TABLE_SIZE);
    unsafe { *table.get_unchecked(idx as usize) }
}

#[inline(always)]
pub(crate) fn get_score_entry(idx: u16) -> Score {
    debug_assert!((idx as usize) < LINE_TABLE_SIZE);
    let score_table = &stores().score;
    unsafe { *score_table.get_unchecked(idx as usize) }
}
