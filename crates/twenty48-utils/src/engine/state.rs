use rand::Rng;
use std::fmt;

use super::ops;
use serde::{Deserialize, Serialize};

// Internal type aliases for packed representation
pub(crate) type BoardRaw = u64;
pub(crate) type Line = u64;
pub(crate) type Tile = u64;
pub(crate) type Score = u64;

/// A direction to move/merge tiles.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Move {
    Up,
    Down,
    Left,
    Right,
}

/// Packed 4x4 2048 board as 16 4-bit nibbles in a `u64`.
///
/// Public methods provide ergonomic, safe operations while preserving
/// an escape hatch to the raw packed representation for advanced use.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Board(pub(crate) BoardRaw);

impl Board {
    /// A constant empty board (all zeros).
    pub const EMPTY: Board = Board(0);

    /// Construct a `Board` from its raw packed representation.
    #[inline]
    pub fn from_raw(raw: BoardRaw) -> Self {
        Board(raw)
    }

    /// Consume this `Board`, returning the raw packed `u64`.
    #[inline]
    pub fn into_raw(self) -> BoardRaw {
        self.0
    }

    /// Borrow the raw packed `u64` for this `Board`.
    #[inline]
    pub fn raw(&self) -> BoardRaw {
        self.0
    }

    /// Return the board resulting from sliding/merging tiles in `dir` (no random insert).
    ///
    /// Example
    /// ```
    /// use twenty48_utils::engine::{self as GameEngine, Board, Move};
    /// GameEngine::new();
    /// let b = Board::EMPTY;
    /// let _ = b.shift(Move::Left);
    /// ```
    #[inline]
    pub fn shift(self, dir: Move) -> Self {
        ops::shift(self, dir)
    }

    /// Insert a random 2 (90%) or 4 (10%) tile into a random empty slot, using the provided RNG.
    ///
    /// Deterministic example using a seeded RNG:
    /// ```
    /// use twenty48_utils::engine::{self as GameEngine, Board};
    /// use rand::{SeedableRng, rngs::StdRng};
    /// GameEngine::new();
    /// let mut rng = StdRng::seed_from_u64(123);
    /// let b = Board::EMPTY.with_random_tile(&mut rng).with_random_tile(&mut rng);
    /// assert!(b.count_empty() <= 14);
    /// ```
    #[inline]
    pub fn with_random_tile<R: Rng + ?Sized>(self, rng: &mut R) -> Self {
        let mut index = rng.gen_range(0..ops::count_empty(self));
        let mut tmp = self.0;
        let mut tile = ops::generate_random_tile(rng);
        loop {
            while (tmp & 0xf) != 0 {
                tmp >>= 4;
                tile <<= 4;
            }
            if index == 0 {
                break;
            }
            index -= 1;
            tmp >>= 4;
            tile <<= 4;
        }
        Board(self.0 | tile)
    }

    /// Convenience: like `with_random_tile` but uses thread-local RNG.
    ///
    /// ```
    /// use twenty48_utils::engine::{self as GameEngine, Board};
    /// GameEngine::new();
    /// let b = Board::EMPTY.with_random_tile_thread();
    /// assert!(b.count_empty() <= 15);
    /// ```
    #[inline]
    pub fn with_random_tile_thread(self) -> Self {
        let mut rng = rand::thread_rng();
        self.with_random_tile(&mut rng)
    }

    /// Perform a move then insert a random tile if the move changed the board, using the provided RNG.
    ///
    /// ```
    /// use twenty48_utils::engine::{self as GameEngine, Board, Move};
    /// use rand::{SeedableRng, rngs::StdRng};
    /// GameEngine::new();
    /// let mut rng = StdRng::seed_from_u64(1);
    /// let b0 = Board::EMPTY.with_random_tile(&mut rng).with_random_tile(&mut rng);
    /// let _b1 = b0.make_move(Move::Up, &mut rng);
    /// ```
    #[inline]
    pub fn make_move<R: Rng + ?Sized>(self, direction: Move, rng: &mut R) -> Self {
        let moved = self.shift(direction);
        if moved != self {
            moved.with_random_tile(rng)
        } else {
            self
        }
    }

    /// Compute the total score for this board.
    ///
    /// ```
    /// use twenty48_utils::engine::{self as GameEngine, Board};
    /// GameEngine::new();
    /// let b = Board::EMPTY;
    /// let _ = b.score();
    /// ```
    #[inline]
    pub fn score(self) -> super::state::Score {
        ops::get_score(self)
    }

    /// Return true if no legal moves remain.
    ///
    /// ```
    /// use twenty48_utils::engine::{self as GameEngine, Board};
    /// GameEngine::new();
    /// // On an empty board, shifting in any direction doesn't change the board,
    /// // so `is_game_over` returns true (no merges/moves possible without a new tile).
    /// assert!(Board::EMPTY.is_game_over());
    /// ```
    #[inline]
    pub fn is_game_over(self) -> bool {
        ops::is_game_over(self)
    }

    /// Return the highest tile value (e.g., 2048) present on the board.
    #[inline]
    pub fn highest_tile(self) -> super::state::Tile {
        ops::get_highest_tile_val(self)
    }

    /// Count the number of empty cells on the board.
    #[inline]
    pub fn count_empty(self) -> u64 {
        ops::count_empty(self)
    }

    /// Get the actual value at index (2^exponent stored at nibble).
    ///
    /// Index runs 0..16 row-major.
    #[inline]
    pub fn tile_value(self, idx: usize) -> u16 {
        ops::get_tile_val(self, idx)
    }

    /// Iterate over tile exponents (nibbles) in row-major order.
    /// Returns 0 for empty, 1 for 2, 2 for 4, etc.
    #[inline]
    pub fn tiles(self) -> TilesIter {
        TilesIter {
            raw: self.0,
            idx: 0,
        }
    }

    /// Convenience: collect tile exponents into a `Vec<u8>`.
    #[inline]
    pub fn to_vec(self) -> Vec<u8> {
        self.tiles().collect()
    }
}

impl fmt::Debug for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Board({:#018x})", self.0)
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let board: Vec<_> = self.tiles().map(|n| super::ops::format_val(&n)).collect();
        write!(
            f,
            "\n{}|{}|{}|{}\n--------------------------------\n{}|{}|{}|{}\n--------------------------------\n{}|{}|{}|{}\n--------------------------------\n{}|{}|{}|{}\n",
            board[0],
            board[1],
            board[2],
            board[3],
            board[4],
            board[5],
            board[6],
            board[7],
            board[8],
            board[9],
            board[10],
            board[11],
            board[12],
            board[13],
            board[14],
            board[15]
        )
    }
}

impl From<BoardRaw> for Board {
    fn from(v: BoardRaw) -> Self {
        Board::from_raw(v)
    }
}
impl From<Board> for BoardRaw {
    fn from(b: Board) -> Self {
        b.into_raw()
    }
}

/// Iterator over board tiles (exponents) in row-major order.
pub struct TilesIter {
    pub raw: BoardRaw,
    pub idx: usize,
}

impl Iterator for TilesIter {
    type Item = u8;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= 16 {
            return None;
        }
        let n = ((self.raw >> (60 - (4 * self.idx))) & 0xf) as u8;
        self.idx += 1;
        Some(n)
    }
}

impl IntoIterator for Board {
    type Item = u8;
    type IntoIter = TilesIter;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.tiles()
    }
}

impl IntoIterator for &Board {
    type Item = u8;
    type IntoIter = TilesIter;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.tiles()
    }
}

/// Convert packed boards to exponent rows into the provided output buffer.
///
/// - `out` must be length = `boards.len() * 16` and will be filled with
///   16 exponents per board (row-major).
/// - If `parallel` is true, uses Rayon to parallelize across cores.
///
/// Example
/// ```
/// use twenty48_utils::engine::{self as GameEngine, state::boards_to_exponents_into};
/// GameEngine::new();
/// let boards: [u64; 2] = [0x1000_0000_0000_0000, 0x2100_0000_0000_0000];
/// let mut out = vec![0u8; boards.len() * 16];
/// boards_to_exponents_into(&mut out, &boards, false);
/// assert_eq!(out.len(), 32);
/// ```
pub fn boards_to_exponents_into(out: &mut [u8], boards: &[u64], parallel: bool) {
    assert_eq!(
        out.len(),
        boards.len() * 16,
        "out buffer must be N*16 bytes"
    );
    if parallel {
        use rayon::prelude::*;
        out.par_chunks_mut(16)
            .zip(boards.par_iter().copied())
            .for_each(|(dst, b)| {
                let exps = Board::from_raw(b).to_vec();
                dst.copy_from_slice(&exps[..16]);
            });
    } else {
        for (i, &b) in boards.iter().enumerate() {
            let exps = Board::from_raw(b).to_vec();
            let dst = &mut out[i * 16..(i + 1) * 16];
            dst.copy_from_slice(&exps[..16]);
        }
    }
}
