use rand::Rng;

use super::state::{Board, BoardRaw, Line, Move, Score, Tile};
use super::tables::{get_line_entry, get_score_entry, stores};

/// Perform a move then insert a random tile if the move changed the board (uses thread RNG).
pub fn make_move(board: Board, direction: Move) -> Board {
    let mut rng = rand::thread_rng();
    board.make_move(direction, &mut rng)
}

/// Slide/merge tiles in the given direction. No randomness.
pub fn shift(board: Board, direction: Move) -> Board {
    match direction {
        Move::Left | Move::Right => shift_rows(board, direction),
        Move::Up | Move::Down => shift_cols(board, direction),
    }
}

// Credit to Nneonneo
pub(crate) fn transpose(x: BoardRaw) -> BoardRaw {
    let a1 = x & 0xF0F00F0FF0F00F0F;
    let a2 = x & 0x0000F0F00000F0F0;
    let a3 = x & 0x0F0F00000F0F0000;
    let a = a1 | (a2 << 12) | (a3 >> 12);
    let b1 = a & 0xFF00FF0000FF00FF;
    let b2 = a & 0x00FF00FF00000000;
    let b3 = a & 0x00000000FF00FF00;
    b1 | (b2 >> 24) | (b3 << 24)
}

pub(crate) fn extract_line(board: BoardRaw, line_idx: u64) -> Line {
    (board >> ((3 - line_idx) * 16)) & 0xffff
}

/// Return the cell's actual value (0 if empty), e.g., 2, 4, 8, ...
pub fn get_tile_val(board: Board, idx: usize) -> u16 {
    let raw_val = (board.0 >> (60 - (4 * idx))) & 0xf;
    if raw_val == 0 {
        0
    } else {
        2_u16.pow(raw_val as u32)
    }
}

pub fn line_to_vec(line: Line) -> Vec<Tile> {
    (0..4).fold(Vec::new(), |mut tiles, tile_idx| {
        tiles.push(line >> ((3 - tile_idx) * 4) & 0xf);
        tiles
    })
}

/// True if no move in any direction changes the board.
pub fn is_game_over(board: Board) -> bool {
    for direction in [Move::Up, Move::Down, Move::Left, Move::Right] {
        let new_board = shift(board, direction);
        if new_board != board {
            return false;
        }
    }
    true
}

// https://stackoverflow.com/questions/38225571/count-number-of-zero-nibbles-in-an-unsigned-64-bit-integer
/// Count the number of zero tiles.
pub fn count_empty(board: Board) -> u64 {
    16 - count_non_empty(board)
}

/// Insert a random 2 (90%) or 4 (10%) tile using thread-local RNG.
///
/// For reproducible behavior, prefer `Board::with_random_tile(&mut impl Rng)`.
pub fn insert_random_tile(board: Board) -> Board {
    board.with_random_tile_thread()
}

pub(crate) fn generate_random_tile<R: Rng + ?Sized>(rng: &mut R) -> Tile {
    if rng.gen_range(0..10) < 9 { 1 } else { 2 }
}

pub fn get_score(board: Board) -> Score {
    (0..4).fold(0, |acc, idx| {
        let row_val = extract_line(board.0, idx) as u16;
        acc + get_score_entry(row_val)
    })
}

fn shift_rows(board: Board, move_dir: Move) -> Board {
    let s = stores();
    let table: &[u64] = match move_dir {
        Move::Left => &s.shift_left,
        Move::Right => &s.shift_right,
        _ => panic!("Trying to move up or down in shift rows"),
    };
    let res = (0..4).fold(0, |new_board, row_idx| {
        let row_val = extract_line(board.0, row_idx) as u16;
        let new_row_val = get_line_entry(table, row_val);
        new_board | (new_row_val << (48 - (16 * row_idx)))
    });
    Board(res)
}

fn shift_cols(board: Board, move_dir: Move) -> Board {
    let transpose_board = transpose(board.0);
    let s = stores();
    let table: &[u64] = match move_dir {
        Move::Up => &s.shift_up,
        Move::Down => &s.shift_down,
        _ => panic!("Trying to move left or right in shift cols"),
    };
    let res = (0..4).fold(0, |new_board, col_idx| {
        let col_val = extract_line(transpose_board, col_idx) as u16;
        let new_col_val = get_line_entry(table, col_val);
        new_board | (new_col_val << (12 - (4 * col_idx)))
    });
    Board(res)
}

pub(crate) fn shift_line(line: Line, direction: Move) -> Line {
    let tiles = line_to_vec(line);
    match direction {
        Move::Left | Move::Right => vec_to_row(shift_vec(tiles, direction)),
        Move::Up | Move::Down => vec_to_col(shift_vec(tiles, direction)),
    }
}

fn vec_to_row(tiles: Vec<Tile>) -> Line {
    tiles[0] << 12 | tiles[1] << 8 | tiles[2] << 4 | tiles[3]
}

fn vec_to_col(tiles: Vec<Tile>) -> Line {
    tiles[0] << 48 | tiles[1] << 32 | tiles[2] << 16 | tiles[3]
}

fn shift_vec(vec: Vec<Tile>, direction: Move) -> Vec<Tile> {
    match direction {
        Move::Left | Move::Up => shift_vec_left(vec),
        Move::Right | Move::Down => shift_vec_right(vec),
    }
}

fn shift_vec_right(vec: Vec<Tile>) -> Vec<Tile> {
    let rev_vec: Vec<Tile> = vec.into_iter().rev().collect();
    shift_vec_left(rev_vec).iter().rev().copied().collect()
}

fn shift_vec_left(mut vec: Vec<Tile>) -> Vec<Tile> {
    for i in 0..4 {
        calculate_left_shift(&mut vec[i..]);
    }
    vec
}

fn calculate_left_shift(slice: &mut [Tile]) {
    let mut acc = 0;
    for s in slice.iter_mut() {
        let val = *s;
        if acc != 0 && acc == val {
            *s = 0;
            acc += 1;
            break;
        } else if acc != 0 && val != 0 && acc != val {
            break;
        } else if acc == 0 && val != 0 {
            *s = 0;
            acc = val;
        }
        // continue scan otherwise
    }
    slice[0] = acc;
}

// Credit to Nneonneo
pub(crate) fn calc_score(line: Line) -> Score {
    let mut score = 0;
    let tiles = line_to_vec(line);
    for &tile_val in tiles.iter().take(4) {
        if tile_val >= 2 {
            // the score is the total sum of the tile and all intermediate merged tiles
            score += (tile_val - 1) * (1 << tile_val);
        }
    }
    score
}

fn count_non_empty(board: Board) -> u64 {
    let mut board_copy = board.0;
    board_copy |= board_copy >> 1;
    board_copy |= board_copy >> 2;
    board_copy &= 0x1111111111111111;
    board_copy.count_ones() as u64
}

pub(crate) fn format_val(val: &u8) -> String {
    match val {
        0 => String::from("       "),
        &x => {
            let mut x = (2_i32.pow(x as u32)).to_string();
            while x.len() < 7 {
                match x.len() {
                    6 => x = format!(" {}", x),
                    _ => x = format!(" {} ", x),
                }
            }
            x
        }
    }
}

pub fn get_highest_tile_val(board: Board) -> Tile {
    let max_tile = (0..16)
        .map(|idx| get_tile(board, idx))
        .max()
        .expect("Could not extract max tile");
    2_u64.pow(max_tile as u32)
}

fn get_tile(board: Board, idx: usize) -> Tile {
    (board.0 >> (60 - (4 * idx))) & 0xf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_shift_vec_left() {
        assert_eq!(shift_vec_left(vec![0, 0, 0, 0]), vec![0, 0, 0, 0]);
        assert_eq!(shift_vec_left(vec![1, 2, 1, 2]), vec![1, 2, 1, 2]);
        assert_eq!(shift_vec_left(vec![1, 1, 2, 2]), vec![2, 3, 0, 0]);
        assert_eq!(shift_vec_left(vec![1, 0, 0, 1]), vec![2, 0, 0, 0]);
    }

    #[test]
    fn it_shift_vec_right() {
        assert_eq!(shift_vec_right(vec![0, 0, 0, 0]), vec![0, 0, 0, 0]);
        assert_eq!(shift_vec_right(vec![1, 2, 1, 2]), vec![1, 2, 1, 2]);
        assert_eq!(shift_vec_right(vec![1, 1, 2, 2]), vec![0, 0, 2, 3]);
        assert_eq!(shift_vec_right(vec![5, 0, 0, 5]), vec![0, 0, 0, 6]);
        assert_eq!(shift_vec_right(vec![0, 2, 2, 2]), vec![0, 0, 2, 3]);
    }

    #[test]
    fn it_test_insert_random_tile() {
        let mut game = Board::EMPTY;
        for _ in 0..16 {
            game = insert_random_tile(game);
        }
        assert_eq!(count_empty(game), 0);
    }

    #[test]
    fn test_shift_left() {
        crate::engine::new();
        assert_eq!(
            shift(Board::from_raw(0x0000), Move::Left),
            Board::from_raw(0x0000)
        );
        assert_eq!(
            shift(Board::from_raw(0x0002), Move::Left),
            Board::from_raw(0x2000)
        );
        assert_eq!(
            shift(Board::from_raw(0x2020), Move::Left),
            Board::from_raw(0x3000)
        );
        assert_eq!(
            shift(Board::from_raw(0x1332), Move::Left),
            Board::from_raw(0x1420)
        );
        assert_eq!(
            shift(Board::from_raw(0x1234), Move::Left),
            Board::from_raw(0x1234)
        );
        assert_eq!(
            shift(Board::from_raw(0x1002), Move::Left),
            Board::from_raw(0x1200)
        );
        assert_ne!(
            shift(Board::from_raw(0x1210), Move::Left),
            Board::from_raw(0x2200)
        );
    }

    #[test]
    fn test_shift_right() {
        crate::engine::new();
        assert_eq!(
            shift(Board::from_raw(0x0000), Move::Right),
            Board::from_raw(0x0000)
        );
        assert_eq!(
            shift(Board::from_raw(0x2000), Move::Right),
            Board::from_raw(0x0002)
        );
        assert_eq!(
            shift(Board::from_raw(0x2020), Move::Right),
            Board::from_raw(0x0003)
        );
        assert_eq!(
            shift(Board::from_raw(0x1332), Move::Right),
            Board::from_raw(0x0142)
        );
        assert_eq!(
            shift(Board::from_raw(0x1234), Move::Right),
            Board::from_raw(0x1234)
        );
        assert_eq!(
            shift(Board::from_raw(0x1002), Move::Right),
            Board::from_raw(0x0012)
        );
        assert_ne!(
            shift(Board::from_raw(0x0121), Move::Right),
            Board::from_raw(0x0022)
        );
    }

    #[test]
    fn test_move_left() {
        crate::engine::new();
        let game = Board::from_raw(0x1234133220021002);
        let game = shift(game, Move::Left);
        assert_eq!(game, Board::from_raw(0x1234142030001200));
    }

    #[test]
    fn test_move_up() {
        crate::engine::new();
        let game = Board::from_raw(0x1121230033004222);
        let game = shift(game, Move::Up);
        assert_eq!(game, Board::from_raw(0x1131240232004000));
    }

    #[test]
    fn test_move_right() {
        crate::engine::new();
        let game = Board::from_raw(0x1234133220021002);
        let game = shift(game, Move::Right);
        assert_eq!(game, Board::from_raw(0x1234014200030012));
    }

    #[test]
    fn test_move_down() {
        crate::engine::new();
        let game = Board::from_raw(0x1121230033004222);
        let game = shift(game, Move::Down);
        assert_eq!(game, Board::from_raw(0x1000210034014232));
    }

    #[test]
    fn it_count_empty() {
        let game = Board::from_raw(0x1111000011110000);
        assert_eq!(count_empty(game), 8);
        assert_eq!(game, Board::from_raw(0x1111000011110000));
        let game = Board::from_raw(0x1100000000000000);
        assert_eq!(count_empty(game), 14);
        assert_eq!(game, Board::from_raw(0x1100000000000000));
    }

    #[test]
    fn it_count_non_empty() {
        let game = Board::from_raw(0x1134000000000000);
        assert_eq!(count_non_empty(game), 4);
    }

    #[test]
    fn it_get_tile_val() {
        let game = Board::from_raw(0x0123456789abcdef);
        assert_eq!(get_tile_val(game, 3), 8);
        assert_eq!(get_tile_val(game, 10), 1024);
        assert_eq!(get_tile_val(game, 15), 32768);

        // Test empty tiles return 0
        let empty_board = Board::from_raw(0x0000000000000000);
        assert_eq!(get_tile_val(empty_board, 0), 0);
        assert_eq!(get_tile_val(empty_board, 8), 0);
        assert_eq!(get_tile_val(empty_board, 15), 0);
    }
}
