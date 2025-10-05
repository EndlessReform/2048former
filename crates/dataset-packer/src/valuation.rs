use std::sync::OnceLock;

const N: usize = 4;
const MAP_SIZE: usize = 1 << (5 * N);

static SCORE_MAPS: OnceLock<(Vec<i64>, Vec<i64>)> = OnceLock::new();

fn tile_score(rank: i32) -> i64 {
    (rank as i64) << rank
}

fn build_score_maps() -> (Vec<i64>, Vec<i64>) {
    let mut score_map_descending = vec![0i64; MAP_SIZE];
    let mut score_map_ascending = vec![0i64; MAP_SIZE];
    for i in 0..MAP_SIZE {
        let mut line = [0i32; N];
        for j in 0..N {
            line[j] = ((i >> (j * 5)) & 0x1F) as i32;
        }
        let mut score: i64 = tile_score(line[0]);
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
        let mut key_desc = 0i32;
        for &r in &line {
            key_desc = key_desc * 32 + r;
        }
        score_map_descending[key_desc as usize] = score;
        score_map_ascending[i] = score;
    }
    (score_map_descending, score_map_ascending)
}

fn encode_row(matrix: &[[i32; N]], y: usize) -> i32 {
    let mut key = 0i32;
    for x in 0..N {
        key = key * 32 + matrix[y][x];
    }
    key
}

fn encode_col(matrix: &[[i32; N]], x: usize) -> i32 {
    let mut key = 0i32;
    for y in 0..N {
        key = key * 32 + matrix[y][x];
    }
    key
}

fn score_maps() -> &'static (Vec<i64>, Vec<i64>) {
    SCORE_MAPS.get_or_init(build_score_maps)
}

pub fn evaluate(board: &[i32], interactive: bool) -> i64 {
    assert_eq!(board.len(), N * N);
    let mut matrix = [[0i32; N]; N];
    for y in 0..N {
        for x in 0..N {
            matrix[y][x] = board[y * N + x];
        }
    }
    let (score_map_descending, score_map_ascending) = score_maps();
    let row_keys: Vec<i32> = (0..N).map(|y| encode_row(&matrix, y)).collect();
    let col_keys: Vec<i32> = (0..N).map(|x| encode_col(&matrix, x)).collect();
    if interactive {
        let score_left = row_keys
            .iter()
            .map(|&k| score_map_descending[k as usize])
            .sum::<i64>();
        let score_right = row_keys
            .iter()
            .map(|&k| score_map_ascending[k as usize])
            .sum::<i64>();
        let score_up = col_keys
            .iter()
            .map(|&k| score_map_descending[k as usize])
            .sum::<i64>();
        let score_down = col_keys
            .iter()
            .map(|&k| score_map_ascending[k as usize])
            .sum::<i64>();
        score_left.max(score_right) + score_up.max(score_down)
    } else {
        let score = row_keys
            .iter()
            .map(|&k| score_map_descending[k as usize])
            .sum::<i64>();
        score
            + col_keys
                .iter()
                .map(|&k| score_map_descending[k as usize])
                .sum::<i64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_score() {
        assert_eq!(tile_score(0), 0);
        assert_eq!(tile_score(1), 2);
        assert_eq!(tile_score(2), 8);
        assert_eq!(tile_score(3), 24);
        assert_eq!(tile_score(4), 64);
    }

    #[test]
    fn test_evaluate_empty_board() {
        let board = [0i32; 16];
        assert_eq!(evaluate(&board, false), 0);
        assert_eq!(evaluate(&board, true), 0);
    }

    #[test]
    fn test_evaluate_single_tile() {
        let mut board = [0i32; 16];
        board[0] = 1;
        assert_eq!(evaluate(&board, false), 8);
        assert_eq!(evaluate(&board, true), 8);
    }

    #[test]
    fn test_evaluate_two_adjacent() {
        let mut board = [0i32; 16];
        board[0] = 1;
        board[1] = 1;
        assert_eq!(evaluate(&board, false), 18);
        assert_eq!(evaluate(&board, true), 18);
    }
}
