use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use serde::Deserialize;
use twenty48_utils::engine::{self, Move};

use crate::schema::{MacroxueStepRow, MacroxueStepRowV1};

use super::board_eval;
use super::types::{MetaRecord, ValuationEncoder};
use super::BOARD_LEN;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum MoveName {
    Up,
    Right,
    Down,
    Left,
}

impl MoveName {
    fn code(self) -> u8 {
        // Canonical UDLR indexing: Up=0, Down=1, Left=2, Right=3
        match self {
            MoveName::Up => 0,
            MoveName::Down => 1,
            MoveName::Left => 2,
            MoveName::Right => 3,
        }
    }
}

impl<'de> Deserialize<'de> for MoveName {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "up" => Ok(MoveName::Up),
            "right" => Ok(MoveName::Right),
            "down" => Ok(MoveName::Down),
            "left" => Ok(MoveName::Left),
            other => Err(serde::de::Error::custom(format!("unknown move '{other}'"))),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct BranchEvs {
    #[serde(default)]
    up: Option<f32>,
    #[serde(default)]
    down: Option<f32>,
    #[serde(default)]
    left: Option<f32>,
    #[serde(default)]
    right: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct StepRecord {
    #[serde(default)]
    seed: Option<u64>,
    #[serde(rename = "step_index")]
    #[serde(default)]
    step_index: Option<u32>,
    #[serde(rename = "move")]
    move_dir: MoveName,
    #[serde(default)]
    valuation_type: Option<String>,
    board: [u8; BOARD_LEN],
    #[serde(default)]
    max_rank: Option<u8>,
    #[serde(default)]
    branch_evs: BranchEvs,
}

/// Decode a packed Macroxue board back into exponent form.
pub fn decode_board(packed: u64, mask: u16) -> [u8; BOARD_LEN] {
    let mut out = [0u8; BOARD_LEN];
    for idx in 0..BOARD_LEN {
        let shift = (BOARD_LEN - 1 - idx) * 4;
        let nib = ((packed >> shift) & 0xF) as u8;
        out[idx] = if (mask >> idx) & 1 == 1 { 16 } else { nib };
    }
    out
}

pub(crate) fn parse_steps_file(
    path: &Path,
    run_id: u32,
    meta: &MetaRecord,
    encoder: &ValuationEncoder,
) -> Result<(Vec<MacroxueStepRowV1>, Vec<i64>, bool)> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let gz = GzDecoder::new(file);
    let reader = BufReader::new(gz);
    let mut rows = Vec::new();
    let mut rewards = Vec::new();
    let mut any_legal = false;
    let mut step_idx = 0u32;
    for line in reader.lines() {
        let line = line.with_context(|| format!("failed to read line from {}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: StepRecord = serde_json::from_str(&line)
            .with_context(|| format!("failed to parse JSON in {}", path.display()))?;
        let idx = record.step_index.unwrap_or(step_idx);
        step_idx = idx.saturating_add(1);
        let encoded = encoder.encode_fast(record.valuation_type.as_deref())?;

        let board_exps = record.board;
        let (board, mask) = pack_board(&board_exps);
        let board_eval = board_eval::evaluate(&board_exps, false)
            .with_context(|| format!("failed to evaluate board in {}", path.display()))?;
        let board_eval = i32::try_from(board_eval)
            .with_context(|| format!("board evaluation overflow in {}", path.display()))?;
        let mut branch_evs = [0f32; 4];
        let mut legal_bits = 0u8;
        if let Some(v) = record.branch_evs.up {
            branch_evs[0] = v;
            legal_bits |= 1u8 << 0;
        }
        if let Some(v) = record.branch_evs.down {
            branch_evs[1] = v;
            legal_bits |= 1u8 << 1;
        }
        if let Some(v) = record.branch_evs.left {
            branch_evs[2] = v;
            legal_bits |= 1u8 << 2;
        }
        if let Some(v) = record.branch_evs.right {
            branch_evs[3] = v;
            legal_bits |= 1u8 << 3;
        }
        any_legal |= legal_bits != 0;
        let step_reward = engine::merge_reward_exps(&board_exps, Move::from_udlr(record.move_dir.code()));
        let step_reward = i64::try_from(step_reward)
            .with_context(|| format!("step reward overflow in {}", path.display()))?;
        rewards.push(step_reward);
        rows.push(MacroxueStepRowV1 {
            run_id,
            step_index: idx,
            board,
            board_eval,
            tile_65536_mask: mask,
            move_dir: record.move_dir.code(),
            valuation_type: encoded,
            ev_legal: legal_bits,
            max_rank: record.max_rank.unwrap_or(meta.max_rank.unwrap_or_default()),
            seed: record.seed.unwrap_or(meta.seed) as u32,
            branch_evs,
        });
    }
    Ok((rows, rewards, any_legal))
}

pub(crate) fn attach_cumulative_reward(
    steps: Vec<MacroxueStepRowV1>,
    rewards: &[i64],
    path: &Path,
) -> Result<Vec<MacroxueStepRow>> {
    if steps.len() != rewards.len() {
        bail!(
            "reward count mismatch in {} (steps={}, rewards={})",
            path.display(),
            steps.len(),
            rewards.len()
        );
    }
    let mut rows: Vec<MacroxueStepRow> = steps.into_iter().map(Into::into).collect();
    let mut cumulative = 0i64;
    for (row, step_reward) in rows.iter_mut().rev().zip(rewards.iter().rev()) {
        cumulative += *step_reward;
        row.cumulative_reward = i32::try_from(cumulative)
            .with_context(|| format!("cumulative reward overflow in {}", path.display()))?;
    }
    Ok(rows)
}

fn pack_board(board: &[u8; BOARD_LEN]) -> (u64, u16) {
    let mut acc = 0u64;
    let mut mask = 0u16;
    for (i, &exp) in board.iter().enumerate() {
        let mut nib = exp;
        if nib >= 16 {
            mask |= 1 << i;
            nib = 15;
        }
        let shift = (15 - i) * 4;
        acc |= (nib as u64 & 0xF) << shift;
    }
    (acc, mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;
    use tempfile::tempdir;

    fn pack_board_msb_ref(cells: &[u8; 16]) -> (u64, u16) {
        let mut acc = 0u64;
        let mut mask = 0u16;
        for (i, &exp) in cells.iter().enumerate() {
            let mut nib = exp;
            if nib >= 16 {
                mask |= 1 << i;
                nib = 15;
            }
            let shift = (15 - i) * 4;
            acc |= (nib as u64 & 0xF) << shift;
        }
        (acc, mask)
    }

    #[test]
    fn round_trip_jsonl_to_step_row_msb_udlr() {
        // Prepare a tiny gzipped JSONL with one step
        let dir = tempdir().unwrap();
        let steps_path = dir.path().join("one.jsonl.gz");
        let mut enc = GzEncoder::new(
            std::fs::File::create(&steps_path).unwrap(),
            Compression::default(),
        );
        // exponents 0..15 with a 16 at position 5 to test mask; move is "right"; branch evs UDLR with Right None (illegal)
        let mut board: Vec<u8> = (0u8..16u8).collect();
        board[5] = 16u8; // force a 65536 tile at index 5
        let board_for_reward = board.clone();
        let record = serde_json::json!({
            "seed": 123u64,
            "step_index": 0u32,
            "move": "right",
            "valuation_type": "search",
            "board": board,
            "branch_evs": {"up": 0.9, "down": 0.7, "left": 0.3, "right": null},
        });
        let line = serde_json::to_string(&record).unwrap();
        enc.write_all(line.as_bytes()).unwrap();
        enc.write_all(b"\n").unwrap();
        enc.finish().unwrap();

        let meta = MetaRecord {
            seed: 123u64,
            num_moves: 1,
            score: 0,
            highest_tile: 0,
            max_rank: Some(0),
        };
        let enc_v = ValuationEncoder::new();
        let (rows, rewards, any_legal) = parse_steps_file(&steps_path, 1u32, &meta, &enc_v).unwrap();
        assert!(
            any_legal,
            "expected at least one legal branch in test record"
        );
        assert_eq!(rows.len(), 1);
        let row = rows[0];
        let rows_with_cumulative = attach_cumulative_reward(rows, &rewards, &steps_path).unwrap();
        let row_with_cumulative = &rows_with_cumulative[0];

        // Verify board packing MSB + 65536 mask behavior
        let mut cell_arr = [0u8; 16];
        for i in 0..16 {
            cell_arr[i] = i as u8;
        }
        // Force a 16 at index 5 in our reference as parse path clamps >=16 to 15 and sets mask
        cell_arr[5] = 16u8;
        let (exp_board, exp_mask) = pack_board_msb_ref(&cell_arr);
        assert_eq!(row.board, exp_board);
        assert_eq!(row.tile_65536_mask, exp_mask);
        let expected_eval = board_eval::evaluate(&cell_arr, false).unwrap();
        assert_eq!(row.board_eval, i32::try_from(expected_eval).unwrap());

        // Verify move_dir UDLR encoding (Right = 3)
        assert_eq!(row.move_dir, 3u8);
        // Verify branch EVs UDLR order and legality mask UDLR (Right illegal -> bit 3 cleared)
        // ORDER: [Up, Down, Left, Right]
        assert!((row.branch_evs[0] - 0.9).abs() < 1e-6);
        assert!((row.branch_evs[1] - 0.7).abs() < 1e-6);
        assert!((row.branch_evs[2] - 0.3).abs() < 1e-6);
        assert!((row.branch_evs[3] - 0.0).abs() < 1e-6);
        // ev_legal bits: Up(1) + Down(2) + Left(4) = 0b0111 = 7
        assert_eq!(row.ev_legal, 0b0111u8);

        let board_exps: [u8; BOARD_LEN] = board_for_reward.try_into().unwrap();
        let step_reward =
            engine::merge_reward_exps(&board_exps, Move::from_udlr(row_with_cumulative.move_dir));
        assert_eq!(row_with_cumulative.cumulative_reward as u64, step_reward);
    }

    #[test]
    fn decode_board_roundtrip() {
        let mut cells = [0u8; BOARD_LEN];
        for (idx, slot) in cells.iter_mut().enumerate() {
            *slot = (idx % 15) as u8;
        }
        cells[5] = 16;
        let (packed, mask) = pack_board_msb_ref(&cells);
        let decoded = decode_board(packed, mask);
        assert_eq!(decoded, cells);
    }

    #[test]
    fn cumulative_reward_is_reverse_sum() {
        let dir = tempdir().unwrap();
        let steps_path = dir.path().join("two.jsonl.gz");
        let mut enc = GzEncoder::new(
            std::fs::File::create(&steps_path).unwrap(),
            Compression::default(),
        );
        let board = vec![1u8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        for step_index in 0..2u32 {
            let record = serde_json::json!({
                "seed": 999u64,
                "step_index": step_index,
                "move": "left",
                "valuation_type": "search",
                "board": board,
                "branch_evs": {"up": 0.1, "down": 0.2, "left": 0.3, "right": 0.4},
            });
            let line = serde_json::to_string(&record).unwrap();
            enc.write_all(line.as_bytes()).unwrap();
            enc.write_all(b"\n").unwrap();
        }
        enc.finish().unwrap();

        let meta = MetaRecord {
            seed: 999u64,
            num_moves: 2,
            score: 0,
            highest_tile: 0,
            max_rank: Some(0),
        };
        let enc_v = ValuationEncoder::new();
        let (rows, rewards, _any_legal) = parse_steps_file(&steps_path, 7u32, &meta, &enc_v).unwrap();
        assert_eq!(rows.len(), 2);
        let rows = attach_cumulative_reward(rows, &rewards, &steps_path).unwrap();

        let mut expected = vec![0i64; rows.len()];
        for i in (0..rows.len()).rev() {
            let step_reward = rewards[i];
            let next = if i + 1 < rows.len() { expected[i + 1] } else { 0 };
            expected[i] = step_reward + next;
        }
        for (row, exp) in rows.iter().zip(expected.iter()) {
            assert_eq!(row.cumulative_reward as i64, *exp);
        }
    }
}
