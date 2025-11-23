//! Compute per-step rewards and discounted returns as a sidecar to packed datasets.
//!
//! The sidecar is aligned 1:1 with `steps-*.npy` rows and keeps both raw and
//! scaled rewards/returns so downstream objectives can choose the transform at
//! training time.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use log::{debug, info};
use npyz::{DType, Field, NpyFile, TypeStr};
use rayon::prelude::*;

use crate::schema::{MacroxueStepRow, StructuredRow};
use crate::writer::StepsWriter;

/// Row layout for the value sidecar aligned with `steps-*.npy`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize,
)]
pub struct ValueRow {
    pub run_id: u32,
    pub step_index: u32,
    /// Raw merge reward (sum of merged tile values) for the recorded move.
    pub reward: f32,
    /// Reward after applying `reward_scale`.
    pub reward_scaled: f32,
    /// Discounted return using raw rewards.
    pub return_raw: f32,
    /// Discounted return using scaled rewards.
    pub return_scaled: f32,
}

impl StructuredRow for ValueRow {
    fn dtype() -> DType {
        let u4: TypeStr = "<u4".parse().unwrap();
        let f4: TypeStr = "<f4".parse().unwrap();
        DType::Record(vec![
            Field {
                name: "run_id".into(),
                dtype: DType::Plain(u4.clone()),
            },
            Field {
                name: "step_index".into(),
                dtype: DType::Plain(u4),
            },
            Field {
                name: "reward".into(),
                dtype: DType::Plain(f4.clone()),
            },
            Field {
                name: "reward_scaled".into(),
                dtype: DType::Plain(f4.clone()),
            },
            Field {
                name: "return_raw".into(),
                dtype: DType::Plain(f4.clone()),
            },
            Field {
                name: "return_scaled".into(),
                dtype: DType::Plain(f4),
            },
        ])
    }
}

/// CLI options for generating a value sidecar.
#[derive(Clone, Debug)]
pub struct ValueSidecarOptions {
    /// Dataset directory containing `steps.npy` / `steps-*.npy` and `metadata.db`.
    pub dataset_dir: PathBuf,
    /// Output directory for the value sidecar (defaults to `dataset_dir`).
    pub output_dir: PathBuf,
    /// Discount factor applied when computing returns.
    pub gamma: f64,
    /// Scale applied to rewards before computing the scaled return.
    pub reward_scale: f64,
    /// Epsilon for the MuZero-style invertible value transform.
    pub epsilon: f64,
    /// Optional override for Rayon worker count.
    pub max_workers: Option<usize>,
    /// Overwrite existing outputs when true.
    pub overwrite: bool,
}

/// Summary returned after writing the value sidecar.
#[derive(Clone, Debug, PartialEq)]
pub struct ValueSidecarSummary {
    pub runs: usize,
    pub steps: usize,
    pub shards: usize,
    pub gamma: f64,
    pub reward_scale: f64,
    pub epsilon: f64,
}

/// Generate `values-*.npy` aligned to the packed Macroxue dataset.
pub fn add_value_sidecar(opts: ValueSidecarOptions) -> Result<ValueSidecarSummary> {
    if opts.gamma < 0.0 {
        bail!("gamma must be non-negative");
    }
    if opts.reward_scale <= 0.0 {
        bail!("reward_scale must be positive");
    }

    twenty48_utils::engine::new();

    let runs = crate::macroxue::load_runs(&opts.dataset_dir)?;
    if runs.is_empty() {
        bail!("metadata.db in {} has no runs", opts.dataset_dir.display());
    }

    let steps_files = collect_step_files(&opts.dataset_dir)?;
    if steps_files.is_empty() {
        bail!(
            "no steps.npy or steps-*.npy files found in {}",
            opts.dataset_dir.display()
        );
    }

    let total_steps: usize = runs.iter().map(|r| r.steps).sum();
    let shard_row_counts = step_shard_row_counts(&steps_files)?;
    let shard_total: usize = shard_row_counts.iter().sum();
    if shard_total != total_steps {
        bail!(
            "steps shards contain {} rows but metadata reports {} steps",
            shard_total,
            total_steps
        );
    }

    let render_sidecar = || -> Result<ValueSidecarSummary> {
        let mut writer = if should_number_values(&steps_files) {
            StepsWriter::<ValueRow>::with_shard_plan(
                &opts.output_dir,
                "values",
                &shard_row_counts,
                opts.overwrite,
            )?
        } else {
            StepsWriter::<ValueRow>::with_prefix(&opts.output_dir, "values", None, opts.overwrite)?
        };

        let pb = crate::macroxue::default_progress_bar(total_steps as u64);
        pb.set_message("computing values");

        let mut written = 0usize;
        let mut expected_run_idx = 0usize;
        let mut current_run_id: Option<u32> = None;
        let mut last_step_index: Option<u32> = None;
        let mut run_steps: Vec<StepValueInput> = Vec::new();

        for (shard_idx, path) in steps_files.iter().enumerate() {
            let file = fs::File::open(path)
                .with_context(|| format!("failed to open steps shard {}", path.display()))?;
            let mut reader = std::io::BufReader::new(file);
            let npy = NpyFile::new(&mut reader)
                .with_context(|| format!("failed to read {}", path.display()))?;

            let mut rows = npy
                .data::<MacroxueStepRow>()
                .map_err(|err| anyhow!("{}: {err}", path.display()))?;
            let mut shard_rows = 0usize;

            while let Some(row) = rows.next() {
                let row =
                    row.with_context(|| format!("failed to decode row in {}", path.display()))?;
                shard_rows += 1;
                pb.inc(1);

                if let Some(current) = current_run_id {
                    if row.run_id < current {
                        bail!(
                            "run_id {} encountered after {} while reading {}",
                            row.run_id,
                            current,
                            path.display()
                        );
                    }
                    if row.run_id != current {
                        written += flush_run(
                            current,
                            expected_run_idx,
                            &mut run_steps,
                            &runs,
                            opts.gamma,
                            opts.reward_scale,
                            opts.epsilon,
                            &mut writer,
                        )?;
                        expected_run_idx += 1;
                        run_steps.clear();
                        last_step_index = None;
                        current_run_id = Some(row.run_id);
                    }
                } else {
                    current_run_id = Some(row.run_id);
                }

                let expected_run = runs.get(expected_run_idx).ok_or_else(|| {
                    anyhow!(
                        "run_id {} encountered but metadata only lists {} run(s)",
                        row.run_id,
                        runs.len()
                    )
                })?;
                if row.run_id != expected_run.run_id {
                    bail!(
                        "run_id {} encountered at position {} but metadata run_id is {}",
                        row.run_id,
                        expected_run_idx,
                        expected_run.run_id
                    );
                }

                if let Some(prev_step) = last_step_index.replace(row.step_index) {
                    if row.step_index < prev_step {
                        bail!(
                            "step_index regressed for run {} ({} after {}) in {}",
                            row.run_id,
                            row.step_index,
                            prev_step,
                            path.display()
                        );
                    }
                }

                run_steps.push(StepValueInput {
                    step_index: row.step_index,
                    board: row.board,
                    tile_65536_mask: row.tile_65536_mask,
                    move_dir: row.move_dir,
                });
            }

            let expected_rows = shard_row_counts
                .get(shard_idx)
                .copied()
                .ok_or_else(|| anyhow!("missing shard metadata for index {}", shard_idx))?;
            if shard_rows != expected_rows {
                bail!(
                    "steps shard {} reported {} rows in header but {} rows were read",
                    path.display(),
                    expected_rows,
                    shard_rows
                );
            }
        }

        if let Some(run_id) = current_run_id {
            written += flush_run(
                run_id,
                expected_run_idx,
                &mut run_steps,
                &runs,
                opts.gamma,
                opts.reward_scale,
                opts.epsilon,
                &mut writer,
            )?;
            expected_run_idx += 1;
        }

        pb.finish_with_message("value sidecar written");

        if expected_run_idx != runs.len() {
            bail!(
                "metadata lists {} run(s) but {} were observed in steps",
                runs.len(),
                expected_run_idx
            );
        }
        if written != total_steps {
            bail!(
                "wrote {} value rows but metadata reports {} steps",
                written,
                total_steps
            );
        }

        let shards = writer.finish()?;
        info!(
            "Wrote value sidecar: {} runs, {} steps, {} shard(s)",
            expected_run_idx, written, shards
        );

        Ok(ValueSidecarSummary {
            runs: expected_run_idx,
            steps: written,
            shards,
            gamma: opts.gamma,
            reward_scale: opts.reward_scale,
            epsilon: opts.epsilon,
        })
    };

    if let Some(n) = opts.max_workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .context("failed to build rayon thread pool")?
            .install(render_sidecar)
    } else {
        render_sidecar()
    }
}

fn step_shard_row_counts(step_paths: &[PathBuf]) -> Result<Vec<usize>> {
    let mut counts = Vec::with_capacity(step_paths.len());
    for path in step_paths {
        let file = fs::File::open(path)
            .with_context(|| format!("failed to open steps shard {}", path.display()))?;
        let mut reader = std::io::BufReader::new(file);
        let npy = NpyFile::new(&mut reader)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let shape = npy.shape();
        if shape.is_empty() {
            bail!("{} has no dimensions", path.display());
        }
        let rows = usize::try_from(shape[0])
            .map_err(|_| anyhow!("{} has too many rows for this platform", path.display()))?;
        if rows == 0 {
            bail!("{} has zero rows", path.display());
        }
        counts.push(rows);
    }
    Ok(counts)
}

fn should_number_values(step_files: &[PathBuf]) -> bool {
    if step_files.len() > 1 {
        return true;
    }
    if let Some(first) = step_files.first() {
        if let Some(name) = first.file_name().and_then(|n| n.to_str()) {
            return name.starts_with("steps-");
        }
    }
    false
}

#[derive(Clone, Copy)]
struct StepValueInput {
    step_index: u32,
    board: u64,
    tile_65536_mask: u16,
    move_dir: u8,
}

fn flush_run(
    run_id: u32,
    run_idx: usize,
    steps: &mut Vec<StepValueInput>,
    runs: &[crate::macroxue::RunSummary],
    gamma: f64,
    reward_scale: f64,
    epsilon: f64,
    writer: &mut StepsWriter<ValueRow>,
) -> Result<usize> {
    let expected = runs.get(run_idx).ok_or_else(|| {
        anyhow!(
            "run_id {} out of range (metadata has {} run(s))",
            run_id,
            runs.len()
        )
    })?;
    if expected.run_id != run_id {
        bail!(
            "run_id {} encountered at index {} but metadata lists run_id {}",
            run_id,
            run_idx,
            expected.run_id
        );
    }
    if steps.len() != expected.steps {
        bail!(
            "run {} reports {} steps in metadata but {} rows were read",
            run_id,
            expected.steps,
            steps.len()
        );
    }

    let values = compute_run(run_id, steps, gamma, reward_scale, epsilon)?;
    writer.write(&values)?;
    let written = values.len();
    steps.clear();
    Ok(written)
}

fn compute_run(
    run_id: u32,
    steps: &[StepValueInput],
    gamma: f64,
    reward_scale: f64,
    epsilon: f64,
) -> Result<Vec<ValueRow>> {
    // Compute per-step rewards in parallel to fully utilise cores even for a single long run.
    let rewards: Vec<f32> = steps
        .par_iter()
        .map(|step| compute_reward(step.board, step.tile_65536_mask, step.move_dir))
        .collect::<Result<Vec<_>>>()?;

    let mut out = Vec::with_capacity(steps.len());
    let mut return_raw = 0f64;

    for (step, &reward) in steps.iter().rev().zip(rewards.iter().rev()) {
        let reward_f64 = reward as f64;
        return_raw = reward_f64 + gamma * return_raw;
        let reward_scaled = reward_f64 * reward_scale;

        let return_scaled = h_transform(return_raw, epsilon, run_id, step.step_index)?;

        out.push(ValueRow {
            run_id,
            step_index: step.step_index,
            reward,
            reward_scaled: reward_scaled as f32,
            return_raw: return_raw as f32,
            return_scaled: return_scaled as f32,
        });
    }

    out.reverse();
    Ok(out)
}

/// MuZero-style invertible transform for value/reward targets.
/// h(x) = sign(x) * (sqrt(x + 1) - 1 + eps * x)
fn h_transform(x: f64, epsilon: f64, run_id: u32, step_index: u32) -> Result<f64> {
    if x < -1.0 {
        bail!(
            "MuZero transform failed for run {}, step {}: input value {} is less than -1",
            run_id,
            step_index,
            x
        );
    }
    // NB: The value inside the sqrt can be negative if x < -1.0, which is handled above.
    let transformed = (x + 1.0).sqrt() - 1.0 + epsilon * x;
    Ok(x.signum() * transformed)
}

fn compute_reward(board_raw: u64, tile_65536_mask: u16, move_dir: u8) -> Result<f32> {
    use crate::macroxue::{decode_board, BOARD_LEN};
    use twenty48_utils::engine::Move;

    let dir = match move_dir {
        0 => Move::Up,
        1 => Move::Down,
        2 => Move::Left,
        3 => Move::Right,
        other => bail!("invalid move_dir {} (expected 0..=3)", other),
    };

    let exps = decode_board(board_raw, tile_65536_mask);
    let mut reward: u64 = 0;

    match dir {
        Move::Left | Move::Right => {
            for row in 0..4 {
                let mut line = [
                    exps[row * 4],
                    exps[row * 4 + 1],
                    exps[row * 4 + 2],
                    exps[row * 4 + 3],
                ];
                if matches!(dir, Move::Right) {
                    line.reverse();
                }
                reward = reward.checked_add(line_reward_left(line)?).ok_or_else(|| {
                    anyhow!("reward overflow while processing row {}", row)
                })?;
            }
        }
        Move::Up | Move::Down => {
            for col in 0..(BOARD_LEN / 4) {
                let mut line = [
                    exps[col],
                    exps[4 + col],
                    exps[8 + col],
                    exps[12 + col],
                ];
                if matches!(dir, Move::Down) {
                    line.reverse();
                }
                reward = reward.checked_add(line_reward_left(line)?).ok_or_else(|| {
                    anyhow!("reward overflow while processing column {}", col)
                })?;
            }
        }
    }

    debug!("move {:?}: reward={}", dir, reward);
    Ok(reward as f32)
}

fn line_reward_left(mut tiles: [u8; 4]) -> Result<u64> {
    let mut reward = 0u64;
    for start in 0..4 {
        reward = reward
            .checked_add(calculate_left_shift_with_reward(&mut tiles[start..])?)
            .ok_or_else(|| anyhow!("reward overflow while shifting line"))?;
    }
    Ok(reward)
}

fn calculate_left_shift_with_reward(slice: &mut [u8]) -> Result<u64> {
    let mut acc = 0u8;
    let mut reward = 0u64;
    for cell in slice.iter_mut() {
        let val = *cell;
        if acc != 0 && acc == val {
            *cell = 0;
            acc = acc
                .checked_add(1)
                .ok_or_else(|| anyhow!("tile exponent overflow while merging"))?;
            let merge_reward = 1u64
                .checked_shl(acc as u32)
                .ok_or_else(|| anyhow!("merge reward overflow for exponent {}", acc))?;
            reward = reward
                .checked_add(merge_reward)
                .ok_or_else(|| anyhow!("reward overflow while merging exponent {}", acc))?;
            break;
        } else if acc != 0 && val != 0 && acc != val {
            break;
        } else if acc == 0 && val != 0 {
            *cell = 0;
            acc = val;
        }
    }
    slice[0] = acc;
    Ok(reward)
}

fn collect_step_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let single = dir.join("steps.npy");
    if single.exists() {
        files.push(single);
    }
    let mut shards: Vec<PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("failed to read {}", dir.display()))?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            if name.starts_with("steps-") && name.ends_with(".npy") {
                Some(entry.path())
            } else {
                None
            }
        })
        .collect();
    shards.sort();
    files.extend(shards);
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macroxue::{RunSummary, write_metadata};
    use crate::writer::{StepsWriter, write_single_shard};
    use tempfile::tempdir;

    fn pack_board(exps: [u8; 16]) -> u64 {
        let mut acc = 0u64;
        for (i, exp) in exps.iter().enumerate() {
            let shift = (15 - i) * 4;
            acc |= (*exp as u64 & 0xF) << shift;
        }
        acc
    }

    fn pack_board_with_mask(exps: [u8; 16]) -> (u64, u16) {
        let mut packed = 0u64;
        let mut mask = 0u16;
        for (i, &exp) in exps.iter().enumerate() {
            let mut nib = exp;
            if nib >= 16 {
                mask |= 1 << i;
                nib = 15;
            }
            let shift = (15 - i) * 4;
            packed |= (nib as u64 & 0xF) << shift;
        }
        (packed, mask)
    }

    fn make_row(run_id: u32, step_index: u32, board: u64, move_dir: u8) -> MacroxueStepRow {
        MacroxueStepRow {
            run_id,
            step_index,
            board,
            board_eval: 0,
            tile_65536_mask: 0,
            move_dir,
            valuation_type: 0,
            ev_legal: 0xFF,
            max_rank: 0,
            seed: 0,
            branch_evs: [0.0; 4],
        }
    }

    #[test]
    fn reward_matches_engine_without_overflow_tiles() {
        twenty48_utils::engine::new();
        // Two merges: 2+2 -> 4, 4+4 -> 8 (total reward 12).
        let board = pack_board([1, 1, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let reward = compute_reward(board, 0, 2).unwrap();

        let before = twenty48_utils::engine::get_score(twenty48_utils::engine::Board::from_raw(board));
        let after = twenty48_utils::engine::get_score(
            twenty48_utils::engine::Board::from_raw(board).shift(twenty48_utils::engine::Move::Left),
        );
        assert_eq!(reward, (after - before) as f32);
    }

    #[test]
    fn reward_accounts_for_merging_into_65536() {
        twenty48_utils::engine::new();
        // Two 32768 tiles merge into a 65536 (reward 65536), mask is zero pre-move.
        let board = pack_board([15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let reward = compute_reward(board, 0, 2).unwrap();
        assert_eq!(reward, 65536.0);
    }

    #[test]
    fn reward_respects_overflow_masked_tiles() {
        twenty48_utils::engine::new();
        // Two 65536 tiles (exp=16) merge into a 131072 tile (reward 131072).
        let (board, mask) =
            pack_board_with_mask([16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let reward = compute_reward(board, mask, 2).unwrap();
        assert_eq!(reward, 131072.0);
    }

    #[test]
    fn discounted_returns_respect_run_boundaries() {
        twenty48_utils::engine::new();

        let tmp = tempdir().unwrap();
        let dataset_dir = tmp.path().join("dataset");
        fs::create_dir_all(&dataset_dir).unwrap();

        // Run 0 has 3 steps; run 1 has 1 step. Shard after every 2 rows.
        let rows_per_shard = Some(2);
        let mut writer = StepsWriter::<MacroxueStepRow>::new(&dataset_dir, rows_per_shard, true)
            .unwrap();

        // Run 0 step boards chosen to produce rewards [4, 8, 16] when moving left.
        let b0 = pack_board([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let b1 = pack_board([2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let b2 = pack_board([3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let b_no_merge = pack_board([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

        writer
            .write(&[
                make_row(0, 0, b0, 2),
                make_row(0, 1, b1, 2),
                make_row(0, 2, b2, 2),
                make_row(1, 0, b_no_merge, 3), // run 1 reward should be zero moving right
            ])
            .unwrap();
        let shard_count = writer.finish().unwrap();
        assert_eq!(shard_count, 2, "expected sharding to split on 2 rows");

        let runs = vec![
            RunSummary {
                run_id: 0,
                seed: 0,
                steps: 3,
                max_score: 0,
                highest_tile: 0,
            },
            RunSummary {
                run_id: 1,
                seed: 0,
                steps: 1,
                max_score: 0,
                highest_tile: 0,
            },
        ];
        write_metadata(&dataset_dir, &runs, true).unwrap();

        let opts = ValueSidecarOptions {
            dataset_dir: dataset_dir.clone(),
            output_dir: dataset_dir.clone(),
            gamma: 0.5,
            reward_scale: 2.0,
            epsilon: 0.1,
            max_workers: None,
            overwrite: true,
        };

        let summary = add_value_sidecar(opts).unwrap();
        assert_eq!(summary.runs, 2);
        assert_eq!(summary.steps, 4);

        let seen_rows = read_values(&dataset_dir);

        assert_eq!(seen_rows.len(), 4);

        let rewards: Vec<f32> = seen_rows.iter().map(|r| r.reward).collect();
        assert_eq!(rewards, vec![4.0, 8.0, 16.0, 0.0]);

        let returns_raw: Vec<f32> = seen_rows.iter().map(|r| r.return_raw).collect();
        assert_eq!(returns_raw, vec![12.0, 16.0, 16.0, 0.0]);

        let rewards_scaled: Vec<f32> = seen_rows.iter().map(|r| r.reward_scaled).collect();
        assert_eq!(rewards_scaled, vec![8.0, 16.0, 32.0, 0.0]);

        let returns_scaled: Vec<f32> = seen_rows.iter().map(|r| r.return_scaled).collect();
        // Raw returns are [12.0, 16.0, 16.0, 0.0]
        // h(12.0, 0.1) = 1.0 * (sqrt(13.0) - 1.0 + 0.1 * 12.0) = 3.60555 - 1.0 + 1.2 = 3.80555
        // h(16.0, 0.1) = 1.0 * (sqrt(17.0) - 1.0 + 0.1 * 16.0) = 4.1231 - 1.0 + 1.6 = 4.7231
        // h(0.0, 0.1) = 0
        let expected = vec![3.8055512, 4.7231054, 4.7231054, 0.0];
        returns_scaled
            .iter()
            .zip(expected.iter())
            .for_each(|(a, e)| assert!((a - e).abs() < 1e-6));
    }

    #[test]
    fn test_h_transform() {
        assert!((h_transform(12.0, 0.1, 0, 0).unwrap() - 3.8055512).abs() < 1e-6);
        assert!((h_transform(16.0, 0.1, 0, 0).unwrap() - 4.7231054).abs() < 1e-6);
        assert_eq!(h_transform(0.0, 0.1, 0, 0).unwrap(), 0.0);
        assert!(h_transform(-2.0, 0.1, 0, 0).is_err());
    }

    #[test]
    fn value_shards_match_step_shards_one_to_one() {
        twenty48_utils::engine::new();

        let tmp = tempdir().unwrap();
        let dataset_dir = tmp.path().join("dataset");
        fs::create_dir_all(&dataset_dir).unwrap();

        let board = pack_board([0; 16]);
        let mut rows = Vec::new();
        for idx in 0..7u32 {
            rows.push(make_row(0, idx, board, 2));
        }

        write_single_shard(&rows[..2], &dataset_dir.join("steps-00000.npy")).unwrap();
        write_single_shard(&rows[2..], &dataset_dir.join("steps-00001.npy")).unwrap();

        let runs = vec![RunSummary {
            run_id: 0,
            seed: 0,
            steps: rows.len(),
            max_score: 0,
            highest_tile: 0,
        }];
        write_metadata(&dataset_dir, &runs, true).unwrap();

        let opts = ValueSidecarOptions {
            dataset_dir: dataset_dir.clone(),
            output_dir: dataset_dir.clone(),
            gamma: 1.0,
            reward_scale: 1.0,
            epsilon: 0.001,
            max_workers: None,
            overwrite: true,
        };

        let summary = add_value_sidecar(opts).unwrap();
        assert_eq!(summary.shards, 2);
        assert_eq!(summary.steps, rows.len());

        let mut value_paths: Vec<PathBuf> = fs::read_dir(&dataset_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("values-") && n.ends_with(".npy"))
                    .unwrap_or(false)
            })
            .collect();
        value_paths.sort();
        let shard_sizes = step_shard_row_counts(&value_paths).unwrap();
        assert_eq!(shard_sizes, vec![2, 5]);
    }

    fn read_values(dir: &Path) -> Vec<ValueRow> {
        let mut paths: Vec<PathBuf> = fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name();
                let name = name.to_str()?;
                if (name == "values.npy") || (name.starts_with("values-") && name.ends_with(".npy")) {
                    Some(e.path())
                } else {
                    None
                }
            })
            .collect();
        paths.sort();
        let mut rows = Vec::new();
        for path in paths {
            let file = fs::File::open(&path).unwrap();
            let mut reader = std::io::BufReader::new(file);
            let npy = NpyFile::new(&mut reader).unwrap();
            rows.extend(npy.into_vec::<ValueRow>().unwrap());
        }
        rows
    }
}
