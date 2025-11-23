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

    let mut shard_row_counts = Vec::with_capacity(steps_files.len());
    let mut per_run: Vec<Vec<StepValueInput>> = vec![Vec::new(); runs.len()];
    let mut last_run_seen: Option<u32> = None;
    let mut last_step_index: Vec<Option<u32>> = vec![None; runs.len()];
    let total_steps: usize = runs.iter().map(|r| r.steps).sum();

    let pb = crate::macroxue::default_progress_bar(total_steps as u64);
    pb.set_message("loading steps");

    for path in &steps_files {
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
            let row = row.with_context(|| format!("failed to decode row in {}", path.display()))?;
            shard_rows += 1;
            pb.inc(1);
            if let Some(prev) = last_run_seen {
                if row.run_id < prev {
                    bail!(
                        "run_id {} encountered after {} while reading {}",
                        row.run_id,
                        prev,
                        path.display()
                    );
                }
            }
            last_run_seen = Some(row.run_id);

            let run_id = row.run_id as usize;
            let run_buf = per_run
                .get_mut(run_id)
                .ok_or_else(|| anyhow!("run_id {} out of range", row.run_id))?;
            if let Some(prev_step) = last_step_index
                .get_mut(run_id)
                .and_then(|slot| slot.replace(row.step_index))
            {
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
            run_buf.push(StepValueInput {
                step_index: row.step_index,
                board: row.board,
                move_dir: row.move_dir,
            });
        }
        shard_row_counts.push(shard_rows);
    }

    pb.finish_with_message("steps loaded");

    for (idx, run) in runs.iter().enumerate() {
        let collected = per_run
            .get(idx)
            .ok_or_else(|| anyhow!("missing run buffer for id {}", idx))?;
        if collected.len() != run.steps {
            bail!(
                "run {} reports {} steps in metadata but {} rows were read",
                run.run_id,
                run.steps,
                collected.len()
            );
        }
    }

    let mut writer = StepsWriter::<ValueRow>::with_prefix(
        &opts.output_dir,
        "values",
        shard_rows_per_output(&shard_row_counts),
        opts.overwrite,
    )?;

    let process = || -> Result<Vec<Vec<ValueRow>>> {
        per_run
            .par_iter()
            .enumerate()
            .map(|(run_id, steps)| compute_run(run_id as u32, steps, opts.gamma, opts.reward_scale))
            .collect()
    };

    let results: Vec<Vec<ValueRow>> = if let Some(n) = opts.max_workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .context("failed to build rayon thread pool")?
            .install(process)?
    } else {
        process()?
    };

    let mut written = 0usize;
    for rows in &results {
        writer.write(rows)?;
        written += rows.len();
    }
    let shards = writer.finish()?;

    info!(
        "Wrote value sidecar: {} runs, {} steps, {} shard(s)",
        results.len(),
        written,
        shards
    );

    if written != total_steps {
        bail!(
            "wrote {} value rows but metadata reports {} steps",
            written,
            total_steps
        );
    }

    Ok(ValueSidecarSummary {
        runs: results.len(),
        steps: written,
        shards,
        gamma: opts.gamma,
        reward_scale: opts.reward_scale,
    })
}

fn shard_rows_per_output(input_shards: &[usize]) -> Option<usize> {
    if input_shards.len() <= 1 {
        None
    } else {
        input_shards.first().copied()
    }
}

#[derive(Clone, Copy)]
struct StepValueInput {
    step_index: u32,
    board: u64,
    move_dir: u8,
}

fn compute_run(
    run_id: u32,
    steps: &[StepValueInput],
    gamma: f64,
    reward_scale: f64,
) -> Result<Vec<ValueRow>> {
    // Compute per-step rewards in parallel to fully utilise cores even for a single long run.
    let rewards: Vec<f32> = steps
        .par_iter()
        .map(|step| compute_reward(step.board, step.move_dir))
        .collect::<Result<Vec<_>>>()?;

    let mut out = Vec::with_capacity(steps.len());
    let mut return_raw = 0f64;
    let mut return_scaled = 0f64;

    for ((step, &reward), reward_scaled) in steps
        .iter()
        .rev()
        .zip(rewards.iter().rev())
        .zip(rewards.iter().map(|r| *r as f64 * reward_scale).rev())
    {
        let reward_f64 = reward as f64;
        return_raw = reward_f64 + gamma * return_raw;
        return_scaled = reward_scaled + gamma * return_scaled;

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

fn compute_reward(board_raw: u64, move_dir: u8) -> Result<f32> {
    use twenty48_utils::engine::{Board, Move};

    let dir = match move_dir {
        0 => Move::Up,
        1 => Move::Down,
        2 => Move::Left,
        3 => Move::Right,
        other => bail!("invalid move_dir {} (expected 0..=3)", other),
    };

    let board = Board::from_raw(board_raw);
    let before = twenty48_utils::engine::get_score(board);
    let after = twenty48_utils::engine::get_score(board.shift(dir));
    debug!("move {:?}: before={} after={}", dir, before, after);
    Ok((after as f64 - before as f64) as f32)
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
    use crate::writer::StepsWriter;
    use tempfile::tempdir;

    fn pack_board(exps: [u8; 16]) -> u64 {
        let mut acc = 0u64;
        for (i, exp) in exps.iter().enumerate() {
            let shift = (15 - i) * 4;
            acc |= (*exp as u64 & 0xF) << shift;
        }
        acc
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
        assert_eq!(returns_scaled, vec![24.0, 32.0, 32.0, 0.0]);
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
