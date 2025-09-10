//! Dataset builder that flattens `.a2run2` runs into a NumPy structured array
//! (`steps.npy`) plus a SQLite metadata database (`metadata.db`).
//!
//! Differences vs. the previous revision
//! -------------------------------------
//! * Uses the `npyz` crate (with `features = ["derive"]`) to define and write a
//!   structured dtype instead of manually emitting an NPY v1.0 header + raw
//!   rows. The dtype is specified explicitly to match NumPy exactly.
//! * Uses `npyz` to count rows and to append by round-tripping
//!   `StepRow` values, avoiding any manual byte slicing.
//! * Maintains identical row layout and field names so Python/NumPy and
//!   PyTorch code remain compatible.
//!
//! Cargo features required
//! -----------------------
//! ```toml
//! npyz = { version = "0.8", features = ["derive"] }
//! rusqlite = { version = "0.31", features = ["bundled" ] }
//! rayon = "1"
//! walkdir = "2"
//! anyhow = "1"
//! ```

#![allow(unexpected_cfgs)]
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use anyhow::Context as _;
use rayon::prelude::*;
use rusqlite::{params, Connection};
use walkdir::WalkDir;

use npyz::{DType, Field, TypeStr, WriterBuilder};

use crate::serialization::{self as ser, BranchV2, RunV2};

/// A single flattened step row written to steps.npy.
/// Field names are chosen to match the NumPy dtype exactly.
#[derive(Clone, Copy, Debug, npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize)]
pub struct StepRow {
    pub board: u64,
    pub r#move: u8,
    pub ev_legal: u8,
    pub ev_values: [f32; 4],
    pub run_id: u32,
    pub step_index: u16,
}

/// Summary returned by a dataset build.
#[derive(Clone, Copy, Debug)]
pub struct BuildReport {
    pub runs: usize,
    pub steps: usize,
}

#[derive(Clone, Debug)]
struct RunRow {
    id: u32,
    first_step_idx: u32,
    num_steps: u32,
    max_score: u64,
    highest_tile: u32,
    engine: String,
    start_time: u64,
    elapsed_s: f32,
}

/// Build a dataset directory from `.a2run2` files under `runs_dir`.
///
/// Writes `steps.npy` (structured array rows) and `metadata.db` atomically to `output_dir`.
pub fn build_dataset(runs_dir: &Path, output_dir: &Path) -> anyhow::Result<BuildReport> {
    anyhow::ensure!(runs_dir.is_dir(), "input must be a directory");
    std::fs::create_dir_all(output_dir)?;

    // Discover runs
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    for e in WalkDir::new(runs_dir).into_iter().filter_map(Result::ok) {
        if e.file_type().is_file() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("a2run2") {
                files.push(p.to_path_buf());
            }
        }
    }
    files.sort();
    anyhow::ensure!(!files.is_empty(), "no .a2run2 files found");

    // Load and decode all runs in parallel
    let runs: Vec<RunV2> = files
        .par_iter()
        .map(|p| ser::read_postcard_from_path(p).map_err(|e| anyhow::anyhow!(e)))
        .collect::<Result<Vec<_>, _>>()?;

    write_from_runs(&runs, output_dir)
}

/// Build a dataset directory from already-decoded runs.
///
/// Writes `steps.npy` (structured array rows) and `metadata.db` atomically to `output_dir`.
pub fn build_dataset_from_runs(runs: &[RunV2], output_dir: &Path) -> anyhow::Result<BuildReport> {
    anyhow::ensure!(!runs.is_empty(), "no runs provided");
    std::fs::create_dir_all(output_dir)?;
    write_from_runs(runs, output_dir)
}

fn write_from_runs(runs: &[RunV2], output_dir: &Path) -> anyhow::Result<BuildReport> {
    // Flatten to steps and run metadata
    let mut steps: Vec<StepRow> =
        Vec::with_capacity(runs.iter().map(|r| r.meta.steps as usize).sum());

    let mut run_rows: Vec<RunRow> = Vec::with_capacity(runs.len());

    let mut next_first = 0u32;
    for (run_idx, run) in runs.iter().enumerate() {
        let rid = run_idx as u32;
        let mut ev_tmp = [0.0f32; 4];
        for (si, s) in run.steps.iter().enumerate() {
            let dir_idx = match s.chosen {
                crate::engine::Move::Up => 0u8,
                crate::engine::Move::Down => 1u8,
                crate::engine::Move::Left => 2u8,
                crate::engine::Move::Right => 3u8,
            };
            let branches = s.branches.as_ref().ok_or_else(|| {
                anyhow::anyhow!("run {} missing branches; required for dataset", run_idx)
            })?;
            let mut mask: u8 = 0;
            for (i, b) in branches.iter().enumerate() {
                match *b {
                    BranchV2::Illegal => {
                        ev_tmp[i] = 0.0;
                    }
                    BranchV2::Legal(v) => {
                        mask |= 1u8 << (i as u8);
                        // Write exact 1.0 for the chosen branch to ensure stable labels
                        ev_tmp[i] = if i as u8 == dir_idx {
                            1.0
                        } else {
                            v.max(0.0).min(1.0)
                        };
                    }
                }
            }
            steps.push(StepRow {
                board: s.pre_board,
                r#move: dir_idx,
                ev_legal: mask,
                ev_values: ev_tmp,
                run_id: rid,
                step_index: si as u16,
            });
        }
        let rs = run.meta.steps;
        run_rows.push(RunRow {
            id: rid,
            first_step_idx: next_first,
            num_steps: rs,
            max_score: run.meta.max_score,
            highest_tile: run.meta.highest_tile,
            engine: run.meta.engine_str.clone().unwrap_or_default(),
            start_time: run.meta.start_unix_s,
            elapsed_s: run.meta.elapsed_s,
        });
        next_first = next_first.saturating_add(rs);
    }

    // Write steps.npy atomically using `npyz` with an explicit structured dtype.
    let steps_tmp = output_dir.join("steps.npy.tmp");
    write_steps_npy(&steps, &steps_tmp).context("write steps.npy.tmp")?;
    let steps_path = output_dir.join("steps.npy");
    std::fs::rename(&steps_tmp, &steps_path)?;

    // Write metadata.db
    let db_path = output_dir.join("metadata.db");
    create_metadata_db(&db_path, &run_rows)?;

    Ok(BuildReport { runs: run_rows.len(), steps: steps.len() })
}

/// Build the exact dtype we want for the row struct.
fn step_row_dtype() -> DType {
    // Matches: [('board','<u8'),('move','u1'),('ev_legal','u1'),('ev_values','<f4',(4,)),('run_id','<u4'),('step_index','<u2')]
    let u8_le: TypeStr = "<u8".parse().unwrap();
    let u4_le: TypeStr = "<u4".parse().unwrap();
    let u2_le: TypeStr = "<u2".parse().unwrap();
    let u1: TypeStr = "|u1".parse().unwrap(); // endianness irrelevant for 1-byte types
    let f4_le: TypeStr = "<f4".parse().unwrap();

    DType::Record(vec![
        Field {
            name: "board".into(),
            dtype: DType::Plain(u8_le),
        },
        Field {
            name: "move".into(),
            dtype: DType::Plain(u1.clone()),
        },
        Field {
            name: "ev_legal".into(),
            dtype: DType::Plain(u1),
        },
        Field {
            name: "ev_values".into(),
            dtype: DType::Array(4, Box::new(DType::Plain(f4_le))),
        },
        Field {
            name: "run_id".into(),
            dtype: DType::Plain(u4_le),
        },
        Field {
            name: "step_index".into(),
            dtype: DType::Plain(u2_le),
        },
    ])
}

fn write_steps_npy(steps: &[StepRow], path: &Path) -> anyhow::Result<()> {
    let dtype = step_row_dtype();
    let file = BufWriter::new(File::create(path)?);
    let mut writer = npyz::WriteOptions::new()
        .dtype(dtype)
        .shape(&[steps.len() as u64])
        .writer(file)
        .begin_nd()?;

    // `extend` takes owned rows; `StepRow` is `Copy`, so `copied()` is fine.
    writer.extend(steps.iter().copied())?;
    writer.finish()?;
    Ok(())
}

/// Append new runs from `new_runs_dir` to an existing dataset dir.
/// Performs an atomic rewrite of `steps.npy` and inserts new rows into `metadata.db`.
pub fn append_runs(dataset_dir: &Path, new_runs_dir: &Path) -> anyhow::Result<BuildReport> {
    anyhow::ensure!(dataset_dir.is_dir(), "dataset must be a directory");
    anyhow::ensure!(new_runs_dir.is_dir(), "input must be a directory");

    let steps_path = dataset_dir.join("steps.npy");
    let db_path = dataset_dir.join("metadata.db");
    anyhow::ensure!(steps_path.exists(), "missing steps.npy in dataset");
    anyhow::ensure!(db_path.exists(), "missing metadata.db in dataset");

    // Compute existing rows by reading header shape.
    let old_rows = npy_row_count(&steps_path)?;

    // Load and flatten new runs (reuse builder logic)
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    for e in WalkDir::new(new_runs_dir)
        .into_iter()
        .filter_map(Result::ok)
    {
        if e.file_type().is_file() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("a2run2") {
                files.push(p.to_path_buf());
            }
        }
    }
    files.sort();
    anyhow::ensure!(!files.is_empty(), "no .a2run2 files found");

    let runs: Vec<RunV2> = files
        .par_iter()
        .map(|p| ser::read_postcard_from_path(p).map_err(|e| anyhow::anyhow!(e)))
        .collect::<Result<Vec<_>, _>>()?;

    let new_rows_count: usize = runs.iter().map(|r| r.meta.steps as usize).sum();
    let mut steps: Vec<StepRow> = Vec::with_capacity(new_rows_count);
    let mut run_rows: Vec<RunRow> = Vec::with_capacity(runs.len());

    let mut next_first = 0u32; // relative to the new segment
    for (run_idx, run) in runs.iter().enumerate() {
        let rid = run_idx as u32;
        let mut ev_tmp = [0.0f32; 4];
        for (si, s) in run.steps.iter().enumerate() {
            let dir_idx = match s.chosen {
                crate::engine::Move::Up => 0u8,
                crate::engine::Move::Down => 1u8,
                crate::engine::Move::Left => 2u8,
                crate::engine::Move::Right => 3u8,
            };
            let branches = s.branches.as_ref().ok_or_else(|| {
                anyhow::anyhow!("run {} missing branches; required for dataset", run_idx)
            })?;
            let mut mask: u8 = 0;
            for (i, b) in branches.iter().enumerate() {
                match *b {
                    BranchV2::Illegal => {
                        ev_tmp[i] = 0.0;
                    }
                    BranchV2::Legal(v) => {
                        mask |= 1u8 << (i as u8);
                        ev_tmp[i] = if i as u8 == dir_idx {
                            1.0
                        } else {
                            v.max(0.0).min(1.0)
                        };
                    }
                }
            }
            steps.push(StepRow {
                board: s.pre_board,
                r#move: dir_idx,
                ev_legal: mask,
                ev_values: ev_tmp,
                run_id: rid,
                step_index: si as u16,
            });
        }
        run_rows.push(RunRow {
            id: rid,
            first_step_idx: next_first,
            num_steps: run.meta.steps,
            max_score: run.meta.max_score,
            highest_tile: run.meta.highest_tile,
            engine: run.meta.engine_str.clone().unwrap_or_default(),
            start_time: run.meta.start_unix_s,
            elapsed_s: run.meta.elapsed_s,
        });
        next_first = next_first.saturating_add(run.meta.steps);
    }

    // Rewrite steps.npy atomically by reading old rows, concatenating, and writing structured records
    rewrite_npy_with_append(&steps_path, &steps, old_rows)?;

    // Update metadata.db (id offset = max id + 1; first_step offset = old_rows)
    let conn = Connection::open(&db_path)?;
    let (max_id, _total_steps_existing): (i64, i64) = {
        let mut stmt =
            conn.prepare("SELECT COALESCE(MAX(id), -1), COALESCE(SUM(num_steps), 0) FROM runs")?;
        stmt.query_row([], |r| Ok((r.get(0)?, r.get(1)?)))?
    };
    let id_offset = (max_id + 1).max(0) as u32;
    let first_step_off = old_rows as u32;
    let mut stmt = conn.prepare(
        "INSERT INTO runs (id, first_step_idx, num_steps, max_score, highest_tile, engine, start_time, elapsed_s) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
    )?;
    for r in &run_rows {
        stmt.execute(params![
            (r.id + id_offset) as i64,
            (r.first_step_idx + first_step_off) as i64,
            r.num_steps as i64,
            r.max_score as i64,
            r.highest_tile as i64,
            r.engine,
            r.start_time as i64,
            r.elapsed_s as f64,
        ])?;
    }

    Ok(BuildReport {
        runs: run_rows.len(),
        steps: steps.len(),
    })
}

pub fn npy_row_count(path: &Path) -> anyhow::Result<usize> {
    // Use `npyz` header to find shape and compute the row count.
    let mut r = BufReader::new(File::open(path)?);
    let hdr = npyz::NpyHeader::from_reader(&mut r)?;
    let shape = hdr.shape();
    anyhow::ensure!(!shape.is_empty(), "npy has zero-dimensional shape");
    Ok(shape.iter().product::<u64>() as usize)
}

fn rewrite_npy_with_append(
    path: &Path,
    new_steps: &[StepRow],
    _old_rows: usize,
) -> anyhow::Result<()> {
    // Read existing rows
    let mut reader = BufReader::new(File::open(path)?);
    let npy = npyz::NpyFile::new(&mut reader)?;
    let mut existing: Vec<StepRow> = npy
        .into_vec()
        .context("decoding existing steps.npy into StepRow")?;

    existing.extend_from_slice(new_steps);

    // Overwrite atomically
    let tmp = path.with_extension("npy.tmp");
    write_steps_npy(&existing, &tmp)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::{tempdir, NamedTempFile};

    #[test]
    fn npy_header_and_rows() {
        let steps = vec![
            StepRow {
                board: 0x0123_4567_89ab_cdef,
                r#move: 2,
                ev_legal: 0b0101,
                ev_values: [1.0, 0.25, 0.0, 0.75],
                run_id: 42,
                step_index: 3,
            },
            StepRow {
                board: 0x0000_0000_0000_0001,
                r#move: 0,
                ev_legal: 0b1111,
                ev_values: [0.0, 0.0, 1.0, 0.0],
                run_id: 42,
                step_index: 4,
            },
        ];
        let tmp = NamedTempFile::new().unwrap();
        write_steps_npy(&steps, tmp.path()).unwrap();

        // Parse header and verify dtype & shape
        let mut r = BufReader::new(File::open(tmp.path()).unwrap());
        let hdr = npyz::NpyHeader::from_reader(&mut r).unwrap();
        let dt = hdr.dtype();
        match dt {
            DType::Record(fields) => {
                let names: Vec<_> = fields.iter().map(|f| f.name.as_str()).collect();
                assert_eq!(
                    names,
                    [
                        "board",
                        "move",
                        "ev_legal",
                        "ev_values",
                        "run_id",
                        "step_index"
                    ]
                );
                // spot-check a few field dtypes
                assert!(matches!(fields[0].dtype, DType::Plain(_))); // board <u8
                assert!(matches!(fields[1].dtype, DType::Plain(_))); // move u1
                assert!(matches!(fields[2].dtype, DType::Plain(_))); // ev_legal u1
                assert!(matches!(fields[3].dtype, DType::Array(4, _))); // ev_values <f4,(4,)
                assert!(matches!(fields[4].dtype, DType::Plain(_))); // run_id <u4
                assert!(matches!(fields[5].dtype, DType::Plain(_))); // step_index <u2
            }
            _ => panic!("expected record dtype"),
        }
        assert_eq!(hdr.shape(), &[2]);

        // Decode rows and sanity-check fields
        let mut r2 = BufReader::new(File::open(tmp.path()).unwrap());
        let npy = npyz::NpyFile::new(&mut r2).unwrap();
        let decoded: Vec<StepRow> = npy.into_vec().unwrap();
        assert_eq!(decoded.len(), steps.len());
        assert_eq!(decoded[0].board, steps[0].board);
        assert_eq!(decoded[0].r#move, steps[0].r#move);
        assert_eq!(decoded[0].ev_legal, steps[0].ev_legal);
        assert!((decoded[0].ev_values[1] - steps[0].ev_values[1]).abs() < 1e-6);
        assert_eq!(decoded[0].run_id, steps[0].run_id);
        assert_eq!(decoded[0].step_index, steps[0].step_index);
    }

    #[test]
    fn build_dataset_small() {
        use crate::engine::Move;
        use crate::serialization::{write_postcard_to_path, RunV2, StepV2};
        use crate::trace::Meta;

        let dir = tempdir().unwrap();
        let runs_dir = dir.path().join("runs");
        std::fs::create_dir_all(&runs_dir).unwrap();

        // Create a tiny v2 run with branches
        let meta = Meta {
            steps: 2,
            start_unix_s: 1_700_000_000,
            elapsed_s: 1.0,
            max_score: 100,
            highest_tile: 8,
            engine_str: Some("test".into()),
        };
        let steps_v2 = vec![
            StepV2 {
                pre_board: 1,
                chosen: Move::Right,
                branches: Some([
                    BranchV2::Legal(0.2),
                    BranchV2::Illegal,
                    BranchV2::Illegal,
                    BranchV2::Legal(0.9),
                ]),
            },
            StepV2 {
                pre_board: 2,
                chosen: Move::Up,
                branches: Some([
                    BranchV2::Legal(0.6),
                    BranchV2::Legal(0.5),
                    BranchV2::Illegal,
                    BranchV2::Illegal,
                ]),
            },
        ];
        let run = RunV2 {
            meta,
            steps: steps_v2,
            final_board: 3,
        };
        let run_path = runs_dir.join("run-000000001.a2run2");
        write_postcard_to_path(&run_path, &run).unwrap();

        let out_dir = dir.path().join("out");
        let rep = build_dataset(&runs_dir, &out_dir).unwrap();
        assert_eq!(rep.runs, 1);
        assert_eq!(rep.steps, 2);
        assert!(out_dir.join("steps.npy").exists());
        assert!(out_dir.join("metadata.db").exists());

        // Count rows using the new helper
        let n = npy_row_count(&out_dir.join("steps.npy")).unwrap();
        assert_eq!(n, 2);
    }
}

fn create_metadata_db(path: &Path, runs: &[impl RunLike]) -> anyhow::Result<()> {
    let conn = Connection::open(path)?;
    conn.execute_batch(
        r#"
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY,
            first_step_idx INTEGER NOT NULL,
            num_steps INTEGER NOT NULL,
            max_score INTEGER NOT NULL,
            highest_tile INTEGER NOT NULL,
            engine TEXT,
            start_time INTEGER,
            elapsed_s REAL
        );
        CREATE INDEX IF NOT EXISTS idx_runs_score ON runs(max_score);
        CREATE INDEX IF NOT EXISTS idx_runs_len ON runs(num_steps);
        "#,
    )?;
    let mut stmt = conn.prepare(
        "INSERT INTO runs (id, first_step_idx, num_steps, max_score, highest_tile, engine, start_time, elapsed_s)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
    )?;
    for r in runs {
        stmt.execute(params![
            r.id() as i64,
            r.first_step_idx() as i64,
            r.num_steps() as i64,
            r.max_score() as i64,
            r.highest_tile() as i64,
            r.engine(),
            r.start_time() as i64,
            r.elapsed_s() as f64,
        ])?;
    }
    Ok(())
}

trait RunLike {
    fn id(&self) -> u32;
    fn first_step_idx(&self) -> u32;
    fn num_steps(&self) -> u32;
    fn max_score(&self) -> u64;
    fn highest_tile(&self) -> u32;
    fn engine(&self) -> &str;
    fn start_time(&self) -> u64;
    fn elapsed_s(&self) -> f32;
}

impl RunLike for RunRow {
    fn id(&self) -> u32 {
        self.id
    }
    fn first_step_idx(&self) -> u32 {
        self.first_step_idx
    }
    fn num_steps(&self) -> u32 {
        self.num_steps
    }
    fn max_score(&self) -> u64 {
        self.max_score
    }
    fn highest_tile(&self) -> u32 {
        self.highest_tile
    }
    fn engine(&self) -> &str {
        &self.engine
    }
    fn start_time(&self) -> u64 {
        self.start_time
    }
    fn elapsed_s(&self) -> f32 {
        self.elapsed_s
    }
}
