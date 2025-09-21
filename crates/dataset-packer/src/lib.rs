#![allow(unexpected_cfgs, non_local_definitions)]

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, warn};
use npyz::{DType, Field, TypeStr, WriterBuilder, NpyWriter};
use rayon::prelude::*;
use rusqlite::{Connection, params};
use serde::Deserialize;

/// Packing configuration supplied by the CLI.
#[derive(Clone, Debug)]
pub struct PackOptions {
    pub input_root: PathBuf,
    pub output_dir: PathBuf,
    pub rows_per_shard: Option<usize>,
    pub max_workers: Option<usize>,
    pub overwrite: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackSummary {
    pub runs: usize,
    pub steps: usize,
    pub shards: usize,
}

#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize,
)]
pub struct StepRow {
    pub run_id: u32,
    pub step_index: u32,
    pub board: u64,
    pub tile_65536_mask: u16,
    pub move_dir: u8,
    pub valuation_type: u8,
    pub ev_legal: u8,
    pub max_rank: u8,
    pub seed: u32,
    pub branch_evs: [f32; 4],
}

impl StepRow {
    fn dtype() -> DType {
        let u1: TypeStr = "<u1".parse().unwrap();
        let u2: TypeStr = "<u2".parse().unwrap();
        let u4: TypeStr = "<u4".parse().unwrap();
        let u8: TypeStr = "<u8".parse().unwrap();
        let f4: TypeStr = "<f4".parse().unwrap();
        DType::Record(vec![
            Field {
                name: "run_id".into(),
                dtype: DType::Plain(u4.clone()),
            },
            Field {
                name: "step_index".into(),
                dtype: DType::Plain(u4.clone()),
            },
            Field {
                name: "board".into(),
                dtype: DType::Plain(u8),
            },
            Field {
                name: "tile_65536_mask".into(),
                dtype: DType::Plain(u2.clone()),
            },
            Field {
                name: "move_dir".into(),
                dtype: DType::Plain(u1.clone()),
            },
            Field {
                name: "valuation_type".into(),
                dtype: DType::Plain(u1.clone()),
            },
            Field {
                name: "ev_legal".into(),
                dtype: DType::Plain(u1.clone()),
            },
            Field {
                name: "max_rank".into(),
                dtype: DType::Plain(u1),
            },
            Field {
                name: "seed".into(),
                dtype: DType::Plain(u4),
            },
            Field {
                name: "branch_evs".into(),
                dtype: DType::Array(4, Box::new(DType::Plain(f4))),
            },
        ])
    }
}

#[derive(Debug, Clone)]
struct RunInput {
    meta_path: PathBuf,
    steps_path: PathBuf,
}

#[derive(Debug, Clone)]
struct RunOutput {
    summary: RunSummary,
    steps: Vec<StepRow>,
}

#[derive(Debug, Clone)]
struct RunSummary {
    run_id: u32,
    seed: u64,
    steps: usize,
    max_score: u64,
    highest_tile: u32,
}

#[derive(Debug, Deserialize)]
struct MetaRecord {
    seed: u64,
    num_moves: usize,
    score: u64,
    #[serde(rename = "max_tile")]
    highest_tile: u32,
    #[serde(default)]
    max_rank: Option<u8>,
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
    board: Vec<u8>,
    #[serde(default)]
    max_rank: Option<u8>,
    #[serde(default)]
    branch_evs: HashMap<String, Option<f32>>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum MoveName {
    Up,
    Right,
    Down,
    Left,
}

impl MoveName {
    fn code(self) -> u8 {
        match self {
            MoveName::Up => 0,
            MoveName::Right => 1,
            MoveName::Down => 2,
            MoveName::Left => 3,
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

#[derive(Default)]
struct ValuationEncoder {
    inner: parking_lot::Mutex<ValuationInner>,
}

#[derive(Default)]
struct ValuationInner {
    map: HashMap<String, u8>,
    names: Vec<String>,
}

impl ValuationEncoder {
    fn new() -> Self {
        let mut inner = ValuationInner::default();
        for name in ["search", "tuple11", "tuple10"] {
            inner.register(name);
        }
        Self {
            inner: parking_lot::Mutex::new(inner),
        }
    }

    fn encode(&self, name: &str) -> Result<u8> {
        let mut inner = self.inner.lock();
        if let Some(&id) = inner.map.get(name) {
            return Ok(id);
        }
        let id = inner.register(name);
        Ok(id)
    }

    fn as_vec(&self) -> Vec<String> {
        self.inner.lock().names.clone()
    }
}

impl ValuationInner {
    fn register(&mut self, name: &str) -> u8 {
        if let Some(&id) = self.map.get(name) {
            return id;
        }
        let id = self.names.len();
        if id >= u8::MAX as usize {
            panic!("valuation enum overflows u8");
        }
        self.names.push(name.to_string());
        self.map.insert(name.to_string(), id as u8);
        id as u8
    }
}

pub fn pack_dataset(opts: PackOptions) -> Result<PackSummary> {
    if opts.rows_per_shard == Some(0) {
        bail!("rows_per_shard must be > 0 when specified");
    }
    if !opts.input_root.exists() {
        bail!(
            "input directory '{}' does not exist",
            opts.input_root.display()
        );
    }
    fs::create_dir_all(&opts.output_dir)
        .with_context(|| format!("failed to create output dir {}", opts.output_dir.display()))?;

    let runs = discover_runs(&opts.input_root)?;
    if runs.is_empty() {
        bail!(
            "no .meta.json files found under {}",
            opts.input_root.display()
        );
    }

    info!("Discovered {} runs", runs.len());
    let pb = ProgressBar::new(runs.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] {wide_bar} {pos}/{len} ({eta})",
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  "),
    );

    let encoder = Arc::new(ValuationEncoder::new());

    let process = || -> Result<Vec<RunOutput>> {
        runs.into_par_iter()
            .enumerate()
            .map(|(idx, run)| {
                let enc = encoder.clone();
                let out = process_run(run, idx as u32, &enc);
                pb.inc(1);
                out
            })
            .collect()
    };

    let mut outputs: Vec<RunOutput> = if let Some(n) = opts.max_workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .context("failed to build rayon thread pool")?
            .install(process)?
    } else {
        process()?
    };

    pb.finish_with_message("runs processed");

    outputs.sort_by_key(|out| out.summary.run_id);

    let total_steps: usize = outputs.iter().map(|out| out.steps.len()).sum();
    let run_summaries: Vec<RunSummary> = outputs.iter().map(|out| out.summary.clone()).collect();

    info!(
        "Writing {} steps across {} runs",
        total_steps,
        run_summaries.len()
    );

    let shards = write_steps_stream(
        &mut outputs,
        &opts.output_dir,
        opts.rows_per_shard,
        opts.overwrite,
        total_steps,
    )?;
    write_metadata(&opts.output_dir, &run_summaries, opts.overwrite)?;
    write_valuation_types(&opts.output_dir, &encoder.as_vec(), opts.overwrite)?;

    Ok(PackSummary {
        runs: run_summaries.len(),
        steps: total_steps,
        shards,
    })
}

fn discover_runs(root: &Path) -> Result<Vec<RunInput>> {
    let mut runs = Vec::new();
    for entry in walkdir::WalkDir::new(root) {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let name = match path.file_name().and_then(|s| s.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if !name.ends_with(".meta.json") {
            continue;
        }
        let steps_path = candidate_steps_path(path);
        if !steps_path.is_file() {
            warn!(
                "Skipping meta file {} because {} is missing",
                path.display(),
                steps_path.display()
            );
            continue;
        }
        runs.push(RunInput {
            meta_path: path.to_path_buf(),
            steps_path,
        });
    }
    runs.sort_by(|a, b| a.meta_path.cmp(&b.meta_path));
    Ok(runs)
}

fn candidate_steps_path(meta_path: &Path) -> PathBuf {
    let file_name = meta_path.file_name().and_then(|s| s.to_str()).unwrap();
    let base = file_name.strip_suffix(".meta.json").unwrap_or(file_name);
    let mut steps = meta_path.to_path_buf();
    steps.set_file_name(format!("{base}.jsonl.gz"));
    steps
}

fn process_run(run: RunInput, run_id: u32, encoder: &ValuationEncoder) -> Result<RunOutput> {
    let meta_content = fs::read_to_string(&run.meta_path)
        .with_context(|| format!("failed to read {}", run.meta_path.display()))?;
    let meta: MetaRecord = serde_json::from_str(&meta_content)
        .with_context(|| format!("failed to parse {}", run.meta_path.display()))?;

    let (steps, observed_moves) = parse_steps_file(&run.steps_path, run_id, &meta, encoder)?;
    if steps.is_empty() {
        warn!(
            "Run {} produced zero steps (meta path {})",
            run_id,
            run.meta_path.display()
        );
    }
    if steps.len() != meta.num_moves {
        warn!(
            "Run {}: step count mismatch meta {} vs parsed {}",
            run_id,
            meta.num_moves,
            steps.len()
        );
    }

    if !observed_moves {
        warn!(
            "Run {} did not record branch EVs with any legal moves; check input at {}",
            run_id,
            run.steps_path.display()
        );
    }

    Ok(RunOutput {
        summary: RunSummary {
            run_id,
            seed: meta.seed,
            steps: steps.len(),
            max_score: meta.score,
            highest_tile: meta.highest_tile,
        },
        steps,
    })
}

fn parse_steps_file(
    steps_path: &Path,
    run_id: u32,
    meta: &MetaRecord,
    encoder: &ValuationEncoder,
) -> Result<(Vec<StepRow>, bool)> {
    let file = File::open(steps_path)
        .with_context(|| format!("failed to open {}", steps_path.display()))?;
    let gz = GzDecoder::new(file);
    let reader = BufReader::new(gz);
    let mut rows = Vec::new();
    let mut any_legal = false;
    for (line_idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read line {}", line_idx + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: StepRecord = serde_json::from_str(&line).with_context(|| {
            format!(
                "failed to parse step JSON in {} at line {}",
                steps_path.display(),
                line_idx + 1
            )
        })?;
        let arr: [u8; 16] = record
            .board
            .try_into()
            .map_err(|_| anyhow!("expected 16 board entries"))?;
        let (board, mask) = pack_board(&arr);
        let valuation_name = record.valuation_type.as_deref().unwrap_or("search");
        let valuation_type = encoder.encode(valuation_name)?;
        let step_index = record.step_index.unwrap_or(rows.len() as u32);
        let seed = record.seed.unwrap_or(meta.seed);
        let seed32 =
            u32::try_from(seed).with_context(|| format!("seed {} does not fit into u32", seed))?;
        let (branch_evs, legal_mask) = gather_branch_evs(&record.branch_evs);
        if legal_mask != 0 {
            any_legal = true;
        }
        let step = StepRow {
            run_id,
            step_index,
            board,
            tile_65536_mask: mask,
            move_dir: record.move_dir.code(),
            valuation_type,
            ev_legal: legal_mask,
            max_rank: record.max_rank.or(meta.max_rank).unwrap_or_default(),
            seed: seed32,
            branch_evs,
        };
        rows.push(step);
    }
    Ok((rows, any_legal))
}

fn pack_board(cells: &[u8; 16]) -> (u64, u16) {
    let mut acc = 0u64;
    let mut mask = 0u16;
    for (idx, &exp) in cells.iter().enumerate() {
        let mut nibble = exp;
        if nibble >= 16 {
            mask |= 1 << idx;
            nibble = 15;
        }
        acc |= (nibble as u64 & 0xF) << (idx * 4);
    }
    (acc, mask)
}

fn gather_branch_evs(map: &HashMap<String, Option<f32>>) -> ([f32; 4], u8) {
    const ORDER: [&str; 4] = ["up", "right", "down", "left"];
    let mut evs = [0.0f32; 4];
    let mut mask = 0u8;
    for (idx, key) in ORDER.iter().enumerate() {
        if let Some(Some(val)) = map.get(*key) {
            evs[idx] = *val;
            mask |= 1 << idx;
        }
    }
    (evs, mask)
}

fn write_steps_stream(
    outputs: &mut [RunOutput],
    out_dir: &Path,
    rows_per_shard: Option<usize>,
    overwrite: bool,
    total_steps: usize,
) -> Result<usize> {
    if total_steps == 0 {
        warn!("No steps to write; skipping steps.npy");
        return Ok(0);
    }
    let per_shard = rows_per_shard.unwrap_or(total_steps);
    if per_shard == 0 {
        bail!("rows_per_shard must be > 0");
    }
    let shard_count = (total_steps + per_shard - 1) / per_shard;
    let mut shard_idx = 0usize;
    let mut remaining_total = total_steps;

    let mut active: Option<ShardWriter> = None;
    let chunk_cap = per_shard.min(65_536).max(1_024);
    let mut chunk: Vec<StepRow> = Vec::with_capacity(chunk_cap);

    let flush_chunk = |chunk: &mut Vec<StepRow>,
                           active: &mut Option<ShardWriter>,
                           shard_idx: &mut usize,
                           remaining_total: &mut usize|
     -> Result<()> {
        let mut start = 0usize;
        while start < chunk.len() {
            if active.is_none() {
                if *shard_idx >= shard_count {
                    bail!("exceeded expected shard count while writing steps.npy");
                }
                let rows_this_shard = if shard_count == 1 {
                    *remaining_total
                } else if *shard_idx == shard_count - 1 {
                    *remaining_total
                } else {
                    per_shard
                };
                *active = Some(ShardWriter::new(
                    out_dir,
                    *shard_idx,
                    shard_count,
                    rows_this_shard,
                    overwrite,
                )?);
            }

            let writer = active.as_mut().expect("active shard must exist");
            let take = writer.remaining.min(chunk.len() - start);
            let slice = &chunk[start..start + take];
            writer.write(slice)?;
            start += take;
            *remaining_total = remaining_total.saturating_sub(take);

            if writer.is_complete() {
                let finished = active.take().unwrap();
                finished.finish()?;
                *shard_idx += 1;
            }
        }
        chunk.clear();
        Ok(())
    };

    for out in outputs.iter_mut() {
        for step in out.steps.drain(..) {
            chunk.push(step);
            if chunk.len() >= chunk_cap {
                flush_chunk(
                    &mut chunk,
                    &mut active,
                    &mut shard_idx,
                    &mut remaining_total,
                )?;
            }
        }
    }

    if !chunk.is_empty() {
        flush_chunk(
            &mut chunk,
            &mut active,
            &mut shard_idx,
            &mut remaining_total,
        )?;
    }

    if let Some(writer) = active.take() {
        if !writer.is_complete() {
            return Err(anyhow!(
                "final shard incomplete: {} rows remaining",
                writer.remaining
            ));
        }
        writer.finish()?;
        shard_idx += 1;
    }

    if shard_idx != shard_count {
        return Err(anyhow!(
            "expected {} shard(s) but wrote {}",
            shard_count,
            shard_idx
        ));
    }

    Ok(shard_count)
}

struct ShardWriter {
    writer: NpyWriter<StepRow, BufWriter<File>>,
    tmp_path: PathBuf,
    final_path: PathBuf,
    remaining: usize,
}

impl ShardWriter {
    fn new(
        out_dir: &Path,
        shard_idx: usize,
        total_shards: usize,
        rows: usize,
        overwrite: bool,
    ) -> Result<Self> {
        let final_name = if total_shards == 1 {
            "steps.npy".to_string()
        } else {
            format!("steps-{shard_idx:05}.npy")
        };
        let final_path = out_dir.join(final_name);
        if final_path.exists() {
            if overwrite {
                fs::remove_file(&final_path)
                    .with_context(|| format!("failed to remove {}", final_path.display()))?;
            } else {
                bail!(
                    "steps file {} exists (use --overwrite)",
                    final_path.display()
                );
            }
        }
        let tmp_path = final_path.with_extension("npy.tmp");
        let file = File::create(&tmp_path)
            .with_context(|| format!("failed to create {}", tmp_path.display()))?;
        let writer = npyz::WriteOptions::new()
            .dtype(StepRow::dtype())
            .shape(&[rows as u64])
            .writer(BufWriter::new(file))
            .begin_nd()?;
        Ok(Self {
            writer,
            tmp_path,
            final_path,
            remaining: rows,
        })
    }

    fn write(&mut self, rows: &[StepRow]) -> Result<()> {
        if rows.len() > self.remaining {
            bail!(
                "attempted to write {} rows but only {} remaining in shard",
                rows.len(),
                self.remaining
            );
        }
        self.writer.extend(rows.iter().copied())?;
        self.remaining -= rows.len();
        Ok(())
    }

    fn is_complete(&self) -> bool {
        self.remaining == 0
    }

    fn finish(self) -> Result<()> {
        let ShardWriter {
            writer,
            tmp_path,
            final_path,
            ..
        } = self;
        writer.finish()?;
        fs::rename(&tmp_path, &final_path).with_context(|| {
            format!(
                "failed to rename {} -> {}",
                tmp_path.display(),
                final_path.display()
            )
        })?;
        Ok(())
    }
}

fn write_metadata(out_dir: &Path, runs: &[RunSummary], overwrite: bool) -> Result<()> {
    let path = out_dir.join("metadata.db");
    if path.exists() {
        if overwrite {
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
        } else {
            bail!("metadata.db already exists (use --overwrite)");
        }
    }
    let mut conn =
        Connection::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
    conn.pragma_update(None, "journal_mode", &"WAL")?;
    conn.pragma_update(None, "synchronous", &"NORMAL")?;
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY,
            seed BIGINT NOT NULL,
            steps INT NOT NULL,
            max_score INT NOT NULL,
            highest_tile INT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS session (
            meta_key TEXT PRIMARY KEY,
            meta_value TEXT NOT NULL
        );
        ",
    )?;
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO runs (id, seed, steps, max_score, highest_tile) VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        for run in runs {
            stmt.execute(params![
                run.run_id as i64,
                run.seed as i64,
                run.steps as i64,
                run.max_score as i64,
                run.highest_tile as i64
            ])?;
        }
    }
    tx.commit()?;
    info!("Wrote metadata.db with {} runs", runs.len());
    Ok(())
}

fn write_valuation_types(out_dir: &Path, names: &[String], overwrite: bool) -> Result<()> {
    let path = out_dir.join("valuation_types.json");
    if path.exists() {
        if overwrite {
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
        } else {
            bail!("valuation_types.json already exists (use --overwrite)");
        }
    }
    let mut file =
        File::create(&path).with_context(|| format!("failed to create {}", path.display()))?;
    let mapping: HashMap<&str, u8> = names
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.as_str(), idx as u8))
        .collect();
    let json = serde_json::to_string_pretty(&mapping)?;
    file.write_all(json.as_bytes())?;
    file.write_all(b"\n")?;
    info!("Wrote valuation_types.json with {} entries", names.len());
    Ok(())
}
