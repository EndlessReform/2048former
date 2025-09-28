#![allow(unexpected_cfgs, non_local_definitions)]

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, warn};
use npyz::{DType, Field, NpyWriter, TypeStr, WriterBuilder};
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

#[derive(Clone, Debug)]
pub struct MergeOptions {
    pub left_dir: PathBuf,
    pub right_dir: PathBuf,
    pub output_dir: PathBuf,
    pub rows_per_shard: Option<usize>,
    pub overwrite: bool,
    pub delete_inputs: bool,
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

fn is_meta_filename(name: &str) -> bool {
    name.ends_with(".meta.json") || name.ends_with(".meta.json.gz")
}

fn meta_base_name(name: &str) -> Option<&str> {
    if let Some(stem) = name.strip_suffix(".meta.json") {
        Some(stem)
    } else if let Some(stem) = name.strip_suffix(".meta.json.gz") {
        Some(stem)
    } else {
        None
    }
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

#[derive(Debug)]
struct DatasetInfo {
    runs: Vec<RunSummary>,
    steps_files: Vec<PathBuf>,
    valuation_names: Vec<String>,
    total_steps: usize,
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
    let pb = default_progress_bar(runs.len() as u64);

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
        let (rows, any_legal) = parse_steps_file(&steps_path, 1u32, &meta, &enc_v).unwrap();
        assert!(
            any_legal,
            "expected at least one legal branch in test record"
        );
        assert_eq!(rows.len(), 1);
        let row = &rows[0];

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
    }
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
        if !is_meta_filename(name) {
            continue;
        }
        let Some(steps_path) = candidate_steps_path(path) else {
            continue;
        };
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

fn candidate_steps_path(meta_path: &Path) -> Option<PathBuf> {
    let file_name = meta_path.file_name().and_then(|s| s.to_str())?;
    let base = meta_base_name(file_name)?;
    let mut steps = meta_path.to_path_buf();
    steps.set_file_name(format!("{base}.jsonl.gz"));
    Some(steps)
}

fn read_json_text(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    if path
        .extension()
        .and_then(|s| s.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("gz"))
        .unwrap_or(false)
    {
        let mut gz = GzDecoder::new(file);
        let mut buf = String::new();
        gz.read_to_string(&mut buf)
            .with_context(|| format!("failed to decompress {}", path.display()))?;
        Ok(buf)
    } else {
        let mut reader = BufReader::new(file);
        let mut buf = String::new();
        reader
            .read_to_string(&mut buf)
            .with_context(|| format!("failed to read {}", path.display()))?;
        Ok(buf)
    }
}

fn process_run(run: RunInput, run_id: u32, encoder: &ValuationEncoder) -> Result<RunOutput> {
    let meta_content = read_json_text(&run.meta_path)?;
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
    // Canonical MSB-first nibble order: cell 0 -> bits 63..60, ..., cell 15 -> bits 3..0
    let mut acc = 0u64;
    let mut mask = 0u16;
    for (idx, &exp) in cells.iter().enumerate() {
        let mut nibble = exp;
        if nibble >= 16 {
            mask |= 1 << idx;
            nibble = 15;
        }
        let shift = (15 - idx) * 4;
        acc |= (nibble as u64 & 0xF) << shift;
    }
    (acc, mask)
}

fn gather_branch_evs(map: &HashMap<String, Option<f32>>) -> ([f32; 4], u8) {
    // Canonical branch order: UDLR (Up, Down, Left, Right)
    const ORDER: [&str; 4] = ["up", "down", "left", "right"];
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

fn default_progress_bar(len: u64) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] {wide_bar} {pos}/{len} ({eta})",
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  "),
    );
    pb
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
    if let Some(limit) = rows_per_shard {
        if limit == 0 {
            bail!("rows_per_shard must be > 0 when specified");
        }
    }

    let mut writer = StepsWriter::new(out_dir, rows_per_shard, overwrite)?;
    for out in outputs.iter_mut() {
        if !out.steps.is_empty() {
            writer.write(&out.steps)?;
            out.steps.clear();
        }
    }
    writer.finish()
}

struct StepsWriter {
    out_dir: PathBuf,
    rows_per_shard: Option<usize>,
    overwrite: bool,
    shard_idx: usize,
    shards_written: usize,
    current: Option<ShardWriter>,
    cleaned: bool,
}

impl StepsWriter {
    fn new(out_dir: &Path, rows_per_shard: Option<usize>, overwrite: bool) -> Result<Self> {
        Ok(Self {
            out_dir: out_dir.to_path_buf(),
            rows_per_shard,
            overwrite,
            shard_idx: 0,
            shards_written: 0,
            current: None,
            cleaned: false,
        })
    }

    fn prepare(&mut self) -> Result<()> {
        if self.cleaned {
            return Ok(());
        }
        if self.overwrite {
            let remove = |path: PathBuf| -> Result<()> {
                if path.exists() {
                    fs::remove_file(&path)
                        .with_context(|| format!("failed to remove {}", path.display()))?;
                }
                Ok(())
            };
            remove(self.out_dir.join("steps.npy"))?;
            for entry in fs::read_dir(&self.out_dir)
                .with_context(|| format!("failed to read {}", self.out_dir.display()))?
            {
                let entry = entry?;
                let name = entry.file_name();
                if let Some(name_str) = name.to_str() {
                    if name_str.starts_with("steps-") && name_str.ends_with(".npy") {
                        fs::remove_file(entry.path()).with_context(|| {
                            format!("failed to remove {}", entry.path().display())
                        })?;
                    }
                }
            }
        } else {
            if self.out_dir.join("steps.npy").exists() {
                bail!(
                    "steps.npy already exists in {} (use --overwrite)",
                    self.out_dir.display()
                );
            }
            let numbered_exists = fs::read_dir(&self.out_dir)
                .with_context(|| format!("failed to read {}", self.out_dir.display()))?
                .filter_map(|e| e.ok())
                .any(|entry| {
                    entry
                        .file_name()
                        .to_str()
                        .map(|n| n.starts_with("steps-") && n.ends_with(".npy"))
                        .unwrap_or(false)
                });
            if numbered_exists {
                bail!(
                    "found existing steps-*.npy in {} (use --overwrite)",
                    self.out_dir.display()
                );
            }
        }
        self.cleaned = true;
        Ok(())
    }

    fn ensure_writer(&mut self) -> Result<()> {
        if self.current.is_some() {
            return Ok(());
        }
        self.prepare()?;
        let shard = ShardWriter::new(&self.out_dir, self.shard_idx, self.rows_per_shard)?;
        self.current = Some(shard);
        Ok(())
    }

    fn write(&mut self, rows: &[StepRow]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut start = 0usize;
        while start < rows.len() {
            self.ensure_writer()?;
            let limit = self.rows_per_shard.unwrap_or(usize::MAX);
            let shard = self.current.as_mut().expect("active shard must exist");
            let written = shard.write(&rows[start..], limit)?;
            if written == 0 {
                bail!("failed to make progress while writing steps.npy");
            }
            start += written;
            if shard.is_full(limit) {
                let shard_state = self.current.take().unwrap();
                shard_state.finish()?;
                self.shards_written += 1;
                self.shard_idx += 1;
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<usize> {
        if let Some(shard) = self.current.take() {
            shard.finish()?;
            self.shards_written += 1;
        }
        if self.shards_written == 0 {
            warn!(
                "No steps were written; no steps.npy produced in {}",
                self.out_dir.display()
            );
        }
        Ok(self.shards_written)
    }
}

struct ShardWriter {
    writer: NpyWriter<StepRow, BufWriter<File>>,
    tmp_path: PathBuf,
    final_path: PathBuf,
    rows_written: usize,
    limit: Option<usize>,
}

impl ShardWriter {
    fn new(out_dir: &Path, shard_idx: usize, limit: Option<usize>) -> Result<Self> {
        let final_name = if limit.is_some() {
            format!("steps-{shard_idx:05}.npy")
        } else {
            "steps.npy".to_string()
        };
        let final_path = out_dir.join(final_name);
        let tmp_path = final_path.with_extension("npy.tmp");
        let file = File::create(&tmp_path)
            .with_context(|| format!("failed to create {}", tmp_path.display()))?;
        let writer = npyz::WriteOptions::new()
            .dtype(StepRow::dtype())
            .writer(BufWriter::new(file))
            .begin_1d()?;
        Ok(Self {
            writer,
            tmp_path,
            final_path,
            rows_written: 0,
            limit,
        })
    }

    fn write(&mut self, rows: &[StepRow], limit: usize) -> Result<usize> {
        let remaining = limit.saturating_sub(self.rows_written);
        let take = if self.limit.is_some() {
            rows.len().min(remaining)
        } else {
            rows.len()
        };
        if take > 0 {
            self.writer.extend(rows[..take].iter().copied())?;
            self.rows_written += take;
        }
        Ok(take)
    }

    fn is_full(&self, limit: usize) -> bool {
        self.limit.is_some() && self.rows_written >= limit
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

pub fn merge_datasets(opts: MergeOptions) -> Result<PackSummary> {
    if opts.rows_per_shard == Some(0) {
        bail!("rows_per_shard must be > 0 when specified");
    }
    if !opts.left_dir.exists() {
        bail!("left dataset '{}' does not exist", opts.left_dir.display());
    }
    if !opts.right_dir.exists() {
        bail!(
            "right dataset '{}' does not exist",
            opts.right_dir.display()
        );
    }
    fs::create_dir_all(&opts.output_dir)
        .with_context(|| format!("failed to create output dir {}", opts.output_dir.display()))?;

    let left = load_dataset_info(&opts.left_dir)?;
    let right = load_dataset_info(&opts.right_dir)?;

    let mut new_runs = Vec::with_capacity(left.runs.len() + right.runs.len());
    let mut left_map = HashMap::new();
    let mut right_map = HashMap::new();
    let mut next_id: u32 = 0;

    for run in &left.runs {
        left_map.insert(run.run_id, next_id);
        let mut updated = run.clone();
        updated.run_id = next_id;
        new_runs.push(updated);
        next_id += 1;
    }
    for run in &right.runs {
        right_map.insert(run.run_id, next_id);
        let mut updated = run.clone();
        updated.run_id = next_id;
        new_runs.push(updated);
        next_id += 1;
    }

    let mut valuation_names = left.valuation_names.clone();
    let mut valuation_ids: HashMap<String, u8> = HashMap::new();
    for (idx, name) in valuation_names.iter().enumerate() {
        valuation_ids.insert(name.clone(), idx as u8);
    }
    for name in right.valuation_names.iter() {
        if !valuation_ids.contains_key(name) {
            let id = valuation_names.len() as u8;
            valuation_names.push(name.clone());
            valuation_ids.insert(name.clone(), id);
        }
    }

    let total_steps = left.total_steps + right.total_steps;

    let mut pb = if total_steps > 0 {
        let pb = default_progress_bar(total_steps as u64);
        pb.set_message("merging steps");
        Some(pb)
    } else {
        None
    };

    let mut writer = StepsWriter::new(&opts.output_dir, opts.rows_per_shard, opts.overwrite)?;
    stream_dataset(&left, &left_map, &valuation_ids, &mut writer, pb.as_ref())?;
    stream_dataset(&right, &right_map, &valuation_ids, &mut writer, pb.as_ref())?;
    let shards = writer.finish()?;

    if let Some(pb) = &pb {
        pb.finish_with_message("merge complete");
    }

    write_metadata(&opts.output_dir, &new_runs, opts.overwrite)?;
    write_valuation_types(&opts.output_dir, &valuation_names, opts.overwrite)?;

    if opts.delete_inputs {
        let output_canon = fs::canonicalize(&opts.output_dir)
            .with_context(|| format!("failed to canonicalize {}", opts.output_dir.display()))?;
        for dir in [&opts.left_dir, &opts.right_dir] {
            let canon = fs::canonicalize(dir)
                .with_context(|| format!("failed to canonicalize {}", dir.display()))?;
            if canon == output_canon {
                bail!(
                    "refusing to delete input '{}' because it matches output directory",
                    dir.display()
                );
            }
        }
        fs::remove_dir_all(&opts.left_dir)
            .with_context(|| format!("failed to remove {}", opts.left_dir.display()))?;
        fs::remove_dir_all(&opts.right_dir)
            .with_context(|| format!("failed to remove {}", opts.right_dir.display()))?;
    }

    Ok(PackSummary {
        runs: new_runs.len(),
        steps: total_steps,
        shards,
    })
}

fn load_dataset_info(dir: &Path) -> Result<DatasetInfo> {
    let steps_files = collect_step_files(dir)?;
    if steps_files.is_empty() {
        bail!("no steps.npy files found in {}", dir.display());
    }
    let runs = load_runs(dir)?;
    if runs.is_empty() {
        bail!("metadata.db in {} has no runs", dir.display());
    }
    let valuation_names = load_valuation_names(dir)?;
    let total_steps = runs.iter().map(|r| r.steps).sum();
    Ok(DatasetInfo {
        runs,
        steps_files,
        valuation_names,
        total_steps,
    })
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

fn load_runs(dir: &Path) -> Result<Vec<RunSummary>> {
    let path = dir.join("metadata.db");
    if !path.exists() {
        bail!("missing metadata.db in {}", dir.display());
    }
    let conn =
        Connection::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut stmt = conn
        .prepare("SELECT id, seed, steps, max_score, highest_tile FROM runs ORDER BY id ASC")
        .with_context(|| format!("failed to prepare query for {}", path.display()))?;
    let rows = stmt
        .query_map([], |row| {
            Ok(RunSummary {
                run_id: row.get::<_, i64>(0)? as u32,
                seed: row.get::<_, i64>(1)? as u64,
                steps: row.get::<_, i64>(2)? as usize,
                max_score: row.get::<_, i64>(3)? as u64,
                highest_tile: row.get::<_, i64>(4)? as u32,
            })
        })
        .with_context(|| format!("failed to query runs from {}", path.display()))?;
    let mut summaries = Vec::new();
    for row in rows {
        summaries.push(row?);
    }
    Ok(summaries)
}

fn load_valuation_names(dir: &Path) -> Result<Vec<String>> {
    let path = dir.join("valuation_types.json");
    if !path.exists() {
        bail!("missing valuation_types.json in {}", dir.display());
    }
    let text =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let map: HashMap<String, u8> = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    if map.is_empty() {
        return Ok(Vec::new());
    }
    let max_id = map
        .values()
        .copied()
        .max()
        .ok_or_else(|| anyhow!("valuation_types.json has no entries"))?;
    let mut names = vec![String::new(); max_id as usize + 1];
    for (name, id) in map {
        let slot = names
            .get_mut(id as usize)
            .ok_or_else(|| anyhow!("valuation id {} out of range", id))?;
        *slot = name;
    }
    if names.iter().any(|s| s.is_empty()) {
        bail!("valuation_types.json is sparse or has missing indices");
    }
    Ok(names)
}

fn stream_dataset(
    dataset: &DatasetInfo,
    run_map: &HashMap<u32, u32>,
    valuation_ids: &HashMap<String, u8>,
    writer: &mut StepsWriter,
    pb: Option<&ProgressBar>,
) -> Result<usize> {
    const CHUNK: usize = 131_072;
    let mut total_rows = 0usize;
    let names = &dataset.valuation_names;
    for path in &dataset.steps_files {
        let file = File::open(path)
            .with_context(|| format!("failed to open steps shard {}", path.display()))?;
        let mut reader = BufReader::new(file);
        let npy = npyz::NpyFile::new(&mut reader)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let mut data = npy
            .data::<StepRow>()
            .map_err(|err| anyhow!("{}: {err}", path.display()))?;
        let mut chunk = Vec::with_capacity(CHUNK);
        while let Some(row) = data.next() {
            let mut row =
                row.with_context(|| format!("failed to decode row in {}", path.display()))?;
            let new_run = run_map
                .get(&row.run_id)
                .ok_or_else(|| anyhow!("missing run_id {} mapping", row.run_id))?;
            row.run_id = *new_run;
            let name = names.get(row.valuation_type as usize).ok_or_else(|| {
                anyhow!(
                    "valuation_type {} out of range for {}",
                    row.valuation_type,
                    path.display()
                )
            })?;
            let new_id = valuation_ids
                .get(name)
                .ok_or_else(|| anyhow!("unknown valuation '{}' during merge", name))?;
            row.valuation_type = *new_id;
            chunk.push(row);
            if chunk.len() >= CHUNK {
                total_rows += chunk.len();
                writer.write(&chunk)?;
                if let Some(pb) = pb {
                    pb.inc(chunk.len() as u64);
                }
                chunk.clear();
            }
        }
        if !chunk.is_empty() {
            total_rows += chunk.len();
            writer.write(&chunk)?;
            if let Some(pb) = pb {
                pb.inc(chunk.len() as u64);
            }
            chunk.clear();
        }
    }
    Ok(total_rows)
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
