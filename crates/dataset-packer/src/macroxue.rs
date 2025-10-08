pub mod board_eval;
pub mod tokenizer;

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, warn};
use rayon::prelude::*;
use rusqlite::{Connection, params};
use serde::Deserialize;

use crate::PackSummary;
use crate::schema::MacroxueStepRow;
use crate::writer::StepsWriter;

/// Configuration for packing raw Macroxue JSON logs into the training layout.
#[derive(Clone, Debug)]
pub struct PackOptions {
    /// Root directory containing `*.meta.json[.gz]` and `*.jsonl.gz` files.
    pub input_root: PathBuf,
    /// Output directory for `steps-*.npy`, `metadata.db`, and `valuation_types.json`.
    pub output_dir: PathBuf,
    /// Maximum rows per shard (omit or `None` to emit a single `steps.npy`).
    pub rows_per_shard: Option<usize>,
    /// Optional override for Rayon worker count.
    pub max_workers: Option<usize>,
    /// Replace existing outputs when true.
    pub overwrite: bool,
}

/// Summary of a packed run extracted from the metadata sidecar.
#[derive(Debug, Clone)]
pub struct RunSummary {
    /// Contiguous run identifier assigned during packing.
    pub run_id: u32,
    /// PRNG seed recorded in the sidecar.
    pub seed: u64,
    /// Number of step rows emitted for this run.
    pub steps: usize,
    /// Final score reported by the Macroxue engine.
    pub max_score: u64,
    /// Highest tile (2^k) reached in the run.
    pub highest_tile: u32,
}

#[derive(Debug, Clone)]
struct RunInput {
    meta_path: PathBuf,
    steps_path: PathBuf,
}

#[derive(Debug, Clone)]
struct RunOutput {
    summary: RunSummary,
    steps: Vec<MacroxueStepRow>,
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

/// Pack a directory of Macroxue logs into the training dataset layout.
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

    let mut writer =
        StepsWriter::<MacroxueStepRow>::new(&opts.output_dir, opts.rows_per_shard, opts.overwrite)?;
    for out in &outputs {
        writer.write(&out.steps)?;
    }
    let shards = writer.finish()?;
    write_metadata(&opts.output_dir, &run_summaries, opts.overwrite)?;
    write_valuation_types(&opts.output_dir, &encoder.as_vec(), opts.overwrite)?;

    Ok(PackSummary {
        runs: run_summaries.len(),
        steps: total_steps,
        shards,
    })
}

pub(crate) fn default_progress_bar(len: u64) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {wide_bar} {pos}/{len}")
            .unwrap()
            .progress_chars("=> "),
    );
    pb
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

fn is_meta_filename(name: &str) -> bool {
    name.ends_with(".meta.json") || name.ends_with(".meta.json.gz")
}

fn candidate_steps_path(meta_path: &Path) -> Option<PathBuf> {
    let file_name = meta_path.file_name().and_then(|s| s.to_str())?;
    let base = meta_base_name(file_name)?;
    let mut steps = meta_path.to_path_buf();
    steps.set_file_name(format!("{base}.jsonl.gz"));
    Some(steps)
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
    path: &Path,
    run_id: u32,
    meta: &MetaRecord,
    encoder: &ValuationEncoder,
) -> Result<(Vec<MacroxueStepRow>, bool)> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let gz = GzDecoder::new(file);
    let reader = BufReader::new(gz);
    let mut rows = Vec::new();
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
        let encoded = encoder.encode(record.valuation_type.as_deref().unwrap_or("search"))?;

        let (board, mask) = pack_board(&record.board);
        let board_eval = board_eval::evaluate(&record.board, false)
            .with_context(|| format!("failed to evaluate board in {}", path.display()))?;
        let board_eval = i32::try_from(board_eval)
            .with_context(|| format!("board evaluation overflow in {}", path.display()))?;
        let mut branch_evs = [0f32; 4];
        let mut legal_bits = 0u8;
        for (name, value) in record.branch_evs.iter() {
            let code = match name.as_str() {
                "up" => 0,
                "down" => 1,
                "left" => 2,
                "right" => 3,
                other => {
                    warn!("unknown branch '{other}' in {}", path.display());
                    continue;
                }
            };
            if let Some(v) = value {
                branch_evs[code] = *v;
                legal_bits |= 1u8 << code;
            }
        }
        any_legal |= legal_bits != 0;
        rows.push(MacroxueStepRow {
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
    Ok((rows, any_legal))
}

fn pack_board(board: &[u8]) -> (u64, u16) {
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

pub(crate) fn write_metadata(out_dir: &Path, runs: &[RunSummary], overwrite: bool) -> Result<()> {
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

pub(crate) fn write_valuation_types(
    out_dir: &Path,
    names: &[String],
    overwrite: bool,
) -> Result<()> {
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

pub fn load_runs(dir: &Path) -> Result<Vec<RunSummary>> {
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

pub fn load_valuation_names(dir: &Path) -> Result<Vec<String>> {
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

#[cfg(test)]
mod tests {
    use super::board_eval;
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
    }
}
