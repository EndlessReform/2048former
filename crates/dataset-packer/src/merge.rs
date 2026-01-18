use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use indicatif::ProgressBar;
use npyz::{DType, Deserialize, NpyFile};
use zstd::stream::read::Decoder;

use crate::PackSummary;
use crate::macroxue::{RunSummary, load_cumulative_reward_flag};
use crate::schema::{MacroxueStepRow, MacroxueStepRowV1, StructuredRow};
use crate::writer::StepsWriter;

/// Options controlling dataset merges.
#[derive(Clone, Debug)]
pub struct MergeOptions {
    /// First packed dataset directory.
    pub left_dir: PathBuf,
    /// Second packed dataset directory.
    pub right_dir: PathBuf,
    /// Output directory for the merged dataset.
    pub output_dir: PathBuf,
    /// Maximum rows per shard in the merged output (omit for a single file).
    pub rows_per_shard: Option<usize>,
    /// Overwrite existing outputs when true.
    pub overwrite: bool,
    /// Compress shards with zstd when true.
    pub compress: bool,
    /// Delete input directories after a successful merge.
    pub delete_inputs: bool,
}

#[derive(Debug)]
struct DatasetInfo {
    runs: Vec<RunSummary>,
    steps_files: Vec<PathBuf>,
    valuation_names: Vec<String>,
    total_steps: usize,
    has_cumulative_reward: bool,
}

/// Merge two Macroxue-format datasets into a single output directory.
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

    if left.has_cumulative_reward != right.has_cumulative_reward {
        bail!("merge requires both inputs to agree on cumulative_reward presence");
    }

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

    let pb = if total_steps > 0 {
        let pb = crate::macroxue::default_progress_bar(total_steps as u64);
        pb.set_message("merging steps");
        Some(pb)
    } else {
        None
    };

    let shards = if left.has_cumulative_reward {
        let mut writer = StepsWriter::<MacroxueStepRow>::new(
            &opts.output_dir,
            opts.rows_per_shard,
            opts.overwrite,
            opts.compress,
        )?;
        stream_dataset(&left, &left_map, &valuation_ids, &mut writer, pb.as_ref())?;
        stream_dataset(&right, &right_map, &valuation_ids, &mut writer, pb.as_ref())?;
        writer.finish()?
    } else {
        let mut writer = StepsWriter::<MacroxueStepRowV1>::new(
            &opts.output_dir,
            opts.rows_per_shard,
            opts.overwrite,
            opts.compress,
        )?;
        stream_dataset(&left, &left_map, &valuation_ids, &mut writer, pb.as_ref())?;
        stream_dataset(&right, &right_map, &valuation_ids, &mut writer, pb.as_ref())?;
        writer.finish()?
    };

    if let Some(pb) = &pb {
        pb.finish_with_message("merge complete");
    }

    crate::macroxue::write_metadata(
        &opts.output_dir,
        &new_runs,
        opts.overwrite,
        left.has_cumulative_reward,
    )?;
    crate::macroxue::write_valuation_types(&opts.output_dir, &valuation_names, opts.overwrite)?;

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
        bail!("no steps.npy[.zst] files found in {}", dir.display());
    }
    let runs = crate::macroxue::load_runs(dir)?;
    if runs.is_empty() {
        bail!("metadata.db in {} has no runs", dir.display());
    }
    let valuation_names = crate::macroxue::load_valuation_names(dir)?;
    let total_steps = runs.iter().map(|r| r.steps).sum();
    let meta_flag = load_cumulative_reward_flag(dir)?;
    let dtype_flag = detect_cumulative_reward(&steps_files)?;
    if let Some(flag) = meta_flag {
        if flag != dtype_flag {
            bail!(
                "cumulative_reward metadata mismatch in {} (session={}, dtype={})",
                dir.display(),
                flag,
                dtype_flag
            );
        }
    }
    let has_cumulative_reward = meta_flag.unwrap_or(dtype_flag);
    Ok(DatasetInfo {
        runs,
        steps_files,
        valuation_names,
        total_steps,
        has_cumulative_reward,
    })
}

fn collect_step_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let single = dir.join("steps.npy");
    let single_zst = dir.join("steps.npy.zst");
    match (single.exists(), single_zst.exists()) {
        (true, true) => {
            bail!(
                "both steps.npy and steps.npy.zst exist in {}",
                dir.display()
            );
        }
        (true, false) => files.push(single),
        (false, true) => files.push(single_zst),
        _ => {}
    }
    let mut shards: Vec<PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("failed to read {}", dir.display()))?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            if name.starts_with("steps-") && (name.ends_with(".npy") || name.ends_with(".npy.zst"))
            {
                Some(entry.path())
            } else {
                None
            }
        })
        .collect();
    shards.sort();
    for path in shards {
        if path.extension().and_then(|s| s.to_str()) == Some("zst") {
            let raw = path.with_extension("npy");
            if raw.exists() {
                bail!(
                    "both {} and {} exist in {}",
                    raw.file_name().unwrap_or_default().to_string_lossy(),
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    dir.display()
                );
            }
        }
        files.push(path);
    }
    Ok(files)
}

fn stream_dataset<T: MacroxueRow>(
    dataset: &DatasetInfo,
    run_map: &HashMap<u32, u32>,
    valuation_ids: &HashMap<String, u8>,
    writer: &mut StepsWriter<T>,
    pb: Option<&ProgressBar>,
) -> Result<usize> {
    const CHUNK: usize = 131_072;
    let mut total_rows = 0usize;
    let names = &dataset.valuation_names;
    for path in &dataset.steps_files {
        let mut reader = open_npy_reader(path)?;
        let npy = NpyFile::new(&mut reader)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let mut data = npy
            .data::<T>()
            .map_err(|err| anyhow!("{}: {err}", path.display()))?;
        let mut chunk = Vec::with_capacity(CHUNK);
        while let Some(row) = data.next() {
            let mut row =
                row.with_context(|| format!("failed to decode row in {}", path.display()))?;
            let new_run = run_map
                .get(&row.run_id())
                .ok_or_else(|| anyhow!("missing run_id {} mapping", row.run_id()))?;
            row.set_run_id(*new_run);
            let name = names.get(row.valuation_type() as usize).ok_or_else(|| {
                anyhow!(
                    "valuation_type {} out of range for {}",
                    row.valuation_type(),
                    path.display()
                )
            })?;
            let new_id = valuation_ids
                .get(name)
                .ok_or_else(|| anyhow!("unknown valuation '{}' during merge", name))?;
            row.set_valuation_type(*new_id);
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

fn open_npy_reader(path: &Path) -> Result<Box<dyn Read>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open steps shard {}", path.display()))?;
    let reader = BufReader::new(file);
    if path.extension().and_then(|s| s.to_str()) == Some("zst") {
        let decoder = Decoder::new(reader)
            .with_context(|| format!("failed to decode zstd shard {}", path.display()))?;
        Ok(Box::new(decoder))
    } else {
        Ok(Box::new(reader))
    }
}

trait MacroxueRow: StructuredRow + Deserialize + Copy {
    fn run_id(&self) -> u32;
    fn set_run_id(&mut self, run_id: u32);
    fn valuation_type(&self) -> u8;
    fn set_valuation_type(&mut self, valuation_type: u8);
}

impl MacroxueRow for MacroxueStepRow {
    fn run_id(&self) -> u32 {
        self.run_id
    }

    fn set_run_id(&mut self, run_id: u32) {
        self.run_id = run_id;
    }

    fn valuation_type(&self) -> u8 {
        self.valuation_type
    }

    fn set_valuation_type(&mut self, valuation_type: u8) {
        self.valuation_type = valuation_type;
    }
}

impl MacroxueRow for MacroxueStepRowV1 {
    fn run_id(&self) -> u32 {
        self.run_id
    }

    fn set_run_id(&mut self, run_id: u32) {
        self.run_id = run_id;
    }

    fn valuation_type(&self) -> u8 {
        self.valuation_type
    }

    fn set_valuation_type(&mut self, valuation_type: u8) {
        self.valuation_type = valuation_type;
    }
}

fn detect_cumulative_reward(step_files: &[PathBuf]) -> Result<bool> {
    let mut expected: Option<bool> = None;
    for path in step_files {
        let mut reader = open_npy_reader(path)?;
        let npy =
            NpyFile::new(&mut reader).with_context(|| format!("failed to read {}", path.display()))?;
        let has_field = dtype_has_field(&npy.dtype(), "cumulative_reward");
        match expected {
            None => expected = Some(has_field),
            Some(prev) if prev != has_field => {
                bail!(
                    "mixed step schemas in {} (cumulative_reward presence differs)",
                    path.display()
                );
            }
            _ => {}
        }
    }
    Ok(expected.unwrap_or(false))
}

fn dtype_has_field(dtype: &DType, name: &str) -> bool {
    match dtype {
        DType::Record(fields) => fields.iter().any(|field| field.name == name),
        _ => false,
    }
}
