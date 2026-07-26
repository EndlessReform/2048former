use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;

use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, warn};
use rayon::prelude::*;

use crate::PackSummary;
use crate::schema::{MacroxueStepRow, MacroxueStepRowV1, StructuredRow};
use crate::writer::StepsWriter;

use super::metadata::{RunSummary, write_metadata, write_valuation_types};
use super::parse::{attach_cumulative_reward, parse_steps_file};
use super::types::{MetaRecord, ValuationEncoder};

/// Configuration for packing raw Macroxue JSON logs into the training layout.
#[derive(Clone, Debug)]
pub struct PackOptions {
    /// Root directories containing `*.meta.json[.gz]` and `*.jsonl.gz` files.
    pub input_roots: Vec<PathBuf>,
    /// Output directory for `steps-*.npy`, `metadata.db`, and `valuation_types.json`.
    pub output_dir: PathBuf,
    /// Maximum rows per shard (omit or `None` to emit a single `steps.npy`).
    pub rows_per_shard: Option<usize>,
    /// Optional override for Rayon worker count.
    pub max_workers: Option<usize>,
    /// Replace existing outputs when true.
    pub overwrite: bool,
    /// Compress shards with zstd when true.
    pub compress: bool,
    /// Include cumulative reward in the packed step schema when true.
    pub include_cumulative_reward: bool,
}

#[derive(Debug, Clone)]
struct RunInput {
    meta_path: PathBuf,
    steps_path: PathBuf,
}

#[derive(Debug, Clone)]
struct RunOutput<T> {
    summary: RunSummary,
    steps: Vec<T>,
}

/// Pack a directory of Macroxue logs into the training dataset layout.
pub fn pack_dataset(opts: PackOptions) -> Result<PackSummary> {
    if opts.rows_per_shard == Some(0) {
        bail!("rows_per_shard must be > 0 when specified");
    }
    validate_input_roots(&opts.input_roots)?;
    fs::create_dir_all(&opts.output_dir)
        .with_context(|| format!("failed to create output dir {}", opts.output_dir.display()))?;

    let runs = discover_runs_multi(&opts.input_roots)?;
    if runs.is_empty() {
        bail!(
            "no .meta.json files found under {}",
            format_input_roots(&opts.input_roots)
        );
    }
    ensure_unique_seeds(&runs)?;

    info!("Discovered {} runs", runs.len());
    let pb = default_progress_bar(runs.len() as u64);

    if opts.include_cumulative_reward {
        pack_dataset_with::<MacroxueStepRow>(&opts, &runs, &pb, process_run_with_cumulative)
    } else {
        pack_dataset_with::<MacroxueStepRowV1>(&opts, &runs, &pb, process_run_legacy)
    }
}

fn pack_dataset_with<T: StructuredRow + Send + Sync + Copy + 'static>(
    opts: &PackOptions,
    runs: &[RunInput],
    pb: &ProgressBar,
    process_run_fn: fn(RunInput, u32, &ValuationEncoder) -> Result<RunOutput<T>>,
) -> Result<PackSummary> {
    let encoder = Arc::new(ValuationEncoder::new());
    let run_count = runs.len();
    let worker_count = opts
        .max_workers
        .unwrap_or_else(|| rayon::current_num_threads());
    let (tx, rx) = mpsc::sync_channel::<(u32, Result<RunOutput<T>>)>(worker_count * 2);
    let runs_vec = runs.to_vec();
    let max_workers = opts.max_workers;
    let tx_worker = tx.clone();
    let pb_worker = pb.clone();
    let pb_main = pb.clone();
    let producer = std::thread::spawn({
        let encoder = encoder.clone();
        move || -> Result<()> {
            let process = || {
                runs_vec.par_iter().cloned().enumerate().for_each_with(
                    tx_worker,
                    |tx, (idx, run)| {
                        let enc = encoder.clone();
                        let out = process_run_fn(run, idx as u32, &enc);
                        pb_worker.inc(1);
                        let _ = tx.send((idx as u32, out));
                    },
                );
            };
            if let Some(n) = max_workers {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .context("failed to build rayon thread pool")?
                    .install(process);
            } else {
                process();
            }
            Ok(())
        }
    });
    drop(tx);

    let mut writer = StepsWriter::<T>::new(
        &opts.output_dir,
        opts.rows_per_shard,
        opts.overwrite,
        opts.compress,
    )?;
    let mut pending: BTreeMap<u32, Option<RunOutput<T>>> = BTreeMap::new();
    let mut next_id = 0u32;
    let mut total_steps = 0usize;
    let mut run_summaries = Vec::with_capacity(run_count);
    let mut first_err: Option<anyhow::Error> = None;

    for (id, res) in rx {
        match res {
            Ok(out) => {
                pending.insert(id, Some(out));
            }
            Err(err) => {
                if first_err.is_none() {
                    first_err = Some(err);
                }
                pending.insert(id, None);
            }
        }
        while let Some(out_opt) = pending.remove(&next_id) {
            if let Some(out) = out_opt {
                if first_err.is_none() {
                    total_steps += out.steps.len();
                    run_summaries.push(out.summary.clone());
                    writer.write(&out.steps)?;
                }
            }
            next_id += 1;
        }
    }

    if let Err(err) = producer
        .join()
        .unwrap_or_else(|_| Err(anyhow::anyhow!("pack worker thread panicked")))
    {
        if first_err.is_none() {
            first_err = Some(err);
        }
    }

    pb_main.finish_with_message("runs processed");

    if let Some(err) = first_err {
        drop(writer);
        crate::writer::clean_outputs(&opts.output_dir, "steps", true)?;
        return Err(err);
    }

    info!(
        "Writing {} steps across {} runs",
        total_steps,
        run_summaries.len()
    );

    let shards = writer.finish()?;
    write_metadata(
        &opts.output_dir,
        &run_summaries,
        opts.overwrite,
        opts.include_cumulative_reward,
    )?;
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

fn discover_runs_multi(roots: &[PathBuf]) -> Result<Vec<RunInput>> {
    let mut runs = Vec::new();
    for root in roots {
        runs.extend(discover_runs(root)?);
    }
    runs.sort_by(|a, b| a.meta_path.cmp(&b.meta_path));
    Ok(runs)
}

fn validate_input_roots(roots: &[PathBuf]) -> Result<()> {
    if roots.is_empty() {
        bail!("at least one input directory is required");
    }
    let mut missing = Vec::new();
    for root in roots {
        if !root.is_dir() {
            missing.push(root);
        }
    }
    if !missing.is_empty() {
        let formatted = missing
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        bail!("input directories missing or not directories: {formatted}");
    }
    Ok(())
}

fn format_input_roots(roots: &[PathBuf]) -> String {
    roots
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn ensure_unique_seeds(runs: &[RunInput]) -> Result<()> {
    let mut seen = std::collections::HashMap::<u64, PathBuf>::new();
    for run in runs {
        let meta_content = read_json_text(&run.meta_path)?;
        let meta: MetaRecord = serde_json::from_str(&meta_content)
            .with_context(|| format!("failed to parse {}", run.meta_path.display()))?;
        if let Some(prev) = seen.insert(meta.seed, run.meta_path.clone()) {
            bail!(
                "duplicate seed {} detected ({} and {}); overlapping seed+step is not allowed",
                meta.seed,
                prev.display(),
                run.meta_path.display()
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    fn write_run(dir: &Path, name: &str, seed: u64) -> Result<()> {
        let meta_path = dir.join(format!("{name}.meta.json"));
        let mut meta = File::create(&meta_path)?;
        writeln!(
            meta,
            "{{\"seed\":{seed},\"num_moves\":1,\"score\":0,\"max_tile\":2}}"
        )?;
        let steps_path = dir.join(format!("{name}.jsonl.gz"));
        let _steps = File::create(&steps_path)?;
        Ok(())
    }

    #[test]
    fn discover_runs_across_multiple_roots() -> Result<()> {
        let root_a = tempfile::tempdir()?;
        let root_b = tempfile::tempdir()?;
        write_run(root_a.path(), "a_run", 1)?;
        write_run(root_b.path(), "b_run", 2)?;

        let runs = discover_runs_multi(&vec![
            root_a.path().to_path_buf(),
            root_b.path().to_path_buf(),
        ])?;
        assert_eq!(runs.len(), 2);
        assert!(runs.windows(2).all(|w| w[0].meta_path <= w[1].meta_path));
        Ok(())
    }

    #[test]
    fn ensure_unique_seeds_errors_on_duplicates() -> Result<()> {
        let root_a = tempfile::tempdir()?;
        let root_b = tempfile::tempdir()?;
        write_run(root_a.path(), "a_run", 42)?;
        write_run(root_b.path(), "b_run", 42)?;

        let runs = discover_runs_multi(&vec![
            root_a.path().to_path_buf(),
            root_b.path().to_path_buf(),
        ])?;
        let err = ensure_unique_seeds(&runs).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("duplicate seed"));
        Ok(())
    }

    #[test]
    fn validate_input_roots_errors_on_missing() -> Result<()> {
        let root = tempfile::tempdir()?;
        let missing = root.path().join("nope");
        let err = validate_input_roots(&[root.path().to_path_buf(), missing.clone()]).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains(&missing.display().to_string()));
        Ok(())
    }
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

fn process_run_legacy(
    run: RunInput,
    run_id: u32,
    encoder: &ValuationEncoder,
) -> Result<RunOutput<MacroxueStepRowV1>> {
    let meta_content = read_json_text(&run.meta_path)?;
    let meta: MetaRecord = serde_json::from_str(&meta_content)
        .with_context(|| format!("failed to parse {}", run.meta_path.display()))?;

    let (steps, _rewards, observed_moves) =
        parse_steps_file(&run.steps_path, run_id, &meta, encoder)?;
    warn_on_counts(&run, run_id, &meta, steps.len(), observed_moves);

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

fn process_run_with_cumulative(
    run: RunInput,
    run_id: u32,
    encoder: &ValuationEncoder,
) -> Result<RunOutput<MacroxueStepRow>> {
    let meta_content = read_json_text(&run.meta_path)?;
    let meta: MetaRecord = serde_json::from_str(&meta_content)
        .with_context(|| format!("failed to parse {}", run.meta_path.display()))?;

    let (steps, rewards, observed_moves) =
        parse_steps_file(&run.steps_path, run_id, &meta, encoder)?;
    let steps = attach_cumulative_reward(steps, &rewards, &run.steps_path)?;
    warn_on_counts(&run, run_id, &meta, steps.len(), observed_moves);

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

fn warn_on_counts(
    run: &RunInput,
    run_id: u32,
    meta: &MetaRecord,
    steps: usize,
    observed_moves: bool,
) {
    if steps == 0 {
        warn!(
            "Run {} produced zero steps (meta path {})",
            run_id,
            run.meta_path.display()
        );
    }
    if steps != meta.num_moves {
        warn!(
            "Run {}: step count mismatch meta {} vs parsed {}",
            run_id, meta.num_moves, steps
        );
    }

    if !observed_moves {
        warn!(
            "Run {} did not record branch EVs with any legal moves; check input at {}",
            run_id,
            run.steps_path.display()
        );
    }
}
