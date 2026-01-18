use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use npyz::NpyFile;
use zstd::stream::read::Decoder;

use crate::macroxue::RunSummary;
use crate::schema::SelfplayStepRow;
use crate::writer::write_single_shard;

/// Write lean self-play step rows to a `.npy` file (single shard).
pub fn write_selfplay_steps(rows: &[SelfplayStepRow], out_path: &Path) -> Result<()> {
    write_single_shard(rows, out_path)
}

/// Load all rows from a single self-play shard.
pub fn load_selfplay_shard(path: &Path) -> Result<Vec<SelfplayStepRow>> {
    let mut reader = open_npy_reader(path)?;
    let npy =
        NpyFile::new(&mut reader).with_context(|| format!("failed to read {}", path.display()))?;
    npy.into_vec()
        .map_err(|err| anyhow!("{}: {err}", path.display()))
}

/// Collect `steps.npy` / `steps-*.npy` files from a self-play dataset directory.
pub fn collect_selfplay_step_files(dir: &Path) -> Result<Vec<PathBuf>> {
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

/// Load run summaries from `metadata.db` alongside a self-play dataset.
pub fn load_selfplay_runs(dir: &Path) -> Result<Vec<RunSummary>> {
    crate::macroxue::load_runs(dir)
}

/// Ensure that a self-play dataset directory looks structurally valid.
pub fn validate_selfplay_dataset(dir: &Path) -> Result<()> {
    let steps = collect_selfplay_step_files(dir)?;
    if steps.is_empty() {
        bail!("no steps.npy[.zst] files found in {}", dir.display());
    }
    if crate::macroxue::load_runs(dir)?.is_empty() {
        bail!("metadata.db in {} has no runs", dir.display());
    }
    Ok(())
}

fn open_npy_reader(path: &Path) -> Result<Box<dyn Read>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open shard {}", path.display()))?;
    let reader = BufReader::new(file);
    if path.extension().and_then(|s| s.to_str()) == Some("zst") {
        let decoder = Decoder::new(reader)
            .with_context(|| format!("failed to decode zstd shard {}", path.display()))?;
        Ok(Box::new(decoder))
    } else {
        Ok(Box::new(reader))
    }
}
