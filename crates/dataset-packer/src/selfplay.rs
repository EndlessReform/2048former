use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use npyz::NpyFile;

use crate::macrosxue::RunSummary;
use crate::schema::SelfplayStepRow;
use crate::writer::write_single_shard;

/// Write lean self-play step rows to a `.npy` file (single shard).
pub fn write_selfplay_steps(rows: &[SelfplayStepRow], out_path: &Path) -> Result<()> {
    write_single_shard(rows, out_path)
}

/// Load all rows from a single self-play shard.
pub fn load_selfplay_shard(path: &Path) -> Result<Vec<SelfplayStepRow>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open shard {}", path.display()))?;
    let mut reader = std::io::BufReader::new(file);
    let npy =
        NpyFile::new(&mut reader).with_context(|| format!("failed to read {}", path.display()))?;
    npy.into_vec()
        .map_err(|err| anyhow!("{}: {err}", path.display()))
}

/// Collect `steps.npy` / `steps-*.npy` files from a self-play dataset directory.
pub fn collect_selfplay_step_files(dir: &Path) -> Result<Vec<PathBuf>> {
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

/// Load run summaries from `metadata.db` alongside a self-play dataset.
pub fn load_selfplay_runs(dir: &Path) -> Result<Vec<RunSummary>> {
    crate::macrosxue::load_runs(dir)
}

/// Ensure that a self-play dataset directory looks structurally valid.
pub fn validate_selfplay_dataset(dir: &Path) -> Result<()> {
    let steps = collect_selfplay_step_files(dir)?;
    if steps.is_empty() {
        bail!("no steps.npy files found in {}", dir.display());
    }
    if crate::macrosxue::load_runs(dir)?.is_empty() {
        bail!("metadata.db in {} has no runs", dir.display());
    }
    Ok(())
}
