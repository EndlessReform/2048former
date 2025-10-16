//! Helpers for writing structured dataset shards to disk.

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use npyz::{NpyWriter, WriteOptions, WriterBuilder};

use crate::schema::StructuredRow;

/// Manage writing `.npy` shards in either single-file or rotating mode.
pub struct StepsWriter<T: StructuredRow> {
    out_dir: PathBuf,
    rows_per_shard: Option<usize>,
    overwrite: bool,
    prefix: String,
    shard_idx: usize,
    shards_written: usize,
    current: Option<ShardWriter<T>>,
    cleaned: bool,
}

impl<T: StructuredRow> StepsWriter<T> {
    /// Create a new writer that targets `out_dir`.
    pub fn new(out_dir: &Path, rows_per_shard: Option<usize>, overwrite: bool) -> Result<Self> {
        Self::with_prefix(out_dir, "steps", rows_per_shard, overwrite)
    }

    /// Create a writer with a custom shard prefix (e.g., "annotations").
    pub fn with_prefix(
        out_dir: &Path,
        prefix: &str,
        rows_per_shard: Option<usize>,
        overwrite: bool,
    ) -> Result<Self> {
        Ok(Self {
            out_dir: out_dir.to_path_buf(),
            rows_per_shard,
            overwrite,
            prefix: prefix.to_string(),
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
        fs::create_dir_all(&self.out_dir)
            .with_context(|| format!("failed to create {}", self.out_dir.display()))?;
        if self.overwrite {
            let remove = |path: PathBuf| -> Result<()> {
                if path.exists() {
                    fs::remove_file(&path)
                        .with_context(|| format!("failed to remove {}", path.display()))?;
                }
                Ok(())
            };
            remove(self.out_dir.join(format!("{}.npy", self.prefix)))?;
            for entry in fs::read_dir(&self.out_dir)
                .with_context(|| format!("failed to read {}", self.out_dir.display()))?
            {
                let entry = entry?;
                let name = entry.file_name();
                if let Some(name_str) = name.to_str() {
                    if name_str.starts_with(&format!("{}-", self.prefix))
                        && name_str.ends_with(".npy")
                    {
                        fs::remove_file(entry.path()).with_context(|| {
                            format!("failed to remove {}", entry.path().display())
                        })?;
                    }
                }
            }
        } else {
            if self.out_dir.join(format!("{}.npy", self.prefix)).exists() {
                bail!(
                    "{}.npy already exists in {} (use overwrite option)",
                    self.prefix,
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
                        .map(|n| n.starts_with(&format!("{}-", self.prefix)) && n.ends_with(".npy"))
                        .unwrap_or(false)
                });
            if numbered_exists {
                bail!(
                    "found existing {}-*.npy in {} (use overwrite option)",
                    self.prefix,
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
        let shard = ShardWriter::new(
            &self.out_dir,
            &self.prefix,
            self.shard_idx,
            self.rows_per_shard,
        )?;
        self.current = Some(shard);
        Ok(())
    }

    /// Append the provided rows, rotating shards when the configured
    /// `rows_per_shard` threshold is reached.
    pub fn write(&mut self, rows: &[T]) -> Result<()> {
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

    /// Finalise the current shard and return the number of shards written.
    pub fn finish(mut self) -> Result<usize> {
        if let Some(shard) = self.current.take() {
            shard.finish()?;
            self.shards_written += 1;
        }
        Ok(self.shards_written)
    }
}

/// Write a single `.npy` file containing the provided rows.
pub fn write_single_shard<T: StructuredRow>(rows: &[T], out_path: &Path) -> Result<()> {
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let tmp = out_path.with_extension("npy.tmp");
    let file = BufWriter::new(
        File::create(&tmp).with_context(|| format!("failed to create {}", tmp.display()))?,
    );
    let mut writer = WriteOptions::new()
        .dtype(T::dtype())
        .shape(&[rows.len() as u64])
        .writer(file)
        .begin_nd()?;
    writer.extend(rows.iter().copied())?;
    writer.finish()?;
    fs::rename(&tmp, out_path).with_context(|| {
        format!(
            "failed to rename {} -> {}",
            tmp.display(),
            out_path.display()
        )
    })?;
    Ok(())
}

struct ShardWriter<T: StructuredRow> {
    writer: NpyWriter<T, BufWriter<File>>,
    tmp_path: PathBuf,
    final_path: PathBuf,
    rows_written: usize,
    limit: Option<usize>,
}

impl<T: StructuredRow> ShardWriter<T> {
    fn new(out_dir: &Path, prefix: &str, shard_idx: usize, limit: Option<usize>) -> Result<Self> {
        let final_name = if limit.is_some() {
            format!("{prefix}-{shard_idx:05}.npy")
        } else {
            format!("{prefix}.npy")
        };
        let final_path = out_dir.join(final_name);
        let tmp_path = final_path.with_extension("npy.tmp");
        let file = File::create(&tmp_path)
            .with_context(|| format!("failed to create {}", tmp_path.display()))?;
        let writer = WriteOptions::new()
            .dtype(T::dtype())
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

    fn write(&mut self, rows: &[T], limit: usize) -> Result<usize> {
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
