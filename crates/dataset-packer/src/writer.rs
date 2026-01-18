//! Helpers for writing structured dataset shards to disk.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use log::info;
use npyz::{NpyWriter, WriteOptions, WriterBuilder};
use zstd::stream::Encoder;

use crate::schema::StructuredRow;

const IO_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const LOG_EVERY_ROWS: usize = 1_000_000;

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
    compress: bool,
    total_rows_written: usize,
    next_log_at: usize,
    row_bytes: usize,
}

pub fn write_shards_parallel<T: StructuredRow + Send + Sync>(
    out_dir: &Path,
    prefix: &str,
    shards: Vec<Vec<T>>,
    overwrite: bool,
    compress: bool,
) -> Result<usize> {
    clean_outputs(out_dir, prefix, overwrite)?;
    if shards.is_empty() {
        info!("writer: no shards to write for prefix \"{prefix}\"");
        return Ok(0);
    }
    let total_rows: usize = shards.iter().map(|rows| rows.len()).sum();
    info!(
        "writer: preparing {} shard(s), {} total rows (compress={})",
        shards.len(),
        total_rows,
        compress
    );
    if shards.len() == 1 {
        let out_path = if compress {
            out_dir.join(format!("{prefix}.npy.zst"))
        } else {
            out_dir.join(format!("{prefix}.npy"))
        };
        info!(
            "writer: writing single shard {} ({} rows, compress={})",
            out_path.display(),
            shards[0].len(),
            compress
        );
        if compress {
            write_shard_with_compression(&shards[0], &out_path)?;
        } else {
            write_single_shard(&shards[0], &out_path)?;
        }
        info!(
            "writer: finished single shard {}",
            out_path.display()
        );
        return Ok(1);
    }

    use rayon::prelude::*;
    info!(
        "writer: writing {} shards in parallel (compress={})",
        shards.len(),
        compress
    );
    shards
        .par_iter()
        .enumerate()
        .try_for_each(|(idx, rows)| write_indexed_shard(rows, out_dir, prefix, idx, compress))?;
    Ok(shards.len())
}

impl<T: StructuredRow> StepsWriter<T> {
    /// Create a new writer that targets `out_dir`.
    pub fn new(
        out_dir: &Path,
        rows_per_shard: Option<usize>,
        overwrite: bool,
        compress: bool,
    ) -> Result<Self> {
        Self::with_prefix(out_dir, "steps", rows_per_shard, overwrite, compress)
    }

    /// Create a writer with a custom shard prefix (e.g., "annotations").
    pub fn with_prefix(
        out_dir: &Path,
        prefix: &str,
        rows_per_shard: Option<usize>,
        overwrite: bool,
        compress: bool,
    ) -> Result<Self> {
        let row_bytes = T::dtype()
            .num_bytes()
            .ok_or_else(|| anyhow::anyhow!("failed to determine row size"))?;
        Ok(Self {
            out_dir: out_dir.to_path_buf(),
            rows_per_shard,
            overwrite,
            prefix: prefix.to_string(),
            shard_idx: 0,
            shards_written: 0,
            current: None,
            cleaned: false,
            compress,
            total_rows_written: 0,
            next_log_at: LOG_EVERY_ROWS,
            row_bytes,
        })
    }

    fn prepare(&mut self) -> Result<()> {
        if self.cleaned {
            return Ok(());
        }
        clean_outputs(&self.out_dir, &self.prefix, self.overwrite)?;
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
            self.compress,
            self.row_bytes,
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
            self.total_rows_written = self.total_rows_written.saturating_add(written);
            if self.total_rows_written >= self.next_log_at {
                let mb = (self.total_rows_written as f64 * self.row_bytes as f64) / (1024.0 * 1024.0);
                info!(
                    "writer: {} rows written total (~{:.1} MiB), shard {} at {} rows",
                    self.total_rows_written,
                    mb,
                    self.shard_idx,
                    shard.rows_written
                );
                self.next_log_at = self.next_log_at.saturating_add(LOG_EVERY_ROWS);
            }
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
        info!(
            "writer: finished {} shard(s), {} rows total",
            self.shards_written, self.total_rows_written
        );
        Ok(self.shards_written)
    }
}

pub(crate) fn clean_outputs(out_dir: &Path, prefix: &str, overwrite: bool) -> Result<()> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create {}", out_dir.display()))?;
    if overwrite {
        let remove = |path: PathBuf| -> Result<()> {
            if path.exists() {
                fs::remove_file(&path)
                    .with_context(|| format!("failed to remove {}", path.display()))?;
            }
            Ok(())
        };
        remove(out_dir.join(format!("{prefix}.npy")))?;
        remove(out_dir.join(format!("{prefix}.npy.zst")))?;
        for entry in fs::read_dir(out_dir)
            .with_context(|| format!("failed to read {}", out_dir.display()))?
        {
            let entry = entry?;
            let name = entry.file_name();
            if let Some(name_str) = name.to_str() {
                if name_str.starts_with(&format!("{prefix}-"))
                    && (name_str.ends_with(".npy") || name_str.ends_with(".npy.zst"))
                {
                    fs::remove_file(entry.path()).with_context(|| {
                        format!("failed to remove {}", entry.path().display())
                    })?;
                }
            }
        }
    } else {
        if out_dir.join(format!("{prefix}.npy")).exists() {
            bail!(
                "{prefix}.npy already exists in {} (use overwrite option)",
                out_dir.display()
            );
        }
        if out_dir.join(format!("{prefix}.npy.zst")).exists() {
            bail!(
                "{prefix}.npy.zst already exists in {} (use overwrite option)",
                out_dir.display()
            );
        }
        let numbered_exists = fs::read_dir(out_dir)
            .with_context(|| format!("failed to read {}", out_dir.display()))?
            .filter_map(|e| e.ok())
            .any(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .map(|n| n.starts_with(&format!("{prefix}-")) && (n.ends_with(".npy") || n.ends_with(".npy.zst")))
                    .unwrap_or(false)
            });
        if numbered_exists {
            bail!(
                "found existing {prefix}-*.npy[.zst] in {} (use overwrite option)",
                out_dir.display()
            );
        }
    }
    Ok(())
}

fn write_indexed_shard<T: StructuredRow>(
    rows: &[T],
    out_dir: &Path,
    prefix: &str,
    shard_idx: usize,
    compress: bool,
) -> Result<()> {
    let out_path = if compress {
        out_dir.join(format!("{prefix}-{shard_idx:05}.npy.zst"))
    } else {
        out_dir.join(format!("{prefix}-{shard_idx:05}.npy"))
    };
    info!(
        "writer: start shard {} ({} rows, compress={})",
        out_path.display(),
        rows.len(),
        compress
    );
    if compress {
        write_shard_with_compression(rows, &out_path)?;
    } else {
        write_single_shard(rows, &out_path)?;
    }
    info!(
        "writer: finished shard {}",
        out_path.display()
    );
    Ok(())
}

fn write_shard_with_compression<T: StructuredRow>(rows: &[T], out_path: &Path) -> Result<()> {
    info!(
        "writer: writing {} rows to {} (tmp)",
        rows.len(),
        out_path.display()
    );
    let tmp = out_path.with_extension("npy.tmp");
    let file = BufWriter::with_capacity(
        IO_BUFFER_BYTES,
        File::create(&tmp).with_context(|| format!("failed to create {}", tmp.display()))?,
    );
    let mut writer = WriteOptions::new()
        .dtype(T::dtype())
        .shape(&[rows.len() as u64])
        .writer(file)
        .begin_nd()?;
    writer.extend(rows.iter().copied())?;
    writer.finish()?;
    let raw_bytes = fs::metadata(&tmp)
        .with_context(|| format!("failed to stat {}", tmp.display()))?
        .len();
    let raw_mb = raw_bytes as f64 / (1024.0 * 1024.0);

    let zst_tmp = out_path.with_extension("zst.tmp");
    info!(
        "writer: compressing {} -> {} (~{:.1} MiB raw)",
        tmp.display(),
        zst_tmp.display(),
        raw_mb
    );
    let input = File::open(&tmp).with_context(|| format!("failed to open {}", tmp.display()))?;
    let output =
        File::create(&zst_tmp).with_context(|| format!("failed to create {}", zst_tmp.display()))?;
    let mut reader = BufReader::with_capacity(IO_BUFFER_BYTES, input);
    let encoder = Encoder::new(BufWriter::with_capacity(IO_BUFFER_BYTES, output), 3)
        .with_context(|| format!("failed to create zstd encoder for {}", zst_tmp.display()))?
        .auto_finish();
    let mut writer = encoder;
    std::io::copy(&mut reader, &mut writer)
        .with_context(|| format!("failed to compress {} -> {}", tmp.display(), zst_tmp.display()))?;
    let zst_bytes = fs::metadata(&zst_tmp)
        .with_context(|| format!("failed to stat {}", zst_tmp.display()))?
        .len();
    let zst_mb = zst_bytes as f64 / (1024.0 * 1024.0);
    fs::rename(&zst_tmp, out_path).with_context(|| {
        format!(
            "failed to rename {} -> {}",
            zst_tmp.display(),
            out_path.display()
        )
    })?;
    fs::remove_file(&tmp).with_context(|| format!("failed to remove {}", tmp.display()))?;
    info!(
        "writer: compressed shard complete ({:.1} MiB -> {:.1} MiB)",
        raw_mb,
        zst_mb
    );
    Ok(())
}

/// Write a single `.npy` file containing the provided rows.
pub fn write_single_shard<T: StructuredRow>(rows: &[T], out_path: &Path) -> Result<()> {
    info!(
        "writer: writing {} rows to {}",
        rows.len(),
        out_path.display()
    );
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let tmp = out_path.with_extension("npy.tmp");
    let file = BufWriter::with_capacity(
        IO_BUFFER_BYTES,
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
    info!(
        "writer: finished writing {}",
        out_path.display()
    );
    Ok(())
}

struct ShardWriter<T: StructuredRow> {
    writer: NpyWriter<T, BufWriter<File>>,
    tmp_path: PathBuf,
    final_path: PathBuf,
    rows_written: usize,
    limit: Option<usize>,
    compress: bool,
    row_bytes: usize,
    next_log_at: usize,
}

impl<T: StructuredRow> ShardWriter<T> {
    fn new(
        out_dir: &Path,
        prefix: &str,
        shard_idx: usize,
        limit: Option<usize>,
        compress: bool,
        row_bytes: usize,
    ) -> Result<Self> {
        let final_name = if limit.is_some() {
            if compress {
                format!("{prefix}-{shard_idx:05}.npy.zst")
            } else {
                format!("{prefix}-{shard_idx:05}.npy")
            }
        } else {
            if compress {
                format!("{prefix}.npy.zst")
            } else {
                format!("{prefix}.npy")
            }
        };
        let final_path = out_dir.join(final_name);
        let tmp_name = if limit.is_some() {
            format!("{prefix}-{shard_idx:05}.npy.tmp")
        } else {
            format!("{prefix}.npy.tmp")
        };
        let tmp_path = out_dir.join(tmp_name);
        let file = File::create(&tmp_path)
            .with_context(|| format!("failed to create {}", tmp_path.display()))?;
        let writer = WriteOptions::new()
            .dtype(T::dtype())
            .writer(BufWriter::with_capacity(IO_BUFFER_BYTES, file))
            .begin_1d()?;
        info!(
            "writer: open shard {} (limit {:?}, compress={})",
            final_path.display(),
            limit,
            compress
        );
        Ok(Self {
            writer,
            tmp_path,
            final_path,
            rows_written: 0,
            limit,
            compress,
            row_bytes,
            next_log_at: LOG_EVERY_ROWS,
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
            if self.rows_written >= self.next_log_at {
                let mb = (self.rows_written as f64 * self.row_bytes as f64) / (1024.0 * 1024.0);
                info!(
                    "writer: shard {} at {} rows (~{:.1} MiB)",
                    self.final_path.display(),
                    self.rows_written,
                    mb
                );
                self.next_log_at = self.next_log_at.saturating_add(LOG_EVERY_ROWS);
            }
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
            compress,
            ..
        } = self;
        info!("writer: finalizing shard {}", final_path.display());
        writer.finish()?;
        if compress {
            let zst_tmp = final_path.with_extension("zst.tmp");
            let raw_bytes = fs::metadata(&tmp_path)
                .with_context(|| format!("failed to stat {}", tmp_path.display()))?
                .len();
            let raw_mb = raw_bytes as f64 / (1024.0 * 1024.0);
            info!(
                "writer: compressing shard {} (~{:.1} MiB raw)",
                tmp_path.display(),
                raw_mb
            );
            let input = File::open(&tmp_path)
                .with_context(|| format!("failed to open {}", tmp_path.display()))?;
            let output = File::create(&zst_tmp)
                .with_context(|| format!("failed to create {}", zst_tmp.display()))?;
            let mut reader = BufReader::with_capacity(IO_BUFFER_BYTES, input);
            let encoder = Encoder::new(BufWriter::with_capacity(IO_BUFFER_BYTES, output), 3)
                .with_context(|| format!("failed to create zstd encoder for {}", zst_tmp.display()))?
                .auto_finish();
            let mut writer = encoder;
            std::io::copy(&mut reader, &mut writer).with_context(|| {
                format!(
                    "failed to compress {} -> {}",
                    tmp_path.display(),
                    zst_tmp.display()
                )
            })?;
            let zst_bytes = fs::metadata(&zst_tmp)
                .with_context(|| format!("failed to stat {}", zst_tmp.display()))?
                .len();
            let zst_mb = zst_bytes as f64 / (1024.0 * 1024.0);
            fs::rename(&zst_tmp, &final_path).with_context(|| {
                format!(
                    "failed to rename {} -> {}",
                    zst_tmp.display(),
                    final_path.display()
                )
            })?;
            fs::remove_file(&tmp_path).with_context(|| {
                format!("failed to remove {}", tmp_path.display())
            })?;
            info!(
                "writer: compressed shard done ({:.1} MiB -> {:.1} MiB)",
                raw_mb,
                zst_mb
            );
        } else {
            fs::rename(&tmp_path, &final_path).with_context(|| {
                format!(
                    "failed to rename {} -> {}",
                    tmp_path.display(),
                    final_path.display()
                )
            })?;
        }
        Ok(())
    }
}
