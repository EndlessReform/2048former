use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use npyz::NpyFile;

#[derive(Clone)]
pub struct StudentBinsRecord {
    pub probs: [Vec<f32>; 4],
}

pub struct StudentBinsData {
    pub num_bins: usize,
    pub records: Vec<StudentBinsRecord>,
}

pub fn load_student_bins_shards(dir: &Path) -> Result<Option<StudentBinsData>> {
    let shards = collect_student_shards(dir)?;
    if shards.is_empty() {
        return Ok(None);
    }

    let mut records = Vec::new();
    let mut observed_bins: Option<usize> = None;

    for shard in shards {
        let file = File::open(&shard).with_context(|| format!("open {}", shard.display()))?;
        let mut reader = std::io::BufReader::new(file);
        let npy = NpyFile::new(&mut reader)
            .with_context(|| format!("failed to read {}", shard.display()))?;
        let shape = npy.shape().to_vec();
        if shape.len() != 3 {
            return Err(anyhow!(
                "{}: expected 3D array [steps, branch, bins], got shape {:?}",
                shard.display(),
                shape
            ));
        }
        if shape[1] != 4 {
            return Err(anyhow!(
                "{}: expected branch dimension of 4, got {}",
                shard.display(),
                shape[1]
            ));
        }
        let steps = shape[0] as usize;
        let bins = shape[2] as usize;
        if bins == 0 {
            continue;
        }
        match observed_bins {
            Some(existing) if existing != bins => {
                return Err(anyhow!(
                    "{}: inconsistent bin count (expected {}, found {})",
                    shard.display(),
                    existing,
                    bins
                ));
            }
            None => observed_bins = Some(bins),
            _ => {}
        }
        let data: Vec<f32> = npy
            .into_vec()
            .map_err(|err| anyhow!("{}: {err}", shard.display()))?;
        if data.len() != steps * 4 * bins {
            return Err(anyhow!(
                "{}: data length {} does not match shape {:?}",
                shard.display(),
                data.len(),
                shape
            ));
        }
        let mut offset = 0;
        for _ in 0..steps {
            let mut heads: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(bins));
            for head in 0..4 {
                heads[head].extend_from_slice(&data[offset..offset + bins]);
                offset += bins;
            }
            records.push(StudentBinsRecord { probs: heads });
        }
    }

    let num_bins = observed_bins.unwrap_or(0);
    Ok(Some(StudentBinsData { num_bins, records }))
}

fn collect_student_shards(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let prefix = "annotations-student";
    let single = dir.join(format!("{prefix}.npy"));
    if single.exists() {
        files.push(single);
    }
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("failed to read {}", dir.display()))?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            if name.starts_with(&format!("{prefix}-")) && name.ends_with(".npy") {
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
