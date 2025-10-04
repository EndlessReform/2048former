use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::Result;
use dataset_packer::{SelfplayStepRow, write_selfplay_steps};
use npyz::{DType, TypeStr, WriteOptions, WriterBuilder};

/// Re-export the shared step row layout from `dataset-packer` so engine code can
/// construct rows without duplicating the schema.
pub type StepRow = SelfplayStepRow;

/// Write the aggregated `steps.npy` shard via the dataset-packer helper.
pub fn write_steps_npy(rows: &[StepRow], out_path: &Path) -> Result<()> {
    write_selfplay_steps(rows, out_path)
}

/// Write a float32 embeddings shard `[N, D]` to `path`.
pub fn write_embeddings_npy(floats: &[f32], n_rows: usize, dim: usize, path: &Path) -> Result<()> {
    assert_eq!(floats.len(), n_rows * dim, "embeddings length must be N*D");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = BufWriter::new(File::create(path)?);
    let f4_le: TypeStr = "<f4".parse().unwrap();
    let mut w = WriteOptions::new()
        .dtype(DType::Plain(f4_le))
        .shape(&[n_rows as u64, dim as u64])
        .writer(file)
        .begin_nd()?;
    w.extend(floats.iter().copied())?;
    w.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::BufReader;
    use tempfile::tempdir;

    #[test]
    fn steps_roundtrip_header_and_rows() {
        let td = tempdir().unwrap();
        let path = td.path().join("steps.npy");
        let rows = vec![
            StepRow {
                run_id: 1,
                step_idx: 0,
                exps: [0u8; 16],
            },
            StepRow {
                run_id: 1,
                step_idx: 1,
                exps: [1u8; 16],
            },
            StepRow {
                run_id: 2,
                step_idx: 0,
                exps: [2u8; 16],
            },
        ];
        write_steps_npy(&rows, &path).unwrap();

        // Read header and deserialize back into StepRow via npyz
        let mut r = BufReader::new(File::open(&path).unwrap());
        let hdr = npyz::NpyHeader::from_reader(&mut r).unwrap();
        assert_eq!(hdr.shape(), &[3]);
        drop(r);
        let mut r = BufReader::new(File::open(&path).unwrap());
        let npy = npyz::NpyFile::new(&mut r).unwrap();
        let back: Vec<StepRow> = npy.into_vec().unwrap();
        assert_eq!(back.len(), rows.len());
        assert_eq!(back[0].run_id, 1);
        assert_eq!(back[1].step_idx, 1);
        assert_eq!(back[2].exps[0], 2);
    }

    #[test]
    fn embeddings_roundtrip() {
        let td = tempdir().unwrap();
        let path = td.path().join("embeddings-000001.npy");
        let n = 2usize;
        let d = 3usize;
        let floats: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        write_embeddings_npy(&floats, n, d, &path).unwrap();

        let mut r = BufReader::new(File::open(&path).unwrap());
        let npy = npyz::NpyFile::new(&mut r).unwrap();
        assert_eq!(npy.shape(), &[n as u64, d as u64]);
        let back: Vec<f32> = npy.into_vec().unwrap();
        assert_eq!(back, floats);
    }
}
