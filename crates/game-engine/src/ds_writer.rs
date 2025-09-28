use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use npyz::{DType, Field, TypeStr, WriterBuilder};

/// Minimal v1 step row: matches docs/self-play-v1.md
/// Fields: run_id: u64, step_idx: u32, exps: [u8; 16]
#[derive(
    Clone, Copy, Debug, PartialEq, npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize,
)]
pub struct StepRow {
    pub run_id: u64,
    pub step_idx: u32,
    pub exps: [u8; 16],
    pub move_dir: u8,
    pub logp: [f32; 4],
}

/// Build the exact dtype for StepRow to ensure NumPy parity.
/// [('run_id','<u8'), ('step_idx','<u4'), ('exps','|u1',(16,))]
fn step_row_dtype() -> DType {
    let u8_le: TypeStr = "<u8".parse().unwrap();
    let u4_le: TypeStr = "<u4".parse().unwrap();
    let u1: TypeStr = "|u1".parse().unwrap();
    let f4_le: TypeStr = "<f4".parse().unwrap();
    DType::Record(vec![
        Field {
            name: "run_id".into(),
            dtype: DType::Plain(u8_le),
        },
        Field {
            name: "step_idx".into(),
            dtype: DType::Plain(u4_le),
        },
        Field {
            name: "exps".into(),
            dtype: DType::Array(16, Box::new(DType::Plain(u1.clone()))),
        },
        Field {
            name: "move_dir".into(),
            dtype: DType::Plain(u1),
        },
        Field {
            name: "logp".into(),
            dtype: DType::Array(4, Box::new(DType::Plain(f4_le))),
        },
    ])
}

/// Write steps.npy with a structured dtype, atomically via a temp path.
pub fn write_steps_npy(
    rows: &[StepRow],
    out_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = out_path.with_extension("npy.tmp");
    let file = BufWriter::new(File::create(&tmp)?);
    let mut w = npyz::WriteOptions::new()
        .dtype(step_row_dtype())
        .shape(&[rows.len() as u64])
        .writer(file)
        .begin_nd()?;
    // StepRow is Copy; extend consumes owned values.
    w.extend(rows.iter().copied())?;
    w.finish()?;
    std::fs::rename(&tmp, out_path)?;
    Ok(())
}

/// Write a float32 embeddings shard `[N, D]` to `path`.
pub fn write_embeddings_npy(
    floats: &[f32],
    n_rows: usize,
    dim: usize,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(floats.len(), n_rows * dim, "embeddings length must be N*D");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = BufWriter::new(File::create(path)?);
    let f4_le: TypeStr = "<f4".parse().unwrap();
    let mut w = npyz::WriteOptions::new()
        .dtype(DType::Plain(f4_le))
        .shape(&[n_rows as u64, dim as u64])
        .writer(file)
        .begin_nd()?;
    // Write in one go; npyz accepts an iterator of f32
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
                move_dir: 0,
                logp: [0.0, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
            },
            StepRow {
                run_id: 1,
                step_idx: 1,
                exps: [1u8; 16],
                move_dir: 2,
                logp: [f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY],
            },
            StepRow {
                run_id: 2,
                step_idx: 0,
                exps: [2u8; 16],
                move_dir: 3,
                logp: [f32::NEG_INFINITY; 4],
            },
        ];
        write_steps_npy(&rows, &path).unwrap();

        // Read header and deserialize back into StepRow via npyz
        let mut r = BufReader::new(File::open(&path).unwrap());
        let hdr = npyz::NpyHeader::from_reader(&mut r).unwrap();
        assert_eq!(hdr.shape(), &[3]);
        // Reset reader and pull typed rows
        drop(r);
        let mut r = BufReader::new(File::open(&path).unwrap());
        let npy = npyz::NpyFile::new(&mut r).unwrap();
        let back: Vec<StepRow> = npy.into_vec().unwrap();
        assert_eq!(back.len(), rows.len());
        assert_eq!(back[0].run_id, 1);
        assert_eq!(back[1].step_idx, 1);
        assert_eq!(back[2].exps[0], 2);
        assert_eq!(back[0].move_dir, 0);
        assert_eq!(back[0].logp[0], 0.0);
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
