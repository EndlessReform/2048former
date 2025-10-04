//! Shared structured array schemas for dataset shards.
//!
//! Generate API docs with `cargo doc -p dataset-packer --open` to explore the
//! available step layouts and helper traits.

use npyz::{DType, Field, TypeStr};

/// Trait implemented by structured rows that can be written to an `.npy` file.
///
/// Types implementing this trait provide both the NumPy dtype description and
/// the ability to serialise themselves via the `npyz` crate derives. All step
/// rows in this crate implement `StructuredRow` so generic writers can operate
/// over them.
pub trait StructuredRow: Copy + npyz::Serialize {
    /// Return the NumPy dtype descriptor for the row.
    fn dtype() -> DType;
}

/// Row layout for Macroxue self-play packs.
///
/// This mirrors the compound dtype documented in
/// `docs/macroxue_data/data_format.md`. Boards are encoded as packed nibbles in
/// `board` with overflow tracked in `tile_65536_mask`. Branch EVs follow the
/// canonical UDLR order.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize,
)]
pub struct MacroxueStepRow {
    pub run_id: u32,
    pub step_index: u32,
    pub board: u64,
    pub tile_65536_mask: u16,
    pub move_dir: u8,
    pub valuation_type: u8,
    pub ev_legal: u8,
    pub max_rank: u8,
    pub seed: u32,
    pub branch_evs: [f32; 4],
}

impl StructuredRow for MacroxueStepRow {
    fn dtype() -> DType {
        let u1: TypeStr = "<u1".parse().unwrap();
        let u2: TypeStr = "<u2".parse().unwrap();
        let u4: TypeStr = "<u4".parse().unwrap();
        let u8: TypeStr = "<u8".parse().unwrap();
        let f4: TypeStr = "<f4".parse().unwrap();
        DType::Record(vec![
            Field {
                name: "run_id".into(),
                dtype: DType::Plain(u4.clone()),
            },
            Field {
                name: "step_index".into(),
                dtype: DType::Plain(u4.clone()),
            },
            Field {
                name: "board".into(),
                dtype: DType::Plain(u8),
            },
            Field {
                name: "tile_65536_mask".into(),
                dtype: DType::Plain(u2.clone()),
            },
            Field {
                name: "move_dir".into(),
                dtype: DType::Plain(u1.clone()),
            },
            Field {
                name: "valuation_type".into(),
                dtype: DType::Plain(u1.clone()),
            },
            Field {
                name: "ev_legal".into(),
                dtype: DType::Plain(u1.clone()),
            },
            Field {
                name: "max_rank".into(),
                dtype: DType::Plain(u1),
            },
            Field {
                name: "seed".into(),
                dtype: DType::Plain(u4),
            },
            Field {
                name: "branch_evs".into(),
                dtype: DType::Array(4, Box::new(DType::Plain(f4))),
            },
        ])
    }
}

/// Row layout for the lean self-play v1 dataset.
///
/// The format keeps only the run identifier, per-run step index, and board
/// exponents (0 = empty, 1 = 2, ...). This struct matches the writer used by
/// the Rust engine under `docs/self-play-v1.md`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize,
)]
pub struct SelfplayStepRow {
    pub run_id: u64,
    pub step_idx: u32,
    pub exps: [u8; 16],
}

impl StructuredRow for SelfplayStepRow {
    fn dtype() -> DType {
        let u8_le: TypeStr = "<u8".parse().unwrap();
        let u4_le: TypeStr = "<u4".parse().unwrap();
        let u1: TypeStr = "|u1".parse().unwrap();
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
                dtype: DType::Array(16, Box::new(DType::Plain(u1))),
            },
        ])
    }
}

/// Convenience alias kept for backwards compatibility with older imports.
pub type StepRow = MacroxueStepRow;

/// Annotation row written by the offline policy annotator.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize,
)]
pub struct AnnotationRow {
    pub run_id: u32,
    pub step_index: u32,
    pub teacher_move: u8,
    pub legal_mask: u8,
    pub argmax_head: u8,
    pub argmax_prob: f32,
    pub policy_p1: [f32; 4],
}

impl StructuredRow for AnnotationRow {
    fn dtype() -> DType {
        let u1: TypeStr = "<u1".parse().unwrap();
        let u4: TypeStr = "<u4".parse().unwrap();
        let f4: TypeStr = "<f4".parse().unwrap();
        DType::Record(vec![
            Field {
                name: "run_id".into(),
                dtype: DType::Plain(u4.clone()),
            },
            Field {
                name: "step_index".into(),
                dtype: DType::Plain(u4.clone()),
            },
            Field {
                name: "teacher_move".into(),
                dtype: DType::Plain(u1.clone()),
            },
            Field {
                name: "legal_mask".into(),
                dtype: DType::Plain(u1.clone()),
            },
            Field {
                name: "argmax_head".into(),
                dtype: DType::Plain(u1.clone()),
            },
            Field {
                name: "argmax_prob".into(),
                dtype: DType::Plain(f4.clone()),
            },
            Field {
                name: "policy_p1".into(),
                dtype: DType::Array(4, Box::new(DType::Plain(f4))),
            },
        ])
    }
}
