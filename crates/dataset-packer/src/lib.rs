#![allow(unexpected_cfgs, non_local_definitions)]

//! Dataset tooling for 2048 self-play corpora.
//!
//! The crate exposes reusable helpers for packing raw Macroxue logs, merging
//! packed datasets, and working with the lean self-play v1 rails. Run
//! `cargo doc -p dataset-packer --open` to browse the public API.

pub mod macroxue;
pub mod merge;
pub mod schema;
pub mod selfplay;
pub mod valuation;
pub mod writer;

pub use crate::macroxue::{PackOptions, pack_dataset};
pub use crate::merge::{MergeOptions, merge_datasets};
pub use crate::schema::{MacroxueStepRow, SelfplayStepRow, StepRow};
pub use crate::selfplay::{collect_selfplay_step_files, load_selfplay_shard, write_selfplay_steps};
pub use crate::valuation::evaluate;
pub use crate::writer::{StepsWriter, write_single_shard};

/// Summary information returned from packing or merging operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackSummary {
    /// Total number of runs discovered in the input.
    pub runs: usize,
    /// Total number of step rows written to the output.
    pub steps: usize,
    /// Count of output shards (1 when a single `steps.npy` is produced).
    pub shards: usize,
}
