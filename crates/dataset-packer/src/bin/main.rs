use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use dataset_packer::{PackOptions, pack_dataset};
use env_logger::Env;
use log::info;

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Pack Macroxue self-play JSON into steps.npy + metadata.db"
)]
struct Cli {
    /// Root directory containing *.meta.json + *.jsonl.gz files
    #[arg(long, value_name = "DIR")]
    input: PathBuf,

    /// Output directory for steps.npy, metadata.db, valuation_types.json
    #[arg(long, value_name = "DIR")]
    output: PathBuf,

    /// Maximum rows per shard (0 or omit => single steps.npy)
    #[arg(long, value_name = "N")]
    shard_rows: Option<usize>,

    /// Number of worker threads (defaults to Rayon default)
    #[arg(long, value_name = "N")]
    workers: Option<usize>,

    /// Overwrite existing outputs if present
    #[arg(long)]
    overwrite: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let options = PackOptions {
        input_root: cli.input,
        output_dir: cli.output,
        rows_per_shard: cli.shard_rows.filter(|&n| n > 0),
        max_workers: cli.workers,
        overwrite: cli.overwrite,
    };

    let summary = pack_dataset(options)?;
    info!(
        "Completed packing: {} runs, {} steps, {} shard(s)",
        summary.runs, summary.steps, summary.shards
    );
    Ok(())
}
