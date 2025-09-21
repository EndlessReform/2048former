use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use dataset_packer::{MergeOptions, PackOptions, merge_datasets, pack_dataset};
use env_logger::Env;
use log::info;

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Dataset utilities for Macroxue self-play",
    subcommand_required = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Pack raw Macroxue JSON logs into steps.npy + metadata.db
    Pack(PackArgs),
    /// Merge two packed datasets into a single directory
    Merge(MergeArgs),
}

#[derive(Debug, Args)]
struct PackArgs {
    /// Root directory containing *.meta.json[.gz] + *.jsonl.gz files
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

#[derive(Debug, Args)]
struct MergeArgs {
    /// First packed dataset directory
    #[arg(long, value_name = "DIR")]
    left: PathBuf,

    /// Second packed dataset directory
    #[arg(long, value_name = "DIR")]
    right: PathBuf,

    /// Output directory for the merged dataset
    #[arg(long, value_name = "DIR")]
    output: PathBuf,

    /// Maximum rows per shard in the merged output (0 or omit => single steps.npy)
    #[arg(long, value_name = "N")]
    shard_rows: Option<usize>,

    /// Overwrite existing outputs if present
    #[arg(long)]
    overwrite: bool,

    /// Delete the source dataset directories after a successful merge
    #[arg(long)]
    delete_inputs: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    match cli.command {
        Command::Pack(args) => {
            let options = PackOptions {
                input_root: args.input,
                output_dir: args.output,
                rows_per_shard: args.shard_rows.filter(|&n| n > 0),
                max_workers: args.workers,
                overwrite: args.overwrite,
            };
            let summary = pack_dataset(options)?;
            info!(
                "Completed packing: {} runs, {} steps, {} shard(s)",
                summary.runs, summary.steps, summary.shards
            );
        }
        Command::Merge(args) => {
            let options = MergeOptions {
                left_dir: args.left,
                right_dir: args.right,
                output_dir: args.output,
                rows_per_shard: args.shard_rows.filter(|&n| n > 0),
                overwrite: args.overwrite,
                delete_inputs: args.delete_inputs,
            };
            let summary = merge_datasets(options)?;
            info!(
                "Merged datasets: {} runs, {} steps, {} shard(s)",
                summary.runs, summary.steps, summary.shards
            );
        }
    }

    Ok(())
}
