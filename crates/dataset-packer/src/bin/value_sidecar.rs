use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use dataset_packer::{ValueSidecarOptions, add_value_sidecar};
use env_logger::Env;
use log::info;

/// Compute per-step rewards and discounted returns for packed datasets.
#[derive(Debug, Parser)]
#[command(author, version, about = "Generate value sidecar aligned to steps.npy", long_about = None)]
struct Cli {
    /// Packed dataset directory containing steps.npy and metadata.db
    #[arg(long, value_name = "DIR")]
    dataset: PathBuf,

    /// Output directory for the value sidecar (defaults to dataset dir)
    #[arg(long, value_name = "DIR")]
    output: Option<PathBuf>,

    /// Discount factor used for the return (G_t = r_t + gamma * G_{t+1})
    #[arg(long, default_value_t = 0.997, value_name = "FLOAT")]
    gamma: f64,

    /// Scale applied to rewards before computing the scaled return
    #[arg(long, default_value_t = 1.0, value_name = "FLOAT")]
    reward_scale: f64,

    /// Optional override for Rayon worker count
    #[arg(long, value_name = "N")]
    workers: Option<usize>,

    /// Overwrite existing outputs when present
    #[arg(long)]
    overwrite: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let output_dir = cli.output.clone().unwrap_or_else(|| cli.dataset.clone());
    let workers = cli.workers.unwrap_or_else(num_cpus::get);

    let opts = ValueSidecarOptions {
        dataset_dir: cli.dataset,
        output_dir,
        gamma: cli.gamma,
        reward_scale: cli.reward_scale,
        max_workers: Some(workers),
        overwrite: cli.overwrite,
    };

    let summary = add_value_sidecar(opts)?;
    info!(
        "Value sidecar complete: {} runs, {} steps, {} shard(s), gamma={}, reward_scale={}",
        summary.runs, summary.steps, summary.shards, summary.gamma, summary.reward_scale
    );

    Ok(())
}
