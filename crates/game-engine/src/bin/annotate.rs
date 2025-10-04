use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use game_engine::{annotation, config};

#[derive(Parser, Debug)]
struct Args {
    /// Path to orchestrator TOML config
    #[arg(long, value_name = "FILE")]
    config: PathBuf,
    /// Input dataset directory (Macroxue or self-play v1)
    #[arg(long, value_name = "DIR")]
    dataset: PathBuf,
    /// Output directory for annotation shards
    #[arg(long, value_name = "DIR")]
    output: PathBuf,
    /// Overwrite existing outputs
    #[arg(long)]
    overwrite: bool,
    /// Optional: limit number of steps to annotate (for smoke tests)
    #[arg(long, value_name = "N")]
    limit: Option<usize>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let cfg = config::AnnotationConfig::from_toml(&args.config)
        .map_err(|e| anyhow::anyhow!("failed to load config: {e}"))?;
    annotation::run(annotation::JobConfig {
        dataset_dir: args.dataset,
        output_dir: args.output,
        overwrite: args.overwrite,
        limit: args.limit,
        config: cfg,
    })
    .await
}
