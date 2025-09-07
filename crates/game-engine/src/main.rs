mod config;
mod grpc;

use clap::Parser;
use std::path::PathBuf;

use config::Config;

#[derive(Parser, Debug)]
struct Args {
    /// Path to configuration file
    #[arg(long, value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    config: PathBuf,
}

fn main() {
    println!("Hello, world!");
    let args = Args::parse();
    println!("Using configuration file: {}", args.config.display());
    let config = Config::from_toml(&args.config).unwrap();
    // gRPC server bootstrap will be added once backend is implemented.
    // For now, we just ensure the gRPC module compiles and is linkable.
    let _ = config;
}
