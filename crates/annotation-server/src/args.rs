use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
pub struct Args {
    /// Path to the base Macroxue dataset directory (steps-*.npy + metadata.db).
    #[arg(long)]
    pub dataset: PathBuf,
    /// Path to the annotation directory containing annotations-*.npy.
    #[arg(long)]
    pub annotations: PathBuf,
    /// Optional path to the tokenizer JSON file for tokenization support.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// Optional path to the UI dist directory to serve static files.
    #[arg(long)]
    pub ui_path: Option<PathBuf>,
    /// Host interface to bind (default 0.0.0.0).
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,
    /// Port to bind (default 8080).
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    /// Optional tracing filter, e.g. "info", "debug".
    #[arg(long, default_value = "info")]
    pub log: String,
}
