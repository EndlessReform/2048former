use std::io::Read;

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub enum SamplingStrategyKind {
    Argmax,
    // Future strategies can be added here, e.g.
    // SoftMax { temperature: f64 },
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub struct SamplingStrategy {
    #[serde(rename = "strategy")]
    pub kind: SamplingStrategyKind,

    // Optional parameters that belong to specific variants.
    // If a variant does not need them, they can be omitted in the TOML
    // and will default to `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub struct Config {
    pub num_seeds: u32,
    pub max_concurrent_games: u32,
    pub max_retries: u32,

    pub sampling: SamplingStrategy,

    // Group orchestrator-specific settings under one nested key.
    #[serde(default)]
    pub orchestrator: Orchestrator,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub struct Orchestrator {
    #[serde(default)]
    pub connection: Connection,
    #[serde(default)]
    pub batch: Batch,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, Default)]
pub struct Connection {
    #[serde(default)]
    pub uds_path: Option<std::path::PathBuf>,
    #[serde(default)]
    pub tcp_addr: Option<String>,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub struct Batch {
    #[serde(default = "defaults::flush_us")]
    pub flush_us: u64,
    #[serde(default = "defaults::target_batch")]
    pub target_batch: usize,
    #[serde(default = "defaults::max_batch")]
    pub max_batch: usize,
    #[serde(default = "defaults::inflight_batches")]
    pub inflight_batches: usize,
    #[serde(default = "defaults::per_game_inflight")]
    pub per_game_inflight: usize,
    #[serde(default = "defaults::queue_cap")]
    pub queue_cap: usize,
    #[serde(default)]
    pub metrics_file: Option<std::path::PathBuf>,
    #[serde(default = "defaults::metrics_interval_s")]
    pub metrics_interval_s: f64,
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self { connection: Connection::default(), batch: Batch::default() }
    }
}

impl Default for Batch {
    fn default() -> Self {
        Self {
            flush_us: defaults::flush_us(),
            target_batch: defaults::target_batch(),
            max_batch: defaults::max_batch(),
            inflight_batches: defaults::inflight_batches(),
            per_game_inflight: defaults::per_game_inflight(),
            queue_cap: defaults::queue_cap(),
            metrics_file: None,
            metrics_interval_s: defaults::metrics_interval_s(),
        }
    }
}

impl Config {
    pub fn from_toml<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = std::fs::File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let cfg: Self = toml::from_str(&contents)?;
        Ok(cfg)
    }
}

mod defaults {
    pub fn flush_us() -> u64 { 250 }
    pub fn target_batch() -> usize { 512 }
    pub fn max_batch() -> usize { 1024 }
    pub fn inflight_batches() -> usize { 2 }
    pub fn per_game_inflight() -> usize { 16 }
    pub fn queue_cap() -> usize { 65_536 }
    pub fn metrics_interval_s() -> f64 { 5.0 }
}
