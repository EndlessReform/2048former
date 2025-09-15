use std::io::Read;

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub enum SamplingStrategyKind {
    Argmax,
    /// Softmax over bin-1 values across heads
    Softmax,
    /// Nucleus (top-p) over heads with an additional top-k cap (default k=2)
    TopPTopK,
    /// Tail aggregation over near-1 bins: score = p1 + alpha_p2*p2 + beta_p3*p3 (argmax)
    TailAgg,
    /// Tail aggregation with confidence-based weighting of p2 using margin (p1 - p2):
    /// score = p1 + w(margin) * p2, where w(m) = alpha / (1 + beta*m)^gamma.
    TailAggConf,
    // Future strategies can be added here
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

    /// Margin for gating on p1 across heads (0..1). If None, defaults per strategy.
    /// Nucleus threshold p in (0,1]; cumulative probability mass across heads to retain.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Cap the candidate set to at most K heads after applying top-p (default 2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,

    /// Tail aggregation weights for the second-to-last and third-to-last bins.
    /// If omitted, defaults are alpha_p2=0.02, beta_p3=0.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha_p2: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beta_p3: Option<f64>,

    /// Advanced tail aggregation: include this many bins below p1 (p2..p_{1+N})
    /// using a geometric decay `tail_decay` for weights. When set (>0), this
    /// takes precedence over alpha_p2/beta_p3.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tail_bins: Option<usize>,

    /// Geometric decay for tail bin weights (0..1]. Weight for p2 is 1.0,
    /// for p3 is decay, then decay^2, etc. Defaults to 0.5 if not set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tail_decay: Option<f64>,

    /// Optional gates for when to apply non-argmax sampling.
    /// If `start_gate` > 0, use pure argmax before this step count.
    /// If `stop_gate` is set, revert to argmax at and after this step count.
    /// Defaults: start_gate=0, stop_gate=None.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_gate: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_gate: Option<u64>,

    // Confidence-based tail aggregation parameters (TailAggConf)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conf_alpha: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conf_beta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conf_gamma: Option<f64>,
}

impl SamplingStrategy {
    /// Resolve temperature to a sane default (1.0) if not provided.
    pub fn temperature_or_default(&self) -> f64 {
        match self.temperature {
            Some(t) if t.is_finite() && t > 0.0 => t,
            _ => 1.0,
        }
    }

    /// Resolve top-p to a sane default (0.8) if not provided.
    pub fn top_p_or_default(&self) -> f64 {
        match self.top_p {
            Some(p) if p.is_finite() && p > 0.0 && p <= 1.0 => p,
            _ => 0.8,
        }
    }

    /// Resolve top-k to a sane default (2) if not provided; clamp to [1,4].
    pub fn top_k_or_default(&self) -> usize {
        let k = self.top_k.unwrap_or(2);
        k.max(1).min(4)
    }

    /// Resolve alpha (p2 weight) to default 0.02
    pub fn alpha_p2_or_default(&self) -> f64 {
        match self.alpha_p2 {
            Some(a) if a.is_finite() && a >= 0.0 => a,
            _ => 0.02,
        }
    }

    /// Resolve beta (p3 weight) to default 0.0
    pub fn beta_p3_or_default(&self) -> f64 {
        match self.beta_p3 {
            Some(b) if b.is_finite() && b >= 0.0 => b,
            _ => 0.0,
        }
    }

    /// Number of extra tail bins to include beyond p1 (p2..). 0 or None means disabled.
    pub fn tail_bins_or_zero(&self) -> usize {
        self.tail_bins.unwrap_or(0)
    }

    /// Decay for advanced tail aggregation. Defaults to 0.5.
    pub fn tail_decay_or_default(&self) -> f64 {
        match self.tail_decay {
            Some(d) if d.is_finite() && d > 0.0 && d <= 1.0 => d,
            _ => 0.5,
        }
    }

    /// Step count at which non-argmax sampling becomes enabled. Defaults to 0.
    pub fn start_gate_or_default(&self) -> u64 {
        self.start_gate.unwrap_or(0)
    }

    /// Optional step count at which to stop applying non-argmax sampling.
    pub fn stop_gate(&self) -> Option<u64> {
        self.stop_gate
    }

    /// Max p2 weight at zero margin (TailAggConf). Defaults to 0.20.
    pub fn conf_alpha_or_default(&self) -> f64 {
        match self.conf_alpha {
            Some(a) if a.is_finite() && a >= 0.0 => a,
            _ => 0.20,
        }
    }

    /// Slope for margin-to-weight mapping (TailAggConf). Defaults to 10.0.
    pub fn conf_beta_or_default(&self) -> f64 {
        match self.conf_beta {
            Some(b) if b.is_finite() && b >= 0.0 => b,
            _ => 10.0,
        }
    }

    /// Exponent shaping for margin-to-weight mapping (TailAggConf). Defaults to 1.0.
    pub fn conf_gamma_or_default(&self) -> f64 {
        match self.conf_gamma {
            Some(g) if g.is_finite() && g > 0.0 => g,
            _ => 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub struct Config {
    /// Optional: number of games to run. If omitted, the orchestrator will run until `max_steps` is reached.
    #[serde(default)]
    pub num_seeds: Option<u32>,
    /// Optional: total steps budget across all games. If set, the orchestrator stops once this is reached.
    #[serde(default)]
    pub max_steps: Option<u64>,
    /// Max per-actor games scheduled concurrently.
    pub max_concurrent_games: u32,
    /// Transient RPC retry attempts (batch-level).
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
    #[serde(default)]
    pub report: Report,
    /// If true, request embeddings from the server and write them to NPY shards.
    #[serde(default)]
    pub inline_embeddings: bool,
    /// Optional: use this as a fixed base seed for reproducibility. If omitted, seeds are fully random.
    #[serde(default)]
    pub fixed_seed: Option<u64>,
    /// Opt-out: when true, ignore fixed/default base seed and use full randomness.
    #[serde(default)]
    pub random_seeds: bool,
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
        Self {
            connection: Connection::default(),
            batch: Batch::default(),
            report: Report::default(),
            inline_embeddings: false,
            fixed_seed: None,
            random_seeds: false,
        }
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

#[derive(Clone, Debug, PartialEq, serde::Deserialize, Default)]
pub struct Report {
    #[serde(default)]
    pub results_file: Option<std::path::PathBuf>,
    #[serde(default)]
    pub session_dir: Option<std::path::PathBuf>,
    /// Optional: in-memory buffering cap to avoid OOM (approximate). If exceeded, stop collecting new rows.
    #[serde(default)]
    pub max_ram_mb: Option<usize>,
    /// Optional: disk failsafe for dataset artifacts. If estimated size exceeds this many GB, skip writing.
    #[serde(default)]
    pub max_gb: Option<f64>,
    /// Shard steps/embeddings NPY files after this many steps. Defaults to 1M.
    #[serde(default = "defaults::shard_max_steps")]
    pub shard_max_steps: usize,
}

impl Report {
    pub fn shard_max_steps_or_default(&self) -> usize {
        if self.shard_max_steps == 0 {
            defaults::shard_max_steps()
        } else {
            self.shard_max_steps
        }
    }
}

mod defaults {
    pub fn flush_us() -> u64 {
        250
    }
    pub fn shard_max_steps() -> usize {
        1_000_000
    }
    pub fn target_batch() -> usize {
        512
    }
    pub fn max_batch() -> usize {
        1024
    }
    pub fn inflight_batches() -> usize {
        2
    }
    pub fn per_game_inflight() -> usize {
        16
    }
    pub fn queue_cap() -> usize {
        65_536
    }
    pub fn metrics_interval_s() -> f64 {
        5.0
    }
}
