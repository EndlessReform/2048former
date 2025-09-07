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
