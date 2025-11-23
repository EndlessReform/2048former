use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue};

const NUM_BRANCHES: usize = 4;
const SEARCH_VALUE_SCALE: f64 = 1000.0;
const SEARCH_ZERO_ATOL: f64 = 0.5;
const DEFAULT_ZERO_TOLERANCE: f64 = 1e-9;
const TOKEN_ILLEGAL: u16 = 0;
const TOKEN_FAILURE: u16 = 1;
const TOKEN_OFFSET: u16 = 2;

/// Token indices returned by a tokenizer implementation.
pub type TokenArray = [u16; NUM_BRANCHES];

/// Trait implemented by concrete tokenizers capable of binning branch EVs.
pub trait Tokenizer {
    /// Unique identifier for the tokenizer family (e.g. "macroxue_ev_advantage_v2").
    fn tokenizer_type(&self) -> &str;
    /// Number of disadvantage bins exposed in the shared vocabulary.
    fn num_bins(&self) -> usize;
    /// Canonical vocabulary order shared across valuation families.
    fn vocab_order(&self) -> &[String];
    /// Valuation family names supported by this tokenizer.
    fn valuation_types(&self) -> &[String];

    /// Encode a single step into per-branch token IDs.
    fn encode_row(
        &self,
        valuation_type: &str,
        branch_evs: &[f32; NUM_BRANCHES],
        move_dir: u8,
        legal_mask: u8,
        board_eval: Option<i32>,
    ) -> Result<TokenArray>;
}

fn default_metadata() -> JsonMap<String, JsonValue> {
    JsonMap::new()
}

/// Quantile configuration for a specific valuation family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MacroxueTokenizerV2TypeConfig {
    pub bin_edges: Vec<f64>,
    #[serde(default)]
    pub failure_cutoff: Option<i32>,
}

/// Serialized configuration for the advantage-based tokenizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MacroxueTokenizerV2Spec {
    pub tokenizer_type: String,
    pub version: u32,
    pub num_bins: u32,
    pub vocab_order: Vec<String>,
    pub valuation_types: Vec<String>,
    pub search: MacroxueTokenizerV2TypeConfig,
    pub tuple10: MacroxueTokenizerV2TypeConfig,
    pub tuple11: MacroxueTokenizerV2TypeConfig,
    #[serde(default = "default_metadata")]
    pub metadata: JsonMap<String, JsonValue>,
}

impl MacroxueTokenizerV2Spec {
    /// Load a tokenizer specification from a JSON reader.
    pub fn from_reader<R: Read>(reader: R) -> Result<Self> {
        serde_json::from_reader(reader).context("failed to deserialize tokenizer spec")
    }

    /// Load a tokenizer specification from a JSON file on disk.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        Self::from_reader(file)
    }
}

/// Advantage-based tokenizer (v2) mirroring the Python implementation.
#[derive(Debug, Clone)]
pub struct MacroxueTokenizerV2 {
    spec: MacroxueTokenizerV2Spec,
    num_bins: usize,
    token_illegal: u16,
    token_failure: u16,
    token_offset: u16,
    token_winner: u16,
    zero_bin: u16,
    search_failure_cutoff: f64,
    zero_tolerance: f64,
    search_edges: Arc<[f64]>,
    tuple10_edges: Arc<[f64]>,
    tuple11_edges: Arc<[f64]>,
}

impl MacroxueTokenizerV2 {
    /// Construct the tokenizer from a parsed specification using the default zero tolerance.
    pub fn from_spec(spec: MacroxueTokenizerV2Spec) -> Result<Self> {
        Self::with_zero_tolerance(spec, DEFAULT_ZERO_TOLERANCE)
    }

    /// Construct the tokenizer with a custom zero tolerance for tuple valuations.
    pub fn with_zero_tolerance(spec: MacroxueTokenizerV2Spec, zero_tolerance: f64) -> Result<Self> {
        ensure!(zero_tolerance >= 0.0, "zero_tolerance must be non-negative");

        let num_bins = usize::try_from(spec.num_bins)
            .with_context(|| format!("num_bins {} does not fit usize", spec.num_bins))?;
        ensure!(num_bins > 0, "num_bins must be positive");
        ensure!(
            num_bins <= (u16::MAX as usize) - 2,
            "num_bins {num_bins} would overflow token ids"
        );

        let search_edges_vec = spec.search.bin_edges.clone();
        let tuple10_edges_vec = spec.tuple10.bin_edges.clone();
        let tuple11_edges_vec = spec.tuple11.bin_edges.clone();

        validate_edges("search", &search_edges_vec, num_bins)?;
        validate_edges("tuple10", &tuple10_edges_vec, num_bins)?;
        validate_edges("tuple11", &tuple11_edges_vec, num_bins)?;

        let search_failure_cutoff = spec
            .search
            .failure_cutoff
            .ok_or_else(|| anyhow::anyhow!("search failure cutoff must be present in spec"))?
            as f64;

        let token_illegal = TOKEN_ILLEGAL;
        let token_failure = TOKEN_FAILURE;
        let token_offset = TOKEN_OFFSET;
        let token_winner = token_offset + num_bins as u16;
        let zero_bin = token_offset + (num_bins as u16 - 1);

        Ok(Self {
            spec,
            num_bins,
            token_illegal,
            token_failure,
            token_offset,
            token_winner,
            zero_bin,
            search_failure_cutoff,
            zero_tolerance,
            search_edges: Arc::from(search_edges_vec.into_boxed_slice()),
            tuple10_edges: Arc::from(tuple10_edges_vec.into_boxed_slice()),
            tuple11_edges: Arc::from(tuple11_edges_vec.into_boxed_slice()),
        })
    }

    /// Load the tokenizer from a specification file on disk.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let spec = MacroxueTokenizerV2Spec::from_path(path)?;
        Self::from_spec(spec)
    }

    /// Return the parsed specification used to construct this tokenizer.
    pub fn spec(&self) -> &MacroxueTokenizerV2Spec {
        &self.spec
    }

    #[inline]
    fn zero_bin_token(&self) -> u16 {
        self.zero_bin
    }

    #[inline]
    fn edges_for(&self, kind: ValuationKind) -> &[f64] {
        match kind {
            ValuationKind::Search => &self.search_edges,
            ValuationKind::Tuple10 => &self.tuple10_edges,
            ValuationKind::Tuple11 => &self.tuple11_edges,
        }
    }

    fn encode_search(
        &self,
        tokens: &mut TokenArray,
        branch_values: &[f64; NUM_BRANCHES],
        move_dir: usize,
        legal: &[bool; NUM_BRANCHES],
        board_eval: f64,
    ) {
        let edges = self.edges_for(ValuationKind::Search);
        let zero_bin = self.zero_bin_token();

        let mut scaled = [0.0f64; NUM_BRANCHES];
        for i in 0..NUM_BRANCHES {
            scaled[i] = (branch_values[i] * SEARCH_VALUE_SCALE).round();
        }

        let mut advantages = [0.0f64; NUM_BRANCHES];
        for i in 0..NUM_BRANCHES {
            advantages[i] = scaled[i] - board_eval;
        }

        let mut failure_mask = [false; NUM_BRANCHES];
        for i in 0..NUM_BRANCHES {
            if legal[i] && advantages[i] < self.search_failure_cutoff {
                failure_mask[i] = true;
            }
        }
        failure_mask[move_dir] = false;

        for i in 0..NUM_BRANCHES {
            if failure_mask[i] {
                tokens[i] = self.token_failure;
            }
        }

        let winner_adv = advantages[move_dir];
        let mut disadvantages = [0.0f64; NUM_BRANCHES];
        for i in 0..NUM_BRANCHES {
            disadvantages[i] = advantages[i] - winner_adv;
        }

        let mut zero_mask = [false; NUM_BRANCHES];
        for i in 0..NUM_BRANCHES {
            zero_mask[i] = disadvantages[i].abs() <= SEARCH_ZERO_ATOL;
        }
        zero_mask[move_dir] = false;
        for i in 0..NUM_BRANCHES {
            if !legal[i] {
                zero_mask[i] = true;
            }
        }

        let mut remainder_mask = [false; NUM_BRANCHES];
        for i in 0..NUM_BRANCHES {
            remainder_mask[i] = legal[i] && !failure_mask[i] && !zero_mask[i];
        }
        remainder_mask[move_dir] = false;

        for i in 0..NUM_BRANCHES {
            if zero_mask[i] && legal[i] {
                tokens[i] = zero_bin;
            }
        }

        for i in 0..NUM_BRANCHES {
            if remainder_mask[i] {
                let idx = searchsorted_bin(edges, disadvantages[i]);
                tokens[i] = self.token_offset + idx as u16;
            }
        }
    }

    fn encode_tuple(
        &self,
        tokens: &mut TokenArray,
        branch_values: &[f64; NUM_BRANCHES],
        move_dir: usize,
        legal: &[bool; NUM_BRANCHES],
        edges: &[f64],
    ) {
        let zero_bin = self.zero_bin_token();
        let winner_val = branch_values[move_dir];
        let mut disadvantages = [0.0f64; NUM_BRANCHES];
        for i in 0..NUM_BRANCHES {
            disadvantages[i] = branch_values[i] - winner_val;
        }

        let mut failure_mask = [false; NUM_BRANCHES];
        for i in 0..NUM_BRANCHES {
            if legal[i] && branch_values[i] <= self.zero_tolerance {
                failure_mask[i] = true;
            }
        }
        failure_mask[move_dir] = false;

        for i in 0..NUM_BRANCHES {
            if failure_mask[i] {
                tokens[i] = self.token_failure;
            }
        }

        let mut zero_mask = [false; NUM_BRANCHES];
        for i in 0..NUM_BRANCHES {
            zero_mask[i] = disadvantages[i].abs() <= self.zero_tolerance;
        }
        zero_mask[move_dir] = false;
        for i in 0..NUM_BRANCHES {
            if !legal[i] {
                zero_mask[i] = true;
            }
        }

        for i in 0..NUM_BRANCHES {
            if zero_mask[i] {
                tokens[i] = zero_bin;
            }
        }

        let mut remainder_mask = [false; NUM_BRANCHES];
        for i in 0..NUM_BRANCHES {
            remainder_mask[i] = legal[i] && !failure_mask[i] && !zero_mask[i];
        }
        remainder_mask[move_dir] = false;

        for i in 0..NUM_BRANCHES {
            if remainder_mask[i] {
                let idx = searchsorted_bin(edges, disadvantages[i]);
                tokens[i] = self.token_offset + idx as u16;
            }
        }
    }
}

impl Tokenizer for MacroxueTokenizerV2 {
    fn tokenizer_type(&self) -> &str {
        &self.spec.tokenizer_type
    }

    fn num_bins(&self) -> usize {
        self.num_bins
    }

    fn vocab_order(&self) -> &[String] {
        &self.spec.vocab_order
    }

    fn valuation_types(&self) -> &[String] {
        &self.spec.valuation_types
    }

    fn encode_row(
        &self,
        valuation_type: &str,
        branch_evs: &[f32; NUM_BRANCHES],
        move_dir: u8,
        legal_mask: u8,
        board_eval: Option<i32>,
    ) -> Result<TokenArray> {
        let kind = ValuationKind::from_str(valuation_type)?;
        let move_dir = move_dir as usize;
        ensure!(move_dir < NUM_BRANCHES, "move_dir {move_dir} out of range");

        let legal = decode_legal_mask(legal_mask);
        ensure!(
            legal.iter().any(|&flag| flag),
            "legal mask has no legal branches"
        );

        let mut tokens = initial_tokens(
            self.token_illegal,
            self.token_failure,
            self.token_winner,
            &legal,
            move_dir,
        )?;

        let branch_values: [f64; NUM_BRANCHES] = (*branch_evs).map(|v| v as f64);

        match kind {
            ValuationKind::Search => {
                let board_eval = board_eval.map(|v| v as f64).ok_or_else(|| {
                    anyhow::anyhow!("board_eval is required for search tokenization")
                })?;
                self.encode_search(&mut tokens, &branch_values, move_dir, &legal, board_eval);
            }
            ValuationKind::Tuple10 => {
                self.encode_tuple(
                    &mut tokens,
                    &branch_values,
                    move_dir,
                    &legal,
                    self.edges_for(ValuationKind::Tuple10),
                );
            }
            ValuationKind::Tuple11 => {
                self.encode_tuple(
                    &mut tokens,
                    &branch_values,
                    move_dir,
                    &legal,
                    self.edges_for(ValuationKind::Tuple11),
                );
            }
        }

        Ok(tokens)
    }
}

#[derive(Debug, Clone, Copy)]
enum ValuationKind {
    Search,
    Tuple10,
    Tuple11,
}

impl ValuationKind {
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "search" => Ok(Self::Search),
            "tuple10" => Ok(Self::Tuple10),
            "tuple11" => Ok(Self::Tuple11),
            other => bail!("unknown valuation_type '{other}'"),
        }
    }
}

fn validate_edges(label: &str, edges: &[f64], num_bins: usize) -> Result<()> {
    ensure!(
        edges.len() == num_bins + 1,
        "{} bin_edges must contain num_bins + 1 entries (got {} vs expected {})",
        label,
        edges.len(),
        num_bins + 1
    );
    for window in edges.windows(2) {
        ensure!(
            window[1] > window[0],
            "{} bin_edges must be strictly increasing",
            label
        );
    }
    Ok(())
}

fn initial_tokens(
    token_illegal: u16,
    token_failure: u16,
    token_winner: u16,
    legal: &[bool; NUM_BRANCHES],
    move_dir: usize,
) -> Result<TokenArray> {
    ensure!(move_dir < NUM_BRANCHES, "move_dir {move_dir} out of range");
    let mut tokens = [token_illegal; NUM_BRANCHES];
    for i in 0..NUM_BRANCHES {
        if legal[i] {
            tokens[i] = token_failure;
        }
    }
    if !legal[move_dir] {
        bail!("winner branch marked illegal");
    }
    tokens[move_dir] = token_winner;
    Ok(tokens)
}

fn decode_legal_mask(bits: u8) -> [bool; NUM_BRANCHES] {
    let mut out = [false; NUM_BRANCHES];
    for i in 0..NUM_BRANCHES {
        out[i] = (bits >> i) & 1 != 0;
    }
    out
}

fn searchsorted_bin(edges: &[f64], value: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = edges.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if edges[mid] <= value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let idx = lo.saturating_sub(1);
    idx.min(edges.len().saturating_sub(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_spec() -> MacroxueTokenizerV2Spec {
        MacroxueTokenizerV2Spec {
            tokenizer_type: "macroxue_ev_advantage_v2".to_string(),
            version: 2,
            num_bins: 4,
            vocab_order: vec![
                "ILLEGAL".into(),
                "FAILURE".into(),
                "BIN_0".into(),
                "BIN_1".into(),
                "BIN_2".into(),
                "BIN_3".into(),
                "WINNER".into(),
            ],
            valuation_types: vec!["search".into(), "tuple10".into(), "tuple11".into()],
            search: MacroxueTokenizerV2TypeConfig {
                bin_edges: vec![-5000.0, -1000.0, -200.0, -20.0, 0.0],
                failure_cutoff: Some(-1500),
            },
            tuple10: MacroxueTokenizerV2TypeConfig {
                bin_edges: vec![-2.0, -1.0, -0.2, -0.05, 0.0],
                failure_cutoff: None,
            },
            tuple11: MacroxueTokenizerV2TypeConfig {
                bin_edges: vec![-2.5, -1.5, -0.4, -0.05, 0.0],
                failure_cutoff: None,
            },
            metadata: JsonMap::new(),
        }
    }

    #[test]
    fn tuple_tokenization_assigns_failure_zero_and_bins() {
        let tokenizer =
            MacroxueTokenizerV2::from_spec(test_spec()).expect("failed to build tokenizer");
        let tokens = tokenizer
            .encode_row("tuple10", &[1.0, 0.2, 0.0, 0.0], 0, 0b0111, None)
            .expect("tuple tokenization should succeed");
        let winner_token = TOKEN_OFFSET + tokenizer.num_bins() as u16;
        let bin_token = TOKEN_OFFSET + 1;
        let failure_token = TOKEN_FAILURE;
        let zero_bin = TOKEN_OFFSET + tokenizer.num_bins() as u16 - 1;
        assert_eq!(
            tokens[0], winner_token,
            "winner branch should map to WINNER token"
        );
        assert_eq!(
            tokens[1], bin_token,
            "disadvantage should map into quantised bin"
        );
        assert_eq!(
            tokens[2], failure_token,
            "branches with zero value fall back to FAILURE"
        );
        assert_eq!(
            tokens[3], zero_bin,
            "illegal branches inherit zero-disadvantage bin"
        );
    }

    #[test]
    fn search_tokenization_bins_disadvantages() {
        let tokenizer =
            MacroxueTokenizerV2::from_spec(test_spec()).expect("failed to build tokenizer");
        let tokens = tokenizer
            .encode_row("search", &[0.90, 0.80, 0.60, 0.95], 3, 0b1111, Some(500))
            .expect("search tokenization should succeed");
        let winner_token = TOKEN_OFFSET + tokenizer.num_bins() as u16;
        let expected = [
            TOKEN_OFFSET + 2,
            TOKEN_OFFSET + 2,
            TOKEN_OFFSET + 1,
            winner_token,
        ];
        assert_eq!(tokens, expected);
    }

    #[test]
    fn search_requires_board_eval() {
        let tokenizer =
            MacroxueTokenizerV2::from_spec(test_spec()).expect("failed to build tokenizer");
        let err = tokenizer.encode_row("search", &[0.5, 0.4, 0.3, 0.6], 0, 0b1111, None);
        assert!(
            err.is_err(),
            "board_eval should be mandatory for search tokenization"
        );
    }

    #[test]
    fn search_winner_not_overwritten_by_failure_mask() {
        let tokenizer =
            MacroxueTokenizerV2::from_spec(test_spec()).expect("failed to build tokenizer");
        // Simulate: advantages = [-93, 0, -31, -74784], winner is branch 1
        // Branch 3 should be FAILURE (advantage < -1500)
        // Branches 0, 2 should be binned
        // Branch 1 should remain WINNER
        let tokens = tokenizer
            .encode_row(
                "search",
                &[0.407, 0.500, 0.469, -74.284], // scaled: [407, 500, 469, -74284]
                1,                               // winner
                0b1111,
                Some(500), // board_eval = 500
            )
            .expect("search tokenization should succeed");

        let winner_token = TOKEN_OFFSET + tokenizer.num_bins() as u16;

        // Expected: [4, 6, 4, 1] (matching Python implementation)
        assert_eq!(
            tokens[0], 4,
            "Branch 0: disadvantage -93 should map to bin 2 (token 4)"
        );
        assert_eq!(
            tokens[1], winner_token,
            "Branch 1: winner must be WINNER token (6)"
        );
        assert_eq!(
            tokens[2], 4,
            "Branch 2: disadvantage -31 should map to bin 2 (token 4)"
        );
        assert_eq!(
            tokens[3], TOKEN_FAILURE,
            "Branch 3: advantage -74784 should be FAILURE (1)"
        );
    }
}
