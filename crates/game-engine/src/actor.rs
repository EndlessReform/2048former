use crate::feeder::{Bins, FeederHandle};
use crate::config;
use ai_2048::engine as GameEngine;
use ai_2048::engine::{Board, Move};
use rand::SeedableRng;
use rand::distributions::Distribution;
// use tokio::task::JoinHandle;

/// Per-game actor that drives a single board to completion by
/// querying the model via the Feeder and applying selected moves.
pub struct GameActor {
    pub game_id: u32,
    pub handle: FeederHandle,
    pub board: Board,
    pub seed: u64,
    pub sampling: config::SamplingStrategy,
}

pub struct GameResult {
    pub game_id: u32,
    pub seed: u64,
    pub steps: u64,
    pub score: u64,
    pub highest_tile: u32,
}

impl GameActor {
    pub fn new(game_id: u32, handle: FeederHandle, seed: u64, sampling: config::SamplingStrategy) -> Self {
        // Initialize a fresh board with two random tiles
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut board: Board = Board::EMPTY;
        board = board.with_random_tile(&mut rng);
        board = board.with_random_tile(&mut rng);
        Self {
            game_id,
            handle,
            board,
            seed,
            sampling,
        }
    }

    /// Run the actor loop to completion and return the result.
    pub async fn run(mut self) -> GameResult {
        let mut steps: u64 = 0;
        let mut seq: u64 = 0;
        let mut rng = rand::rngs::StdRng::from_entropy();

        while !self.board.is_game_over() {
            let id = ((self.game_id as u64) << 32) | seq;
            let board_bytes = board_to_exponents(self.board);
            let rx = self.handle.submit(id, self.game_id, board_bytes).await;
            let bins = match rx.await {
                Ok(Ok(b)) => b,
                Ok(Err(_status)) => {
                    // On RPC failure, end the game; outer orchestrator may decide to retry/restart.
                    break;
                }
                Err(_canceled) => {
                    break;
                }
            };
            // Compute legal mask and select move according to configured sampling strategy
            let legal = legal_mask(self.board);
            // Gate non-argmax sampling by steps: before start_gate or at/after stop_gate -> argmax
            let start_gate = self.sampling.start_gate_or_default();
            let stop_gate = self.sampling.stop_gate();
            let outside_window = (steps < start_gate)
                || (stop_gate.map(|s| steps >= s).unwrap_or(false));
            let mv = if matches!(self.sampling.kind, config::SamplingStrategyKind::Argmax) {
                // Strategy is argmax; no gating effect
                select_move(&bins, &legal, &self.sampling, &mut rng)
            } else if outside_window {
                // Force argmax outside the sampling window
                select_move_max_p1(&bins, &legal)
            } else {
                // Within window: apply configured non-argmax strategy
                select_move(&bins, &legal, &self.sampling, &mut rng)
            };
            if let Some(m) = mv {
                // Apply move and spawn new tile using the Board API
                self.board = self.board.make_move(m, &mut rng);
                steps += 1;
            } else {
                break;
            }
            seq += 1;
        }

        GameResult {
            game_id: self.game_id,
            seed: self.seed,
            steps,
            score: self.board.score(),
            highest_tile: self.board.highest_tile() as u32,
        }
    }
}

fn board_to_exponents(b: Board) -> [u8; 16] {
    // ai_2048 packs 16 nibbles in a u64. To match the Python `to_exponents()`
    // row-major ordering expected by the model, map MSB-first to (row, col).
    // That is, nibble 15 -> index 0 (top-left), ..., nibble 0 -> index 15 (bottom-right).
    let raw = b.raw();
    let mut out = [0u8; 16];
    for idx in 0..16 {
        let nib = 15 - idx; // MSB-first
        out[idx] = ((raw >> (nib * 4)) & 0xF) as u8;
    }
    out
}

fn legal_mask(board: Board) -> [bool; 4] {
    let dirs = [Move::Up, Move::Down, Move::Left, Move::Right];
    let mut mask = [false; 4];
    for (i, &m) in dirs.iter().enumerate() {
        let after = GameEngine::make_move(board, m);
        mask[i] = after.raw() != board.raw();
    }
    mask
}

fn select_move_max_p1(bins: &Bins, legal: &[bool; 4]) -> Option<Move> {
    if bins.len() != 4 {
        return None;
    }
    let n_bins = bins[0].len();
    if n_bins == 0 {
        return None;
    }
    let one_idx = n_bins - 1; // default: dedicated '1' bin at the end
    let mut best_i: Option<usize> = None;
    let mut best_v: f32 = f32::NEG_INFINITY;
    for (i, head) in bins.iter().enumerate() {
        if !legal[i] {
            continue;
        }
        let p1 = *head.get(one_idx).unwrap_or(&0.0);
        if p1 > best_v {
            best_v = p1;
            best_i = Some(i);
        }
    }
    // If none legal (should imply game over), fallback to global argmax
    let idx = best_i.or_else(|| {
        let mut bi: usize = 0;
        let mut bv = f32::NEG_INFINITY;
        for (i, head) in bins.iter().enumerate() {
            let p1 = *head.get(one_idx).unwrap_or(&0.0);
            if p1 > bv {
                bv = p1;
                bi = i;
            }
        }
        Some(bi)
    })?;
    Some(match idx {
        0 => Move::Up,
        1 => Move::Down,
        2 => Move::Left,
        _ => Move::Right,
    })
}

fn select_move_softmax(bins: &Bins, legal: &[bool; 4], temperature: f32, rng: &mut rand::rngs::StdRng) -> Option<Move> {
    if bins.len() != 4 { return None; }
    let n_bins = bins[0].len();
    if n_bins == 0 { return None; }
    let one_idx = n_bins - 1; // dedicated '1' bin at the end

    // Collect p1 for legal moves only
    let mut p1 = [0.0f32; 4];
    let mut any_legal = false;
    for (i, head) in bins.iter().enumerate() {
        if !legal[i] { continue; }
        any_legal = true;
        p1[i] = *head.get(one_idx).unwrap_or(&0.0);
    }

    if !any_legal { return select_move_max_p1(bins, legal); }

    let t = if temperature.is_finite() && temperature > 0.0 { temperature } else { 1.0 };

    // Treat provided values as probabilities and apply temperature via log-probs:
    // weights ∝ exp((ln p) / t) == p^(1/t). Use max ln for stability.
    let mut max_ln = f32::NEG_INFINITY;
    for &p in &p1 {
        if p > 0.0 && p.is_finite() { let ln = p.ln(); if ln > max_ln { max_ln = ln; } }
    }
    let mut weights = [0.0f64; 4];
    for (i, &p) in p1.iter().enumerate() {
        if !legal[i] || !(p > 0.0) { continue; }
        let ln = p.ln();
        let z = ((ln - max_ln) / t) as f64;
        weights[i] = z.exp();
    }

    // If all weights are zero (numeric underflow), fallback to argmax
    if weights.iter().all(|&w| w == 0.0) {
        return select_move_max_p1(bins, legal);
    }

    // Sample index according to weights
    let dist = match rand::distributions::WeightedIndex::new(&weights) {
        Ok(d) => d,
        Err(_) => return select_move_max_p1(bins, legal),
    };
    let idx = dist.sample(rng);
    Some(match idx { 0 => Move::Up, 1 => Move::Down, 2 => Move::Left, _ => Move::Right })
}

fn select_move_top_p_top_k(bins: &Bins, legal: &[bool; 4], top_p: f32, top_k: usize, temperature: f32, rng: &mut rand::rngs::StdRng) -> Option<Move> {
    if bins.len() != 4 { return None; }
    let n_bins = bins[0].len();
    if n_bins == 0 { return None; }
    let one_idx = n_bins - 1;

    // Collect p1 across legal heads
    let mut scores = [0.0f64; 4];
    let mut legal_count = 0usize;
    for (i, head) in bins.iter().enumerate() {
        if !legal[i] { continue; }
        let p1 = *head.get(one_idx).unwrap_or(&0.0);
        scores[i] = p1 as f64;
        legal_count += 1;
    }
    if legal_count == 0 { return None; }

    // If only one legal, take it
    if legal_count == 1 {
        let idx = scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i)?;
        return Some(match idx { 0 => Move::Up, 1 => Move::Down, 2 => Move::Left, _ => Move::Right });
    }

    // Normalize scores to distribution over legal heads
    let sum_s: f64 = scores.iter().sum();
    if sum_s <= 0.0 {
        // degenerate: fallback to argmax p1
        return select_move_max_p1(bins, legal);
    }
    let mut w: Vec<(usize, f64)> = (0..4).filter(|&i| legal[i]).map(|i| (i, scores[i] / sum_s)).collect();
    // Sort descending by weight
    w.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Build nucleus up to cumulative >= top_p
    let p = top_p.clamp(0.0, 1.0) as f64;
    let mut nucleus: Vec<(usize, f64)> = Vec::with_capacity(4);
    let mut cum = 0.0f64;
    for (i, wi) in w.into_iter() {
        if wi <= 0.0 { continue; }
        nucleus.push((i, wi));
        cum += wi;
        if cum >= p { break; }
    }
    if nucleus.is_empty() {
        // Shouldn't happen if sum_s>0; fallback
        return select_move_max_p1(bins, legal);
    }

    // Cap to top-k (default 2)
    let k = top_k.max(1).min(nucleus.len());
    nucleus.truncate(k);

    // Temperature shaping: weights ∝ exp((ln w)/T)
    let t = if temperature.is_finite() && temperature > 0.0 { temperature } else { 1.0 } as f64;
    let mut max_ln = f64::NEG_INFINITY;
    for &(_, wi) in &nucleus { let lnw = wi.ln(); if lnw > max_ln { max_ln = lnw; } }
    let mut weights: Vec<f64> = Vec::with_capacity(nucleus.len());
    for &(_, wi) in &nucleus {
        let lnw = wi.ln();
        let z = (lnw - max_ln) / t;
        weights.push(z.exp());
    }
    // If all weights underflow, fallback to argmax p1
    if weights.iter().all(|&x| x == 0.0) {
        return select_move_max_p1(bins, legal);
    }
    let dist = match rand::distributions::WeightedIndex::new(&weights) {
        Ok(d) => d,
        Err(_) => return select_move_max_p1(bins, legal),
    };
    let pick = dist.sample(rng);
    let idx = nucleus[pick].0;
    Some(match idx { 0 => Move::Up, 1 => Move::Down, 2 => Move::Left, _ => Move::Right })
}

fn select_move_tail_agg_simple(bins: &Bins, legal: &[bool; 4], alpha_p2: f32, beta_p3: f32) -> Option<Move> {
    if bins.len() != 4 { return None; }
    let n_bins = bins[0].len();
    if n_bins == 0 { return None; }
    let one_idx = n_bins.saturating_sub(1);
    let two_idx = n_bins.saturating_sub(2);
    let three_idx = n_bins.saturating_sub(3);

    let mut best_i: Option<usize> = None;
    let mut best_s: f32 = f32::NEG_INFINITY;
    for (i, head) in bins.iter().enumerate() {
        if !legal[i] { continue; }
        let p1 = *head.get(one_idx).unwrap_or(&0.0);
        let p2 = if two_idx < n_bins { *head.get(two_idx).unwrap_or(&0.0) } else { 0.0 };
        let p3 = if three_idx < n_bins { *head.get(three_idx).unwrap_or(&0.0) } else { 0.0 };
        let s = p1 + alpha_p2 * p2 + beta_p3 * p3;
        if s > best_s { best_s = s; best_i = Some(i); }
    }
    let idx = best_i?;
    Some(match idx { 0 => Move::Up, 1 => Move::Down, 2 => Move::Left, _ => Move::Right })
}

fn select_move_tail_agg_adv(bins: &Bins, legal: &[bool; 4], extra_bins: usize, decay: f32) -> Option<Move> {
    if bins.len() != 4 { return None; }
    let n_bins = bins[0].len();
    if n_bins == 0 { return None; }
    let one_idx = n_bins - 1; // p1
    let max_extra = extra_bins.min(one_idx); // cannot include below first bin

    let mut best_i: Option<usize> = None;
    let mut best_s: f32 = f32::NEG_INFINITY;
    for (i, head) in bins.iter().enumerate() {
        if !legal[i] { continue; }
        let mut s = *head.get(one_idx).unwrap_or(&0.0); // always include p1 with weight 1
        // Add p2.. with geometric decay: weight for p2 is 1.0, p3 is decay, p4 is decay^2, ...
        let mut w = 1.0f32;
        for j in 1..=max_extra { // j=1 -> p2
            let idx = one_idx.saturating_sub(j);
            let pj = *head.get(idx).unwrap_or(&0.0);
            s += w * pj;
            w *= decay.max(0.0).min(1.0);
            if w == 0.0 { break; }
        }
        if s > best_s { best_s = s; best_i = Some(i); }
    }
    let idx = best_i?;
    Some(match idx { 0 => Move::Up, 1 => Move::Down, 2 => Move::Left, _ => Move::Right })
}

fn select_move_tail_agg_conf(bins: &Bins, legal: &[bool; 4], alpha: f32, beta: f32, gamma: f32) -> Option<Move> {
    if bins.len() != 4 { return None; }
    let n_bins = bins[0].len();
    if n_bins < 2 { return select_move_max_p1(bins, legal); }
    let one_idx = n_bins - 1; // p1
    let two_idx = n_bins - 2; // p2

    let mut best_i: Option<usize> = None;
    let mut best_s: f32 = f32::NEG_INFINITY;
    for (i, head) in bins.iter().enumerate() {
        if !legal[i] { continue; }
        let p1 = *head.get(one_idx).unwrap_or(&0.0);
        let p2 = *head.get(two_idx).unwrap_or(&0.0);
        // Margin between top bin and second bin (confidence proxy)
        let mut m = p1 - p2;
        if !m.is_finite() { m = 0.0; }
        if m < 0.0 { m = 0.0; }
        // w(m) = alpha / (1 + beta*m)^gamma
        let a = if alpha.is_finite() && alpha >= 0.0 { alpha } else { 0.20 };
        let b = if beta.is_finite() && beta >= 0.0 { beta } else { 10.0 };
        let g = if gamma.is_finite() && gamma > 0.0 { gamma } else { 1.0 };
        let denom = 1.0 + b * m;
        let w = if denom > 0.0 { a / denom.powf(g) } else { a };
        let s = p1 + w * p2;
        if s > best_s { best_s = s; best_i = Some(i); }
    }
    let idx = best_i?;
    Some(match idx { 0 => Move::Up, 1 => Move::Down, 2 => Move::Left, _ => Move::Right })
}

fn select_move(bins: &Bins, legal: &[bool; 4], sampling: &config::SamplingStrategy, rng: &mut rand::rngs::StdRng) -> Option<Move> {
    match sampling.kind {
        config::SamplingStrategyKind::Argmax => select_move_max_p1(bins, legal),
        config::SamplingStrategyKind::Softmax => {
            let t = sampling.temperature_or_default() as f32;
            select_move_softmax(bins, legal, t, rng)
        }
        config::SamplingStrategyKind::TopPTopK => {
            let p = sampling.top_p_or_default() as f32;
            let k = sampling.top_k_or_default();
            let t = sampling.temperature_or_default() as f32;
            select_move_top_p_top_k(bins, legal, p, k, t, rng)
        }
        config::SamplingStrategyKind::TailAgg => {
            let extra = sampling.tail_bins_or_zero();
            if extra > 0 {
                let decay = sampling.tail_decay_or_default() as f32;
                select_move_tail_agg_adv(bins, legal, extra, decay)
            } else {
                let a = sampling.alpha_p2_or_default() as f32;
                let b = sampling.beta_p3_or_default() as f32;
                select_move_tail_agg_simple(bins, legal, a, b)
            }
        }
        config::SamplingStrategyKind::TailAggConf => {
            let a = sampling.conf_alpha_or_default() as f32;
            let b = sampling.conf_beta_or_default() as f32;
            let g = sampling.conf_gamma_or_default() as f32;
            select_move_tail_agg_conf(bins, legal, a, b, g)
        }
    }
}
