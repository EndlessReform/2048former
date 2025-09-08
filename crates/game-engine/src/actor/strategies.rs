use crate::config;
use crate::feeder::Bins;
use ai_2048::engine::Move;
use rand::distributions::Distribution;

pub(crate) fn select_move_max_p1(bins: &Bins, legal: &[bool; 4]) -> Option<Move> {
    if bins.len() != 4 { return None; }
    let n_bins = bins[0].len();
    if n_bins == 0 { return None; }
    let one_idx = n_bins - 1; // dedicated '1' bin at end
    let mut best_i: Option<usize> = None;
    let mut best_v: f32 = f32::NEG_INFINITY;
    for (i, head) in bins.iter().enumerate() {
        if !legal[i] { continue; }
        let p1 = *head.get(one_idx).unwrap_or(&0.0);
        if p1 > best_v { best_v = p1; best_i = Some(i); }
    }
    let idx = best_i.or_else(|| {
        let mut bi: usize = 0; let mut bv = f32::NEG_INFINITY;
        for (i, head) in bins.iter().enumerate() { let p1 = *head.get(one_idx).unwrap_or(&0.0); if p1 > bv { bv = p1; bi = i; } }
        Some(bi)
    })?;
    Some(match idx { 0 => Move::Up, 1 => Move::Down, 2 => Move::Left, _ => Move::Right })
}

fn select_move_softmax(bins: &Bins, legal: &[bool; 4], temperature: f32, rng: &mut rand::rngs::StdRng) -> Option<Move> {
    if bins.len() != 4 { return None; }
    let n_bins = bins[0].len();
    if n_bins == 0 { return None; }
    let one_idx = n_bins - 1;
    let mut p1 = [0.0f32; 4];
    let mut any_legal = false;
    for (i, head) in bins.iter().enumerate() { if !legal[i] { continue; } any_legal = true; p1[i] = *head.get(one_idx).unwrap_or(&0.0); }
    if !any_legal { return select_move_max_p1(bins, legal); }
    let t = if temperature.is_finite() && temperature > 0.0 { temperature } else { 1.0 };
    let mut max_ln = f32::NEG_INFINITY;
    for &p in &p1 { if p > 0.0 && p.is_finite() { let ln = p.ln(); if ln > max_ln { max_ln = ln; } } }
    let mut weights = [0.0f64; 4];
    for (i, &p) in p1.iter().enumerate() { if !legal[i] || !(p > 0.0) { continue; } let ln = p.ln(); let z = ((ln - max_ln) / t) as f64; weights[i] = z.exp(); }
    if weights.iter().all(|&w| w == 0.0) { return select_move_max_p1(bins, legal); }
    let dist = match rand::distributions::WeightedIndex::new(&weights) { Ok(d) => d, Err(_) => return select_move_max_p1(bins, legal) };
    let idx = dist.sample(rng);
    Some(match idx { 0 => Move::Up, 1 => Move::Down, 2 => Move::Left, _ => Move::Right })
}

fn select_move_top_p_top_k(
    bins: &Bins,
    legal: &[bool; 4],
    top_p: f32,
    top_k: usize,
    temperature: f32,
    rng: &mut rand::rngs::StdRng,
) -> Option<Move> {
    if bins.len() != 4 { return None; }
    let n_bins = bins[0].len();
    if n_bins == 0 { return None; }
    let one_idx = n_bins - 1;
    let mut scores = [0.0f64; 4];
    let mut legal_count = 0usize;
    for (i, head) in bins.iter().enumerate() { if !legal[i] { continue; } let p1 = *head.get(one_idx).unwrap_or(&0.0); scores[i] = p1 as f64; legal_count += 1; }
    if legal_count == 0 { return None; }
    if legal_count == 1 { let idx = scores.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).map(|(i,_)| i)?; return Some(match idx { 0 => Move::Up, 1 => Move::Down, 2 => Move::Left, _ => Move::Right }); }
    let sum_s: f64 = scores.iter().sum();
    if sum_s <= 0.0 { return select_move_max_p1(bins, legal); }
    let mut w: Vec<(usize, f64)> = (0..4).filter(|&i| legal[i]).map(|i| (i, scores[i] / sum_s)).collect();
    w.sort_by(|a, b| b.1.total_cmp(&a.1));
    let p = top_p.clamp(0.0, 1.0) as f64;
    let mut nucleus: Vec<(usize, f64)> = Vec::with_capacity(4);
    let mut cum = 0.0f64;
    for (i, wi) in w.into_iter() { if wi <= 0.0 { continue; } nucleus.push((i, wi)); cum += wi; if cum >= p { break; } }
    if nucleus.is_empty() { return select_move_max_p1(bins, legal); }
    let k = top_k.max(1).min(nucleus.len());
    nucleus.truncate(k);
    let t = if temperature.is_finite() && temperature > 0.0 { temperature } else { 1.0 } as f64;
    let mut max_ln = f64::NEG_INFINITY;
    for &(_, wi) in &nucleus { let lnw = wi.ln(); if lnw > max_ln { max_ln = lnw; } }
    let mut weights: Vec<f64> = Vec::with_capacity(nucleus.len());
    for &(_, wi) in &nucleus { let lnw = wi.ln(); let z = (lnw - max_ln) / t; weights.push(z.exp()); }
    if weights.iter().all(|&x| x == 0.0) { return select_move_max_p1(bins, legal); }
    let dist = match rand::distributions::WeightedIndex::new(&weights) { Ok(d) => d, Err(_) => return select_move_max_p1(bins, legal) };
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
    let max_extra = extra_bins.min(one_idx);

    let mut best_i: Option<usize> = None;
    let mut best_s: f32 = f32::NEG_INFINITY;
    for (i, head) in bins.iter().enumerate() {
        if !legal[i] { continue; }
        let mut s = *head.get(one_idx).unwrap_or(&0.0);
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

pub(crate) fn select_move(
    bins: &Bins,
    legal: &[bool; 4],
    sampling: &config::SamplingStrategy,
    rng: &mut rand::rngs::StdRng,
) -> Option<Move> {
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

