# Afterstate EV Tokenization Proposal

## Goals and Constraints
- Preserve as much teacher signal as possible for KD without exposing tuple heuristics directly.
- Normalise heterogeneous valuation scales (tuple probabilities in [0,1], expectimax scores spanning roughly [-3k, 3k]).
- Stay robust to late-game scale drift where absolute magnitudes explode.
- Yield targets that plug into the existing `binned_ev` path or a small extension of it.

## Candidate Strategies

### 1. Temperature-Scaled Advantage Distillation (TSAD)
Core idea: convert per-move scores into relative advantages, then distil via a softened cross-entropy over the four moves. Offsets cancel shifts and a robust scale factor tempers type-specific magnitudes.
Recipe: (1) subtract the best legal valuation from every legal branch; (2) divide by a scale parameter `s` computed per position, e.g. `s = max(|v_best - v_median|, epsilon)` or a learned function of simple board features such as max tile exponent; (3) apply softmax with temperature `tau` to obtain weights; (4) train with KL divergence against these weights alongside the hard move target.
Why it helps: the subtraction removes additive bias (tuple11 vs expectimax), and the adaptive divisor ensures that when search scores swing wildly the distribution remains informative rather than collapsing to the argmax.
References: matches the soft target recipe of Hinton et al. 2015 (Knowledge Distillation) and the energy-based Boltzmann policies discussed in Sutton & Barto 2018 §13.1.
Pros: smooth signal for all moves, easy to integrate with current loaders, degrades gracefully when only one move is legal. Cons: requires calibrating `s` and `tau`; bad scaling choices can still over-flatten confident tuple outputs.

### 2. Type-Calibrated Survival Probability Binning
Core idea: learn calibration curves that map each valuation type to an estimated “survival probability” or expected score delta, then reuse the existing binning stack on the calibrated outputs. Tuple policies would use near-identity mappings, while expectimax scores pass through an isotonic or Platt-style calibrator fit against downstream outcomes.
Implementation sketch: assemble a held-out slice where the final max tile or termination depth is known; fit monotonic calibrators per `valuation_type`; transform branch EVs through the matching calibrator before binning; store the type ID (already available in logs) to pick the correct mapping at load time.
References: Platt 1999 and Zadrozny & Elkan 2002 on probability calibration; similar per-head calibration appears in DeepMind’s AlphaGo ladder models.
Pros: keeps the familiar bin targets, separates concerns (calibration vs. tokenization), and offers a clear offline validation loop. Cons: needs reliable outcome labels and enough data per type; calibration may drift if teacher heuristics change.

### 3. Rank-and-Margin Tokens
Core idea: ignore absolute magnitudes and encode ordinal structure plus coarse margin classes. For each step record (a) the rank of every legal move, (b) a signed margin bucket computed from `(v_move - v_best)` divided by a board-scale normaliser (e.g., `2^max_tile_exp` or running MAD). Concatenate rank and margin class into a discrete token bank (roughly 12–16 bins).
Why it helps: tuple probabilities and expectimax scores both induce meaningful orderings even if their scales differ. Margin classes retain information about how decisive the choice was without assuming unit comparability.
References: inspired by listwise ranking losses (e.g., Cao et al. 2007’s ListNet) and the pairwise margin distillation used in policy distillation for Go (Aja Huang et al. 2016).
Pros: robust to outliers, cheap to compute, and highlights “really bad” negatives because they land in extreme margin bins. Cons: loses calibration information within a margin class; requires carefully picking the normaliser so margins stay in a reasonable range.

### 4. Board-Scale Normalised Quantile Binning
Core idea: regress a fast board-scale estimator `g(board)` that predicts the typical magnitude of expectimax scores at that stage (features: max tile exponent, empty-tile count, sum of tiles). Divide raw expectimax values by `g(board)` before mixing with tuple probabilities. Feed the normalised values into a quantile-based binning scheme computed over a large corpus so that bins reflect relative standing rather than raw numbers.
Implementation notes: precompute quantile edges on a stratified sample, updating them when the teacher dataset shifts; clamp the normalised scores to a limited range (e.g., [-4, 4]) before binning to avoid tail blow-ups.
References: echoes the “dynamic value scaling” used in MuZero (Schrittwieser et al. 2020 Appendix B) and the normalised advantage functions of Gu et al. 2016.
Pros: produces a single scalar per branch compatible with current bin infrastructure, captures negative catastrophes via bins on the lower tail. Cons: adds an extra model (`g`) that must stay in sync with new datasets; misestimation of `g` can reintroduce scale mismatch.

## Next Steps to De-Risk
- Prototype TSAD to get immediate signal, since it only needs per-step statistics. Measure KL agreement between calibrated teacher and current student policy.
- In parallel, collect outcome-labelled data to fit per-type calibrators; this unlocks strategy 2 if TSAD underperforms.
- Run small ablations comparing rank-and-margin vs. quantile binning on held-out Macroxue logs to test which preserves tuple-style confidence best.
