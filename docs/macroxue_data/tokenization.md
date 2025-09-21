Title: Winner+Margins Tokenization for Cross-Type 2048 Action Values

Motivation
- Convey teacher action-value structure to a transformer while hiding the valuation backend (`search/tuple10/tuple11`).
- Encode both the chosen move and how wrong each alternative is, despite wildly different EV scales between early/late game and heuristics.

Method Overview
- Fit an empirical CDF `F_t` over all branch EVs per valuation type `t` (ignoring `None`).
- Map each branch EV `v` to percentile `p = F_t(v)` in `[0,1]`.
- Identify the best action `a*` with percentile `p*`. Emit a single `[WINNER]_{a*}` token.
- For every other legal action `a`, compute the regret margin `Δ_a = p* - p_a`. Bin `Δ_a` and emit `a#d{k}` (optionally append `#r{2|3|4}` if loser rank is useful).
- For illegal branches (no board change), emit `[ILLEGAL]_{a}` and skip the margin token.
- Optionally bin the decision hardness `gap_top2 = p* - p_(2nd)` on the same grid and emit one `gap#d{k}` token per state.

Clarifications (ECDFs & Spec)
- ECDFs are fit over absolute EVs. Margins are computed *after* the EV→percentile mapping; this keeps Δ on a common scale, so early-game values like `5.2` and late-game `1100.8` compare fairly.
- Tokenization relies on a frozen spec: `out/tokenizer_v1.json` stores 2049 quantile knots for each valuation type plus the Δ-bin edges (see below). Training consumes only the tokens; the ECDFs are needed only when converting fresh logs.
- Retokenizing new logs is cheap: load the spec, map EVs → percentiles via piecewise-linear interpolation between knots, then produce tokens.

Empirical Evaluation (d6_test_v2_240)
- Dataset: depth-6 test set with 240 games (~6 M states, 15.1 M non-winner legal branches). Median `gap_top2 = 2.7e-4`; 90th percentile `6.9e-2`; 99th percentile `6.7e-1`.
- We swept non-winner bin grids with 2–128 interior bins. Quantile-based grids consistently dominated log-spaced grids (2–3× lower MAE).
- Key distortion numbers (MAE of Δ vs bin midpoint):
  - `quantile_16` (17 bins): 0.0153 MAE, 0.0285 RMSE, max bin share 6.0%.
  - `quantile_32` (32 bins): 0.00774 MAE, 0.0160 RMSE, bin shares 2.8–6.1%.
  - `quantile_64` (62 bins): 0.00422 MAE, 0.00844 RMSE, bin shares 1.6–6.2%.
  - `quantile_128` (122 bins): 0.00217 MAE, 0.00449 RMSE, diminishing returns.
- Recommendation: adopt the 32-bin quantile grid. It halves MAE relative to 17 bins, keeps vocabulary practical, and retains balanced occupancy.

Recommended Δ-Bin Grid (percentile space)
- Edges (rounded to 7 significant figures):
  `[0, 2.7133e-06, 1.54449e-05, 3.25596e-05, 5.54139e-05, 8.71386e-05, 0.000133056, 0.000202975, 0.000318306, 0.000543703, 0.000907203, 0.00135637, 0.0019873, 0.0031199, 0.00481617, 0.007449, 0.0116777, 0.0183356, 0.028475, 0.0441634, 0.0684336, 0.105217, 0.161573, 0.247982, 0.376907, 0.454553, 0.510351, 0.579428, 0.637875, 0.700276, 0.780917, 1.0]`.
- Full-precision edges and per-bin occupancies are in `out/bin_grid_report.json` (`scheme == "quantile_32"`). Version the spec whenever the grid is regenerated.

Tokenization Procedure (per state)
- Map each legal branch EV with the frozen ECDF to get percentiles `p_a`.
- Emit `[WINNER]_{a*}` for the best action `a*`.
- For each legal loser `a`:
  - Compute `Δ_a = p* - p_a`.
  - Find bin index `k` via the edges above and emit `a#d{k}` (optionally with `#r{2|3|4}`).
- Emit `[ILLEGAL]_{a}` for illegal branches.
- (Optional) Digitize `gap_top2` on the same grid and emit `gap#d{k}`.

Model Heads & Loss
- Winner head: 4-way softmax over `{up,left,right,down}`.
- Margin heads: four categorical heads (one per action) over 32 Δ-bins + `ILLEGAL`. Mask the loss when the action is the winner.
- Optional gap head over 32 bins for `gap_top2`.
- Loss: `L = CE_winner + Σ_a 1[a legal ∧ a≠a*]·CE_margin(a) + β·Σ_a 1[a illegal]·CE_illegal(a) + γ·CE_gap`.
- HL-Gauss smoothing: construct a Gaussian over neighbouring bins centred at the true `Δ_a`. σ ≈ one bin width near ties and can scale up for high-Δ bins. Train with cross-entropy against the smoothed target distribution.

Inference / Sampling
- Deterministic: pick `argmax` on the Winner head; break near ties via `score(a) = P_winner(a) − λ·E[Δ_a]` (`λ≈0.1`).
- Regret minimisation: choose the legal action with the smallest `E[Δ_a]` (treat `ILLEGAL` as Δ=1).
- Stochastic: sample from the Winner head; if an illegal move is drawn, resample or fall back to regret minimisation. Gumbel-top-k works well for exploration.
- Optional: expose `E[Δ_a]` as an auxiliary log for downstream evaluation.

Engineering Notes
- `scripts/bin_grid_report.py` reproduces the sweep and writes `out/bin_grid_report.json` / `out/bin_grid_summary.csv`.
- `scripts/export_tokenizer.py` freezes the ECDF knots + edges into `out/tokenizer_v1.json`. Ship this file with any dataset snapshot.
- Sequence vocab: 4 winner tokens, 4 illegal tokens, 4×32 margin tokens (only three fire per state), optional 32 gap tokens, and optional loser-rank suffixes.
- Keep unit tests that re-run a held-out chunk to verify (a) ECDF inversion monotonicity, (b) Δ bin indices stay in range, and (c) histogram JS divergence against the reference spec remains small.

Deliverables
- `out/bin_grid_report.json`, `out/bin_grid_summary.csv`: quantitative sweep results (2–128 interior bins, quantile vs log grids).
- `out/tokenizer_v1.json`: frozen ECDF + 32-bin grid spec.
- `out/strategy_a_*`: legacy plots/csvs for the earlier 17-bin prototype (useful for regression checks).
- `scripts/bin_grid_report.py`, `scripts/export_tokenizer.py`: tooling to regenerate the statistics and spec.

Appendix: Bin Sweep Snapshot (quantile grids)
- 3 bins → MAE 0.0807, entropy 1.58 bits, max bin share 33.6%.
- 5 bins → MAE 0.0503, entropy 2.32 bits, max bin share 20.0%.
- 9 bins → MAE 0.0296, entropy 3.17 bits, max bin share 12.0%.
- 17 bins → MAE 0.0153, entropy 4.09 bits, max bin share 6.0%.
- 32 bins → MAE 0.00774, entropy 4.98 bits, max bin share 6.1%.
- 62 bins → MAE 0.00422, entropy 5.89 bits, max bin share 6.2%.
- 122 bins → MAE 0.00217, entropy 6.82 bits, max bin share 5.4%.