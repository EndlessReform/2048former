# 2048 Transformer Pretraining

This package handles the pretraining loop for our 2048 transformer. During pretraining, we perform knowledge distillation based on oracle predictions from an expectimax game agent: either [Matt Kennedy](https://github.com/EndlessReform/2048_bench)'s Rust-based implementation or [MacroXue](https://github.com/EndlessReform/macroxue-expectimax-2048)'s C++ hybrid with expectimax and end-position lookup.


## Training objectives

Three supervision modes are available via `[target].mode` in the TOML config:

- `binned_ev` *(default)* — discretises per-direction EVs into histogram bins using `[binning]`. The encoder keeps four independent heads (up/down/left/right) that emit logits over `n_bins`.
- `hard_move` — trains a single 4-way policy head with one-hot cross-entropy on the logged move. Illegal moves are masked out, but no extra signal is provided beyond the argmax.
- `tsad_soft` — enables temperature-scaled advantage distillation. Branch EVs are normalised per position, softened with `[target].tsad_temperature`, and optionally blended with the logged move through `[target].tsad_mix_with_hard`. The loader emits a soft distribution over the legal moves plus the original hard label, and the trainer minimises KL(student‖teacher) with an optional auxiliary CE term (`[target].tsad_aux_ce_weight`).

When `target.mode` is `hard_move` or `tsad_soft`, the trainer automatically swaps the encoder to a single 4-way policy head before compilation; existing inits that expose binned heads continue to work. Inference helpers (`forward_distributions`, `bin/play_2048.py`) already understand the policy head and convert it back into per-move probabilities for sampling.

Example configs:

- `config/config.example.toml` (binned EV baseline)
- `config/pretraining/v1-hard-10m.toml` (one-hot policy distillation)
- `config/pretraining/v1-tsad-10m.toml` (TSAD with move-bias mixing)

Key TSAD knobs:

- `[target].tsad_scale_kind` — choose `"max_abs"` (default) or `"mad"` to scale per-position advantages before temperature.
- `[target].tsad_min_scale` — clamps the denominator to avoid flat late-game boards from exploding the softmax.
- `[target].tsad_mix_with_hard` — convex blend weight towards the logged move (`0.0` = pure KD, `1.0` = one-hot).
- `[target].tsad_aux_ce_weight` — adds a separate cross-entropy penalty that sharpens the chosen move without discarding the soft distribution.


## Training data

### Raws: Macroxue format

An expectimax selfplay loop produces two files per 2048 run: a `.meta.json` containing run-level metadata, and a `.jsonl.gz` containing each game state in the run.

Metadata example record:

```json
// Note: the actual file is not pretty-printed
{
    "seed": 710363253,
    "depth": 2, // depth of expectimax during run
    "game_index": 0, // Index during selfplay run. Irrelevant downstream
    "steps_file": "selfplay_logs/d2_test_v2/depth02_worker00_seed0710363253_game000000.jsonl.gz",
    "num_moves": 31671,
    "score": 1008096,
    "max_tile": 65536,
    "max_rank": 16,
    "sum_tile":69774,
    "seconds": 1.419195
}
```

Example step row from the `.jsonl.gz`:

```json
// Also not pretty-printed
{
    "seed": 710363253,
    "step_index": 0,
    "max_rank": 1, // max rank on board at this step
    "move": "left",
    "valuation_type": "search",
    "valuation": 0.034000,
    "board": [0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0],
    "branch_evs": {"up":0.033000,"left":0.034000,"right":0.013000,"down":0.011000}
}
```




### 
