# Interpretation Scripts

This folder contains lightweight analysis helpers for activation dumps.

## Count Outliers

`count_outliers.py` inspects a 4D activation tensor (N, L, T, H) from
`activations.safetensors` and reports outlier statistics.

### Topline (fast)

Use this when you only need the headline result (top dims, sign, layers, tokens).

```bash
uv run --locked scripts/interp/count_outliers.py \
  --input activations.safetensors \
  --key attn_norm \
  --threshold 6.0
```

### Exhaustive (slower)

Use this when you want extra comparisons or heavier summaries.

```bash
uv run --locked scripts/interp/count_outliers.py \
  --input activations.safetensors \
  --key attn_norm \
  --compare-key mlp_input \
  --threshold 6.0 \
  --outlier-samples 0 \
  --iqr-samples 2_000_000
```

### Optional vs necessary outputs

Always shown (topline):
- Outlier counts and rates.
- Hidden dimension ranking + one-sided sign check.
- Layer-wise and token-position distributions.
- IQR estimate for the full activation distribution.

Optional (exhaustive or slower):
- `--compare-key`: compare two activation keys in the same dump.
- `--outlier-samples 0`: exact outlier magnitude stats (uses all outliers).
- `--iqr-samples`: adjust for faster or more accurate IQR estimation.
- `--no-layer-fraction`: disable the same-location layer fraction requirement.

### Tips

- Use `--quiet` to suppress debug progress lines.
- For very large dumps, increase `--outlier-samples` if you only need a rough
  magnitude summary.

## Plot Attention Matrices

`plot_attention_matrices.py` loads `attn_norm` + `boards` from an activation dump
and renders attention heatmaps for a few random boards at a chosen layer.

```bash
uv run --locked scripts/interp/plot_attention_matrices.py \
  --input activations.safetensors \
  --layer 1 \
  --n-boards 2
```

Optional knobs:
- `--heads 0,4`: plot specific GQA heads.
- `--no-avg`: skip the all-heads average heatmap.
- `--output-dir out/attn_viz/custom_run`: customize output folder.

## Check RMSNorm Gammas

`check_rmsnorm_gammas.py` reads RMSNorm gamma weights from an activation dump
and reports whether they are all exactly 1.0. Make sure the dump was created
with `--outputs rmsnorm_gammas`.

```bash
uv run --locked scripts/interp/check_rmsnorm_gammas.py \
  --input activations_with_rmsnorms.safetensors
```
