# 2048 Value Head Bootstrapping — Design Doc (repo‑aligned)

Audience: junior engineer. Plain language. No handwaving.

Goal: Train and compare two plug‑in value heads on top of a frozen transformer trunk (board → 16 tokens → trunk embeddings). Use 50M+ saved game‑states to bootstrap. Evaluate which head better predicts future game strength and highest tile.

Heads to compare:
- Ordinal/BCE (reaching‑rate) — for each milestone tile T, predict P(reach ≥T | state).
- Categorical/CE — distribution over highest‑tile buckets or over log‑score bins.

Important: This document is updated to match the actual repository layout, data formats, and training loop. It avoids introducing a parallel package and instead plugs into existing modules.

---

## 0) What exists in this repo

Data layout (self‑play v1/v2 rails):
- Steps: `steps.npy` or shards `steps-*.npy` under a session directory; structured dtype with at least `run_id`, `step_index`, and a packed board. In current training, additional fields like `branch_evs`, `ev_legal`, and `move_dir` may be present; the value head does not require them.
- Metadata: `metadata.db` (SQLite) with a `runs` table containing per‑game facts. The columns of interest are `id` (run_id), `steps` (count), `highest_tile`, and often `max_score`/`final_score`. Older dumps may name columns slightly differently; the loader already probes variants where needed.

Loader/train loop:
- Dataloader and samplers live in `packages/train_2048/src/train_2048/dataloader/steps.py` (see file for fields and samplers).
- Training entrypoint is `packages/train_2048/src/train_2048/main.py`, which calls `training_loop.run_training`.
- Objectives live under `packages/train_2048/src/train_2048/objectives/` and are selected via `target.mode` in the TOML config.
- The transformer trunk is `packages/core_2048/src/core_2048/model.py` (class `Encoder`), which returns both hidden states and current policy/EV heads.

Tokens/board codec:
- Steps store boards packed; collate decodes to 16 token IDs using `BoardCodec.decode_packed_board_to_exps_u8` (see `packages/train_2048/src/train_2048/dataloader/collate.py`).

---

## 1) Label definitions (derived from metadata.db)

Labels are per state but derived from the run‑level outcome (`runs.highest_tile` and optionally `runs.max_score`/`final_score`).

### 1.1 Milestone thresholds (for Ordinal/BCE)

Define milestones: `T = [1024, 2048, 4096, 8192, 16384, 32768]` (configurable).
For a finished episode’s `highest_tile`, the per‑threshold label is:

* `y_k = 1` if `max_tile >= T[k]`, else `0`.
  So one state yields a **vector** like `[1,1,1,1,0,0]` if the episode’s top tile was 8192.

### 1.2 Buckets for Categorical/CE

Pick one target:
- Highest‑tile buckets: `B0:<1024`, `B1:[1024,2048)`, …, `B6:>=32768` (K+1 classes).
- Log‑score bins: `x = log1p(score)` where score comes from `runs.max_score` or `runs.final_score` depending on schema. Choose `n_bins` (e.g., 64), uniformly spaced in x. Use soft labels by splitting mass linearly across adjacent bins.

---

## 2) Sampling and splits (use existing samplers)

We do not introduce a separate sampling manifest. Use the existing dataset and split machinery in `train_2048`:
- Build loaders via `build_steps_dataloaders(...)` in `packages/train_2048/src/train_2048/dataloader/steps.py:500`.
- Split by run to avoid leakage using either SQL predicates (`run_sql`/`val_run_sql`) or a random run split (`val_run_pct`) — see `packages/train_2048/src/train_2048/dataloader/steps.py:360` and `:404`.
- Control epoch size with `dataset.num_steps` to stream a fixed number of steps per epoch via `StreamingRandomSampler` or filter by `step_index_min/max` for time‑slicing — see `packages/train_2048/src/train_2048/dataloader/steps.py:86`, `:163`, and `:500`.
- Enable `mmap_mode=true` in config to memory-map large step shards.
- To keep each episode equally represented, enable `dataset.value_sampler` in the training TOML. The helper caps states per run and draws evenly spaced quantiles inside early/mid/late segments. See `config/value_head/quantile_stratified.toml` for concrete defaults (512 states per run across `[0.0, 0.25, 0.75, 1.0]`).
- Select the objective with `[target].mode = "value_ordinal"` (BCE reach-rate) or `"value_categorical"` (tile CE). Tune the head via `[value_head]` — freeze/unfreeze the transformer trunk, pick `head_type = "probe"|"mlp"`, adjust `mlp_hidden_dim`/`mlp_dropout`, and override `tile_thresholds` (defaults: `[1024, 2048, 4096, 8192, 16384, 32768]`).

Recommended scale: start with 5–10M steps per epoch, validate on a held‑out run split. Always split by run, not by state.

---

## 3) Model variants (shared trunk)

Trunk: the existing `Encoder` in `packages/core_2048/src/core_2048/model.py` produces hidden states `(B, L, D)` and head outputs. We will reuse its hidden states and mean‑pooled board representation `(B, D)`.

Implementation (repo-aligned):
- Add a new objective that attaches a value head to the encoder during `prepare_model(...)` (see `packages/train_2048/src/train_2048/objectives/base.py`).
- The objective will call `model(tokens)` to get hidden states `x` and then compute `board_repr = x.mean(dim=1)` inside the loss.
- Freeze the trunk by setting `requires_grad=False` on embeddings/blocks/ln; only the value head parameters remain trainable.
- The implementation is wired into `packages/train_2048/src/train_2048/objectives/value_head.py`; `prepare_model` now installs either a linear probe or a 1-hidden-layer MLP on top of the pooled board state and respects `[value_head]` toggles when deciding whether to keep the trunk frozen.

Heads:
- Linear probe: `Linear(D → K)` (ordinal) or `Linear(D → K+1 / n_bins)` (categorical).
- 1‑hidden MLP: `LayerNorm(D) → Linear(D→H) → GELU → Dropout(0.1) → Linear(H→out)`.
  - `out = K` (ordinal), `K+1` (tile CE), or `n_bins` (log‑score CE). Default `H=1024`.
  - Optional monotone parameterization for ordinal logits: `t = b − cumsum(softplus(inc))`.

---

## 4) Objectives (explicit formulas)

### 4.1 Ordinal/BCE (Reaching-Rate)

Let the head output **K logits** `t_k`. Convert to probabilities `q_k = sigmoid(t_k)`.
For a label vector `y ∈ {0,1}^K`:

* **Per-threshold binary cross-entropy**:
  [\text{BCE}(y_k,q_k) = -y_k \log q_k - (1-y_k)\log(1-q_k).]
* **Loss (averaged over thresholds and batch):**
  [\mathcal{L}*\text{ord} = \frac{1}{K}\sum*{k=1}^K \text{BCE}(y_k,q_k).]
* **Optional class-balancing:** compute dataset positive rate `p_k = P(y_k=1)`. Weight positives and negatives:
  [w^+_k = \tfrac{1}{2p_k},\quad w^-*k = \tfrac{1}{2(1-p_k)}] (cap `w^-_k` at 10 to avoid extremes). Then
  [\mathcal{L}*\text{ord} = \frac{1}{K}\sum_k \big( w^+_k y_k \cdot \text{BCE}^+ + w^-_k (1-y_k) \cdot \text{BCE}^- \big).]
* **Optional focal term** to down-weight easy examples: multiply each term by `(1 - p_t)^γ`, where `p_t = y_k*q_k + (1-y_k)*(1-q_k)` and `γ ∈ [1,2]`.
* **Monotone logits (recommended):** enforce `q_1 ≥ q_2 ≥ … ≥ q_K` by parameterizing
  `t_k = b - cumsum(softplus(inc_k))`. (This is a single head; not 7 heads.)

From `q` to exact buckets: `p0 = 1 - q1`, `p_i = q_i - q_{i+1}` (i=1..K-1), `pK = q_K`.

### 4.2 Categorical/CE

Choose **one** target:

* **Highest-tile buckets (K+1 classes)** with **one-hot** labels.
* **Log-score bins (`n_bins` classes)** with **soft labels** (linear interpolation between adjacent bins in log-score).

Let the head output logits `z` over classes; `p = softmax(z)`.

* **Loss:** [\mathcal{L}_\text{ce} = -\sum_j y_j \log p_j] (averaged over batch). For soft labels, `y_j` can be fractional.
* **Optional distance-aware smoothing:** Gaussian label smoothing in log-score space (bandwidth set by the bin width).

---

## 5) Training plan (using existing CLI/loop)

Defaults:
- Optimizer: `AdamW(lr=3e-4, weight_decay=1e-4)`
- Batch size: as large as fits (e.g., 2048–4096)
- Epochs: 5–10; early stop on val NLL.
- Trunk frozen; train only the value head.

Steps:
1) Build DataLoaders with `build_steps_dataloaders(...)` using `dataset.*` config. Use `val_run_sql` or `val_run_pct` to split by run.
2) In a new objective, attach the value head in `prepare_model(...)`, freeze trunk params, and reuse the encoder forward for hidden states.
3) Compute loss: ordinal BCE or categorical CE depending on mode. Log NLL, Brier/ECE, and AUROC where relevant.
4) Validate each epoch; save best by val loss using existing checkpoint hooks.

Artifacts: rely on existing checkpointing in `packages/train_2048/src/train_2048/checkpointing.py` for safetensors and optional `.pt` bundles. Metrics stream to stdout and optionally W&B if enabled.

---

## 6) Benchmarks & metrics

Evaluate on the **test split (by game)**.

### 6.1 Core metrics

* **Log-loss / NLL** (primary):

  * Ordinal/BCE: average BCE over thresholds (lower is better).
  * Categorical/CE: cross-entropy over classes (lower is better).
* **Calibration:**

  * **Brier score** per threshold (ordinal).
  * **ECE** (Expected Calibration Error):

    * thresholds (ordinal),
    * bins (categorical),
    * show reliability plots.
* **Ranking quality:**

  * **AUROC** per threshold (e.g., especially for `≥32768`).
  * **Top-k bucket accuracy** (exact and within ±1 bucket).
* **Correlation with outcomes:**

  * Spearman/Pearson between predicted **expected score** and actual final score on held-out games (compute expected score from ordinal head via a precomputed `delta_k` vector; see §6.3).

### 6.2 Slices (error analysis)

* **By timestep**: early (first 25%), mid (25–75%), late (last 25%).
* **By episode length**: short, median, long (tertiles).
* **By highest-tile bucket**: where do we under/over-estimate?
* **Confusions**: for categorical, plot confusion matrix over buckets; for ordinal, plot per-threshold precision/recall.

### 6.3 Expected score from ordinal (optional scalar)

Compute a vector `delta[k]` = typical **score increment** from crossing `T[k]` (median difference; precompute once from finished episodes). Then for a state with ordinal probs `q`:

* `E[score] ≈ base + sum_k delta[k] * q[k]` (base can be 0). Use this for correlation metrics.

---

## 7) Reporting

For each model × objective (4 combos: linear/MLP × CE/BCE):

* Table of **NLL**, **Brier/ECE**, **AUROC**@`≥16384` and `≥32768`, **Top-1** and **Top±1** accuracy (categorical), and **score correlation**.
* Reliability plots (PNG) and confusion matrix (categorical).
* Training curves (train/val loss).
* In 1 page of plain English: **which head wins and why** (cite the numbers). No jargon.

**Win criteria (initial):**

* Primary: Best **test NLL** in its family (ordinal vs categorical).
* Secondary: Better **AUROC@≥32768** and better **ECE** on high thresholds, or better **Top±1** for categorical.
* Tertiary: Stable training and no obvious calibration pathologies.

---

## 8) Implementation (where to put things)

We integrate with existing packages instead of creating a new top‑level module:
- New objective(s): add `value_ordinal` and `value_categorical` under `packages/train_2048/src/train_2048/objectives/` and register in `packages/train_2048/src/train_2048/objectives/__init__.py:8`.
  - Each objective implements `prepare_model` (attach head; freeze trunk), `train_step` (compute pooled repr, logits, loss, metrics), and `evaluate`.
  - Heads can live inline in the objective file to start; promote to a small `heads.py` later if reused.
- Collate function: add `make_collate_value(...)` to `packages/train_2048/src/train_2048/dataloader/collate.py` to return `(tokens, labels)` for the chosen target. It should:
  - Preload `run_id → highest_tile` and `run_id → score` maps once from `metadata.db`.
  - During collate, fetch rows via `StepsDataset.get_rows(...)`, decode tokens via `BoardCodec`, and build labels for ordinal or categorical targets.
- Config: extend `TargetConfig.mode` in `packages/train_2048/src/train_2048/config.py:15` to accept the two new modes and thread any extra knobs (e.g., thresholds, n_bins) via `cfg.binning` or a small `value_head` section if needed.
- Freezing: within `prepare_model`, set `requires_grad=False` for `tok_emb`, `pos_emb`, `blocks`, and `final_ln`. Leave the new value head trainable. Optionally gate with a config flag if you want to compare frozen vs finetune.
- Optional: if you prefer heads attached to the core model, you can add an optional `value_head` attribute in `packages/core_2048/src/core_2048/model.py:300` and let objectives toggle it; however, attaching in the objective avoids touching the core until we commit the design.

---

## 9) How to run (uv + existing CLI)

- Install deps: `uv sync`
- Train value head (ordinal, linear probe) with a new config that sets `target.mode = "value_ordinal"` and points `dataset.dataset_dir` to a session containing `steps.npy` and `metadata.db`:
  - `uv run python packages/train_2048/src/train_2048/main.py --config config/pretraining/v2/50m-hard-move-continue.toml` (replace with your value‑head config TOML).
- Override device: `--device cpu|cuda`.

Config knobs to use:
- `dataset`: `run_sql`/`val_run_sql` or `val_run_pct`, `num_steps`, `mmap_mode`, `step_index_min/max`.
- `binning`: reuse for thresholds (`edges`) or add a small `value_head` section with `milestones` and `n_bins` if preferred.

---

## 10) Guardrails & pitfalls

* **Split by game**, not by state (avoid leakage).
* **Sampler seed** must be saved; otherwise you cannot reproduce the subset.
* **Length bias:** always sample a fixed number per game with time strata.
* **Class imbalance:** for ordinal, consider dropping the two easiest thresholds, or use class-balancing.
* **Calibration:** don’t trust accuracy alone; always check ECE/Brier and reliability plots.
* **Numerical safety:** clamp probabilities before logs; add eps in divisions.

---

## 11) Deliverables checklist

* [ ] `sampling_manifest.csv` + `config.json` + `metrics.json` and seeds recorded.
* [ ] Four trained models (linear/MLP × ordinal/categorical) with checkpoints.
* [ ] `eval/` directory with: test metrics JSON, reliability plots, confusion matrix, training curves.
* [ ] **One-page writeup** (bullet points) comparing the two objectives and two architectures, grounded in the numbers.

Success looks like: a clear winner on test NLL and at least one of: better AUROC on high thresholds or better calibration, plus a simple recommendation for what to co‑train with policy next. Use the repo’s loaders, configs, and CLI so results are reproducible via `uv run`.
