# Adding value head

## Background

The 2048 score is the sum of all merges made throughout the game (episode). In addition to the supervised _policy_ head (trained from brute-force expectimax depth-6 relative disadvantage predictions), we are adding a _value head_ that predicts Monte Carlo discounted return per state.

With full episodes available, the target should start as full Monte Carlo discounted reward,
$$G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k},$$
where the raw reward is the step-wise merge gain. Later we can layer on N-step TD for self-play/distillation. Following MuZero (Schrittwieser et al. 2019) and Stochastic MuZero (Antonoglou et al. 2022):
- Apply the invertible transform $h(x) = \mathrm{sign}(x)(\sqrt{x + 1} - 1 + \epsilon x)$ with $\epsilon = 0.001$ to stabilise large targets.
- Use **two-hot** discrete cross-entropy over $n$ bins (e.g., 601).

### Literature on two-hot cross-entropy

From Stochastic MuZero paper:
> In both the afterstate dynamics and dynamics network the parent states and codes or actions were combined by simple concatenation. The reward and value predictions used the categorical representation introduced in MuZero (Schrittwieser et al., 2020). We used 601 bins for both the value and the reward predictions with both the value and the reward being able to represent values between [0, 600.]. Furthermore, similarly to MuZero, we used the invertible transform h(x) = sign(x)(√x + 1 − 1 + x) with = 0.001 for scaling the targets. The value targets were computed using n-step TD(λ) bootstrapping with n = 10, a discount of 0.999 and λ = 0.5.

From original MuZero paper:
> We use a discrete support set of size 601 with one support for every integer between −300 and 300. Under this transformation, each scalar is represented as the linear combination of its two adjacent supports, such that the original value can be recovered by x = xlow ∗ plow + xhigh ∗ phigh. As an example, a target of 3.7 would be represented as a weight of 0.3 on the support for 3 and a weight of 0.7 on the support for 4. The value and reward outputs of the network are also modeled using a softmax output of size 601. During inference the actual value and rewards are obtained by first computing their expected value under their respective softmax distribution and subsequently by inverting the scaling transformation. Scaling and transformation of the value and reward happens transparently on the network side and is not visible to the rest of the algorithm.

## Where things stand (2025-xx)

Status snapshot (Apr 2025):
- ✅ Value sidecar CLI (`value-sidecar`) computes per-step rewards via `twenty48-utils` merge scores and writes discounted returns (`reward`, `reward_scaled`, `return_raw`, `return_scaled`) aligned to `steps-*.npy`. Flags: `--gamma`, `--reward-scale`, `--workers`, `--overwrite`.
- ✅ Value head + config plumbing: `core_2048` exposes an optional mean-pooled value head with either MSE (scalar) or discrete cross-entropy (vocab_size + vocab_type) output. A pre-pool SwiGLU block can be enabled for the value path via `value_head.pre_pool_mlp` (inherits the trunk width/expansion defaults). See `ValueHeadConfig`/`ValueObjectiveConfig` in `packages/core_2048/src/core_2048/model.py` and the init schema notes in `packages/core_2048/README.md`.
- ✅ Training path: dataloaders can join the value sidecar and `BinnedEV`/`MacroxueTokens` mix value loss alongside policy loss (tunable via `value_training.value_loss_policy_scale`; defaults to 1.0). Value path now supports MSE **and** MuZero two-hot cross-entropy (default support [0,600], vocab=601, configurable in `value_training.ce_*`). Inference returns `(hidden_states, policy_out, value_out)`; policy-only callers stay compatible.

- Macroxue packs only store policy metadata: see `crates/dataset-packer/src/schema.rs` and `docs/macroxue_data/data_format.md`. There is no reward/return field yet; `runs.max_score` in SQLite is the final score only.
- The transformer now returns `(hidden_states, policy_out, value_out)` where `value_out` is `None` when disabled; policy heads are unchanged (binned EV or action_policy). Legacy callers still work because unpacking is handled centrally.
- Training defaults to policy-only, but when `value_training.enabled=true` the dataloaders emit `value_targets` and the objectives consume them (policy/objective wiring stays backward compatible when value is disabled).
- Inference now streams policy + optional value (`InferRequest.output_mode` controls the mix) and advertises tokenizer/value metadata via a new `Describe` RPC. Annotation now writes value sidecars (`annotations-value-*.npy` with `value`/`value_xform`, `annotations-value-bins-*.npy` when the value head is categorical) unless `--no-value-sidecar` is passed.

## Data pipeline plan

**Value sidecar for packed datasets** (new Rust CLI):
- [x] Compute per-step merge reward from packed boards + move (`crates/twenty48-utils` shift/score tables give deterministic merge score per move).
- [x] For each run (ordered by `run_id`, `step_index`), compute discounted returns backward. Expose `--gamma` and `--reward_scale` to manage the 1k–1.5m score range.
- [x] Write a NumPy sidecar (aligned with `steps-*.npy`) that keeps `run_id`, `step_index`, `reward`, and `return` (raw and transformed). Keep binning/transform in Python to allow objective swaps.

Run the sidecar generation on an existing packed dataset:

```bash
cargo run -p dataset-packer --bin value-sidecar -- \
  --dataset datasets/macroxue/d7_10g_v1 \
  --gamma 0.997 \
  --reward-scale 1.0 \
  --overwrite
```

The sidecar uses all cores by default; use `--workers N` if you need to cap threads.
Per-step reward computation is parallelized even for a single long run to avoid single-core bottlenecks.
- [ ] Consider sharding alongside annotations (`annotations-*.npy`) and copy `metadata.db` to keep joins simple.

## Model and training plan

- [x] Add a value head atop the pooled board representation (same mean-pooled hidden state as policy heads). Keep the option to share trunk weights with policy.
- [x] Config knobs to enable value head, choose objective (MSE vs discrete/two-hot), and freeze/unfreeze the encoder. (`value_head.enabled/pooling/objective.type|vocab_size|vocab_type` in config.json; pooling is mean-only for now. Trunk freezing to be handled in trainer.)
- [x] Data loader joins: load value sidecar and align on `(run_id, step_index)` to batch `return` targets alongside existing policy fields (`value_training.enabled`).
- [x] Training modes: (a) train value head only on frozen trunk; (b) fine-tune both heads jointly with weighted losses; (c) from-scratch co-training (`value_training.freeze_trunk`, `value_only`, loss weights/scales).
- [ ] Optional: experiment with Perceiver-style aggregation instead of mean-pooling for the value readout.

## Inference and evaluation plan

- [x] Extend the protobuf + server to stream value predictions (raw + inverse-transformed) optionally, without forcing 1-ply inference to pay for it. `InferRequest.output_mode` controls policy/value combinations (auto, policy-only, value-only, policy+value w/ require-value), and `Output.value` now carries the current-board value alongside policy heads when present. `ModelMetadata.value` surfaces objective/support/transform hints for downstream decoding and `Describe` advertises tokenizer/value support up front.
- [x] Add value fields to annotation sidecars (scalars + optional per-bin probs) and manifest bits, with an opt-out flag (`--no-value-sidecar`) when policy-only annotation is desired.
- [ ] Add a smoke-test script mirroring `bin/play_2048.py` that logs predicted value vs realised returns over a short rollout.

## Two-hot cross-entropy touchpoints (what changed / what to edit next)

- Training config now accepts `objective = "cross_entropy"` with MuZero two-hot support controls (`ce_vocab_size/support_[min|max]/transform_epsilon/apply_transform` in `ValueTrainingConfig`). Default support is `[0, 600]` → vocab 601.
- Dataloaders build two-hot targets when CE is selected via `scalar_to_two_hot` in `packages/train_2048/src/train_2048/value_support.py`; enable/disable the MuZero transform via config (use `return_raw` + apply_transform=true, or `return_scaled` + apply_transform=false to avoid double scaling).
- Objectives (`BinnedEV`, `MacroxueTokens`) train either MSE or CE; CE uses soft two-hot targets (`-target * log_softmax`) and logs `value_loss` accordingly.
- Model/helpers: shared MuZero transform + inverse + two-hot utilities live in `packages/train_2048/src/train_2048/value_support.py`. Inference now decodes value heads (best-effort inverse when metadata is available) and returns both transformed and inverse-transformed scalars.
- Logging: training/val payloads and progress bar label value loss as `value_ce` when CE is active; wandb tags follow the same naming.
- Still pending: viewer rails and smoke-tests that exercise value predictions end-to-end; revisit manifest metadata if we add more transforms.

## Contradictions vs current stack (review)

- The value head exists and is optional; policy-only configs remain backward compatible (omit `value_head` in config.json).
- Datasets **do not carry reward/return columns**; only `max_score` per run exists, so a new sidecar/pack step is required before value training is possible.
- Inference rails now stream value outputs + metadata; annotation sidecars capture value predictions, but viewers/analysis still need to consume them.
