# Value head (v3)

## Overview and design goals

**Motivation:** 
- Currently (18 Jan 2026), our model only implements a policy head (per-move linear probes on mean-pooled hidden state, predicting winning move or relative disadvantage based on macroxue data - see `docs/macroxue_data/valuation.md` and `docs/macroxue_data/data_format.md` for details. This (a) limits us to 1-ply methods, (b) locks us in to be at most as good as the teacher
- We need to create a value head based on existing trajectories (to start with)

### Core implementation touchpoints

**Codebase map (where to wire the value head + loss):**
- `packages/train_2048/src/train_2048/objectives/base.py`: objective contract (`prepare_model`, `train_step`, `evaluate`). Today assumes a single objective; likely needs a composite/multi-loss path with backward-compatible return shape.
- `packages/train_2048/src/train_2048/objectives/__init__.py`: `make_objective` dispatches `target.mode` (`binned_ev`, `hard_move`, `macroxue_tokens`); new `value_head` path or a composite objective would be added here.
- `packages/train_2048/src/train_2048/training_loop.py` + `packages/train_2048/src/train_2048/training_model.py`: objective construction, `prepare_model`, and train/eval loops. Loss scaling and mixed precision flow live here.
- `packages/train_2048/src/train_2048/training_metrics.py`: WandB metric formatting assumes policy-only losses for existing modes; will need value metrics and combined-loss reporting.
- `packages/train_2048/src/train_2048/config.py`: `target.mode` enum + dataset/model config knobs (loss weights, value head type, thresholds, pooling, etc.).
- `packages/core_2048/src/core_2048/model.py` and `packages/train_2048/src/train_2048/modeling/te_encoder.py`: model head definitions and pooled trunk representation; will need a new value head fed by pooled trunk, plus config surface in model JSON.

**Data format touchpoints (highest_tile labels):**
- Packed datasets are `steps.npy` (record dtype) + `metadata.db` + `valuation_types.json`. Example: `datasets/d7_test_v1/` contains all three (may be outdated but good for shape checks).
- `metadata.db` has a `runs` table with `highest_tile`; steps only store `run_id` and `step_index`, so labels are derived via run-level lookup. See `docs/macroxue_data/data_format.md` and `docs/self-play-v1.md`.
- Collation/loader path: `packages/train_2048/src/train_2048/dataloader/steps.py`, `packages/train_2048/src/train_2048/dataloader/steps_v2.py`, and `packages/train_2048/src/train_2048/dataloader/collate.py`. These will need to surface `run_id` (already present) and/or add a per-batch `highest_tile` tensor via a run lookup table.

**Work outline (key methods to change / add):**
- **Objective layer:** add a value-head objective (CORAL loss) or a composite objective that mixes policy + value. Implement in `packages/train_2048/src/train_2048/objectives/` and extend `make_objective`.
- **Model head:** add a value head module wired to the pooled trunk state (likely same pooled representation as the policy head). Update `core_2048` model config to declare value head class/size + `num_classes`/thresholds; keep policy head backwards compatible.
- **Data flow:** build a `run_id -> highest_tile` lookup (SQLite read into memory) and add label generation in the collate path or dataset wrapper; ensure it works for both training and eval loaders.
- **Metrics/logging:** extend `training_metrics.py` to log `value/loss`, `loss_total`, and policy/value breakdown; make WandB keys consistent with existing `train/` and `val/` prefixes.
- **Config / CLI:** extend `target.mode` or add a `target.value_*` block; add loss weights (policy/value), thresholds, and pooling options; ensure `main.py` and `config/` examples are updated.

**Unexpected barriers to plan for:**
- **Objective contract assumes one loss:** `Objective.train_step` returns a single `loss` plus optional per-head losses; multi-loss needs a combined scalar for backprop plus structured metrics for logging.
- **Model output shape assumptions:** current objectives expect 4-way heads or a single policy head; value head needs a clean, named output path without breaking existing objectives.
- **Run-level labels only:** highest tile is in `metadata.db`, not `steps.npy`. Need to avoid per-batch SQLite hits; preload a LUT (run_id contiguous but not guaranteed across merged packs) and handle missing runs.
- **Dataset heterogeneity:** Macroxue uses packed `board` and optional `cumulative_reward`; self-play v1 may use different encodings. Value labels should be independent of board encoding, but collate must be robust to dataset type.

## Phase 1

Create [CORAL-style](https://arxiv.org/abs/1901.07884v6) ordinal sigmoid loss for highest tile reached. (since cumulative reward has small step-wise merge rewards e.g. 4,8,etc.)
- Start to exponents 1024, 2048, ...: ignore 512 and below 
- pseudocode:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoralLayer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # We only need ONE weight vector for all tasks
        self.fc = nn.Linear(input_dim, 1, bias=False)
        
        # We have a separate bias for each threshold (N-1)
        # For 2048, if tiles are [512, 1024, 2048, 4096, 8192, 16384]
        # num_classes = 6, so we need 5 thresholds.
        self.biases = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x):
        # 1. Get the general 'score' for the board state
        score = self.fc(x) # Shape: [batch, 1]
        
        # 2. Add the task-specific biases
        # logits shape: [batch, num_classes - 1]
        logits = score + self.biases
        return logits
```

Loss:
```python
def coral_loss(logits, target_indices, num_classes):
    """
    logits: [batch, num_classes - 1]
    target_indices: [batch] (integers representing the highest tile reached)
    """
    # Create the binary mask (the "levels passed" representation)
    # Example: index 2 becomes [1, 1, 0, 0...]
    batch_size = logits.size(0)
    num_thresholds = num_classes - 1
    
    # Building the binary targets
    levels = torch.arange(num_thresholds).to(logits.device)
    # target_indices[:, None] broadcasts to [batch, num_thresholds]
    binary_targets = (target_indices[:, None] > levels).float()
    
    # Use Binary Cross Entropy with Logits
    # This treats each threshold as a separate sigmoid task
    loss = F.binary_cross_entropy_with_logits(logits, binary_targets, reduction='none')
    
    # Sum the losses across thresholds, then average over the batch
    return loss.sum(dim=1).mean()
```

Implementation notes:
- Highest_tile is kept in the step metadata db but not per-step. Steps are indexed by run ID. You might want to optimize performance by either indexing on run ID in the `.sqlite` or as a LUT: empirically, there's ~120K-1M games/db so this could be kept in memory eg as a tensor.
- Please create this as a separate objective in `packages/train_2048`. You might need to change the objective contract but ensure backwards compatibility
- Ensure logging in wandb is descriptive: total loss and loss (as a separate section).
- We should support options for ablation:
    - From-scratch training (both losses)
    - SFT on pretrained checkpoint
    - Probe on top of frozen trunk
- Knobs to tweak:
    - Relative weight of policy vs value losses (default equal as per AlphaGo)
    - Pooling before final (default: reuse mean-pooled representation before per-move layers; could require separate projection or even separate final block)
- Please ensure there's sufficient docs and knobs in:
    - The training config
    - The modeling (`packages/core_2048`) and model json config

NOTE: using cumulative reward is technically possible but vexed for implementation. We should consider this only if/when reaching-rate based value head fails.

## Phase 2: Implementing MCTS at inference

TODO
