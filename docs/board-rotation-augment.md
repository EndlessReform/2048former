Board Rotation + Flip Augmentation (1x)
=======================================

Goal
----
Add a training-time augmentation that rotates and optionally flips each board
once (no dataset expansion). Each sample is replaced by a rotated and/or
mirrored version (0, 90, 180, 270 degrees; left-right or up-down flips) and
all UDLR-aligned targets are permuted to match. No inference or wire protocol
changes are required.

Scope and rules
---------------
- Applies to training collate only (no changes to dataset shards on disk).
- Rotations and flips must keep UDLR semantics correct for:
  - branch EVs
  - legal mask (ev_legal bits)
  - move_dir / hard_move labels
  - macroxue tokens (v2; v1 is not supported with rotation/flip)
- For macroxue v2 "search" tokenization: board_eval must be recomputed for the
  rotated/flipped board.

UDLR permutation reference
--------------------------
UDLR indices: Up=0, Down=1, Left=2, Right=3.

Rotate 90 deg CW:
- perm (rot[i] = orig[perm[i]]) = [2, 3, 1, 0]
- move_dir_rot = inv_perm[move_dir_orig] = [3, 2, 0, 1]

Rotate 180 deg:
- perm = [1, 0, 3, 2]
- move_dir_rot = [1, 0, 3, 2]

Rotate 270 deg CW:
- perm = [3, 2, 0, 1]
- move_dir_rot = [2, 3, 1, 0]

Flip left-right:
- perm = [0, 1, 3, 2]
- move_dir_flip = [0, 1, 3, 2]

Flip up-down:
- perm = [1, 0, 2, 3]
- move_dir_flip = [1, 0, 2, 3]


Current implementation
======================
- Config: `packages/train_2048/src/train_2048/config.py` adds
  `RotationAugmentConfig`, `FlipAugmentConfig`, and the corresponding
  `DatasetConfig.rotation_augment` / `DatasetConfig.flip_augment`.
- Utilities: `packages/train_2048/src/train_2048/augmentation/rotation.py`
  provides `rotate_board_exps`, `rotate_branch_udlr`, `rotate_move_dir`,
  `rotate_legal_bits`, plus `sample_rotation_k` and `make_rotation_rng`.
- Utilities: `packages/train_2048/src/train_2048/augmentation/flip.py`
  provides `flip_board_exps`, `flip_branch_udlr`, `flip_move_dir`,
  `flip_legal_bits`, plus `sample_flip_axis` and `make_flip_rng`.
- Collate integration:
  - `packages/train_2048/src/train_2048/dataloader/collate.py` applies rotation
    and flip in `make_collate_steps`, `make_collate_steps_worker_safe`,
    `make_collate_macroxue`, and `make_collate_macroxue_worker_safe`.
  - Board eval recomputation for macroxue v2 happens only for rotated/flipped
    rows.
  - Rotation/flip with macroxue v1 raises an assertion.
- Wiring: `packages/train_2048/src/train_2048/dataloader/__init__.py`,
  `packages/train_2048/src/train_2048/dataloader/steps.py`,
  `packages/train_2048/src/train_2048/dataloader/steps_v2.py`, and
  `packages/train_2048/src/train_2048/training_loop.py` pass
  `rotation_augment` and `flip_augment` into collate builders.
- Tests:
  - `packages/train_2048/tests/test_rotation_augment.py` validates index and
    UDLR permutations.
  - `packages/train_2048/tests/test_collate_rotation.py` validates collate
    rotation behavior.
  - `packages/train_2048/tests/test_flip_augment.py` validates flip index and
    UDLR permutations.
  - `packages/train_2048/tests/test_collate_flip.py` validates collate flip
    behavior.


Performance considerations
==========================
- Avoid per-row Python loops; use vectorized NumPy operations.
- Keep copies minimal: avoid `copy()` unless needed to break memmap views.
- Rotation/flip should be opt-in and should not slow the no-augmentation path.
- Recompute `board_eval` only for rotated/flipped rows to avoid extra CPU.
- For multi-worker DataLoader, use deterministic per-worker RNG or pass
  augmentations in from a seeded generator to keep reproducibility.
- Be careful with pinned-memory paths (inference is unaffected; training
  collate should remain CPU-bound and parallelized by workers).


Implementation checklist
========================
- [x] Add rotation config (Task A).
- [x] Implement rotation utilities (Task B).
- [x] Integrate into collate paths (Task C).
- [x] Add unit tests (Task E).
- [x] Run existing tokenizer tests plus new rotation tests.
- [x] Add flip config (Task F).
- [x] Implement flip utilities (Task G).
- [x] Integrate flip into collate paths (Task H).
- [x] Add flip unit tests (Task I).


Usage
=====

To enable board rotation augmentation, add the following to your training config
TOML:

```toml
[dataset.rotation_augment]
mode = "random_k"
allow_noop = true  # Includes 0, 90, 180, and 270 degree rotations
```

If `mode` is `"none"` (default), no augmentation is applied. If `mode` is
`"random_k"`, each sample in a batch is independently rotated by 0, 90, 180, or
270 degrees.

Notes:
- Rotation is supported for macroxue v2; macroxue v1 will assert if rotation is enabled.

To enable board flip augmentation, add the following to your training config
TOML:

```toml
[dataset.flip_augment]
mode = "random_axis"
allow_noop = true  # Includes no-flip, left-right, and up-down
```

If `mode` is `"random_axis"`, each sample in a batch is independently flipped
left-right or up-down (and can include no-op if `allow_noop` is true). If both
rotation and flip are enabled, rotation is applied first, then flip.

Notes:
- Flip is supported for macroxue v2; macroxue v1 will assert if flip is enabled.


Flip augmentation
=================
Flip augmentation mirrors the board either left-right or up-down. Use
`mode = "random_axis"` to sample per-row flips in `{none, left-right, up-down}`.
