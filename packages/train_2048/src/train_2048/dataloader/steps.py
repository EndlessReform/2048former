from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple

import numpy as np
import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader

from ..binning import Binner

if TYPE_CHECKING:
    from ..config import TargetConfig


class StepsDataset(Dataset):
    """Map-style dataset over steps.npy returning global indices.

    Collate performs slicing + fast exponent conversion via ai_2048.batch_from_steps.
    """

    def __init__(
        self,
        dataset_dir: str,
        indices: Optional[np.ndarray] = None,
    ) -> None:
        import numpy as _np  # local alias for type checkers

        self.dataset_dir = str(dataset_dir)
        steps_path = Path(self.dataset_dir) / "steps.npy"
        if not steps_path.is_file():
            raise FileNotFoundError(f"Missing steps.npy at {steps_path}")
        self.steps = _np.load(str(steps_path))
        if indices is None:
            self.indices = _np.arange(self.steps.shape[0], dtype=_np.int64)
        else:
            # Ensure int64 for torch interop
            self.indices = _np.asarray(indices, dtype=_np.int64)

    def __len__(self) -> int:  # type: ignore[override]
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> int:  # type: ignore[override]
        # Return the global step index; slicing happens in collate
        return int(self.indices[int(idx)])


def _select_run_ids(
    conn: sqlite3.Connection,
    steps: np.ndarray,
    *,
    run_sql: Optional[str],
    sql_params: Sequence | None,
    val_run_sql: Optional[str],
    val_sql_params: Sequence | None,
    val_run_pct: float,
    val_split_seed: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (train_run_ids, val_run_ids_or_None) as numpy arrays (dtype matches steps['run_id']).

    - If `val_run_sql` is provided, it defines validation runs; training are the rest in universe.
    - Else if `val_run_pct` in (0,1), do a random split of unique universe run IDs.
    - Else no validation split (second item is None).
    Universe is defined by `run_sql` when provided, else all runs present in steps.
    """

    # Determine dtype for run_id column from steps.npy
    if 'run_id' not in steps.dtype.names:
        raise KeyError("steps.npy is missing 'run_id' field needed for splitting")
    run_dtype = steps['run_id'].dtype

    def _fetch_ids(sql: str, params: Sequence | None) -> np.ndarray:
        rows = conn.execute(sql, tuple(params or ())).fetchall()
        return np.asarray([r[0] for r in rows], dtype=run_dtype)

    # Universe of runs
    if run_sql:
        uni = _fetch_ids(run_sql, sql_params)
    else:
        # Fallback: infer from steps directly
        uni = np.unique(steps['run_id'].astype(run_dtype))

    if val_run_sql:
        val_ids = _fetch_ids(val_run_sql, val_sql_params)
        # Intersect to ensure val is within universe
        val_ids = np.intersect1d(val_ids, uni, assume_unique=False)
        train_ids = np.setdiff1d(uni, val_ids, assume_unique=False)
        return train_ids, val_ids

    if val_run_pct > 0.0:
        uni_unique = np.unique(uni)
        rng = np.random.default_rng(int(val_split_seed))
        n_val = max(1, int(np.ceil(val_run_pct * len(uni_unique))))
        perm = rng.permutation(len(uni_unique))
        val_sel = np.sort(perm[:n_val])
        val_ids = uni_unique[val_sel]
        train_ids = np.setdiff1d(uni_unique, val_ids, assume_unique=True)
        return train_ids, val_ids

    return uni, None


def _indices_from_run_ids(steps: np.ndarray, run_ids: np.ndarray) -> np.ndarray:
    mask = np.isin(steps['run_id'], run_ids.astype(steps['run_id'].dtype, copy=False))
    return np.flatnonzero(mask).astype(np.int64, copy=False)


def make_collate_steps(
    target_mode: str,
    steps: np.ndarray,
    binner: Optional[Binner],
    target_cfg: "TargetConfig" | None = None,
) -> Callable:
    """Collate function that gathers tokens/mask/values and builds training targets."""

    if target_mode not in {"binned_ev", "hard_move", "tsad_soft"}:
        raise ValueError(f"Unknown target mode: {target_mode}")
    if target_mode == "binned_ev" and binner is None:
        raise ValueError("Binner is required when target_mode='binned_ev'")
    if target_mode == "tsad_soft" and target_cfg is None:
        raise ValueError("Target config is required when target_mode='tsad_soft'")

    import ai_2048 as a2  # lazy import

    def _collate(batch_indices: Sequence[int]):
        import numpy as _np

        idxs = _np.asarray(batch_indices, dtype=_np.int64)
        exps_buf, dirs, evs = a2.batch_from_steps(steps, idxs, parallel=True)

        # Tokens from exponents buffer
        exps = _np.frombuffer(exps_buf, dtype=_np.uint8).reshape(-1, 16)
        tokens = torch.from_numpy(exps.copy()).to(dtype=torch.int64)

        # EVs and legality mask from ev_legal bitfield (Up=1, Down=2, Left=4, Right=8)
        if not isinstance(evs, _np.ndarray):
            evs = _np.asarray(evs, dtype=_np.float32)
        evs = evs.astype(_np.float32, copy=False)

        if 'ev_legal' in steps.dtype.names:
            bits = steps['ev_legal'][idxs].astype(_np.uint8, copy=False)
            legal = _np.stack([
                (bits & 1) != 0,   # Up -> col 0
                (bits & 2) != 0,   # Down -> col 1
                (bits & 4) != 0,   # Left -> col 2
                (bits & 8) != 0,   # Right -> col 3
            ], axis=1)
        else:
            # Fallback: treat finite EVs as legal if mask not present
            legal = _np.isfinite(evs)

        evs_clean = (evs * legal.astype(_np.float32, copy=False)).astype(_np.float32, copy=False)

        branch_values = torch.from_numpy(evs_clean.copy())  # (N,4) float32
        branch_mask = torch.from_numpy(legal.astype(_np.bool_, copy=False))  # (N,4) bool

        out = {
            "tokens": tokens,
            "branch_mask": branch_mask,
            "branch_values": branch_values,
        }
        if target_mode == "binned_ev":
            assert binner is not None  # for type checkers
            binner.to_device(branch_values.device)
            out["branch_bin_targets"] = binner.bin_values(branch_values).long()
            out["n_bins"] = int(binner.n_bins)
        elif target_mode == "hard_move":
            dirs_arr = _np.asarray(dirs, dtype=_np.int64)
            out["move_targets"] = torch.from_numpy(dirs_arr.copy()).to(dtype=torch.long)
        else:  # tsad_soft
            assert target_cfg is not None
            dirs_arr = _np.asarray(dirs, dtype=_np.int64)
            mix_with_hard = float(getattr(target_cfg, "tsad_mix_with_hard", 0.0))
            temperature = float(getattr(target_cfg, "tsad_temperature", 1.0))
            min_scale = float(getattr(target_cfg, "tsad_min_scale", 1e-3))
            scale_kind = getattr(target_cfg, "tsad_scale_kind", "max_abs")

            # Teacher distribution initialised to zeros; fill legal moves per sample.
            soft_targets = _np.zeros_like(evs_clean, dtype=_np.float32)

            for i in range(evs_clean.shape[0]):
                legal_mask_i = legal[i]
                if not legal_mask_i.any():
                    continue
                legal_vals = evs_clean[i, legal_mask_i]
                if legal_vals.size == 0:
                    continue
                best = float(legal_vals.max())
                advantages = legal_vals - best

                if scale_kind == "mad":
                    center = _np.median(legal_vals)
                    scale = float(_np.mean(_np.abs(legal_vals - center)))
                else:  # max_abs
                    scale = float(_np.max(_np.abs(advantages)))
                scale = max(scale, min_scale)
                denom = max(scale * temperature, min_scale)
                logits = advantages / denom
                logits = logits - _np.max(logits)
                probs = _np.exp(logits)
                probs_sum = float(_np.sum(probs))
                if not _np.isfinite(probs_sum) or probs_sum <= 0.0:
                    probs = _np.ones_like(probs)
                    probs_sum = float(probs.size)
                probs = probs / probs_sum

                probs_full = _np.zeros(4, dtype=_np.float32)
                probs_full[legal_mask_i] = probs.astype(_np.float32, copy=False)

                move_idx = int(dirs_arr[i])
                hard_full = _np.zeros(4, dtype=_np.float32)
                if 0 <= move_idx < 4 and legal_mask_i[move_idx]:
                    hard_full[move_idx] = 1.0
                else:
                    # Fallback: pin to current argmax probability.
                    best_idx_rel = int(probs.argmax())
                    legal_indices = _np.flatnonzero(legal_mask_i)
                    best_global = int(legal_indices[best_idx_rel])
                    hard_full[best_global] = 1.0

                mixed = (1.0 - mix_with_hard) * probs_full + mix_with_hard * hard_full
                mixed_sum = float(_np.sum(mixed[legal_mask_i]))
                if mixed_sum <= 0.0 or not _np.isfinite(mixed_sum):
                    mixed = probs_full
                    mixed_sum = float(_np.sum(mixed[legal_mask_i]))
                if mixed_sum <= 0.0:
                    # Uniform over legal moves as last resort.
                    legal_indices = _np.flatnonzero(legal_mask_i)
                    mixed = _np.zeros(4, dtype=_np.float32)
                    mixed[legal_indices] = 1.0 / max(len(legal_indices), 1)
                    mixed_sum = 1.0
                soft_targets[i] = (mixed / mixed_sum).astype(_np.float32, copy=False)

            out["policy_targets"] = torch.from_numpy(soft_targets.copy()).to(
                dtype=torch.float32
            )
            out["move_targets"] = torch.from_numpy(dirs_arr.copy()).to(dtype=torch.long)
        return out

    return _collate


def build_steps_dataloaders(
    dataset_dir: str,
    binner: Optional[Binner],
    target_mode: str,
    target_cfg: "TargetConfig" | None,
    batch_size: int,
    *,
    run_sql: Optional[str] = None,
    sql_params: Sequence | None = None,
    val_run_sql: Optional[str] = None,
    val_sql_params: Sequence | None = None,
    val_run_pct: float = 0.0,
    val_split_seed: int = 42,
    num_workers_train: int = 12,
) -> Tuple[DataLoader, Optional[DataLoader], int]:
    """Create train/val DataLoaders from steps.npy + metadata.db.

    Returns (dl_train, dl_val_or_None, per_epoch_steps).
    """

    steps_path = Path(dataset_dir) / "steps.npy"
    meta_path = Path(dataset_dir) / "metadata.db"
    if not steps_path.is_file():
        raise FileNotFoundError(f"Missing steps.npy at {steps_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata.db at {meta_path}")

    steps = np.load(str(steps_path))
    conn = sqlite3.connect(str(meta_path))
    try:
        train_rids, val_rids = _select_run_ids(
            conn,
            steps,
            run_sql=run_sql,
            sql_params=sql_params or (),
            val_run_sql=val_run_sql,
            val_sql_params=val_sql_params or (),
            val_run_pct=float(val_run_pct or 0.0),
            val_split_seed=int(val_split_seed or 42),
        )
    finally:
        conn.close()

    train_indices = _indices_from_run_ids(steps, train_rids)
    val_indices = _indices_from_run_ids(steps, val_rids) if val_rids is not None else None

    ds_train = StepsDataset(dataset_dir, indices=train_indices)
    collate = make_collate_steps(target_mode, ds_train.steps, binner, target_cfg)

    prefetch_train = 8 if num_workers_train > 0 else None
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers_train,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=True if num_workers_train > 0 else False,
        prefetch_factor=prefetch_train if prefetch_train is not None else 2,
    )

    dl_val: Optional[DataLoader] = None
    if val_indices is not None and val_indices.size > 0:
        num_workers_val = max(2, num_workers_train // 3)
        prefetch_val = 4 if num_workers_val > 0 else None
        ds_val = StepsDataset(dataset_dir, indices=val_indices)
        collate_v = make_collate_steps(target_mode, ds_val.steps, binner, target_cfg)
        dl_val = DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers_val,
            collate_fn=collate_v,
            pin_memory=True,
            persistent_workers=True if num_workers_val > 0 else False,
            prefetch_factor=prefetch_val if prefetch_val is not None else 2,
        )

    per_epoch_steps = ceil(len(train_indices) / max(1, int(batch_size)))
    return dl_train, dl_val, per_epoch_steps


__all__ = [
    "StepsDataset",
    "make_collate_steps",
    "build_steps_dataloaders",
]
