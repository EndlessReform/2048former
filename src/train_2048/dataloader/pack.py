from __future__ import annotations

import ai_2048 as a2
import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import Callable, Optional

from ..binning import Binner


class StepBatchDataset(IterableDataset):
    def __init__(
        self,
        pack_path: str,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int | None = 0,
        *,
        split_role: str = "none",  # "none" | "train" | "val"
        split_unit: str = "run",  # "run" | "step"
        split_test_pct: float | None = None,
        split_test_size: int | None = None,
        split_seed: int = 42,
    ):
        self.pack_path = pack_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = 0 if seed is None else int(seed)
        self.split_role = split_role
        self.split_unit = split_unit
        self.split_test_pct = split_test_pct
        self.split_test_size = split_test_size
        self.split_seed = int(split_seed)

    def __iter__(self):
        info = get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers

        reader = a2.PackReader.open(self.pack_path)

        # Optional split view
        view = reader
        if self.split_role in ("train", "val"):
            unit = self.split_unit or "run"
            if unit == "step" and (self.split_test_size or 0) > 0:
                train_v, val_v = reader.split(
                    unit="step", test_size=int(self.split_test_size), seed=int(self.split_seed)
                )
            elif (self.split_test_pct or 0.0) > 0.0:
                train_v, val_v = reader.split(
                    unit=unit, test_pct=float(self.split_test_pct), seed=int(self.split_seed)
                )
            else:
                train_v, val_v = reader, reader
            view = train_v if self.split_role == "train" else val_v

        # Per-worker seed for deterministic but distinct shuffles
        worker_seed = (self.seed or 0) + worker_id
        it = view.iter_step_batches(self.batch_size, shuffle=self.shuffle, seed=worker_seed)

        # Batch-level sharding by worker to avoid duplicate work
        for i, batch in enumerate(it):
            if i % num_workers == worker_id:
                if isinstance(batch, tuple) and len(batch) == 3:
                    pre_boards, _chosen_dirs, branch_evs = batch
                    yield (pre_boards, branch_evs)
                else:
                    yield batch

    @property
    def total_steps(self) -> int:
        reader = a2.PackReader.open(self.pack_path)
        try:
            if self.split_role in ("train", "val"):
                unit = self.split_unit or "run"
                if unit == "step" and (self.split_test_size or 0) > 0:
                    train_v, val_v = reader.split(
                        unit="step",
                        test_size=int(self.split_test_size),
                        seed=int(self.split_seed),
                    )
                elif (self.split_test_pct or 0.0) > 0.0:
                    train_v, val_v = reader.split(
                        unit=unit,
                        test_pct=float(self.split_test_pct),
                        seed=int(self.split_seed),
                    )
                else:
                    train_v, val_v = reader, reader
                view = train_v if self.split_role == "train" else val_v
                return int(getattr(view, "total_steps", 0))
            return int(getattr(reader, "total_steps", 0))
        finally:
            close_fn = getattr(reader, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass


def collate_step_batches(batch_list):
    # Normalize input shape
    if isinstance(batch_list, tuple) and len(batch_list) in (2, 3):
        batch_list = [batch_list]

    pre_list = []
    mask_list = []
    val_list = []

    for elem in batch_list:
        if isinstance(elem, tuple) and len(elem) == 3:
            pre_boards, _chosen_dirs, branch_evs = elem
        else:
            pre_boards, branch_evs = elem
        pre_list.extend(pre_boards)
        for brs in branch_evs:
            mask = [1 if b.is_legal else 0 for b in brs]
            vals = [float(b.value) if b.is_legal else 0.0 for b in brs]
            mask_list.append(mask)
            val_list.append(vals)

    pre = torch.tensor(pre_list, dtype=torch.int64)
    branch_mask = torch.tensor(mask_list, dtype=torch.bool)
    branch_vals = torch.tensor(val_list, dtype=torch.float32)
    tokens = pre
    return {"tokens": tokens, "branch_mask": branch_mask, "branch_values": branch_vals}


def make_collate_step_batches(binner: Optional[Binner]) -> Callable:
    def _collate(batch_list):
        out = collate_step_batches(batch_list)
        if binner is None:
            return out
        vals = out["branch_values"]
        binner.to_device(vals.device)
        out["branch_bin_targets"] = binner.bin_values(vals).long()
        out["n_bins"] = binner.n_bins
        return out

    return _collate


__all__ = ["StepBatchDataset", "collate_step_batches", "make_collate_step_batches"]

