from __future__ import annotations

from math import ceil
from typing import Optional, Tuple

from torch.utils.data import DataLoader

from ..binning import Binner
from ..config import TrainingConfig
from .pack import StepBatchDataset, make_collate_step_batches
from .jsonl import make_collate_hf_steps


def build_dataloaders(
    cfg: TrainingConfig,
    binner: Binner,
    *,
    num_workers_train: int = 12,
) -> Tuple[DataLoader, Optional[DataLoader], int]:
    """Construct train/val DataLoaders and return (dl_train, dl_val, per_epoch_steps).

    - If `cfg.dataset.packfile` ends with `.jsonl`, builds HF Dataset loaders.
    - Otherwise, uses ai_2048 PackReader via StepBatchDataset.
    """

    from datasets import Dataset as HFDataset  # type: ignore

    pack_path = cfg.dataset.resolved_packfile()
    is_json = pack_path.lower().endswith(".jsonl")

    prefetch_train = 8 if num_workers_train > 0 else None

    if is_json:
        # JSONL via HF datasets (map-style); step-based split
        ds_all = HFDataset.from_json(pack_path)
        ds_train_hf = ds_all
        ds_val_hf = None

        split_enabled = (getattr(cfg.dataset, "val_steps", 0) or 0) > 0 or (
            getattr(cfg.dataset, "val_pct", 0.0) or 0.0
        ) > 0.0
        if split_enabled:
            val_steps = int(getattr(cfg.dataset, "val_steps", 0) or 0)
            val_pct = float(getattr(cfg.dataset, "val_pct", 0.0) or 0.0)
            test_size = (
                val_steps if val_steps > 0 else (val_pct if val_pct > 0.0 else None)
            )
            if test_size is not None and (
                (isinstance(test_size, int) and test_size > 0)
                or (isinstance(test_size, float) and test_size > 0.0)
            ):
                split = ds_all.train_test_split(test_size=test_size, shuffle=True, seed=42)
                ds_train_hf, ds_val_hf = split["train"], split["test"]

        collate_json = make_collate_hf_steps(binner)
        dl_train = DataLoader(
            ds_train_hf,
            batch_size=cfg.batch.batch_size,
            shuffle=True,
            num_workers=num_workers_train,
            collate_fn=collate_json,
            pin_memory=True,
            persistent_workers=True if num_workers_train > 0 else False,
            prefetch_factor=prefetch_train if prefetch_train is not None else 2,
        )

        dl_val = None
        if ds_val_hf is not None:
            num_workers_val = max(2, num_workers_train // 3)
            prefetch_val = 4 if num_workers_val > 0 else None
            dl_val = DataLoader(
                ds_val_hf,
                batch_size=cfg.batch.batch_size,
                shuffle=False,
                num_workers=num_workers_val,
                collate_fn=collate_json,
                pin_memory=True,
                persistent_workers=True if num_workers_val > 0 else False,
                prefetch_factor=prefetch_val if prefetch_val is not None else 2,
            )

        per_epoch_steps = ceil(len(ds_train_hf) / max(1, int(cfg.batch.batch_size)))
        return dl_train, dl_val, per_epoch_steps

    # a2pack via ai_2048 PackReader
    split_enabled = (getattr(cfg.dataset, "val_steps", 0) or 0) > 0 or (
        getattr(cfg.dataset, "val_pct", 0.0) or 0.0
    ) > 0.0
    split_unit = "step" if (cfg.dataset.val_steps or 0) > 0 else "run"
    split_test_size = int(cfg.dataset.val_steps or 0) if split_unit == "step" else None
    split_test_pct = float(cfg.dataset.val_pct or 0.0) if split_unit == "run" else None

    collate_fn = make_collate_step_batches(binner)
    ds_train = StepBatchDataset(
        pack_path=pack_path,
        batch_size=cfg.batch.batch_size,
        shuffle=True,
        seed=cfg.seed,
        split_role=("train" if split_enabled else "none"),
        split_unit=split_unit,
        split_test_size=split_test_size,
        split_test_pct=split_test_pct,
        split_seed=42,
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=None,
        num_workers=num_workers_train,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers_train > 0 else False,
        prefetch_factor=prefetch_train if prefetch_train is not None else 2,
    )

    dl_val = None
    if split_enabled:
        num_workers_val = max(2, num_workers_train // 3)
        prefetch_val = 4 if num_workers_val > 0 else None
        ds_val = StepBatchDataset(
            pack_path=pack_path,
            batch_size=cfg.batch.batch_size,
            shuffle=False,
            seed=cfg.seed,
            split_role="val",
            split_unit=split_unit,
            split_test_size=split_test_size,
            split_test_pct=split_test_pct,
            split_seed=42,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=None,
            num_workers=num_workers_val,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if num_workers_val > 0 else False,
            prefetch_factor=prefetch_val if prefetch_val is not None else 2,
        )

    per_epoch_steps = getattr(ds_train, "total_steps", 0) or 0
    return dl_train, dl_val, int(per_epoch_steps)


__all__ = [
    "build_dataloaders",
    "StepBatchDataset",
    "make_collate_step_batches",
    "make_collate_hf_steps",
]

