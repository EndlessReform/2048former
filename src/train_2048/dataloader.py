import ai_2048 as a2
import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import Callable, Optional

from .binning import Binner


class StepBatchDataset(IterableDataset):
    def __init__(
        self,
        pack_path: str,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int | None = 0,
    ):
        self.pack_path = pack_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = 0 if seed is None else int(seed)

    def __iter__(self):
        info = get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers

        # Open the packfile in this process
        reader = a2.PackReader.open(self.pack_path)

        # Derive a per-worker seed for deterministic but distinct shuffles
        worker_seed = (self.seed or 0) + worker_id
        it = reader.iter_step_batches(
            self.batch_size, shuffle=self.shuffle, seed=worker_seed
        )

        # Simple batch-level sharding by worker to avoid duplicate work
        for i, batch in enumerate(it):
            if i % num_workers == worker_id:
                # Some pack versions yield (pre_boards, chosen_dirs, branch_evs).
                # Drop chosen_dirs so downstream only sees (pre_boards, branch_evs).
                if isinstance(batch, tuple) and len(batch) == 3:
                    pre_boards, _chosen_dirs, branch_evs = batch
                    yield (pre_boards, branch_evs)
                else:
                    yield batch  # expected: (pre_boards, branch_evs)


def collate_step_batches(batch_list):
    """
    batch_list is a list of tuples produced by StepBatchDataset, where each
    element is already a batch: (pre_boards, branch_evs). For backward
    compatibility, elements may also be 3-tuples (pre_boards, chosen_dirs, branch_evs);
    in that case chosen_dirs is ignored.
    This collate merges them and applies custom tokenization.
    """

    # Normalize input: DataLoader with batch_size=None may pass a single sample
    # tuple directly instead of a list. Ensure we always iterate a list of
    # (pre_boards, branch_evs) or (pre_boards, chosen_dirs, branch_evs).
    if isinstance(batch_list, tuple) and len(batch_list) in (2, 3):
        batch_list = [batch_list]

    pre_list = []  # (N, 16) exponents
    mask_list = []  # (N, 4) 1 if legal else 0
    val_list = []  # (N, 4) EVs, 0 where illegal

    for elem in batch_list:
        if isinstance(elem, tuple) and len(elem) == 3:
            pre_boards, _chosen_dirs, branch_evs = elem
        else:
            pre_boards, branch_evs = elem
        pre_list.extend(pre_boards)
        for brs in branch_evs:
            # brs: list[BranchV2] in [Up, Down, Left, Right]
            mask = [1 if b.is_legal else 0 for b in brs]
            vals = [float(b.value) if b.is_legal else 0.0 for b in brs]
            mask_list.append(mask)
            val_list.append(vals)

    pre = torch.tensor(pre_list, dtype=torch.int64)  # (N, 16)
    branch_mask = torch.tensor(mask_list, dtype=torch.bool)  # (N, 4)
    branch_vals = torch.tensor(val_list, dtype=torch.float32)  # (N, 4)

    # Example custom tokenization on exponents (optional):
    # Map exponent e -> token id e (0..15), or embed elsewhere.
    tokens = pre  # replace with your tokenizer if desired

    return {
        "tokens": tokens,  # (N, 16) int64
        "branch_mask": branch_mask,  # (N, 4) bool
        "branch_values": branch_vals,  # (N, 4) float32
    }


def make_collate_step_batches(binner: Optional[Binner]) -> Callable:
    """
    Factory for a collate_fn that also discretizes branch EVs using the provided binner.

    Returns a function suitable for DataLoader.collate_fn that produces:
    - tokens: (N, 16) int64
    - branch_mask: (N, 4) bool
    - branch_values: (N, 4) float32 (raw EVs)
    - branch_bin_targets: (N, 4) int64 (only if binner is provided)
    - n_bins: int (only if binner is provided)
    """

    def _collate(batch_list):
        from .dataloader import collate_step_batches as _base

        out = _base(batch_list)
        if binner is None:
            return out

        vals = out["branch_values"]  # (N, 4)
        # Keep bins on same device as values
        binner.to_device(vals.device)
        bins = binner.bin_values(vals)
        out["branch_bin_targets"] = bins.long()
        out["n_bins"] = binner.n_bins
        return out

    return _collate


"""
# Usage
ds = StepBatchDataset(
    "/path/to/dataset.a2pack", batch_size=1024, shuffle=True, seed=123
)
dl = DataLoader(
    ds, batch_size=None, num_workers=4, collate_fn=collate_step_batches, pin_memory=True
)
"""
