from __future__ import annotations

from typing import Optional

import numpy as np

_BASE_IDX = np.arange(16, dtype=np.int64).reshape(4, 4)
_FLIP_INDEX_MAPS = np.stack(
    [
        _BASE_IDX,
        np.fliplr(_BASE_IDX),
        np.flipud(_BASE_IDX),
    ],
    axis=0,
).reshape(3, 16)

_UDLR_PERMS = np.array(
    [
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [1, 0, 2, 3],
    ],
    dtype=np.int64,
)
_INV_UDLR_PERMS = np.array(
    [
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [1, 0, 2, 3],
    ],
    dtype=np.int64,
)
_UDLR_BIT_POSITIONS = np.array([0, 1, 2, 3], dtype=np.uint8)


def make_flip_rng(seed: Optional[int]) -> np.random.Generator:
    """Create a numpy RNG with a deterministic worker-specific seed."""
    if seed is None:
        return np.random.default_rng()

    try:
        import torch

        worker_info = torch.utils.data.get_worker_info()
    except Exception:
        worker_info = None

    if worker_info is None:
        return np.random.default_rng(seed)

    seed_seq = np.random.SeedSequence(seed, spawn_key=(worker_info.id,))
    return np.random.default_rng(seed_seq)


def sample_flip_axis(
    batch_size: int,
    *,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    allow_noop: bool = True,
) -> np.ndarray:
    """Sample per-row flip axis in {0,1,2} for a batch."""
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if rng is None:
        rng = make_flip_rng(seed)
    low = 0 if allow_noop else 1
    return rng.integers(low, 3, size=batch_size, dtype=np.int8)


def flip_board_exps(exps: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Flip [B,16] row-major exponents with per-row axis (0=noop, 1=LR, 2=UD)."""
    exps = np.asarray(exps)
    if exps.ndim != 2 or exps.shape[1] != 16:
        raise ValueError("exps must have shape [B, 16]")
    axis_arr = _normalize_axis(axis, exps.shape[0])
    idx = _FLIP_INDEX_MAPS[axis_arr]
    return np.take_along_axis(exps, idx, axis=1)


def flip_branch_udlr(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Flip [B,4] UDLR-aligned values with per-row axis (0=noop, 1=LR, 2=UD)."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("values must have shape [B, 4]")
    axis_arr = _normalize_axis(axis, values.shape[0])
    idx = _UDLR_PERMS[axis_arr]
    return np.take_along_axis(values, idx, axis=1)


def flip_move_dir(move_dir: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Flip [B] move_dir indices (UDLR order) with per-row axis."""
    move_dir = np.asarray(move_dir)
    if move_dir.ndim != 1:
        raise ValueError("move_dir must have shape [B]")
    axis_arr = _normalize_axis(axis, move_dir.shape[0])
    if np.any((move_dir < 0) | (move_dir > 3)):
        raise ValueError("move_dir values must be in [0, 3]")
    return _INV_UDLR_PERMS[axis_arr, move_dir.astype(np.int64, copy=False)]


def flip_legal_bits(bits: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Flip [B] UDLR bitmask values with per-row axis."""
    bits = np.asarray(bits)
    if bits.ndim != 1:
        raise ValueError("bits must have shape [B]")
    axis_arr = _normalize_axis(axis, bits.shape[0])
    expanded = ((bits[:, None] >> _UDLR_BIT_POSITIONS) & 1).astype(np.uint8)
    flipped = np.take_along_axis(expanded, _UDLR_PERMS[axis_arr], axis=1)
    merged = np.sum(flipped << _UDLR_BIT_POSITIONS, axis=1, dtype=np.uint8)
    return merged.astype(bits.dtype, copy=False)


def _normalize_axis(axis: np.ndarray, batch_size: int) -> np.ndarray:
    axis_arr = np.asarray(axis, dtype=np.int64)
    if axis_arr.ndim == 0:
        axis_arr = np.full((batch_size,), int(axis_arr), dtype=np.int64)
    if axis_arr.ndim != 1 or axis_arr.shape[0] != batch_size:
        raise ValueError("axis must have shape [B]")
    if np.any((axis_arr < 0) | (axis_arr > 2)):
        raise ValueError("axis values must be in {0,1,2}")
    return axis_arr
