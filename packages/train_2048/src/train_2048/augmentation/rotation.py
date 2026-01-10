from __future__ import annotations

from typing import Optional

import numpy as np

_BASE_IDX = np.arange(16, dtype=np.int64).reshape(4, 4)
_ROTATION_INDEX_MAPS = np.stack(
    [
        _BASE_IDX,
        np.rot90(_BASE_IDX, -1),
        np.rot90(_BASE_IDX, 2),
        np.rot90(_BASE_IDX, 1),
    ],
    axis=0,
).reshape(4, 16)

_UDLR_PERMS = np.array(
    [
        [0, 1, 2, 3],
        [2, 3, 1, 0],
        [1, 0, 3, 2],
        [3, 2, 0, 1],
    ],
    dtype=np.int64,
)
_INV_UDLR_PERMS = np.array(
    [
        [0, 1, 2, 3],
        [3, 2, 0, 1],
        [1, 0, 3, 2],
        [2, 3, 1, 0],
    ],
    dtype=np.int64,
)
_UDLR_BIT_POSITIONS = np.array([0, 1, 2, 3], dtype=np.uint8)


def make_rotation_rng(seed: Optional[int]) -> np.random.Generator:
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


def sample_rotation_k(
    batch_size: int,
    *,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    allow_noop: bool = True,
) -> np.ndarray:
    """Sample per-row rotation k in {0,1,2,3} for a batch."""
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if rng is None:
        rng = make_rotation_rng(seed)
    low = 0 if allow_noop else 1
    return rng.integers(low, 4, size=batch_size, dtype=np.int8)


def rotate_board_exps(exps: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Rotate [B,16] row-major exponents with per-row rotation k."""
    exps = np.asarray(exps)
    if exps.ndim != 2 or exps.shape[1] != 16:
        raise ValueError("exps must have shape [B, 16]")
    k_arr = _normalize_k(k, exps.shape[0])
    idx = _ROTATION_INDEX_MAPS[k_arr]
    return np.take_along_axis(exps, idx, axis=1)


def rotate_branch_udlr(values: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Rotate [B,4] UDLR-aligned values with per-row rotation k."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("values must have shape [B, 4]")
    k_arr = _normalize_k(k, values.shape[0])
    idx = _UDLR_PERMS[k_arr]
    return np.take_along_axis(values, idx, axis=1)


def rotate_move_dir(move_dir: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Rotate [B] move_dir indices (UDLR order) with per-row rotation k."""
    move_dir = np.asarray(move_dir)
    if move_dir.ndim != 1:
        raise ValueError("move_dir must have shape [B]")
    k_arr = _normalize_k(k, move_dir.shape[0])
    if np.any((move_dir < 0) | (move_dir > 3)):
        raise ValueError("move_dir values must be in [0, 3]")
    return _INV_UDLR_PERMS[k_arr, move_dir.astype(np.int64, copy=False)]


def rotate_legal_bits(bits: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Rotate [B] UDLR bitmask values with per-row rotation k."""
    bits = np.asarray(bits)
    if bits.ndim != 1:
        raise ValueError("bits must have shape [B]")
    k_arr = _normalize_k(k, bits.shape[0])
    expanded = ((bits[:, None] >> _UDLR_BIT_POSITIONS) & 1).astype(np.uint8)
    rotated = np.take_along_axis(expanded, _UDLR_PERMS[k_arr], axis=1)
    merged = np.sum(rotated << _UDLR_BIT_POSITIONS, axis=1, dtype=np.uint8)
    return merged.astype(bits.dtype, copy=False)


def _normalize_k(k: np.ndarray, batch_size: int) -> np.ndarray:
    k_arr = np.asarray(k, dtype=np.int64)
    if k_arr.ndim == 0:
        k_arr = np.full((batch_size,), int(k_arr), dtype=np.int64)
    if k_arr.ndim != 1 or k_arr.shape[0] != batch_size:
        raise ValueError("k must have shape [B]")
    if np.any((k_arr < 0) | (k_arr > 3)):
        raise ValueError("k values must be in {0,1,2,3}")
    return k_arr
