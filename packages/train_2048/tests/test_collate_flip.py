from __future__ import annotations

import numpy as np

from train_2048.augmentation.flip import (
    flip_board_exps,
    flip_branch_udlr,
    flip_move_dir,
)
from train_2048.augmentation.rotation import (
    rotate_board_exps,
    rotate_branch_udlr,
    rotate_move_dir,
)
from train_2048.dataloader.collate import make_collate_steps


class _DummyDataset:
    def __init__(self, rows: np.ndarray) -> None:
        self._rows = rows

    def get_rows(self, idxs: np.ndarray) -> np.ndarray:
        return self._rows[idxs]


def _pack_exps_to_u64(exps: np.ndarray) -> np.uint64:
    packed = np.uint64(0)
    for i, value in enumerate(exps.tolist()):
        shift = np.uint64((15 - i) * 4)
        packed |= np.uint64(int(value) & 0xF) << shift
    return packed


def _make_rows() -> np.ndarray:
    exps = np.arange(16, dtype=np.uint8)
    packed = _pack_exps_to_u64(exps)
    dtype = np.dtype(
        [
            ("board", "<u8"),
            ("branch_evs", "<f4", (4,)),
            ("ev_legal", "<u1"),
            ("move_dir", "<u1"),
        ],
        align=True,
    )
    rows = np.zeros(1, dtype=dtype)
    rows[0]["board"] = packed
    rows[0]["branch_evs"] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    rows[0]["ev_legal"] = np.uint8(0b1011)
    rows[0]["move_dir"] = np.uint8(2)
    return rows


def test_make_collate_steps_flip(monkeypatch) -> None:
    rows = _make_rows()
    exps = np.arange(16, dtype=np.uint8)
    dataset = _DummyDataset(rows)

    def _fixed_axis(*args, **kwargs) -> np.ndarray:
        return np.array([1], dtype=np.int64)

    monkeypatch.setattr("train_2048.dataloader.collate.sample_flip_axis", _fixed_axis)

    flip_cfg = type(
        "FlipCfg",
        (),
        {"mode": "random_axis", "seed": 123, "allow_noop": True},
    )()

    collate = make_collate_steps(
        "hard_move",
        dataset,
        binner=None,
        ev_tokenizer=None,
        flip_augment=flip_cfg,
    )
    out = collate([0])

    expected_tokens = flip_board_exps(exps[None, :], np.array([1], dtype=np.int64))
    expected_evs = flip_branch_udlr(rows["branch_evs"], np.array([1], dtype=np.int64))
    expected_move = flip_move_dir(
        rows["move_dir"].astype(np.int64, copy=False),
        np.array([1], dtype=np.int64),
    )

    assert np.array_equal(out["tokens"].numpy(), expected_tokens)
    assert np.array_equal(out["branch_values"].numpy(), expected_evs)
    assert np.array_equal(out["move_targets"].numpy(), expected_move)


def test_make_collate_steps_rotation_flip(monkeypatch) -> None:
    rows = _make_rows()
    exps = np.arange(16, dtype=np.uint8)
    dataset = _DummyDataset(rows)

    def _fixed_k(*args, **kwargs) -> np.ndarray:
        return np.array([1], dtype=np.int64)

    def _fixed_axis(*args, **kwargs) -> np.ndarray:
        return np.array([2], dtype=np.int64)

    monkeypatch.setattr("train_2048.dataloader.collate.sample_rotation_k", _fixed_k)
    monkeypatch.setattr("train_2048.dataloader.collate.sample_flip_axis", _fixed_axis)

    rotation_cfg = type(
        "RotationCfg",
        (),
        {"mode": "random_k", "seed": 123, "allow_noop": True},
    )()
    flip_cfg = type(
        "FlipCfg",
        (),
        {"mode": "random_axis", "seed": 456, "allow_noop": True},
    )()

    collate = make_collate_steps(
        "hard_move",
        dataset,
        binner=None,
        ev_tokenizer=None,
        rotation_augment=rotation_cfg,
        flip_augment=flip_cfg,
    )
    out = collate([0])

    rotated = rotate_board_exps(exps[None, :], np.array([1], dtype=np.int64))
    expected_tokens = flip_board_exps(rotated, np.array([2], dtype=np.int64))
    expected_evs = flip_branch_udlr(
        rotate_branch_udlr(rows["branch_evs"], np.array([1], dtype=np.int64)),
        np.array([2], dtype=np.int64),
    )
    rotated_move = rotate_move_dir(
        rows["move_dir"].astype(np.int64, copy=False),
        np.array([1], dtype=np.int64),
    )
    expected_move = flip_move_dir(rotated_move, np.array([2], dtype=np.int64))

    assert np.array_equal(out["tokens"].numpy(), expected_tokens)
    assert np.array_equal(out["branch_values"].numpy(), expected_evs)
    assert np.array_equal(out["move_targets"].numpy(), expected_move)
