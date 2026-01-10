from __future__ import annotations

import numpy as np

from train_2048.augmentation.flip import (
    flip_board_exps,
    flip_branch_udlr,
    flip_legal_bits,
    flip_move_dir,
)


def test_flip_board_exps():
    base = np.arange(16, dtype=np.int64)
    exps = np.stack(
        [
            base,
            base + 100,
            base + 200,
        ],
        axis=0,
    )
    axis = np.array([0, 1, 2], dtype=np.int64)
    flipped = flip_board_exps(exps, axis)

    expected_0 = base
    expected_1 = np.array(
        [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12],
        dtype=np.int64,
    )
    expected_2 = np.array(
        [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3],
        dtype=np.int64,
    )
    expected = np.stack(
        [expected_0, expected_1 + 100, expected_2 + 200],
        axis=0,
    )

    assert np.array_equal(flipped, expected)


def test_flip_branch_udlr_permutations():
    values = np.array([[10, 11, 12, 13]], dtype=np.float32)
    perms = [
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [1, 0, 2, 3],
    ]
    for axis, perm in enumerate(perms):
        flipped = flip_branch_udlr(values, np.array([axis], dtype=np.int64))
        expected = values[:, perm]
        assert np.array_equal(flipped, expected)


def test_flip_move_dir_permutations():
    move_dir = np.array([0, 1, 2, 3], dtype=np.int64)
    cases = {
        1: np.array([0, 1, 3, 2], dtype=np.int64),
        2: np.array([1, 0, 2, 3], dtype=np.int64),
    }
    for axis, expected in cases.items():
        flipped = flip_move_dir(move_dir, np.full(4, axis, dtype=np.int64))
        assert np.array_equal(flipped, expected)


def test_flip_legal_bits_matches_udlr_perm():
    bits = np.array([0b0101, 0b1010, 0b0011], dtype=np.uint8)
    axis = np.array([1, 2, 1], dtype=np.int64)

    flipped = flip_legal_bits(bits, axis)

    expanded = ((bits[:, None] >> np.arange(4)) & 1).astype(np.uint8)
    flipped_expanded = flip_branch_udlr(expanded, axis)
    expected = np.sum(flipped_expanded << np.arange(4), axis=1, dtype=np.uint8)

    assert np.array_equal(flipped, expected)
