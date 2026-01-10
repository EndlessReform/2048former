from __future__ import annotations

import numpy as np

from train_2048.augmentation.rotation import (
    rotate_board_exps,
    rotate_branch_udlr,
    rotate_legal_bits,
    rotate_move_dir,
)


def test_rotate_board_exps():
    base = np.arange(16, dtype=np.int64)
    exps = np.stack(
        [
            base,
            base + 100,
            base + 200,
            base + 300,
        ],
        axis=0,
    )
    k = np.array([0, 1, 2, 3], dtype=np.int64)
    rotated = rotate_board_exps(exps, k)

    expected_0 = base
    expected_1 = np.array(
        [12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3],
        dtype=np.int64,
    )
    expected_2 = np.array(
        [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        dtype=np.int64,
    )
    expected_3 = np.array(
        [3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12],
        dtype=np.int64,
    )
    expected = np.stack(
        [expected_0, expected_1 + 100, expected_2 + 200, expected_3 + 300],
        axis=0,
    )

    assert np.array_equal(rotated, expected)


def test_rotate_branch_udlr_permutations():
    values = np.array([[10, 11, 12, 13]], dtype=np.float32)
    perms = [
        [0, 1, 2, 3],
        [2, 3, 1, 0],
        [1, 0, 3, 2],
        [3, 2, 0, 1],
    ]
    for k, perm in enumerate(perms):
        rotated = rotate_branch_udlr(values, np.array([k], dtype=np.int64))
        expected = values[:, perm]
        assert np.array_equal(rotated, expected)


def test_rotate_move_dir_permutations():
    move_dir = np.array([0, 1, 2, 3], dtype=np.int64)
    cases = {
        1: np.array([3, 2, 0, 1], dtype=np.int64),
        2: np.array([1, 0, 3, 2], dtype=np.int64),
        3: np.array([2, 3, 1, 0], dtype=np.int64),
    }
    for k, expected in cases.items():
        rotated = rotate_move_dir(move_dir, np.full(4, k, dtype=np.int64))
        assert np.array_equal(rotated, expected)


def test_rotate_legal_bits_matches_udlr_perm():
    bits = np.array([0b0101, 0b1010, 0b0011], dtype=np.uint8)
    k = np.array([1, 2, 3], dtype=np.int64)

    rotated = rotate_legal_bits(bits, k)

    expanded = ((bits[:, None] >> np.arange(4)) & 1).astype(np.uint8)
    rotated_expanded = rotate_branch_udlr(expanded, k)
    expected = np.sum(rotated_expanded << np.arange(4), axis=1, dtype=np.uint8)

    assert np.array_equal(rotated, expected)
