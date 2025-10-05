"""Heuristic scoring for 2048 boards.

This module mirrors the behaviour of ``Node::BuildScoreMap`` and
``Node::Evaluate`` from ``node.h``.  Boards are encoded using tile ranks
(exponents) exactly as stored inside ``Board::board`` (see ``board.h``) and as
emitted by the engine's JSONL step logs in ``2048.cc`` (see the block that writes
``"board":[...]`` when the ``-J`` flag is in use).  The utility functions accept
both 4×4 matrices and flat length-16 sequences of ranks ordered row-major.
"""

from __future__ import annotations

import argparse
import numbers

# Board size constant (same as the C++ ``N``)
N = 4

# Map size: 1 << (5 * N) = 1 << 20
MAP_SIZE = 1 << (5 * N)

# Score maps: ascending and descending scores for each encoded line
score_map_descending = [0] * MAP_SIZE
score_map_ascending = [0] * MAP_SIZE


def tile_score(rank: int) -> int:
    """Return the score contribution for a tile of the given rank."""
    return rank << rank


def build_score_map() -> None:
    """Populate score lookup tables mirroring ``Node::BuildScoreMap``.

    The outer loop enumerates every possible 4-tile configuration encoded as
    five-bit little-endian ranks.  We re-encode the configuration into the
    big-endian format used by ``Board::Row``/``Board::Col`` when storing the
    descending scores so Python observes the same aliasing as the original C++
    code.
    """
    for i in range(MAP_SIZE):
        # Decode the line into 4 ranks using little-endian 5-bit chunks.
        line = [(i >> (j * 5)) & 0x1F for j in range(N)]

        # Compute score for the line (same TileScore arithmetic as the engine).
        score = tile_score(line[0])
        for x in range(N - 1):
            a = tile_score(line[x])
            b = tile_score(line[x + 1])
            if a >= b:
                score += a + b
            else:
                score += (a - b) * 12
            if a == b:
                score += a

        # Map the little-endian index to the big-endian index used on lookup.
        key_desc = 0
        for j in range(N):
            key_desc = key_desc * 32 + line[j]

        score_map_descending[key_desc] = score
        score_map_ascending[i] = score


def _is_scalar(value: object) -> bool:
    return isinstance(value, numbers.Integral)


def _normalize_board(board) -> list[list[int]]:
    """Coerce a board into a 4×4 matrix of ranks.

    The engine serialises boards as flat, row-major sequences of ranks
    (``board"[...]`` in ``2048.cc``) whereas most callers in Python prefer a
    2-D list.  This helper accepts either representation (including NumPy
    arrays) and returns a freshly materialised ``list`` of rows.
    """
    if hasattr(board, "tolist") and not isinstance(board, list):  # NumPy etc.
        board = board.tolist()  # type: ignore[assignment]

    if isinstance(board, list):
        data = board
    else:
        try:
            data = list(board)
        except TypeError as exc:  # pragma: no cover - defensive
            raise TypeError("Board must be a sequence of ranks") from exc

    if len(data) == N and all(
        hasattr(row, "__len__") and not _is_scalar(row) for row in data
    ):
        matrix: list[list[int]] = []
        for row in data:
            row_data = row.tolist() if hasattr(row, "tolist") else list(row)
            if len(row_data) != N:
                raise ValueError("Expected rows of length 4")
            matrix.append([int(value) for value in row_data])
        return matrix

    if len(data) == N * N and all(_is_scalar(value) for value in data):
        flat = [int(value) for value in data]
        return [flat[i * N : (i + 1) * N] for i in range(N)]

    raise ValueError("Board must be a 4x4 grid or a row-major sequence of 16 ranks")


def _encode_rows_from_matrix(matrix) -> list[int]:
    keys = []
    for y in range(N):
        key = 0
        for x in range(N):
            key = key * 32 + int(matrix[y][x])
        keys.append(key)
    return keys


def _encode_cols_from_matrix(matrix) -> list[int]:
    keys = []
    for x in range(N):
        key = 0
        for y in range(N):
            key = key * 32 + int(matrix[y][x])
        keys.append(key)
    return keys


def encode_rows(board) -> list[int]:
    """Encode each row into a base-32 integer (matches ``Board::Row``).

    The input can be a 4×4 matrix or a flat row-major sequence of ranks, the
    same shape emitted in the engine's ``.jsonl`` logs.
    """
    matrix = _normalize_board(board)
    return _encode_rows_from_matrix(matrix)


def encode_cols(board) -> list[int]:
    """Encode each column into a base-32 integer (matches ``Board::Col``)."""
    matrix = _normalize_board(board)
    return _encode_cols_from_matrix(matrix)


def evaluate(board, options: dict) -> int:
    """Evaluate a board via the heuristic used by ``Node::Evaluate``.

    Parameters
    ----------
    board:
        Either a 4×4 matrix of ranks or a flat row-major sequence of 16 ranks.
    options:
        Mapping with the boolean ``interactive`` flag (defaults to ``False``).
    """
    matrix = _normalize_board(board)
    interactive = bool(options.get("interactive", False))

    row_keys = _encode_rows_from_matrix(matrix)
    col_keys = _encode_cols_from_matrix(matrix)

    if interactive:
        score_left = sum(score_map_descending[k] for k in row_keys)
        score_right = sum(score_map_ascending[k] for k in row_keys)
        score_up = sum(score_map_descending[k] for k in col_keys)
        score_down = sum(score_map_ascending[k] for k in col_keys)
        return int(max(score_left, score_right) + max(score_up, score_down))

    score = sum(score_map_descending[k] for k in row_keys)
    score += sum(score_map_descending[k] for k in col_keys)
    return int(score)


# Build tables once at import time, matching the C++ static initialiser.
build_score_map()
