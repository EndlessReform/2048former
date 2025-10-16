from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from train_2048.tokenization.macroxue import (
    MacroxueTokenizerV2,
    MacroxueTokenizerV2Spec,
    fit_macroxue_tokenizer_v2,
)


def _macroxue_dtype() -> np.dtype:
    return np.dtype(
        [
            ("run_id", "<u4"),
            ("step_index", "<u4"),
            ("board", "<u8"),
            ("board_eval", "<i4"),
            ("tile_65536_mask", "<u2"),
            ("move_dir", "<u1"),
            ("valuation_type", "<u1"),
            ("ev_legal", "<u1"),
            ("max_rank", "<u1"),
            ("seed", "<u4"),
            ("branch_evs", "<f4", (4,)),
        ],
        align=True,
    )


def _legal_mask(bits: int) -> list[bool]:
    return [bool((bits >> i) & 1) for i in range(4)]


def test_fit_macroxue_tokenizer_v2(tmp_path):
    dataset_dir = Path(tmp_path)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dtype = _macroxue_dtype()
    rows = np.zeros(6, dtype=dtype)

    # Search rows
    rows[0]["valuation_type"] = 0
    rows[0]["move_dir"] = 0
    rows[0]["ev_legal"] = 0b1111
    rows[0]["board_eval"] = 500
    rows[0]["branch_evs"] = np.array([0.7, 0.4, 0.3, 0.5], dtype=np.float32)

    rows[1]["valuation_type"] = 0
    rows[1]["move_dir"] = 1
    rows[1]["ev_legal"] = 0b1111
    rows[1]["board_eval"] = 200
    rows[1]["branch_evs"] = np.array([0.1, 0.65, 0.25, 0.4], dtype=np.float32)

    # Tuple10 rows
    rows[2]["valuation_type"] = 2
    rows[2]["move_dir"] = 3
    rows[2]["ev_legal"] = 0b1111
    rows[2]["branch_evs"] = np.array([0.75, 0.6, 0.55, 0.9], dtype=np.float32)

    rows[3]["valuation_type"] = 2
    rows[3]["move_dir"] = 1
    rows[3]["ev_legal"] = 0b1111
    rows[3]["branch_evs"] = np.array([0.0, 0.85, 0.3, 0.2], dtype=np.float32)

    # Tuple11 rows
    rows[4]["valuation_type"] = 1
    rows[4]["move_dir"] = 2
    rows[4]["ev_legal"] = 0b1111
    rows[4]["branch_evs"] = np.array([0.55, 0.65, 0.92, 0.5], dtype=np.float32)

    rows[5]["valuation_type"] = 1
    rows[5]["move_dir"] = 0
    rows[5]["ev_legal"] = 0b1111
    rows[5]["branch_evs"] = np.array([0.8, 0.12, 0.3, 0.0], dtype=np.float32)

    np.save(dataset_dir / "steps.npy", rows)
    (dataset_dir / "valuation_types.json").write_text(
        json.dumps({"search": 0, "tuple11": 1, "tuple10": 2})
    )

    spec = fit_macroxue_tokenizer_v2(
        dataset_dir,
        num_bins=3,
        search_failure_cutoff=-1000,
        zero_tolerance=1e-6,
        show_progress=False,
    )

    assert isinstance(spec, MacroxueTokenizerV2Spec)
    assert spec.tokenizer_type == "macroxue_ev_advantage_v2"
    assert spec.num_bins == 3
    assert spec.search.failure_cutoff == -1000
    assert len(spec.search.bin_edges) == spec.num_bins + 1
    assert len(spec.tuple10.bin_edges) == spec.num_bins + 1
    assert len(spec.tuple11.bin_edges) == spec.num_bins + 1
    assert max(spec.search.bin_edges[:-1]) <= 0
    assert spec.search.bin_edges[-1] == 0.0
    assert np.all(np.diff(spec.search.bin_edges) > 0)

    tokenizer = MacroxueTokenizerV2(spec)

    search_tokens = tokenizer.encode_row(
        "search",
        rows[0]["branch_evs"],
        int(rows[0]["move_dir"]),
        _legal_mask(int(rows[0]["ev_legal"])),
        board_eval=int(rows[0]["board_eval"]),
    )
    assert search_tokens.shape == (4,)
    assert search_tokens[int(rows[0]["move_dir"])] == tokenizer.token_winner
    assert search_tokens.min() >= 0
    assert search_tokens.max() <= tokenizer.token_winner

    tuple_tokens = tokenizer.encode_row(
        "tuple10",
        rows[3]["branch_evs"],
        int(rows[3]["move_dir"]),
        _legal_mask(int(rows[3]["ev_legal"])),
    )
    assert tuple_tokens[int(rows[3]["move_dir"])] == tokenizer.token_winner
    # Branch with zero probability should map to FAILURE (token index 1)
    assert tuple_tokens[0] == tokenizer.token_failure
