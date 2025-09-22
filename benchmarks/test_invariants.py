from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from train_2048.tokenization.base import (
    remap_labels_urdl_to_udlr,
    reorder_cols_urdl_to_udlr,
    BoardCodec,
)
from train_2048.dataloader.steps import make_collate_steps, make_collate_macroxue


class _DummyDS:
    def __init__(self, rows: np.ndarray, dataset_dir: str = ".") -> None:
        self._rows = rows
        self.dataset_dir = dataset_dir

    def get_rows(self, idxs: np.ndarray) -> np.ndarray:
        return self._rows[idxs]


def _mk_rows_basic(n: int = 4, *, include_move: bool = True) -> np.ndarray:
    dt: list[tuple[str, Any]] = [
        ("board", np.uint64),
        ("tile_65536_mask", np.uint16),
        ("branch_evs", (np.float32, (4,))),  # URDL
        ("ev_legal", np.uint8),  # URDL bits
    ]
    if include_move:
        dt.append(("move_dir", np.int64))  # URDL numeric 0..3
    rows = np.zeros((n,), dtype=np.dtype(dt))
    # Simple boards: pack 16 zero exponents -> 0
    rows["board"] = 0
    rows["tile_65536_mask"] = 0
    # Construct EVs and legality in URDL order
    # Example: Up better than others to test reordering robustness
    urdl = np.array([0.9, 0.2, 0.3, 0.4], dtype=np.float32)
    rows["branch_evs"] = urdl
    # Legal bits URDL: all legal (0b1111 = 15)
    rows["ev_legal"] = np.uint8(0xF)
    if include_move:
        # move_dir URDL e.g. Right=1
        rows["move_dir"] = 1
    return rows


def test_reorder_helpers() -> None:
    labels = np.array([0, 1, 2, 3], dtype=np.int64)  # URDL
    mapped = remap_labels_urdl_to_udlr(labels)
    assert mapped.tolist() == [0, 3, 1, 2], "URDL→UDLR class remap incorrect"

    cols = np.arange(8, dtype=np.int64).reshape(2, 4)  # URDL columns
    reordered = reorder_cols_urdl_to_udlr(cols)
    assert reordered.tolist() == [[0, 2, 3, 1], [4, 6, 7, 5]], "URDL→UDLR column reorder incorrect"


def test_board_codec() -> None:
    # Pack a 2x16 board with a single 65536 tile at position 5 for first row
    packed = np.zeros((2,), dtype=np.uint64)
    mask = np.zeros((2,), dtype=np.uint16)
    # First row: cell 5 has exponent 16 (via mask)
    mask[0] = np.uint16(1 << 5)
    exps = BoardCodec.decode_packed_board_to_exps_u8(packed, mask65536=mask)
    assert exps.shape == (2, 16)
    assert int(exps[0, 5]) == 16 and int(exps[1, 5]) == 0


def test_make_collate_steps_udlr_canonicalization() -> None:
    rows = _mk_rows_basic(n=3, include_move=True)
    ds = _DummyDS(rows)
    collate = make_collate_steps("binned_ev", ds, binner=None, ev_tokenizer=None)
    out = collate(np.array([0, 1, 2], dtype=np.int64))
    # tokens
    assert out["tokens"].shape == (3, 16)
    # branch_values/mask are UDLR; original URDL [0.9,0.2,0.3,0.4] becomes [0.9,0.3,0.4,0.2]
    evs = out["branch_values"].numpy()[0].tolist()
    mask = out["branch_mask"].numpy()[0].tolist()
    assert evs == [0.9, 0.3, 0.4, 0.2], f"Expected UDLR values, got {evs}"
    assert mask == [True, True, True, True], "Legal mask should be all True"

    # Hard-move remap: move_dir=Right(1 URDL) -> class 3 in UDLR
    collate_h = make_collate_steps("hard_move", ds, binner=None, ev_tokenizer=None)
    out_h = collate_h(np.array([0], dtype=np.int64))
    assert int(out_h["move_targets"][0].item()) == 3, "Expected Right→class 3 under UDLR"


def test_macroxue_collate_targets(tmp_path: Path) -> None:
    # Minimal tokenizer spec with simple edges and one valuation type
    spec = {
        "version": 1,
        "quantile_count": 5,
        "actions": ["up", "left", "right", "down"],
        "valuation_types": ["top_score"],
        "ecdf_knots": {"top_score": [0.0, 0.5, 1.0]},
        "delta_edges": [0.0, 0.25, 0.5, 0.75, 1.0],
        "percentile_grid": "uniform_0_1",
    }
    spec_path = tmp_path / "tok.json"
    spec_path.write_text(json.dumps(spec))

    rows_dt = np.dtype([
        ("board", np.uint64),
        ("tile_65536_mask", np.uint16),
        ("branch_evs", (np.float32, (4,))),
        ("ev_legal", np.uint8),
        ("valuation_type", np.int64),
    ])
    rows = np.zeros((2,), dtype=rows_dt)
    rows["board"] = 0
    rows["tile_65536_mask"] = 0
    rows["valuation_type"] = 0  # top_score
    # URDL EVs -> Up best, others smaller
    rows["branch_evs"][0] = np.array([1.0, 0.0, 0.5, 0.25], dtype=np.float32)
    rows["ev_legal"][0] = np.uint8(0xF)
    # One illegal branch: Right -> bit 1<<1 zeroed
    rows["branch_evs"][1] = np.array([0.6, 0.4, 0.3, 0.2], dtype=np.float32)
    rows["ev_legal"][1] = np.uint8(0xD)  # 1101b = U,D,L legal; Right illegal

    ds = _DummyDS(rows, dataset_dir=str(tmp_path))
    collate = make_collate_macroxue(ds, str(spec_path))
    out = collate(np.array([0, 1], dtype=np.int64))
    targets = out["targets"].numpy()
    n_classes = int(out["n_classes"])  # bins + 2
    winner = n_classes - 1
    illegal = n_classes - 2

    # Row 0: Up best → winner in col 0 (UDLR)
    assert int(targets[0, 0]) == winner
    # Row 1: Right illegal → illegal in col 3 (UDLR)
    assert int(targets[1, 3]) == illegal


def test_server_token_staging_parity() -> None:
    # Given model tokens (exponents U8) the server should see the same after packing into bytes
    B = 3
    tok = torch.randint(0, 17, (B, 16), dtype=torch.int64)
    boards_bytes = bytes(tok.to(torch.uint8).cpu().numpy().tobytes())
    np_view = np.frombuffer(boards_bytes, dtype=np.uint8).reshape(B, 16)
    # Simulate server pinned host buffer
    cpu_buf = torch.empty((B, 16), dtype=torch.uint8)
    cpu_buf.numpy()[...] = np_view
    server_tokens = cpu_buf.to(dtype=torch.long)
    assert torch.equal(server_tokens, tok.to(dtype=torch.long)), "Server staging must preserve token values"


if __name__ == "__main__":
    # Run all tests when executed directly
    test_reorder_helpers()
    test_board_codec()
    test_make_collate_steps_udlr_canonicalization()
    test_server_token_staging_parity()
    # Macroxue test uses temp path; emulate a small tmp dir
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_macroxue_collate_targets(Path(td))
    print("All invariant tests passed.")

