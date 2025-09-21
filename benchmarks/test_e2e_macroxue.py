from __future__ import annotations

"""
End-to-end sanity for Macroxue token pipeline without training.

Covers: synthetic pack -> steps.npy -> collate (percentiles, legality, winner/illegal) ->
model inference softmax over heads -> argmax over winner bin selects correct move.

Run: uv run python benchmarks/test_e2e_macroxue.py
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from train_2048.dataloader.steps import StepsDataset, make_collate_macroxue
from train_2048.tokenization.macroxue import MacroxueTokenizerSpec
from core_2048.model import Encoder, EncoderConfig
from core_2048.infer import forward_distributions


def _pack_board_from_exponents(exps: np.ndarray) -> tuple[np.uint64, np.uint16]:
    """Pack 16 4-bit exponents (uint8 0..15) into a u64, matching dataset packer."""
    assert exps.shape == (16,)
    acc = np.uint64(0)
    mask = np.uint16(0)
    for i, v in enumerate(exps.tolist()):
        vi = int(v)
        if vi >= 16:
            mask |= np.uint16(1 << i)
            vi = 15
        acc |= (np.uint64(vi & 0xF) << np.uint64(4 * i))
    return acc, mask


def test_e2e_macroxue(tmpdir: Path) -> None:
    # 1) Build a tiny synthetic dataset folder with steps.npy (1 row)
    ds_dir = tmpdir / "dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic board exponents 0..15
    exps = np.arange(16, dtype=np.uint8)
    board_u64, mask_u16 = _pack_board_from_exponents(exps)

    # URDL legality bits: enable Up and Right only (bits 1<<0, 1<<1)
    ev_legal = np.uint8(1 | 2)
    # Branch EVs aligned to URDL (floats in [0,1] to match our knots)
    branch = np.array([0.9, 0.7, 0.3, 0.2], dtype=np.float32)

    # NumPy dtype matching crates/dataset-packer StepRow::dtype
    dtype = np.dtype([
        ("run_id", "<u4"),
        ("step_index", "<u4"),
        ("board", "<u8"),
        ("tile_65536_mask", "<u2"),
        ("move_dir", "<u1"),
        ("valuation_type", "<u1"),
        ("ev_legal", "<u1"),
        ("max_rank", "<u1"),
        ("seed", "<u4"),
        ("branch_evs", ("<f4", 4)),
    ])

    row = np.zeros((), dtype=dtype)
    row["run_id"] = np.uint32(1)
    row["step_index"] = np.uint32(0)
    row["board"] = np.uint64(board_u64)
    row["tile_65536_mask"] = np.uint16(mask_u16)
    row["move_dir"] = np.uint8(0)
    row["valuation_type"] = np.uint8(0)  # spec index 0 -> "val"
    row["ev_legal"] = ev_legal
    row["max_rank"] = np.uint8(0)
    row["seed"] = np.uint32(123)
    row["branch_evs"] = branch

    steps = np.empty((1,), dtype=dtype)
    steps[0] = row
    np.save(ds_dir / "steps.npy", steps)

    # 2) Create a minimal tokenizer spec (percentile ~ identity on [0,1])
    #    and simple margin bins
    spec = MacroxueTokenizerSpec(
        version=1,
        quantile_count=5,
        actions=("up", "left", "right", "down"),  # actions are unused in collate
        valuation_types=["val"],
        ecdf_knots={"val": [0.0, 0.25, 0.5, 0.75, 1.0]},
        delta_edges=[0.0, 0.25, 0.5, 0.75, 1.0],
        notes="e2e synthetic",
    )
    tok_path = tmpdir / "tokenizer.json"
    spec.to_json(tok_path)

    # 3) Collate one batch using Macroxue path
    ds = StepsDataset(str(ds_dir), mmap_mode=False)
    collate = make_collate_macroxue(ds, str(tok_path))
    batch = collate([0])

    tokens = batch["tokens"]  # (1,16) long
    targets = batch["targets"]  # (1,4) long
    n_classes = int(batch["n_classes"])  # n_bins + 2

    assert tokens.shape == (1, 16)
    assert tokens.dtype == torch.int64
    assert targets.shape == (1, 4)
    assert n_classes == len(spec.delta_edges) - 1 + 2

    # Winner should be Up (index 0) with URDL and our EVs; Down/Left illegal
    n_bins = len(spec.delta_edges) - 1
    ILLEGAL = n_bins
    WINNER = n_bins + 1
    t = targets[0].tolist()
    assert t[0] == WINNER, f"expected winner@Up; got {t}"
    assert t[2] == ILLEGAL and t[3] == ILLEGAL, f"expected illegal@Down/Left; got {t}"

    # 4) Build a tiny model with heads sized to n_classes and bias Up head to pick WINNER
    enc_cfg = EncoderConfig(
        input_vocab_size=17,
        output_n_bins=n_classes,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        layer_norm_eps=1e-5,
        dropout_prob=0.0,
        max_position_embeddings=16,
        head_type="binned_ev",
    )
    model = Encoder(enc_cfg)
    # Zero all weights/biases for determinism
    for p in model.parameters():
        torch.nn.init.zeros_(p)
    # Bias last class (WINNER) high on Up head (0), low elsewhere
    with torch.no_grad():
        for h in range(4):
            model.ev_heads[h].bias.zero_()
        model.ev_heads[0].bias[WINNER] = 10.0  # Up head favors WINNER
        # others remain at 0 -> softmax lower p1

    head_probs = forward_distributions(model, tokens, set_eval=True)
    p1 = torch.stack([hp[:, -1] for hp in head_probs], dim=1)  # (1,4)
    choice = int(torch.argmax(p1, dim=1).item())
    assert choice == 0, f"expected argmax p1=head 0 (Up), got {choice}"

    print("OK: e2e Macroxue pipeline sanity passed")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        test_e2e_macroxue(Path(td))
