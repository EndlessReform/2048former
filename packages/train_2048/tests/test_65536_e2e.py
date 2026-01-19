import json
from pathlib import Path

import numpy as np
import torch

from train_2048.dataloader.collate import make_collate_steps
from train_2048.modeling.te_encoder import TEEncoder
from core_2048 import EncoderConfig


def _pack_exps_to_u64(exps: np.ndarray) -> np.uint64:
    packed = np.uint64(0)
    for i, value in enumerate(exps):
        shift = (15 - i) * 4
        packed |= np.uint64(int(value) & 0xF) << np.uint64(shift)
    return packed


class _DummyDataset:
    def __init__(self, rows: np.ndarray) -> None:
        self._rows = rows

    def get_rows(self, idxs):
        return self._rows[idxs]


def test_dataloader_model_sees_65536_token():
    # Build a single-row pseudo dataset with a 65536 tile at cell 0.
    # Packed nibble uses 0 there; mask marks it as 65536.
    exps = np.zeros(16, dtype=np.uint8)
    packed = _pack_exps_to_u64(exps)
    mask65536 = np.uint16(1 << 0)

    dtype = np.dtype(
        [
            ("board", "<u8"),
            ("tile_65536_mask", "<u2"),
            ("branch_evs", "<f4", (4,)),
            ("ev_legal", "<u1"),
            ("move_dir", "<i1"),
        ]
    )
    rows = np.zeros(1, dtype=dtype)
    rows[0]["board"] = packed
    rows[0]["tile_65536_mask"] = mask65536
    rows[0]["branch_evs"] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    rows[0]["ev_legal"] = np.uint8(0b1111)
    rows[0]["move_dir"] = np.int8(0)

    ds = _DummyDataset(rows)
    collate = make_collate_steps("hard_move", ds)
    batch = collate([0])

    tokens = batch["tokens"]
    assert tokens.shape == (1, 16)
    assert int(tokens[0, 0].item()) == 16

    # Ensure model embedding can index token 16.
    cfg_path = Path("checkpoints/20260118_134032/config.json")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = EncoderConfig.model_validate(json.load(f))
    model = TEEncoder(cfg)

    assert model.tok_emb.num_embeddings >= 17
    emb = model.tok_emb(tokens)
    assert torch.allclose(emb[0, 0], model.tok_emb.weight[16])
