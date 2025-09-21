from __future__ import annotations

"""
Policy argmax agreement check (offline, no server).

Validates that argmax sampling from a checkpoint agrees with recorded moves in
steps.npy. Supports both action_policy (UDLR) and binned_ev (URDL) heads.

Outputs a concise summary and optionally writes a small JSONL of disagreements.

Run:
  uv run python benchmarks/policy_argmax_agreement.py \
    --init inits/v2/10m_hard_target \
    --dataset datasets/macroxue/d6_1b \
    --num 20000 --device cuda --out disagreements.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from core_2048 import load_encoder_from_init
from train_2048.dataloader.steps import StepsDataset


def _unpack_board_to_exps_u8(packed: np.ndarray, *, mask65536: Optional[np.ndarray] = None) -> np.ndarray:
    arr = packed.astype(np.uint64, copy=False)
    n = int(arr.shape[0])
    out = np.empty((n, 16), dtype=np.uint8)
    for i in range(16):
        out[:, i] = ((arr >> (4 * i)) & np.uint64(0xF)).astype(np.uint8, copy=False)
    if mask65536 is not None:
        m = mask65536.astype(np.uint16, copy=False)
        for i in range(16):
            sel = ((m >> i) & np.uint16(1)) != 0
            if np.any(sel):
                out[sel, i] = np.uint8(16)
    return out


def urdl_bits(bits: np.ndarray) -> np.ndarray:
    # Up=1, Right=2, Down=4, Left=8
    return np.stack([
        (bits & 1) != 0,
        (bits & 2) != 0,
        (bits & 4) != 0,
        (bits & 8) != 0,
    ], axis=1)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", required=True, help="Init dir with config.json and weights")
    ap.add_argument("--dataset", required=True, help="Dataset dir containing steps.npy and metadata.db")
    ap.add_argument("--num", type=int, default=20000, help="Number of random samples to check (default: %(default)s)")
    ap.add_argument("--batch", type=int, default=4096, help="Chunk size for streaming eval (default: %(default)s)")
    ap.add_argument("--device", type=str, default=None, help="Device: cuda|cpu (default: auto)")
    ap.add_argument("--out", type=str, default=None, help="Optional JSONL file to write first K disagreements")
    ap.add_argument("--max-write", type=int, default=500, help="Max disagreements to write (default: %(default)s)")
    args = ap.parse_args(argv)

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading model: {args.init}")
    model = load_encoder_from_init(args.init).to(dev)
    model.eval()

    # Detect head type
    head_type = getattr(getattr(model, "config", None), "head_type", "binned_ev")
    print(f"Detected head_type={head_type}")

    ds = StepsDataset(args.dataset, mmap_mode=True)
    rng = np.random.default_rng(123)
    total = len(ds)
    n = min(int(args.num), total)
    idxs = rng.integers(0, total, size=n, dtype=np.int64)

    # Streaming evaluation in chunks to avoid large host allocations
    write_out = args.out is not None
    out_fp = None
    written = 0
    if write_out:
        out_fp = Path(args.out).open("w", encoding="utf-8")

    agree_cnt = 0
    illegal_cnt = 0
    conf = np.zeros((4, 4), dtype=np.int64)

    B = max(1, int(args.batch))
    inv_perm_cpu = np.array([0, 2, 3, 1], dtype=np.int64)  # UDLR->URDL mapping on CPU

    for off in range(0, n, B):
        sel = idxs[off : off + B]
        batch = ds.get_rows(sel)

        mask65536 = batch['tile_65536_mask'] if 'tile_65536_mask' in batch.dtype.names else None
        exps = _unpack_board_to_exps_u8(batch['board'], mask65536=mask65536)
        tokens = torch.from_numpy(exps).to(device=dev, dtype=torch.long)

        if 'move_dir' not in batch.dtype.names:
            raise KeyError("steps.npy missing move_dir field; cannot compare against oracle moves")
        oracle_move_urdl = batch['move_dir'].astype(np.int64, copy=False)
        legal_bits = batch['ev_legal'].astype(np.uint8, copy=False)
        legal_urdl = urdl_bits(legal_bits)

        with torch.inference_mode():
            _, head_out = model(tokens)

        if isinstance(head_out, (list, tuple)):
            head_probs = [F.softmax(h.float(), dim=-1) for h in head_out]
            p1 = torch.stack([hp[:, -1] for hp in head_probs], dim=1)  # (b,4)
            pred_urdl = torch.argmax(p1, dim=1).cpu().numpy().astype(np.int64)
            pred_scores = p1.gather(1, torch.tensor(pred_urdl, device=p1.device).view(-1, 1)).squeeze(1).cpu().numpy()
        else:
            probs = F.softmax(head_out.float(), dim=-1)  # (b,4) UDLR
            pred_udlr = torch.argmax(probs, dim=1).cpu().numpy().astype(np.int64)
            pred_urdl = inv_perm_cpu[pred_udlr]
            pred_scores = probs.gather(1, torch.from_numpy(pred_udlr).to(probs.device).view(-1, 1)).squeeze(1).cpu().numpy()

        agree = (pred_urdl == oracle_move_urdl)
        illegal_pred = ~legal_urdl[np.arange(pred_urdl.shape[0]), pred_urdl]
        agree_cnt += int(agree.sum())
        illegal_cnt += int(illegal_pred.sum())
        # Confusion
        for t, p in zip(oracle_move_urdl.tolist(), pred_urdl.tolist()):
            conf[int(t), int(p)] += 1

        if write_out and written < int(args.max_write):
            remaining = int(args.max_write) - written
            for i in range(min(remaining, agree.shape[0])):
                if agree[i]:
                    continue
                rec = {
                    "index": int(sel[i]),
                    "oracle_move_urdl": int(oracle_move_urdl[i]),
                    "pred_move_urdl": int(pred_urdl[i]),
                    "pred_score": float(pred_scores[i]),
                    "legal_urdl": [bool(x) for x in legal_urdl[i].tolist()],
                }
                out_fp.write(json.dumps(rec) + "\n")
                written += 1
                if written >= int(args.max_write):
                    break

    if out_fp is not None:
        out_fp.close()

    acc = float(agree_cnt) / float(n) if n > 0 else 0.0
    il_frac = float(illegal_cnt) / float(n) if n > 0 else 0.0
    print(f"Samples: {n}")
    print(f"Agreement with oracle (argmax): {acc:.4f}")
    print(f"Predicted illegal by ev_legal: {il_frac:.4f}")
    moves = ["Up", "Right", "Down", "Left"]
    print("Confusion (rows=oracle URDL, cols=pred URDL):")
    for i in range(4):
        row = ", ".join(f"{conf[i,j]}" for j in range(4))
        print(f"  {moves[i]:>5}: {row}")

    if args.out:
        print(f"Wrote up to {args.max_write} disagreements to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
