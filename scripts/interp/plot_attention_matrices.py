#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from safetensors.torch import safe_open

from core_2048 import load_encoder_from_init


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot attention matrices from activation dumps.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("activations.safetensors"),
        help="Activation dump path (.safetensors or .pt).",
    )
    parser.add_argument(
        "--init",
        type=str,
        default=None,
        help="Init/checkpoint path (defaults to activation metadata if present).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=1,
        help="Layer index (1-based, default: 1).",
    )
    parser.add_argument(
        "--n-boards",
        type=int,
        default=2,
        help="Number of boards to visualize (randomly sampled).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for sampling boards.",
    )
    parser.add_argument(
        "--heads",
        type=str,
        default=None,
        help="Comma-separated head indices to plot (e.g., 0,4).",
    )
    parser.add_argument(
        "--no-avg",
        action="store_true",
        help="Disable average-attention plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: out/attn_viz/<input>_layerX_<ts>).",
    )
    return parser.parse_args()


def _load_activation_bundle(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
    if path.suffix.lower() == ".safetensors":
        with safe_open(path, framework="pt") as f:
            if "attn_norm" not in f.keys() or "boards" not in f.keys():
                available = ", ".join(sorted(f.keys()))
                msg = f"Expected attn_norm + boards in {path}. Available keys: {available}"
                raise KeyError(msg)
            attn_norm = f.get_tensor("attn_norm")
            boards = f.get_tensor("boards")
            metadata = f.metadata() or {}
        return attn_norm, boards, {k: str(v) for k, v in metadata.items()}
    if path.suffix.lower() in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if "attn_norm" not in payload or "boards" not in payload:
            available = ", ".join(sorted(payload.keys()))
            msg = f"Expected attn_norm + boards in {path}. Available keys: {available}"
            raise KeyError(msg)
        metadata = payload.get("metadata", {})
        return payload["attn_norm"], payload["boards"], {k: str(v) for k, v in metadata.items()}
    raise ValueError("Activation input must be .safetensors or .pt")


def _parse_heads(raw: str | None, num_heads: int, num_kv: int, groups: int) -> list[int]:
    if raw:
        heads = [int(item) for item in raw.split(",") if item.strip() != ""]
        for h in heads:
            if h < 0 or h >= num_heads:
                raise ValueError(f"Head index {h} is out of range (0..{num_heads - 1})")
        return heads
    return list(range(num_heads))


def _attention_weights(
    attn_module,
    x: torch.Tensor,
    head_idx: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    q, k, _v = attn_module._shape_qkv(x)
    q = q.float()
    k = k.float()
    n_kv = k.shape[1]
    head_dim = q.shape[-1]
    kv_idx = head_idx % n_kv

    qh = q[:, head_idx]
    kh = k[:, kv_idx]
    qk = torch.matmul(qh, kh.transpose(-2, -1))
    if attn_module.use_attention_sinks:
        qk = qk * float(attn_module.sm_scale)
        sink = attn_module.sinks[head_idx].float().to(qk)
        sink_term = sink.view(1, 1, 1).expand(qk.size(0), qk.size(1), 1)
        qk = torch.cat([qk, sink_term], dim=-1)
        weights_full = torch.softmax(qk, dim=-1)
        sink_weights = weights_full[..., -1]
        weights = weights_full[..., :-1]
    else:
        qk = qk / (head_dim**0.5)
        weights = torch.softmax(qk, dim=-1)
        sink_weights = None
    return weights, sink_weights


def _format_board(board: np.ndarray) -> str:
    board = board.reshape(4, 4)
    rows = [" ".join(f"{int(v):2d}" for v in row) for row in board]
    return "\n".join(rows)


def _plot_attention(
    weights: np.ndarray,
    *,
    out_path: Path,
    title: str,
    board_text: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(weights, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(title, fontsize=10)
    labels = [f"r{r}c{c}" for r in range(1, 5) for c in range(1, 5)]
    ax.set_xticks(range(len(labels)), labels=labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=6)
    ax.text(
        1.02,
        0.5,
        board_text,
        transform=ax.transAxes,
        va="center",
        ha="left",
        fontsize=8,
        family="monospace",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_sink_weights(
    sink_weights: np.ndarray,
    *,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 1.6))
    im = ax.imshow(sink_weights[None, :], cmap="viridis", vmin=0.0, vmax=1.0)
    labels = [f"r{r}c{c}" for r in range(1, 5) for c in range(1, 5)]
    ax.set_xticks(range(len(labels)), labels=labels, rotation=90, fontsize=6)
    ax.set_yticks([0], labels=["sink"], fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    attn_norm, boards, metadata = _load_activation_bundle(args.input)

    if attn_norm.ndim != 4:
        raise ValueError(f"Expected attn_norm to be 4D (N,L,T,H); got {attn_norm.shape}")

    n_samples, n_layers, seq_len, hidden_size = attn_norm.shape
    layer_idx = args.layer - 1
    if layer_idx < 0 or layer_idx >= n_layers:
        raise ValueError(f"Layer must be in 1..{n_layers} (got {args.layer})")

    init_path = args.init or metadata.get("checkpoint")
    if not init_path:
        raise ValueError("Missing --init and activation metadata does not include checkpoint.")

    model = load_encoder_from_init(init_path)
    model.eval()
    attn = model.blocks[layer_idx].attn

    heads = _parse_heads(args.heads, attn.num_heads, attn.num_kv_heads, attn.groups)

    rng = np.random.default_rng(args.seed)
    n_pick = min(max(args.n_boards, 1), n_samples)
    chosen = rng.choice(n_samples, size=n_pick, replace=False)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        stem = args.input.stem
        out_dir = Path("out") / "attn_viz" / f"{stem}_layer{args.layer}_{timestamp}"
    else:
        out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        target_dtype = next(attn.parameters()).dtype
        x = attn_norm[chosen, layer_idx].to(dtype=target_dtype)
        weights_by_head: dict[int, torch.Tensor] = {}
        sink_by_head: dict[int, torch.Tensor] = {}
        for head_idx in heads:
            weights, sink_weights = _attention_weights(attn, x, head_idx)
            weights_by_head[head_idx] = weights.cpu()
            if sink_weights is not None:
                sink_by_head[head_idx] = sink_weights.cpu()
        avg_weights = None
        avg_sink = None
        if not args.no_avg:
            all_weights = []
            all_sinks = []
            for head_idx in range(attn.num_heads):
                weights, sink_weights = _attention_weights(attn, x, head_idx)
                all_weights.append(weights.cpu())
                if sink_weights is not None:
                    all_sinks.append(sink_weights.cpu())
            avg_weights = torch.stack(all_weights, dim=0).mean(dim=0)
            if all_sinks:
                avg_sink = torch.stack(all_sinks, dim=0).mean(dim=0)

    for i, sample_idx in enumerate(chosen):
        board = boards[sample_idx].cpu().numpy()
        board_text = _format_board(board)
        board_dir = out_dir / f"board_{int(sample_idx)}"
        board_dir.mkdir(parents=True, exist_ok=True)
        (board_dir / "board.txt").write_text(board_text + "\n", encoding="utf-8")

        for head_idx, weights in weights_by_head.items():
            weights_np = weights[i].numpy()
            out_path = board_dir / f"head_{head_idx:02d}.png"
            title = f"Layer {args.layer} Head {head_idx} (sample {int(sample_idx)})"
            _plot_attention(
                weights_np,
                out_path=out_path,
                title=title,
                board_text=board_text,
            )
            if head_idx in sink_by_head:
                sink_vals = sink_by_head[head_idx][i].numpy()
                sink_path = board_dir / f"head_{head_idx:02d}_sink.png"
                sink_title = (
                    f"Layer {args.layer} Head {head_idx} Sink (mean={sink_vals.mean():.3f})"
                )
                _plot_sink_weights(
                    sink_vals,
                    out_path=sink_path,
                    title=sink_title,
                )

        if avg_weights is not None:
            out_path = board_dir / "head_avg.png"
            title = f"Layer {args.layer} Avg Heads (sample {int(sample_idx)})"
            _plot_attention(
                avg_weights[i].numpy(),
                out_path=out_path,
                title=title,
                board_text=board_text,
            )
            if avg_sink is not None:
                sink_vals = avg_sink[i].numpy()
                sink_path = board_dir / "head_avg_sink.png"
                sink_title = f"Layer {args.layer} Avg Sink (mean={sink_vals.mean():.3f})"
                _plot_sink_weights(
                    sink_vals,
                    out_path=sink_path,
                    title=sink_title,
                )

    print(f"Wrote attention visualizations to {out_dir}")


if __name__ == "__main__":
    main()
