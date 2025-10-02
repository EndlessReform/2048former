#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
import torch
from safetensors.torch import load_file as safe_load_file
from torch.utils.data import DataLoader

from train_2048.config import ValueHeadConfig, load_config
from train_2048.dataloader.steps import (
    StepsDataset,
    make_collate_value,
    _indices_from_run_ids,
)
from train_2048.objectives.value_head import ValueOrdinal
from train_2048.config import normalize_state_dict_keys
from core_2048 import load_encoder_from_init


@dataclass
class EvalBatch:
    probs: np.ndarray
    targets: np.ndarray
    progress: np.ndarray
    run_ids: np.ndarray


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.uint8, copy=False)
    y_score = y_score.astype(np.float64, copy=False)
    pos = int(y_true.sum())
    neg = y_true.size - pos
    if pos == 0 or neg == 0:
        return float('nan')
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.arange(1, y_true.size + 1, dtype=np.float64)
    pos_rank_sum = ranks[y_true[order] == 1].sum()
    auroc = (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auroc)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int) -> float:
    if y_true.size == 0:
        return float('nan')
    y_prob = np.clip(y_prob.astype(np.float64, copy=False), 0.0, 1.0)
    y_true = y_true.astype(np.float64, copy=False)
    # Bin edges excluding 0 and 1 for digitize
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize expects interior edges
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=False)
    bin_totals = np.bincount(bin_ids, minlength=n_bins)
    if not bin_totals.any():
        return float('nan')
    prob_sums = np.bincount(bin_ids, weights=y_prob, minlength=n_bins)
    true_sums = np.bincount(bin_ids, weights=y_true, minlength=n_bins)
    valid = bin_totals > 0
    accuracies = np.zeros_like(prob_sums)
    confidences = np.zeros_like(prob_sums)
    accuracies[valid] = true_sums[valid] / bin_totals[valid]
    confidences[valid] = prob_sums[valid] / bin_totals[valid]
    gap = np.abs(accuracies - confidences)
    ece = float((gap[valid] * bin_totals[valid]).sum() / y_true.size)
    return ece


def compute_stage_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    stage_mask: np.ndarray,
    thresholds: Sequence[int],
    *,
    stage_name: str,
    n_bins: int,
) -> list[dict[str, float | int | str]]:
    if not stage_mask.any():
        return []
    stage_probs = probs[stage_mask]
    stage_targets = targets[stage_mask]
    rows: list[dict[str, float | int | str]] = []
    sample_count = stage_probs.shape[0]
    positives = stage_targets.sum(axis=0)
    for idx, tile in enumerate(thresholds):
        y_true = stage_targets[:, idx]
        y_prob = stage_probs[:, idx]
        brier = float(np.mean((y_prob - y_true) ** 2)) if y_true.size else float('nan')
        auroc = binary_auroc(y_true, y_prob)
        ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
        rows.append(
            {
                "stage": stage_name,
                "threshold": int(tile),
                "samples": int(sample_count),
                "positives": int(positives[idx]),
                "prevalence": float(y_true.mean()) if y_true.size else float('nan'),
                "auroc": auroc,
                "brier": brier,
                "ece": ece,
            }
        )
    return rows


def print_table_plain(records: Sequence[dict[str, float | int | str]]) -> None:
    if not records:
        return
    columns = ["stage", "threshold", "samples", "positives", "prevalence", "auroc", "brier", "ece"]
    widths = []
    for col in columns:
        cell_width = max(len(col), max(len(format_cell(row.get(col))) for row in records))
        widths.append(cell_width)

    header = " ".join(col.ljust(width) for col, width in zip(columns, widths))
    print(header)
    print("-" * len(header))
    for row in records:
        line = " ".join(format_cell(row.get(col)).ljust(width) for col, width in zip(columns, widths))
        print(line)


def format_cell(value: float | int | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def stage_boundaries_to_names(bounds: Sequence[float]) -> list[str]:
    names: list[str] = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        lo_pct = int(round(lo * 100))
        hi_pct = int(round(hi * 100))
        names.append(f"{lo_pct:02d}-{hi_pct:02d}%")
    return names


def load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.suffix == ".safetensors":
        state = load_state_dict_safetensors(path)
    else:
        raw = torch.load(path, map_location="cpu")
        if isinstance(raw, dict) and "model" in raw:
            state = raw["model"]
        else:
            state = raw
    state_cpu = {k: v.detach().to("cpu") for k, v in state.items()}
    state_norm = normalize_state_dict_keys(state_cpu)
    model.load_state_dict(state_norm, strict=False)


def load_state_dict_safetensors(path: Path) -> dict[str, torch.Tensor]:
    raw = safe_load_file(str(path))
    tensors = {k: v.detach().to("cpu") for k, v in raw.items()}
    return normalize_state_dict_keys(tensors)


def fetch_run_ids(meta_path: Path, sql: str, params: Sequence[str]) -> np.ndarray:
    with sqlite3.connect(str(meta_path)) as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    if not rows:
        return np.empty((0,), dtype=np.int64)
    return np.asarray([int(r[0]) for r in rows], dtype=np.int64)


def build_dataloader(
    dataset_dir: Path,
    *,
    thresholds: Sequence[int],
    batch_size: int,
    num_workers: int,
    run_sql: str | None,
    sql_params: Sequence[str],
    mmap_mode: bool,
) -> tuple[StepsDataset, DataLoader]:
    ds = StepsDataset(str(dataset_dir), mmap_mode=mmap_mode)
    if run_sql:
        meta_path = dataset_dir / "metadata.db"
        run_ids = fetch_run_ids(meta_path, run_sql, sql_params)
        indices = _indices_from_run_ids(ds, run_ids)
        ds.indices = indices
    collate = make_collate_value("value_ordinal", ds, tile_thresholds=list(thresholds))
    loader_kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 8
    loader = DataLoader(**loader_kwargs)
    return ds, loader


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
) -> EvalBatch:
    model.eval()
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    progress: list[np.ndarray] = []
    run_ids: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            hidden, _ = model(tokens)
            board_repr = hidden.mean(dim=1)
            logits = model.value_head(board_repr)  # type: ignore[attr-defined]
            prob = torch.sigmoid(logits).float().cpu().numpy()
            probs.append(prob)
            targets.append(batch["value_targets_bce"].cpu().numpy())
            progress.append(batch["step_fraction"].cpu().numpy())
            run_ids.append(batch["run_id"].cpu().numpy())

    if not probs:
        return EvalBatch(
            probs=np.zeros((0, 0), dtype=np.float32),
            targets=np.zeros((0, 0), dtype=np.float32),
            progress=np.zeros((0,), dtype=np.float32),
            run_ids=np.zeros((0,), dtype=np.int64),
        )

    probs_arr = np.concatenate(probs, axis=0)
    targets_arr = np.concatenate(targets, axis=0)
    progress_arr = np.concatenate(progress, axis=0)
    run_ids_arr = np.concatenate(run_ids, axis=0)
    return EvalBatch(probs=probs_arr, targets=targets_arr, progress=progress_arr, run_ids=run_ids_arr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate value-head metrics on a holdout dataset")
    parser.add_argument("--config", type=str, help="Training TOML config used for the value head", required=False)
    parser.add_argument("--init", type=str, required=True, help="Path to init directory used for loading the base encoder")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained value-head checkpoint (.safetensors or .pt)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset directory with steps.npy/metadata.db")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mmap", action="store_true", help="Enable numpy.memmap when loading step shards")
    parser.add_argument("--run-sql", type=str, default=None, help="Optional SQL to select run IDs for evaluation")
    parser.add_argument("--sql-param", action="append", default=[], help="Parameter to bind in --run-sql (may repeat)")
    parser.add_argument("--threshold", action="append", type=int, default=None, help="Override tile thresholds (may repeat)")
    parser.add_argument("--stage-boundaries", type=float, nargs="+", default=[0.0, 0.25, 0.75, 1.0])
    parser.add_argument("--no-stage", action="store_true", help="Disable stage-wise breakdown (report overall only)")
    parser.add_argument("--ece-bins", type=int, default=20, help="Number of bins for Expected Calibration Error")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu / cuda / cuda:0 / mps)")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write metrics as JSON")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional path to write metrics as CSV")
    return parser.parse_args()


def resolve_thresholds(cfg: ValueHeadConfig, override: Iterable[int] | None) -> list[int]:
    if override is None:
        return list(cfg.tile_thresholds)
    thresholds = sorted(set(int(t) for t in override))
    if not thresholds:
        raise ValueError("At least one --threshold must be provided when overriding")
    return thresholds


def main() -> None:
    args = parse_args()

    config = None
    if args.config:
        config = load_config(args.config)
        if config.target.mode != "value_ordinal":
            raise ValueError(f"Config target.mode must be 'value_ordinal' (got {config.target.mode})")
        value_cfg = config.value_head
    else:
        value_cfg = ValueHeadConfig()

    thresholds = resolve_thresholds(value_cfg, args.threshold)
    value_cfg = value_cfg.model_copy(update={"tile_thresholds": thresholds})

    cfg_ns = SimpleNamespace(value_head=value_cfg)

    ckpt_path = Path(args.checkpoint)
    init_dir = Path(args.init)
    dataset_dir = Path(args.dataset)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    model = load_encoder_from_init(str(init_dir))
    objective = ValueOrdinal()
    model = objective.prepare_model(model, device, cfg=cfg_ns)
    load_checkpoint(model, ckpt_path)
    model.to(device)

    _ds, loader = build_dataloader(
        dataset_dir,
        thresholds=thresholds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        run_sql=args.run_sql,
        sql_params=args.sql_param,
        mmap_mode=args.mmap,
    )

    batch = evaluate_model(model, loader, device=device)
    if batch.probs.size == 0:
        print("No samples available for evaluation.")
        return

    if not args.no_stage:
        bounds = np.asarray(args.stage_boundaries, dtype=np.float64)
        if bounds.ndim != 1 or bounds.size < 2:
            raise ValueError("stage-boundaries must contain at least two values")
        if not math.isclose(bounds[0], 0.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("stage-boundaries must start at 0.0")
        if not math.isclose(bounds[-1], 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("stage-boundaries must end at 1.0")
        if np.any(np.diff(bounds) < 0):
            raise ValueError("stage-boundaries must be non-decreasing")
        stage_ids = np.digitize(batch.progress, bounds[1:-1], right=False)
        stage_names = stage_boundaries_to_names(bounds)
        stage_map = [("all", np.ones(batch.progress.shape[0], dtype=bool))]
        for idx, name in enumerate(stage_names):
            mask = stage_ids == idx
            stage_map.append((name, mask))
    else:
        stage_map = [("all", np.ones(batch.progress.shape[0], dtype=bool))]

    rows: list[dict[str, float | int | str]] = []
    for stage_name, mask in stage_map:
        rows.extend(
            compute_stage_metrics(
                batch.probs,
                batch.targets,
                mask,
                thresholds,
                stage_name=stage_name,
                n_bins=args.ece_bins,
            )
        )

    rows.sort(key=lambda r: (str(r["stage"]), int(r["threshold"])))

    if pd is not None:
        df = pd.DataFrame(rows)
        if df.empty:
            print("No metrics computed (check stage masks and data availability).")
            return
        df.sort_values(["stage", "threshold"], inplace=True)
        pd.options.display.float_format = "{:.4f}".format
        print("\nPer-threshold metrics:")
        print(df.to_string(index=False))
        records = df.to_dict(orient="records")
    else:
        if not rows:
            print("No metrics computed (check stage masks and data availability).")
            return
        print("\nPer-threshold metrics:")
        print_table_plain(rows)
        records = rows

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        if pd is not None:
            pd.DataFrame(records).to_csv(args.output_csv, index=False)
        else:
            import csv

            with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["stage", "threshold", "samples", "positives", "prevalence", "auroc", "brier", "ece"])
                writer.writeheader()
                writer.writerows(records)


if __name__ == "__main__":
    main()
