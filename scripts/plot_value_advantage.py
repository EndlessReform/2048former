#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from rich import print as rprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare value-head predictions to a base rate for reaching a target tile "
            "after first hitting a base tile (default 1024 -> 2048)."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset directory with steps.npy/values.npy.")
    parser.add_argument(
        "--target-tile",
        type=int,
        default=2048,
        help="Target tile to reach (default: 2048).",
    )
    parser.add_argument(
        "--base-tile",
        type=int,
        default=None,
        help="Base tile conditioning value (default: target_tile // 2).",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=12,
        help="Number of games to plot (default: 12).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for sampling games.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Chunk size for scanning steps (default: 1,000,000).",
    )
    parser.add_argument(
        "--output",
        default="value_advantage.png",
        help="Output plot path (default: value_advantage.png).",
    )
    parser.add_argument(
        "--advantage",
        action="store_true",
        help="Plot advantage (p_next - base_rate) instead of raw probability.",
    )
    parser.add_argument(
        "--report-knowledge",
        action="store_true",
        help="Print how early the model exceeds the baseline by the cutoff.",
    )
    parser.add_argument(
        "--knowledge-cutoff",
        type=float,
        default=0.1,
        help="Advantage threshold for 'knowing' the outcome (default: 0.1).",
    )
    parser.add_argument(
        "--last-n-steps",
        type=int,
        default=None,
        help="Restrict plot to the last N steps before outcome (e.g. 128).",
    )
    parser.add_argument(
        "--restrict-dies",
        action="store_true",
        help="Only include games that die before reaching target.",
    )
    parser.add_argument(
        "--restrict-wins",
        action="store_true",
        help="Only include games that reach the target tile.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip alignment check between steps.npy and values.npy.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Balance sampled games across reach vs die outcomes when possible.",
    )
    return parser.parse_args()


def _chunked(n_rows: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, int(n_rows), chunk_size):
        end = min(int(n_rows), start + chunk_size)
        yield start, end


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _tile_to_rank(tile: int) -> int:
    if not _is_power_of_two(tile):
        raise ValueError(f"tile must be a power of two, got {tile}")
    return int(math.log2(tile))


def _load_highest_tiles(metadata_path: Path) -> np.ndarray:
    conn = sqlite3.connect(str(metadata_path))
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(runs)")]
        if "highest_tile" in cols:
            tile_col = "highest_tile"
        elif "max_tile" in cols:
            tile_col = "max_tile"
        else:
            raise KeyError("runs table missing highest_tile/max_tile")
        rows = conn.execute(f"SELECT id, {tile_col} FROM runs").fetchall()
    finally:
        conn.close()
    if not rows:
        raise RuntimeError("metadata.db contains no runs")
    ids = np.asarray([r[0] for r in rows], dtype=np.int64)
    tiles = np.asarray([r[1] for r in rows], dtype=np.int64)
    max_id = int(ids.max())
    highest_tiles = np.full((max_id + 1,), -1, dtype=np.int64)
    highest_tiles[ids] = tiles
    return highest_tiles


def _sample_runs(
    rng: np.random.Generator,
    eligible_reach: np.ndarray,
    eligible_fail: np.ndarray,
    n_total: int,
    balanced: bool,
) -> np.ndarray:
    if n_total <= 0:
        return np.asarray([], dtype=np.int64)
    if balanced and eligible_reach.size and eligible_fail.size:
        n_reach = min(eligible_reach.size, n_total // 2)
        n_fail = min(eligible_fail.size, n_total - n_reach)
        reach_ids = rng.choice(eligible_reach, size=n_reach, replace=False)
        fail_ids = rng.choice(eligible_fail, size=n_fail, replace=False)
        return np.concatenate([reach_ids, fail_ids]).astype(np.int64, copy=False)
    eligible = eligible_reach if eligible_fail.size == 0 else np.concatenate([eligible_reach, eligible_fail])
    n_pick = min(eligible.size, n_total)
    return rng.choice(eligible, size=n_pick, replace=False).astype(np.int64, copy=False)


def _latest_knowledge_start(step_rel: np.ndarray, advantage: np.ndarray, cutoff: float) -> int | None:
    if step_rel.size == 0:
        return None
    above = advantage >= cutoff
    if not above.any():
        return None
    prev_below = np.concatenate(([True], ~above[:-1]))
    starts = np.flatnonzero(above & prev_below)
    if starts.size == 0:
        return None
    return int(step_rel[starts[-1]])


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset)
    steps_path = dataset_dir / "steps.npy"
    values_path = dataset_dir / "values.npy"
    metadata_path = dataset_dir / "metadata.db"
    if not steps_path.is_file():
        raise FileNotFoundError(f"Missing {steps_path}")
    if not values_path.is_file():
        raise FileNotFoundError(f"Missing {values_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing {metadata_path}")

    target_tile = int(args.target_tile)
    base_tile = int(args.base_tile) if args.base_tile is not None else target_tile // 2
    if not _is_power_of_two(target_tile):
        raise ValueError("target_tile must be a power of two")
    if not _is_power_of_two(base_tile):
        raise ValueError("base_tile must be a power of two")
    if base_tile >= target_tile:
        raise ValueError("base_tile must be smaller than target_tile")
    base_rank = _tile_to_rank(base_tile)
    target_rank = _tile_to_rank(target_tile)

    steps = np.load(str(steps_path), mmap_mode="r")
    values = np.load(str(values_path), mmap_mode="r")
    if steps.shape[0] != values.shape[0]:
        raise RuntimeError("steps.npy and values.npy have different lengths")
    if not args.no_validate:
        check = min(10_000, steps.shape[0])
        if not np.array_equal(steps[:check]["run_id"], values[:check]["run_id"]):
            raise RuntimeError("steps.npy and values.npy run_id mismatch")
        if not np.array_equal(steps[:check]["step_index"], values[:check]["step_index"]):
            raise RuntimeError("steps.npy and values.npy step_index mismatch")

    highest_tiles = _load_highest_tiles(metadata_path)
    eligible_reach = np.flatnonzero(highest_tiles >= target_tile)
    eligible_fail = np.flatnonzero((highest_tiles >= base_tile) & (highest_tiles < target_tile))
    rng = np.random.default_rng(args.seed)
    if args.restrict_dies and args.restrict_wins:
        raise ValueError("Choose at most one of --restrict-dies or --restrict-wins.")
    if args.restrict_dies:
        eligible_reach = np.asarray([], dtype=np.int64)
    if args.restrict_wins:
        eligible_fail = np.asarray([], dtype=np.int64)
    sample_run_ids = _sample_runs(rng, eligible_reach, eligible_fail, args.max_games, args.balanced)
    sample_set = set(int(x) for x in sample_run_ids.tolist())

    base_total = 0
    base_reach = 0
    series: dict[int, list[tuple[int, float]]] = {int(r): [] for r in sample_run_ids.tolist()}
    last_step: dict[int, int] = {int(r): -1 for r in sample_run_ids.tolist()}
    reach_step: dict[int, int | None] = {int(r): None for r in sample_run_ids.tolist()}

    for start, end in _chunked(int(steps.shape[0]), args.chunk_size):
        chunk = steps[start:end]
        chunk_vals = values[start:end]
        mask = chunk["max_rank"] == base_rank
        if mask.any():
            base_rows = chunk[mask]
            base_vals = chunk_vals[mask]
            run_ids = base_rows["run_id"].astype(np.int64, copy=False)
            valid = run_ids < highest_tiles.shape[0]
            run_ids = run_ids[valid]
            if run_ids.size != 0:
                outcomes = highest_tiles[run_ids] >= target_tile
                base_total += int(run_ids.size)
                base_reach += int(outcomes.sum())

                if sample_set:
                    sample_mask = np.isin(run_ids, sample_run_ids)
                    if sample_mask.any():
                        sample_rows = base_rows[valid][sample_mask]
                        sample_vals = base_vals[valid][sample_mask]
                        for rid, step_idx, p_next in zip(
                            sample_rows["run_id"],
                            sample_rows["step_index"],
                            sample_vals["p_next"],
                        ):
                            series[int(rid)].append((int(step_idx), float(p_next)))

        if sample_set:
            sample_mask_all = np.isin(chunk["run_id"], sample_run_ids)
            if sample_mask_all.any():
                sample_rows_all = chunk[sample_mask_all]
                for rid, step_idx, max_rank in zip(
                    sample_rows_all["run_id"],
                    sample_rows_all["step_index"],
                    sample_rows_all["max_rank"],
                ):
                    rid_i = int(rid)
                    step_i = int(step_idx)
                    prev_last = last_step.get(rid_i, -1)
                    if step_i > prev_last:
                        last_step[rid_i] = step_i
                    if int(max_rank) >= target_rank:
                        prev_reach = reach_step.get(rid_i)
                        if prev_reach is None or step_i < prev_reach:
                            reach_step[rid_i] = step_i

    if base_total == 0:
        raise RuntimeError(f"No steps found with max_rank == {base_rank} (tile {base_tile}).")
    base_rate = base_reach / base_total
    rprint(
        f"Base rate p(reach {target_tile} | {base_tile}) = {base_rate:.6f} "
        f"(p(die | {base_tile}) = {1.0 - base_rate:.6f}, steps={base_total})"
    )

    if not series:
        rprint("No sampled games found for plotting.")
        return

    # Build plots
    plotted = 0
    run_ids_sorted = sorted(series.keys())
    n_plots = sum(1 for rid in run_ids_sorted if series[rid])
    if n_plots == 0:
        rprint("No eligible steps for sampled games.")
        return
    ncols = min(4, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    axes = axes.flatten()

    knowledge_report: list[tuple[int, str, int | None]] = []
    cutoff = float(args.knowledge_cutoff)

    for rid in run_ids_sorted:
        data = series[rid]
        if not data:
            continue
        data.sort(key=lambda x: x[0])
        step_idx = np.asarray([d[0] for d in data], dtype=np.int64)
        p_next = np.asarray([d[1] for d in data], dtype=np.float32)
        outcome_step = reach_step.get(rid)
        if outcome_step is None:
            outcome_step = last_step.get(rid, step_idx.max() if step_idx.size else 0)
        step_rel = step_idx - int(outcome_step)
        outcome = "reach" if highest_tiles[rid] >= target_tile else "die"
        if outcome == "reach":
            advantage = p_next - base_rate
        else:
            advantage = base_rate - p_next
        knowledge_start = _latest_knowledge_start(step_rel, advantage, cutoff)
        if args.last_n_steps is not None:
            window_start = -int(args.last_n_steps)
            keep = step_rel >= window_start
            step_rel = step_rel[keep]
            p_next = p_next[keep]
            advantage = advantage[keep]
        if args.advantage:
            y_vals = advantage
            baseline = 0.0
        else:
            y_vals = p_next
            baseline = base_rate
        ax = axes[plotted]
        ax.plot(step_rel, y_vals, color="tab:blue", linewidth=1.0)
        ax.axhline(baseline, color="tab:orange", linestyle="--", linewidth=1.0)
        if knowledge_start is not None:
            ax.axvline(knowledge_start, color="tab:green", linestyle=":", linewidth=1.0)
        ax.set_title(f"run {rid} ({outcome})", fontsize=9)
        ax.set_xlabel("steps before outcome")
        if plotted % ncols == 0:
            if args.advantage:
                ax.set_ylabel(f"advantage vs base p(reach {target_tile})")
            else:
                ax.set_ylabel(f"p(reach {target_tile})")

        if args.report_knowledge or args.knowledge_cutoff != 0.1:
            knowledge_report.append((rid, outcome, knowledge_start))
        plotted += 1

    for ax in axes[plotted:]:
        ax.axis("off")

    fig.tight_layout()
    out_path = Path(args.output)
    fig.savefig(out_path, dpi=150)
    rprint(f"Wrote plot to {out_path}")

    if knowledge_report:
        rprint(f"Knowledge cutoff: {cutoff}")
        for rid, outcome, know_step in knowledge_report:
            if know_step is None:
                rprint(f"run {rid} ({outcome}): never exceeds cutoff")
            else:
                rprint(f"run {rid} ({outcome}): knows at {know_step} steps before outcome")


if __name__ == "__main__":
    main()
