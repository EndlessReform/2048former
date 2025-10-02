# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///
import argparse
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


TILE_VALUES: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384, 32768)


@dataclass(slots=True)
class RunInfo:
    max_score: int
    highest_tile: int
    steps: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the visualization script."""
    parser = argparse.ArgumentParser(description="Visualize self-play runs")
    parser.add_argument(
        "--input-dataset",
        "-i",
        type=Path,
        required=True,
        help="Path to the dataset directory or metadata.db file",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory where visualizations will be written",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins when aggregating many distinct scores",
    )
    return parser.parse_args()


def resolve_metadata_path(dataset_path: Path) -> Path:
    """Return the metadata.db path for the given dataset location."""
    if dataset_path.is_file():
        if dataset_path.name != "metadata.db" and dataset_path.suffix != ".db":
            raise ValueError(f"Expected a metadata.db file, got {dataset_path}")
        return dataset_path

    metadata_path = dataset_path / "metadata.db"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.db not found in {dataset_path}")
    return metadata_path


def load_run_metadata(metadata_path: Path) -> Dict[int, RunInfo]:
    """Load run metadata keyed by run_id."""
    with sqlite3.connect(str(metadata_path)) as conn:
        cursor = conn.execute("SELECT id, max_score, highest_tile, steps FROM runs")
        rows = cursor.fetchall()

    runs: Dict[int, RunInfo] = {}
    for row in rows:
        if not row:
            continue
        run_id, max_score, highest_tile, steps = row
        if max_score is None or steps is None:
            continue
        runs[int(run_id)] = RunInfo(
            max_score=int(max_score),
            highest_tile=int(highest_tile or 0),
            steps=int(steps),
        )
    return runs


def plot_max_score_histogram(
    max_scores: Sequence[int], output_path: Path, dataset_label: str, bins: int
) -> tuple[plt.Figure, plt.Axes]:
    """Plot and write the max score histogram for the dataset."""
    if not max_scores:
        raise ValueError("No max_score values found in dataset")

    counter = Counter(max_scores)
    unique_scores = sorted(counter)
    fig_width = min(max(6.0, 0.35 * len(unique_scores)), 18.0)

    fig, ax = plt.subplots(figsize=(fig_width, 4.5))

    if len(unique_scores) <= 60:
        positions = range(len(unique_scores))
        counts = [counter[score] for score in unique_scores]
        ax.bar(positions, counts, width=0.8)
        ax.set_xticks(list(positions))
        ax.set_xticklabels([str(score) for score in unique_scores], rotation=45, ha="right")
    else:
        ax.hist(max_scores, bins=max(bins, 1), edgecolor="black")

    ax.set_xlabel("Max score")
    ax.set_ylabel("Run count")
    ax.set_title(f"Max score distribution ({dataset_label})")
    return fig, ax


def plot_run_length_histogram(
    run_lengths: Sequence[int], dataset_label: str, bins: int
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the run length distribution."""

    if not run_lengths:
        raise ValueError("No run length values found in dataset")

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.hist(run_lengths, bins=max(bins, 1), edgecolor="black")
    ax.set_xlabel("Steps in run")
    ax.set_ylabel("Run count")
    ax.set_title(f"Run length distribution ({dataset_label})")
    return fig, ax


def sanitize_label(path: Path) -> str:
    """Return a label usable in filenames for the dataset path."""
    stem = path.name if path.is_dir() else path.stem
    safe = stem.replace(" ", "_")
    return safe or "dataset"


def dataset_root_for_steps(input_path: Path) -> Path:
    """Return the directory that should contain step shards."""
    return input_path if input_path.is_dir() else input_path.parent


def find_step_files(dataset_dir: Path) -> list[Path]:
    """Return sorted step shard paths for the dataset."""
    direct = dataset_dir / "steps.npy"
    if direct.exists():
        return [direct]
    return sorted(dataset_dir.glob("steps-*.npy"))


def compute_tile_hits(
    dataset_dir: Path,
    run_metadata: Mapping[int, RunInfo],
    tile_values: Sequence[int],
    chunk_size: int = 200_000,
) -> Dict[int, Dict[int, int]]:
    """Return earliest step index each run reaches the requested tiles."""

    step_files = find_step_files(dataset_dir)
    if not step_files:
        return {}

    tile_to_exp = {tile: int(math.log2(tile)) for tile in tile_values}

    pending: Dict[int, set[int]] = {
        tile: {run_id for run_id, info in run_metadata.items() if info.highest_tile >= tile}
        for tile in tile_values
    }
    hits: Dict[int, Dict[int, int]] = {tile: {} for tile in tile_values}

    # Short-circuit when no runs reach the requested tiles.
    if all(not runs for runs in pending.values()):
        return hits

    for shard_path in step_files:
        steps = np.load(shard_path, mmap_mode="r")
        total = steps.shape[0]
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk = steps[start:end]
            run_ids = chunk["run_id"]
            step_idx = chunk["step_idx"]
            exps = chunk["exps"]

            for tile in tile_values:
                remaining = pending[tile]
                if not remaining:
                    continue

                exp = tile_to_exp[tile]
                run_mask = np.isin(run_ids, list(remaining))
                if not np.any(run_mask):
                    continue

                candidate_indices = np.nonzero(run_mask)[0]
                exp_matches = (exps[candidate_indices] == exp).any(axis=1)
                if not np.any(exp_matches):
                    continue

                hit_indices = candidate_indices[exp_matches]

                newly_completed: set[int] = set()
                for idx in hit_indices:
                    run = int(run_ids[idx])
                    if run in hits[tile]:
                        continue
                    hits[tile][run] = int(step_idx[idx])
                    newly_completed.add(run)

                if newly_completed:
                    remaining.difference_update(newly_completed)

        # Early exit if all tiles resolved.
        if all(not runs for runs in pending.values()):
            break

    return hits


def attach_tile_step_overlays(
    ax_counts: plt.Axes,
    tile_hits: Mapping[int, Mapping[int, int]],
    tile_values: Sequence[int],
) -> bool:
    """Overlay tile reach step indices on the shared x-axis."""

    if not tile_hits:
        return False

    ax_tiles = ax_counts.twinx()
    colors = plt.get_cmap("plasma", len(tile_values))
    plotted = False

    for idx, tile in enumerate(tile_values):
        run_steps = tile_hits.get(tile)
        if not run_steps:
            continue

        xs = [step_index for step_index in run_steps.values()]
        if not xs:
            continue

        ax_tiles.scatter(
            xs,
            [tile] * len(xs),
            s=16,
            alpha=0.3,
            color=colors(idx),
            label=f"Reached {tile}",
            edgecolors="none",
        )
        plotted = True

    if not plotted:
        return False

    ax_tiles.set_ylabel("Tile value")
    ax_tiles.set_yticks(list(tile_values))
    ax_tiles.set_ylim(tile_values[0] * 0.9, tile_values[-1] * 1.05)
    ax_tiles.legend(loc="upper right")
    return True


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = resolve_metadata_path(args.input_dataset)
    run_metadata = load_run_metadata(metadata_path)
    if not run_metadata:
        raise ValueError("No runs found in metadata")

    max_scores = [info.max_score for info in run_metadata.values()]
    run_lengths = [info.steps for info in run_metadata.values()]

    dataset_label = sanitize_label(args.input_dataset)
    output_path = args.out_dir / f"{dataset_label}_max_score_hist.png"

    fig, ax = plot_max_score_histogram(max_scores, output_path, dataset_label, args.bins)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Wrote histogram to {output_path}")

    run_steps_output = args.out_dir / f"{dataset_label}_run_steps_hist.png"
    fig_steps, ax_steps_counts = plot_run_length_histogram(
        run_lengths, dataset_label, args.bins
    )

    overlayed = False
    try:
        dataset_root = dataset_root_for_steps(args.input_dataset)
        tile_hits = compute_tile_hits(dataset_root, run_metadata, TILE_VALUES)
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"Warning: failed to compute tile overlays ({exc})")
    else:
        overlayed = attach_tile_step_overlays(ax_steps_counts, tile_hits, TILE_VALUES)
        if overlayed:
            print("Overlayed tile hit scatter points on run length chart")

    fig_steps.tight_layout()
    fig_steps.savefig(run_steps_output)
    plt.close(fig_steps)
    print(f"Wrote histogram to {run_steps_output}")


if __name__ == "__main__":
    main()
