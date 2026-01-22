#!/usr/bin/env -S uv run --script
#
# /// script
# dependencies = [
#   "duckdb>=1.0",
#   "matplotlib>=3.8",
#   "numpy>=1.26",
# ]
# ///

"""
Survivorship analysis for 2048 training/eval datasets.

Usage:
    ./survivorship_analysis.py metadata.db
    ./survivorship_analysis.py metadata.db -o ./my_outputs
    ./survivorship_analysis.py results.jsonl -o ./analysis
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import duckdb
import numpy as np
import matplotlib.pyplot as plt


def load_data(filepath: Path) -> list[dict]:
    """Load game data from SQLite db or JSONL file."""
    suffix = filepath.suffix.lower()

    if suffix == ".db":
        conn = duckdb.connect(str(filepath), read_only=True)
        rows = conn.execute(
            "SELECT seed, steps, max_score as score, highest_tile FROM runs"
        ).fetchall()
        conn.close()
        return [
            {"seed": r[0], "steps": r[1], "score": r[2], "highest_tile": r[3]}
            for r in rows
        ]
    elif suffix in (".jsonl", ".json"):
        games = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    games.append(json.loads(line))
        return games
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def analyze_survivorship(games: list[dict], output_dir: Path, name: str):
    """Generate survivorship analysis plots and stats."""

    steps = np.array([g["steps"] for g in games])
    scores = np.array([g["score"] for g in games])
    tiles = np.array([g["highest_tile"] for g in games])

    # Tile distribution
    tile_counts = defaultdict(int)
    for t in tiles:
        tile_counts[t] += 1

    # Print summary stats
    print(f"\n{'='*60}")
    print(f"SURVIVORSHIP ANALYSIS: {name}")
    print(f"{'='*60}")
    print(f"\nLoaded {len(games)} games")
    print(f"Steps: min={steps.min()}, max={steps.max()}, mean={steps.mean():.1f}, median={np.median(steps):.1f}")
    print(f"Scores: min={scores.min()}, max={scores.max()}, mean={scores.mean():.1f}")

    print("\nTile distribution:")
    for t in sorted(tile_counts.keys()):
        pct = 100 * tile_counts[t] / len(tiles)
        print(f"  {t:>6}: {tile_counts[t]:>6} ({pct:.2f}%)")

    # Create main 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{name} ({len(games):,} games) - Survivorship Analysis', fontsize=14, fontweight='bold')

    # Tile threshold approximate step counts
    tile_steps = {4096: 3500, 8192: 7000, 16384: 14500, 32768: 22000, 65536: 30000}

    # Plot 1: Survivorship curve
    ax1 = axes[0, 0]
    sorted_steps = np.sort(steps)
    survival_prob = 1 - np.arange(1, len(sorted_steps) + 1) / len(sorted_steps)
    ax1.plot(sorted_steps, survival_prob, 'b-', linewidth=1.5)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Fraction of games still alive')
    ax1.set_title('Survivorship Curve')
    ax1.grid(True, alpha=0.3)
    for tile, step in tile_steps.items():
        ax1.axvline(x=step, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
        ax1.text(step, 0.95, f'{tile}', rotation=90, fontsize=8, alpha=0.7)

    # Plot 2: Death distribution histogram
    ax2 = axes[0, 1]
    bin_width = 500
    bins = np.arange(0, steps.max() + bin_width, bin_width)
    ax2.hist(steps, bins=bins, alpha=0.7, color='steelblue', edgecolor='none')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Deaths per bin')
    ax2.set_title(f'Death Distribution (bin={bin_width} steps)')
    ax2.grid(True, alpha=0.3, axis='y')
    for tile, step in tile_steps.items():
        ax2.axvline(x=step, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    # Plot 3: Final tile distribution
    ax3 = axes[1, 0]
    tile_labels = sorted(tile_counts.keys())
    tile_values = [tile_counts[t] for t in tile_labels]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(tile_labels)))
    bars = ax3.bar([str(t) for t in tile_labels], tile_values, color=colors)
    ax3.set_xlabel('Highest Tile Reached')
    ax3.set_ylabel('Number of Games')
    ax3.set_title('Final Tile Distribution')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, tile_values):
        pct = 100 * val / len(tiles)
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(tile_values) * 0.01,
                 f'{pct:.1f}%', ha='center', fontsize=9)

    # Plot 4: Hazard rate
    ax4 = axes[1, 1]
    hazard_bins = np.arange(0, steps.max() + 1000, 1000)
    deaths_per_bin, _ = np.histogram(steps, bins=hazard_bins)
    survivors_at_bin = len(steps) - np.cumsum(np.concatenate([[0], deaths_per_bin[:-1]]))
    hazard_rate = deaths_per_bin / np.maximum(survivors_at_bin, 1)
    bin_centers = (hazard_bins[:-1] + hazard_bins[1:]) / 2
    ax4.plot(bin_centers, hazard_rate * 100, 'b-', linewidth=1.5)
    ax4.fill_between(bin_centers, hazard_rate * 100, alpha=0.3)
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Hazard Rate (%)')
    ax4.set_title('Instantaneous Death Rate (% of survivors per 1000 steps)')
    ax4.grid(True, alpha=0.3)
    for tile, step in tile_steps.items():
        ax4.axvline(x=step, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_survivorship.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved {output_dir / f'{name}_survivorship.png'}")

    # Create transition zone figure
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle(f'{name} - Transition Zone Deaths', fontsize=12, fontweight='bold')

    zones = [
        ('8192→16384', 6000, 16000, 7000, 14500),
        ('16384→32768', 14000, 24000, 14500, 22000),
        ('32768→65536', 21000, 55000, 22000, 30000),
    ]

    for ax, (title, lo, hi, appear1, appear2) in zip(axes2, zones):
        zone_steps = steps[(steps > lo) & (steps < hi)]
        if len(zone_steps) > 0:
            ax.hist(zone_steps, bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.axvline(x=appear1, color='red', linestyle='--', label=f'~{title.split("→")[0]} appears')
        ax.axvline(x=appear2, color='green', linestyle='--', label=f'~{title.split("→")[1]} appears')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Deaths')
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_transitions.png', dpi=150, bbox_inches='tight')
    print(f"Saved {output_dir / f'{name}_transitions.png'}")

    # Print detailed analysis
    print(f"\n{'='*60}")
    print("DEATH STATISTICS BY TILE")
    print('='*60)

    died_at = defaultdict(list)
    for i in range(len(tiles)):
        died_at[tiles[i]].append(steps[i])

    for tile in sorted(died_at.keys()):
        s = np.array(died_at[tile])
        print(f"\n{tile}:")
        print(f"  Count: {len(s)} ({100*len(s)/len(tiles):.2f}%)")
        print(f"  Steps: min={s.min()}, max={s.max()}, mean={s.mean():.1f}, std={s.std():.1f}")

    # Corner death analysis
    print(f"\n{'='*60}")
    print("CORNER DEATH ANALYSIS")
    print('='*60)

    for thresh in [8192, 16384, 32768]:
        next_thresh = thresh * 2
        missed_steps = steps[tiles == thresh]
        if len(missed_steps) == 0:
            continue

        thresh_appear = {8192: 6500, 16384: 14000, 32768: 21500}
        next_appear = {8192: 14000, 16384: 21500, 32768: 29500}

        early_cutoff = thresh_appear[thresh] + 2000
        late_cutoff = next_appear[thresh] - 2000

        early = np.sum(missed_steps < early_cutoff)
        late = np.sum(missed_steps > late_cutoff)
        mid = len(missed_steps) - early - late

        print(f"\n{thresh} → {next_thresh} (n={len(missed_steps)}):")
        print(f"  Early (<{early_cutoff}): {early} ({100*early/len(missed_steps):.1f}%)")
        print(f"  Mid: {mid} ({100*mid/len(missed_steps):.1f}%)")
        print(f"  Late (>{late_cutoff}): {late} ({100*late/len(missed_steps):.1f}%)")

    # 32768 vs 65536
    print(f"\n{'='*60}")
    print("32768 vs 65536 ANALYSIS")
    print('='*60)

    reached_32k = np.sum(tiles >= 32768)
    reached_65k = np.sum(tiles >= 65536)
    stuck_32k = np.sum(tiles == 32768)

    print(f"\nReaching 32768: {reached_32k} ({100*reached_32k/len(tiles):.2f}%)")
    print(f"Reaching 65536: {reached_65k} ({100*reached_65k/len(tiles):.2f}%)")
    print(f"Stuck at 32768: {stuck_32k} ({100*stuck_32k/len(tiles):.2f}%)")

    if stuck_32k > 0:
        stuck_steps = steps[tiles == 32768]
        close = np.sum(stuck_steps > 28000)
        print(f"\nStuck-at-32768 games close to 65536 (>28000 steps): {close} ({100*close/stuck_32k:.1f}%)")

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(
        description='Survivorship analysis for 2048 game data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s metadata.db
    %(prog)s metadata.db -o ./analysis
    %(prog)s eval_results.jsonl
        """
    )
    parser.add_argument('input', type=Path, help='Input file (.db or .jsonl)')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help='Output directory (default: out/<input_stem>)')
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    output_dir = args.output or Path('out') / args.input.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    games = load_data(args.input)
    analyze_survivorship(games, output_dir, args.input.stem)

    print(f"\n✓ Analysis complete. Outputs in {output_dir}/")


if __name__ == '__main__':
    main()
