"""CLI entrypoint for fitting the Macroxue v2 tokenizer on packed datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from .macroxue import fit_macroxue_tokenizer_v2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit the Macroxue advantage tokenizer (v2) from a packed dataset"
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the packed dataset directory (expects steps.npy / valuation_types.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tokenizer.json"),
        help="Where to write the tokenizer spec (default: %(default)s)",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=32,
        help="Number of disadvantage bins shared across all valuation types (default: %(default)s)",
    )
    parser.add_argument(
        "--search-failure-cutoff",
        type=int,
        default=-1500,
        help="Cutoff (in heuristic score units) for marking search branches as FAILURE (default: %(default)s)",
    )
    parser.add_argument(
        "--zero-tolerance",
        type=float,
        default=1e-9,
        help="Tolerance for treating tuple disadvantages as zero (default: %(default)s)",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional path to dump metadata summary as JSON",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation width for the saved JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output file if it already exists",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar during fitting",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    spec = fit_macroxue_tokenizer_v2(
        args.dataset,
        num_bins=args.num_bins,
        search_failure_cutoff=args.search_failure_cutoff,
        zero_tolerance=args.zero_tolerance,
        show_progress=not args.no_progress,
    )

    spec.to_json(args.output, indent=args.indent, overwrite=args.overwrite)
    print(f"[tokenizer] wrote spec to {args.output}")

    meta_summary = spec.metadata.copy()
    meta_summary["tokenizer_type"] = spec.tokenizer_type
    meta_summary["version"] = spec.version
    meta_summary["num_bins"] = spec.num_bins
    meta_summary["search_failure_cutoff"] = spec.search.failure_cutoff

    def _fmt_stats(name: str, stats: dict) -> str:
        return (
            f"{name}: rows={stats.get('rows', 'n/a')}, legal={stats.get('legal_branches', 'n/a')}, "
            f"failure={stats.get('failure_branches', 'n/a')}, zero={stats.get('zero_disadvantage', 'n/a')}"
        )

    for key in ("search", "tuple10", "tuple11"):
        stats = meta_summary.get(key)
        if isinstance(stats, dict):
            print(f"[tokenizer] {_fmt_stats(key, stats)}")

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(meta_summary, indent=args.indent))
        print(f"[tokenizer] wrote summary to {args.summary_out}")


if __name__ == "__main__":  # pragma: no cover
    main()
