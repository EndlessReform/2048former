"""CLI helpers for fitting Macroxue tokenizers."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable, List, Sequence

from .macroxue import fit_macroxue_tokenizer


def _expand_globs(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        paths.extend(Path(match) for match in matches)
    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for path in sorted(paths):
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a Macroxue percentile tokenizer from raw rollouts")
    parser.add_argument(
        "globs",
        nargs="*",
        default=["datasets/raws/macroxue_d6_240g_tokenization/*.jsonl.gz"],
        help="Glob(s) for gzipped JSONL files (default: %(default)s)",
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=2049,
        help="Number of ECDF knots per valuation type (default: %(default)s)",
    )
    parser.add_argument(
        "--margin-bins",
        type=int,
        default=32,
        help="Number of loser-margin bins (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/raws/macroxue_d6_240g_tokenization/tokenizer_v1.json"),
        help="Where to write the tokenizer spec (default: %(default)s)",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional path to dump metadata summary as JSON",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    paths = _expand_globs(args.globs)
    if not paths:
        parser.error("No input files matched the provided globs")

    spec = fit_macroxue_tokenizer(
        paths,
        quantile_count=args.quantiles,
        margin_bins=args.margin_bins,
        show_progress=not args.no_progress,
    )

    spec.to_json(args.out)
    print(f"Tokenizer spec written to {args.out}")
    margin_counts = spec.metadata.get("margin_bin_counts")
    margin_preview: str
    if isinstance(margin_counts, list) and margin_counts:
        margin_preview = ", ".join(str(x) for x in margin_counts[:5])
    else:
        margin_preview = "n/a"

    print(
        f"States: {spec.metadata.get('total_states', 'n/a')}, "
        f"valuation types: {', '.join(spec.valuation_types)}"
    )
    print(
        f"Loser bins: {spec.metadata.get('margin_bins')}, "
        f"bin counts (first 5): {margin_preview}"
    )

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(spec.metadata, indent=2))
        print(f"Summary metadata written to {args.summary_out}")


if __name__ == "__main__":  # pragma: no cover
    main()
