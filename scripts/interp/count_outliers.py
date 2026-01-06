#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from safetensors.torch import safe_open


@dataclass
class OutlierAnalysis:
    key: str
    shape: tuple[int, int, int, int]
    dtype: torch.dtype
    total_elements: int
    element_hits: int
    outlier_occurrences: int
    outlier_occurrence_rate: float
    location_total: int
    location_outliers: int | None
    location_rate: float | None
    mean_layer_hit: float
    dim_counts: torch.Tensor
    pos_counts: torch.Tensor
    neg_counts: torch.Tensor
    layer_counts: torch.Tensor
    token_counts: torch.Tensor
    first_layer_counts: torch.Tensor
    outlier_mag_stats: dict[str, float]
    iqr_stats: dict[str, float]
    iqr_used: int
    size_mib: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count activation outliers in a safetensors dump.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("activations.safetensors"),
        help="Path to activations.safetensors.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="attn_norm",
        help="Activation key to analyze.",
    )
    parser.add_argument(
        "--compare-key",
        type=str,
        default=None,
        help="Optional second activation key to compare.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=6.0,
        help="Magnitude threshold for outliers.",
    )
    parser.add_argument(
        "--min-layer-fraction",
        type=float,
        default=0.25,
        help="Minimum fraction of layers with outliers at the same location.",
    )
    parser.add_argument(
        "--no-layer-fraction",
        action="store_true",
        help="Disable the same-location layer fraction requirement.",
    )
    parser.add_argument(
        "--iqr-samples",
        type=int,
        default=2_000_000,
        help="Number of samples to estimate IQR (<=0 disables).",
    )
    parser.add_argument(
        "--iqr-seed",
        type=int,
        default=0,
        help="RNG seed for IQR sampling.",
    )
    parser.add_argument(
        "--topk-dims",
        type=int,
        default=10,
        help="Show top-k hidden dimensions by outlier count.",
    )
    parser.add_argument(
        "--one-sided-threshold",
        type=float,
        default=0.95,
        help="Fraction for marking a dimension as one-sided.",
    )
    parser.add_argument(
        "--max-dim-list",
        type=int,
        default=48,
        help="Max number of outlier dims to print in the summary list.",
    )
    parser.add_argument(
        "--outlier-samples",
        type=int,
        default=0,
        help="Samples for outlier magnitude stats (<=0 uses all).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress debug output.",
    )
    return parser.parse_args()


def _load_tensor(path: Path, key: str) -> torch.Tensor:
    with safe_open(path, framework="pt") as f:
        if key not in f.keys():
            available = ", ".join(sorted(f.keys()))
            msg = f"Key '{key}' not found. Available keys: {available}"
            raise KeyError(msg)
        return f.get_tensor(key)


def _estimate_iqr(
    tensor: torch.Tensor,
    sample_size: int,
    seed: int,
) -> tuple[float, float, float, int]:
    if sample_size <= 0:
        return 0.0, 0.0, 0.0, 0

    flat = tensor.reshape(-1)
    total = flat.numel()
    if total == 0:
        return 0.0, 0.0, 0.0, 0

    if sample_size >= total:
        sample = flat.float()
        used = total
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        indices = torch.randint(0, total, (sample_size,), generator=generator)
        sample = flat[indices].float()
        used = sample_size

    quantiles = torch.quantile(sample, torch.tensor([0.25, 0.5, 0.75]))
    q1 = float(quantiles[0].item())
    q2 = float(quantiles[1].item())
    q3 = float(quantiles[2].item())
    return q1, q2, q3, used


def _log(console: Console, quiet: bool, message: str) -> None:
    if not quiet:
        console.print(message)


def _estimate_outlier_magnitudes(
    values: torch.Tensor,
    sample_size: int,
    seed: int,
) -> dict[str, float]:
    if values.numel() == 0:
        return {"min": 0.0, "q1": 0.0, "median": 0.0, "q3": 0.0, "max": 0.0}

    if sample_size > 0 and sample_size < values.numel():
        generator = torch.Generator()
        generator.manual_seed(seed)
        indices = torch.randint(0, values.numel(), (sample_size,), generator=generator)
        sample = values[indices]
    else:
        sample = values

    q1, q2, q3 = torch.quantile(sample, torch.tensor([0.25, 0.5, 0.75]))
    return {
        "min": float(sample.min().item()),
        "q1": float(q1.item()),
        "median": float(q2.item()),
        "q3": float(q3.item()),
        "max": float(sample.max().item()),
    }


def _analyze_tensor(
    tensor: torch.Tensor,
    *,
    key: str,
    threshold: float,
    min_layer_fraction: float,
    require_layer_fraction: bool,
    iqr_samples: int,
    iqr_seed: int,
    outlier_samples: int,
    quiet: bool,
    console: Console,
) -> OutlierAnalysis:
    if tensor.ndim != 4:
        msg = f"Expected 4D tensor (N, L, T, H); got shape {tuple(tensor.shape)}"
        raise ValueError(msg)

    size_bytes = tensor.numel() * tensor.element_size()
    size_mib = size_bytes / (1024 * 1024)
    _log(
        console,
        quiet,
        f"[bold]Loaded[/bold] {key} shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} elements={tensor.numel():,} "
        f"~{size_mib:,.1f} MiB",
    )

    _log(console, quiet, "[bold]Computing[/bold] IQR estimate")
    q1, q2, q3, iqr_used = _estimate_iqr(tensor, iqr_samples, iqr_seed)
    iqr_stats = {"q1": q1, "median": q2, "q3": q3, "iqr": q3 - q1}

    _log(console, quiet, "[bold]Computing[/bold] outlier mask")
    mask = (tensor > threshold) | (tensor < -threshold)
    total_elements = mask.numel()
    element_hits = int(mask.sum().item())

    if require_layer_fraction:
        _log(console, quiet, "[bold]Filtering[/bold] by layer-fraction requirement")
        layer_hits = mask.float().mean(dim=1)
        location_mask = layer_hits >= min_layer_fraction
        location_outliers = int(location_mask.sum().item())
        location_total = location_mask.numel()
        location_rate = location_outliers / location_total if location_total else 0.0
        mean_layer_hit = (
            float(layer_hits[location_mask].mean().item()) if location_outliers else 0.0
        )
        mask_filtered = mask & location_mask[:, None, :, :]
        del layer_hits
    else:
        location_mask = None
        location_outliers = None
        location_total = tensor.shape[0] * tensor.shape[2] * tensor.shape[3]
        location_rate = None
        mean_layer_hit = 0.0
        mask_filtered = mask

    outlier_occurrences = int(mask_filtered.sum().item())
    outlier_occurrence_rate = (
        outlier_occurrences / total_elements if total_elements else 0.0
    )

    _log(console, quiet, "[bold]Summarizing[/bold] by hidden dimension/layer/token")
    dim_counts = mask_filtered.sum(dim=(0, 1, 2)).to(torch.int64)
    layer_counts = mask_filtered.sum(dim=(0, 2, 3)).to(torch.int64)
    token_counts = mask_filtered.sum(dim=(0, 1, 3)).to(torch.int64)

    _log(console, quiet, "[bold]Summarizing[/bold] sign distribution")
    if location_mask is None:
        pos_mask = tensor > threshold
        neg_mask = tensor < -threshold
    else:
        pos_mask = (tensor > threshold) & location_mask[:, None, :, :]
        neg_mask = (tensor < -threshold) & location_mask[:, None, :, :]
    pos_counts = pos_mask.sum(dim=(0, 1, 2)).to(torch.int64)
    neg_counts = neg_mask.sum(dim=(0, 1, 2)).to(torch.int64)
    del pos_mask, neg_mask

    _log(console, quiet, "[bold]Summarizing[/bold] first-layer emergence")
    first_layer_mask = mask_filtered
    any_mask = first_layer_mask.any(dim=1)
    if any_mask.any():
        first_layer = first_layer_mask.to(torch.uint8).argmax(dim=1)
        first_layer_counts = torch.bincount(
            first_layer[any_mask].flatten(),
            minlength=tensor.shape[1],
        ).to(torch.int64)
    else:
        first_layer_counts = torch.zeros(
            tensor.shape[1],
            dtype=torch.int64,
        )
    del any_mask, first_layer_mask

    _log(console, quiet, "[bold]Summarizing[/bold] outlier magnitudes")
    outlier_values = tensor[mask_filtered].float().abs()
    outlier_mag_stats = _estimate_outlier_magnitudes(
        outlier_values,
        sample_size=outlier_samples,
        seed=iqr_seed,
    )
    del outlier_values

    return OutlierAnalysis(
        key=key,
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        total_elements=total_elements,
        element_hits=element_hits,
        outlier_occurrences=outlier_occurrences,
        outlier_occurrence_rate=outlier_occurrence_rate,
        location_total=location_total,
        location_outliers=location_outliers,
        location_rate=location_rate,
        mean_layer_hit=mean_layer_hit,
        dim_counts=dim_counts,
        pos_counts=pos_counts,
        neg_counts=neg_counts,
        layer_counts=layer_counts,
        token_counts=token_counts,
        first_layer_counts=first_layer_counts,
        outlier_mag_stats=outlier_mag_stats,
        iqr_stats=iqr_stats,
        iqr_used=iqr_used,
        size_mib=size_mib,
    )


def _render_summary(
    console: Console,
    analysis: OutlierAnalysis,
    *,
    threshold: float,
    min_layer_fraction: float,
    require_layer_fraction: bool,
    input_path: Path,
) -> None:
    table = Table(title=f"Activation Outliers ({analysis.key})")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Input", str(input_path))
    table.add_row("Key", analysis.key)
    table.add_row("Threshold", f"{threshold:.4f}")
    if require_layer_fraction:
        table.add_row("Min layer fraction", f"{min_layer_fraction:.2%}")
    else:
        table.add_row("Layer fraction requirement", "disabled")

    n_samples, n_layers, n_tokens, hidden = analysis.shape
    table.add_row("Tensor shape", f"({n_samples}, {n_layers}, {n_tokens}, {hidden})")
    table.add_row("Elements > threshold", f"{analysis.element_hits:,}")
    table.add_row("Outlier occurrences", f"{analysis.outlier_occurrences:,}")
    table.add_row("Outlier occurrence rate", f"{analysis.outlier_occurrence_rate:.4%}")
    if require_layer_fraction:
        if analysis.location_outliers is not None and analysis.location_rate is not None:
            table.add_row("Outlier locations", f"{analysis.location_outliers:,}")
            table.add_row("Outlier location rate", f"{analysis.location_rate:.4%}")
        table.add_row(
            "Mean layer-hit fraction (outliers)",
            f"{analysis.mean_layer_hit:.2%}",
        )
    if analysis.iqr_used:
        table.add_row("Q1 (25%)", f"{analysis.iqr_stats['q1']:.4f}")
        table.add_row("Median (50%)", f"{analysis.iqr_stats['median']:.4f}")
        table.add_row("Q3 (75%)", f"{analysis.iqr_stats['q3']:.4f}")
        table.add_row("IQR (Q3-Q1)", f"{analysis.iqr_stats['iqr']:.4f}")

    console.print(table)


def _render_dim_summary(
    console: Console,
    analysis: OutlierAnalysis,
    *,
    topk: int,
    one_sided_threshold: float,
    max_dim_list: int,
) -> None:
    dim_counts = analysis.dim_counts
    total_outliers = analysis.outlier_occurrences
    nonzero = torch.nonzero(dim_counts, as_tuple=False).flatten()
    dim_total = dim_counts.numel()
    unique_dims = int(nonzero.numel())
    sorted_dims = nonzero
    sorted_counts = dim_counts[nonzero]
    if unique_dims:
        sorted_counts, order = torch.sort(sorted_counts, descending=True)
        sorted_dims = sorted_dims[order]

    list_limit = min(unique_dims, max_dim_list)
    listed_dims = (
        ", ".join(str(int(idx.item())) for idx in sorted_dims[:list_limit])
        if list_limit
        else "-"
    )

    summary = Table(title=f"Outlier Dimensions ({analysis.key})")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("Hidden dims", f"{dim_total}")
    summary.add_row("Dims with outliers", f"{unique_dims}")
    if unique_dims:
        top6 = min(6, unique_dims)
        top6_share = (
            float(sorted_counts[:top6].sum().item()) / total_outliers
            if total_outliers
            else 0.0
        )
        summary.add_row(f"Top {top6} share", f"{top6_share:.2%}")
    if list_limit:
        suffix = "" if unique_dims <= max_dim_list else "…"
        summary.add_row(
            f"Outlier dims (first {list_limit})",
            f"{listed_dims}{suffix}",
        )
    console.print(summary)

    if not unique_dims:
        return

    topk = min(topk, unique_dims)
    table = Table(title=f"Top {topk} Dimensions ({analysis.key})")
    table.add_column("Rank")
    table.add_column("Dim")
    table.add_column("Count")
    table.add_column("% of outliers")
    table.add_column("Pos %")
    table.add_column("Neg %")
    table.add_column("One-sided")

    pos_counts = analysis.pos_counts
    neg_counts = analysis.neg_counts
    for rank in range(topk):
        dim_idx = int(sorted_dims[rank].item())
        count = int(sorted_counts[rank].item())
        pos = int(pos_counts[dim_idx].item())
        neg = int(neg_counts[dim_idx].item())
        total = pos + neg
        pos_frac = pos / total if total else 0.0
        neg_frac = neg / total if total else 0.0
        one_sided = max(pos_frac, neg_frac) >= one_sided_threshold
        table.add_row(
            str(rank + 1),
            str(dim_idx),
            f"{count:,}",
            f"{(count / total_outliers if total_outliers else 0.0):.2%}",
            f"{pos_frac:.2%}",
            f"{neg_frac:.2%}",
            "yes" if one_sided else "no",
        )

    console.print(table)


def _render_layer_tables(console: Console, analysis: OutlierAnalysis) -> None:
    total = analysis.outlier_occurrences
    layer_counts = analysis.layer_counts
    table = Table(title=f"Layer Outlier Occurrences ({analysis.key})")
    table.add_column("Layer")
    table.add_column("Count")
    table.add_column("% of outliers")
    for idx, count in enumerate(layer_counts.tolist()):
        share = count / total if total else 0.0
        table.add_row(str(idx), f"{count:,}", f"{share:.2%}")
    console.print(table)

    first_counts = analysis.first_layer_counts
    first_total = int(first_counts.sum().item())
    first_table = Table(title=f"First-Layer Appearance ({analysis.key})")
    first_table.add_column("Layer")
    first_table.add_column("Count")
    first_table.add_column("% of locations")
    for idx, count in enumerate(first_counts.tolist()):
        share = count / first_total if first_total else 0.0
        first_table.add_row(str(idx), f"{count:,}", f"{share:.2%}")
    console.print(first_table)


def _render_token_table(console: Console, analysis: OutlierAnalysis) -> None:
    total = analysis.outlier_occurrences
    table = Table(title=f"Token Position Outliers ({analysis.key})")
    table.add_column("Token")
    table.add_column("Count")
    table.add_column("% of outliers")
    for idx, count in enumerate(analysis.token_counts.tolist()):
        share = count / total if total else 0.0
        table.add_row(str(idx), f"{count:,}", f"{share:.2%}")
    console.print(table)


def _render_magnitude_table(console: Console, analysis: OutlierAnalysis) -> None:
    stats = analysis.outlier_mag_stats
    table = Table(title=f"Outlier Magnitudes ({analysis.key})")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Min", f"{stats['min']:.4f}")
    table.add_row("Q1 (25%)", f"{stats['q1']:.4f}")
    table.add_row("Median (50%)", f"{stats['median']:.4f}")
    table.add_row("Q3 (75%)", f"{stats['q3']:.4f}")
    table.add_row("Max", f"{stats['max']:.4f}")
    console.print(table)


def _render_compare(
    console: Console,
    left: OutlierAnalysis,
    right: OutlierAnalysis,
) -> None:
    left_dims = set(torch.nonzero(left.dim_counts, as_tuple=False).flatten().tolist())
    right_dims = set(torch.nonzero(right.dim_counts, as_tuple=False).flatten().tolist())
    common_dims = left_dims & right_dims
    union_dims = left_dims | right_dims

    summary = Table(title="Key Comparison")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("Key A", left.key)
    summary.add_row("Key B", right.key)
    summary.add_row("Dims with outliers (A)", f"{len(left_dims)}")
    summary.add_row("Dims with outliers (B)", f"{len(right_dims)}")
    summary.add_row("Common dims", f"{len(common_dims)}")
    summary.add_row(
        "Jaccard (dims)",
        f"{(len(common_dims) / len(union_dims) if union_dims else 0.0):.2%}",
    )
    console.print(summary)

    if not common_dims:
        return

    common = torch.tensor(sorted(common_dims), dtype=torch.int64)
    common_counts = torch.minimum(
        left.dim_counts[common],
        right.dim_counts[common],
    )
    sorted_counts, order = torch.sort(common_counts, descending=True)
    topk = min(10, sorted_counts.numel())
    table = Table(title="Top Common Outlier Dims (min count)")
    table.add_column("Rank")
    table.add_column("Dim")
    table.add_column("Min count")
    for idx in range(topk):
        dim_idx = int(common[order[idx]].item())
        count = int(sorted_counts[idx].item())
        table.add_row(str(idx + 1), str(dim_idx), f"{count:,}")
    console.print(table)


def main() -> None:
    args = _parse_args()
    console = Console()

    analysis = _analyze_tensor(
        _load_tensor(args.input, args.key),
        key=args.key,
        threshold=args.threshold,
        min_layer_fraction=args.min_layer_fraction,
        require_layer_fraction=not args.no_layer_fraction,
        iqr_samples=args.iqr_samples,
        iqr_seed=args.iqr_seed,
        outlier_samples=args.outlier_samples,
        quiet=args.quiet,
        console=console,
    )
    if analysis.iqr_used and not args.quiet:
        console.print(
            f"[bold]IQR sample[/bold] {analysis.iqr_used:,} values "
            f"(seed={args.iqr_seed})"
        )
    _render_summary(
        console,
        analysis,
        threshold=args.threshold,
        min_layer_fraction=args.min_layer_fraction,
        require_layer_fraction=not args.no_layer_fraction,
        input_path=args.input,
    )
    _render_dim_summary(
        console,
        analysis,
        topk=args.topk_dims,
        one_sided_threshold=args.one_sided_threshold,
        max_dim_list=args.max_dim_list,
    )
    _render_layer_tables(console, analysis)
    _render_token_table(console, analysis)
    _render_magnitude_table(console, analysis)

    if args.compare_key:
        compare_analysis = _analyze_tensor(
            _load_tensor(args.input, args.compare_key),
            key=args.compare_key,
            threshold=args.threshold,
            min_layer_fraction=args.min_layer_fraction,
            require_layer_fraction=not args.no_layer_fraction,
            iqr_samples=args.iqr_samples,
            iqr_seed=args.iqr_seed,
            outlier_samples=args.outlier_samples,
            quiet=args.quiet,
            console=console,
        )
        if compare_analysis.iqr_used and not args.quiet:
            console.print(
                f"[bold]IQR sample[/bold] {compare_analysis.iqr_used:,} values "
                f"(seed={args.iqr_seed})"
            )
        _render_summary(
            console,
            compare_analysis,
            threshold=args.threshold,
            min_layer_fraction=args.min_layer_fraction,
            require_layer_fraction=not args.no_layer_fraction,
            input_path=args.input,
        )
        _render_dim_summary(
            console,
            compare_analysis,
            topk=args.topk_dims,
            one_sided_threshold=args.one_sided_threshold,
            max_dim_list=args.max_dim_list,
        )
        _render_layer_tables(console, compare_analysis)
        _render_token_table(console, compare_analysis)
        _render_magnitude_table(console, compare_analysis)
        _render_compare(console, analysis, compare_analysis)


if __name__ == "__main__":
    main()
