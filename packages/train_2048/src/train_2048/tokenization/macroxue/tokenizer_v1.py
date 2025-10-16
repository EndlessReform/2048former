"""Tokenizer fitting and application for Macroxue expectimax rollouts.

The tokenizer maps branch EVs to percentiles via empirical CDF knots and
emits winner/loser/illegal tokens in percentile space. Specification is
serialised as JSON for reuse when tokenising fresh datasets.
"""

from __future__ import annotations

import argparse
import gzip
import json
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from bisect import bisect_right

from tqdm import tqdm


ACTIONS: Tuple[str, str, str, str] = ("up", "left", "right", "down")
Action = str


@dataclass(slots=True)
class MacroxueTokenizerSpec:
    """Frozen tokenizer configuration with ECDF knots and Δ-bin edges."""

    version: int
    quantile_count: int
    actions: Tuple[str, ...]
    valuation_types: List[str]
    ecdf_knots: Dict[str, List[float]]
    delta_edges: List[float]
    percentile_grid: str = "uniform_0_1"
    notes: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "quantile_count": self.quantile_count,
            "actions": list(self.actions),
            "valuation_types": list(self.valuation_types),
            "ecdf_knots": self.ecdf_knots,
            "delta_edges": list(self.delta_edges),
            "percentile_grid": self.percentile_grid,
            "notes": self.notes,
            "metadata": self.metadata,
        }

    def to_json(self, path: Path, *, indent: int = 2) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=indent))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MacroxueTokenizerSpec":
        return cls(
            version=int(payload["version"]),
            quantile_count=int(payload["quantile_count"]),
            actions=tuple(payload["actions"])  # type: ignore[arg-type]
            if "actions" in payload
            else ACTIONS,
            valuation_types=list(payload["valuation_types"]),  # type: ignore[arg-type]
            ecdf_knots=_parse_knots(payload.get("ecdf_knots")),
            delta_edges=list(map(float, payload["delta_edges"])),  # type: ignore[arg-type]
            percentile_grid=str(payload.get("percentile_grid", "uniform_0_1")),
            notes=payload.get("notes")
            if isinstance(payload.get("notes"), str)
            else None,
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, path: Path) -> "MacroxueTokenizerSpec":
        return cls.from_dict(json.loads(path.read_text()))


def _iter_records(path: Path) -> Iterator[Mapping[str, object]]:
    with gzip.open(path, "rt") as fh:
        for line in fh:
            yield json.loads(line)


def _parse_knots(obj: object | None) -> Dict[str, List[float]]:
    if obj is None:
        return {}
    if not isinstance(obj, Mapping):
        raise TypeError("ecdf_knots must be a mapping of valuation_type -> sequence")
    parsed: Dict[str, List[float]] = {}
    for key, values in obj.items():
        if not isinstance(values, Iterable):
            raise TypeError("Each ecdf_knots entry must be iterable")
        parsed[str(key)] = [float(v) for v in values]
    return parsed


def _percentile_from_knots(knots: np.ndarray, value: float) -> float:
    if knots.size == 0:
        return 0.0
    if value <= knots[0]:
        return 0.0
    if value >= knots[-1]:
        return 1.0
    idx = int(np.searchsorted(knots, value, side="right"))
    n = knots.size - 1
    if idx <= 0:
        return 0.0
    lo_val = float(knots[idx - 1])
    hi_val = float(knots[idx])
    if hi_val <= lo_val:
        return idx / n
    frac = (value - lo_val) / (hi_val - lo_val)
    return min(max((idx - 1 + frac) / n, 0.0), 1.0)


def _digitize(delta: float, edges: Sequence[float]) -> int:
    if len(edges) < 2:
        return 0
    idx = bisect_right(edges, float(delta)) - 1
    idx = max(0, min(idx, len(edges) - 2))
    return idx


@dataclass(slots=True)
class TokenizedState:
    """Percentile-space representation of a single board decision."""

    winner_action: Optional[Action]
    percentiles: Dict[Action, float]
    deltas: Dict[Action, float]
    delta_bins: Dict[Action, Optional[int]]
    illegal_actions: Tuple[Action, ...]
    gap_top2: Optional[float]
    gap_bin: Optional[int]

    def winner_token(self) -> Optional[str]:
        if self.winner_action is None:
            return None
        return f"[WINNER]_{self.winner_action}"

    def margin_token(self, action: Action) -> Optional[str]:
        bin_idx = self.delta_bins.get(action)
        if bin_idx is None:
            return None
        return f"{action}#d{bin_idx}"

    def illegal_token(self, action: Action) -> Optional[str]:
        if action in self.illegal_actions:
            return f"[ILLEGAL]_{action}"
        return None

    def gap_token(self) -> Optional[str]:
        if self.gap_bin is None:
            return None
        return f"gap#d{self.gap_bin}"

    def all_tokens(self, *, include_gap: bool = False) -> List[str]:
        tokens: List[str] = []
        win = self.winner_token()
        if win:
            tokens.append(win)
        for action in ACTIONS:
            if action == self.winner_action:
                continue
            illegal = self.illegal_token(action)
            if illegal:
                tokens.append(illegal)
                continue
            margin = self.margin_token(action)
            if margin:
                tokens.append(margin)
        if include_gap:
            gap = self.gap_token()
            if gap:
                tokens.append(gap)
        return tokens


class MacroxueTokenizer:
    """Project best/alternative actions into the frozen percentile space."""

    def __init__(
        self, spec: MacroxueTokenizerSpec, *, include_gap_token: bool = False
    ) -> None:
        self.spec = spec
        self.include_gap_token = include_gap_token
        self._knots = {
            vt: np.array(values, dtype=np.float64)
            for vt, values in spec.ecdf_knots.items()
        }
        self._delta_edges = np.array(spec.delta_edges, dtype=np.float64)

    def percentile(self, valuation_type: str, value: float) -> float:
        knots = self._knots.get(valuation_type)
        if knots is None:
            raise KeyError(f"Unknown valuation type: {valuation_type!r}")
        return float(_percentile_from_knots(knots, float(value)))

    def encode_state(
        self,
        valuation_type: str,
        branch_evs: Mapping[str, Optional[float]],
    ) -> TokenizedState:
        knots = self._knots.get(valuation_type)
        if knots is None:
            raise KeyError(f"Unknown valuation type: {valuation_type!r}")
        percentiles: Dict[str, float] = {}
        deltas: Dict[str, float] = {}
        delta_bins: Dict[str, Optional[int]] = {}
        illegal_actions: List[str] = []
        legal_actions: List[str] = []

        for action in ACTIONS:
            ev = branch_evs.get(action)
            if ev is None:
                percentiles[action] = 0.0
                deltas[action] = 1.0
                delta_bins[action] = None
                illegal_actions.append(action)
                continue
            p = _percentile_from_knots(knots, float(ev))
            percentiles[action] = p
            legal_actions.append(action)

        winner_action: Optional[str] = None
        winner_p = -1.0
        for action in legal_actions:
            p = percentiles[action]
            if p > winner_p:
                winner_p = p
                winner_action = action

        gap_top2: Optional[float]
        if len(legal_actions) >= 2:
            sorted_ps = sorted(percentiles[a] for a in legal_actions)
            gap_top2 = float(sorted_ps[-1] - sorted_ps[-2])
        elif len(legal_actions) == 1:
            gap_top2 = 1.0
        else:
            gap_top2 = None

        gap_bin: Optional[int] = None
        if gap_top2 is not None:
            gap_bin = _digitize(gap_top2, self._delta_edges)

        if winner_action is not None:
            p_best = percentiles[winner_action]
            for action in legal_actions:
                if action == winner_action:
                    deltas[action] = 0.0
                    delta_bins[action] = None
                    continue
                delta = float(np.clip(p_best - percentiles[action], 0.0, 1.0))
                deltas[action] = delta
                delta_bins[action] = _digitize(delta, self._delta_edges)
        return TokenizedState(
            winner_action=winner_action,
            percentiles=percentiles,
            deltas=deltas,
            delta_bins=delta_bins,
            illegal_actions=tuple(illegal_actions),
            gap_top2=gap_top2,
            gap_bin=gap_bin if self.include_gap_token else None,
        )


def _list_paths(glob_pattern: str) -> List[Path]:
    root = Path()
    paths = sorted(p for p in root.glob(glob_pattern) if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No files matched glob: {glob_pattern}")
    return paths


def fit_macroxue_tokenizer(
    paths: Sequence[Path],
    *,
    quantile_count: int = 2049,
    margin_bins: int = 32,
    show_progress: bool = True,
) -> MacroxueTokenizerSpec:
    if quantile_count < 3:
        raise ValueError("quantile_count must be >= 3 for stable interpolation")
    if margin_bins <= 0:
        raise ValueError("margin_bins must be positive")

    buffers: MutableMapping[str, array] = {}
    valuation_state_counts: Dict[str, int] = {}
    legal_counts: Dict[str, int] = {action: 0 for action in ACTIONS}
    illegal_counts: Dict[str, int] = {action: 0 for action in ACTIONS}
    total_states = 0

    iterator = tqdm(paths, desc="Collect EVs", unit="file") if show_progress else paths
    for path in iterator:
        for rec in _iter_records(path):
            valuation_type = str(rec["valuation_type"])
            valuation_state_counts[valuation_type] = (
                valuation_state_counts.get(valuation_type, 0) + 1
            )
            branch_raw = rec.get("branch_evs", {})
            if not isinstance(branch_raw, Mapping):
                raise TypeError("branch_evs must be a mapping of action->EV")
            branch = branch_raw
            total_states += 1

            buf = buffers.get(valuation_type)
            if buf is None:
                buf = buffers[valuation_type] = array("d")

            for action in ACTIONS:
                ev = branch.get(action) if isinstance(branch, Mapping) else None
                if ev is None:
                    illegal_counts[action] += 1
                    continue
                legal_counts[action] += 1
                buf.append(float(ev))

    if not buffers:
        raise RuntimeError("No branch EVs collected; input appears empty")

    ecdf_knots: Dict[str, List[float]] = {}
    p_grid = np.linspace(0.0, 1.0, quantile_count)
    numpy_knots: Dict[str, np.ndarray] = {}

    for valuation_type, buf in buffers.items():
        arr = np.frombuffer(buf, dtype=np.float64)
        if arr.size == 0:
            raise RuntimeError(f"Valuation type {valuation_type} had zero legal EVs")
        arr.sort()
        numpy_knots[valuation_type] = arr
        idx = np.clip((p_grid * (arr.size - 1)).astype(np.int64), 0, arr.size - 1)
        knots = arr[idx]
        ecdf_knots[valuation_type] = knots.astype(float).tolist()

    delta_values = array("f")
    gap_values = array("f")
    winner_counts: Dict[str, int] = {action: 0 for action in ACTIONS}

    iterator2 = tqdm(paths, desc="Gather deltas", unit="file") if show_progress else paths
    for path in iterator2:
        for rec in _iter_records(path):
            valuation_type = str(rec["valuation_type"])
            branch_raw = rec.get("branch_evs", {})
            if not isinstance(branch_raw, Mapping):
                raise TypeError("branch_evs must be a mapping of action->EV")
            branch = branch_raw
            knots = numpy_knots[valuation_type]
            percentiles_by_action: Dict[str, float] = {}
            for action in ACTIONS:
                ev = branch.get(action) if isinstance(branch, Mapping) else None
                if ev is None:
                    continue
                p = _percentile_from_knots(knots, float(ev))
                percentiles_by_action[action] = p

            if not percentiles_by_action:
                continue
            winner_action = max(percentiles_by_action.items(), key=lambda item: item[1])[
                0
            ]
            winner_counts[winner_action] += 1
            p_best = percentiles_by_action[winner_action]

            legal_ps = sorted(percentiles_by_action.values())
            if len(legal_ps) >= 2:
                gap = legal_ps[-1] - legal_ps[-2]
            else:
                gap = 1.0
            gap_values.append(float(gap))

            for action, percentile in percentiles_by_action.items():
                if action == winner_action:
                    continue
                delta = p_best - percentile
                delta_values.append(float(delta))

    delta_array = np.frombuffer(delta_values, dtype=np.float32)
    gap_array = np.frombuffer(gap_values, dtype=np.float32)
    if delta_array.size == 0:
        raise RuntimeError("No non-best deltas observed; dataset may be degenerate")

    quantile_points = np.linspace(0.0, 1.0, margin_bins + 1, endpoint=False)[1:]
    interior = np.quantile(delta_array, quantile_points, method="linear")
    edges = np.concatenate(([0.0], interior, [1.0]))
    edges = np.unique(edges)
    if edges[0] != 0.0:
        edges[0] = 0.0
    if edges[-1] != 1.0:
        edges = np.append(edges, 1.0)
    delta_edges = edges.astype(float).tolist()

    delta_bins = np.clip(
        np.digitize(delta_array, edges, right=True) - 1, 0, len(edges) - 2
    )
    bin_counts = np.bincount(delta_bins, minlength=len(edges) - 1).astype(int).tolist()

    metadata = {
        "total_states": int(total_states),
        "valuation_state_counts": {
            k: int(v) for k, v in valuation_state_counts.items()
        },
        "legal_counts": {k: int(v) for k, v in legal_counts.items()},
        "illegal_counts": {k: int(v) for k, v in illegal_counts.items()},
        "winner_counts": {k: int(v) for k, v in winner_counts.items()},
        "margin_bin_counts": bin_counts,
        "margin_bins": len(delta_edges) - 1,
        "gap_summary": {
            "count": int(gap_array.size),
            "p50": float(np.median(gap_array)) if gap_array.size else 0.0,
            "p90": float(np.quantile(gap_array, 0.9)) if gap_array.size else 0.0,
            "p99": float(np.quantile(gap_array, 0.99)) if gap_array.size else 0.0,
        },
    }

    spec = MacroxueTokenizerSpec(
        version=1,
        quantile_count=quantile_count,
        actions=ACTIONS,
        valuation_types=sorted(ecdf_knots.keys()),
        ecdf_knots=ecdf_knots,
        delta_edges=delta_edges,
        notes=(
            "Macroxue winner+margin tokenizer with ECDF percentiles and "
            f"quantile-based {len(delta_edges) - 1} loser bins"
        ),
        metadata=metadata,
    )
    return spec


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glob",
        default="datasets/raws/macroxue_d6_240g_tokenization/**/*.jsonl.gz",
        help="Glob pattern for input data (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default="out/tokenizer.json",
        type=Path,
        help="Output path for the tokenizer spec (default: %(default)s)",
    )
    parser.add_argument(
        "--quantiles",
        default=2049,
        type=int,
        help="Number of knots for ECDF (default: %(default)s)",
    )
    parser.add_argument(
        "--bins",
        default=32,
        type=int,
        help="Number of margin bins (default: %(default)s)",
    )
    args = parser.parse_args()

    paths = _list_paths(args.glob)
    print(f"Fitting tokenizer from {len(paths)} files...")
    spec = fit_macroxue_tokenizer(
        paths,
        quantile_count=args.quantiles,
        margin_bins=args.bins,
    )
    spec.to_json(args.out)
    print(f"Tokenizer spec written to {args.out}")


if __name__ == "__main__":
    main_cli()


__all__ = [
    "ACTIONS",
    "MacroxueTokenizer",
    "MacroxueTokenizerSpec",
    "TokenizedState",
    "fit_macroxue_tokenizer",
]