"""Advantage-based tokenizer (v2) for Macroxue datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

try:  # pragma: no cover - tqdm is optional during unit tests
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

NUM_BRANCHES = 4
BRANCH_AXIS = np.arange(NUM_BRANCHES, dtype=np.int64)
SEARCH_VALUE_SCALE = 1000.0
ZERO_TOLERANCE = 1e-9


@dataclass(slots=True)
class MacroxueTokenizerV2TypeConfig:
    """Quantile configuration for a specific valuation family."""

    bin_edges: List[float]
    failure_cutoff: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {"bin_edges": list(map(float, self.bin_edges))}
        if self.failure_cutoff is not None:
            payload["failure_cutoff"] = int(self.failure_cutoff)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MacroxueTokenizerV2TypeConfig":
        edges_obj = payload.get("bin_edges")
        if not isinstance(edges_obj, Sequence):
            raise TypeError("bin_edges must be a sequence")
        bin_edges = [float(value) for value in edges_obj]
        failure_cutoff = payload.get("failure_cutoff")
        cutoff_int = int(failure_cutoff) if failure_cutoff is not None else None
        return cls(bin_edges=bin_edges, failure_cutoff=cutoff_int)


@dataclass(slots=True)
class MacroxueTokenizerV2Spec:
    """Serialized configuration for the advantage-based tokenizer."""

    tokenizer_type: str
    version: int
    num_bins: int
    vocab_order: List[str]
    valuation_types: List[str]
    search: MacroxueTokenizerV2TypeConfig
    tuple10: MacroxueTokenizerV2TypeConfig
    tuple11: MacroxueTokenizerV2TypeConfig
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "tokenizer_type": self.tokenizer_type,
            "version": self.version,
            "num_bins": self.num_bins,
            "vocab_order": list(self.vocab_order),
            "valuation_types": list(self.valuation_types),
            "search": self.search.to_dict(),
            "tuple10": self.tuple10.to_dict(),
            "tuple11": self.tuple11.to_dict(),
            "metadata": self.metadata,
        }

    def to_json(self, path: Path, *, indent: int = 2, overwrite: bool = False) -> None:
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists (pass overwrite=True to clobber)")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=indent))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MacroxueTokenizerV2Spec":
        try:
            tokenizer_type = str(payload["tokenizer_type"])
            version = int(payload["version"])
            num_bins = int(payload["num_bins"])
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Missing required field {exc}") from exc

        vocab_obj = payload.get("vocab_order")
        if not isinstance(vocab_obj, Sequence):
            raise TypeError("vocab_order must be a sequence")
        vocab_order = [str(item) for item in vocab_obj]

        valuation_obj = payload.get("valuation_types")
        if not isinstance(valuation_obj, Sequence):
            raise TypeError("valuation_types must be a sequence")
        valuation_types = [str(item) for item in valuation_obj]

        metadata = payload.get("metadata")
        metadata_dict = dict(metadata) if isinstance(metadata, Mapping) else {}

        return cls(
            tokenizer_type=tokenizer_type,
            version=version,
            num_bins=num_bins,
            vocab_order=vocab_order,
            valuation_types=valuation_types,
            search=MacroxueTokenizerV2TypeConfig.from_dict(
                _require_mapping(payload, "search")
            ),
            tuple10=MacroxueTokenizerV2TypeConfig.from_dict(
                _require_mapping(payload, "tuple10")
            ),
            tuple11=MacroxueTokenizerV2TypeConfig.from_dict(
                _require_mapping(payload, "tuple11")
            ),
            metadata=metadata_dict,
        )

    @classmethod
    def from_json(cls, path: Path) -> "MacroxueTokenizerV2Spec":
        return cls.from_dict(json.loads(path.read_text()))


class MacroxueTokenizerV2:
    """Vectorised helper that maps branch EVs to discrete class IDs."""

    def __init__(self, spec: MacroxueTokenizerV2Spec, *, zero_tolerance: float = ZERO_TOLERANCE):
        self.spec = spec
        self.zero_tolerance = float(zero_tolerance)

        self.num_bins = int(spec.num_bins)
        self.token_illegal = 0
        self.token_failure = 1
        self.token_offset = 2
        self.token_winner = self.token_offset + self.num_bins

        self._bin_edges = {
            "search": np.asarray(spec.search.bin_edges, dtype=np.float64),
            "tuple10": np.asarray(spec.tuple10.bin_edges, dtype=np.float64),
            "tuple11": np.asarray(spec.tuple11.bin_edges, dtype=np.float64),
        }
        self._search_cutoff = spec.search.failure_cutoff
        if self._search_cutoff is None:
            raise ValueError("Search cutoff must be present in the spec")

    def encode_row(
        self,
        valuation_type: str,
        branch_evs: Sequence[float],
        move_dir: int,
        legal_mask: Sequence[bool] | np.ndarray,
        *,
        board_eval: Optional[int] = None,
    ) -> np.ndarray:
        """Return per-branch class IDs for a single decision.

        Parameters
        ----------
        valuation_type:
            One of ``search`` / ``tuple10`` / ``tuple11``.
        branch_evs:
            Iterable of four branch EV values in UDLR order.
        move_dir:
            Index of the winning action in UDLR order.
        legal_mask:
            Boolean iterable of length four indicating legal moves.
        board_eval:
            Required for search valuations; ignored otherwise.
        """
        valuation_type = str(valuation_type)
        if valuation_type not in self._bin_edges:
            raise KeyError(f"Unknown valuation_type '{valuation_type}'")

        branch_values = np.asarray(branch_evs, dtype=np.float64)
        if branch_values.shape != (NUM_BRANCHES,):
            raise ValueError("branch_evs must have length 4 in UDLR order")

        legal = np.asarray(legal_mask, dtype=bool)
        if legal.shape != (NUM_BRANCHES,):
            raise ValueError("legal_mask must have length 4")

        tokens = np.full(NUM_BRANCHES, self.token_illegal, dtype=np.int64)
        if move_dir < 0 or move_dir >= NUM_BRANCHES:
            raise ValueError(f"move_dir out of range: {move_dir}")

        tokens[legal] = self.token_failure
        tokens[move_dir] = self.token_winner

        if not legal[move_dir]:
            raise ValueError("Winner branch marked illegal")

        edges = self._bin_edges[valuation_type]

        if valuation_type == "search":
            if board_eval is None:
                raise ValueError("board_eval is required for search tokenization")
            winner_val = float(branch_values[move_dir])
            scaled = np.rint(branch_values * SEARCH_VALUE_SCALE)
            winner_scaled = float(np.rint(winner_val * SEARCH_VALUE_SCALE))
            advantages = scaled - float(board_eval)
            failure_mask = advantages < float(self._search_cutoff)
            failure_mask[move_dir] = False
            tokens[failure_mask & legal] = self.token_failure

            advantages = scaled - float(board_eval)
            winner_adv = advantages[move_dir]
            # Negative (or zero) delta: candidate advantage minus winner advantage.
            disadvantages = advantages - winner_adv
            zero_mask = np.isclose(disadvantages, 0.0, atol=0.5)
            zero_mask[move_dir] = False
            zero_mask |= ~legal
            remainder = ~(failure_mask | zero_mask)
            remainder[move_dir] = False

            zero_bins = self.token_offset + (self.num_bins - 1)
            tokens[zero_mask & legal] = zero_bins

            if np.any(remainder & legal):
                values = disadvantages[remainder & legal]
                idx = _searchsorted_bins(edges, values)
                tokens[np.where(remainder & legal)[0]] = self.token_offset + idx
            return tokens

        winner_val = branch_values[move_dir]
        zero_bins = self.token_offset + (self.num_bins - 1)
        failure_mask = (branch_values <= self.zero_tolerance) & legal
        failure_mask[move_dir] = False
        tokens[failure_mask] = self.token_failure

        disadvantages = branch_values - winner_val
        zero_mask = np.isclose(disadvantages, 0.0, atol=self.zero_tolerance)
        zero_mask[move_dir] = False
        zero_mask |= ~legal
        tokens[zero_mask] = zero_bins

        remainder_mask = legal & ~failure_mask & ~zero_mask
        remainder_mask[move_dir] = False
        if np.any(remainder_mask):
            values = disadvantages[remainder_mask]
            idx = _searchsorted_bins(edges, values)
            tokens[np.where(remainder_mask)[0]] = self.token_offset + idx
        return tokens


def fit_macroxue_tokenizer_v2(
    dataset_dir: Path | str,
    *,
    num_bins: int,
    search_failure_cutoff: int,
    zero_tolerance: float = ZERO_TOLERANCE,
    show_progress: bool = True,
) -> MacroxueTokenizerV2Spec:
    """Fit quantile bins for the advantage-based tokenizer."""
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory '{root}' does not exist")

    shard_paths = _discover_shards(root)
    if not shard_paths:
        raise FileNotFoundError(f"No steps.npy or steps-*.npy files found under {root}")

    valuation_lookup = _load_valuation_lookup(root)
    try:
        search_id = _lookup_valuation_id(valuation_lookup, "search")
        tuple10_id = _lookup_valuation_id(valuation_lookup, "tuple10")
        tuple11_id = _lookup_valuation_id(valuation_lookup, "tuple11")
    except KeyError as exc:
        raise KeyError(f"Valuation type {exc} missing from valuation_types.json") from exc

    stats = {
        "search": _TypeStats(),
        "tuple10": _TypeStats(),
        "tuple11": _TypeStats(),
    }
    search_deltas: List[np.ndarray] = []
    tuple10_deltas: List[np.ndarray] = []
    tuple11_deltas: List[np.ndarray] = []

    iterator: Iterable[Path]
    if show_progress and tqdm is not None:
        iterator = tqdm(shard_paths, desc="Fitting tokenizer v2", unit="shard")
    else:
        iterator = shard_paths

    for shard_path in iterator:
        shard = np.load(str(shard_path), mmap_mode="r")
        dtype_names = shard.dtype.names or ()
        if "board_eval" not in dtype_names:
            raise KeyError(
                f"Shard {shard_path} is missing 'board_eval'. "
                "Re-pack the dataset with the updated packer."
            )

        valuation_type = shard["valuation_type"].astype(np.int64, copy=False)
        branch_evs = shard["branch_evs"].astype(np.float64, copy=False)
        move_dir = shard["move_dir"].astype(np.int64, copy=False)
        legal_bits = shard["ev_legal"].astype(np.uint8, copy=False)
        board_eval = shard["board_eval"].astype(np.int64, copy=False)

        legal_mask = _decode_legal_mask(legal_bits)

        _accumulate_search(
            valuation_type,
            branch_evs,
            move_dir,
            legal_mask,
            board_eval,
            search_id,
            search_failure_cutoff,
            search_deltas,
            stats["search"],
        )
        _accumulate_tuple(
            valuation_type,
            branch_evs,
            move_dir,
            legal_mask,
            tuple10_id,
            zero_tolerance,
            tuple10_deltas,
            stats["tuple10"],
        )
        _accumulate_tuple(
            valuation_type,
            branch_evs,
            move_dir,
            legal_mask,
            tuple11_id,
            zero_tolerance,
            tuple11_deltas,
            stats["tuple11"],
        )

    search_values = _concatenate_or_raise(search_deltas, "search", num_bins)
    tuple10_values = _concatenate_or_raise(tuple10_deltas, "tuple10", num_bins)
    tuple11_values = _concatenate_or_raise(tuple11_deltas, "tuple11", num_bins)

    search_edges = _compute_bin_edges(search_values, num_bins, "search")
    tuple10_edges = _compute_bin_edges(tuple10_values, num_bins, "tuple10")
    tuple11_edges = _compute_bin_edges(tuple11_values, num_bins, "tuple11")

    vocab_order = ["ILLEGAL", "FAILURE"] + [f"BIN_{i}" for i in range(num_bins)] + ["WINNER"]
    metadata = {
        "dataset_dir": str(root),
        "num_bins": num_bins,
        "search": stats["search"].to_metadata(),
        "tuple10": stats["tuple10"].to_metadata(),
        "tuple11": stats["tuple11"].to_metadata(),
        "search_value_scale": SEARCH_VALUE_SCALE,
    }

    return MacroxueTokenizerV2Spec(
        tokenizer_type="macroxue_ev_advantage_v2",
        version=2,
        num_bins=num_bins,
        vocab_order=vocab_order,
        valuation_types=["search", "tuple10", "tuple11"],
        search=MacroxueTokenizerV2TypeConfig(
            bin_edges=list(map(float, search_edges)),
            failure_cutoff=int(search_failure_cutoff),
        ),
        tuple10=MacroxueTokenizerV2TypeConfig(bin_edges=list(map(float, tuple10_edges))),
        tuple11=MacroxueTokenizerV2TypeConfig(bin_edges=list(map(float, tuple11_edges))),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    obj = payload.get(key)
    if not isinstance(obj, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return obj


def _discover_shards(root: Path) -> List[Path]:
    shard_paths = sorted(root.glob("steps-*.npy"))
    if shard_paths:
        return shard_paths
    single = root / "steps.npy"
    if single.is_file():
        return [single]
    return []


def _load_valuation_lookup(root: Path) -> Dict[int, str]:
    path = root / "valuation_types.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing valuation_types.json in {root}")
    payload = json.loads(path.read_text())
    if isinstance(payload, Mapping):
        items = ((int(v), str(k)) for k, v in payload.items())
    elif isinstance(payload, Sequence):
        items = ((idx, str(name)) for idx, name in enumerate(payload))
    else:  # pragma: no cover - defensive
        raise TypeError("valuation_types.json must contain a mapping or list")
    lookup: Dict[int, str] = {}
    for idx, name in items:
        lookup[idx] = name
    return lookup


def _lookup_valuation_id(lookup: Mapping[int, str], target: str) -> int:
    for idx, name in lookup.items():
        if name == target:
            return int(idx)
    raise KeyError(target)


def _decode_legal_mask(bits: np.ndarray) -> np.ndarray:
    return ((bits[:, None] >> BRANCH_AXIS) & 1).astype(bool)


def _accumulate_search(
    valuation_type: np.ndarray,
    branch_evs: np.ndarray,
    move_dir: np.ndarray,
    legal_mask: np.ndarray,
    board_eval: np.ndarray,
    search_id: int,
    search_failure_cutoff: int,
    accumulator: List[np.ndarray],
    stats: "_TypeStats",
) -> None:
    mask = valuation_type == search_id
    if not np.any(mask):
        return
    rows = branch_evs[mask]
    winners = move_dir[mask]
    legal = legal_mask[mask]
    board_scores = board_eval[mask]

    stats.rows += int(rows.shape[0])
    stats.legal_branches += int(np.sum(legal)) - int(np.sum(legal[np.arange(rows.shape[0]), winners]))

    scaled = np.rint(rows * SEARCH_VALUE_SCALE)
    advantages = scaled - board_scores[:, None]

    failure = (advantages < float(search_failure_cutoff)) & legal
    failure[np.arange(rows.shape[0]), winners] = False
    stats.failure_branches += int(failure.sum())

    winner_adv = advantages[np.arange(rows.shape[0]), winners]
    disadvantages = advantages - winner_adv[:, None]

    zero_mask = np.isclose(disadvantages, 0.0, atol=0.5)
    zero_mask[np.arange(rows.shape[0]), winners] = False
    zero_mask &= legal
    zero_mask &= ~failure
    stats.zero_disadvantage += int(zero_mask.sum())

    remainder = legal & ~failure & ~zero_mask
    remainder[np.arange(rows.shape[0]), winners] = False
    if not np.any(remainder):
        return
    accumulator.append(disadvantages[remainder].astype(np.float64, copy=False))


def _accumulate_tuple(
    valuation_type: np.ndarray,
    branch_evs: np.ndarray,
    move_dir: np.ndarray,
    legal_mask: np.ndarray,
    target_id: int,
    zero_tolerance: float,
    accumulator: List[np.ndarray],
    stats: "_TypeStats",
) -> None:
    mask = valuation_type == target_id
    if not np.any(mask):
        return
    rows = branch_evs[mask]
    winners = move_dir[mask]
    legal = legal_mask[mask]

    stats.rows += int(rows.shape[0])
    stats.legal_branches += int(np.sum(legal)) - int(np.sum(legal[np.arange(rows.shape[0]), winners]))

    winner_vals = rows[np.arange(rows.shape[0]), winners]
    disadvantages = rows - winner_vals[:, None]

    failure = (rows <= zero_tolerance) & legal
    failure[np.arange(rows.shape[0]), winners] = False
    stats.failure_branches += int(failure.sum())

    zero_mask = np.isclose(disadvantages, 0.0, atol=zero_tolerance)
    zero_mask[np.arange(rows.shape[0]), winners] = False
    zero_mask &= legal
    zero_mask &= ~failure
    stats.zero_disadvantage += int(zero_mask.sum())

    remainder = legal & ~failure & ~zero_mask
    remainder[np.arange(rows.shape[0]), winners] = False
    if not np.any(remainder):
        return
    accumulator.append(disadvantages[remainder].astype(np.float64, copy=False))


def _concatenate_or_raise(acc: List[np.ndarray], label: str, num_bins: int) -> np.ndarray:
    if not acc:
        raise ValueError(f"No samples collected for {label}; cannot fit tokenizer")
    values = np.concatenate(acc)
    negative = values[values < 0]
    if negative.size < num_bins:
        raise ValueError(
            f"Insufficient negative disadvantages for {label}: "
            f"needed >= {num_bins}, found {negative.size}"
        )
    return negative


def _compute_bin_edges(values: np.ndarray, num_bins: int, label: str) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(values, quantiles, method="linear")
    edges = np.asarray(edges, dtype=np.float64)
    # Ensure monotonic increase; final edge anchored at zero.
    edges[-1] = max(edges[-1], 0.0)
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    if edges[0] > 0:
        edges[0] = min(edges[0], 0.0)
    if not np.all(np.diff(edges) > 0):
        raise ValueError(f"Failed to create strictly increasing edges for {label}")
    return edges


def _searchsorted_bins(edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, values, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return idx.astype(np.int64, copy=False)


@dataclass
class _TypeStats:
    rows: int = 0
    legal_branches: int = 0
    failure_branches: int = 0
    zero_disadvantage: int = 0

    def to_metadata(self) -> Dict[str, int]:
        return {
            "rows": self.rows,
            "legal_branches": self.legal_branches,
            "failure_branches": self.failure_branches,
            "zero_disadvantage": self.zero_disadvantage,
        }
