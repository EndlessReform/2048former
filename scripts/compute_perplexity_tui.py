#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen, Screen
from textual.widgets import Footer, Header, Input, Static

from core_2048 import load_encoder_from_init
from train_2048.binning import BinningConfig
from train_2048.dataloader.steps import StepsDataset
from train_2048.tokenization.base import BoardCodec
from train_2048.tokenization.ev_binning import EVBinnerTokenizer
from train_2048.tokenization.macroxue import MacroxueTokenizerV2, MacroxueTokenizerV2Spec

DIR_NAMES = ("Up", "Down", "Left", "Right")
DIR_SHORT = ("U", "D", "L", "R")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TUI for compute_perplexity per-board outputs.")
    parser.add_argument("--per-board", required=True, help="Path to per-board .npz from compute_perplexity.py")
    parser.add_argument("--init", default=None, help="Checkpoint dir or bundle (defaults to metadata).")
    parser.add_argument("--dataset", default=None, help="Steps dataset dir (defaults to metadata).")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu). Defaults to cuda if available.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random selection.")
    return parser.parse_args()


def _parse_metadata(raw: np.ndarray | None) -> dict[str, object]:
    if raw is None:
        return {}
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {}


def _load_binning_config(init_path: str) -> BinningConfig:
    init = Path(init_path)
    if init.is_file():
        init = init.parent
    cfg_path = init / "training-config.json"
    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        binning = payload.get("binning")
        if isinstance(binning, dict):
            return BinningConfig(**binning)
    return BinningConfig()


def _load_training_target_mode(init_path: str) -> str | None:
    init = Path(init_path)
    if init.is_file():
        init = init.parent
    cfg_path = init / "training-config.json"
    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        target = payload.get("target", {})
        mode = target.get("mode")
        if isinstance(mode, str):
            return mode
    return None


def _load_macroxue_tokenizer(init_path: str) -> tuple[MacroxueTokenizerV2, int] | None:
    init = Path(init_path)
    if init.is_file():
        init = init.parent
    tok_path = init / "tokenizer.json"
    if not tok_path.is_file():
        return None
    with tok_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("tokenizer_type") != "macroxue_ev_advantage_v2":
        return None
    spec = MacroxueTokenizerV2Spec.from_dict(payload)
    tokenizer = MacroxueTokenizerV2(spec)
    n_classes = len(spec.vocab_order)
    return tokenizer, n_classes


def _load_valuation_type_mapping(dataset_dir: str) -> dict[int, str]:
    vt_path = Path(dataset_dir) / "valuation_types.json"
    if vt_path.is_file():
        with vt_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return {i: name for i, name in enumerate(payload)}
        if isinstance(payload, dict):
            keys_are_ints = all(str(k).isdigit() for k in payload.keys())
            if keys_are_ints:
                return {int(k): str(v) for k, v in payload.items()}
            return {int(v): str(k) for k, v in payload.items()}
        raise TypeError("Unsupported valuation_types.json format")
    return {0: "search", 1: "tuple10", 2: "tuple11"}


def _bin_centers_from_config(cfg: BinningConfig) -> np.ndarray:
    edges = np.asarray(cfg.edges, dtype=np.float64)
    if cfg.strategy == "edges":
        base_centers = 0.5 * (edges[:-1] + edges[1:])
        if cfg.special_zero_one:
            return np.concatenate(([0.0], base_centers, [1.0]))
        return base_centers
    # upper_bounds
    centers = []
    for i, upper in enumerate(edges):
        lower = 0.0 if i == 0 else edges[i - 1]
        centers.append(0.5 * (lower + upper))
    return np.asarray(centers, dtype=np.float64)


def _decode_board(packed: np.uint64, mask: Optional[np.uint16]) -> np.ndarray:
    packed_arr = np.asarray([packed], dtype=np.uint64)
    mask_arr = None if mask is None else np.asarray([mask], dtype=np.uint16)
    exps = BoardCodec.decode_packed_board_to_exps_u8(packed_arr, mask65536=mask_arr)[0]
    exps = exps.astype(np.int64, copy=False)
    return np.where(exps == 0, 0, 1 << exps).reshape(4, 4)


def _decode_board_exps(packed: np.uint64, mask: Optional[np.uint16]) -> np.ndarray:
    packed_arr = np.asarray([packed], dtype=np.uint64)
    mask_arr = None if mask is None else np.asarray([mask], dtype=np.uint16)
    return BoardCodec.decode_packed_board_to_exps_u8(packed_arr, mask65536=mask_arr)[0]


def _render_board_ascii(tiles: np.ndarray) -> str:
    cell_w = 6
    border = "+" + "+".join(["-" * cell_w] * 4) + "+"
    lines = [border]
    for row in tiles:
        row_text = "|".join(f"{(val if val > 0 else '.'):^{cell_w}}" for val in row)
        lines.append(f"|{row_text}|")
        lines.append(border)
    return "\n".join(lines)


def _safe_float(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.6f}"


def _format_dir_values(values: Iterable[float | int | None]) -> str:
    parts = []
    for name, val in zip(DIR_SHORT, values):
        if val is None:
            parts.append(f"{name}: n/a")
        elif isinstance(val, float):
            parts.append(f"{name}: {val:.4f}")
        else:
            parts.append(f"{name}: {val}")
    return "  ".join(parts)


def _argmax_legal(values: np.ndarray, legal: np.ndarray) -> int | None:
    if not np.any(legal):
        return None
    masked = np.where(legal, values, -np.inf)
    return int(np.argmax(masked))


def _summary_stats(values: np.ndarray, loss: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "p5": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "mean_nll": float("nan"),
        }
    percentiles = np.percentile(values, [5, 10, 25, 50, 75, 90, 95])
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p5": float(percentiles[0]),
        "p10": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p90": float(percentiles[5]),
        "p95": float(percentiles[6]),
        "mean_nll": float(np.mean(loss)) if loss.size > 0 else float("nan"),
    }


@dataclass(frozen=True)
class SidecarData:
    indices: np.ndarray
    loss: np.ndarray
    perplexity: np.ndarray
    count: np.ndarray
    board: np.ndarray
    mask: Optional[np.ndarray]
    metadata: dict[str, object]

    @classmethod
    def load(cls, path: str) -> "SidecarData":
        payload = np.load(path)
        metadata = _parse_metadata(payload.get("metadata"))
        mask = payload.get("tile_65536_mask")
        return cls(
            indices=payload["indices"].astype(np.int64, copy=False),
            loss=payload["loss"].astype(np.float32, copy=False),
            perplexity=payload["perplexity"].astype(np.float32, copy=False),
            count=payload["count"].astype(np.int64, copy=False),
            board=payload["board"].astype(np.uint64, copy=False),
            mask=mask.astype(np.uint16, copy=False) if mask is not None else None,
            metadata=metadata,
        )


@dataclass
class BoardStats:
    teacher_bins: Optional[np.ndarray] = None
    student_bins: Optional[np.ndarray] = None
    teacher_favored: Optional[int] = None
    student_favored: Optional[int] = None
    teacher_evs: Optional[np.ndarray] = None
    student_evs: Optional[np.ndarray] = None
    legal: Optional[np.ndarray] = None
    match: Optional[bool] = None


class PromptScreen(ModalScreen[Optional[str]]):
    def __init__(self, title: str, placeholder: str, initial: str = "") -> None:
        super().__init__()
        self._title = title
        self._placeholder = placeholder
        self._initial = initial

    def compose(self) -> ComposeResult:
        with Container(id="prompt"):
            yield Static(self._title, id="prompt-title")
            yield Input(value=self._initial, placeholder=self._placeholder, id="prompt-input")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() if event.value is not None else None)

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)


class HelpScreen(Screen):
    BINDINGS = [("q", "app.pop_screen", "Close")]

    def compose(self) -> ComposeResult:
        help_text = "\n".join(
            [
                "Vim-style navigation:",
                "  h / left  - previous board",
                "  l / right - next board",
                "  j / down  - next board",
                "  k / up    - previous board",
                "  g         - jump to dataset index (prefix with # for position)",
                "  G         - jump to last board",
                "  r         - random board",
                "  s         - set RNG seed",
                "  n         - next mismatch (student vs teacher)",
                "  p         - previous mismatch",
                "  F5        - hardest boards",
                "  F6        - easiest boards",
                "  F7        - summary stats",
                "  ?         - this help",
                "  q         - quit",
            ]
        )
        with Container(id="help"):
            yield Static(help_text, id="help-text")


class RankingScreen(ModalScreen[Optional[int]]):
    BINDINGS = [
        ("q", "close", "Close"),
        ("escape", "close", "Close"),
        ("j", "down", "Down"),
        ("k", "up", "Up"),
        ("g", "first", "First"),
        ("G", "last", "Last"),
        ("enter", "select", "Select"),
    ]

    def __init__(self, title: str, positions: np.ndarray, data: SidecarData, *, limit: int = 100) -> None:
        super().__init__()
        self._title = title
        self._positions = positions[:limit]
        self._data = data
        self._limit = limit
        self._cursor = 0

    def compose(self) -> ComposeResult:
        with Container(id="ranking"):
            yield Static(self._title, id="ranking-title")
            yield Static("", id="ranking-list")

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        lines = []
        for i, pos in enumerate(self._positions):
            idx = int(self._data.indices[pos])
            ppl = float(self._data.perplexity[pos])
            marker = ">" if i == self._cursor else " "
            lines.append(f"{marker} {i+1:>3}  pos={int(pos):>6}  idx={idx:>8}  ppl={ppl:>10.6f}")
        if not lines:
            lines = ["(no entries)"]
        self.query_one("#ranking-list", Static).update("\n".join(lines))

    def action_close(self) -> None:
        self.dismiss(None)

    def action_down(self) -> None:
        if self._cursor < len(self._positions) - 1:
            self._cursor += 1
            self._refresh()

    def action_up(self) -> None:
        if self._cursor > 0:
            self._cursor -= 1
            self._refresh()

    def action_first(self) -> None:
        self._cursor = 0
        self._refresh()

    def action_last(self) -> None:
        if len(self._positions) > 0:
            self._cursor = len(self._positions) - 1
            self._refresh()

    def action_select(self) -> None:
        if len(self._positions) == 0:
            self.dismiss(None)
            return
        self.dismiss(int(self._positions[self._cursor]))


class SummaryScreen(ModalScreen[None]):
    BINDINGS = [
        ("q", "close", "Close"),
        ("escape", "close", "Close"),
        ("0", "filter_all", "All"),
        ("1", "filter_search", "Search"),
        ("2", "filter_tuple10", "Tuple10"),
        ("3", "filter_tuple11", "Tuple11"),
        ("a", "head_all", "HeadAll"),
        ("u", "head_up", "HeadUp"),
        ("d", "head_down", "HeadDown"),
        ("l", "head_left", "HeadLeft"),
        ("r", "head_right", "HeadRight"),
    ]

    def __init__(
        self,
        *,
        data: SidecarData,
        n_classes: Optional[int],
        vt_by_position: Optional[np.ndarray],
        head_by_position: Optional[np.ndarray],
        stats_cache: dict[str, dict[str, float]],
    ) -> None:
        super().__init__()
        self._data = data
        self._n_classes = n_classes
        self._vt_by_position = vt_by_position
        self._head_by_position = head_by_position
        self._stats_cache = stats_cache
        self._filter = "all"
        self._head_filter = "all"

    def compose(self) -> ComposeResult:
        with Container(id="summary"):
            yield Static("", id="summary-text", markup=True)

    def on_mount(self) -> None:
        self._refresh()

    def _stats_for_filter(self, label: str, head: str) -> dict[str, float]:
        cache_key = f"{label}|{head}"
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]
        mask = np.ones_like(self._data.perplexity, dtype=bool)
        if label != "all" and self._vt_by_position is not None:
            mask &= self._vt_by_position == label
        if head != "all" and self._head_by_position is not None:
            mask &= self._head_by_position == head
        ppl = self._data.perplexity[mask]
        loss = self._data.loss[mask]
        stats = _summary_stats(ppl, loss)
        self._stats_cache[cache_key] = stats
        return stats

    def _refresh(self) -> None:
        stats = self._stats_for_filter(self._filter, self._head_filter)
        if self._filter == "all" and self._head_filter == "all":
            sample_count = int(self._data.perplexity.size)
        else:
            mask = np.ones_like(self._data.perplexity, dtype=bool)
            if self._filter != "all" and self._vt_by_position is not None:
                mask &= self._vt_by_position == self._filter
            if self._head_filter != "all" and self._head_by_position is not None:
                mask &= self._head_by_position == self._head_filter
            sample_count = int(mask.sum())
        classes = self._n_classes if self._n_classes is not None else "n/a"
        title = f"[bold cyan]SUMMARY[/]  [bold]type={self._filter}[/]  [bold]head={self._head_filter}[/]"
        header = f"{'Metric':<12} {'PPL':>14}"
        divider = "-" * max(len(title), len(header), 36)
        rows = [
            ("min", stats["min"], None),
            ("p5", stats["p5"], None),
            ("p10", stats["p10"], None),
            ("p25", stats["p25"], None),
            ("p50", stats["p50"], None),
            ("p75", stats["p75"], None),
            ("p90", stats["p90"], None),
            ("p95", stats["p95"], None),
            ("max", stats["max"], None),
        ]
        lines = [
            title,
            divider,
            f"[bold]Samples[/]: {sample_count:<8}  [bold]Classes[/]: {classes}",
            f"[bold]Mean NLL[/]: {_safe_float(stats['mean_nll'])}",
            "",
            f"[bold]{header}[/]",
            "-" * len(header),
        ]
        for label, ppl, _ in rows:
            lines.append(f"{label:<12} {_safe_float(ppl):>14}")
        lines.extend(
            [
                "",
                "[bold]Type[/]: 0=all 1=search 2=tuple10 3=tuple11",
                "[bold]Head[/]: a=all u=up d=down l=left r=right",
            ]
        )
        self.query_one("#summary-text", Static).update("\n".join(lines))

    def action_close(self) -> None:
        self.dismiss(None)

    def _set_filter(self, label: str) -> None:
        if label != "all" and self._vt_by_position is None:
            return
        self._filter = label
        self._refresh()

    def action_filter_all(self) -> None:
        self._set_filter("all")

    def action_filter_search(self) -> None:
        self._set_filter("search")

    def action_filter_tuple10(self) -> None:
        self._set_filter("tuple10")

    def action_filter_tuple11(self) -> None:
        self._set_filter("tuple11")

    def _set_head_filter(self, label: str) -> None:
        if label != "all" and self._head_by_position is None:
            return
        self._head_filter = label
        self._refresh()

    def action_head_all(self) -> None:
        self._set_head_filter("all")

    def action_head_up(self) -> None:
        self._set_head_filter("U")

    def action_head_down(self) -> None:
        self._set_head_filter("D")

    def action_head_left(self) -> None:
        self._set_head_filter("L")

    def action_head_right(self) -> None:
        self._set_head_filter("R")


class PerplexityApp(App):
    CSS = """
    #main {
        height: 1fr;
        padding: 1 2;
    }
    #board {
        width: 50%;
        padding-right: 2;
        border: solid gray;
        padding: 1 2;
    }
    #stats {
        width: 1fr;
        border: solid gray;
        padding: 1 2;
    }
    #prompt {
        width: 60%;
        margin: 1 2;
        padding: 1 2;
        border: solid gray;
        background: black;
    }
    #help {
        padding: 1 2;
    }
    #help-text {
        padding: 1;
        border: solid gray;
    }
    #ranking {
        padding: 1 2;
    }
    #ranking-title {
        padding-bottom: 1;
    }
    #summary {
        padding: 1 2;
        border: solid gray;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("?", "help", "Help"),
        ("h", "prev", "Prev"),
        ("left", "prev", "Prev"),
        ("k", "prev", "Prev"),
        ("up", "prev", "Prev"),
        ("l", "next", "Next"),
        ("right", "next", "Next"),
        ("j", "next", "Next"),
        ("down", "next", "Next"),
        ("g", "goto", "Goto"),
        ("G", "last", "Last"),
        ("r", "random", "Random"),
        ("s", "seed", "Seed"),
        ("n", "next_mismatch", "NextMismatch"),
        ("p", "prev_mismatch", "PrevMismatch"),
        ("f5", "hardest", "Hardest"),
        ("f6", "easiest", "Easiest"),
        ("f7", "summary", "Summary"),
    ]

    def __init__(self, *, data: SidecarData, init_path: Optional[str], dataset_path: Optional[str], device: str, seed: int):
        super().__init__()
        self.data = data
        self.position = 0
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.index_to_pos = {int(idx): pos for pos, idx in enumerate(self.data.indices)}

        self.init_path = init_path
        self.dataset_path = dataset_path
        self.device = torch.device(device)

        self.dataset: Optional[StepsDataset] = None
        self.model: Optional[torch.nn.Module] = None
        self.mode: Optional[str] = None
        self.n_classes: Optional[int] = None
        self.tokenizer: Optional[MacroxueTokenizerV2] = None
        self.ev_tokenizer: Optional[EVBinnerTokenizer] = None
        self.bin_centers: Optional[np.ndarray] = None
        self.vt_map: Optional[dict[int, str]] = None
        self.stats_cache: dict[int, BoardStats] = {}
        self._ranked_positions: Optional[np.ndarray] = None
        self._vt_by_position: Optional[np.ndarray] = None
        self._head_by_position: Optional[np.ndarray] = None
        self._summary_cache: dict[str, dict[str, float]] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            yield Static("", id="board", markup=True)
            yield Static("", id="stats", markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._init_data_sources()
        self._refresh()

    def _init_data_sources(self) -> None:
        if self.dataset_path:
            self.dataset = StepsDataset(self.dataset_path, mmap_mode=True)
        if self.init_path:
            self.model = load_encoder_from_init(self.init_path)
            self.model.to(self.device)
            self.model.eval()

        if self.init_path:
            self.mode = _load_training_target_mode(self.init_path)
            macroxue = _load_macroxue_tokenizer(self.init_path)
            if self.mode == "macroxue_tokens" or (self.mode is None and macroxue is not None):
                if macroxue is not None:
                    self.tokenizer, self.n_classes = macroxue
                if self.dataset_path:
                    self.vt_map = _load_valuation_type_mapping(self.dataset_path)
                self.mode = "macroxue_tokens"
            else:
                cfg = _load_binning_config(self.init_path)
                self.ev_tokenizer = EVBinnerTokenizer(cfg).to(self.device)
                self.bin_centers = _bin_centers_from_config(cfg)
                self.n_classes = int(self.ev_tokenizer.n_bins)
                self.mode = "binned_ev"
        else:
            self.mode = str(self.data.metadata.get("mode")) if self.data.metadata else None
            self.n_classes = int(self.data.metadata.get("classes", 0)) if self.data.metadata else None

        if self.dataset_path and self.vt_map is None:
            self.vt_map = _load_valuation_type_mapping(self.dataset_path)

    def _get_rows(self, dataset_index: int) -> Optional[np.ndarray]:
        if self.dataset is None:
            return None
        return self.dataset.get_rows(np.asarray([dataset_index], dtype=np.int64))

    def _compute_stats(self, position: int) -> BoardStats:
        dataset_index = int(self.data.indices[position])
        stats = BoardStats()
        rows = self._get_rows(dataset_index)
        if rows is None:
            return stats

        row = rows[0]
        if "branch_evs" in row.dtype.names:
            evs = row["branch_evs"].astype(np.float32, copy=False)
        elif "ev_values" in row.dtype.names:
            evs = row["ev_values"].astype(np.float32, copy=False)
        else:
            evs = None

        if "ev_legal" in row.dtype.names:
            legal = BoardCodec.legal_mask_from_bits_udlr(np.asarray([row["ev_legal"]]))[0]
        elif evs is not None:
            legal = np.isfinite(evs)
        else:
            legal = np.ones(4, dtype=bool)
        stats.legal = legal

        if self.mode == "binned_ev" and evs is not None and self.ev_tokenizer is not None:
            branch_values = torch.from_numpy(evs.copy()).to(device=self.device, dtype=torch.float32).unsqueeze(0)
            mask = torch.from_numpy(legal.astype(np.bool_, copy=False)).to(self.device).unsqueeze(0)
            targets = self.ev_tokenizer.build_targets(evs=branch_values, legal_mask=mask)["branch_bin_targets"][0]
            stats.teacher_bins = targets.cpu().numpy().astype(np.int64, copy=False)
            stats.teacher_evs = evs.astype(np.float32, copy=False)
            stats.teacher_favored = _argmax_legal(stats.teacher_evs, legal)
        elif self.mode == "macroxue_tokens" and self.tokenizer is not None:
            if evs is None:
                return stats
            if "valuation_type" not in row.dtype.names or "move_dir" not in row.dtype.names:
                return stats
            valuation_type = int(row["valuation_type"])
            move_dir = int(row["move_dir"])
            if "board_eval" in row.dtype.names:
                board_eval = int(row["board_eval"])
            else:
                from train_2048.tokenization.macroxue import board_eval as board_eval_mod

                mask65536 = row["tile_65536_mask"] if "tile_65536_mask" in row.dtype.names else None
                board_exps = _decode_board_exps(row["board"], mask65536)
                if hasattr(board_eval_mod, "evaluate_board_batch"):
                    board_eval = int(board_eval_mod.evaluate_board_batch(np.asarray([board_exps]))[0])
                else:
                    board_eval = int(board_eval_mod.evaluate(board_exps, {}))
            vt_name = self.vt_map.get(valuation_type) if self.vt_map else None
            if vt_name is None:
                return stats
            tokens = self.tokenizer.encode_row(
                valuation_type=vt_name,
                branch_evs=evs,
                move_dir=move_dir,
                legal_mask=legal,
                board_eval=board_eval,
            )
            stats.teacher_bins = tokens.astype(np.int64, copy=False)
            stats.teacher_favored = move_dir

        if self.model is None:
            return stats

        packed = self.data.board[position]
        mask_val = self.data.mask[position] if self.data.mask is not None else None
        board_exps = _decode_board_exps(packed, mask_val)
        tokens = torch.from_numpy(board_exps.reshape(1, 16)).to(device=self.device, dtype=torch.long)
        with torch.inference_mode():
            _hs, head_out = self.model(tokens)
        student_bins = []
        student_evs = []
        student_win_prob = []
        for h, logits_h in enumerate(head_out):
            logits_h = logits_h[0].float()
            student_bins.append(int(torch.argmax(logits_h).item()))
            probs = torch.softmax(logits_h, dim=-1)
            if self.mode == "binned_ev" and self.bin_centers is not None:
                centers = torch.as_tensor(self.bin_centers, device=probs.device, dtype=probs.dtype)
                student_evs.append(float((probs * centers).sum().item()))
            elif self.mode == "macroxue_tokens" and self.tokenizer is not None:
                student_win_prob.append(float(probs[self.tokenizer.token_winner].item()))
        stats.student_bins = np.asarray(student_bins, dtype=np.int64)
        if student_evs:
            stats.student_evs = np.asarray(student_evs, dtype=np.float32)
            stats.student_favored = _argmax_legal(stats.student_evs, legal)
        elif student_win_prob:
            stats.student_evs = np.asarray(student_win_prob, dtype=np.float32)
            stats.student_favored = _argmax_legal(stats.student_evs, legal)
        if stats.teacher_favored is not None and stats.student_favored is not None:
            stats.match = stats.teacher_favored == stats.student_favored
        return stats

    def _get_stats(self, position: int) -> BoardStats:
        if position not in self.stats_cache:
            self.stats_cache[position] = self._compute_stats(position)
        return self.stats_cache[position]

    def _refresh(self) -> None:
        board_widget = self.query_one("#board", Static)
        stats_widget = self.query_one("#stats", Static)
        packed = self.data.board[self.position]
        mask_val = self.data.mask[self.position] if self.data.mask is not None else None
        tiles = _decode_board(packed, mask_val)
        board_widget.update("[bold cyan]BOARD[/]\n" + _render_board_ascii(tiles))

        stats = self._get_stats(self.position)
        dataset_index = int(self.data.indices[self.position])
        loss = float(self.data.loss[self.position])
        ppl = float(self.data.perplexity[self.position])
        count = int(self.data.count[self.position])

        lines = [
            "[bold magenta]STATS[/]",
            f"[bold]Position[/]: {self.position + 1} / {len(self.data.indices)}",
            f"[bold]Dataset index[/]: {dataset_index}",
            f"[bold]Loss[/]: {_safe_float(loss)}",
            f"[bold]Perplexity[/]: {_safe_float(ppl)}",
            f"[bold]Legal count[/]: {count}",
            f"[bold]Mode[/]: {self.mode or 'unknown'}  [bold]Classes[/]: {self.n_classes or 'n/a'}",
            f"[bold]Seed[/]: {self.seed}",
            "",
        ]
        if stats.legal is not None:
            legal_dirs = [DIR_SHORT[i] for i, ok in enumerate(stats.legal) if ok]
            lines.append(f"[bold]Legal moves[/]: {' '.join(legal_dirs) if legal_dirs else 'none'}")
        if stats.teacher_bins is not None:
            lines.append(f"[bold]Teacher bins[/]: {_format_dir_values(stats.teacher_bins.tolist())}")
        if stats.student_bins is not None:
            lines.append(f"[bold]Student bins[/]: {_format_dir_values(stats.student_bins.tolist())}")
        if stats.teacher_evs is not None:
            lines.append(f"[bold]Teacher EVs[/]: {_format_dir_values(stats.teacher_evs.tolist())}")
        if stats.student_evs is not None:
            label = "Student EVs" if self.mode == "binned_ev" else "Student win p"
            lines.append(f"[bold]{label}[/]: {_format_dir_values(stats.student_evs.tolist())}")
        if stats.teacher_favored is not None:
            lines.append(f"[bold]Teacher favored[/]: {DIR_NAMES[stats.teacher_favored]}")
        if stats.student_favored is not None:
            lines.append(f"[bold]Student favored[/]: {DIR_NAMES[stats.student_favored]}")
        if stats.match is not None:
            verdict = "[green]yes[/]" if stats.match else "[red]no[/]"
            lines.append(f"[bold]Match[/]: {verdict}")
        stats_widget.update("\n".join(lines))

    def _ranked(self) -> np.ndarray:
        if self._ranked_positions is None:
            self._ranked_positions = np.argsort(self.data.perplexity, kind="mergesort")
        return self._ranked_positions

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_next(self) -> None:
        if self.position < len(self.data.indices) - 1:
            self.position += 1
            self._refresh()

    def action_prev(self) -> None:
        if self.position > 0:
            self.position -= 1
            self._refresh()

    def action_last(self) -> None:
        self.position = len(self.data.indices) - 1
        self._refresh()

    def _goto_position(self, pos: int) -> None:
        if 0 <= pos < len(self.data.indices):
            self.position = pos
            self._refresh()
        else:
            self.notify(f"Position out of range: {pos}", severity="warning")

    def action_goto(self) -> None:
        async def _handle(result: Optional[str]) -> None:
            if not result:
                return
            if result.startswith("#"):
                try:
                    pos = int(result[1:])
                except ValueError:
                    self.notify("Invalid position format", severity="warning")
                    return
                self._goto_position(pos)
                return
            try:
                idx = int(result)
            except ValueError:
                self.notify("Invalid index format", severity="warning")
                return
            pos = self.index_to_pos.get(idx)
            if pos is None:
                self.notify("Dataset index not found in sidecar", severity="warning")
                return
            self._goto_position(pos)

        self.push_screen(PromptScreen("Jump to dataset index (#pos for position)", "e.g. 12345 or #50"), _handle)

    def action_seed(self) -> None:
        async def _handle(result: Optional[str]) -> None:
            if result is None or result == "":
                return
            try:
                seed = int(result)
            except ValueError:
                self.notify("Seed must be an integer", severity="warning")
                return
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            self._refresh()

        self.push_screen(PromptScreen("Set RNG seed", "integer seed", str(self.seed)), _handle)

    def action_random(self) -> None:
        pos = int(self.rng.integers(0, len(self.data.indices)))
        self._goto_position(pos)

    def _find_mismatch(self, start: int, step: int) -> Optional[int]:
        pos = start + step
        while 0 <= pos < len(self.data.indices):
            stats = self._get_stats(pos)
            if stats.match is False:
                return pos
            pos += step
        return None

    def action_next_mismatch(self) -> None:
        pos = self._find_mismatch(self.position, 1)
        if pos is None:
            self.notify("No next mismatch found", severity="warning")
            return
        self._goto_position(pos)

    def action_prev_mismatch(self) -> None:
        pos = self._find_mismatch(self.position, -1)
        if pos is None:
            self.notify("No previous mismatch found", severity="warning")
            return
        self._goto_position(pos)

    def action_hardest(self) -> None:
        positions = self._ranked()[::-1]

        async def _handle(result: Optional[int]) -> None:
            if result is not None:
                self._goto_position(result)

        self.push_screen(RankingScreen("Hardest boards by perplexity", positions, self.data), _handle)

    def action_easiest(self) -> None:
        positions = self._ranked()

        async def _handle(result: Optional[int]) -> None:
            if result is not None:
                self._goto_position(result)

        self.push_screen(RankingScreen("Easiest boards by perplexity", positions, self.data), _handle)

    def _load_valuation_types(self) -> Optional[np.ndarray]:
        if self.dataset is None or self.vt_map is None:
            return None
        if self._vt_by_position is not None:
            return self._vt_by_position

        vt_arr: list[str] = []
        indices = self.data.indices.astype(np.int64, copy=False)
        batch_size = 2048
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            rows = self.dataset.get_rows(batch)
            if "valuation_type" not in rows.dtype.names:
                self._vt_by_position = None
                return None
            vt_ids = rows["valuation_type"].astype(np.int64, copy=False)
            for vt in vt_ids.tolist():
                vt_arr.append(self.vt_map.get(int(vt), "unknown"))
        self._vt_by_position = np.asarray(vt_arr, dtype=object)
        return self._vt_by_position

    def _load_head_dirs(self) -> Optional[np.ndarray]:
        if self.dataset is None:
            return None
        if self._head_by_position is not None:
            return self._head_by_position

        head_arr: list[str] = []
        indices = self.data.indices.astype(np.int64, copy=False)
        batch_size = 2048
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            rows = self.dataset.get_rows(batch)
            if "move_dir" in rows.dtype.names:
                dirs = rows["move_dir"].astype(np.int64, copy=False)
            elif "move" in rows.dtype.names:
                dirs = rows["move"].astype(np.int64, copy=False)
            else:
                self._head_by_position = None
                return None
            for d in dirs.tolist():
                if 0 <= int(d) < 4:
                    head_arr.append(DIR_SHORT[int(d)])
                else:
                    head_arr.append("unknown")
        self._head_by_position = np.asarray(head_arr, dtype=object)
        return self._head_by_position

    def action_summary(self) -> None:
        vt_by_position = self._load_valuation_types()
        head_by_position = self._load_head_dirs()
        self.push_screen(
            SummaryScreen(
                data=self.data,
                n_classes=self.n_classes,
                vt_by_position=vt_by_position,
                head_by_position=head_by_position,
                stats_cache=self._summary_cache,
            )
        )


def main() -> None:
    args = parse_args()
    sidecar = SidecarData.load(args.per_board)
    metadata = sidecar.metadata
    init_path = args.init or (str(metadata.get("checkpoint")) if metadata else None)
    dataset_path = args.dataset or (str(metadata.get("dataset")) if metadata else None)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    app = PerplexityApp(
        data=sidecar,
        init_path=init_path,
        dataset_path=dataset_path,
        device=device,
        seed=args.seed,
    )
    app.run()


if __name__ == "__main__":
    main()
