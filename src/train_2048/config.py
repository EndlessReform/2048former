from __future__ import annotations

from typing import Literal, Optional

import json
from pathlib import Path

import tomllib
from pydantic import BaseModel, Field, field_validator
from safetensors.torch import load_file as safe_load_file
from .model import Encoder, EncoderConfig
from .binning import BinningConfig


def _find_repo_root() -> Path:
    # Heuristic: repo root is the parent of 'src'
    here = Path(__file__).resolve()
    # e.g., .../repo/src/train_2048/config.py -> repo
    src_dir = here.parent.parent  # .../repo/src
    root = src_dir.parent
    return root


class DatasetConfig(BaseModel):
    packfile: str = "./datasets/dsv1.a2pack"
    # Choose either fixed steps or epochs. If both provided, steps takes priority.
    num_steps: Optional[int] = None
    num_epochs: Optional[int] = None

    @field_validator("num_steps", "num_epochs")
    @classmethod
    def _non_negative(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("num_steps/num_epochs must be >= 0")
        return v

    def resolved_packfile(self) -> str:
        p = Path(self.packfile)
        if not p.is_absolute():
            p = _find_repo_root() / p
        return str(p)


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "train-2048"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = Field(default_factory=list)
    mode: Literal["online", "offline", "disabled"] = "disabled"
    # Log metrics to W&B every N steps (>=1). Defaults to every step.
    report_every: int = 1

    @field_validator("mode")
    @classmethod
    def _sync_mode_with_enabled(cls, v: str, info):
        # If not explicitly enabled, default to disabled regardless of mode.
        data = info.data or {}
        if not data.get("enabled", False):
            return "disabled"
        return v

    @field_validator("report_every")
    @classmethod
    def _report_every_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("wandb.report_every must be >= 1")
        return v


class LRScheduleConfig(BaseModel):
    name: Literal["constant", "warmup-stable-decay"] = "constant"
    # Only used for warmup-stable-decay
    warmup_steps: int = 0
    decay_steps: int = 0
    # Optional: percent of total steps used for cooldown/decay (0..1).
    # If provided (>0), takes precedence over decay_steps and is computed
    # once total training steps are known.
    cooldown_pct: float | None = None
    min_lr_ratio: float = 0.1  # final_lr = base_lr * min_lr_ratio

    @field_validator("warmup_steps", "decay_steps")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("steps must be >= 0")
        return v

    @field_validator("min_lr_ratio")
    @classmethod
    def _ratio_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("min_lr_ratio must be between 0 and 1")
        return v

    @field_validator("cooldown_pct")
    @classmethod
    def _pct_range(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not (0.0 <= v <= 1.0):
            raise ValueError("cooldown_pct must be between 0 and 1")
        return v


class OptimizerConfig(BaseModel):
    name: Literal["adamw", "muon"] = "adamw"
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


class HyperParams(BaseModel):
    learning_rate: float = 3e-4
    muon_lr: Optional[float] = None  # only used if optimizer=muon
    lr_schedule: LRScheduleConfig = Field(default_factory=LRScheduleConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)


class BatchConfig(BaseModel):
    batch_size: int = 1024
    micro_batch_size: Optional[int] = None

    @field_validator("batch_size")
    @classmethod
    def _batch_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("batch_size must be > 0")
        return v

    @field_validator("micro_batch_size")
    @classmethod
    def _micro_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("micro_batch_size must be > 0 when provided")
        return v

    def grad_accum_steps(self) -> int:
        if not self.micro_batch_size:
            return 1
        # Ceil division; training loop may handle last partial microbatch
        return (self.batch_size + self.micro_batch_size - 1) // self.micro_batch_size


class DropoutConfig(BaseModel):
    dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.0

    @field_validator("dropout_prob", "attention_dropout_prob")
    @classmethod
    def _in_unit_interval(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("dropout values must be in [0, 1]")
        return v


class TrainingConfig(BaseModel):
    # IO
    init_dir: str
    checkpoint_dir: str

    # Sections
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    hyperparameters: HyperParams = Field(default_factory=HyperParams)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    dropout: DropoutConfig = Field(default_factory=DropoutConfig)
    # Discretization for EV heads
    binning: BinningConfig = Field(default_factory=BinningConfig)
    # Dataset input
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)

    # Misc
    seed: int = 0

    @classmethod
    def from_toml(cls, path: str) -> "TrainingConfig":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data)


def load_config(path: str) -> TrainingConfig:
    """Load training config from a TOML file."""
    return TrainingConfig.from_toml(path)


def load_encoder_from_init(init_dir: str) -> Encoder:
    """
    Construct an Encoder from an init folder.

    Expects:
    - `<init_dir>/config.json`: JSON matching EncoderConfig fields
    - Optional `<init_dir>/model.safetensors`: weights to load

    Returns a model on CPU with random weights if the safetensors file is absent.
    """
    init_path = Path(init_dir)
    cfg_path = init_path / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing init config: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        enc_cfg_dict = json.load(f)

    enc_cfg = EncoderConfig.model_validate(enc_cfg_dict)
    model = Encoder(enc_cfg)

    weights_path = init_path / "model.safetensors"
    if weights_path.is_file():
        state = safe_load_file(str(weights_path))
        state = normalize_state_dict_keys(state)
        try:
            model.load_state_dict(state, strict=True)
        except Exception as e:
            # Fallback to non-strict if keys still mismatch after normalization
            missing, unexpected = _compare_state_keys(model, state)
            print(
                "Warning: non-strict load due to key mismatch. "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
            model.load_state_dict(state, strict=False)

    return model


__all__ = [
    "WandbConfig",
    "LRScheduleConfig",
    "OptimizerConfig",
    "HyperParams",
    "BatchConfig",
    "DropoutConfig",
    "TrainingConfig",
    "load_config",
    "load_encoder_from_init",
    "normalize_state_dict_keys",
]


def normalize_state_dict_keys(state: dict) -> dict:
    """
    Strip known wrapper prefixes from state_dict keys (e.g., `_orig_mod.`, `module.`).

    Returns a new dict with normalized keys.
    """
    prefixes = ("_orig_mod.", "module.")
    out = {}
    for k, v in state.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p) :]
                    changed = True
        out[nk] = v
    return out


def _compare_state_keys(model: Encoder, state: dict) -> tuple[list[str], list[str]]:
    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())
    missing = sorted(list(model_keys - state_keys))
    unexpected = sorted(list(state_keys - model_keys))
    return missing, unexpected
