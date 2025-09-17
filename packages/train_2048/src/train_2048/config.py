from __future__ import annotations

from typing import Literal, Optional

import json
from pathlib import Path

import tomllib
from pydantic import BaseModel, Field, field_validator
from core_2048 import Encoder, EncoderConfig, load_encoder_from_init, normalize_state_dict_keys
from .binning import BinningConfig


class TargetConfig(BaseModel):
    """Configure which supervision target to use during training."""

    mode: Literal["binned_ev", "hard_move"] = "binned_ev"


def _find_repo_root() -> Path:
    """Return the repository root by walking upwards.

    Prefer a directory that contains a VCS marker ('.git') or monorepo markers
    like 'packages' or top-level 'datasets'. Fallback to the old heuristic
    (parent of 'src') if nothing matches.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists() or (parent / "packages").exists() or (parent / "datasets").exists():
            return parent
    # Fallback: parent of 'src'
    src_dir = here.parent.parent
    return src_dir.parent


class DatasetConfig(BaseModel):
    # Root directory containing steps.npy and metadata.db
    dataset_dir: str = "./dataset"
    # Optional: SQL to define the universe of runs (training+validation)
    run_sql: Optional[str] = None
    sql_params: list[object] = Field(default_factory=list)

    # Validation split options (by run to avoid leakage)
    # Option A: explicit SQL for validation runs
    val_run_sql: Optional[str] = None
    val_sql_params: list[object] = Field(default_factory=list)
    # Option B: random run split when >0; ignored if val_run_sql provided
    val_run_pct: float = 0.0
    val_split_seed: int = 42

    # Choose either fixed steps or epochs. If both provided, steps takes priority.
    num_steps: Optional[int] = None
    num_epochs: Optional[int] = None
    # Run validation every N training steps (when >0)
    val_every: int = 1000

    @field_validator("num_steps", "num_epochs")
    @classmethod
    def _non_negative(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("num_steps/num_epochs must be >= 0")
        return v

    @field_validator("val_run_pct")
    @classmethod
    def _pct_range(cls, v: float) -> float:
        if v < 0.0 or v >= 1.0:
            if v != 0.0:
                raise ValueError("dataset.val_run_pct must be in [0,1) (0 disables)")
        return v

    @field_validator("val_every")
    @classmethod
    def _val_every_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("dataset.val_every must be >= 0 (0 disables)")
        return v

    def resolved_dataset_dir(self) -> str:
        p = Path(self.dataset_dir)
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


class CheckpointConfig(BaseModel):
    # Save an epoch checkpoint at this cadence (keep all)
    every_epochs: Optional[int] = 1
    # Optionally evaluate and save a single best checkpoint every N steps
    save_best_every_steps: Optional[int] = None
    # Minimum improvement required to update the best
    best_min_delta: float = 0.0

    @field_validator("every_epochs", "save_best_every_steps")
    @classmethod
    def _non_negative_or_none(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("checkpoint intervals must be > 0 when provided")
        return v

    @field_validator("best_min_delta")
    @classmethod
    def _delta_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("best_min_delta must be >= 0")
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
    target: TargetConfig = Field(default_factory=TargetConfig)
    # Discretization for EV heads
    binning: BinningConfig = Field(default_factory=BinningConfig)
    # Dataset input
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    # Checkpointing
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

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


__all__ = [
    "WandbConfig",
    "LRScheduleConfig",
    "OptimizerConfig",
    "HyperParams",
    "BatchConfig",
    "DropoutConfig",
    "TargetConfig",
    "CheckpointConfig",
    "TrainingConfig",
    "load_config",
    "load_encoder_from_init",
    "normalize_state_dict_keys",
]
