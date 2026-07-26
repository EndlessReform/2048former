from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from .checkpointing import ResumeState, maybe_resume_optimizer_from_init
from .config import TrainingConfig


@dataclass
class DataCursorTracker:
    """Track prefetched, pending, and optimizer-committed data positions."""

    committed: Optional[dict] = None
    _pending: Optional[dict] = None

    def begin_step(self) -> None:
        self._pending = self.committed

    def observe_microbatch(self, cursor: object) -> None:
        if isinstance(cursor, dict):
            self._pending = cursor

    def commit_step(self) -> Optional[dict]:
        self.committed = self._pending
        return self.committed


def _dump_config_section(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _restart_contract(cfg: TrainingConfig) -> dict:
    return {
        "training_seed": int(cfg.seed),
        "target_steps": cfg.dataset.num_steps,
        "learning_rate": float(cfg.hyperparameters.learning_rate),
        "lr_schedule": _dump_config_section(cfg.hyperparameters.lr_schedule),
        "optimizer": _dump_config_section(cfg.hyperparameters.optimizer),
        "adaptive_batch": _dump_config_section(cfg.batch.adaptive),
        "rotation_augment": _dump_config_section(cfg.dataset.rotation_augment),
        "flip_augment": _dump_config_section(cfg.dataset.flip_augment),
    }


def extract_resume_metadata(init_info: dict[str, Any]) -> tuple[dict, dict, Optional[int]]:
    """Extract resume-related metadata from an init bundle info dict."""
    weight_type = init_info.get("weights_type", "unknown")
    bundle_meta = init_info.get("bundle_metadata") if weight_type == "pt" else {}
    if not isinstance(bundle_meta, dict):
        bundle_meta = {}
    resume_payload_meta = bundle_meta.get("resume") if isinstance(bundle_meta.get("resume"), dict) else {}
    resume_dataset_meta = bundle_meta.get("dataset") if isinstance(bundle_meta.get("dataset"), dict) else {}
    resume_global_step_meta = bundle_meta.get("global_step") if isinstance(bundle_meta.get("global_step"), int) else bundle_meta.get("global_step")
    return resume_payload_meta, resume_dataset_meta, resume_global_step_meta


def resolve_resume_skip_samples(
    cfg: TrainingConfig,
    dataset_signature: dict,
    dataset_fingerprint: str,
    *,
    resume_payload_meta: dict,
    resume_dataset_meta: dict,
    resume_global_step_meta: Optional[int],
) -> int:
    """Compute how many samples to skip when resuming from a checkpoint."""
    effective_batch_size_cfg = int(cfg.batch.batch_size)
    resume_samples_consumed = None
    if isinstance(resume_payload_meta, dict) and "samples_consumed" in resume_payload_meta:
        try:
            resume_samples_consumed = int(resume_payload_meta["samples_consumed"])
        except Exception:
            resume_samples_consumed = None
    if resume_samples_consumed is None and resume_global_step_meta is not None:
        try:
            resume_samples_consumed = int(resume_global_step_meta) * effective_batch_size_cfg
        except Exception:
            resume_samples_consumed = None

    fingerprint_loaded = resume_dataset_meta.get("fingerprint") if isinstance(resume_dataset_meta, dict) else None
    dataset_dirs_match = (
        isinstance(resume_dataset_meta, dict)
        and resume_dataset_meta.get("dataset_dir")
        and resume_dataset_meta.get("dataset_dir") == dataset_signature["dataset_dir"]
    )
    dataset_match = bool(fingerprint_loaded) and fingerprint_loaded == dataset_fingerprint
    if not dataset_match and fingerprint_loaded is None:
        dataset_match = dataset_dirs_match

    resume_skip_samples = int(resume_samples_consumed or 0) if dataset_match else 0
    if resume_skip_samples > 0 and dataset_match:
        approx_steps = resume_skip_samples / max(1, effective_batch_size_cfg)
        print(f"[resume] Resuming data stream after {resume_skip_samples:,} samples (~{approx_steps:,.0f} steps)")
    elif (resume_samples_consumed or 0) > 0 and not dataset_match:
        print(
            "[resume] Dataset metadata mismatch between checkpoint and current configuration; "
            "training data will restart from the beginning."
        )
    return resume_skip_samples


def resolve_resume_data_cursor(
    dataset_signature: dict,
    dataset_fingerprint: str,
    *,
    resume_payload_meta: dict,
    resume_dataset_meta: dict,
) -> Optional[dict]:
    """Return a committed data cursor only when the dataset identity matches."""
    fingerprint_loaded = resume_dataset_meta.get("fingerprint")
    dataset_match = bool(fingerprint_loaded) and fingerprint_loaded == dataset_fingerprint
    if fingerprint_loaded is None:
        dataset_match = (
            resume_dataset_meta.get("dataset_dir") == dataset_signature["dataset_dir"]
        )
    cursor = resume_payload_meta.get("data_cursor")
    if isinstance(cursor, dict) and not dataset_match:
        raise ValueError(
            "Checkpoint data cursor does not match the configured dataset and run split"
        )
    return cursor if isinstance(cursor, dict) else None


def validate_resume_configuration(
    cfg: TrainingConfig,
    resume_payload_meta: dict,
) -> None:
    """Reject incompatible settings for an exact cursor-based restart."""
    if int(resume_payload_meta.get("version", 0) or 0) < 2:
        return
    expected = {
        "effective_batch_size": int(cfg.batch.batch_size),
        "micro_batch_size": int(cfg.batch.physical_batch_size()),
        "grad_accum_steps": int(cfg.batch.grad_accum_steps()),
    }
    mismatches = [
        f"{key}: checkpoint={resume_payload_meta.get(key)} current={value}"
        for key, value in expected.items()
        if int(resume_payload_meta.get(key, -1)) != value
    ]
    if mismatches:
        raise ValueError(
            "Exact restart configuration mismatch: " + "; ".join(mismatches)
        )
    checkpoint_contract = resume_payload_meta.get("restart_contract")
    current_contract = _restart_contract(cfg)
    if checkpoint_contract != current_contract:
        raise ValueError(
            "Exact restart training/sampling contract differs from the checkpoint"
        )


def resolve_resume_bundle_path(init_info: dict[str, Any], cfg: TrainingConfig) -> tuple[Optional[Path], str]:
    """Select the checkpoint bundle used to resume optimizer state when possible."""
    weight_type = init_info.get("weights_type", "unknown")
    available_pt = init_info.get("available_pt", [])
    bundle_path_from_weights = init_info.get("weights_path") if weight_type == "pt" else None

    resolved_init_path = init_info.get("resolved_init_path", cfg.init_dir)
    init_dir_path = Path(resolved_init_path)
    bundle_path_for_resume = None
    if bundle_path_from_weights:
        bundle_path_for_resume = Path(bundle_path_from_weights)
    elif init_dir_path.is_file() and init_dir_path.suffix.lower() in {".pt", ".pth"}:
        bundle_path_for_resume = init_dir_path
    else:
        if available_pt:
            print(
                "[resume] WARNING: PT bundle(s) found alongside safetensors "
                f"{available_pt}, but weights were loaded from safetensors. "
                "Optimizer state will NOT be resumed to avoid desync."
            )
    return bundle_path_for_resume, resolved_init_path


def maybe_resume_optimizer_state(
    init_dir: str,
    optimizer: torch.optim.Optimizer,
    bundle_path: Optional[Path],
) -> Optional[ResumeState]:
    """Attempt to resume optimizer state from the chosen bundle path."""
    if bundle_path is None:
        return None
    try:
        return maybe_resume_optimizer_from_init(init_dir, optimizer, bundle_path=str(bundle_path))
    except Exception as exc:
        print(f"[resume] Optimizer resume attempt failed: {exc}")
        return None


def build_resume_state(
    global_step: int,
    samples_consumed: int,
    skip_samples: int,
    cfg: TrainingConfig,
    *,
    data_cursor: Optional[dict] = None,
) -> dict:
    """Serialize resume data for checkpoint bundles."""
    state = {
        "version": 2,
        "global_step": int(global_step),
        "samples_consumed": int(samples_consumed),
        "skip_samples": int(skip_samples),
        "effective_batch_size": int(cfg.batch.batch_size),
        "micro_batch_size": int(cfg.batch.physical_batch_size()),
        "grad_accum_steps": int(cfg.batch.grad_accum_steps()),
        "restart_contract": _restart_contract(cfg),
    }
    if data_cursor is not None:
        state["data_cursor"] = data_cursor
    return state


__all__ = [
    "extract_resume_metadata",
    "resolve_resume_skip_samples",
    "resolve_resume_data_cursor",
    "validate_resume_configuration",
    "resolve_resume_bundle_path",
    "maybe_resume_optimizer_state",
    "build_resume_state",
    "DataCursorTracker",
]
