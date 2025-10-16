from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json
import time

import torch
from safetensors.torch import save_file as safe_save_file

from .config import normalize_state_dict_keys

CHECKPOINT_METADATA_VERSION = 2


@dataclass
class ResumeState:
    global_step: Optional[int]
    bundle_path: Path
    optimizer_loaded: bool
    bundle_metadata: Dict[str, Any]


def save_pt_bundle(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    training_cfg: Optional[object],
    global_step: Optional[int],
    resume_state: Optional[Dict[str, Any]] = None,
    dataset_metadata: Optional[Dict[str, Any]] = None,
    metadata_version: int = CHECKPOINT_METADATA_VERSION,
) -> Path:
    """Persist a torch ``.pt`` bundle with weights, optimizer state, and metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "global_step": (int(global_step) if global_step is not None else None),
        "model": normalize_state_dict_keys(model.state_dict()),
        "optimizer": (optimizer.state_dict() if optimizer is not None else None),
        "encoder_config": getattr(model, "config", None).model_dump() if getattr(model, "config", None) is not None else None,
        "training_config": training_cfg.model_dump() if training_cfg is not None and hasattr(training_cfg, "model_dump") else None,
        "metadata_version": int(metadata_version),
    }
    if dataset_metadata is not None:
        payload["dataset"] = dataset_metadata
    if resume_state is not None:
        payload["resume"] = resume_state
    torch.save(payload, str(path))
    return path


def maybe_resume_optimizer_from_init(
    init_dir: str,
    optimizer: torch.optim.Optimizer,
    *,
    bundle_path: Optional[str] = None,
) -> Optional[ResumeState]:
    """Attempt to resume optimizer state from the provided init source.

    Returns a :class:`ResumeState` carrying metadata even when no optimizer state is available.
    """
    init_path = Path(init_dir)
    if bundle_path is not None:
        candidates = [Path(bundle_path)]
    elif init_path.is_file():
        candidates = [init_path]
    else:
        candidates = [
            init_path / "model-stable.pt",
            init_path / "model.pt",
        ]

    for pt in candidates:
        if not pt.is_file():
            continue
        try:
            bundle = torch.load(str(pt), map_location="cpu")
        except Exception as e:
            print(f"[resume] Failed to load checkpoint bundle {pt}: {e}")
            return None
        if not isinstance(bundle, dict):
            continue
        optimizer_loaded = False
        opt_blob = bundle.get("optimizer")
        if isinstance(opt_blob, dict):
            try:
                optimizer.load_state_dict(opt_blob)  # type: ignore[arg-type]
                optimizer_loaded = True
                print(f"[resume] Loaded optimizer state from {pt}")
            except Exception as e:
                print(f"[resume] Failed to load optimizer from {pt}: {e}")
                optimizer_loaded = False
        else:
            print(f"[resume] No optimizer state found in {pt}; continuing without optimizer resume")
        metadata = {k: v for k, v in bundle.items() if k not in ("model", "optimizer")}
        gs = metadata.get("global_step")
        try:
            gs_int = int(gs) if gs is not None else None
        except Exception:
            gs_int = None
        return ResumeState(global_step=gs_int, bundle_path=pt, optimizer_loaded=optimizer_loaded, bundle_metadata=metadata)
    return None


def create_run_dir(base_dir: str) -> Path:
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def dump_training_and_model_config(run_dir: Path, training_cfg: object, model: torch.nn.Module) -> None:
    import shutil
    try:
        with (run_dir / "training-config.json").open("w", encoding="utf-8") as f:
            json.dump(training_cfg.model_dump(), f, indent=2)
    except Exception:
        pass
    try:
        enc_cfg = getattr(model, "config", None)
        if enc_cfg is not None:
            with (run_dir / "config.json").open("w", encoding="utf-8") as f:
                json.dump(enc_cfg.model_dump(), f, indent=2)
    except Exception:
        pass

    # If using a macroxue tokenizer, copy the spec to the run dir for reproducibility.
    try:
        if (
            hasattr(training_cfg, "target")
            and getattr(training_cfg.target, "mode", "") == "macroxue_tokens"
        ):
            if hasattr(training_cfg, "dataset") and hasattr(
                training_cfg.dataset, "resolved_tokenizer_path"
            ):
                tokenizer_path_str = training_cfg.dataset.resolved_tokenizer_path()
                if tokenizer_path_str:
                    tokenizer_path = Path(tokenizer_path_str)
                    if tokenizer_path.is_file():
                        shutil.copy(tokenizer_path, run_dir / "tokenizer.json")
    except Exception:
        pass


def save_safetensors(model: torch.nn.Module, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    state = normalize_state_dict_keys(raw)
    safe_save_file(state, str(path), metadata={"format": "pt"})
    return path


def _save_stable_bundle(
    run_dir: Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    training_cfg: Optional[object],
    global_step: Optional[int],
    resume_state: Optional[Dict[str, Any]],
    dataset_metadata: Optional[Dict[str, Any]],
    metadata_version: int,
) -> None:
    """Write both safetensors and a .pt bundle for stable checkpoint.

    The .pt includes optimizer state to support resumption (e.g., WSD/Muon/AdamW).
    """
    # Primary weights (safetensors)
    save_safetensors(model, run_dir / "model-stable.safetensors")
    # PT bundle with optimizer + configs for easy resume
    try:
        save_pt_bundle(
            run_dir / "model-stable.pt",
            model=model,
            optimizer=optimizer,
            training_cfg=training_cfg,
            global_step=global_step,
            resume_state=resume_state,
            dataset_metadata=dataset_metadata,
            metadata_version=metadata_version,
        )
    except Exception:
        pass


def maybe_save_stable(
    model: torch.nn.Module,
    run_dir: Path,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    training_cfg: Optional[object] = None,
    global_step: int,
    decay_steps: int,
    decay_start: int,
    preflag: dict,
    resume_state: Optional[Dict[str, Any]] = None,
    dataset_metadata: Optional[Dict[str, Any]] = None,
    metadata_version: int = CHECKPOINT_METADATA_VERSION,
) -> None:
    if preflag.get("saved", False):
        return
    if decay_steps <= 0:
        return
    if global_step == decay_start:
        _save_stable_bundle(
            run_dir,
            model=model,
            optimizer=optimizer,
            training_cfg=training_cfg,
            global_step=global_step,
            resume_state=resume_state,
            dataset_metadata=dataset_metadata,
            metadata_version=metadata_version,
        )
        preflag["saved"] = True


def maybe_save_best(
    *,
    model: torch.nn.Module,
    run_dir: Path,
    evaluate_fn,
    dl_val,
    device,
    cfg_checkpoint,
    step: int,
    epoch: Optional[int],
    best_tracker: Dict[str, float],
    optimizer: Optional[torch.optim.Optimizer],
    training_cfg: Optional[object],
    wandb_run: Optional[object] = None,
    resume_state: Optional[Dict[str, Any]] = None,
    dataset_metadata: Optional[Dict[str, Any]] = None,
    metadata_version: int = CHECKPOINT_METADATA_VERSION,
):
    if cfg_checkpoint is None or cfg_checkpoint.save_best_every_steps is None or dl_val is None:
        return
    interval = int(cfg_checkpoint.save_best_every_steps)
    if step <= 0 or (step % interval) != 0:
        return
    # `evaluate_fn` is expected to be a bound method like `objective.evaluate`,
    # which takes (model, dl_val, device). Do not pass extra args.
    val_metrics = evaluate_fn(model, dl_val, device)
    val_loss = float(val_metrics["loss"])
    if val_loss + float(cfg_checkpoint.best_min_delta) < best_tracker.get("best_val_loss", float("inf")):
        best_tracker["best_val_loss"] = val_loss
        try:
            save_pt_bundle(
                run_dir / "model-best.pt",
                model=model,
                optimizer=optimizer,
                training_cfg=training_cfg,
                global_step=step,
                resume_state=resume_state,
                dataset_metadata=dataset_metadata,
                metadata_version=metadata_version,
            )
        except Exception as e:
            print(f"[ckpt] Failed to write best checkpoint at step {step}: {e}")
        else:
            print(f"[ckpt] Saved new best checkpoint at step {step}: {run_dir / 'model-best.pt'}")
        if wandb_run is not None:
            try:
                import wandb  # type: ignore

                wandb.log({"best/val_loss": val_loss, "best/epoch": (epoch if epoch is not None else -1)}, step=step)
            except Exception:
                pass


def dangerous_dump_pt(*, cfg, run_dir: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
    if not getattr(cfg, "dangerous_just_checkpoint", False):
        return
    if step <= 0:
        return
    if step < 1000 or (step % 100_000) == 0:
        payload = {
            "global_step": int(step),
            "model": normalize_state_dict_keys(model.state_dict()),
            "optimizer": optimizer.state_dict(),
            "encoder_config": getattr(model, "config", None).model_dump() if getattr(model, "config", None) is not None else None,
            "training_config": cfg.model_dump(),
            "metadata_version": CHECKPOINT_METADATA_VERSION,
        }
        path = run_dir / f"dangerous-step-{step:08d}.pt"
        try:
            torch.save(payload, str(path))
            print(f"[ckpt] Dumped dangerous pt checkpoint: {path}")
        except Exception as e:
            print(f"[ckpt] Failed to write dangerous pt checkpoint at step {step}: {e}")


def maybe_save_pt_interval(
    *,
    model: torch.nn.Module,
    run_dir: Path,
    optimizer: Optional[torch.optim.Optimizer],
    training_cfg: Optional[object],
    step: int,
    interval: Optional[int],
    resume_state: Optional[Dict[str, Any]] = None,
    dataset_metadata: Optional[Dict[str, Any]] = None,
    metadata_version: int = CHECKPOINT_METADATA_VERSION,
) -> None:
    if interval is None or interval <= 0:
        return
    if step <= 0 or (step % interval) != 0:
        return
    path = run_dir / f"model-step-{step:08d}.pt"
    try:
        save_pt_bundle(
            path,
            model=model,
            optimizer=optimizer,
            training_cfg=training_cfg,
            global_step=step,
            resume_state=resume_state,
            dataset_metadata=dataset_metadata,
            metadata_version=metadata_version,
        )
        print(f"[ckpt] Saved step checkpoint: {path}")
    except Exception as e:
        print(f"[ckpt] Failed to write step checkpoint at step {step}: {e}")


__all__ = [
    "create_run_dir",
    "dump_training_and_model_config",
    "save_safetensors",
    "save_pt_bundle",
    "maybe_save_stable",
    "maybe_save_best",
    "dangerous_dump_pt",
    "maybe_resume_optimizer_from_init",
    "maybe_save_pt_interval",
    "ResumeState",
]
