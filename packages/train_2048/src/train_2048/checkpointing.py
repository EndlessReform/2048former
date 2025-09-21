from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import json
import time

import torch
from safetensors.torch import save_file as safe_save_file

from .config import normalize_state_dict_keys


def create_run_dir(base_dir: str) -> Path:
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def dump_training_and_model_config(run_dir: Path, training_cfg: object, model: torch.nn.Module) -> None:
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


def save_safetensors(model: torch.nn.Module, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    state = normalize_state_dict_keys(raw)
    safe_save_file(state, str(path), metadata={"format": "pt"})
    return path


def maybe_save_stable(model: torch.nn.Module, run_dir: Path, *, global_step: int, decay_steps: int, decay_start: int, preflag: dict) -> None:
    if preflag.get("saved", False):
        return
    if decay_steps <= 0:
        return
    if global_step == decay_start:
        save_safetensors(model, run_dir / "model-stable.safetensors")
        preflag["saved"] = True


def maybe_save_best(
    *,
    model: torch.nn.Module,
    run_dir: Path,
    evaluate_fn,
    dl_val,
    device,
    target_mode: str,
    cfg_checkpoint,
    step: int,
    epoch: Optional[int],
    best_tracker: Dict[str, float],
    wandb_run: Optional[object] = None,
):
    if cfg_checkpoint is None or cfg_checkpoint.save_best_every_steps is None or dl_val is None:
        return
    interval = int(cfg_checkpoint.save_best_every_steps)
    if step <= 0 or (step % interval) != 0:
        return
    val_metrics = evaluate_fn(model, dl_val, device, target_mode)
    val_loss = float(val_metrics["loss"])
    if val_loss + float(cfg_checkpoint.best_min_delta) < best_tracker.get("best_val_loss", float("inf")):
        best_tracker["best_val_loss"] = val_loss
        save_safetensors(model, run_dir / "model-best.safetensors")
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
        }
        path = run_dir / f"dangerous-step-{step:08d}.pt"
        try:
            torch.save(payload, str(path))
            print(f"[ckpt] Dumped dangerous pt checkpoint: {path}")
        except Exception as e:
            print(f"[ckpt] Failed to write dangerous pt checkpoint at step {step}: {e}")


__all__ = [
    "create_run_dir",
    "dump_training_and_model_config",
    "save_safetensors",
    "maybe_save_stable",
    "maybe_save_best",
    "dangerous_dump_pt",
]

