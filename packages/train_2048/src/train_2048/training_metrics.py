from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .objectives import Objective


def format_postfix(
    metrics: dict[str, float | list[float] | None],
    lr: float,
    target_mode: str,
    *,
    global_step: Optional[int] = None,
    accum_steps: Optional[int] = None,
    micro_batch_size: Optional[int] = None,
    dt_data_ms: float | None = None,
    dt_comp_ms: float | None = None,
) -> str:
    """Format the tqdm postfix for the current training step."""
    loss = float(metrics.get("loss", 0.0))
    parts: list[str] = []
    if global_step is not None:
        parts.append(f"step={global_step}")
    base = f"loss={loss:.4f}"
    if target_mode in ("binned_ev", "macroxue_tokens"):
        if target_mode == "macroxue_tokens":
            pa = metrics.get("policy_agreement")
            if pa is None:
                pa = metrics.get("policy_agree")
            if pa is not None:
                base += f"  agree={float(pa) * 100:.1f}%"
    else:
        acc = metrics.get("policy_accuracy")
        if acc is None:
            acc = metrics.get("policy_acc")
        if acc is not None:
            base += f"  policy_acc={float(acc):.3f}"
    base += f"  lr={lr:.2e}"
    if accum_steps is not None and micro_batch_size is not None:
        effective = int(accum_steps) * int(micro_batch_size)
        parts.append(f"mb={int(micro_batch_size)} eff={effective}")
    if dt_data_ms is not None and dt_comp_ms is not None:
        base += f"  data={dt_data_ms:.1f}ms  comp={dt_comp_ms:.1f}ms"
    if parts:
        return f"{' '.join(parts)}  {base}"
    return base


def wandb_log(data: dict[str, float | int], step: int) -> None:
    try:
        import wandb  # type: ignore

        wandb.log(data, step=step)
    except Exception:
        pass


def build_train_payload(
    metrics: dict[str, float | list[float] | None],
    lr: float,
    target_mode: str,
    *,
    epoch: Optional[int],
    dt_data_ms: float,
    dt_comp_ms: float,
    effective_batch_size: Optional[int],
    accum_steps: Optional[int],
) -> dict[str, float | int]:
    """Build the wandb payload for a training step."""
    payload: dict[str, float | int] = {
        "train/loss": float(metrics["loss"]),
        "train/lr": float(lr),
        "train/data_time_ms": float(dt_data_ms),
        "train/compute_time_ms": float(dt_comp_ms),
    }
    if epoch is not None:
        payload["train/epoch"] = int(epoch)
    if target_mode in ("binned_ev", "macroxue_tokens"):
        hl = metrics["head_losses"]
        payload.update(
            {
                "train/loss_u": float(hl[0]),
                "train/loss_d": float(hl[1]),
                "train/loss_l": float(hl[2]),
                "train/loss_r": float(hl[3]),
            }
        )
        agreement = metrics.get("policy_agreement")
        if agreement is None:
            agreement = metrics.get("policy_agree")
        if agreement is not None:
            payload["train/policy_agreement"] = float(agreement)
    else:
        # Hard target path (e.g., hard_move): log canonical policy accuracy only.
        acc = metrics.get("policy_accuracy")
        if acc is None:
            acc = metrics.get("policy_acc")
        if acc is not None:
            payload["train/policy_accuracy"] = float(acc)
    if effective_batch_size is not None:
        payload["train/effective_batch_size"] = int(effective_batch_size)
    if accum_steps is not None:
        payload["train/accum_steps"] = int(accum_steps)
    return payload


def build_val_payload(
    metrics: dict[str, float | list[float] | None],
    target_mode: str,
    *,
    epoch: Optional[int],
) -> dict[str, float | int]:
    """Build the wandb payload for a validation step."""
    payload: dict[str, float | int] = {"val/loss": float(metrics["loss"])}
    if epoch is not None:
        payload["train/epoch"] = int(epoch)
    if target_mode in ("binned_ev", "macroxue_tokens"):
        hl = metrics["head_losses"]
        payload.update(
            {
                "val/loss_u": float(hl[0]),
                "val/loss_d": float(hl[1]),
                "val/loss_l": float(hl[2]),
                "val/loss_r": float(hl[3]),
            }
        )
        agreement = metrics.get("policy_agreement")
        if agreement is None:
            agreement = metrics.get("policy_agree")
        if agreement is not None:
            payload["val/policy_agreement"] = float(agreement)
    else:
        # Hard target path (e.g., hard_move): log canonical policy accuracy only.
        acc = metrics.get("policy_accuracy")
        if acc is None:
            acc = metrics.get("policy_acc")
        if acc is not None:
            payload["val/policy_accuracy"] = float(acc)
    return payload


def maybe_log_val(
    objective: Objective,
    model: torch.nn.Module,
    dl_val: Optional[DataLoader],
    device: torch.device,
    *,
    cfg: TrainingConfig,
    target_mode: str,
    step: int,
    wandb_run: Optional[object],
    epoch: Optional[int],
) -> Optional[dict[str, float | list[float] | None]]:
    """Run and optionally log validation metrics when configured."""
    if dl_val is None:
        return None
    if (cfg.dataset.val_every or 0) <= 0:
        return None
    if step <= 0 or (step % int(cfg.dataset.val_every)) != 0:
        return None
    val_metrics = objective.evaluate(model, dl_val, device)
    if wandb_run is not None:
        wandb_log(build_val_payload(val_metrics, target_mode, epoch=epoch), step=step)
    return val_metrics


def accumulate_metric_sums(
    sums: dict[str, list[float] | float],
    counts: dict[str, int],
    metrics: dict[str, float | list[float] | None],
) -> None:
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, list):
            arr = [float(x) for x in value]
            prev = sums.get(key)
            if prev is None or not isinstance(prev, list):
                sums[key] = arr
            else:
                if len(prev) != len(arr):
                    raise ValueError(f"Metric list length mismatch for '{key}'")
                sums[key] = [p + a for p, a in zip(prev, arr)]
            counts[key] = counts.get(key, 0) + 1
        else:
            sums[key] = float(sums.get(key, 0.0)) + float(value)
            counts[key] = counts.get(key, 0) + 1


def finalize_metric_sums(
    sums: dict[str, list[float] | float],
    counts: dict[str, int],
    last_metrics: dict[str, float | list[float] | None],
) -> dict[str, float | list[float] | None]:
    result: dict[str, float | list[float] | None] = {}
    keys = set(last_metrics.keys()) | set(sums.keys())
    for key in keys:
        count = counts.get(key, 0)
        if key in sums and count > 0:
            total = sums[key]
            if isinstance(total, list):
                result[key] = [v / count for v in total]
            else:
                result[key] = total / count
        else:
            result[key] = last_metrics.get(key)
    return result


__all__ = [
    "format_postfix",
    "wandb_log",
    "build_train_payload",
    "build_val_payload",
    "maybe_log_val",
    "accumulate_metric_sums",
    "finalize_metric_sums",
]
