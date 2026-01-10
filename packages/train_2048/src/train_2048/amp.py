from __future__ import annotations

import contextlib

import torch


def resolve_autocast_type(cfg: object | None = None, *, model: torch.nn.Module | None = None) -> str:
    """Resolve the autocast type from config or model metadata."""
    if cfg is not None:
        amp_cfg = getattr(cfg, "amp", None)
        autocast_type = getattr(amp_cfg, "autocast_type", None)
        if autocast_type is not None:
            return str(autocast_type)
    if model is not None:
        autocast_type = getattr(model, "_autocast_type", None)
        if autocast_type is not None:
            return str(autocast_type)
    return "bf16"


def use_transformer_engine(cfg: object | None, device: torch.device, *, model: torch.nn.Module | None = None) -> bool:
    """Return True if TransformerEngine FP8 autocast is requested."""
    if device.type != "cuda":
        return False
    return resolve_autocast_type(cfg, model=model) == "mxfp8"


def require_transformer_engine(cfg: object | None, device: torch.device, *, model: torch.nn.Module | None = None) -> None:
    """Validate that TransformerEngine is available when requested."""
    autocast_type = resolve_autocast_type(cfg, model=model)
    if autocast_type != "mxfp8":
        return
    if device.type != "cuda":
        raise ValueError("amp.autocast_type='mxfp8' requires CUDA.")
    try:
        import transformer_engine  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "TransformerEngine is required for amp.autocast_type='mxfp8'. "
            "Install transformer-engine with CUDA support."
        ) from exc


def autocast_context(
    cfg: object | None,
    device: torch.device,
    *,
    model: torch.nn.Module | None = None,
) -> contextlib.AbstractContextManager[None]:
    """Return the correct autocast context for the current precision mode."""
    autocast_type = resolve_autocast_type(cfg, model=model)
    if device.type != "cuda":
        if autocast_type == "mxfp8":
            raise ValueError("amp.autocast_type='mxfp8' requires CUDA.")
        return contextlib.nullcontext()
    if autocast_type == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if autocast_type == "fp32":
        return contextlib.nullcontext()
    if autocast_type == "mxfp8":
        try:
            from transformer_engine.common import recipe as te_recipe
            from transformer_engine.pytorch import fp8_autocast
        except Exception as exc:
            raise RuntimeError(
                "TransformerEngine is required for amp.autocast_type='mxfp8'. "
                "Install transformer-engine with CUDA support."
            ) from exc
            # TODO this is wrong.
            # Fuck NVIDIA
        return fp8_autocast(enabled=True, fp8_recipe=te_recipe.DelayedScaling())
    raise ValueError(f"Unknown amp.autocast_type: {autocast_type}")


__all__ = [
    "resolve_autocast_type",
    "use_transformer_engine",
    "require_transformer_engine",
    "autocast_context",
]
