from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def forward_distributions(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    set_eval: bool = True,
) -> list[torch.Tensor]:
    """
    Run the model and return per-head probability distributions over bins.

    Args:
      model: Encoder with 4 output heads.
      tokens: Long tensor of shape (B, S) or (S,) of token ids.
      set_eval: If True, temporarily sets model to eval() during the forward.

    Returns:
      List of length 4; each element is a float tensor (B, n_bins) with softmax
      probabilities for that head.
    """
    with torch.inference_mode():
        # Normalize input shape to (B, S)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        assert tokens.dim() == 2, "tokens must be (B, S) or (S,)"

        # Move inputs to the model's device
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = tokens.device
        prev_mode: Optional[bool] = None
        if set_eval:
            prev_mode = model.training
            model.eval()

        try:
            forward_out = model(tokens.to(model_device, dtype=torch.long))
            if isinstance(forward_out, tuple) and len(forward_out) == 3:
                _hs, ev_logits, _value_out = forward_out
            else:
                _hs, ev_logits = forward_out
            # Softmax per head over bins
            head_probs = [F.softmax(logits.float(), dim=-1) for logits in ev_logits]
        finally:
            if set_eval and prev_mode is not None:
                model.train(prev_mode)

        return head_probs


def prepare_model_for_inference(
    model: torch.nn.Module,
    *,
    device: Optional[torch.device | str] = None,
    prefer_bf16: bool = True,
    compile_mode: Optional[str] = "reduce-overhead",
):
    """
    Move model to target device, optionally forcing bf16 on CUDA, then compile.

    Returns (model, used_dtype) where used_dtype is the model's parameter dtype
    after any conversion.
    """
    # Resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)

    # Move to device, forcing bf16 on CUDA if requested
    if device.type == "cuda" and prefer_bf16:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device)

    # Compile for inference (skip on CPU to avoid slowdown/unsupported cases)
    if compile_mode is not None and device.type != "cpu":
        try:
            model = torch.compile(model, mode=compile_mode)
        except Exception:
            # If compile not available/supported, proceed uncompiled
            pass

    # Report the model's dtype after move
    try:
        used_dtype = next(model.parameters()).dtype
    except StopIteration:
        used_dtype = None
    return model, used_dtype
