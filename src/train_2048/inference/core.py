from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def auto_device_name() -> str:
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.inference_mode()
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
        _hs, ev_logits = model(tokens.to(model_device, dtype=torch.long))
        # Softmax per head over bins
        head_probs = [F.softmax(logits.float(), dim=-1) for logits in ev_logits]
    finally:
        if set_eval and prev_mode is not None:
            model.train(prev_mode)

    return head_probs


@torch.inference_mode()
def infer_move(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    legal_mask: Optional[torch.Tensor] = None,
    set_eval: bool = True,
    one_bin_index: Optional[int] = None,
    strategy: str = "max-p1",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Convenience wrapper to run the model and select a move in one call.

    Returns (moves, extras) where moves is (B,) long tensor; extras includes:
    - probs: list[Tensor] of head distributions (B, n_bins)
    - p1: Tensor (B, 4) with per-head probability of the '1' bin
    """
    from .selection import select_move  # local import to avoid cycles

    head_probs = forward_distributions(model, tokens, set_eval=set_eval)
    n_bins = head_probs[0].size(1)
    if one_bin_index is None:
        one_bin_index = n_bins - 1
    p1 = torch.stack([hp[:, one_bin_index] for hp in head_probs], dim=1)
    moves = select_move(
        head_probs,
        legal_mask=legal_mask,
        strategy="max-p1" if strategy is None else strategy,  # keep current default
        one_bin_index=one_bin_index,
    )
    return moves, {"probs": head_probs, "p1": p1}


def prepare_model_for_inference(
    model: torch.nn.Module,
    *,
    device: Optional[torch.device | str] = None,
    prefer_bf16: bool = True,
    compile_mode: Optional[str] = "reduce-overhead",
) -> Tuple[torch.nn.Module, Optional[torch.dtype]]:
    """
    Move model to target device, optionally forcing bf16 on CUDA, then compile.

    - On CUDA and when prefer_bf16 is True, parameters are moved to
      torch.bfloat16 to leverage tensor cores during inference.
    - On other platforms (CPU/MPS) the dtype is left unchanged.

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
