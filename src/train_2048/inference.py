from __future__ import annotations

from typing import Iterable, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
try:
    # Optional: only used by adapter helpers
    from ai_2048 import Board, Move, Rng  # type: ignore
except Exception:  # pragma: no cover - optional dependency for adapters
    Board = Move = Rng = None  # type: ignore


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
def select_move(
    head_probs: list[torch.Tensor],
    *,
    legal_mask: Optional[torch.Tensor] = None,
    strategy: Literal["max-p1"] = "max-p1",
    one_bin_index: Optional[int] = None,
) -> torch.Tensor:
    """
    Select a move index per batch according to a simple strategy.

    Current default strategy "max-p1": choose the direction whose head assigns
    the highest probability to the bin corresponding to value==1.

    Args:
      head_probs: list of 4 tensors (B, n_bins), softmax probabilities per head.
      legal_mask: optional (B, 4) bool tensor; if provided, illegal moves are
                  masked out when selecting the move. If all moves are illegal
                  for a sample, falls back to global argmax without masking.
      strategy: currently only "max-p1" is implemented; reserved for future
                strategies like sampling by distribution or expected value.
      one_bin_index: override the index used for the "1" bin. Defaults to the
                     last bin (n_bins-1) inferred from head_probs.

    Returns:
      Long tensor of shape (B,) with selected move indices in {0,1,2,3}.
    """
    if len(head_probs) != 4:
        raise ValueError("Expected 4 heads (up, down, left, right)")

    B = head_probs[0].size(0)
    n_bins = head_probs[0].size(1)
    if one_bin_index is None:
        one_bin_index = n_bins - 1  # default: dedicated '1' bin at the end

    # Stack p1 across heads -> (B, 4)
    p1 = torch.stack([hp[:, one_bin_index] for hp in head_probs], dim=1)

    if legal_mask is not None:
        # Ensure mask is on same device and boolean dtype
        if legal_mask.device != p1.device or legal_mask.dtype != torch.bool:
            legal_mask = legal_mask.to(device=p1.device, dtype=torch.bool)
        if legal_mask.shape != p1.shape:
            raise ValueError(
                f"legal_mask must have shape {p1.shape}, got {tuple(legal_mask.shape)}"
            )
        # Very negative for illegal so they won't be chosen
        masked = p1.masked_fill(~legal_mask, float("-inf"))
        # If a row is all -inf (no legal moves), fall back to unmasked p1
        no_legal = ~torch.isfinite(masked).any(dim=1)
        # For rows with at least one legal move, take argmax on masked
        best_masked = masked.argmax(dim=1)
        if no_legal.any():
            best_unmasked = p1.argmax(dim=1)
            best_masked = torch.where(no_legal, best_unmasked, best_masked)
        return best_masked.long()

    # No mask: global argmax on p1 across heads
    return p1.argmax(dim=1).long()


@torch.inference_mode()
def infer_move(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    legal_mask: Optional[torch.Tensor] = None,
    set_eval: bool = True,
    one_bin_index: Optional[int] = None,
    strategy: Literal["max-p1"] = "max-p1",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Convenience wrapper to run the model and select a move in one call.

    Args:
      model: Encoder with 4 heads.
      tokens: (B, S) or (S,) long tensor of token ids for the board.
      legal_mask: optional (B, 4) bool mask of legal moves.
      set_eval: set model to eval() during the forward pass.
      one_bin_index: override the index of the '1' bin (defaults to last bin).
      strategy: selection strategy; currently only "max-p1".

    Returns:
      (moves, extras) where moves is (B,) long tensor; extras includes:
        - probs: list[Tensor] of head distributions (B, n_bins)
        - p1: Tensor (B, 4) with per-head probability of the '1' bin
    """
    head_probs = forward_distributions(model, tokens, set_eval=set_eval)
    n_bins = head_probs[0].size(1)
    if one_bin_index is None:
        one_bin_index = n_bins - 1
    p1 = torch.stack([hp[:, one_bin_index] for hp in head_probs], dim=1)
    moves = select_move(
        head_probs,
        legal_mask=legal_mask,
        strategy=strategy,
        one_bin_index=one_bin_index,
    )
    return moves, {"probs": head_probs, "p1": p1}


__all__ = ["forward_distributions", "select_move", "infer_move"]


# ---------------------------
# ai_2048 adapter conveniences
# ---------------------------

def _moves_order() -> list:
    if Move is None:
        raise RuntimeError("ai_2048 not available; adapter helpers require it")
    # Dataset order is [Up, Down, Left, Right]
    return [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]


def board_to_tokens(board: "Board") -> torch.Tensor:
    """Convert a board to token tensor of shape (1, 16)."""
    if Board is None:
        raise RuntimeError("ai_2048 not available; board_to_tokens requires it")
    vals = board.to_exponents()  # iterable of 16 exponents
    return torch.tensor(vals, dtype=torch.long).unsqueeze(0)


def legal_mask_from_board(board: "Board") -> torch.Tensor:
    """
    Compute a (1, 4) bool tensor mask indicating legal moves without consuming
    the caller's RNG by using a local dummy rng for probing.
    """
    if Board is None:
        raise RuntimeError("ai_2048 not available; legal_mask_from_board requires it")
    base_vals = list(board.to_values())
    mask: list[bool] = []
    dummy_rng = Rng(0)  # deterministic local RNG for probing
    for mv in _moves_order():
        nb = board.make_move(mv, rng=dummy_rng)
        mask.append(list(nb.to_values()) != base_vals)
    return torch.tensor(mask, dtype=torch.bool).unsqueeze(0)


class ModelPolicy:
    """
    Simple policy that selects the move whose head assigns highest probability
    to the '1' bin, with legality masking derived from the board.

    Compatible with the ai_2048 Expectimax-like interface: exposes best_move(board).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: Optional[torch.device | str] = None,
        one_bin_index: Optional[int] = None,
        strategy: Literal["max-p1"] = "max-p1",
    ) -> None:
        self.model = model
        self.strategy = strategy
        self.one_bin_index = one_bin_index
        # Prepare model for inference on the desired device and dtype, and compile it
        self.model, self.used_dtype = prepare_model_for_inference(
            self.model, device=device, prefer_bf16=True, compile_mode="reduce-overhead"
        )

    @torch.inference_mode()
    def best_move(self, board: "Board") -> Optional["Move"]:
        if Board is None:
            raise RuntimeError("ai_2048 not available; ModelPolicy requires it")
        tokens = board_to_tokens(board)
        legal_mask = legal_mask_from_board(board)
        # If no legal moves, return None (game over)
        if not legal_mask.any():
            return None
        moves, _extras = infer_move(
            self.model,
            tokens,
            legal_mask=legal_mask,
            set_eval=True,
            one_bin_index=self.one_bin_index,
            strategy=self.strategy,
        )
        idx = int(moves.item())
        return _moves_order()[idx]


__all__ += [
    "board_to_tokens",
    "legal_mask_from_board",
    "ModelPolicy",
]


# ---------------------------
# Inference model preparation
# ---------------------------


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


__all__ += ["prepare_model_for_inference"]
