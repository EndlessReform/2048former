from typing import Literal, Optional

import torch

from .adapters import _moves_order, board_to_tokens, legal_mask_from_board
from .core import infer_move, prepare_model_for_inference


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
    def best_move(self, board):  # type: ignore[no-untyped-def]
        # Lazy import guards live in adapters
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


__all__ = ["ModelPolicy"]
