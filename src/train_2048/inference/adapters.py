import torch

try:
    # Optional: only used by adapter helpers
    from ai_2048 import Board, Move, Rng  # type: ignore
except Exception:  # pragma: no cover - optional dependency for adapters
    Board = Move = Rng = None  # type: ignore


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
