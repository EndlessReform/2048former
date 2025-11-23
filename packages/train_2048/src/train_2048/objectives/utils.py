from __future__ import annotations

from typing import Any, Tuple


def unpack_model_outputs(forward_out: Any) -> Tuple[Any, Any, Any]:
    """
    Normalize Encoder.forward outputs to (hidden_states, policy_out, value_out).

    The encoder returns (hidden_states, policy_out) for legacy callers or
    (hidden_states, policy_out, value_out) when a value head is enabled.
    """
    if not isinstance(forward_out, tuple):
        raise TypeError(f"unexpected model output type: {type(forward_out)}")
    if len(forward_out) == 2:
        hidden_states, policy_out = forward_out
        value_out = None
    elif len(forward_out) == 3:
        hidden_states, policy_out, value_out = forward_out
    else:
        raise ValueError(f"unexpected model output tuple length: {len(forward_out)}")
    return hidden_states, policy_out, value_out


__all__ = ["unpack_model_outputs"]
