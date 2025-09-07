from __future__ import annotations

from typing import Literal, Sequence

import torch
from pydantic import BaseModel, Field, field_validator


class BinningConfig(BaseModel):
    """
    Configuration for discretizing continuous EVs into bins.

    Strategies:
    - "edges": interpret `edges` as histogram edges with length = n_bins + 1.
      Bins are intervals [e[i], e[i+1]) except the last which is [e[-2], e[-1]].
      If `special_zero_one` is True, exact 0 and exact 1 get dedicated bins,
      and the interior intervals cover (0, 1) using the interior edges.
    - "upper_bounds": interpret `edges` as monotonically increasing upper bounds
      with length = n_bins. Bin k is (-inf, edges[k]] with clamping to [0, 1].

    Values are expected in [0, 1]. Out-of-range values are clamped.
    """

    strategy: Literal["edges", "upper_bounds"] = "edges"
    # When True with strategy="edges", put exact 0 and exact 1 into their own bins
    special_zero_one: bool = True
    edges: Sequence[float] = Field(
        default_factory=lambda: [
            0.0,
            0.5,
            0.8,
            0.9,
            0.95,
            0.98,
            0.99,
            0.995,
            0.998,
            0.999,
            1.0,
        ]
    )

    @field_validator("edges")
    @classmethod
    def _validate_edges(cls, v: Sequence[float]) -> Sequence[float]:
        if not v:
            raise ValueError("edges must be non-empty")
        if any(not (0.0 <= x <= 1.0) for x in v):
            raise ValueError("edges must lie within [0, 1]")
        if any(v[i] > v[i + 1] for i in range(len(v) - 1)):
            raise ValueError("edges must be non-decreasing")
        return v

    @property
    def n_bins(self) -> int:
        if self.strategy == "edges":
            if len(self.edges) < 2:
                raise ValueError("'edges' strategy requires at least 2 edges")
            base = len(self.edges) - 1
            return base + 2 if self.special_zero_one else base
        else:  # "upper_bounds"
            return len(self.edges)


class Binner:
    """Applies binning to tensors of EVs in [0, 1]."""

    def __init__(self, config: BinningConfig):
        self.config = config
        self._edges = torch.tensor(list(config.edges), dtype=torch.float32)

    @classmethod
    def from_config(cls, config: BinningConfig) -> "Binner":
        return cls(config)

    @property
    def n_bins(self) -> int:
        return self.config.n_bins

    def to_device(self, device: torch.device) -> "Binner":
        self._edges = self._edges.to(device)
        return self

    def bin_values(self, values: torch.Tensor) -> torch.Tensor:
        """
        Map EVs to integer bin indices.

        Args:
            values: tensor of any shape, with EVs expected in [0, 1].

        Returns:
            Long tensor of same shape containing bin indices in [0, n_bins-1].
        """
        x = values.float().clamp(0.0, 1.0)
        if self.config.strategy == "edges":
            if self.config.special_zero_one:
                # Dedicated bins for exact 0 and exact 1.
                # Interior bins cover (0, 1) split by the interior bounds.
                bounds = self._edges[1:-1]  # excludes 0.0 and 1.0
                idx = torch.bucketize(x, bounds, right=False)  # 0..len(bounds)

                is_zero = (x == 0.0)
                is_one = (x == 1.0)

                # Shift interior to start at bin 1, reserve last bin for exact 1
                out = idx.to(torch.long) + 1
                out = torch.where(is_zero, torch.zeros_like(out), out)
                out = torch.where(is_one, torch.full_like(out, self.n_bins - 1), out)
                return out.long()
            else:
                # Standard edges: [e[i], e[i+1)) with last inclusive of right edge
                bounds = self._edges[1:]
                idx = torch.bucketize(x, bounds, right=False)
                return idx.clamp(max=self.n_bins - 1).long()
        else:  # upper_bounds
            # edges length = n_bins, treat each as upper bound of bin
            idx = torch.bucketize(x, self._edges, right=True)
            return idx.clamp(max=self.n_bins - 1).long()


__all__ = ["BinningConfig", "Binner"]

