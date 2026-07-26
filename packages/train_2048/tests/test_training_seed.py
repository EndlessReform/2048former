from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from train_2048 import training_loop


def test_run_training_seeds_runtime_before_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[float, float, torch.Tensor]] = []

    def collect_signature(_cfg: object) -> dict:
        observed.append((random.random(), np.random.random(), torch.rand(4)))
        raise RuntimeError("stop after seed check")

    monkeypatch.setattr(training_loop, "collect_dataset_signature", collect_signature)
    cfg = SimpleNamespace(seed=1729, target=SimpleNamespace(mode="hard_move"))

    for poison_seed in (1, 999):
        random.seed(poison_seed)
        np.random.seed(poison_seed)
        torch.manual_seed(poison_seed)
        with pytest.raises(RuntimeError, match="stop after seed check"):
            training_loop.run_training(cfg, "cpu")

    assert observed[0][0] == observed[1][0]
    assert observed[0][1] == observed[1][1]
    assert torch.equal(observed[0][2], observed[1][2])
