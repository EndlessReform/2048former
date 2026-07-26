from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from train_2048.training_model import make_scheduler


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        hyperparameters=SimpleNamespace(
            learning_rate=0.1,
            muon_lr=None,
            lr_schedule=SimpleNamespace(
                name="cosine",
                warmup_steps=0,
                decay_steps=0,
                cooldown_pct=None,
                min_lr_ratio=0.1,
                linear_steps=None,
                linear_start_step=0,
                intermediate_ratio=None,
            ),
        )
    )


def test_resumed_scheduler_uses_original_not_decayed_base_lr() -> None:
    cfg = _config()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scale, apply, _ = make_scheduler(cfg, optimizer, total_steps=10)
    apply(scale(4))
    assert optimizer.param_groups[0]["lr"] < 0.1

    resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
    resumed_optimizer = torch.optim.SGD([resumed_parameter], lr=0.1)
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    resumed_scale, resumed_apply, _ = make_scheduler(
        cfg,
        resumed_optimizer,
        total_steps=10,
    )

    expected_lr = 0.1 * scale(5)
    assert resumed_apply(resumed_scale(5)) == pytest.approx(expected_lr)
    assert resumed_optimizer.param_groups[0]["initial_lr"] == pytest.approx(0.1)


def test_legacy_scheduler_checkpoint_uses_configured_base_lr() -> None:
    cfg = _config()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.04)
    legacy_state = optimizer.state_dict()
    legacy_state["param_groups"][0].pop("initial_lr", None)

    resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
    resumed_optimizer = torch.optim.SGD([resumed_parameter], lr=0.1)
    resumed_optimizer.load_state_dict(legacy_state)
    scale, apply, _ = make_scheduler(cfg, resumed_optimizer, total_steps=10)
    assert apply(scale(5)) == pytest.approx(0.1 * scale(5))
