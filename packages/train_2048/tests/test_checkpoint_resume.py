from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from train_2048.checkpointing import restore_runtime_state, save_pt_bundle


class _TinyScaler:
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def state_dict(self) -> dict[str, float]:
        return {"scale": self.scale}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.scale = float(state["scale"])


def _tiny_training_state():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    return model, optimizer


def test_tiny_bundle_restores_rng_scaler_and_cursor(tmp_path: Path) -> None:
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    model, optimizer = _tiny_training_state()
    scaler = _TinyScaler(4096.0)
    path = tmp_path / "resume.pt"
    cursor = {
        "version": 1,
        "seed": 42,
        "epoch": 2,
        "shard": 3,
        "position": 4,
    }

    save_pt_bundle(
        path,
        model=model,
        optimizer=optimizer,
        training_cfg=None,
        global_step=7,
        resume_state={"global_step": 7, "data_cursor": cursor},
        dataset_metadata={"fingerprint": "tiny"},
        grad_scaler=scaler,
    )
    expected = (random.random(), np.random.random(), torch.rand(3))

    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    scaler.scale = 1.0
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["resume"]["data_cursor"] == cursor
    assert restore_runtime_state(
        payload["resume"]["runtime_state"],
        grad_scaler=scaler,
    )
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert scaler.scale == 4096.0
    assert path.stat().st_size < 1_000_000
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_atomic_bundle_failure_preserves_previous_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, optimizer = _tiny_training_state()
    path = tmp_path / "resume.pt"
    path.write_bytes(b"previous-checkpoint")

    def _failed_save(payload, destination: str) -> None:
        Path(destination).write_bytes(b"partial")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(torch, "save", _failed_save)
    with pytest.raises(OSError, match="simulated interrupted write"):
        save_pt_bundle(
            path,
            model=model,
            optimizer=optimizer,
            training_cfg=None,
            global_step=1,
            resume_state={"global_step": 1},
        )

    assert path.read_bytes() == b"previous-checkpoint"
    assert not list(tmp_path.glob(".*.tmp.*"))


def _train_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    start: int,
    stop: int,
) -> None:
    model.train()
    for step in range(start, stop):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs[step])
        loss = torch.nn.functional.mse_loss(prediction, targets[step])
        loss.backward()
        optimizer.step()


def test_tiny_dropout_training_matches_uninterrupted_after_resume(
    tmp_path: Path,
) -> None:
    inputs = torch.arange(60, dtype=torch.float32).reshape(10, 2, 3) / 10
    targets = torch.arange(20, dtype=torch.float32).reshape(10, 2, 1) / 20
    torch.manual_seed(991)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.35),
        torch.nn.Linear(5, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    _train_steps(model, optimizer, inputs, targets, 0, 4)

    path = tmp_path / "tiny-training.pt"
    save_pt_bundle(
        path,
        model=model,
        optimizer=optimizer,
        training_cfg=None,
        global_step=4,
        resume_state={"global_step": 4},
    )
    _train_steps(model, optimizer, inputs, targets, 4, 10)
    uninterrupted = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    payload = torch.load(path, map_location="cpu", weights_only=False)
    resumed_model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.35),
        torch.nn.Linear(5, 1),
    )
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.01)
    resumed_model.load_state_dict(payload["model"])
    resumed_optimizer.load_state_dict(payload["optimizer"])
    assert restore_runtime_state(payload["resume"]["runtime_state"])
    _train_steps(resumed_model, resumed_optimizer, inputs, targets, 4, 10)

    for name, value in resumed_model.state_dict().items():
        assert torch.equal(value, uninterrupted[name]), name
    assert path.stat().st_size < 1_000_000
