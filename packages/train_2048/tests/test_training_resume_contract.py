from __future__ import annotations

from types import SimpleNamespace

import pytest

from train_2048.training_resume import (
    DataCursorTracker,
    build_resume_state,
    resolve_resume_data_cursor,
    validate_resume_configuration,
)


def _config() -> SimpleNamespace:
    batch = SimpleNamespace(
        batch_size=2048,
        physical_batch_size=lambda: 512,
        grad_accum_steps=lambda: 4,
    )
    batch.adaptive = SimpleNamespace(model_dump=lambda: {"enabled": False})
    return SimpleNamespace(
        seed=42,
        batch=batch,
        dataset=SimpleNamespace(
            num_steps=100,
            rotation_augment=SimpleNamespace(model_dump=lambda: {"mode": "none"}),
            flip_augment=SimpleNamespace(model_dump=lambda: {"mode": "none"}),
        ),
        hyperparameters=SimpleNamespace(
            learning_rate=0.001,
            lr_schedule=SimpleNamespace(model_dump=lambda: {"name": "constant"}),
            optimizer=SimpleNamespace(model_dump=lambda: {"name": "adamw"}),
        ),
    )


def test_exact_resume_rejects_batch_shape_change() -> None:
    payload = {
        "version": 2,
        "effective_batch_size": 2048,
        "micro_batch_size": 1024,
        "grad_accum_steps": 2,
    }
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_resume_configuration(_config(), payload)


def test_cursor_resume_rejects_dataset_or_split_change() -> None:
    signature = {"dataset_dir": "/data/pool"}
    payload = {
        "data_cursor": {
            "version": 1,
            "seed": 42,
            "epoch": 0,
            "shard": 0,
            "position": 10,
        }
    }
    with pytest.raises(ValueError, match="does not match"):
        resolve_resume_data_cursor(
            signature,
            "current-fingerprint",
            resume_payload_meta=payload,
            resume_dataset_meta={"fingerprint": "different-fingerprint"},
        )


def test_exact_resume_accepts_only_the_original_training_contract() -> None:
    cfg = _config()
    payload = build_resume_state(4, 8192, 0, cfg)
    validate_resume_configuration(cfg, payload)

    payload["restart_contract"]["training_seed"] = 99
    with pytest.raises(ValueError, match="contract differs"):
        validate_resume_configuration(cfg, payload)


def test_data_cursor_commits_only_after_all_accumulation_microbatches() -> None:
    original = {"position": 4}
    tracker = DataCursorTracker(original)
    tracker.begin_step()
    tracker.observe_microbatch({"position": 5})
    tracker.observe_microbatch({"position": 6})
    tracker.observe_microbatch({"position": 7})
    assert tracker.committed == original
    assert tracker.commit_step() == {"position": 7}

    tracker.begin_step()
    tracker.observe_microbatch({"position": 8})
    # Simulated interruption before optimizer-step completion: no commit call.
    assert tracker.committed == {"position": 7}
