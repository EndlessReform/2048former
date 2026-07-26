from __future__ import annotations

"""Bounded, real-data CPU training and exact-resume integration smoke test."""

import argparse
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch

from train_2048.config import TrainingConfig
from train_2048.training_loop import run_training


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "datasets/raws/d6_test_v2"
MAX_STEPS = 100
MAX_BATCH_SIZE = 256
MAX_THREADS = 32


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_config(
    *,
    init_dir: Path,
    checkpoint_dir: Path,
    dataset_dir: Path,
    steps: int,
    batch_size: int,
    checkpoint_step: int,
) -> TrainingConfig:
    return TrainingConfig.model_validate(
        {
            "init_dir": str(init_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "seed": 1729,
            "compile_enabled": False,
            "wandb": {"enabled": False, "mode": "disabled"},
            "target": {"mode": "hard_move"},
            "hyperparameters": {
                "learning_rate": 1e-3,
                "lr_schedule": {"name": "constant"},
                "optimizer": {"name": "adamw", "weight_decay": 0.01},
            },
            "batch": {"batch_size": batch_size},
            "dropout": {
                "dropout_prob": 0.1,
                "attention_dropout_prob": 0.0,
            },
            "amp": {"autocast_type": "fp32"},
            "dataset": {
                "dataset_dir": str(dataset_dir),
                "mmap_mode": True,
                "num_workers_train": 0,
                "shard_locality": True,
                "shard_cache_in_memory": False,
                "num_steps": steps,
                "val_run_pct": 0.2,
                "val_split_seed": 42,
                "val_num_steps": 1,
                "val_every": checkpoint_step,
            },
            "checkpoint": {
                "every_epochs": None,
                "save_pt_every_steps": checkpoint_step,
            },
        }
    )


def _write_tiny_init(path: Path) -> None:
    path.mkdir(parents=True)
    config = {
        "input_vocab_size": 17,
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 64,
        "layer_norm_eps": 1e-6,
        "dropout_prob": 0.1,
        "attention_dropout_prob": 0.0,
        "max_position_embeddings": 16,
        "head_type": "action_policy",
    }
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def _assert_same_weights(expected_path: Path, actual_path: Path) -> None:
    expected = load_file(str(expected_path), device="cpu")
    actual = load_file(str(actual_path), device="cpu")
    if expected.keys() != actual.keys():
        raise AssertionError("resumed checkpoint has different parameter names")
    for name, expected_tensor in expected.items():
        if not torch.equal(expected_tensor, actual[name]):
            max_delta = (expected_tensor - actual[name]).abs().max().item()
            raise AssertionError(
                f"resumed weight mismatch for {name} (max abs delta {max_delta})"
            )


def _validate_args(args: argparse.Namespace) -> None:
    if not 2 <= args.steps <= MAX_STEPS:
        raise ValueError(f"--steps must be between 2 and {MAX_STEPS}")
    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"--batch-size must be between 1 and {MAX_BATCH_SIZE}")
    if not 1 <= args.threads <= MAX_THREADS:
        raise ValueError(f"--threads must be between 1 and {MAX_THREADS}")
    if not args.dataset.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {args.dataset}")
    for filename in ("metadata.db", "steps.npy"):
        if not (args.dataset / filename).is_file():
            raise FileNotFoundError(f"smoke dataset is missing {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bounded CPU training and prove exact checkpoint resume."
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Keep artifacts here; otherwise use and remove a temporary RAM-backed directory.",
    )
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    _validate_args(args)

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    temporary = args.work_dir is None
    if temporary:
        temp_root = Path("/dev/shm")
        mounts = Path("/proc/mounts").read_text().splitlines()
        if not temp_root.is_dir() or not any(
            fields[1] == str(temp_root) and fields[2] == "tmpfs"
            for line in mounts
            if len(fields := line.split()) >= 3
        ):
            raise RuntimeError(
                "/dev/shm is not tmpfs; refusing implicit checkpoint writes. "
                "Pass --work-dir explicitly to choose an artifact location."
            )
        work_dir = Path(tempfile.mkdtemp(prefix="train-2048-smoke-", dir=temp_root))
    else:
        work_dir = args.work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=False)

    checkpoint_step = max(1, args.steps // 2)
    started = time.monotonic()
    try:
        init_dir = work_dir / "init"
        _write_tiny_init(init_dir)
        baseline_cfg = _make_config(
            init_dir=init_dir,
            checkpoint_dir=work_dir / "baseline",
            dataset_dir=args.dataset,
            steps=args.steps,
            batch_size=args.batch_size,
            checkpoint_step=checkpoint_step,
        )

        print(
            f"[smoke] Real pool: {args.dataset} | steps={args.steps} "
            f"batch={args.batch_size} | checkpoint={checkpoint_step}"
        )
        baseline_final, baseline_step = run_training(baseline_cfg, "cpu")
        if baseline_step != args.steps:
            raise AssertionError(
                f"baseline stopped at step {baseline_step}, expected {args.steps}"
            )
        resume_bundle = baseline_final.parent / f"model-step-{checkpoint_step:08d}.pt"
        if not resume_bundle.is_file():
            raise AssertionError(f"mid-run checkpoint was not written: {resume_bundle}")

        resumed_cfg = _make_config(
            init_dir=resume_bundle,
            checkpoint_dir=work_dir / "resumed",
            dataset_dir=args.dataset,
            steps=args.steps,
            batch_size=args.batch_size,
            checkpoint_step=checkpoint_step,
        )
        _seed_everything(999_999)
        resumed_final, resumed_step = run_training(resumed_cfg, "cpu")
        if resumed_step != args.steps:
            raise AssertionError(
                f"resumed run stopped at step {resumed_step}, expected {args.steps}"
            )
        _assert_same_weights(baseline_final, resumed_final)

        checkpoint_bytes = sum(
            path.stat().st_size for path in work_dir.rglob("*") if path.is_file()
        )
        elapsed = time.monotonic() - started
        print(
            f"[smoke] PASS: uninterrupted and resumed weights match exactly; "
            f"elapsed={elapsed:.1f}s artifacts={checkpoint_bytes / 1024**2:.2f} MiB"
        )
    finally:
        if temporary:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"[smoke] Artifacts retained at {work_dir}")


if __name__ == "__main__":
    main()
