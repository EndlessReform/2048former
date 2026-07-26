# Repository Guidelines

## Project Structure & Module Organization
The core training library lives in `src/train_2048/`, covering configuration, data loading, binning, model definition, inference, and training utilities. The CLI entry point is `main.py`, which wraps training runs via `--config` and optional `--device`. Helper scripts sit in `bin/` (for example `bin/visualize-runs.py`), while reproducible checks belong in `benchmarks/`. The Rust orchestrator lives in `crates/game-engine`. Reference configs are collected in `config/`, and model snapshots or initial weights reside under `inits/`.

## Notes on Astral uv

- **NEVER** edit pyproject.toml directly to add a dependency. Always `uv add` a dependency, e.g. `uv add scikit-learn`.
- **NEVER** run a file with: `python` directly, `venv` directly. always `uv run foo.py --locked` directly, or ephemeral heredocs with `uv run python --locked`. this is ESSENTIAL.
- To avoid network sandbox issues, run uv commands not needing network with --locked whenever possible.

## Build, Test, and Development Commands
Sync the environment with `uv sync`; the project pins dependencies through `pyproject.toml` and `uv.lock`. Launch training with `uv run --locked train --config config/config.example.toml` (set `--device cpu|cuda` as needed). Validate gameplay with the Rust orchestrator: `cargo run -p game-engine -- --config config/inference/top-score.toml`. Benchmark client/server inference via `uv run benchmarks/bench_client_server.py --init inits/v1_pretrained_50m --uds /tmp/2048_infer.sock --device cuda --compile-mode default --config config/inference/top-score.toml --release`.
For Rust commands, use `--locked` when running `cargo` to avoid lockfile/network churn (e.g. `cargo test -p dataset-packer --locked`).

## Coding Style & Naming Conventions
Follow Python 3.12+, four-space indentation, and type hints for every public function. Adhere to PEP 8 and PEP 257; keep functions small and pure where feasible. Modules use snake_case, classes PascalCase, functions and variables snake_case. Prefer explicit `from train_2048 import …` imports over relative wildcards, and add short comments only when clarifying non-obvious logic.

## Testing Guidelines
Run the low-level suite with `uv run --locked python -m pytest packages/train_2048/tests benchmarks/test_invariants.py -q`. Smoke-test gameplay with the Rust orchestrator (see `crates/game-engine/README.md`) and use `benchmarks/bench_client_server.py` for repeatable latency or quality comparisons.

### CPU training integration smoke

Run `uv run --locked benchmarks/smoke_training_cpu.py`. It trains a one-layer model for 10 steps at batch size 32 on the real 9.7 MiB `datasets/raws/d6_test_v2` SQLite + NPY pool, validates once, resumes from step 5, and requires exact equality with the uninterrupted final weights. It uses zero DataLoader workers, two CPU threads, no compile, no W&B, and temporary checkpoints in `/dev/shm` that are deleted after success, so it neither touches the GPU nor writes checkpoints to flash.

Use `--steps 100` for the longer bounded variant. The script rejects more than 100 steps or batch sizes over 256. Pass `--work-dir /tmp/train-2048-smoke-debug` only when artifacts must be retained for inspection; the directory must not already exist.

This is the preferred integration gate for changes to dataloading, collation, objectives, the training loop, validation, checkpointing, or exact resume. It uses production code and real rows, not mocks. It does not exercise compressed multi-shard pools, CUDA/TransformerEngine, distributed training, or scientific convergence.

Notes on pytest in this repo:
- Running `uv run pytest` hit missing deps (numpy/pytest) because it used a different env; `uv run --locked python -m pytest ...` worked once deps were present.
- `uv sync --locked --dev` alone did not install pytest here; explicitly `uv add --group dev pytest` fixed it.

## Commit & Pull Request Guidelines
Write commits in the imperative mood with subjects ≤72 characters (e.g., `Train: tune lr` or `Bench: add top-score`). PRs should summarize behavior changes, link related issues, state the config and device used, and include before/after metrics or logs. Attach artifacts only when they clarify results, and document any config or logging updates alongside the code.

## Configuration & Safety
Bootstrap experiments from `config/config.example.toml` and track tweaks in repo-controlled configs. Never commit secrets (e.g., W&B tokens); prefer using environment variables or offline modes. When adjusting dataset schemas or dtype definitions, update the associated documentation and keep NumPy and Rust layouts in lockstep to avoid breaking self-play pipelines.
