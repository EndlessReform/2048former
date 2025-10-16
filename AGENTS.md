# Repository Guidelines

## Project Structure & Module Organization
The core training library lives in `src/train_2048/`, covering configuration, data loading, binning, model definition, inference, and training utilities. The CLI entry point is `main.py`, which wraps training runs via `--config` and optional `--device`. Helper scripts sit in `bin/` (for example `bin/play_2048.py` to replay saved inits), while reproducible checks belong in `benchmarks/`. Reference configs are collected in `config/`, and model snapshots or initial weights reside under `inits/`.

## Notes on Astral uv

- **NEVER** edit pyproject.toml directly to add a dependency. Always `uv add` a dependency, e.g. `uv add scikit-learn`.
- **NEVER** run a file with: `python` directly, `venv` directly. always `uv run foo.py --locked` directly, or ephemeral heredocs with `uv run python --locked`. this is ESSENTIAL.
- To avoid network sandbox issues, run uv commands not needing network with --locked whenever possible.

## Build, Test, and Development Commands
Sync the environment with `uv sync`; the project pins dependencies through `pyproject.toml` and `uv.lock`. Launch training with `uv run --locked train --config config/config.example.toml` (set `--device cpu|cuda` as needed). Validate gameplay using `uv run bin/play_2048.py --init ./inits/v1_50m --seed 123`. Benchmark client/server inference via `uv run benchmarks/bench_client_server.py --init inits/v1_pretrained_50m --uds /tmp/2048_infer.sock --device cuda --compile-mode default --config config/inference/top-score.toml --release`.

## Coding Style & Naming Conventions
Follow Python 3.12+, four-space indentation, and type hints for every public function. Adhere to PEP 8 and PEP 257; keep functions small and pure where feasible. Modules use snake_case, classes PascalCase, functions and variables snake_case. Prefer explicit `from train_2048 import …` imports over relative wildcards, and add short comments only when clarifying non-obvious logic.

## Testing Guidelines
There is no formal test suite yet. Smoke-test gameplay with `bin/play_2048.py` and capture moves per second, score, and highest tile. Use `benchmarks/bench_client_server.py` for repeatable latency or quality comparisons. When proposing new checks, mirror existing benchmark patterns or add a `tests/` module with pytest-style names aligned to the corresponding package path.

## Commit & Pull Request Guidelines
Write commits in the imperative mood with subjects ≤72 characters (e.g., `Train: tune lr` or `Bench: add top-score`). PRs should summarize behavior changes, link related issues, state the config and device used, and include before/after metrics or logs. Attach artifacts only when they clarify results, and document any config or logging updates alongside the code.

## Configuration & Safety
Bootstrap experiments from `config/config.example.toml` and track tweaks in repo-controlled configs. Never commit secrets (e.g., W&B tokens); prefer using environment variables or offline modes. When adjusting dataset schemas or dtype definitions, update the associated documentation and keep NumPy and Rust layouts in lockstep to avoid breaking self-play pipelines.
