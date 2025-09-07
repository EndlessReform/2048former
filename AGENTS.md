# Repository Guidelines

## Project Structure & Modules
- `src/train_2048/`: core library (config, model, dataloader, binning, inference, train).
- `main.py`: CLI entry for training (`--config`, optional `--device`).
- `bin/`: helper scripts (e.g., `play_2048.py` to run a game with a saved init).
- `benchmarks/`: performance/quality checks (e.g., `bench_client_server.py`).
- `config/`: example configs (`config.example.toml`, benchmark presets).
- `inits/`: model/init snapshots (e.g., `inits/v1_50m/`).

## Environment & Tooling (uv REQUIRED)
- You must use `uv` for all setup and commands (e.g., `uv add`, `uv sync`, `uv run …`).
- Do not create manual venvs and do not use `pip` or direct `python` executions.
- This project is pinned via `pyproject.toml` + `uv.lock` and builds with `uv_build`.

## Build, Run, Test
- Install (editable): `uv sync`.
- Add dependency: `uv add <pkg>` (dev: `uv add --group dev <pkg>`).
- Train: `uv run python main.py --config config/config.example.toml`.
- Play 2048 with a model: `uv run bin/play_2048.py --init ./init --seed 123`.
- Benchmark: `uv run python benchmarks/bench_client_server.py --init inits/v1_pretrained_50m --uds /tmp/2048_infer.sock --device cuda --compile-mode default --config config/inference/top-score.toml --release`
- Python: 3.12+ (see `.python-version`). Torch auto-selects device; override with `--device cpu|cuda`.

## Coding Style & Naming
- Python, 4‑space indent, type hints required for public functions.
- Follow PEP 8/PEP 257; keep functions small and pure where possible.
- Modules: snake_case (`inference.py`), classes: `PascalCase`, functions/vars: `snake_case`.
- Keep dataclasses/Pydantic models in `config.py`; avoid circular imports.
- Prefer explicit `from train_2048 import …` over relative wildcards.

## Testing Guidelines
- No formal test suite yet. Validate via:
  - Quick sim: `bin/play_2048.py` (prints moves/sec, score, highest tile).
  - Benchmarks: `benchmarks/bench_client_server.py` for repeatable comparisons.
- Add new checks under `benchmarks/` or propose `tests/` with `pytest` (mirroring module paths).

## Commit & PR Guidelines
- Commits: short, imperative subject (<= 72 chars), optional scope/tag.
  - Examples: `Add training loop`, `Inference: mask illegal moves`, `bench: add top-score`.
- PRs: clear description, links to issues, config used, before/after metrics (score, MPS), and logs/snippets. Include screenshots only if they add clarity.
- Keep diffs focused; update `README.md` and configs when behavior or flags change.

## Configuration & Safety
- Start from `config/config.example.toml`; do not commit secrets (e.g., W&B tokens).
- W&B: enabled via config; offline/disabled modes are supported. Document any tracking changes in PRs.
