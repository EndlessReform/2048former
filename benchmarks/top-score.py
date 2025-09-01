import argparse
import asyncio
import time
from pathlib import Path
from typing import Literal, Optional, Union

import statistics
import tomllib
import torch
from pydantic import BaseModel, Field
from tqdm import tqdm

from ai_2048 import Board, Move, Rng  # type: ignore

from train_2048.config import load_encoder_from_init
from train_2048.inference import (
    InferenceEngine,
    AsyncBatchPool,
    EngineInput,
    auto_device_name,
    board_to_tokens,
    legal_mask_from_board,
    select_move,
)


class OpenAIStrategy(BaseModel):
    type: Literal["openai"] = "openai"
    base_url: str = "https://api.openai.com/v1/chat/completions"
    model_id: str


class SLMStrategy(BaseModel):
    type: Literal["slm"] = "slm"
    init_folder: str
    sampling: Literal["top-1"] = "top-1"


class EvalConfig(BaseModel):
    strategy: Union[OpenAIStrategy, SLMStrategy] = Field(discriminator="type")
    num_seeds: int = 10
    max_retries: int = 3
    max_concurrent_games: int = 64

    @classmethod
    def from_toml(cls, path: str | Path):
        path = Path(path)
        with path.open("rb") as f:
            data = tomllib.load(f)
        return cls(**data)


class Strategy:
    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def submit(self, board: Board) -> "asyncio.Future[Optional[Move]]":  # noqa: F821
        raise NotImplementedError


class SLMStrategyImpl(Strategy):
    def __init__(self, init_folder: str) -> None:
        device_str = auto_device_name()
        model = load_encoder_from_init(init_folder)
        self.engine = InferenceEngine(model, device=device_str)
        self.pool = AsyncBatchPool(self.engine, max_batch=256, flush_interval_ms=1.0)

    async def start(self) -> None:
        await self.pool.start()

    async def close(self) -> None:
        await self.pool.close()

    def submit(self, board: Board) -> "asyncio.Future[Optional[Move]]":  # noqa: F821
        # Precompute legality; if no legal moves, resolve to None immediately
        lm = legal_mask_from_board(board)
        loop = asyncio.get_running_loop()
        if not lm.any():
            fut: "asyncio.Future[Optional[Move]]" = loop.create_future()
            fut.set_result(None)
            return fut

        tokens = board_to_tokens(board).squeeze(0)
        move_order = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        efut = self.pool.submit(EngineInput(tokens=tokens))

        async def _await_move() -> Optional[Move]:
            out = await efut
            # Build a fake batch of size 1 for selection
            head_probs = [hp.unsqueeze(0) for hp in out.head_probs]
            mv_idx = int(select_move(head_probs, legal_mask=lm).item())
            return move_order[mv_idx]

        return _await_move()


async def run_batched_games(config: EvalConfig) -> tuple[list[int], list[int], list[int], dict[int, int], int]:
    seeds = [42 + i * 1009 for i in range(config.num_seeds)]

    # Strategy selection (keeps engine details internal to the strategy)
    if config.strategy.type == "slm":
        strategy: Strategy = SLMStrategyImpl(config.strategy.init_folder)  # type: ignore[attr-defined]
    else:
        raise NotImplementedError(f"Strategy {config.strategy.type} not implemented")

    await strategy.start()

    scores: list[int] = []
    steps: list[int] = []
    top_tiles: list[int] = []
    reached_counts = {1024: 0, 2048: 0, 4096: 0, 8192: 0}
    aborted_games = 0

    max_conc = max(1, int(config.max_concurrent_games))
    seeds_iter = iter(seeds)

    class Game:
        __slots__ = ("seed", "rng", "board", "moves", "retries", "score", "top_tile", "done", "aborted")

        def __init__(self, seed: int):
            self.seed = seed
            self.rng = Rng(int(seed))
            self.board = Board.empty().with_random_tile(rng=self.rng).with_random_tile(rng=self.rng)
            self.moves = 0
            self.retries = 0
            self.score = int(self.board.score())
            self.top_tile = int(self.board.highest_tile())
            self.done = False
            self.aborted = False

    active: list[Game] = []

    def start_next_game() -> bool:
        try:
            s = next(seeds_iter)
        except StopIteration:
            return False
        active.append(Game(s))
        return True

    while len(active) < min(max_conc, len(seeds)):
        if not start_next_game():
            break

    completed = 0
    t0 = time.perf_counter()
    last_t = t0
    last_completed = 0
    total_moves = 0
    pbar = tqdm(total=len(seeds), desc="Games", dynamic_ncols=True)

    try:
        while completed < len(seeds):
            # Build batch of alive boards and submit to strategy
            pending: list[asyncio.Task] = []
            idxs: list[int] = []

            for idx, g in enumerate(active):
                if g.done:
                    continue
                if g.board.is_game_over():
                    g.done = True
                    continue
                pending.append(asyncio.create_task(strategy.submit(g.board)))
                idxs.append(idx)

            if not pending:
                newly_completed = 0
                for g in list(active):
                    if g.done:
                        scores.append(int(g.score))
                        steps.append(int(g.moves))
                        top_tiles.append(int(g.top_tile))
                        for thr in reached_counts:
                            if g.top_tile >= thr:
                                reached_counts[thr] += 1
                        if g.aborted:
                            aborted_games += 1
                        active.remove(g)
                        completed += 1
                        newly_completed += 1
                        start_next_game()
                if newly_completed == 0:
                    break
                pbar.update(newly_completed)
                now = time.perf_counter()
                if now - last_t >= 1.0:
                    window_c = completed - last_completed
                    dt = now - last_t
                    pbar.set_postfix_str(
                        f"active={len(active)} avg_mps={(total_moves/(now-t0)) if (now-t0)>0 else 0:.2f} last_s={window_c/dt if dt>0 else 0:.2f}"
                    )
                last_t = now
                last_completed = completed
                continue

            # Await all predictions for this tick
            moves: list[Optional[Move]] = await asyncio.gather(*pending)

            for bi, mv in enumerate(moves):
                g = active[idxs[bi]]
                if mv is None:
                    g.done = True
                    continue
                prev = g.board.raw
                g.board = g.board.make_move(mv, rng=g.rng)
                if g.board.raw == prev:
                    g.retries += 1
                    if g.retries > config.max_retries:
                        g.done = True
                        g.aborted = True
                    continue
                g.retries = 0
                g.moves += 1
                total_moves += 1
                g.score = int(g.board.score())
                g.top_tile = int(g.board.highest_tile())

            newly_completed = 0
            for g in list(active):
                if g.done or g.board.is_game_over():
                    g.done = True
                    scores.append(int(g.score))
                    steps.append(int(g.moves))
                    top_tiles.append(int(g.top_tile))
                    for thr in reached_counts:
                        if g.top_tile >= thr:
                            reached_counts[thr] += 1
                    if g.aborted:
                        aborted_games += 1
                    active.remove(g)
                    completed += 1
                    newly_completed += 1
                    start_next_game()

            if newly_completed:
                pbar.update(newly_completed)
                now = time.perf_counter()
                if now - last_t >= 1.0:
                    window_c = completed - last_completed
                    dt = now - last_t
                    pbar.set_postfix_str(
                        f"active={len(active)} avg_mps={(total_moves/(now-t0)) if (now-t0)>0 else 0:.2f} last_s={window_c/dt if dt>0 else 0:.2f}"
                    )
                    last_t = now
                    last_completed = completed
    finally:
        pbar.close()
        try:
            await backend.close()
        except Exception:
            pass

    return scores, steps, top_tiles, reached_counts, aborted_games


parser = argparse.ArgumentParser(description="Benchmark top score over multiple seeds")
parser.add_argument(
    "--config",
    type=str,
    default="config/top-score-benchmark.toml",
    help="Path to benchmark config TOML file",
)


def main():
    args = parser.parse_args()
    config = EvalConfig.from_toml(args.config)
    scores, steps, top_tiles, reached_counts, aborted_games = asyncio.run(run_batched_games(config))

    def fmt_mean(vals: list[int]) -> str:
        return f"{statistics.fmean(vals):.2f}" if vals else "n/a"

    if scores and steps:
        print("\n=== Benchmark Summary ===")
        print(f"games: {len(scores)}  aborted: {aborted_games}")
        print(
            "scores    -> mean: {mean}  max: {mx}  min: {mn}".format(
                mean=fmt_mean(scores), mx=max(scores), mn=min(scores)
            )
        )
        print(
            "steps     -> mean: {mean}  max: {mx}  min: {mn}".format(
                mean=fmt_mean(steps), mx=max(steps), mn=min(steps)
            )
        )
        print(
            "reached   -> "
            + ", ".join(
                f"{thr}: {reached_counts[thr]}/{len(scores)}" for thr in [1024, 2048, 4096, 8192]
            )
        )
        print(
            "top tile  -> max: {mx}  min: {mn}".format(
                mx=max(top_tiles), mn=min(top_tiles)
            )
        )
    else:
        print("No games played; check configuration.")


if __name__ == "__main__":
    main()
