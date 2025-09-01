from abc import ABC, abstractmethod
import argparse
from pathlib import Path
from pydantic import BaseModel, Field
import tomllib
from tqdm import tqdm
from typing import Literal, Optional, Union

from ai_2048 import Board, Move, Rng  # type: ignore
import statistics


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
    """
    Number of retries allowed if the policy makes an illegal move
    """

    @classmethod
    def from_toml(cls, path: str | Path):
        path = Path(path)
        with path.open("rb") as f:
            data = tomllib.load(f)
        return cls(**data)


class Strategy(ABC):
    @abstractmethod
    def get_move(self, board: Board) -> Optional[Move]:
        pass

    def reset(self):
        pass


class SLMStrategyImpl(Strategy):
    def __init__(self, config: SLMStrategy):
        from train_2048.config import load_encoder_from_init
        from train_2048.inference import ModelPolicy, auto_device_name

        device_str = auto_device_name()
        model = load_encoder_from_init(config.init_folder)
        self.policy = ModelPolicy(model, device=device_str)

    def get_move(self, board):
        return self.policy.best_move(board)


def create_strategy(config: Union[OpenAIStrategy, SLMStrategy]) -> Strategy:
    if config.type == "slm":
        assert isinstance(config, SLMStrategy)
        return SLMStrategyImpl(config)
    else:
        raise NotImplementedError(f"Strategy {config.type} not implemented yet")


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

    seeds = [42 + i * 1009 for i in range(config.num_seeds)]  # 1009 is prime

    policy = create_strategy(config.strategy)

    # Stats across games
    scores: list[int] = []
    steps: list[int] = []
    top_tiles: list[int] = []
    reached_counts = {1024: 0, 2048: 0, 4096: 0, 8192: 0}
    aborted_games = 0

    # TODO add parallelism
    for seed in tqdm(seeds):
        rng = Rng(int(seed))
        board = Board.empty().with_random_tile(rng=rng).with_random_tile(rng=rng)

        n_moves = 0
        n_retries = 0
        score = board.score()
        top_tile = board.highest_tile()

        while not board.is_game_over():
            mv = policy.get_move(board)
            if mv is None:
                break

            board_prev_state = board.raw
            board = board.make_move(mv, rng=rng)
            if board.raw == board_prev_state:
                n_retries += 1
                if n_retries > config.max_retries:
                    print(f"Too many illegal moves, aborting game with seed {seed}")
                    aborted_games += 1
                    break
                continue

            n_retries = 0
            n_moves += 1
            # Score and top tile monotonically increase
            score = board.score()
            top_tile = board.highest_tile()

        # Record results for this game
        scores.append(int(score))
        steps.append(int(n_moves))
        top_tiles.append(int(top_tile))
        for thr in reached_counts:
            if top_tile >= thr:
                reached_counts[thr] += 1

        # Reset strategy between games if needed
        try:
            policy.reset()
        except Exception:
            pass

    # Pretty-print aggregate stats
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
                f"{thr}: {reached_counts[thr]}/{len(scores)}"
                for thr in [1024, 2048, 4096, 8192]
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
