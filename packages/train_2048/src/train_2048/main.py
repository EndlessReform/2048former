import argparse
import torch
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.style import Style

from train_2048.config import load_config
from train_2048.training_loop import TrainingInterrupted, run_training

console = Console()


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Train 2048 transformer scaffold")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.example.toml",
        help="Path to a TOML config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g., cuda, cpu). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable torch profiler for a step window (torch.compile/inductor-friendly).",
    )
    parser.add_argument(
        "--profile-start",
        type=int,
        default=2,
        help="Global step to start torch profiling (inclusive).",
    )
    parser.add_argument(
        "--profile-end",
        type=int,
        default=10,
        help="Global step to stop torch profiling (inclusive).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    # Optional Weights & Biases setup
    wandb_run = None
    if args.profile and getattr(cfg, "wandb", None):
        console.print("[profile] W&B disabled for profiler run.", style="yellow")
    # Loud warnings if W&B isn't going to log online
    if getattr(cfg, "wandb", None) and not args.profile:
        if not getattr(cfg.wandb, "enabled", False):
            console.print(
                Panel(
                    "W&B DISABLED: [wandb].enabled=false — no metrics will be logged.\n"
                    'Set [wandb].enabled=true and [wandb].mode="online" to log to wandb.ai',
                    title="Warning",
                    style=Style(color="red", bold=True),
                    border_style="red",
                )
            )
        elif getattr(cfg.wandb, "mode", "online") != "online":
            mode = cfg.wandb.mode
            if mode == "disabled":
                message = (
                    "W&B MODE=disabled — run will be a dummy; nothing appears online."
                )
            else:
                message = (
                    "W&B MODE=offline — metrics stored locally; not visible online."
                )
            console.print(
                Panel(
                    f"{message}\n"
                    'Use [wandb].mode="online" (and run `wandb login`) to upload runs.',
                    title="Warning",
                    style=Style(color="red", bold=True),
                    border_style="red",
                )
            )
        elif getattr(cfg.wandb, "mode", "online") != "online":
            mode = cfg.wandb.mode
            print("\n" + "=" * 88)
            if mode == "disabled":
                print(
                    "[1;31mW&B MODE=disabled — run will be a dummy; nothing appears online.[0m"
                )
            else:
                print(
                    "[1;31mW&B MODE=offline — metrics stored locally; not visible online.[0m"
                )
            print('Use [wandb].mode="online" (and run `wandb login`) to upload runs.')
            print("=" * 88 + "\n")

    if getattr(cfg, "wandb", None) and cfg.wandb.enabled and not args.profile:
        try:
            import wandb  # type: ignore

            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=(cfg.wandb.entity or None),
                name=(cfg.wandb.run_name or None),
                tags=(cfg.wandb.tags or None),
                mode=cfg.wandb.mode,
                config={
                    "config_path": args.config,
                    "seed": cfg.seed,
                    "wandb_report_every": getattr(cfg.wandb, "report_every", 1),
                    "optimizer": cfg.hyperparameters.optimizer.model_dump(),
                    "lr": cfg.hyperparameters.learning_rate,
                    "lr_schedule": cfg.hyperparameters.lr_schedule.model_dump(),
                    "batch": cfg.batch.model_dump(),
                    "dropout": cfg.dropout.model_dump(),
                    "target": cfg.target.model_dump(),
                    "binning": cfg.binning.model_dump(),
                    "dataset": cfg.dataset.model_dump(),
                },
            )
            console.print(
                f"W&B run initialized: [bold cyan]{wandb_run.name}[/bold cyan] ([dim]{wandb_run.id}[/dim])"
            )
        except Exception as e:
            console.print(
                f"W&B init failed ({e}); continuing without W&B logging.",
                style="yellow",
            )
            wandb_run = None

    device_str = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    console.print(f"Config loaded from: [dim]{args.config}[/dim]")

    try:
        _ckpt_path, _global_step = run_training(
            cfg,
            device_str,
            wandb_run,
            profile=args.profile,
            profile_start=args.profile_start,
            profile_end=args.profile_end,
        )
    except TrainingInterrupted as exc:
        lines = [
            "Training interrupted (Ctrl+C).",
            f"Checkpoint directory: {exc.run_dir}",
        ]
        console.print(
            Panel(
                "\n".join(lines),
                title="Shutdown",
                style=Style(color="yellow", bold=True),
                border_style="yellow",
            )
        )
        raise SystemExit(130) from None
    except KeyboardInterrupt:
        lines = [
            "Training interrupted (Ctrl+C).",
            f"Checkpoint directory: {Path(cfg.checkpoint_dir).resolve()}",
        ]
        console.print(
            Panel(
                "\n".join(lines),
                title="Shutdown",
                style=Style(color="yellow", bold=True),
                border_style="yellow",
            )
        )
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
