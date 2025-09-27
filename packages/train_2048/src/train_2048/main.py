import argparse
import torch
from typing import Optional

from train_2048.config import load_config
from train_2048.training_loop import run_training


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
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    # Optional Weights & Biases setup
    wandb_run = None
    # Loud warnings if W&B isn't going to log online
    if getattr(cfg, "wandb", None):
        if not getattr(cfg.wandb, "enabled", False):
            print("\n" + "=" * 88)
            print("[1;31mW&B DISABLED: [wandb].enabled=false — no metrics will be logged.[0m")
            print("Set [wandb].enabled=true and [wandb].mode=\"online\" to log to wandb.ai")
            print("=" * 88 + "\n")
        elif getattr(cfg.wandb, "mode", "online") != "online":
            mode = cfg.wandb.mode
            print("\n" + "=" * 88)
            if mode == "disabled":
                print("[1;31mW&B MODE=disabled — run will be a dummy; nothing appears online.[0m")
            else:
                print("[1;31mW&B MODE=offline — metrics stored locally; not visible online.[0m")
            print("Use [wandb].mode=\"online\" (and run `wandb login`) to upload runs.")
            print("=" * 88 + "\n")

    if getattr(cfg, "wandb", None) and cfg.wandb.enabled:
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
            print(f"W&B run initialized: {wandb_run.name} ({wandb_run.id})")
        except Exception as e:
            print(f"W&B init failed ({e}); continuing without W&B logging.")
            wandb_run = None

    device_str = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Config loaded from: {args.config}")

    _ckpt_path, _global_step = run_training(cfg, device_str, wandb_run)


if __name__ == "__main__":
    main()
