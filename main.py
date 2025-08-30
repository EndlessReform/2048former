import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from train_2048.config import load_config, load_encoder_from_init
from train_2048.binning import Binner
from train_2048.dataloader import StepBatchDataset, make_collate_step_batches


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

    device_str = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = torch.device(device_str)

    model = load_encoder_from_init(cfg.init_dir)
    # Use bf16 on CUDA for better perf/memory; keep default on CPU
    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device)
    # model = torch.compile(model, mode="reduce-overhead")

    # Print a brief summary
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded Encoder on {device_str} with {n_params:,} params")
    print(f"Config loaded from: {args.config}")

    # Build binner and dataloader
    binner = Binner.from_config(cfg.binning)
    ds = StepBatchDataset(
        pack_path=cfg.dataset.resolved_packfile(),
        batch_size=cfg.batch.batch_size,
        shuffle=True,
        seed=cfg.seed,
    )
    collate_fn = make_collate_step_batches(binner)
    dl = DataLoader(
        ds,
        batch_size=None,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Stubbed training loop: microbatch over each dataloader batch, forward only
    model.train()
    steps = cfg.dataset.num_steps or 0
    if steps > 0:
        it = iter(dl)
        for step in tqdm(range(steps)):
            batch = next(it)
            tokens = batch["tokens"].to(device, non_blocking=True)
            N = tokens.size(0)
            micro = cfg.batch.micro_batch_size or N
            for start in range(0, N, micro):
                end = min(start + micro, N)
                mb_tokens = tokens[start:end]
                _hs, ev_logits = model(mb_tokens)
                # Optional: sanity check shapes
                if step == 0 and start == 0:
                    print(
                        f"First forward: micro {mb_tokens.shape} -> logits {[t.shape for t in ev_logits]}"
                    )
    else:
        epochs = cfg.dataset.num_epochs or 1
        for epoch in tqdm(range(epochs)):
            for batch in dl:
                tokens = batch["tokens"].to(device, non_blocking=True)
                N = tokens.size(0)
                micro = cfg.batch.micro_batch_size or N
                for start in range(0, N, micro):
                    end = min(start + micro, N)
                    mb_tokens = tokens[start:end]
                    _hs, ev_logits = model(mb_tokens)
            print(f"Completed epoch {epoch + 1}/{epochs}")


if __name__ == "__main__":
    main()
