import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict
from safetensors.torch import save_file as safe_save_file

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

    # Optimizer setup from config
    opt_cfg = cfg.hyperparameters.optimizer
    if opt_cfg.name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.hyperparameters.learning_rate,
            betas=(opt_cfg.beta1, opt_cfg.beta2),
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay,
        )
    elif opt_cfg.name == "muon":
        raise NotImplementedError("'muon' optimizer not implemented; use adamw.")
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg.name}")

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

    # Training loop: compute CE loss per head
    model.train()
    steps = cfg.dataset.num_steps or 0
    if steps > 0:
        it = iter(dl)
        pbar = tqdm(range(steps))
        for _ in pbar:
            batch = next(it)
            metrics = train_step(model, batch, optimizer, device)
            pbar.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "u": f"{metrics['head_losses'][0]:.3f}",
                    "d": f"{metrics['head_losses'][1]:.3f}",
                    "l": f"{metrics['head_losses'][2]:.3f}",
                    "r": f"{metrics['head_losses'][3]:.3f}",
                }
            )
    else:
        epochs = cfg.dataset.num_epochs or 1
        for epoch in range(epochs):
            pbar = tqdm(dl)
            for batch in pbar:
                metrics = train_step(model, batch, optimizer, device)
                pbar.set_postfix(
                    {
                        "loss": f"{metrics['loss']:.4f}",
                        "u": f"{metrics['head_losses'][0]:.3f}",
                        "d": f"{metrics['head_losses'][1]:.3f}",
                        "l": f"{metrics['head_losses'][2]:.3f}",
                        "r": f"{metrics['head_losses'][3]:.3f}",
                    }
                )
            print(f"Completed epoch {epoch + 1}/{epochs}")

    # Save final checkpoint to safetensors
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model-final.safetensors"
    state_cpu = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    safe_save_file(state_cpu, str(ckpt_path), metadata={"format": "pt"})
    print(f"Saved final checkpoint: {ckpt_path}")


def train_step(
    model,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """
    Single optimization step over one DataLoader batch (no micro-batching).

    Batch must include:
    - tokens: (N, S) int64
    - branch_mask: (N, 4) bool
    - branch_bin_targets: (N, 4) int64
    """
    tokens = batch["tokens"].to(device, non_blocking=True)
    branch_mask = batch["branch_mask"].to(device, non_blocking=True)
    targets_bins = batch["branch_bin_targets"].to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:

        class _NullCtx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        autocast = _NullCtx()

    with autocast:
        _hs, ev_logits = model(tokens)

        per_head_losses = []
        for h in range(4):
            logits_h = ev_logits[h].float()  # compute loss in fp32 for stability
            tgt_h = targets_bins[:, h]
            mask_h = branch_mask[:, h]

            # Cross-entropy per-sample
            loss_h = F.cross_entropy(logits_h, tgt_h, reduction="none")
            # Mask illegal moves
            if mask_h.any():
                loss_h = loss_h[mask_h].mean()
            else:
                loss_h = torch.zeros((), device=logits_h.device, dtype=torch.float32)
            per_head_losses.append(loss_h)

        loss = sum(per_head_losses)

    loss.backward()
    optimizer.step()

    head_losses = [lh.detach().item() for lh in per_head_losses]
    total_loss = loss.detach().item()

    return {"loss": total_loss, "head_losses": head_losses}


if __name__ == "__main__":
    main()
