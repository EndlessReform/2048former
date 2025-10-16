from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import Objective


class MacroxueTokens(Objective):
    name = "macroxue_tokens"

    def __init__(self, *, tokenizer_path: Optional[str] = None) -> None:
        self.tokenizer_path = tokenizer_path

    def prepare_model(
        self, model: torch.nn.Module, device: torch.device, *, cfg: object, dl_train: Optional[DataLoader]
    ) -> torch.nn.Module:
        # If we can infer n_classes from tokenizer spec, ensure heads match.
        n_classes: Optional[int] = None
        if self.tokenizer_path:
            try:
                from train_2048.tokenization.macroxue import MacroxueTokenizerSpec

                spec = MacroxueTokenizerSpec.from_json(self.tokenizer_path)
                n_bins = int(len(spec.delta_edges) - 1)
                n_classes = n_bins + 2
            except Exception:
                n_classes = None
        if n_classes is None and dl_train is not None:
            # Fall back to peeking at a batch (can stall with mmap, so last resort)
            try:
                sample = next(iter(dl_train))
                n_classes = int(sample.get("n_classes", 0)) or None
            except Exception:
                n_classes = None

        if n_classes is not None and getattr(getattr(model, "config", None), "output_n_bins", None) != n_classes:
            model.config.output_n_bins = int(n_classes)
            for i in range(4):
                model.ev_heads[i] = torch.nn.Linear(model.config.hidden_size, int(n_classes)).to(device)

        return model.to(device)

    def train_step(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        cfg: object,
        zero_grad: bool = True,
        optimizer_step: bool = True,
        loss_scale: float = 1.0,
    ) -> Dict[str, float | list[float] | None]:
        tokens = batch["tokens"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        if zero_grad:
            optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            class _Null:
                def __enter__(self):
                    return None

                def __exit__(self, *args):
                    return False

            autocast = _Null()

        # Bounds check on tokens for clearer errors
        vocab = getattr(getattr(model, "tok_emb", None), "num_embeddings", None)
        if vocab is not None and tokens.numel():
            tmin = int(tokens.min().item())
            tmax = int(tokens.max().item())
            if tmin < 0 or tmax >= int(vocab):
                raise RuntimeError(f"Token id out of range: min={tmin} max={tmax} vocab={int(vocab)}")

        with autocast:
            _hs, head_out = model(tokens)
            per_head_losses: list[torch.Tensor] = []
            agree_sum = torch.zeros((), device=device, dtype=torch.float32)
            agree_cnt = 0
            for h in range(4):
                logits_h = head_out[h].float()
                tgt_h = targets[:, h]
                # Optional range validation for targets based on output width
                n_classes = int(logits_h.shape[-1])
                if tgt_h.numel():
                    tmin = int(tgt_h.min().item())
                    tmax = int(tgt_h.max().item())
                    if tmin < 0 or tmax >= n_classes:
                        raise RuntimeError(
                            f"Target out of range for head {h}: min={tmin} max={tmax} n_classes={n_classes}"
                        )
                loss_h = F.cross_entropy(logits_h, tgt_h)
                per_head_losses.append(loss_h)
                # Winner probability agreement
                winner_idx = n_classes - 1
                win_mask = tgt_h == winner_idx
                if win_mask.any():
                    probs = F.softmax(logits_h[win_mask], dim=-1)[:, winner_idx]
                    agree_sum = agree_sum + probs.sum()
                    agree_cnt += int(win_mask.sum().item())

            loss = sum(per_head_losses)

        scaled_loss = loss * float(loss_scale)
        scaled_loss.backward()
        if optimizer_step:
            if cfg.hyperparameters.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hyperparameters.grad_clip_norm)
            optimizer.step()

        head_losses = [float(l.detach().item()) for l in per_head_losses]
        policy_agreement = float((agree_sum / max(1, agree_cnt)).detach().item()) if agree_cnt > 0 else None
        return {"loss": float(loss.detach().item()), "head_losses": head_losses, "policy_accuracy": None, "policy_agreement": policy_agreement}

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        dl_val: DataLoader,
        device: torch.device,
    ) -> Dict[str, float | list[float] | None]:
        was_training = model.training
        model.eval()

        total_loss = 0.0
        total_heads = torch.zeros(4, dtype=torch.float64)
        n_batches = 0
        agree_sum = 0.0
        agree_cnt = 0

        if device.type == "cuda":
            autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            class _Null:
                def __enter__(self):
                    return None

                def __exit__(self, *args):
                    return False

            autocast = _Null()

        for batch in dl_val:
            tokens = batch["tokens"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            with autocast:
                _hs, head_out = model(tokens)
                per_head_losses: list[torch.Tensor] = []
                for h in range(4):
                    logits_h = head_out[h].float()
                    tgt_h = targets[:, h]
                    loss_h = F.cross_entropy(logits_h, tgt_h)
                    per_head_losses.append(loss_h)
                    # Winner-bin agreement
                    n_classes = int(logits_h.shape[-1])
                    winner_idx = n_classes - 1
                    win_mask = tgt_h == winner_idx
                    if win_mask.any():
                        probs = F.softmax(logits_h[win_mask], dim=-1)[:, winner_idx]
                        agree_sum += float(probs.sum().item())
                        agree_cnt += int(win_mask.sum().item())

                loss = sum(per_head_losses)

            total_loss += float(loss.detach().item())
            total_heads += torch.tensor([lh.detach().item() for lh in per_head_losses], dtype=torch.float64)
            n_batches += 1

        if was_training:
            model.train()

        if n_batches == 0:
            return {"loss": 0.0, "head_losses": [0.0, 0.0, 0.0, 0.0], "policy_accuracy": None, "policy_agreement": None}

        avg_loss = float(total_loss / n_batches)
        avg_heads = (total_heads / n_batches).tolist()
        policy_agreement = (agree_sum / agree_cnt) if (agree_cnt > 0) else None
        return {"loss": avg_loss, "head_losses": avg_heads, "policy_accuracy": None, "policy_agreement": policy_agreement}


__all__ = ["MacroxueTokens"]
