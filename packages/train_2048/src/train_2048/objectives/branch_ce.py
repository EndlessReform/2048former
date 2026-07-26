from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from pathlib import Path

from core_2048.tokenization.abs_ev_binning import BinningConfig, AbsEVBinningTokenizer
from train_2048.amp import autocast_context

from .base import Objective


class BranchCE(Objective):
    """Unified branch-wise cross-entropy objective for binned EV + macroxue tokens.

    Batch contract:
      - tokens: (B, S) long
      - branch_targets: (B, 4) long (legacy: branch_bin_targets / targets)
      - branch_mask: (B, 4) bool (optional; ignored unless masking is enabled)
    """

    name = "branch_ce"

    def __init__(self, *, target_mode: str, tokenizer_path: Optional[str] = None) -> None:
        self.target_mode = target_mode.strip().lower()
        self.tokenizer_path = tokenizer_path
        self._expected_n_classes: Optional[int] = None
        self._agreement_index: Optional[int] = None
        # Preserve prior masking behavior: binned_ev masks illegal; macroxue does not.
        self._mask_illegal = self.target_mode == "binned_ev"

    def _resolve_expected_n_classes(
        self,
        *,
        cfg: object,
        dl_train: Optional[DataLoader],
    ) -> Optional[int]:
        if self.target_mode == "macroxue_tokens":
            if not self.tokenizer_path:
                raise ValueError("tokenizer_path is required for macroxue_tokens objective")
            from core_2048.tokenization.macroxue import MacroxueTokenizerV2Spec

            spec = MacroxueTokenizerV2Spec.from_json(Path(self.tokenizer_path))
            return int(len(spec.vocab_order))
        if self.target_mode == "binned_ev":
            try:
                bin_cfg = BinningConfig(**getattr(cfg, "binning", {}).model_dump())
                tok = AbsEVBinningTokenizer(bin_cfg)
                return int(tok.n_bins)
            except Exception:
                return None
        return None

    def _resolve_agreement_index(self, n_classes: Optional[int]) -> Optional[int]:
        if n_classes is None:
            return None
        if self.target_mode == "binned_ev":
            # Agreement token is bin 0 for binned EV.
            return 0
        if self.target_mode == "macroxue_tokens":
            # Agreement token is the final class for macroxue.
            return max(0, int(n_classes) - 1)
        return None

    def _ensure_ev_heads(
        self,
        model: torch.nn.Module,
        device: torch.device,
        n_classes: Optional[int],
    ) -> None:
        ev_heads = getattr(model, "ev_heads", None)
        if not isinstance(ev_heads, (list, torch.nn.ModuleList)) or len(ev_heads) != 4:
            raise RuntimeError("BranchCE expects 4 per-branch EV heads (model.ev_heads length = 4)")
        if n_classes is None:
            return

        cfg = getattr(model, "config", None)
        current_bins = getattr(cfg, "output_n_bins", None)
        head_bins = None
        try:
            head_bins = int(ev_heads[0].out_features)
        except Exception:
            head_bins = None

        if current_bins != n_classes or head_bins != n_classes:
            if cfg is not None:
                cfg.output_n_bins = int(n_classes)
            for i in range(4):
                ev_heads[i] = torch.nn.Linear(model.config.hidden_size, int(n_classes)).to(device)

    def prepare_model(
        self,
        model: torch.nn.Module,
        device: torch.device,
        *,
        cfg: object,
        dl_train: Optional[DataLoader],
    ) -> torch.nn.Module:
        n_classes = self._resolve_expected_n_classes(cfg=cfg, dl_train=dl_train)

        self._expected_n_classes = n_classes
        self._agreement_index = self._resolve_agreement_index(n_classes)
        self._ensure_ev_heads(model, device, n_classes)
        return model.to(device)

    def _extract_targets(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "branch_targets" in batch:
            return batch["branch_targets"]
        if "branch_bin_targets" in batch:
            return batch["branch_bin_targets"]
        if "targets" in batch:
            return batch["targets"]
        raise KeyError("Expected 'branch_targets' (or legacy 'branch_bin_targets'/'targets') in batch")

    def _extract_mask(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        mask = batch.get("branch_mask")
        if mask is not None:
            return mask
        return None

    def _check_token_bounds(self, model: torch.nn.Module, tokens: torch.Tensor) -> None:
        vocab = getattr(getattr(model, "tok_emb", None), "num_embeddings", None)
        if vocab is not None and tokens.numel():
            tmin = int(tokens.min().item())
            tmax = int(tokens.max().item())
            if tmin < 0 or tmax >= int(vocab):
                raise RuntimeError(f"Token id out of range: min={tmin} max={tmax} vocab={int(vocab)}")

    def _check_target_bounds(self, targets: torch.Tensor, n_classes: Optional[int]) -> None:
        if n_classes is None or not targets.numel():
            return
        tmin = int(targets.min().item())
        tmax = int(targets.max().item())
        if tmin < 0 or tmax >= int(n_classes):
            raise RuntimeError(
                f"Target out of range: min={tmin} max={tmax} n_classes={int(n_classes)}"
            )

    def train_step(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        cfg: object,
        grad_scaler: Optional[GradScaler] = None,
        zero_grad: bool = True,
        optimizer_step: bool = True,
        loss_scale: float = 1.0,
    ) -> Dict[str, float | list[float] | None]:
        tokens = batch["tokens"].to(device, non_blocking=True)
        targets = self._extract_targets(batch).to(device, non_blocking=True)
        branch_mask = self._extract_mask(batch)
        if branch_mask is not None:
            branch_mask = branch_mask.to(device, non_blocking=True)

        if zero_grad:
            optimizer.zero_grad(set_to_none=True)

        self._check_token_bounds(model, tokens)

        with autocast_context(cfg, device, model=model):
            _hs, head_out = model(tokens)
            if not isinstance(head_out, (list, tuple)) or len(head_out) != 4:
                raise RuntimeError("BranchCE expects model to return 4 branch heads")

            per_head_losses: list[torch.Tensor] = []
            agree_sum = torch.zeros((), device=device, dtype=torch.float32)
            agree_cnt = 0
            for h in range(4):
                logits_h = head_out[h].float()
                if self._expected_n_classes is not None and int(logits_h.shape[-1]) != int(self._expected_n_classes):
                    raise RuntimeError(
                        "BranchCE head width does not match tokenizer classes: "
                        f"head={int(logits_h.shape[-1])} expected={int(self._expected_n_classes)}"
                    )
                tgt_h = targets[:, h]
                mask_h = branch_mask[:, h] if (branch_mask is not None) else None
                self._check_target_bounds(tgt_h, self._expected_n_classes or int(logits_h.shape[-1]))

                loss_h = F.cross_entropy(logits_h, tgt_h, reduction="none")
                if self._mask_illegal and mask_h is not None:
                    loss_h = loss_h[mask_h].mean() if mask_h.any() else torch.zeros(
                        (), device=logits_h.device, dtype=torch.float32
                    )
                else:
                    loss_h = loss_h.mean()
                per_head_losses.append(loss_h)

                if self._agreement_index is not None:
                    agree_sel = tgt_h == int(self._agreement_index)
                    if self._mask_illegal and mask_h is not None:
                        agree_sel = agree_sel & mask_h
                    if agree_sel.any():
                        probs = F.softmax(logits_h[agree_sel], dim=-1)[:, int(self._agreement_index)]
                        agree_sum = agree_sum + probs.sum()
                        agree_cnt += int(agree_sel.sum().item())

            loss = sum(per_head_losses)

        scaled_loss = loss * float(loss_scale)
        if grad_scaler is not None:
            grad_scaler.scale(scaled_loss).backward()
            if optimizer_step:
                if cfg.hyperparameters.grad_clip_norm is not None:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hyperparameters.grad_clip_norm)
                grad_scaler.step(optimizer)
                grad_scaler.update()
        else:
            scaled_loss.backward()
            if optimizer_step:
                if cfg.hyperparameters.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hyperparameters.grad_clip_norm)
                optimizer.step()

        head_losses = [float(l.detach().item()) for l in per_head_losses]
        policy_agreement = float((agree_sum / max(1, agree_cnt)).detach().item()) if agree_cnt > 0 else None
        return {
            "loss": float(loss.detach().item()),
            "head_losses": head_losses,
            "policy_accuracy": None,
            "policy_agreement": policy_agreement,
        }

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

        for batch in dl_val:
            tokens = batch["tokens"].to(device, non_blocking=True)
            targets = self._extract_targets(batch).to(device, non_blocking=True)
            branch_mask = self._extract_mask(batch)
            if branch_mask is not None:
                branch_mask = branch_mask.to(device, non_blocking=True)

            with autocast_context(None, device, model=model):
                _hs, head_out = model(tokens)
                if not isinstance(head_out, (list, tuple)) or len(head_out) != 4:
                    raise RuntimeError("BranchCE expects model to return 4 branch heads")
                per_head_losses: list[torch.Tensor] = []
                for h in range(4):
                    logits_h = head_out[h].float()
                    if self._expected_n_classes is not None and int(logits_h.shape[-1]) != int(self._expected_n_classes):
                        raise RuntimeError(
                            "BranchCE head width does not match tokenizer classes: "
                            f"head={int(logits_h.shape[-1])} expected={int(self._expected_n_classes)}"
                        )
                    tgt_h = targets[:, h]
                    mask_h = branch_mask[:, h] if (branch_mask is not None) else None
                    loss_h = F.cross_entropy(logits_h, tgt_h, reduction="none")
                    if self._mask_illegal and mask_h is not None:
                        loss_h = loss_h[mask_h].mean() if mask_h.any() else torch.zeros(
                            (), device=logits_h.device, dtype=torch.float32
                        )
                    else:
                        loss_h = loss_h.mean()
                    per_head_losses.append(loss_h)

                    if self._agreement_index is not None:
                        agree_sel = tgt_h == int(self._agreement_index)
                        if self._mask_illegal and mask_h is not None:
                            agree_sel = agree_sel & mask_h
                        if agree_sel.any():
                            probs = F.softmax(logits_h[agree_sel], dim=-1)[:, int(self._agreement_index)]
                            agree_sum += float(probs.sum().item())
                            agree_cnt += int(agree_sel.sum().item())

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


__all__ = ["BranchCE"]
