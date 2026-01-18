from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from pathlib import Path

from core_2048.tokenization.abs_ev_binning import BinningConfig, AbsEVBinningTokenizer
from train_2048.amp import autocast_context

from .base import Objective


class PolicyValueCoral(Objective):
    """Composite objective: policy head + CORAL value head (ordinal regression)."""

    name = "policy_value_coral"

    def __init__(
        self,
        *,
        policy_mode: str,
        tokenizer_path: Optional[str],
        tiles: Sequence[int],
        include_underflow: bool = True,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        value_pooling: str = "mean",
        value_proj_dim: Optional[int] = None,
    ) -> None:
        self.policy_mode = policy_mode.strip().lower()
        self.tokenizer_path = tokenizer_path
        self.include_underflow = bool(include_underflow)
        self.policy_weight = float(policy_weight)
        self.value_weight = float(value_weight)
        self.value_pooling = str(value_pooling)
        self.value_proj_dim = value_proj_dim

        tiles_list = [int(x) for x in tiles]
        if not tiles_list:
            raise ValueError("PolicyValueCoral requires non-empty tiles list")
        tiles_list = sorted(set(tiles_list))
        if any(t <= 0 for t in tiles_list):
            raise ValueError("tiles must be positive tile values")
        self.tiles = tiles_list
        self.value_num_classes = len(self.tiles) + (1 if self.include_underflow else 0)
        if self.value_num_classes <= 1:
            raise ValueError("value_num_classes must be > 1 for CORAL")
        if self.policy_mode not in ("binned_ev", "hard_move", "macroxue_tokens"):
            raise ValueError(f"Unsupported policy_mode: {self.policy_mode}")

        self._expected_n_classes: Optional[int] = None
        self._agreement_index: Optional[int] = None
        self._mask_illegal = self.policy_mode == "binned_ev"

    def _resolve_expected_n_classes(
        self,
        *,
        cfg: object,
        dl_train: Optional[DataLoader],
    ) -> Optional[int]:
        if self.policy_mode == "macroxue_tokens":
            if not self.tokenizer_path:
                raise ValueError("tokenizer_path is required for macroxue_tokens objective")
            from core_2048.tokenization.macroxue import MacroxueTokenizerV2Spec

            spec = MacroxueTokenizerV2Spec.from_json(Path(self.tokenizer_path))
            return int(len(spec.vocab_order))
        if self.policy_mode == "binned_ev":
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
        if self.policy_mode == "binned_ev":
            return 0
        if self.policy_mode == "macroxue_tokens":
            return max(0, int(n_classes) - 1)
        return None

    def _ensure_ev_heads(
        self,
        model: torch.nn.Module,
        device: torch.device,
        n_classes: Optional[int],
    ) -> None:
        if self.policy_mode not in ("binned_ev", "macroxue_tokens"):
            return
        ev_heads = getattr(model, "ev_heads", None)
        if not isinstance(ev_heads, (list, torch.nn.ModuleList)) or len(ev_heads) != 4:
            raise RuntimeError("PolicyValueCoral expects 4 per-branch EV heads (model.ev_heads length = 4)")
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

    def _ensure_value_head(self, model: torch.nn.Module, device: torch.device) -> None:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            cfg.value_head_type = "coral"
            cfg.value_head_num_classes = int(self.value_num_classes)
            cfg.value_head_pooling = self.value_pooling
            cfg.value_head_proj_dim = self.value_proj_dim

        from core_2048.model import CoralHead

        value_head = getattr(model, "value_head", None)
        value_head_proj = getattr(model, "value_head_proj", None)
        value_head_pooling = getattr(model, "value_head_pooling", None)

        hidden_size = int(getattr(cfg, "hidden_size", getattr(model, "hidden_size", 0)))
        if hidden_size <= 0:
            raise RuntimeError("Unable to resolve hidden_size for value head")

        needs_proj = self.value_pooling == "mean_proj"
        proj_dim = int(self.value_proj_dim or hidden_size)

        rebuild = value_head is None
        if value_head is not None and hasattr(value_head, "biases"):
            try:
                if int(value_head.biases.shape[0]) + 1 != int(self.value_num_classes):
                    rebuild = True
            except Exception:
                rebuild = True

        if value_head_pooling != self.value_pooling:
            rebuild = True

        if needs_proj:
            if value_head_proj is None:
                rebuild = True
            else:
                try:
                    if int(value_head_proj.weight.shape[0]) != proj_dim:
                        rebuild = True
                except Exception:
                    rebuild = True
        else:
            if value_head_proj is not None:
                rebuild = True

        if rebuild:
            if needs_proj:
                model.value_head_proj = torch.nn.Linear(hidden_size, proj_dim, bias=False).to(device)
                model.value_head_act = torch.nn.SiLU()
                value_in_dim = proj_dim
            else:
                model.value_head_proj = None
                model.value_head_act = None
                value_in_dim = hidden_size
            model.value_head = CoralHead(value_in_dim, int(self.value_num_classes)).to(device)
            model.value_head_pooling = self.value_pooling
            model.value_head_type = "coral"

    def prepare_model(
        self,
        model: torch.nn.Module,
        device: torch.device,
        *,
        cfg: object,
        dl_train: Optional[DataLoader],
    ) -> torch.nn.Module:
        n_classes = self._resolve_expected_n_classes(cfg=cfg, dl_train=dl_train)
        if n_classes is None and dl_train is not None:
            try:
                sample = next(iter(dl_train))
                if isinstance(sample, dict):
                    if "n_classes" in sample:
                        n_classes = int(sample["n_classes"])
                    elif "n_bins" in sample:
                        n_classes = int(sample["n_bins"])
            except Exception:
                n_classes = None

        self._expected_n_classes = n_classes
        self._agreement_index = self._resolve_agreement_index(n_classes)
        self._ensure_ev_heads(model, device, n_classes)
        self._ensure_value_head(model, device)
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

    def _map_highest_tile(self, highest_tile: torch.Tensor) -> torch.Tensor:
        tiles = torch.tensor(self.tiles, device=highest_tile.device, dtype=highest_tile.dtype)
        counts = (highest_tile.unsqueeze(1) >= tiles.unsqueeze(0)).sum(dim=1)
        if self.include_underflow:
            return counts.to(dtype=torch.long)
        max_idx = max(0, len(self.tiles) - 1)
        return torch.clamp(counts, max=max_idx).to(dtype=torch.long)

    def _coral_loss(self, logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        num_thresholds = logits.shape[1]
        levels = torch.arange(num_thresholds, device=logits.device)
        binary_targets = (target_idx[:, None] > levels).float()
        loss = F.binary_cross_entropy_with_logits(logits, binary_targets, reduction="none")
        return loss.sum(dim=1).mean()

    def _check_token_bounds(self, model: torch.nn.Module, tokens: torch.Tensor) -> None:
        vocab = getattr(getattr(model, "tok_emb", None), "num_embeddings", None)
        if vocab is not None and tokens.numel():
            tmin = int(tokens.min().item())
            tmax = int(tokens.max().item())
            if tmin < 0 or tmax >= int(vocab):
                raise RuntimeError(f"Token id out of range: min={tmin} max={tmax} vocab={int(vocab)}")

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
        highest_tile = batch["highest_tile"].to(device, non_blocking=True)

        if zero_grad:
            optimizer.zero_grad(set_to_none=True)

        self._check_token_bounds(model, tokens)

        with autocast_context(cfg, device, model=model):
            out = model(tokens, return_value=True)
            if not isinstance(out, (list, tuple)) or len(out) != 3:
                raise RuntimeError("Expected model(tokens, return_value=True) -> (hidden, policy, value)")
            _hs, policy_out, value_logits = out
            if value_logits is None:
                raise RuntimeError("Value head logits missing from model output")

            # Policy loss
            policy_loss = torch.zeros((), device=device, dtype=torch.float32)
            head_losses: list[torch.Tensor] = []
            policy_accuracy = None
            policy_agreement = None

            if self.policy_mode in ("binned_ev", "macroxue_tokens"):
                targets = self._extract_targets(batch).to(device, non_blocking=True)
                branch_mask = self._extract_mask(batch)
                if branch_mask is not None:
                    branch_mask = branch_mask.to(device, non_blocking=True)
                if not isinstance(policy_out, (list, tuple)) or len(policy_out) != 4:
                    raise RuntimeError("PolicyValueCoral expects 4 branch heads for binned_ev/macroxue_tokens")

                agree_sum = torch.zeros((), device=device, dtype=torch.float32)
                agree_cnt = 0
                for h in range(4):
                    logits_h = policy_out[h].float()
                    tgt_h = targets[:, h]
                    mask_h = branch_mask[:, h] if (branch_mask is not None) else None
                    loss_h = F.cross_entropy(logits_h, tgt_h, reduction="none")
                    if self._mask_illegal and mask_h is not None:
                        loss_h = loss_h[mask_h].mean() if mask_h.any() else torch.zeros(
                            (), device=logits_h.device, dtype=torch.float32
                        )
                    else:
                        loss_h = loss_h.mean()
                    head_losses.append(loss_h)

                    if self._agreement_index is not None:
                        agree_sel = tgt_h == int(self._agreement_index)
                        if self._mask_illegal and mask_h is not None:
                            agree_sel = agree_sel & mask_h
                        if agree_sel.any():
                            probs = F.softmax(logits_h[agree_sel], dim=-1)[:, int(self._agreement_index)]
                            agree_sum = agree_sum + probs.sum()
                            agree_cnt += int(agree_sel.sum().item())

                policy_loss = sum(head_losses)
                if self._agreement_index is not None:
                    if agree_cnt > 0:
                        policy_agreement = float((agree_sum / float(agree_cnt)).detach().item())
            else:
                move_targets = batch["move_targets"].to(device, non_blocking=True)
                branch_mask = batch.get("branch_mask")
                if branch_mask is not None:
                    branch_mask = branch_mask.to(device, non_blocking=True)

                if isinstance(policy_out, (list, tuple)):
                    if not all(t.shape[-1] == 1 for t in policy_out):
                        raise RuntimeError("hard_move expects single policy head or 4x1 logits list")
                    logits = torch.stack([t.float().squeeze(-1) for t in policy_out], dim=1)
                else:
                    logits = policy_out.float()
                loss_per_sample = F.cross_entropy(logits, move_targets, reduction="none")
                if branch_mask is not None and branch_mask.numel() == move_targets.numel() * 4:
                    chosen_legal = branch_mask[torch.arange(move_targets.size(0), device=device), move_targets]
                    policy_loss = loss_per_sample[chosen_legal].mean() if bool(chosen_legal.any()) else torch.zeros(
                        (), device=logits.device, dtype=torch.float32
                    )
                else:
                    policy_loss = loss_per_sample.mean()

                preds = logits.argmax(dim=1)
                if branch_mask is not None and branch_mask.numel() == move_targets.numel() * 4:
                    acc_mask = branch_mask[torch.arange(move_targets.size(0), device=device), move_targets]
                    if bool(acc_mask.any()):
                        policy_accuracy = float(((preds == move_targets) & acc_mask).float()[acc_mask].mean().detach().item())
                    else:
                        policy_accuracy = 0.0
                else:
                    policy_accuracy = float((preds == move_targets).float().mean().detach().item())

                probs = F.softmax(logits, dim=-1)
                p_t = probs[torch.arange(move_targets.size(0), device=device), move_targets]
                if branch_mask is not None and branch_mask.numel() == move_targets.numel() * 4:
                    if 'acc_mask' in locals() and bool(acc_mask.any()):
                        policy_agreement = float(p_t[acc_mask].mean().detach().item())
                    else:
                        policy_agreement = None
                else:
                    policy_agreement = float(p_t.mean().detach().item())

                head_losses = []
                for h in range(4):
                    sel = move_targets == h
                    if sel.any():
                        head_losses.append(loss_per_sample[sel].mean())
                    else:
                        head_losses.append(torch.zeros((), device=logits.device, dtype=torch.float32))

            # Value loss
            value_logits_f = value_logits.float()
            if int(value_logits_f.shape[-1]) != int(self.value_num_classes - 1):
                raise RuntimeError(
                    f"Value head width mismatch: got {int(value_logits_f.shape[-1])} "
                    f"expected {int(self.value_num_classes - 1)}"
                )
            target_idx = self._map_highest_tile(highest_tile)
            value_loss = self._coral_loss(value_logits_f, target_idx)

            loss = self.policy_weight * policy_loss + self.value_weight * value_loss

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

        return {
            "loss": float(loss.detach().item()),
            "policy_loss": float(policy_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "head_losses": [float(x.detach().item()) for x in head_losses],
            "policy_accuracy": policy_accuracy,
            "policy_agreement": policy_agreement,
        }

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, dl_val: DataLoader, device: torch.device) -> Dict[str, float | list[float] | None]:
        was_training = model.training
        model.eval()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_heads = torch.zeros(4, dtype=torch.float64)
        n_batches = 0
        total_correct = 0.0
        total_examples = 0
        agree_sum = 0.0
        agree_cnt = 0

        for batch in dl_val:
            tokens = batch["tokens"].to(device, non_blocking=True)
            highest_tile = batch["highest_tile"].to(device, non_blocking=True)

            with autocast_context(None, device, model=model):
                out = model(tokens, return_value=True)
                if not isinstance(out, (list, tuple)) or len(out) != 3:
                    raise RuntimeError("Expected model(tokens, return_value=True) -> (hidden, policy, value)")
                _hs, policy_out, value_logits = out
                if value_logits is None:
                    raise RuntimeError("Value head logits missing from model output")

                policy_loss = torch.zeros((), device=device, dtype=torch.float32)
                head_losses: list[torch.Tensor] = []

                if self.policy_mode in ("binned_ev", "macroxue_tokens"):
                    targets = self._extract_targets(batch).to(device, non_blocking=True)
                    branch_mask = self._extract_mask(batch)
                    if branch_mask is not None:
                        branch_mask = branch_mask.to(device, non_blocking=True)
                    if not isinstance(policy_out, (list, tuple)) or len(policy_out) != 4:
                        raise RuntimeError("PolicyValueCoral expects 4 branch heads for binned_ev/macroxue_tokens")

                    for h in range(4):
                        logits_h = policy_out[h].float()
                        tgt_h = targets[:, h]
                        mask_h = branch_mask[:, h] if (branch_mask is not None) else None
                        loss_h = F.cross_entropy(logits_h, tgt_h, reduction="none")
                        if self._mask_illegal and mask_h is not None:
                            loss_h = loss_h[mask_h].mean() if mask_h.any() else torch.zeros(
                                (), device=logits_h.device, dtype=torch.float32
                            )
                        else:
                            loss_h = loss_h.mean()
                        head_losses.append(loss_h)

                        if self._agreement_index is not None:
                            agree_sel = tgt_h == int(self._agreement_index)
                            if self._mask_illegal and mask_h is not None:
                                agree_sel = agree_sel & mask_h
                            if agree_sel.any():
                                probs = F.softmax(logits_h[agree_sel], dim=-1)[:, int(self._agreement_index)]
                                agree_sum += float(probs.sum().item())
                                agree_cnt += int(agree_sel.sum().item())

                    policy_loss = sum(head_losses)
                else:
                    move_targets = batch["move_targets"].to(device, non_blocking=True)
                    branch_mask = batch.get("branch_mask")
                    if branch_mask is not None:
                        branch_mask = branch_mask.to(device, non_blocking=True)

                    if isinstance(policy_out, (list, tuple)):
                        if not all(t.shape[-1] == 1 for t in policy_out):
                            raise RuntimeError("hard_move expects single policy head or 4x1 logits list")
                        logits = torch.stack([t.float().squeeze(-1) for t in policy_out], dim=1)
                    else:
                        logits = policy_out.float()
                    loss_per_sample = F.cross_entropy(logits, move_targets, reduction="none")
                    if branch_mask is not None and branch_mask.numel() == move_targets.numel() * 4:
                        chosen_legal = branch_mask[torch.arange(move_targets.size(0), device=device), move_targets]
                        policy_loss = loss_per_sample[chosen_legal].mean() if bool(chosen_legal.any()) else torch.zeros(
                            (), device=logits.device, dtype=torch.float32
                        )
                    else:
                        policy_loss = loss_per_sample.mean()

                    preds = logits.argmax(dim=1)
                    if branch_mask is not None and branch_mask.numel() == move_targets.numel() * 4:
                        acc_mask = branch_mask[torch.arange(move_targets.size(0), device=device), move_targets]
                        total_correct += float(((preds == move_targets) & acc_mask).sum().item())
                        total_examples += int(acc_mask.sum().item())
                    else:
                        total_correct += float((preds == move_targets).sum().item())
                        total_examples += int(move_targets.numel())

                    probs = F.softmax(logits, dim=-1)
                    p_t = probs[torch.arange(move_targets.size(0), device=device), move_targets]
                    if branch_mask is not None and branch_mask.numel() == move_targets.numel() * 4:
                        if bool(acc_mask.any()):
                            agree_sum += float(p_t[acc_mask].sum().item())
                            agree_cnt += int(acc_mask.sum().item())
                    else:
                        agree_sum += float(p_t.sum().item())
                        agree_cnt += int(move_targets.numel())

                    head_losses = []
                    for h in range(4):
                        sel = move_targets == h
                        if sel.any():
                            head_losses.append(loss_per_sample[sel].mean())
                        else:
                            head_losses.append(torch.zeros((), device=logits.device, dtype=torch.float32))

                value_logits_f = value_logits.float()
                if int(value_logits_f.shape[-1]) != int(self.value_num_classes - 1):
                    raise RuntimeError(
                        f"Value head width mismatch: got {int(value_logits_f.shape[-1])} "
                        f"expected {int(self.value_num_classes - 1)}"
                    )
                target_idx = self._map_highest_tile(highest_tile)
                value_loss = self._coral_loss(value_logits_f, target_idx)

                loss = self.policy_weight * policy_loss + self.value_weight * value_loss

            total_loss += float(loss.detach().item())
            total_policy_loss += float(policy_loss.detach().item())
            total_value_loss += float(value_loss.detach().item())
            total_heads += torch.tensor([lh.detach().item() for lh in head_losses], dtype=torch.float64)
            n_batches += 1

        if was_training:
            model.train()

        if n_batches == 0:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "head_losses": [0.0, 0.0, 0.0, 0.0],
                "policy_accuracy": None,
                "policy_agreement": None,
            }

        avg_loss = float(total_loss / n_batches)
        avg_policy_loss = float(total_policy_loss / n_batches)
        avg_value_loss = float(total_value_loss / n_batches)
        avg_heads = (total_heads / n_batches).tolist()
        policy_accuracy = None
        if self.policy_mode == "hard_move":
            policy_accuracy = float(total_correct / total_examples) if total_examples > 0 else None
        policy_agreement = (agree_sum / agree_cnt) if (agree_cnt > 0) else None
        return {
            "loss": avg_loss,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "head_losses": avg_heads,
            "policy_accuracy": policy_accuracy,
            "policy_agreement": policy_agreement,
        }


__all__ = ["PolicyValueCoral"]
