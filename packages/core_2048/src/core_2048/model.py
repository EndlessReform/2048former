from typing import Literal

from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueObjectiveConfig(BaseModel):
    type: Literal["mse", "cross_entropy"] = "mse"
    vocab_size: int | None = None
    vocab_type: str | None = None

    @classmethod
    def model_validate(cls, obj):
        cfg = super().model_validate(obj)
        if cfg.type == "cross_entropy":
            if cfg.vocab_size is None or int(cfg.vocab_size) <= 0:
                raise ValueError("value objective 'cross_entropy' requires vocab_size > 0")
            if not cfg.vocab_type:
                raise ValueError("value objective 'cross_entropy' requires a vocab_type string label")
        return cfg


class ValueHeadConfig(BaseModel):
    # Toggle value head; omit this block or set enabled=False to disable.
    enabled: bool = True
    # Pooling strategy for value readout. Only mean pooling is supported today.
    pooling: Literal["mean"] = "mean"
    # Optional extra MLP applied before value pooling.
    pre_pool_mlp: bool = False
    objective: ValueObjectiveConfig = ValueObjectiveConfig()

    @classmethod
    def model_validate(cls, obj):
        cfg = super().model_validate(obj)
        if not cfg.enabled:
            return cfg
        if cfg.pooling != "mean":
            raise ValueError("value_head.pooling currently supports only 'mean'")
        # Defer objective validation to ValueObjectiveConfig.model_validate
        _ = ValueObjectiveConfig.model_validate(cfg.objective)
        return cfg


class EncoderConfig(BaseModel):
    input_vocab_size: int = 16
    output_n_bins: int | None = None
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    layer_norm_eps: float
    dropout_prob: float = 0.1
    # GQA: number of key/value heads. If None, defaults to num_attention_heads (MHA)
    num_key_value_heads: int | None = None
    # Attention dropout (used by scaled_dot_product_attention during training)
    attention_dropout_prob: float = 0.0
    # Absolute positional embeddings length
    max_position_embeddings: int = 16
    # Output head type: default binned EV per direction; alternative single policy head over 4 moves
    head_type: str = "binned_ev"  # accepted: "binned_ev", "action_policy"
    # Optional value head configuration (disabled when null)
    value_head: ValueHeadConfig | None = None

    @classmethod
    def model_validate(cls, obj):
        cfg = super().model_validate(obj)
        # Validate head/output compatibility
        if cfg.head_type == "binned_ev":
            if cfg.output_n_bins is None or int(cfg.output_n_bins) <= 0:
                raise ValueError("output_n_bins must be set (>0) when head_type='binned_ev'")
        else:
            # action_policy: output_n_bins is unused; allow None
            pass
        if cfg.value_head is not None and not cfg.value_head.enabled:
            # Treat disabled value_head the same as None
            cfg.value_head = None
        return cfg


class EncoderAttention(nn.Module):
    """
    LLaMA 3-style self-attention with GQA using F.scaled_dot_product_attention.

    Notes:
    - Absolute positional encoding is expected to be added outside this module.
    - Implements grouped-query attention (GQA): queries have `num_attention_heads`,
      keys/values have `num_key_value_heads`. K/V are shared across groups without
      explicit head-wise repeat-interleave; expansion is handled efficiently via
      batch shaping before attention.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = (
            config.num_key_value_heads
            if config.num_key_value_heads is not None
            else config.num_attention_heads
        )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_attention_heads must be a multiple of num_key_value_heads for GQA"
            )
        self.groups = self.num_heads // self.num_kv_heads

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.head_dim = self.hidden_size // self.num_heads

        # Separate projections for Q, K, V
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        self.wq = nn.Linear(self.hidden_size, q_size, bias=False)
        self.wk = nn.Linear(self.hidden_size, kv_size, bias=False)
        self.wv = nn.Linear(self.hidden_size, kv_size, bias=False)

        self.wo = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.attn_dropout_p = float(
            getattr(config, "attention_dropout_prob", 0.0) or 0.0
        )
        self.resid_dropout = nn.Dropout(config.dropout_prob)

    def _shape_qkv(self, x: torch.Tensor):
        # x: (B, S, H)
        B, S, _ = x.size()
        # Projections
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape to (B, S, n_heads, head_dim) for q
        q = q.view(B, S, self.num_heads, self.head_dim)
        # Reshape to (B, S, n_kv_heads, head_dim) for k and v
        k = k.view(B, S, self.num_kv_heads, self.head_dim)
        v = v.view(B, S, self.num_kv_heads, self.head_dim)

        # Transpose to (B, n_*, S, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Do not repeat-interleave here; return:
        # q: (B, n_heads, S, D), k/v: (B, n_kv_heads, S, D)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ):
        # Expect x as (B, S, H) — assumed pre-normalized by caller
        q, k, v = self._shape_qkv(x)

        # Compute attention. If using GQA (groups > 1), avoid head-wise repeats by
        # reshaping into (B * groups, n_kv_heads, S, D) and sharing K/V across groups.
        if self.groups > 1:
            B, n_h, S, D = q.shape
            n_kv = k.shape[1]
            g = self.groups
            # q -> (B, g, n_kv, S, D) -> (B*g, n_kv, S, D)
            q_g = q.view(B, g, n_kv, S, D).reshape(B * g, n_kv, S, D)
            # k/v -> (B, 1, n_kv, S, D) expanded across groups, then merge batch
            k_g = k.unsqueeze(1).expand(B, g, n_kv, S, D).reshape(B * g, n_kv, S, D)
            v_g = v.unsqueeze(1).expand(B, g, n_kv, S, D).reshape(B * g, n_kv, S, D)

            # Adjust mask if provided: expect broadcastable to (B, n_heads, S, S).
            # We reshape to (B*g, n_kv, S, S) by similar expansion when needed.
            if attn_mask is not None:
                # Try to broadcast by expanding along group dimension if possible.
                # Accept masks shaped (B, 1, S, S) or (B, n_h, S, S).
                if attn_mask.dim() == 4 and attn_mask.size(0) == B:
                    if attn_mask.size(1) == 1:
                        # (B, 1, S, S) -> (B, g, 1, S, S) -> (B*g, 1, S, S) -> (B*g, n_kv, S, S)
                        m = attn_mask.unsqueeze(1).expand(B, g, 1, S, S)
                        attn_mask_g = m.reshape(B * g, 1, S, S)
                    else:
                        # (B, n_heads, S, S) -> (B, g, n_kv, S, S) -> (B*g, n_kv, S, S)
                        m = attn_mask.view(B, g, n_kv, S, S)
                        attn_mask_g = m.reshape(B * g, n_kv, S, S)
                else:
                    # Fallback: let PyTorch attempt broadcasting
                    attn_mask_g = attn_mask
            else:
                attn_mask_g = None

            attn_out_g = F.scaled_dot_product_attention(
                q_g,
                k_g,
                v_g,
                attn_mask=attn_mask_g,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )  # (B*g, n_kv, S, D)

            # Reshape back to (B, n_heads, S, D)
            attn_out = attn_out_g.view(B, g, n_kv, S, D).reshape(B, n_h, S, D)
        else:
            # Standard MHA case: heads match
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )  # (B, n_heads, S, D)

        # Merge heads back: (B, S, H)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(x.size(0), x.size(1), self.hidden_size)
        )

        out = self.wo(attn_out)
        out = self.resid_dropout(out)
        return out


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: silu(Wg x) * (Wu x) -> Wd (no residual/norm)."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, dropout_prob: float, eps: float
    ):
        super().__init__()
        self.w_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.w_up(x)
        gate = self.w_gate(x)
        act = F.silu(gate) * up
        out = self.w_down(act)
        out = self.dropout(out)
        return out


class EncoderBlock(nn.Module):
    """Transformer encoder block: Pre-norm RMSNorm + GQA attention + SwiGLU MLP."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.attn = EncoderAttention(config)
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout_prob=config.dropout_prob,
            eps=config.layer_norm_eps,
        )
        # Pre-norm RMSNorms
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ):
        # Encoder uses non-causal (bidirectional) self-attention.
        x = x + self.attn(self.attn_norm(x), attn_mask=attn_mask, is_causal=False)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, max_position_embeddings: int, hidden_size: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_position_embeddings, hidden_size)

    def forward(self, seq_len: int, device: torch.device):
        pos = torch.arange(seq_len, device=device)
        return self.pos_emb(pos)  # (S, H)


class Encoder(nn.Module):
    """
    Stack of EncoderBlocks with token + absolute positional embeddings.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.input_vocab_size, config.hidden_size)
        # TODO add 2D inductive bias
        self.pos_emb = AbsolutePositionalEmbedding(
            max_position_embeddings=config.max_position_embeddings,
            hidden_size=config.hidden_size,
        )
        # No embedding pre-norm; blocks use pre-norm RMSNorm

        self.blocks = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.num_hidden_layers)]
        )
        # Final RMSNorm
        self.final_ln = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Output heads are modular; default remains binned EV per direction
        self.head_type = getattr(config, "head_type", "binned_ev")
        if self.head_type == "binned_ev":
            if self.config.output_n_bins is None or int(self.config.output_n_bins) <= 0:
                raise ValueError("EncoderConfig.output_n_bins must be > 0 for binned_ev head")
            self.ev_heads = nn.ModuleList(
                [nn.Linear(config.hidden_size, int(config.output_n_bins)) for _ in range(4)]
            )
            self.policy_head = None
        elif self.head_type == "action_policy":
            # Single 4-way policy head (Up, Down, Left, Right)
            self.policy_head = nn.Linear(config.hidden_size, 4)
            self.ev_heads = None
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}")

        # Optional value head (mean-pooled linear probe)
        vh_cfg = getattr(config, "value_head", None)
        self.value_head_cfg = vh_cfg if vh_cfg is not None and vh_cfg.enabled else None
        self.value_pre_pool_mlp: nn.Module | None = None
        self.value_pre_pool_norm: nn.Module | None = None
        if self.value_head_cfg is not None:
            objective = self.value_head_cfg.objective
            out_dim = 1 if objective.type == "mse" else int(objective.vocab_size)
            self.value_head = nn.Linear(config.hidden_size, out_dim)
            if bool(getattr(self.value_head_cfg, "pre_pool_mlp", False)):
                self.value_pre_pool_norm = nn.RMSNorm(
                    config.hidden_size, eps=config.layer_norm_eps
                )
                self.value_pre_pool_mlp = SwiGLU(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    dropout_prob=config.dropout_prob,
                    eps=config.layer_norm_eps,
                )
        else:
            self.value_head = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            input_ids: Long tensor (batch, seq_len) with token IDs.
            attn_mask: Optional mask broadcastable to (batch, num_heads, seq_len, seq_len).
                Use bool (True=keep, False=mask) or additive float mask.

        Returns:
            hidden_states: Tensor (batch, seq_len, hidden_size)
            policy_logits: List[Tensor] of length 4 (binned_ev) or Tensor (batch, 4) (action_policy)
            value_out: Tensor (batch,) for mse or (batch, vocab_size) for cross_entropy; None when disabled
        """
        # input_ids: (B, S)
        B, S = input_ids.shape
        device = input_ids.device

        # Clone to avoid potential CUDA Graphs output-buffer aliasing when compiled
        x = self.tok_emb(input_ids).clone()  # (B, S, H)
        pe = self.pos_emb(S, device)  # (S, H)
        x = x + pe.unsqueeze(0)

        # No embedding pre-norm; position added directly

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        x = self.final_ln(x)

        # Final mean pooling for policy heads
        board_repr = x.mean(dim=1)  # (B, H)
        value_out: torch.Tensor | None = None
        if self.value_head is not None:
            value_tokens = x
            if (
                self.value_head_cfg
                and bool(getattr(self.value_head_cfg, "pre_pool_mlp", False))
                and self.value_pre_pool_mlp is not None
                and self.value_pre_pool_norm is not None
            ):
                value_tokens = value_tokens + self.value_pre_pool_mlp(
                    self.value_pre_pool_norm(value_tokens)
                )
            value_repr = value_tokens.mean(dim=1)
            value_out = self.value_head(value_repr)
            if self.value_head_cfg and self.value_head_cfg.objective.type == "mse":
                value_out = value_out.squeeze(-1)
        if self.head_type == "binned_ev":
            ev_logits = [head(board_repr) for head in self.ev_heads]  # List of (B, n_bins)
            return x, ev_logits, value_out
        else:  # action_policy
            policy_logits = self.policy_head(board_repr)  # (B, 4)
            return x, policy_logits, value_out
