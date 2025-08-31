from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderConfig(BaseModel):
    input_vocab_size: int = 16
    output_n_bins: int
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


class EncoderAttention(nn.Module):
    """
    LLaMA 3-style self-attention with GQA using F.scaled_dot_product_attention.

    Notes:
    - Absolute positional encoding is expected to be added outside this module.
    - Implements grouped-query attention (GQA): queries have `num_attention_heads`,
      keys/values have `num_key_value_heads`. K/V are expanded across query groups
      by repeat-interleave along the head dimension.
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

        # Single projection for QKV with GQA-compatible output size
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        self.wqkv = nn.Linear(self.hidden_size, q_size + 2 * kv_size, bias=False)

        self.wo = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.attn_dropout_p = float(
            getattr(config, "attention_dropout_prob", 0.0) or 0.0
        )
        self.resid_dropout = nn.Dropout(config.dropout_prob)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def _shape_qkv(self, x: torch.Tensor):
        # x: (B, S, H)
        B, S, _ = x.size()
        qkv = self.wqkv(x)  # (B, S, q + k + v)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

        # Reshape to (B, S, n_heads, head_dim) for q
        q = q.view(B, S, self.num_heads, self.head_dim)
        # Reshape to (B, S, n_kv_heads, head_dim) for k and v
        k = k.view(B, S, self.num_kv_heads, self.head_dim)
        v = v.view(B, S, self.num_kv_heads, self.head_dim)

        # Transpose to (B, n_*, S, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand K/V across query groups by interleaving heads
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        return q, k, v  # shapes: (B, n_heads, S, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ):
        # Expect x as (B, S, H)
        residual = x

        q, k, v = self._shape_qkv(x)

        # Use PyTorch scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )  # (B, n_heads, S, head_dim)

        # Merge heads back: (B, S, H)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(x.size(0), x.size(1), self.hidden_size)
        )

        out = self.wo(attn_out)
        out = self.resid_dropout(out)
        out = self.layer_norm(residual + out)
        return out


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: silu(Wg x) * (Wu x) -> Wd"""

    def __init__(
        self, hidden_size: int, intermediate_size: int, dropout_prob: float, eps: float
    ):
        super().__init__()
        self.w_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        up = self.w_up(x)
        gate = self.w_gate(x)
        act = F.silu(gate) * up
        out = self.w_down(act)
        out = self.dropout(out)
        return self.layer_norm(residual + out)


class EncoderBlock(nn.Module):
    """Transformer encoder block: GQA attention + SwiGLU MLP (pre-existing absolute PE expected externally)."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.attn = EncoderAttention(config)
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout_prob=config.dropout_prob,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ):
        # Encoder uses non-causal (bidirectional) self-attention.
        x = self.attn(x, attn_mask=attn_mask, is_causal=False)
        x = self.mlp(x)
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
        # Pre-norm right after embeddings (token + position)
        self.emb_pre_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.blocks = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.num_hidden_layers)]
        )
        # Optional final LN (kept simple; can be removed if not desired)
        self.final_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ev_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.output_n_bins) for _ in range(4)]
        )

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
            ev_logits: List[Tensor] of length 4, each (batch, output_n_bins)
        """
        # input_ids: (B, S)
        B, S = input_ids.shape
        device = input_ids.device

        # Clone to avoid potential CUDA Graphs output-buffer aliasing when compiled
        x = self.tok_emb(input_ids).clone()  # (B, S, H)
        pe = self.pos_emb(S, device)  # (S, H)
        x = x + pe.unsqueeze(0)

        # Pre-norm after embeddings as requested
        x = self.emb_pre_ln(x)

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        x = self.final_ln(x)

        # Final mean pooling
        board_repr = x.mean(dim=1)  # (B, H)
        ev_logits = [head(board_repr) for head in self.ev_heads]  # List of (B, n_bins)
        return x, ev_logits
