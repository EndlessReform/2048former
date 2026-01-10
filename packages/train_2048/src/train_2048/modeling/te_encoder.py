from git.util import HIDE_WINDOWS_KNOWN_ERRORS
import math
import transformer_engine.pytorch as te
import torch
import torch.nn as nn

from core_2048 import EncoderConfig
from core_2048.model import AbsolutePositionalEmbedding

class TEEncoderBlock(nn.Module):
    def __init__(self, config: EncoderConfig):
        """
        TODO: Post-processing script to change key names
        """
        super().__init__()
        self.config = config
        self.layer_norm = te.LayerNorm(config.hidden_size)

        self.attn = te.MultiheadAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_gqa_groups=config.num_key_value_heads,
            attention_dropout=config.attention_dropout_prob,
            # norm
            layernorm_epsilon=config.layer_norm_eps,
            attn_mask_type="no_mask", # Bidirectional full attention,
            # TODO: experiment with this, seems to be one-click
            # zero_centered_gamma=True,
            # Pre-norm
            input_layernorm=True,
            normalization="RMSNorm",
            qk_norm_type="RMSNorm" if config.use_qk_norm else None,
            qk_norm_eps=config.layer_norm_eps,
            softmax_type="learnable" if config.use_attention_sinks else "vanilla",
            bias=False,
        )
        self.mlp = te.LayerNormMLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            eps=config.layer_norm_eps,
            activation="swiglu",
            normalization="RMSNorm",
            bias=False,
        )

    def forward(self, x: torch.Tensor):
        # Norms are fused
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class TEEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.input_vocab_size, config.hidden_size)
        self.pos_emb = AbsolutePositionalEmbedding(
            max_position_embeddings=config.max_position_embeddings,
            hidden_size=config.hidden_size
        )

        # TODO: does TE do bookkeeping for keeping last N blocks in BF16 for NVFP4 or do we have to manage it here?
        self.blocks = nn.ModuleList(
            [TEEncoderBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_ln = te.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Out proj still has to be from torch due to precision issues
        self.head_type = config.head_type
        if self.head_type == "binned_ev":
            assert config.output_n_bins is not None, "output_n_bins must be specified for binned_ev head"
            self.ev_heads = nn.ModuleList(
                [nn.Linear(config.hidden_size, config.output_n_bins) for _ in range(4)]
            )
            self.policy_head = None
        else:
            self.policy_head = nn.Linear(config.hidden_size, 4)
            self.ev_heads = None

    def forward(self, input_ids: torch.Tensor):
        B, S = input_ids.shape
        device = input_ids.device

        # Clone to avoid CUDA graph issues
        x = self.tok_emb(input_ids).clone()
        pe = self.pos_emb(S, device)
        x = x + pe.unsqueeze(0)

        # No pre-norm: handled by blocks
        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)

        # Final mean pool
        board_repr = x.mean(dim=1) # (bsz, hidden_dim)
        if self.head_type == "binned_ev":
            assert self.ev_heads is not None, "ev_heads must be specified for binned_ev head"
            ev_preds = [head(board_repr) for head in self.ev_heads]
            return ev_preds
        elif self.head_type == "action_policy":
            assert self.policy_head is not None, "policy_head must be specified for policy head"
            policy_pred = self.policy_head(board_repr)
            return policy_pred
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")
