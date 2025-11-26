Core 2048 Model Init Spec

This document defines the minimal JSON schema for an init directory that `core_2048` can load via `load_encoder_from_init(<init_dir>)`. An init directory contains:

- config.json — model architecture and head specification
- model.safetensors — optional weights (if absent, random initialization is used)

config.json fields (EncoderConfig)

Required
- input_vocab_size (int) — token vocabulary size for board exponents. Common values:
  - 16 (0..15) for datasets without 65536 tiles
  - 17 (0..16) when 65536 tiles are possible and encoded as exponent 16
- hidden_size (int)
- num_hidden_layers (int)
- num_attention_heads (int)
- intermediate_size (int)
- layer_norm_eps (float)
- max_position_embeddings (int) — must be ≥ 16 (the board token length)

Output head configuration (choose one)
1) Binned EV heads (four independent heads, one per move)
   - head_type = "binned_ev"
   - output_n_bins (int) — number of classes per head. For Macroxue tokenization this equals loser_bins + 2 (winner + illegal). Example: 32 loser bins → 34 classes.

2) Action policy head (single 4‑way head over moves)
   - head_type = "action_policy"
   - output_n_bins must be omitted or null

Optional
- dropout_prob (float, default 0.1)
- attention_dropout_prob (float, default 0.0)
- num_key_value_heads (int|null) — for GQA; defaults to num_attention_heads when null
- value_head (object|null) — optional scalar/value readout on the mean-pooled board embedding
  - enabled (bool, default true when the block is present)
  - pooling (string, default "mean"; only mean pooling is supported)
  - pre_pool_mlp (bool, default false) — add a SwiGLU MLP before pooling the value readout
  - objective (object)
    - type (string) — "mse" or "cross_entropy"
    - vocab_size (int) — required when type="cross_entropy"; output dimension
    - vocab_type (string) — required when type="cross_entropy"; label for the value vocab/bins

Weights layout (model.safetensors)

The loader infers the head type if `head_type` is missing by inspecting keys:
- If keys start with `policy_head.` → action_policy
- If keys start with `ev_heads.0.` → binned_ev
- If keys include `value_head.` → value head enabled (objective inferred from out_features)

Additional notes
- Token order is the board’s 16 exponents in row‑major cell order using MSB‑first nibble packing in a u64.
- Canonical branch/head order is UDLR (Up, Down, Left, Right) across dataset, training, and inference.
- `value_head` is backward compatible: omit it entirely to match older policy-only configs.

Examples

Action policy (hard move)
{
  "input_vocab_size": 17,
  "hidden_size": 288,
  "num_hidden_layers": 10,
  "num_attention_heads": 6,
  "intermediate_size": 1152,
  "layer_norm_eps": 1e-6,
  "dropout_prob": 0.1,
  "num_key_value_heads": 2,
  "attention_dropout_prob": 0.0,
  "max_position_embeddings": 16,
  "head_type": "action_policy"
}

Binned EV (macroxue tokens, 32 loser bins → 34 classes)
{
  "input_vocab_size": 17,
  "hidden_size": 512,
  "num_hidden_layers": 12,
  "num_attention_heads": 8,
  "intermediate_size": 2048,
  "layer_norm_eps": 1e-6,
  "dropout_prob": 0.1,
  "max_position_embeddings": 16,
  "head_type": "binned_ev",
  "output_n_bins": 34
}

Binned EV + value (discrete cross-entropy over return bins)
{
  "input_vocab_size": 17,
  "hidden_size": 512,
  "num_hidden_layers": 12,
  "num_attention_heads": 8,
  "intermediate_size": 2048,
  "layer_norm_eps": 1e-6,
  "dropout_prob": 0.1,
  "max_position_embeddings": 16,
  "head_type": "binned_ev",
  "output_n_bins": 34,
  "value_head": {
    "pooling": "mean",
    "objective": {
      "type": "cross_entropy",
      "vocab_size": 601,
      "vocab_type": "return_bins"
    }
  }
}
