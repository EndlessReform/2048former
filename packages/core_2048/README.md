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

Weights layout (model.safetensors)

The loader infers the head type if `head_type` is missing by inspecting keys:
- If keys start with `policy_head.` → action_policy
- If keys start with `ev_heads.0.` → binned_ev

Additional notes
- Token order is the board’s 16 exponents in row‑major cell order (LSB‑first packing in dataset).
- For Macroxue tokens, the orchestrator’s head order should be `URDL`, and argmax over the WINNER bin (last class) is the default.
- For legacy hard move policy, class order is `UDLR`.

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

