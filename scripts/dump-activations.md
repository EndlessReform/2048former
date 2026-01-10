# Dump Activations

This script dumps selected transformer activations for offline analysis.

## Usage

```bash
uv run --locked scripts/dump_activations.py \
  --init inits/hf/2048former-50m-v0.1 \
  --dataset ./datasets/raws/d7_test_v1 \
  --n-samples 4096 \
  --batch-size 1024 \
  --seed 42 \
  --outputs final_hidden,layer_outputs,head_logits,attn_norm,mlp_input \
  -o activations.safetensors \
  --device cuda
```

Benchmark without writing:

```bash
uv run --locked scripts/dump_activations.py \
  --init inits/hf/2048former-50m-v0.1 \
  --dataset ./datasets/raws/d7_test_v1 \
  --n-samples 4096 \
  --batch-size 1024 \
  --outputs final_hidden,layer_outputs \
  --no-write \
  --device cuda
```

## Notes

- CUDA only; `--device` must resolve to `cuda`.
- Output format is inferred from `-o/--output` extension: `.safetensors`, `.pt`, or `.npz`.
- `.npz` does not support `bf16`.
- `attn_weights` is not implemented and will raise.
- Speed summary prints after the run; if `rich` is installed, output is a table.

## Flags

| Flag | Default | Description |
| --- | --- | --- |
| `--init` | required | Checkpoint directory, `.pt` bundle, or `hf://` path. |
| `--dataset` | required | Steps dataset directory (expects `steps.npy` or `steps-*.npy`). |
| `--indices` | — | Optional `.npy` int64 array of step indices. |
| `--n-samples` | 1024 | Number of samples (ignored if `--indices` provided). |
| `--batch-size` | 1024 | Forward-pass batch size. |
| `--seed` | 0 | RNG seed for sampling. |
| `--outputs` | `final_hidden` | Comma-separated activation keys. |
| `-o`, `--output` | `activations.safetensors` | Output path; extension selects format. |
| `--dtype` | `fp32` | Storage dtype: `fp32`, `fp16`, `bf16`. |
| `--no-write` | — | Run inference and discard activations without writing outputs. |
| `--device` | `cuda` | Compute device. |

## Activation keys

| Key | Shape | Description |
| --- | --- | --- |
| `final_hidden` | `(N, 16, H)` | Output of final RMSNorm. |
| `pooled` | `(N, H)` | Mean-pooled board representation. |
| `head_logits` | `(N, 4, n_bins)` | EV head logits for binned EV, or `(N, 4)` for action policy. |
| `head_probs` | same | Softmax of `head_logits`. |
| `layer_outputs` | `(N, L, 16, H)` | Hidden states after each encoder block. |
| `attn_norm` | `(N, L, 16, H)` | Hidden state after attention norm, before QKV projection. |
| `mlp_input` | `(N, L, 16, H)` | Hidden state after MLP norm, before expansion. |
| `rmsnorm_gammas` | per-module | Dumps RMSNorm gamma weights under `rmsnorm_gammas/<module_name>`. |
| `attn_weights` | — | Not implemented. |

## RMSNorm gammas

Use `--outputs rmsnorm_gammas` (or include it with other keys) to dump RMSNorm
weights from the model. Each RMSNorm module is written as its own tensor with
key `rmsnorm_gammas/<module_name>` in the output.

```bash
uv run --locked scripts/dump_activations.py \
  --init inits/hf/2048former-50m-v0.1 \
  --dataset ./datasets/raws/d7_test_v1 \
  --n-samples 1 \
  --outputs rmsnorm_gammas \
  -o activations_with_rmsnorms.safetensors \
  --device cuda
```

## Output schema

### safetensors

```python
from safetensors import safe_open
with safe_open("activations.safetensors", framework="pt") as f:
    final_hidden = f.get_tensor("final_hidden")
    layer_outputs = f.get_tensor("layer_outputs")
    indices = f.get_tensor("indices")
    boards = f.get_tensor("boards")
    metadata = f.metadata()
```

### .pt

```python
data = torch.load("activations.pt")
# data["final_hidden"], data["indices"], data["boards"], data["metadata"]
```

### .npz

```python
data = np.load("activations.npz")
# data["final_hidden"], data["indices"], data["boards"]
# metadata stored as JSON string in data["metadata"]
```
