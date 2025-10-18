# Playing 2048 with transformers

## Inference 

### gRPC Server + Client

Python server (Torch):
- Generate stubs once:

```bash
uv run python -m grpc_tools.protoc -I proto --python_out=packages/infer_2048/src/infer_2048/proto --grpc_python_out=packages/infer_2048/src/infer_2048/proto proto/train_2048/inference/v1/inference.proto
```

Start server over:
- UDS (recommended local): uv run infer-2048 --init inits/v1_50m --uds unix:/tmp/2048_infer.sock --device cuda
- TCP: uv run infer-2048 --init inits/v1_50m --tcp 127.0.0.1:50051 --device cpu

Rust orchestrator client (game-engine):
- Configure one of:
- In config TOML: [orchestrator.connection] uds_path = "/tmp/2048_infer.sock"
- Or TCP: tcp_addr = "http://127.0.0.1:50051"
- Build/run: cargo run -p game-engine -- --config config/inference/top-score.toml

### Python-side inference

Example: choose the move with the highest probability of the '1' bin.

```bash
import torch
from train_2048.config import load_encoder_from_init
from train_2048.inference import infer_move

model = load_encoder_from_init("./init")
board_tokens = torch.tensor([0,0,1,2, 0,0,1,3, 0,0,2,3, 0,0,1,1], dtype=torch.long)
moves, extras = infer_move(model, board_tokens)
print(moves)  # tensor([dir]) where dir in {0:Up,1:Down,2:Left,3:Right}
```

Use with ai_2048 loop (legal masking handled automatically):

```
from ai_2048 import Board, Rng
from train_2048.config import load_encoder_from_init
from train_2048.inference import ModelPolicy

rng = Rng(42)
board = Board.empty().with_random_tile(rng=rng).with_random_tile(rng=rng)
model = load_encoder_from_init("./init")
policy = ModelPolicy(model)

while not board.is_game_over():
    m = policy.best_move(board)
    if m is None:
        break
    board = board.make_move(m, rng=rng)

print("final score:", board.score(), "highest:", board.highest_tile())
```

## CLI script to run a game

A standalone script lives in `bin/play_2048.py` (kept outside the package).
It runs a single game with the default policy, auto-selects device (mps > cuda > cpu), and supports an optional seed.

```
python bin/play_2048.py --init ./init --seed 123
# or let it pick a random seed
python bin/play_2048.py --init ./init
# optionally override device
python bin/play_2048.py --init ./init --device cpu
# adjust progress logging (prints every N moves)
python bin/play_2048.py --init ./init --log-interval 25
```

The CLI prints periodic status lines like:

```
Starting game: device=cuda dtype=torch.bfloat16 seed=123 initial score=0 highest=2
[    50] last=Left  score=   544 highest=  16 mps(window)= 1200.50 mps(avg)= 1180.23
...
Done: device=cuda seed=123 moves=732 time=0.89s avg_mps=823.74 final score=37640 highest=1024
```

Notes
- Dtype: the model runs in its native dtype; no implicit conversion.
- Device: defaults to mps on macOS if available, then cuda, else cpu.

## Dataset (NPY + SQLite)

Build a dataset from `.a2run2` runs using the Rust CLI (see upstream repo):

```bash
cargo run -q -p ai-2048 --bin dataset -- build --input /path/to/runs --out dataset_dir
```

Python usage (NumPy + sqlite3):

```python
from pathlib import Path
import sqlite3
import numpy as np
import ai_2048 as a2

dataset = Path("dataset_dir")
steps = np.load(dataset / "steps.npy")  # structured array (kept in RAM for speed)
conn = sqlite3.connect(dataset / "metadata.db")

# Filter runs via SQL (example: score in [50k, 500k])
rids = [row[0] for row in conn.execute(
    "SELECT id FROM runs WHERE max_score BETWEEN ? AND ?", (50_000, 500_000)
)]
mask = np.isin(steps['run_id'], np.array(rids, dtype=steps['run_id'].dtype))
idxs = np.flatnonzero(mask)

# Batch slice + fast exponent conversion
exps_buf, dirs, evs = a2.batch_from_steps(steps, idxs, parallel=True)
exps = np.frombuffer(exps_buf, dtype=np.uint8).reshape(-1, 16)

# EV encoding (branches, UDLR canonical)
# - Order: Up=0, Down=1, Left=2, Right=3 (UDLR)
# - ev_values are normalized to [0,1]; chosen branch is exactly 1.0
# - ev_legal is a u8 bitmask (UDLR: Up=1, Down=2, Left=4, Right=8)
#   Illegal branches also store 0.0 in ev_values, so use the mask
# Tip: expand mask to (N,4) and zero-out illegal entries (UDLR)
mask_bits = steps['ev_legal'][idxs]
legal = np.stack([
    (mask_bits & 1) != 0,   # Up
    (mask_bits & 2) != 0,   # Down
    (mask_bits & 4) != 0,   # Left
    (mask_bits & 8) != 0,   # Right
], axis=1)
evs = (evs * legal.astype(np.float32))
```

PyTorch DataLoader
```
import torch
from torch.utils.data import Dataset, DataLoader

class StepsDataset(Dataset):
    def __init__(self, dataset_dir: str, run_sql: str | None = None, sql_params: tuple = ()):  # optional SQL filter
        import sqlite3, numpy as np
        self.steps = np.load(f"{dataset_dir}/steps.npy")
        conn = sqlite3.connect(f"{dataset_dir}/metadata.db")
        if run_sql:
            rids = [row[0] for row in conn.execute(run_sql, sql_params)]
            mask = np.isin(self.steps['run_id'], np.array(rids, dtype=self.steps['run_id'].dtype))
            self.indices = np.flatnonzero(mask)
        else:
            self.indices = np.arange(self.steps.shape[0], dtype=np.int64)

    def __len__(self):
        return self.indices.size

    def __getitem__(self, idx):
        return int(self.indices[idx])

def collate_steps(batch_indices, steps):
    import numpy as np, ai_2048 as a2
    idxs = np.array(batch_indices, dtype=np.int64)
    exps_buf, dirs, evs = a2.batch_from_steps(steps, idxs, parallel=True)
    exps = np.frombuffer(exps_buf, dtype=np.uint8).reshape(-1, 16)
    # Apply ev_legal bitmask to build (N,4) legality and zero-out illegal EVs
    bits = steps['ev_legal'][idxs].astype(np.uint8)
    legal = np.stack([(bits & 1)!=0, (bits & 2)!=0, (bits & 4)!=0, (bits & 8)!=0], axis=1)
    evs = evs * legal.astype(np.float32)
    # Convert to tensors
    exps_t = torch.from_numpy(exps.copy()).to(dtype=torch.int64)
    mask_t = torch.from_numpy(legal.astype(np.bool_))
    evs_t  = torch.from_numpy(evs.copy())
    return exps_t, mask_t, evs_t
```

Config changes
- Use `dataset_dir` instead of `packfile`.
- Validation uses disjoint runs via percentage (`val_run_pct`) or explicit SQL (`val_run_sql`).
- See `config/config.example.toml` and `config/long-ctxt-sft.toml` for templates.

## Training

- Install deps once via `uv sync`, then train with `uv run python main.py --config config/config.example.toml` (override `--device` as needed).
- To resume from a previous run, point `init_dir` at either the original init folder or a saved `.pt` bundle (for example `init_dir = "checkpoints/20240901_120000/model-stable.pt"`).
- When a `.pt` bundle is used, the encoder weights, optimizer state, and `global_step` are restored automatically before training continues.
- Checkpoints are still written into the `checkpoint_dir` from the active config, so update it if you want the resumed run to land in a new folder.
- `[checkpoint].save_pt_every_steps` dumps numbered `model-step-XXXXXXXX.pt` bundles that include optimizer state for straight resumes.
- Validation best checkpoints now write `model-best.pt` bundles; convert to safetensors for inference with `uv run python bin/pt_to_safetensors.py <path/to/model-best.pt> --write-config`.
- Learning rate schedulers support `constant`, `warmup-stable-decay`, `cosine`, or `linear` (with optional warmup).
- Gradient accumulation is managed through `batch.micro_batch_size`; the loader emits the micro batch while the trainer accumulates updates until it reaches `batch.batch_size`.
- With `batch.adaptive` enabled (cosine LR only), the harness doubles the effective batch once the learning rate falls to 50% of its peak and quadruples it at 25%, keeping the physical micro batch unchanged.
