Inference
---------

Example: choose the move with the highest probability of the '1' bin.

```
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

CLI script to run a game
------------------------

A standalone script lives in `bin/play_2048.py` (kept outside the package).
It runs a single game with the default policy, auto-selects device (mps > cuda > cpu), and supports an optional seed.

```
python bin/play_2048.py --init ./init --seed 123
# or let it pick a random seed
python bin/play_2048.py --init ./init
# optionally override device
python bin/play_2048.py --init ./init --device cpu
```
