# Dataloader Recovery Plan

## Preserved State

The inherited mixed work is preserved as commit `5544643` on branch
`archive-low-hanging-fruit-wip-20260725`. Active repair starts from
`origin/low-hanging-fruit` at `ab3f660`, which includes the 65536 and autocast
fixes without the unfinished feature bundle.

Do not begin a serious training run until both phases below are complete.

## Phase 1: Validation Correctness

Establish a trustworthy validation path before evaluating annealing,
weighted-dataset, reannotation, or learning-rate experiments.

Required behavior:

1. Training and validation run IDs are disjoint.
2. Every validation row belongs to the selected validation run set.
3. Validation rows and order are deterministic for a fixed seed.
4. Validation never uses random rotation or flip augmentation.
5. Loader modes that cannot honor a requested run split fail explicitly.
6. Low-level pytest coverage exercises these properties with a tiny synthetic
   SQLite + NPY packed dataset.

Implementation status on `fix/dataloader-correctness`:

- validation is materialized once in RAM from the first physical shard that
  contains selected held-out run IDs;
- the fixed rows are deterministically sampled with `seed + 1` and reused;
- validation collate is constructed with rotation and flip disabled;
- validation runs in-process so the fixed array is not copied to workers;
- non-shard-local training now rejects a requested run split because those
  samplers cannot enforce `train_run_ids`;
- uncapped validation rejects the configuration instead of scanning or
  materializing an entire production pool;
- the low-level baseline is 25 passing tests using only kilobyte-scale temporary
  files.

For the normal compressed, tmpfs-backed, shard-local training path, validation
may materialize a fixed filtered subset from one shard. This is scientifically
acceptable because all shards in a source pool are generated with the same
algorithm and board depth. The selected rows must still be filtered by held-out
run ID; "first shard" alone is not a validation split.

## Phase 2: Exact Restart

Frequent interruption makes restart behavior part of the sampling protocol.
DataLoader prefetch advances the sampler producer beyond batches consumed by
the optimizer, so producer state is not a valid committed restart cursor.

The repaired contract is checkpoint-boundary exactness:

- model, optimizer, global step, and schedule describe the same completed
  optimizer step;
- a cursor carried through the consumed batch identifies the next unconsumed
  training row;
- sampler state is committed only after a successful optimizer step, including
  the final microbatch under gradient accumulation;
- dataset and split fingerprints are validated on resume;
- Python, NumPy, Torch CPU/CUDA, augmentation, and scaler state needed for
  deterministic continuation are persisted;
- SIGINT/SIGTERM requests a stop, finishes a safe optimizer-step boundary, and
  writes an atomic resumable checkpoint;
- hard termination falls back to the latest periodic atomic checkpoint.

Tests must compare uninterrupted and interrupted/resumed sample sequences with
zero workers, worker prefetch, gradient accumulation, shard boundaries, epoch
boundaries, and run filtering.

## Flash-Wear Constraint

Correctness tests use small temporary synthetic datasets and must not copy or
rewrite production pools. Restart tests use tiny CPU models and small checkpoint
payloads; they must not repeatedly write production-scale checkpoints.

Production checkpoint cadence remains coarse and configurable. Lightweight
metadata or cursor state must not trigger a full model checkpoint. Full bundles
are written atomically only at intentional periodic, best-model,
graceful-stop, or final boundaries.

## Deferred Feature Order

After both phases establish a green baseline:

1. Separate interrupted-run resume from starting a new phase from weights.
2. Restore and repair weighted merge, including emitted-row metadata and source
   provenance.
3. Restore and test batched reannotation.
4. Evaluate TailAgg/MaxP1 changes in named configs on identical game seeds.
5. Restore operational tmpfs cleanup independently.

Exact sampler restart is Phase 2, not deferred experimental work.
