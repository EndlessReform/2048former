# Dataloader Recovery Plan

## Preserved State

The inherited mixed work is preserved as commit `5544643` on branch
`archive-low-hanging-fruit-wip-20260725`. Active repair starts from
`origin/low-hanging-fruit` at `ab3f660`, which includes the 65536 and autocast
fixes without the unfinished feature bundle.

Do not begin a serious training run until both phases below are complete.

## Phase 1: Validation Correctness (Complete)

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
- low-level tests cover the split, filtering, fixed ordering, disabled
  augmentation, bounded materialization, and unsupported-mode failures using
  only kilobyte-scale temporary files.

For the normal compressed, tmpfs-backed, shard-local training path, validation
may materialize a fixed filtered subset from one shard. This is scientifically
acceptable because all shards in a source pool are generated with the same
algorithm and board depth. The selected rows must still be filtered by held-out
run ID; "first shard" alone is not a validation split.

## Phase 2: Exact Restart (Complete)

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

Implementation status on `fix/dataloader-correctness`:

- shard permutations derive independently from `(training seed, epoch, shard)`;
- each queued sample carries a compact cursor for the next row, and the training
  loop commits only the cursor returned by the completed optimizer step;
- cursor tests advance the producer beyond consumed data and prove exact
  continuation across shard and epoch boundaries;
- a real two-worker, eight-batch-prefetch test proves exact continuation with
  rotation and flip enabled;
- rotation and flip are stateless functions of training seed and sample cursor,
  so worker scheduling does not alter augmented inputs after restart;
- resumable bundles include Python, NumPy, Torch CPU/CUDA, and gradient-scaler
  state and are written to a temporary filename followed by atomic rename;
- scheduler base learning rates survive optimizer serialization, avoiding
  double decay after restart, with a configured-LR fallback for old bundles;
- SIGINT/SIGTERM finishes the current optimizer step and writes one atomic
  `model-interrupted.pt`, unless that step already wrote its configured periodic
  checkpoint; a second signal aborts immediately;
- tiny CPU dropout-model tests compare uninterrupted and resumed parameters
  exactly;
- a real-data CPU integration test trains a one-layer model on the 9.7 MiB
  `d6_test_v2` SQLite + NPY pool, validates, resumes from its midpoint bundle,
  and requires bit-identical final parameters;
- DataLoader generators are isolated from the model RNG so constructing a new
  iterator after resume cannot shift dropout;
- full project bundles explicitly opt out of PyTorch's tensor-only loader,
  allowing optimizer and NumPy RNG state to be reopened on current PyTorch;
- fresh runs seed Python, NumPy, Torch CPU, and Torch CUDA before model or data
  initialization, so `training.seed` governs more than sampler order;
- objective setup derives head widths from configuration/tokenizer metadata and
  never probes the one-shot resumable training iterator;
- interrupt handling preserves configured dataset symlinks and limits shutdown
  mutation to the intended atomic checkpoint.

The combined validation/restart baseline is 41 passing low-level tests plus the
real-data CPU integration test. Its 10-step and 100-step forms both complete in
about one second here, generate about 0.55 MiB of temporary checkpoint data in
`/dev/shm`, and remove it afterward. CUDA RNG state is captured and restored in
code; end-to-end CUDA parameter equivalence is not part of the low-write local
suite and remains a short smoke test before the next production run.

## Flash-Wear Constraint

Correctness tests use small temporary synthetic datasets and must not copy or
rewrite production pools. Restart tests use tiny CPU models and small checkpoint
payloads; they must not repeatedly write production-scale checkpoints.

Production checkpoint cadence remains coarse and configurable. Lightweight
metadata or cursor state must not trigger a full model checkpoint. Full bundles
are written atomically only at intentional periodic, best-model,
graceful-stop, or final boundaries.

The restart implementation does not add periodic writes. Cursor state remains
in memory between existing checkpoint boundaries. Tests constrain generated
checkpoint fixtures to less than one megabyte.

## Deferred Feature Order

With both phases green, deferred feature work proceeds in this order:

1. Separate interrupted-run resume from starting a new phase from weights.
2. Restore and repair weighted merge, including emitted-row metadata and source
   provenance.
3. Restore and test batched reannotation.
4. Evaluate TailAgg/MaxP1 changes in named configs on identical game seeds.
5. Restore operational tmpfs cleanup independently.

Exact sampler restart is Phase 2, not deferred experimental work.

## Known Baseline Debt

- The dataset packer's ordered parallel writer can accumulate an unbounded
  out-of-order result map when an early run is much slower than later runs.
  This behavior already exists on `master`; repair it with the weighted-merge
  work, where packer memory and provenance are reviewed together.
- CUDA/TransformerEngine parameter equivalence and a real compressed
  multi-shard restart remain pre-production smoke tests rather than local CI.
