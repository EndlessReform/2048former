#!/usr/bin/env python3
"""Test that training starts without OOM."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "packages/train_2048/src"))

from train_2048.dataloader.steps_v2 import build_steps_dataloaders

def main():
    dataset_dir = "./datasets/d6_3b_v1"
    tokenizer_path = "./out/macroxue_v2/tokenizer_16.json"

    print(f"Building dataloaders...")
    start = time.time()

    dl_train, dl_val, per_epoch_steps, metadata = build_steps_dataloaders(
        dataset_dir=dataset_dir,
        target_mode="macroxue_tokens",
        batch_size=1024,
        tokenizer_path=tokenizer_path,
        train_num_steps=100_000,
        num_workers_train=12,  # Will be overridden to 0 for .zst
        shard_locality=True,
        shard_cache_in_memory=True,
        val_num_steps=200,
        val_run_pct=0.005,
    )

    elapsed = time.time() - start
    print(f"\n✓ Dataloaders created in {elapsed:.2f}s")
    print(f"  Train batches: {len(dl_train)}")
    print(f"  Val batches: {len(dl_val) if dl_val else 0}")
    print(f"  Train workers: {dl_train.num_workers}")

    # Try to get one batch
    print(f"\nFetching first batch...")
    start = time.time()
    batch = next(iter(dl_train))
    elapsed = time.time() - start
    print(f"✓ First batch fetched in {elapsed:.2f}s")
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  Input shape: {batch['input_ids'].shape}")

if __name__ == "__main__":
    main()
