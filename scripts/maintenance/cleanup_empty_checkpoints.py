#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


WEIGHT_EXTENSIONS = (".safetensors", ".pt")


def has_weights(path: Path) -> bool:
    return any(p.suffix in WEIGHT_EXTENSIONS for p in path.rglob("*") if p.is_file())


def find_empty_runs(root: Path) -> list[Path]:
    return [
        child
        for child in sorted(root.iterdir())
        if child.is_dir() and not has_weights(child)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove checkpoint run folders that do not contain any weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  uv run --locked scripts/maintenance/cleanup_empty_checkpoints.py \\\n"
            "    checkpoints/ --dry-run\n"
        ),
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="checkpoints",
        help="Root folder containing checkpoint run subfolders (default: checkpoints).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List candidate folders without deleting them.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Delete without confirmation.",
    )
    return parser.parse_args()


def confirm_delete(count: int) -> bool:
    prompt = f"Delete {count} checkpoint folder(s)? [y/N]: "
    response = input(prompt).strip().lower()
    return response in {"y", "yes"}


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")
    if not root.is_dir():
        raise SystemExit(f"Root path is not a directory: {root}")

    candidates = find_empty_runs(root)
    if not candidates:
        print("No empty checkpoint folders found.")
        return

    print("Empty checkpoint folders:")
    for path in candidates:
        print(f"- {path}")

    if args.dry_run:
        print("Dry run enabled; nothing deleted.")
        return

    if not args.yes and not confirm_delete(len(candidates)):
        print("Aborted; nothing deleted.")
        return

    for path in candidates:
        shutil.rmtree(path)
        print(f"Deleted {path}")


if __name__ == "__main__":
    main()
