"""Metadata database utilities - trust the DB completely."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Sequence, Tuple
import numpy as np


class MetadataDB:
    """Interface to metadata.db for run selection.

    Trust the metadata DB completely - no verification against actual data.
    The DB is the source of truth for run counts, step counts, etc.
    """

    def __init__(self, dataset_dir: str):
        self.db_path = Path(dataset_dir) / "metadata.db"
        if not self.db_path.is_file():
            raise FileNotFoundError(f"metadata.db not found at {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def get_all_run_ids(self) -> np.ndarray:
        """Return all run IDs from metadata."""
        with self._connect() as conn:
            rows = conn.execute("SELECT id FROM runs").fetchall()
            return np.array([r[0] for r in rows], dtype=np.uint32)

    def get_run_ids_by_sql(self, sql: str, params: Sequence | None = None) -> np.ndarray:
        """Execute custom SQL to select run IDs."""
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params or ())).fetchall()
            return np.array([r[0] for r in rows], dtype=np.uint32)

    def get_total_steps_for_runs(self, run_ids: Optional[np.ndarray] = None) -> int:
        """Get total step count for given runs (or all runs if None).

        Tries column names in order: steps, num_steps, num_moves (back-compat).
        """
        with self._connect() as conn:
            # Find the right column name
            step_col = None
            for col in ("steps", "num_steps", "num_moves"):
                try:
                    conn.execute(f"SELECT {col} FROM runs LIMIT 1")
                    step_col = col
                    break
                except sqlite3.OperationalError:
                    continue

            if step_col is None:
                return 0

            rows = conn.execute(f"SELECT id, {step_col} FROM runs").fetchall()

            if run_ids is None:
                return sum(cnt or 0 for _, cnt in rows)

            run_set = set(run_ids.tolist())
            return sum(cnt or 0 for rid, cnt in rows if rid in run_set)

    def split_runs_train_val(
        self,
        *,
        run_sql: Optional[str] = None,
        sql_params: Sequence | None = None,
        val_run_sql: Optional[str] = None,
        val_sql_params: Sequence | None = None,
        val_run_pct: float = 0.0,
        val_split_seed: int = 42,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Split runs into train/val sets.

        Returns (train_run_ids, val_run_ids_or_None).

        Logic:
        1. If val_run_sql provided: use it for val, rest for train
        2. Elif val_run_pct > 0: random split
        3. Else: all runs for train, no val
        """
        # Get universe of runs
        if run_sql:
            universe = self.get_run_ids_by_sql(run_sql, sql_params)
        else:
            universe = self.get_all_run_ids()

        # Explicit validation set
        if val_run_sql:
            val_ids = self.get_run_ids_by_sql(val_run_sql, val_sql_params)
            val_ids = np.intersect1d(val_ids, universe)
            train_ids = np.setdiff1d(universe, val_ids)
            return train_ids, val_ids

        # Percentage split
        if val_run_pct > 0.0:
            universe_unique = np.unique(universe)
            rng = np.random.default_rng(val_split_seed)
            n_val = max(1, int(np.ceil(val_run_pct * len(universe_unique))))
            perm = rng.permutation(len(universe_unique))
            val_indices = np.sort(perm[:n_val])
            val_ids = universe_unique[val_indices]
            train_ids = np.setdiff1d(universe_unique, val_ids)
            return train_ids, val_ids

        # No validation split
        return universe, None

    def get_run_count(self) -> int:
        """Total number of runs."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

    def __repr__(self) -> str:
        return f"MetadataDB({self.db_path})"
