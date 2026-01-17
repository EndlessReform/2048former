from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from ..augmentation.rotation import (
    make_rotation_rng,
    rotate_board_exps,
    rotate_branch_udlr,
    rotate_legal_bits,
    rotate_move_dir,
    sample_rotation_k,
)
from ..augmentation.flip import (
    flip_board_exps,
    flip_branch_udlr,
    flip_legal_bits,
    flip_move_dir,
    make_flip_rng,
    sample_flip_axis,
)
from ..tokenization.base import (
    BoardCodec,
)


def _rotation_settings(rotation_augment: Optional[object]) -> tuple[str, Optional[int], bool]:
    if rotation_augment is None:
        return "none", None, True
    mode = getattr(rotation_augment, "mode", "none")
    seed = getattr(rotation_augment, "seed", None)
    allow_noop = bool(getattr(rotation_augment, "allow_noop", True))
    return str(mode), seed, allow_noop


def _flip_settings(flip_augment: Optional[object]) -> tuple[str, Optional[int], bool]:
    if flip_augment is None:
        return "none", None, True
    mode = getattr(flip_augment, "mode", "none")
    seed = getattr(flip_augment, "seed", None)
    allow_noop = bool(getattr(flip_augment, "allow_noop", True))
    return str(mode), seed, allow_noop


def make_collate_macroxue(
    dataset,
    tokenizer_path: str,
    *,
    rotation_augment: Optional[object] = None,
    flip_augment: Optional[object] = None,
) -> Callable:
    """Collate function for macroxue advantage tokenization (v2 spec)."""
    import json
    from pathlib import Path
    from . import steps as steps_module
    from core_2048.tokenization.macroxue import (
        MacroxueTokenizerV2,
        MacroxueTokenizerV2Spec,
    )

    p = Path(tokenizer_path)
    if not p.is_file():
        raise FileNotFoundError(f"Tokenizer spec not found at {p}")
    with open(p) as f:
        payload = json.load(f)

    tokenizer_type = payload.get("tokenizer_type")
    rotation_mode, rotation_seed, rotation_allow_noop = _rotation_settings(rotation_augment)
    flip_mode, flip_seed, flip_allow_noop = _flip_settings(flip_augment)

    if tokenizer_type != "macroxue_ev_advantage_v2":
        raise ValueError(f"Unsupported macroxue tokenizer_type: {tokenizer_type}")

    spec = MacroxueTokenizerV2Spec.from_dict(payload)
    tokenizer = MacroxueTokenizerV2(spec)
    n_classes = len(spec.vocab_order)

    # Get valuation type mapping from dataset string name -> tokenizer integer id
    ds = dataset
    if isinstance(ds, steps_module.StepsDataset):
        try:
            vt_path = Path(ds.dataset_dir) / "valuation_types.json"
            if vt_path.is_file():
                with open(vt_path) as f:
                    # Invert the lookup: name -> id
                    vt_payload = json.load(f)
                    if isinstance(vt_payload, list):
                        vt_name_to_ds_id = {name: i for i, name in enumerate(vt_payload)}
                    elif isinstance(vt_payload, dict):
                        vt_name_to_ds_id = {v: int(k) for k, v in vt_payload.items()}
                    else:
                        raise TypeError("Unsupported valuation_types.json format")
            else:
                # Fallback for older datasets
                vt_name_to_ds_id = {"search": 0, "tuple10": 1, "tuple11": 2}
        except Exception:
            vt_name_to_ds_id = {"search": 0, "tuple10": 1, "tuple11": 2}
    else:
        # Fallback for other dataset types
        vt_name_to_ds_id = {"search": 0, "tuple10": 1, "tuple11": 2}

    def _unpack_board_to_exps_u8(packed, *, mask65536=None):
        return BoardCodec.decode_packed_board_to_exps_u8(packed, mask65536=mask65536)

    # Track if we've warned about missing board_eval
    _warned_missing_board_eval = [False]

    _rotation_rng_holder = [None]
    _flip_rng_holder = [None]

    def _get_rotation_rng():
        if _rotation_rng_holder[0] is None:
            _rotation_rng_holder[0] = make_rotation_rng(rotation_seed)
        return _rotation_rng_holder[0]

    def _get_flip_rng():
        if _flip_rng_holder[0] is None:
            _flip_rng_holder[0] = make_flip_rng(flip_seed)
        return _flip_rng_holder[0]

    def _collate_v2(batch_indices):
        import numpy as _np

        idxs = _np.asarray(batch_indices, dtype=_np.int64)
        batch = dataset.get_rows(idxs)

        if "board" not in batch.dtype.names:
            raise KeyError("Expected 'board' field in steps.npy for macroxue dataset")
        mask65536 = (
            batch["tile_65536_mask"]
            if "tile_65536_mask" in batch.dtype.names
            else None
        )
        exps = _unpack_board_to_exps_u8(batch["board"], mask65536=mask65536)
        branch_evs = batch["branch_evs"]
        valuation_types_ds = batch["valuation_type"].astype(_np.int64, copy=False)
        ev_legal = batch["ev_legal"]
        move_dirs = batch["move_dir"]

        rotation_k = None
        flip_axis = None

        if rotation_mode != "none":
            if rotation_mode != "random_k":
                raise ValueError(f"Unknown rotation_augment mode: {rotation_mode}")
            rotation_k = sample_rotation_k(
                len(idxs),
                rng=_get_rotation_rng(),
                allow_noop=rotation_allow_noop,
            )
            if _np.any(rotation_k != 0):
                exps = rotate_board_exps(exps, rotation_k)
                branch_evs = rotate_branch_udlr(branch_evs, rotation_k)
                ev_legal = rotate_legal_bits(ev_legal, rotation_k)
                move_dirs = rotate_move_dir(move_dirs, rotation_k)

        if flip_mode != "none":
            if flip_mode != "random_axis":
                raise ValueError(f"Unknown flip_augment mode: {flip_mode}")
            flip_axis = sample_flip_axis(
                len(idxs),
                rng=_get_flip_rng(),
                allow_noop=flip_allow_noop,
            )
            if _np.any(flip_axis != 0):
                exps = flip_board_exps(exps, flip_axis)
                branch_evs = flip_branch_udlr(branch_evs, flip_axis)
                ev_legal = flip_legal_bits(ev_legal, flip_axis)
                move_dirs = flip_move_dir(move_dirs, flip_axis)
        tokens = torch.from_numpy(exps.copy()).to(dtype=torch.int64)

        # Check for board_eval field
        has_board_eval = "board_eval" in batch.dtype.names
        if has_board_eval:
            board_evals = batch["board_eval"]
        else:
            # Compute board_eval on the fly
            if not _warned_missing_board_eval[0]:
                import warnings
                warnings.warn(
                    "Dataset missing 'board_eval' field - computing on-the-fly (this may impact performance). "
                    "Consider re-packing dataset with latest Rust tooling.",
                    UserWarning,
                    stacklevel=2
                )
                _warned_missing_board_eval[0] = True

            # Import board eval function
            from core_2048.tokenization.macroxue.board_eval import evaluate_board_batch
            board_evals = evaluate_board_batch(exps)

        if has_board_eval and (rotation_mode != "none" or flip_mode != "none"):
            changed_mask = None
            if rotation_mode != "none":
                rot_mask = rotation_k != 0
                changed_mask = rot_mask if changed_mask is None else (changed_mask | rot_mask)
            if flip_mode != "none":
                flip_mask = flip_axis != 0
                changed_mask = flip_mask if changed_mask is None else (changed_mask | flip_mask)
            if changed_mask is not None and _np.any(changed_mask):
                from core_2048.tokenization.macroxue.board_eval import evaluate_board_batch
                board_evals = board_evals.copy()
                board_evals[changed_mask] = evaluate_board_batch(exps[changed_mask])

        legal_mask = BoardCodec.legal_mask_from_bits_udlr(ev_legal)

        targets = np.zeros((len(idxs), 4), dtype=np.int64)

        for i in range(len(idxs)):
            vt_ds_id = valuation_types_ds[i]
            vt_name = next(
                (name for name, ds_id in vt_name_to_ds_id.items() if ds_id == vt_ds_id),
                None,
            )
            if vt_name is None:
                raise KeyError(f"Unrecognized valuation_type ID from dataset: {vt_ds_id}")

            targets[i, :] = tokenizer.encode_row(
                valuation_type=vt_name,
                branch_evs=branch_evs[i],
                move_dir=move_dirs[i],
                legal_mask=legal_mask[i],
                board_eval=board_evals[i],
            )

        branch_targets = torch.from_numpy(targets.copy()).long()
        branch_mask = torch.from_numpy(legal_mask.astype(_np.bool_, copy=False))
        return {
            "tokens": tokens,
            "branch_targets": branch_targets,
            "branch_mask": branch_mask,
            "targets": branch_targets,
            "n_classes": n_classes,
        }

    return _collate_v2


def make_collate_steps(
    target_mode: str,
    dataset,
    *,
    ev_tokenizer: Optional[object] = None,
    rotation_augment: Optional[object] = None,
    flip_augment: Optional[object] = None,
) -> Callable:
    """Collate function for regular steps datasets (binned_ev or hard_move)."""
    import numpy as _np

    if target_mode not in {"binned_ev", "hard_move"}:
        raise ValueError(f"Unknown target mode: {target_mode}")
    if target_mode == "binned_ev" and ev_tokenizer is None:
        raise ValueError("ev_tokenizer is required for binned_ev mode")

    rotation_mode, rotation_seed, rotation_allow_noop = _rotation_settings(rotation_augment)
    flip_mode, flip_seed, flip_allow_noop = _flip_settings(flip_augment)
    _rotation_rng_holder = [None]
    _flip_rng_holder = [None]

    def _get_rotation_rng():
        if _rotation_rng_holder[0] is None:
            _rotation_rng_holder[0] = make_rotation_rng(rotation_seed)
        return _rotation_rng_holder[0]

    def _get_flip_rng():
        if _flip_rng_holder[0] is None:
            _flip_rng_holder[0] = make_flip_rng(flip_seed)
        return _flip_rng_holder[0]

    def _collate(batch_indices):
        idxs = _np.asarray(batch_indices, dtype=_np.int64)
        batch = dataset.get_rows(idxs)

        # Decode board exponents (dataset packs MSB-first into uint64 + optional 65536 mask)
        if 'board' not in batch.dtype.names:
            raise KeyError("'board' field is required in steps.npy")
        mask65536 = batch['tile_65536_mask'] if 'tile_65536_mask' in batch.dtype.names else None
        # Always decode MSB-first packed boards, then apply 65536 mask
        from ..tokenization.base import BoardCodec as _BC
        exps_np = _BC.decode_packed_board_to_exps_u8(batch['board'], mask65536=mask65536)

        # Branch EVs and legal moves are UDLR in the dataset
        # Support both new ('branch_evs') and old ('ev_values') field names
        if 'branch_evs' in batch.dtype.names:
            evs = batch['branch_evs'].astype(_np.float32, copy=False)
        elif 'ev_values' in batch.dtype.names:
            evs = batch['ev_values'].astype(_np.float32, copy=False)
        else:
            raise KeyError("'branch_evs' or 'ev_values' missing from steps.npy")

        ev_legal_bits = batch['ev_legal'] if 'ev_legal' in batch.dtype.names else None
        rotation_k = None
        flip_axis = None

        if rotation_mode != "none":
            if rotation_mode != "random_k":
                raise ValueError(f"Unknown rotation_augment mode: {rotation_mode}")
            rotation_k = sample_rotation_k(
                len(idxs),
                rng=_get_rotation_rng(),
                allow_noop=rotation_allow_noop,
            )
            if _np.any(rotation_k != 0):
                exps_np = rotate_board_exps(exps_np, rotation_k)
                evs = rotate_branch_udlr(evs, rotation_k)
                if ev_legal_bits is not None:
                    ev_legal_bits = rotate_legal_bits(ev_legal_bits, rotation_k)

        if flip_mode != "none":
            if flip_mode != "random_axis":
                raise ValueError(f"Unknown flip_augment mode: {flip_mode}")
            flip_axis = sample_flip_axis(
                len(idxs),
                rng=_get_flip_rng(),
                allow_noop=flip_allow_noop,
            )
            if _np.any(flip_axis != 0):
                exps_np = flip_board_exps(exps_np, flip_axis)
                evs = flip_branch_udlr(evs, flip_axis)
                if ev_legal_bits is not None:
                    ev_legal_bits = flip_legal_bits(ev_legal_bits, flip_axis)

        tokens = torch.from_numpy(exps_np.copy()).to(dtype=torch.int64)
        legal = (
            BoardCodec.legal_mask_from_bits_udlr(ev_legal_bits)
            if ev_legal_bits is not None
            else _np.isfinite(evs)
        )
        branch_values = torch.from_numpy(evs.copy()).to(dtype=torch.float32)
        branch_mask = torch.from_numpy(legal.astype(_np.bool_, copy=False))

        out = {
            "tokens": tokens,
            "branch_values": branch_values,
        }
        if target_mode == "binned_ev":
            try:
                ev_tokenizer.to(branch_values.device)  # type: ignore[attr-defined]
            except Exception:
                pass
            targets = ev_tokenizer.build_targets(evs=branch_values, legal_mask=branch_mask)  # type: ignore[call-arg]
            branch_targets = targets.get("branch_bin_targets")
            if branch_targets is None:
                raise KeyError("Expected 'branch_bin_targets' from EV tokenizer")
            out["branch_targets"] = branch_targets
            out["branch_mask"] = branch_mask
            if "n_bins" in targets:
                out["n_classes"] = int(targets["n_bins"])
            out.update(targets)
        else:
            # Support both new ('move_dir') and old ('move') label fields
            if 'move_dir' in batch.dtype.names:
                dirs_arr = batch['move_dir'].astype(_np.int64, copy=False)
            elif 'move' in batch.dtype.names:
                dirs_arr = batch['move'].astype(_np.int64, copy=False)
            else:
                raise KeyError("move_dir/move missing from steps.npy for hard_move target")
            if rotation_mode != "none" and rotation_k is not None:
                if _np.any(rotation_k != 0):
                    dirs_arr = rotate_move_dir(dirs_arr, rotation_k)
            if flip_mode != "none" and flip_axis is not None:
                if _np.any(flip_axis != 0):
                    dirs_arr = flip_move_dir(dirs_arr, flip_axis)
            out["move_targets"] = torch.from_numpy(dirs_arr.copy()).to(dtype=torch.long)
        return out

    return _collate


def make_collate_macroxue_worker_safe(
    dataset_dir: str,
    tokenizer_path: str,
    *,
    rotation_augment: Optional[object] = None,
    flip_augment: Optional[object] = None,
) -> Callable:
    """Worker-safe collate that creates its own shard loader per worker."""
    # Import here to avoid circular deps
    from .shard_loader import ShardLoader

    # Each worker will create its own loader (lazy, thread-safe)
    _worker_loader = [None]  # Mutable container to cache per-worker

    def _get_loader():
        if _worker_loader[0] is None:
            _worker_loader[0] = ShardLoader(dataset_dir, mmap_mode=True)
        return _worker_loader[0]

    # Load tokenizer config once
    import json
    from pathlib import Path
    from core_2048.tokenization.macroxue import (
        MacroxueTokenizerV2,
        MacroxueTokenizerV2Spec,
    )

    p = Path(tokenizer_path)
    if not p.is_file():
        raise FileNotFoundError(f"Tokenizer spec not found at {p}")
    with open(p) as f:
        payload = json.load(f)

    tokenizer_type = payload.get("tokenizer_type")
    rotation_mode, rotation_seed, rotation_allow_noop = _rotation_settings(rotation_augment)
    flip_mode, flip_seed, flip_allow_noop = _flip_settings(flip_augment)

    if tokenizer_type != "macroxue_ev_advantage_v2":
        raise ValueError(f"Unsupported macroxue tokenizer_type: {tokenizer_type}")

    spec = MacroxueTokenizerV2Spec.from_dict(payload)
    tokenizer = MacroxueTokenizerV2(spec)
    n_classes = len(spec.vocab_order)

    # Load valuation type mapping
    try:
        vt_path = Path(dataset_dir) / "valuation_types.json"
        if vt_path.is_file():
            with open(vt_path) as f:
                vt_payload = json.load(f)
                if isinstance(vt_payload, list):
                    vt_name_to_ds_id = {name: i for i, name in enumerate(vt_payload)}
                elif isinstance(vt_payload, dict):
                    vt_name_to_ds_id = {v: int(k) for k, v in vt_payload.items()}
                else:
                    raise TypeError("Unsupported valuation_types.json format")
        else:
            vt_name_to_ds_id = {"search": 0, "tuple10": 1, "tuple11": 2}
    except Exception:
        vt_name_to_ds_id = {"search": 0, "tuple10": 1, "tuple11": 2}

    _warned_missing_board_eval = [False]
    _rotation_rng_holder = [None]
    _flip_rng_holder = [None]

    def _get_rotation_rng():
        if _rotation_rng_holder[0] is None:
            _rotation_rng_holder[0] = make_rotation_rng(rotation_seed)
        return _rotation_rng_holder[0]

    def _get_flip_rng():
        if _flip_rng_holder[0] is None:
            _flip_rng_holder[0] = make_flip_rng(flip_seed)
        return _flip_rng_holder[0]

    def _collate(batch_indices):
        import numpy as _np

        loader = _get_loader()
        idxs = _np.asarray(batch_indices, dtype=_np.int64)
        batch = loader.get_rows(idxs)

        if "board" not in batch.dtype.names:
            raise KeyError("Expected 'board' field in steps.npy")
        mask65536 = batch["tile_65536_mask"] if "tile_65536_mask" in batch.dtype.names else None
        exps = BoardCodec.decode_packed_board_to_exps_u8(batch["board"], mask65536=mask65536)
        branch_evs = batch["branch_evs"]
        valuation_types_ds = batch["valuation_type"].astype(_np.int64, copy=False)
        ev_legal = batch["ev_legal"]
        move_dirs = batch["move_dir"]

        rotation_k = None
        flip_axis = None

        if rotation_mode != "none":
            if rotation_mode != "random_k":
                raise ValueError(f"Unknown rotation_augment mode: {rotation_mode}")
            rotation_k = sample_rotation_k(
                len(idxs),
                rng=_get_rotation_rng(),
                allow_noop=rotation_allow_noop,
            )
            if _np.any(rotation_k != 0):
                exps = rotate_board_exps(exps, rotation_k)
                branch_evs = rotate_branch_udlr(branch_evs, rotation_k)
                ev_legal = rotate_legal_bits(ev_legal, rotation_k)
                move_dirs = rotate_move_dir(move_dirs, rotation_k)

        if flip_mode != "none":
            if flip_mode != "random_axis":
                raise ValueError(f"Unknown flip_augment mode: {flip_mode}")
            flip_axis = sample_flip_axis(
                len(idxs),
                rng=_get_flip_rng(),
                allow_noop=flip_allow_noop,
            )
            if _np.any(flip_axis != 0):
                exps = flip_board_exps(exps, flip_axis)
                branch_evs = flip_branch_udlr(branch_evs, flip_axis)
                ev_legal = flip_legal_bits(ev_legal, flip_axis)
                move_dirs = flip_move_dir(move_dirs, flip_axis)
        tokens = torch.from_numpy(exps.copy()).to(dtype=torch.int64)

        has_board_eval = "board_eval" in batch.dtype.names
        if has_board_eval:
            board_evals = batch["board_eval"]
        else:
            if not _warned_missing_board_eval[0]:
                import warnings
                warnings.warn(
                    "Dataset missing 'board_eval' field - computing on-the-fly",
                    UserWarning,
                    stacklevel=2
                )
                _warned_missing_board_eval[0] = True
            from core_2048.tokenization.macroxue.board_eval import evaluate_board_batch
            board_evals = evaluate_board_batch(exps)

        if has_board_eval and (rotation_mode != "none" or flip_mode != "none"):
            changed_mask = None
            if rotation_mode != "none":
                rot_mask = rotation_k != 0
                changed_mask = rot_mask if changed_mask is None else (changed_mask | rot_mask)
            if flip_mode != "none":
                flip_mask = flip_axis != 0
                changed_mask = flip_mask if changed_mask is None else (changed_mask | flip_mask)
            if changed_mask is not None and _np.any(changed_mask):
                from core_2048.tokenization.macroxue.board_eval import evaluate_board_batch
                board_evals = board_evals.copy()
                board_evals[changed_mask] = evaluate_board_batch(exps[changed_mask])

        legal_mask = BoardCodec.legal_mask_from_bits_udlr(ev_legal)
        targets = np.zeros((len(idxs), 4), dtype=np.int64)

        for i in range(len(idxs)):
            vt_ds_id = valuation_types_ds[i]
            vt_name = next(
                (name for name, ds_id in vt_name_to_ds_id.items() if ds_id == vt_ds_id),
                None,
            )
            if vt_name is None:
                raise KeyError(f"Unrecognized valuation_type ID: {vt_ds_id}")

            targets[i, :] = tokenizer.encode_row(
                valuation_type=vt_name,
                branch_evs=branch_evs[i],
                move_dir=move_dirs[i],
                legal_mask=legal_mask[i],
                board_eval=board_evals[i],
            )

        branch_targets = torch.from_numpy(targets.copy()).long()
        branch_mask = torch.from_numpy(legal_mask.astype(_np.bool_, copy=False))
        return {
            "tokens": tokens,
            "branch_targets": branch_targets,
            "branch_mask": branch_mask,
            "targets": branch_targets,
            "n_classes": n_classes,
        }

    return _collate


def make_collate_steps_worker_safe(
    dataset_dir: str,
    target_mode: str,
    *,
    ev_tokenizer: Optional[object] = None,
    rotation_augment: Optional[object] = None,
    flip_augment: Optional[object] = None,
) -> Callable:
    """Worker-safe collate for regular steps datasets."""
    from .shard_loader import ShardLoader

    _worker_loader = [None]

    def _get_loader():
        if _worker_loader[0] is None:
            _worker_loader[0] = ShardLoader(dataset_dir, mmap_mode=True)
        return _worker_loader[0]

    if target_mode not in {"binned_ev", "hard_move"}:
        raise ValueError(f"Unknown target mode: {target_mode}")
    if target_mode == "binned_ev" and ev_tokenizer is None:
        raise ValueError("ev_tokenizer is required for binned_ev mode")

    rotation_mode, rotation_seed, rotation_allow_noop = _rotation_settings(rotation_augment)
    flip_mode, flip_seed, flip_allow_noop = _flip_settings(flip_augment)
    _rotation_rng_holder = [None]
    _flip_rng_holder = [None]

    def _get_rotation_rng():
        if _rotation_rng_holder[0] is None:
            _rotation_rng_holder[0] = make_rotation_rng(rotation_seed)
        return _rotation_rng_holder[0]

    def _get_flip_rng():
        if _flip_rng_holder[0] is None:
            _flip_rng_holder[0] = make_flip_rng(flip_seed)
        return _flip_rng_holder[0]

    def _collate(batch_indices):
        import numpy as _np

        loader = _get_loader()
        idxs = _np.asarray(batch_indices, dtype=_np.int64)
        batch = loader.get_rows(idxs)

        if 'board' not in batch.dtype.names:
            raise KeyError("'board' field required")
        mask65536 = batch['tile_65536_mask'] if 'tile_65536_mask' in batch.dtype.names else None
        exps_np = BoardCodec.decode_packed_board_to_exps_u8(batch['board'], mask65536=mask65536)

        if 'branch_evs' in batch.dtype.names:
            evs = batch['branch_evs'].astype(_np.float32, copy=False)
        elif 'ev_values' in batch.dtype.names:
            evs = batch['ev_values'].astype(_np.float32, copy=False)
        else:
            raise KeyError("'branch_evs' or 'ev_values' missing")

        ev_legal_bits = batch['ev_legal'] if 'ev_legal' in batch.dtype.names else None
        rotation_k = None
        flip_axis = None

        if rotation_mode != "none":
            if rotation_mode != "random_k":
                raise ValueError(f"Unknown rotation_augment mode: {rotation_mode}")
            rotation_k = sample_rotation_k(
                len(idxs),
                rng=_get_rotation_rng(),
                allow_noop=rotation_allow_noop,
            )
            if _np.any(rotation_k != 0):
                exps_np = rotate_board_exps(exps_np, rotation_k)
                evs = rotate_branch_udlr(evs, rotation_k)
                if ev_legal_bits is not None:
                    ev_legal_bits = rotate_legal_bits(ev_legal_bits, rotation_k)

        if flip_mode != "none":
            if flip_mode != "random_axis":
                raise ValueError(f"Unknown flip_augment mode: {flip_mode}")
            flip_axis = sample_flip_axis(
                len(idxs),
                rng=_get_flip_rng(),
                allow_noop=flip_allow_noop,
            )
            if _np.any(flip_axis != 0):
                exps_np = flip_board_exps(exps_np, flip_axis)
                evs = flip_branch_udlr(evs, flip_axis)
                if ev_legal_bits is not None:
                    ev_legal_bits = flip_legal_bits(ev_legal_bits, flip_axis)

        tokens = torch.from_numpy(exps_np.copy()).to(dtype=torch.int64)
        legal = (
            BoardCodec.legal_mask_from_bits_udlr(ev_legal_bits)
            if ev_legal_bits is not None
            else _np.isfinite(evs)
        )
        branch_values = torch.from_numpy(evs.copy()).to(dtype=torch.float32)
        branch_mask = torch.from_numpy(legal.astype(_np.bool_, copy=False))

        out = {
            "tokens": tokens,
            "branch_values": branch_values,
        }

        if target_mode == "binned_ev":
            try:
                ev_tokenizer.to(branch_values.device)
            except Exception:
                pass
            targets = ev_tokenizer.build_targets(evs=branch_values, legal_mask=branch_mask)
            branch_targets = targets.get("branch_bin_targets")
            if branch_targets is None:
                raise KeyError("Expected 'branch_bin_targets' from EV tokenizer")
            out["branch_targets"] = branch_targets
            out["branch_mask"] = branch_mask
            if "n_bins" in targets:
                out["n_classes"] = int(targets["n_bins"])
            out.update(targets)
        else:
            if 'move_dir' in batch.dtype.names:
                dirs_arr = batch['move_dir'].astype(_np.int64, copy=False)
            elif 'move' in batch.dtype.names:
                dirs_arr = batch['move'].astype(_np.int64, copy=False)
            else:
                raise KeyError("move_dir/move missing")
            if rotation_mode != "none" and rotation_k is not None:
                if _np.any(rotation_k != 0):
                    dirs_arr = rotate_move_dir(dirs_arr, rotation_k)
            if flip_mode != "none" and flip_axis is not None:
                if _np.any(flip_axis != 0):
                    dirs_arr = flip_move_dir(dirs_arr, flip_axis)
            out["move_targets"] = torch.from_numpy(dirs_arr.copy()).to(dtype=torch.long)

        return out

    return _collate


__all__ = [
    "make_collate_macroxue",
    "make_collate_steps",
    "make_collate_macroxue_worker_safe",
    "make_collate_steps_worker_safe",
]
