from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from ..binning import Binner
from ..tokenization.base import (
    BoardCodec,
)


def make_collate_macroxue(dataset, tokenizer_path: str) -> Callable:
    """Collate function for macroxue tokenization (v1 or v2 spec)."""
    import json
    from pathlib import Path
    from . import steps as steps_module
    from ..tokenization.macroxue.tokenizer_v2 import (
        MacroxueTokenizerV2,
        MacroxueTokenizerV2Spec,
    )
    # v1 spec for fallback
    from ..tokenization.macroxue import MacroxueTokenizerSpec as MacroxueTokenizerV1Spec

    p = Path(tokenizer_path)
    if not p.is_file():
        raise FileNotFoundError(f"Tokenizer spec not found at {p}")
    with open(p) as f:
        payload = json.load(f)

    tokenizer_type = payload.get("tokenizer_type")

    # V2 TOKENIZER PATH
    if tokenizer_type == "macroxue_ev_advantage_v2":
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
            tokens = torch.from_numpy(exps.copy()).to(dtype=torch.int64)

            branch_evs = batch["branch_evs"]
            valuation_types_ds = batch["valuation_type"].astype(_np.int64, copy=False)
            ev_legal = batch["ev_legal"]
            move_dirs = batch["move_dir"]
            board_evals = (
                batch["board_eval"] if "board_eval" in batch.dtype.names else None
            )

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
                    board_eval=board_evals[i] if board_evals is not None else None,
                )

            return {
                "tokens": tokens,
                "targets": torch.from_numpy(targets.copy()).long(),
                "n_classes": n_classes,
            }

        return _collate_v2
    
    # V1 TOKENIZER PATH (ORIGINAL IMPLEMENTATION)
    else:
        spec = MacroxueTokenizerV1Spec.from_dict(payload)
        knots = {vt: np.array(k) for vt, k in spec.ecdf_knots.items()}
        delta_edges = np.array(spec.delta_edges)
        # Number of margin bins inferred from edges
        n_bins = len(delta_edges) - 1
        # Class layout (keep semantics consistent with v1 bins: worst -> best):
        #   0                 : ILLEGAL
        #   1 .. n_bins      : margin bins (worst .. best)
        #   n_bins + 1       : WINNER (p1)
        illegal_token = 0
        winner_token = n_bins + 1

        # Precompute valuation type mapping (dataset id -> spec index).
        vt_name_to_spec_idx = {name: i for i, name in enumerate(spec.valuation_types)}
        ds_vt_mapping: Optional[dict[int, int]] = None
        try:
            import json as _json
            vt_path = Path(dataset.dataset_dir) / "valuation_types.json"
            if vt_path.is_file():
                payload = _json.loads(vt_path.read_text())
                if isinstance(payload, list):
                    ds_id_to_name = {int(i): str(name) for i, name in enumerate(payload)}
                elif isinstance(payload, dict):
                    ds_id_to_name = {int(k): str(v) for k, v in payload.items()}
                else:
                    raise TypeError("valuation_types.json must be list or dict")
                tmp_map: dict[int, int] = {}
                for ds_id, name in ds_id_to_name.items():
                    if name not in vt_name_to_spec_idx:
                        raise KeyError(
                            f"Valuation type '{name}' (id {ds_id}) missing from tokenizer spec"
                        )
                    tmp_map[ds_id] = vt_name_to_spec_idx[name]
                ds_vt_mapping = tmp_map
        except Exception:
            ds_vt_mapping = None

        def _unpack_board_to_exps_u8(packed, *, mask65536=None):
            return BoardCodec.decode_packed_board_to_exps_u8(packed, mask65536=mask65536)

        def _collate_v1(batch_indices):
            import numpy as _np

            idxs = _np.asarray(batch_indices, dtype=_np.int64)
            batch = dataset.get_rows(idxs)

            if 'board' not in batch.dtype.names:
                raise KeyError("Expected 'board' field in steps.npy for macroxue dataset")
            mask65536 = batch['tile_65536_mask'] if 'tile_65536_mask' in batch.dtype.names else None
            exps = _unpack_board_to_exps_u8(batch['board'], mask65536=mask65536)
            tokens = torch.from_numpy(exps.copy()).to(dtype=torch.int64)

            branch_evs = batch["branch_evs"]
            valuation_types = batch["valuation_type"].astype(_np.int64, copy=False)
            ev_legal = batch["ev_legal"]

            percentiles = np.zeros_like(branch_evs, dtype=np.float32)

            # If dataset->spec mapping is available, remap ids; else assume identity
            if ds_vt_mapping is not None:
                vt_spec_ids = _np.empty_like(valuation_types)
                if valuation_types.size:
                    uniq = _np.unique(valuation_types)
                    for ds_id in uniq.tolist():
                        if ds_id not in ds_vt_mapping:
                            raise KeyError(f"Dataset valuation_type id {ds_id} missing from mapping")
                        vt_spec_ids[valuation_types == ds_id] = int(ds_vt_mapping[ds_id])
            else:
                vt_spec_ids = valuation_types

            for vt_name, vt_id in vt_name_to_spec_idx.items():
                mask = vt_spec_ids == vt_id
                if not np.any(mask):
                    continue
                vt_knots = knots[vt_name]
                evs = branch_evs[mask]
                idx = np.searchsorted(vt_knots, evs, side="right")
                n = len(vt_knots) - 1
                p = np.zeros_like(evs, dtype=np.float32)
                valid = (idx > 0) & (idx < len(vt_knots))
                if np.any(valid):
                    lo = vt_knots[idx[valid] - 1]
                    hi = vt_knots[idx[valid]]
                    ratio = np.divide(
                        evs[valid] - lo,
                        hi - lo,
                        out=np.zeros_like(evs[valid]),
                        where=(hi > lo),
                    )
                    p[valid] = (idx[valid] - 1 + ratio) / n
                p[evs <= vt_knots[0]] = 0.0
                p[evs >= vt_knots[-1]] = 1.0
                percentiles[mask] = np.clip(p, 0.0, 1.0)

            # Dataset now stores UDLR, no reordering needed
            legal_mask = BoardCodec.legal_mask_from_bits_udlr(ev_legal)

            percentiles = percentiles.astype(np.float32, copy=False)
            percentiles[~legal_mask] = -np.inf

            winner_indices = np.argmax(percentiles, axis=1)
            winner_percentiles = np.max(percentiles, axis=1, keepdims=True)
            deltas = np.clip(winner_percentiles - percentiles, 0, 1)
            margin_bins = np.searchsorted(delta_edges, deltas, side="right") - 1
            margin_bins = np.clip(margin_bins, 0, n_bins - 1)
            # Reorder margin bins so that indices increase from worst->best.
            # Original macroxue bins are 0=best .. n_bins-1=worst.
            # Map to 1..n_bins with 1=worst .. n_bins=best to align with v1 semantics.
            margin_bins_inv = (n_bins - 1) - margin_bins  # 0..n_bins-1 (worst..best)
            class_bins = 1 + margin_bins_inv               # 1..n_bins

            targets = class_bins.astype(np.int64, copy=False)
            # Mark illegal moves explicitly
            targets[~legal_mask] = illegal_token
            # Assign winner token only for rows with at least one legal move
            rows = np.arange(len(idxs), dtype=np.int64)
            any_legal = legal_mask.any(axis=1)
            if np.any(any_legal):
                r_sel = rows[any_legal]
                w_sel = winner_indices[any_legal]
                targets[r_sel, w_sel] = winner_token

            return {
                "tokens": tokens,
                "targets": torch.from_numpy(targets.copy()).long(),
                # n_classes = margin bins + ILLEGAL + WINNER
                "n_classes": n_bins + 2,
            }

        return _collate_v1


def make_collate_steps(target_mode: str, dataset, binner: Optional[Binner], *, ev_tokenizer: Optional[object] = None) -> Callable:
    """Collate function for regular steps datasets (binned_ev or hard_move)."""
    import numpy as _np

    if target_mode not in {"binned_ev", "hard_move"}:
        raise ValueError(f"Unknown target mode: {target_mode}")

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
        tokens = torch.from_numpy(exps_np.copy()).to(dtype=torch.int64)

        # Branch EVs and legal moves are UDLR in the dataset
        # Support both new ('branch_evs') and old ('ev_values') field names
        if 'branch_evs' in batch.dtype.names:
            evs = batch['branch_evs'].astype(_np.float32, copy=False)
        elif 'ev_values' in batch.dtype.names:
            evs = batch['ev_values'].astype(_np.float32, copy=False)
        else:
            raise KeyError("'branch_evs' or 'ev_values' missing from steps.npy")
        legal = BoardCodec.legal_mask_from_bits_udlr(batch['ev_legal']) if 'ev_legal' in batch.dtype.names else _np.isfinite(evs)
        branch_values = torch.from_numpy(evs.copy()).to(dtype=torch.float32)
        branch_mask = torch.from_numpy(legal.astype(_np.bool_, copy=False))

        out = {
            "tokens": tokens,
            "branch_values": branch_values,
        }
        if target_mode == "binned_ev":
            if ev_tokenizer is not None:
                try:
                    ev_tokenizer.to(branch_values.device)  # type: ignore[attr-defined]
                except Exception:
                    pass
                targets = ev_tokenizer.build_targets(evs=branch_values, legal_mask=branch_mask)  # type: ignore[call-arg]
                out.update(targets)
            else:
                assert binner is not None
                binner.to_device(branch_values.device)
                out["branch_bin_targets"] = binner.bin_values(branch_values).long()
                out["n_bins"] = int(binner.n_bins)
        else:
            # Support both new ('move_dir') and old ('move') label fields
            if 'move_dir' in batch.dtype.names:
                dirs_arr = batch['move_dir'].astype(_np.int64, copy=False)
            elif 'move' in batch.dtype.names:
                dirs_arr = batch['move'].astype(_np.int64, copy=False)
            else:
                raise KeyError("move_dir/move missing from steps.npy for hard_move target")
            out["move_targets"] = torch.from_numpy(dirs_arr.copy()).to(dtype=torch.long)
        return out

    return _collate


__all__ = [
    "make_collate_macroxue",
    "make_collate_steps",
]
