"""Split SVD components into SLC (important) and MLC (rest) and write them to disk.

Output layout (one file per layer in each tier folder):
    <out>/SLC/<layer>.pt   {"indices", "U", "SVh", "importance"}   important components
    <out>/MLC/<layer>.pt   {"indices", "U", "SVh", "importance"}   remaining components
    <out>/tier_map.json    per-layer shapes, rank, SLC/MLC indices
Layer output on hardware = U_slc @ (SVh_slc @ x) + U_mlc @ (SVh_mlc @ x) + bias.
`indices` are positions in the original rank order (descending σ), so the two
tiers can be merged back exactly.
"""
import json
import math
from pathlib import Path

import torch

from .svd_linear import SVDLinear

TIERS = ("SLC", "MLC")


def select_slc(scores: dict[str, torch.Tensor], slc_frac: float, scope: str) -> dict[str, torch.Tensor]:
    """Boolean SLC mask per layer.

    scope="layer":  top ceil(p*k) components inside every layer.
    scope="global": top p of all components across layers (layers compete).
    """
    if scope == "layer":
        masks = {}
        for name, s in scores.items():
            n_slc = math.ceil(slc_frac * s.numel())
            mask = torch.zeros_like(s, dtype=torch.bool)
            mask[torch.topk(s, n_slc).indices] = True
            masks[name] = mask
        return masks

    flat = torch.cat(list(scores.values()))
    n_slc = math.ceil(slc_frac * flat.numel())
    threshold = torch.topk(flat, n_slc).values.min()
    return {name: s >= threshold for name, s in scores.items()}


def save_tiers(out_dir: Path, layers: dict[str, SVDLinear], scores: dict[str, torch.Tensor],
               masks: dict[str, torch.Tensor]) -> dict:
    for tier in TIERS:
        (out_dir / tier).mkdir(parents=True, exist_ok=True)

    tier_map = {}
    for name, layer in layers.items():
        U, SVh = layer.crossbar_factors()
        entry = {"out_features": layer.out_features, "in_features": layer.in_features, "rank": layer.rank}
        for tier, mask in (("SLC", masks[name]), ("MLC", ~masks[name])):
            idx = mask.nonzero().flatten()
            torch.save({"indices": idx, "U": U[:, idx].contiguous(), "SVh": SVh[idx].contiguous(),
                        "importance": scores[name][idx]}, out_dir / tier / f"{name}.pt")
            entry[tier] = {"count": idx.numel(), "indices": idx.tolist(),
                           "U_shape": [layer.out_features, idx.numel()],
                           "SVh_shape": [idx.numel(), layer.in_features]}
        tier_map[name] = entry

    with open(out_dir / "tier_map.json", "w") as f:
        json.dump(tier_map, f, indent=1)
    return tier_map


def load_layer(out_dir: Path, name: str) -> dict[str, dict]:
    return {tier: torch.load(out_dir / tier / f"{name}.pt") for tier in TIERS}


def reconstruct_weight(out_dir: Path, name: str) -> torch.Tensor:
    """W_eff = U_slc @ SVh_slc + U_mlc @ SVh_mlc (the two tiers summed back)."""
    parts = load_layer(out_dir, name)
    return sum(p["U"] @ p["SVh"] for p in parts.values())
