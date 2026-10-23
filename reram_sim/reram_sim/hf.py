"""Build ReRAM versions of the SVD-tiered transformers from svd_tiering outputs.

Reads outputs/<model>/{svd_model.pt, tier_map.json, importance.pt} written by
../svd_tiering and replaces every factorised encoder layer with a ReRAMLinear.
The SLC/MLC split is recomputed from the saved importance so that other
policies (all-SLC, all-MLC, magnitude, random) use the same factorised weights.
"""
import json
import sys
from pathlib import Path

import torch

SVD_ROOT = Path(__file__).resolve().parents[2] / "svd_tiering"
sys.path.insert(0, str(SVD_ROOT))

from svd_tiering import MODELS, SVDLinear, load_model, select_slc  # noqa: E402
from svd_tiering.data import evaluate, sst2_loader  # noqa: E402

from .layer import ReRAMLinear, replace_modules  # noqa: E402

POLICIES = ("all_slc", "all_mlc", "gradient", "magnitude", "random")
DEFAULT_OUTPUTS = SVD_ROOT / "outputs"


def load_svd_model(key: str, out_dir: Path):
    """HF model with SVDLinear layers holding the fine-tuned factors from svd_model.pt."""
    model, tok = load_model(key)
    tier_map = json.loads((out_dir / "tier_map.json").read_text())
    shells = {}
    for name, e in tier_map.items():
        linear = model.get_submodule(name)
        k, m, n = e["rank"], e["out_features"], e["in_features"]
        bias = torch.zeros(m) if linear.bias is not None else None
        shells[name] = SVDLinear(torch.zeros(m, k), torch.zeros(k), torch.zeros(k, n), bias)
    replace_modules(model, shells)
    model.load_state_dict(torch.load(out_dir / "svd_model.pt", map_location="cpu"))
    model.eval()
    return model, tok, shells, tier_map


def tier_masks(layers: dict[str, SVDLinear], importance: dict, policy: str,
               slc_frac: float, seed: int = 0) -> dict[str, torch.Tensor]:
    if policy == "all_slc":
        return {n: torch.ones(l.rank, dtype=torch.bool) for n, l in layers.items()}
    if policy == "all_mlc":
        return {n: torch.zeros(l.rank, dtype=torch.bool) for n, l in layers.items()}
    if policy in ("gradient", "magnitude"):
        scores = importance[policy]
    elif policy == "random":
        g = torch.Generator().manual_seed(seed)
        scores = {n: torch.rand(l.rank, generator=g) for n, l in layers.items()}
    else:
        raise ValueError(f"unknown policy {policy!r}; choose from {POLICIES}")
    return select_slc(scores, slc_frac, "layer")


def to_reram(model, layers: dict[str, SVDLinear], masks: dict[str, torch.Tensor]) -> dict[str, ReRAMLinear]:
    """Replace each SVDLinear with a ReRAMLinear split by `masks`. Call program_model() after."""
    built = {}
    for name, svd in layers.items():
        U, SVh = svd.crossbar_factors()
        built[name] = ReRAMLinear(U, SVh, masks[name], svd.bias)
    replace_modules(model, built)
    return built


__all__ = ["MODELS", "POLICIES", "DEFAULT_OUTPUTS", "load_svd_model", "tier_masks", "to_reram",
           "evaluate", "sst2_loader"]
