"""Tiered ReRAM linear layer: SVD components split over SLC and MLC crossbars.

    y = U_slc (SVh_slc x) + U_mlc (SVh_mlc x) + bias

Each tier is two crossbars back to back: SVh (k_tier x in) then U (out x k_tier).
The intermediate h is digital (ADC out, DAC in again).
"""
import torch
import torch.nn as nn

from .crossbar import Crossbar, CrossbarConfig

TIERS = ("SLC", "MLC")


class ReRAMLinear(nn.Module):
    def __init__(self, U: torch.Tensor, SVh: torch.Tensor, slc_mask: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.in_features = SVh.shape[1]
        self.out_features = U.shape[0]
        self.tiers = nn.ModuleDict()
        for tier, mask in zip(TIERS, (slc_mask, ~slc_mask)):
            idx = mask.nonzero().flatten()
            if idx.numel():
                self.tiers[tier] = nn.Sequential(Crossbar(SVh[idx]), Crossbar(U[:, idx]))
        self.register_buffer("bias", None if bias is None else bias.detach().clone().float())

    @classmethod
    def from_linear(cls, linear: nn.Linear, slc_mask_fn, rank: int | None = None) -> "ReRAMLinear":
        """SVD an nn.Linear (full rank by default, so nothing is lost) and tier it.

        slc_mask_fn(S) -> bool mask over the rank components, e.g. top-p by σ.
        """
        U, S, Vh = torch.linalg.svd(linear.weight.detach().float(), full_matrices=False)
        k = S.numel() if rank is None else rank
        U, S, Vh = U[:, :k], S[:k], Vh[:k]
        return cls(U, S[:, None] * Vh, slc_mask_fn(S), linear.bias)

    def program(self, cfgs: dict[str, CrossbarConfig], gen: torch.Generator | None = None) -> None:
        for tier, pair in self.tiers.items():
            for xbar in pair:
                xbar.program(cfgs[tier], gen)

    @torch.no_grad()
    def effective_weight(self) -> torch.Tensor:
        """Weight the layer computes with when inputs and ADC are ideal."""
        return sum(pair[1].effective_weight() @ pair[0].effective_weight() for pair in self.tiers.values())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = sum(pair(x) for pair in self.tiers.values())
        return y if self.bias is None else y + self.bias

    def extra_repr(self) -> str:
        counts = {t: p[0].out_features for t, p in self.tiers.items()}
        return f"in={self.in_features}, out={self.out_features}, components={counts}"


def replace_modules(model: nn.Module, build: dict[str, nn.Module]) -> None:
    """Swap model.<name> for build[name] in place."""
    for name, new in build.items():
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, new)


def program_model(model: nn.Module, cfgs: dict[str, CrossbarConfig], seed: int) -> None:
    """Program every ReRAMLinear in module order from one seeded generator (one simulated chip)."""
    device = next(model.buffers()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    for m in model.modules():
        if isinstance(m, ReRAMLinear):
            m.program(cfgs, gen)
