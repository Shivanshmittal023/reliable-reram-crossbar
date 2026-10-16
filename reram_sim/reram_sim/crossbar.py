"""One ReRAM crossbar holding a weight matrix W (out x in), simulated in PyTorch.

Mapping of a weight to cells (programmed once, then used for inference):
  1. Quantise W per output row to signed w_bits integers q in [-qmax, qmax].
  2. Split q into a differential pair: q = q_pos - q_neg, both non-negative.
  3. Bit-slice each magnitude into base-L digits, L = 2**bits_per_cell
     (SLC: 1 bit per cell, MLC: 2+ bits per cell). Digit s sits in its own column.
  4. Digit d is programmed as conductance g = g_min + d * delta, with
     delta = (g_max - g_min) / (L - 1) and g_max = 1 (all conductances are
     normalised to G_max).
  5. Programming noise: g += sigma * G_max * N(0, 1), clamped at 0.
     Stuck-at faults: a cell is stuck at G_max (SA1) or G_min (SA0) with
     probability fault_rate, whatever it was meant to hold.

Inference: the analog column current for slice s is x · (g_pos - g_neg); dividing
by delta gives the (noisy) signed digit. Slices are combined digitally by
shift-and-add (x L**s) and rescaled by the row scale.

The same sigma hurts MLC more than SLC: MLC levels are closer together
(smaller delta), so the same absolute conductance error is a larger digit error.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CrossbarConfig:
    bits_per_cell: int = 1           # 1 = SLC, 2/3 = MLC
    sigma: float = 0.0               # programming noise std, as a fraction of G_max
    fault_rate: float = 0.0          # probability that a cell is stuck
    sa1_frac: float = 0.5            # share of stuck cells stuck at G_max (rest at G_min)
    on_off: float = 100.0            # G_max / G_min
    w_bits: int = 8                  # signed weight precision (sign + w_bits-1 magnitude bits)
    in_bits: int | None = 8          # DAC precision (signed, per-tensor scale); None = ideal input
    adc_bits: int | None = None      # ADC precision per slice and tile; None = ideal readout
    tile_rows: int = 128             # crossbar rows; inputs beyond this go to another tile

    @property
    def levels(self) -> int:
        return 2 ** self.bits_per_cell

    @property
    def n_slices(self) -> int:
        return math.ceil((self.w_bits - 1) / self.bits_per_cell)


def quantize_symmetric(x: torch.Tensor, bits: int, dim=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Round x to signed integers in [-qmax, qmax]. Returns (integers as float, scale)."""
    qmax = 2 ** (bits - 1) - 1
    absmax = x.abs().amax() if dim is None else x.abs().amax(dim=dim, keepdim=True)
    scale = absmax.clamp(min=1e-12) / qmax
    return torch.round(x / scale).clamp(-qmax, qmax), scale


class Crossbar(nn.Module):
    """y = x @ W.T computed through simulated ReRAM cells. Call `program` before use."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.detach().clone().float())
        # Set by program(): per-row weight scale, then either the merged weight
        # (ideal readout) or the per-slice signed digits (ADC readout).
        for name in ("scale", "w_hat", "digits"):
            self.register_buffer(name, None, persistent=False)
        self.cfg: CrossbarConfig | None = None

    @torch.no_grad()
    def program(self, cfg: CrossbarConfig, gen: torch.Generator | None = None) -> None:
        """Write the weights into cells: quantise, bit-slice, add noise and faults."""
        self.cfg = cfg
        q, scale = quantize_symmetric(self.weight, cfg.w_bits, dim=1)
        self.scale = scale
        mag = {"pos": q.clamp(min=0), "neg": (-q).clamp(min=0)}

        L = cfg.levels
        g_min = 1.0 / cfg.on_off
        delta = (1.0 - g_min) / (L - 1)
        digits = []
        for s in range(cfg.n_slices):
            g = {k: self._cells(torch.div(m, L ** s, rounding_mode="floor") % L, cfg, g_min, delta, gen)
                 for k, m in mag.items()}
            digits.append((g["pos"] - g["neg"]) / delta)

        if cfg.adc_bits is None:
            # Ideal readout is linear, so the slices can be merged once here.
            w_int = sum(L ** s * d for s, d in enumerate(digits))
            self.w_hat, self.digits = w_int * scale, None
        else:
            self.w_hat, self.digits = None, torch.stack(digits)

    @staticmethod
    def _cells(d: torch.Tensor, cfg: CrossbarConfig, g_min: float, delta: float,
               gen: torch.Generator | None) -> torch.Tensor:
        g = g_min + d * delta
        if cfg.sigma > 0:
            g = (g + cfg.sigma * torch.randn(g.shape, generator=gen, device=g.device)).clamp(min=0)
        if cfg.fault_rate > 0:
            stuck = torch.rand(g.shape, generator=gen, device=g.device) < cfg.fault_rate
            sa1 = torch.rand(g.shape, generator=gen, device=g.device) < cfg.sa1_frac
            g = torch.where(stuck, torch.where(sa1, torch.ones_like(g), torch.full_like(g, g_min)), g)
        return g

    @torch.no_grad()
    def effective_weight(self) -> torch.Tensor:
        """The weight the chip actually holds (after quantisation, noise and faults)."""
        if self.w_hat is not None:
            return self.w_hat
        L = self.cfg.levels
        return sum(L ** s * d for s, d in enumerate(self.digits)) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg is None:
            raise RuntimeError("Crossbar.program() must be called before forward()")
        cfg = self.cfg
        if cfg.in_bits is not None:
            x, x_scale = quantize_symmetric(x, cfg.in_bits)
        else:
            x_scale = 1.0
        if cfg.adc_bits is None:
            return F.linear(x, self.w_hat) * x_scale
        return self._forward_adc(x) * x_scale

    def _forward_adc(self, x: torch.Tensor) -> torch.Tensor:
        """Per slice and per tile of tile_rows inputs: analog MAC, then ADC, then shift-and-add.

        The ADC range is set to the largest |partial sum| in the batch for that slice
        and tile (an ideally calibrated ADC).
        """
        cfg = self.cfg
        R = cfg.tile_rows
        n_tiles = math.ceil(self.in_features / R)
        pad = n_tiles * R - self.in_features
        xt = F.pad(x, (0, pad)).unflatten(-1, (n_tiles, R))              # (..., T, R)
        qmax = 2 ** (cfg.adc_bits - 1) - 1
        y = 0
        for s, d in enumerate(self.digits):
            dt = F.pad(d, (0, pad)).unflatten(-1, (n_tiles, R))         # (out, T, R)
            part = torch.einsum("...tr,otr->...ot", xt, dt)              # (..., out, T)
            rng = part.abs().flatten(0, -2).amax(0).clamp(min=1e-12)     # (T,)
            step = rng / qmax
            part = torch.round(part / step).clamp(-qmax, qmax) * step
            y = y + cfg.levels ** s * part.sum(-1)
        return y * self.scale.squeeze(1)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, cfg={self.cfg}"
