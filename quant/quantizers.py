"""Weight and activation quantisers (docs/step1_decisions.md, D3-D5, D12).

All quantisers are symmetric with zero-point 0 and use torch.round
(round half to even), followed by a clamp.
"""
import math

import torch
import torch.nn as nn

W_QMAX = 127   # signed weights live in [-127, 127] (D3); -128 is never used
A_QMAX = 255   # unsigned activations live in [0, 255] (D4)


def _per_channel_view(s, w):
    """Reshape a per-output-channel vector so it broadcasts against w."""
    return s.view(-1, *([1] * (w.dim() - 1)))


def weight_scale(w):
    """Per-output-channel max-abs scale: s[c] = max|W[c]| / 127."""
    amax = w.detach().abs().flatten(1).amax(dim=1)
    return amax.clamp_min(1e-12) / W_QMAX


def quantize_weight(w):
    """Return (w_int int8, w_scale float32) for a conv/linear weight."""
    s = weight_scale(w)
    w_int = torch.round(w.detach() / _per_channel_view(s, w)).clamp(-W_QMAX, W_QMAX)
    return w_int.to(torch.int8), s.float()


def fake_quant_weight(w):
    """Quantise-dequantise with a straight-through estimator (QAT)."""
    s = _per_channel_view(weight_scale(w), w)
    ws = w / s
    q = ws + (torch.round(ws).clamp(-W_QMAX, W_QMAX) - ws).detach()
    return q * s


def fake_quant_act(x, scale, qmax=A_QMAX):
    """Unsigned activation quantise-dequantise with STE inside [0, qmax]."""
    xs = (x / scale).clamp(0, qmax)
    q = xs + (torch.round(xs) - xs).detach()
    return q * scale


class ActQuant(nn.Module):
    """Unsigned per-tensor activation quantiser.

    Modes (not saved in the state dict; set with quant.prepare.set_mode):
      'off'   - pass-through (FP32 training)
      'calib' - pass-through, records the activation percentile (D5)
      'quant' - fake-quantise with the frozen scale
    """

    def __init__(self, fixed_scale=None, percentile=99.99):
        super().__init__()
        self.fixed = fixed_scale is not None
        self.percentile = percentile
        self.register_buffer("scale", torch.tensor(float(fixed_scale or 0.0)))
        self.mode = "off"
        self._calib = []

    def forward(self, x):
        if self.mode == "off":
            return x
        if self.mode == "calib":
            if not self.fixed:
                self._calib.append(self._percentile(x))
            return x
        return fake_quant_act(x, self.scale)

    def _percentile(self, x):
        flat = x.detach().flatten().float()
        k = max(1, math.ceil(self.percentile / 100.0 * flat.numel()))
        return flat.kthvalue(k).values.item()

    def finish_calibration(self):
        if self.fixed:
            return
        if not self._calib:
            raise RuntimeError("ActQuant saw no calibration data")
        p = sum(self._calib) / len(self._calib)
        self.scale.fill_(max(p, 1e-8) / A_QMAX)
        self._calib = []

    def extra_repr(self):
        return f"scale={self.scale.item():.6g}, fixed={self.fixed}, mode={self.mode}"
