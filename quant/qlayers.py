"""Crossbar-mappable layers: QConv2d and QLinear.

Each layer runs in one of three modes (set with quant.prepare.set_mode):
  'fp'        - plain float layer (FP32 training, BN folding, calibration)
  'fakequant' - INT8 fake-quantised weights with STE (QAT)
  'int'       - integer reference: exact integer MAC + shared periphery (D10)

In 'int' mode the MAC can be replaced through `mac_backend`, a callable
`fn(layer, x_int) -> acc`. This is the hook the Step 2 crossbar simulator
plugs into; everything else stays identical.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import periphery
from .quantizers import fake_quant_weight


class _QMixin:
    def _init_quant_buffers(self):
        out_ch = self.weight.shape[0]
        self.register_buffer("w_int", torch.zeros_like(self.weight, dtype=torch.int8))
        self.register_buffer("w_scale", torch.zeros(out_ch))
        self.register_buffer("in_scale", torch.tensor(0.0))
        self.mode = "fp"
        self.mac_backend = None

    def _int_forward(self, x):
        x_int = periphery.to_int(x, self.in_scale)
        if self.mac_backend is not None:
            acc = self.mac_backend(self, x_int)
        else:
            acc = self.exact_mac(x_int)
        return periphery.dequantize_acc(acc, self.in_scale, self.w_scale, self.bias)

    def crossbar_matrix(self):
        """w_int as the crossbar matrix [rows = in*kh*kw, cols = out] (D13)."""
        return self.w_int.reshape(self.w_int.shape[0], -1).t()


class QConv2d(_QMixin, nn.Conv2d):
    # Bias is always present: before folding it is redundant with BN,
    # after folding it carries the BN shift. Keeps one structure (D6).
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=True)
        self._init_quant_buffers()

    def exact_mac(self, x_int):
        return F.conv2d(x_int, self.w_int.double(), None, self.stride, self.padding)

    def forward(self, x):
        if self.mode == "fp":
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        if self.mode == "fakequant":
            return F.conv2d(x, fake_quant_weight(self.weight), self.bias, self.stride, self.padding)
        return self._int_forward(x)


class QLinear(_QMixin, nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)
        self._init_quant_buffers()

    def exact_mac(self, x_int):
        return F.linear(x_int, self.w_int.double())

    def forward(self, x):
        if self.mode == "fp":
            return F.linear(x, self.weight, self.bias)
        if self.mode == "fakequant":
            return F.linear(x, fake_quant_weight(self.weight), self.bias)
        return self._int_forward(x)
