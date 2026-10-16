"""Digital periphery shared by the INT8 reference and the crossbar simulator (D8).

The reference model and the Step 2 simulator differ only in how the
integer accumulator `acc = sum x_int * w_int` is produced. Everything that
happens before and after it lives here, so both use identical code.
"""
import torch


def to_int(x, in_scale):
    """Recover the integer activations from a dequantised UINT8 tensor.

    Inputs to crossbar layers always come out of an ActQuant, so x is
    exactly k * in_scale (in float32) and the rounding is exact.
    Returned as float64 so that later sums of integers are exact (D11).
    """
    return torch.round(x.double() / in_scale.double())


def dequantize_acc(acc, in_scale, w_scale, bias):
    """y = acc * s_x * s_w[c] + b[c], computed in float64, returned as float32.

    acc has the output channel on dim 1 (conv: [N, C, H, W], linear: [N, C]).
    """
    shape = [1, -1] + [1] * (acc.dim() - 2)
    s = (in_scale.double() * w_scale.double()).view(shape)
    y = acc.double() * s + bias.double().view(shape)
    return y.float()


def global_avgpool(x):
    """Global average pool, reduced in float64 so the result does not depend
    on device or reduction order."""
    return x.double().mean(dim=(2, 3)).float()
