"""Switch a model between FP32, fake-quant and integer-reference modes."""
import torch

from .qlayers import QConv2d, QLinear
from .quantizers import ActQuant, quantize_weight

MODES = ("fp", "fakequant", "int")


def act_quants(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, ActQuant)}


def set_mode(model, mode):
    """'fp': float layers, quantisers off. 'fakequant': QAT. 'int': integer reference.
    'int' requires freeze_int() (or loading an export) first."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    for m in model.modules():
        if isinstance(m, (QConv2d, QLinear)):
            m.mode = mode
        elif isinstance(m, ActQuant):
            m.mode = "off" if mode == "fp" else "quant"
    return model


@torch.no_grad()
def calibrate(model, loader, n_batches, device):
    """D5: record the 99.99th percentile of every activation on `n_batches`
    batches of the float (folded) model, then freeze the scales."""
    set_mode(model, "fp")
    for q in act_quants(model).values():
        q.mode = "calib"
    model.eval()
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(x.to(device))
    for q in act_quants(model).values():
        q.finish_calibration()
    set_mode(model, "fakequant")
    return model


@torch.no_grad()
def freeze_int(model):
    """Fill w_int / w_scale / in_scale from the current float weights and
    activation scales, and switch to the integer reference."""
    quants = act_quants(model)
    for _, layer, qname in model.crossbar_layers():
        w_int, w_scale = quantize_weight(layer.weight)
        layer.w_int.copy_(w_int)
        layer.w_scale.copy_(w_scale)
        layer.in_scale.copy_(quants[qname].scale)
    return set_mode(model, "int")
