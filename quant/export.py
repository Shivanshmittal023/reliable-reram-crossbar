"""Export / load the INT8 golden model (D13, D14).

The export holds, for every crossbar layer in forward order:
name, type, w_int, w_scale, bias, in_scale, in_quant, stride, padding,
plus the scale of every activation quantiser. The graph itself is implied
by `arch` (see models/<arch>.crossbar_layers), so loading needs the repo.
"""
import json

import numpy as np
import torch

from models import build_model
from .prepare import act_quants, set_mode
from .qlayers import QConv2d

FORMAT_VERSION = 1


def export_int8(model, arch, meta, stem):
    """Write `<stem>.pt` and `<stem>.npz`. `model` must be frozen (freeze_int)."""
    layers = []
    for name, layer, qname in model.crossbar_layers():
        is_conv = isinstance(layer, QConv2d)
        layers.append({
            "name": name,
            "type": "conv" if is_conv else "linear",
            "w_int": layer.w_int.detach().cpu().clone(),
            "w_scale": layer.w_scale.detach().cpu().clone(),
            "bias": layer.bias.detach().cpu().float().clone(),
            "in_scale": float(layer.in_scale),
            "in_quant": qname,
            "stride": list(layer.stride) if is_conv else None,
            "padding": list(layer.padding) if is_conv else None,
        })
    blob = {
        "format_version": FORMAT_VERSION,
        "arch": arch,
        "meta": dict(meta),
        "act_scales": {n: float(q.scale) for n, q in act_quants(model).items()},
        "layers": layers,
    }
    torch.save(blob, f"{stem}.pt")
    _save_npz(blob, f"{stem}.npz")
    return blob


def _save_npz(blob, path):
    arrays = {}
    header = {k: blob[k] for k in ("format_version", "arch", "meta", "act_scales")}
    header["layers"] = []
    for L in blob["layers"]:
        header["layers"].append({k: v for k, v in L.items() if not torch.is_tensor(v)})
        for k in ("w_int", "w_scale", "bias"):
            arrays[f"{L['name']}/{k}"] = L[k].numpy()
    arrays["header_json"] = np.array(json.dumps(header))
    np.savez(path, **arrays)


def load_int8(path):
    return torch.load(path, map_location="cpu", weights_only=True)


@torch.no_grad()
def build_int_model(blob, device="cpu"):
    """Rebuild the integer-reference model from an export blob."""
    model = build_model(blob["arch"], folded=True)
    quants = act_quants(model)
    for name, scale in blob["act_scales"].items():
        quants[name].scale.fill_(scale)
    layers = {name: layer for name, layer, _ in model.crossbar_layers()}
    if set(layers) != {L["name"] for L in blob["layers"]}:
        raise RuntimeError("export does not match the model's crossbar layers")
    for L in blob["layers"]:
        layer = layers[L["name"]]
        layer.w_int.copy_(L["w_int"])
        layer.w_scale.copy_(L["w_scale"])
        layer.bias.copy_(L["bias"])
        layer.in_scale.fill_(L["in_scale"])
        # Keep the float weight consistent so fakequant mode reproduces w_int.
        shape = [-1] + [1] * (L["w_int"].dim() - 1)
        layer.weight.copy_(L["w_int"].float() * L["w_scale"].view(shape))
    return set_mode(model.to(device).eval(), "int")
