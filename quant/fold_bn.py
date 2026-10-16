"""Fold BatchNorm into the preceding conv (D6).

Convention: a conv named `convX` is followed by a BatchNorm named `bnX`
in the same parent module (e.g. conv1/bn1, conv2/bn2).
"""
import copy

import torch
import torch.nn as nn


@torch.no_grad()
def _fold_pair(conv, bn):
    std = torch.sqrt(bn.running_var + bn.eps)
    gamma = bn.weight if bn.affine else torch.ones_like(std)
    beta = bn.bias if bn.affine else torch.zeros_like(std)
    k = gamma / std
    conv.weight.mul_(k.view(-1, 1, 1, 1))
    b = conv.bias if conv.bias is not None else torch.zeros_like(std)
    conv.bias.copy_(beta + (b - bn.running_mean) * k)


def fold_bn(model):
    """Return a copy of `model` with every convX/bnX pair folded; bnX becomes Identity."""
    model = copy.deepcopy(model).eval()
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if not isinstance(child, nn.BatchNorm2d):
                continue
            conv_name = "conv" + name[len("bn"):] if name.startswith("bn") else None
            conv = getattr(parent, conv_name, None) if conv_name else None
            if conv is None:
                raise RuntimeError(f"BatchNorm '{name}' has no matching '{conv_name}'")
            _fold_pair(conv, child)
            setattr(parent, name, nn.Identity())
    return model
