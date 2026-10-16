import torch.nn as nn

from .resnet_cifar import resnet20

ARCHS = {"resnet20": resnet20}


def build_model(arch, folded=False):
    """Build `arch`. With folded=True every BatchNorm is replaced by Identity,
    giving the structure of a BN-folded checkpoint (D6)."""
    if arch not in ARCHS:
        raise ValueError(f"unknown arch '{arch}', choose from {sorted(ARCHS)}")
    model = ARCHS[arch]()
    if folded:
        for parent in list(model.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.BatchNorm2d):
                    setattr(parent, name, nn.Identity())
    return model
