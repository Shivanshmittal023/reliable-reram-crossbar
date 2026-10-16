"""Shared fixtures. Tests run on CPU with random weights and random images,
so they need no dataset download and finish in seconds."""
import pytest
import torch

from models import build_model
from quant.fold_bn import fold_bn
from quant.prepare import calibrate, freeze_int
from utils.common import setup


def random_images(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, 3, 32, 32), generator=g).float() / 255.0


def randomise_bn(model, seed=0):
    """Give BN layers non-trivial statistics so folding is actually tested."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                c = m.num_features
                m.running_mean.copy_(torch.randn(c, generator=g) * 0.1)
                m.running_var.copy_(torch.rand(c, generator=g) + 0.5)
                m.weight.copy_(torch.rand(c, generator=g) + 0.5)
                m.bias.copy_(torch.randn(c, generator=g) * 0.1)
    return model


@pytest.fixture
def fp32_model():
    setup(0)
    return randomise_bn(build_model("resnet20")).eval()


@pytest.fixture
def quant_model(fp32_model):
    """Folded, calibrated, fake-quant ResNet-20."""
    model = fold_bn(fp32_model)
    calib = [(random_images(32, seed=s), None) for s in range(4)]
    return calibrate(model, calib, n_batches=4, device="cpu")


@pytest.fixture
def int_model(quant_model):
    import copy
    return freeze_int(copy.deepcopy(quant_model)).eval()
