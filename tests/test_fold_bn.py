import torch

from models import build_model
from quant.fold_bn import fold_bn
from tests.conftest import random_images


def test_folded_matches_unfolded(fp32_model):
    folded = fold_bn(fp32_model)
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in folded.modules())
    x = random_images(16, seed=1)
    with torch.no_grad():
        diff = (fp32_model(x) - folded(x)).abs().max().item()
    assert diff < 1e-4


def test_folded_state_dict_loads_into_folded_structure(fp32_model):
    folded = fold_bn(fp32_model)
    fresh = build_model("resnet20", folded=True)
    fresh.load_state_dict(folded.state_dict())   # strict: same keys and shapes
