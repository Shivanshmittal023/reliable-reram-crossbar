"""The integer reference is the golden model (D10, D11)."""
import torch
import torch.nn.functional as F

from quant.qlayers import QConv2d
from tests.conftest import random_images


def test_int_matches_fakequant(quant_model, int_model):
    x = random_images(64, seed=2)
    with torch.no_grad():
        fq = quant_model.eval()(x)
        it = int_model(x)
    # Same function up to float rounding; predictions should all agree.
    assert (fq - it).abs().max().item() < 1e-2 * fq.abs().max().item()
    assert (fq.argmax(1) == it.argmax(1)).float().mean().item() >= 0.98


def test_int_reference_is_deterministic(int_model):
    x = random_images(32, seed=3)
    with torch.no_grad():
        assert torch.equal(int_model(x), int_model(x))


def test_exact_mac_equals_int64_im2col(int_model):
    """The float64 conv must equal a plain int64 im2col matmul (D11, D13)."""
    captured = {}

    def capture(layer, x_int):
        acc = layer.exact_mac(x_int)
        captured.setdefault(layer, (x_int, acc))
        return acc

    convs = [m for m in int_model.modules() if isinstance(m, QConv2d)]
    for m in convs:
        m.mac_backend = capture
    with torch.no_grad():
        int_model(random_images(4, seed=4))
    for m in convs:
        m.mac_backend = None

    for layer, (x_int, acc) in captured.items():
        assert torch.equal(x_int, torch.round(x_int)) and x_int.min() >= 0 and x_int.max() <= 255
        cols = F.unfold(x_int, layer.kernel_size, padding=layer.padding, stride=layer.stride).long()
        ref = layer.crossbar_matrix().long().t() @ cols          # [out, L] per image
        assert torch.equal(acc.flatten(2).long(), ref)
    assert len(captured) == 19


def test_mac_backend_hook_is_used(int_model):
    """Step 2 plugs the crossbar simulator in here; an identical backend
    must give bit-identical logits."""
    x = random_images(8, seed=5)
    with torch.no_grad():
        ref = int_model(x)
        for _, layer, _ in int_model.crossbar_layers():
            layer.mac_backend = lambda l, xi: l.exact_mac(xi)
        out = int_model(x)
    assert torch.equal(ref, out)
