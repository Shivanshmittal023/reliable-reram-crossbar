import torch

from quant.quantizers import A_QMAX, fake_quant_act, fake_quant_weight, quantize_weight


def test_weight_range_symmetric_per_channel():
    w = torch.randn(16, 8, 3, 3) * torch.linspace(0.01, 2.0, 16).view(-1, 1, 1, 1)
    w_int, s = quantize_weight(w)
    assert w_int.dtype == torch.int8
    assert w_int.min() >= -127 and w_int.max() <= 127          # -128 never used (D3)
    assert s.shape == (16,)
    # max-abs scaling: every channel reaches +-127
    assert torch.all(w_int.abs().flatten(1).amax(1) == 127)


def test_fake_quant_weight_matches_integer_weight():
    w = torch.randn(10, 64)
    w_int, s = quantize_weight(w)
    assert torch.allclose(fake_quant_weight(w), w_int.float() * s.view(-1, 1))


def test_ste_gradient_passes_through():
    w = torch.randn(4, 4, requires_grad=True)
    fake_quant_weight(w).sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w))


def test_activation_unsigned_and_clamped():
    s = torch.tensor(0.01)
    x = torch.tensor([-1.0, 0.0, 0.004, 0.006, 1.0, 10.0])
    q = fake_quant_act(x, s) / s
    assert torch.equal(torch.round(q), q)
    assert q.min() == 0 and q.max() == A_QMAX


def test_pixel_input_is_exact():
    # D2: ToTensor pixels k/255 quantise back to exactly k.
    k = torch.arange(256).float()
    s = torch.tensor(1.0 / 255.0)
    assert torch.equal(torch.round(fake_quant_act(k / 255.0, s) / s), k)
