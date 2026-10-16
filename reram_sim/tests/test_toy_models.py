"""Whole toy networks: nn.Linear layers swapped for ReRAMLinear, compared with the PyTorch originals."""
import copy
import math

import pytest
import torch
import torch.nn as nn

from reram_sim import CrossbarConfig, ReRAMLinear, program_model, replace_modules

SLC = CrossbarConfig(bits_per_cell=1)
MLC = CrossbarConfig(bits_per_cell=2)


def with_sigma(sigma):
    return {"SLC": CrossbarConfig(bits_per_cell=1, sigma=sigma), "MLC": CrossbarConfig(bits_per_cell=2, sigma=sigma)}


def top_sigma(frac):
    """SLC mask: top `frac` of components by singular value."""
    def fn(S):
        mask = torch.zeros_like(S, dtype=torch.bool)
        mask[torch.topk(S, math.ceil(frac * S.numel())).indices] = True
        return mask
    return fn


def to_reram(model, mask_fn):
    reram = {n: ReRAMLinear.from_linear(m, mask_fn) for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    replace_modules(model, reram)
    return model


@pytest.fixture(scope="module")
def toy_task():
    """Trained 3-layer MLP on a 10-class Gaussian-blob task."""
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(10, 32, generator=g) * 2
    y = torch.randint(0, 10, (4000,), generator=g)
    x = centers[y] + torch.randn(4000, 32, generator=g)
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10))
    opt = torch.optim.Adam(mlp.parameters(), 1e-2)
    for _ in range(300):
        opt.zero_grad()
        nn.functional.cross_entropy(mlp(x[:3000]), y[:3000]).backward()
        opt.step()
    mlp.eval()
    return mlp, x[3000:], y[3000:]


def accuracy(model, x, y):
    with torch.no_grad():
        return (model(x).argmax(-1) == y).float().mean().item()


def clone_reram(mlp, mask_fn):
    return to_reram(copy.deepcopy(mlp), mask_fn)


def test_mlp_ideal_equals_pytorch(toy_task):
    mlp, x, y = toy_task
    reram = clone_reram(mlp, top_sigma(0.2))
    ideal = CrossbarConfig(w_bits=24, in_bits=None)
    program_model(reram, {"SLC": ideal, "MLC": ideal}, seed=0)
    with torch.no_grad():
        torch.testing.assert_close(reram(x), mlp(x), rtol=1e-4, atol=1e-4)


def test_mlp_8bit_noise_free_keeps_accuracy(toy_task):
    mlp, x, y = toy_task
    base = accuracy(mlp, x, y)
    assert base > 0.9
    reram = clone_reram(mlp, top_sigma(0.2))
    program_model(reram, {"SLC": SLC, "MLC": MLC}, seed=0)
    assert accuracy(reram, x, y) >= base - 0.01


def test_mlp_noise_degrades_and_slc_protects(toy_task):
    """Accuracy falls as sigma grows; all-SLC beats all-MLC; important-in-SLC sits between."""
    mlp, x, y = toy_task
    seeds = range(5)

    def mean_acc(frac, sigma):
        model = clone_reram(mlp, top_sigma(frac))
        accs = []
        for s in seeds:
            program_model(model, with_sigma(sigma), seed=s)
            accs.append(accuracy(model, x, y))
        return sum(accs) / len(accs)

    sweep = [mean_acc(0.0, s) for s in (0.0, 0.05, 0.15)]
    assert sweep[0] > sweep[1] > sweep[2]
    acc = {frac: mean_acc(frac, 0.15) for frac in (0.0, 0.2, 1.0)}
    assert acc[1.0] > acc[0.2] > acc[0.0]


def test_transformer_encoder_ffn(toy_task):
    """nn.TransformerEncoderLayer with its FFN Linears on ReRAM: ideal cells give the same output."""
    torch.manual_seed(0)
    # train mode with dropout 0: deterministic, and keeps PyTorch off its fused fast path,
    # which reads linear1.weight directly
    enc = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, dropout=0.0, batch_first=True)
    x = torch.randn(4, 10, 64)
    with torch.no_grad():
        ref = enc(x)
    replace_modules(enc, {n: ReRAMLinear.from_linear(getattr(enc, n), top_sigma(0.2)) for n in ("linear1", "linear2")})
    ideal = CrossbarConfig(w_bits=24, in_bits=None)
    program_model(enc, {"SLC": ideal, "MLC": ideal}, seed=0)
    with torch.no_grad():
        torch.testing.assert_close(enc(x), ref, rtol=1e-4, atol=1e-4)
        program_model(enc, with_sigma(0.05), seed=0)
        assert ((enc(x) - ref).norm() / ref.norm()).item() > 1e-3
