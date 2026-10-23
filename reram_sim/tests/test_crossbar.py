"""Crossbar and ReRAMLinear checked against torch.nn.Linear on toy matrices."""
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from reram_sim import Crossbar, CrossbarConfig, ReRAMLinear, program_model, quantize_symmetric

IDEAL = dict(sigma=0.0, fault_rate=0.0, in_bits=None, adc_bits=None)


def make_linear(out_f=48, in_f=300, seed=0):
    torch.manual_seed(seed)
    return nn.Linear(in_f, out_f)


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


def test_ideal_high_precision_matches_linear():
    lin = make_linear()
    x = torch.randn(16, lin.in_features)
    xb = Crossbar(lin.weight)
    xb.program(CrossbarConfig(w_bits=20, **IDEAL))
    assert rel_err(xb(x) + lin.bias, lin(x)) < 1e-5


@pytest.mark.parametrize("bits_per_cell", [1, 2, 3])
def test_bit_slicing_is_exact_for_quantised_weight(bits_per_cell):
    """Noise-free cells reproduce the 8-bit quantised weight exactly, for SLC and MLC."""
    lin = make_linear()
    q, scale = quantize_symmetric(lin.weight.detach(), 8, dim=1)
    xb = Crossbar(lin.weight)
    xb.program(CrossbarConfig(bits_per_cell=bits_per_cell, w_bits=8, **IDEAL))
    torch.testing.assert_close(xb.effective_weight(), q * scale, rtol=0, atol=1e-6)
    x = torch.randn(16, lin.in_features)
    torch.testing.assert_close(xb(x), F.linear(x, q * scale), rtol=1e-5, atol=1e-5)


def test_input_dac_quantisation():
    lin = make_linear()
    x = torch.randn(16, lin.in_features)
    xb = Crossbar(lin.weight)
    xb.program(CrossbarConfig(w_bits=8, in_bits=8, sigma=0.0, adc_bits=None))
    xq, xs = quantize_symmetric(x, 8)
    torch.testing.assert_close(xb(x), F.linear(xq * xs, xb.effective_weight()), rtol=1e-5, atol=1e-5)
    assert rel_err(xb(x), F.linear(x, lin.weight)) < 0.02


@pytest.mark.parametrize("bits_per_cell", [1, 2])
def test_adc_path(bits_per_cell):
    """Tiled, per-slice ADC path: ~exact with a 24-bit ADC, coarser with fewer bits."""
    lin = make_linear(in_f=300)          # 300 rows -> 3 tiles of 128 (last one padded)
    x = torch.randn(32, lin.in_features)
    ref = Crossbar(lin.weight)
    ref.program(CrossbarConfig(bits_per_cell=bits_per_cell, w_bits=8, in_bits=8))
    errs = []
    for adc_bits in (24, 8, 4):
        xb = Crossbar(lin.weight)
        xb.program(CrossbarConfig(bits_per_cell=bits_per_cell, w_bits=8, in_bits=8, adc_bits=adc_bits))
        torch.testing.assert_close(xb.effective_weight(), ref.effective_weight())
        errs.append(rel_err(xb(x), ref(x)))
    assert errs[0] < 1e-5
    assert errs[0] < errs[1] < errs[2]


@pytest.mark.parametrize("bits_per_cell", [1, 2, 3])
def test_noise_matches_analytic_std(bits_per_cell):
    """Weight error std = scale * sigma/delta * sqrt(2 * sum_s L^(2s)) (two cells per digit)."""
    sigma = 0.002                                   # small, so clamping at G=0 never triggers
    W = torch.randn(256, 256)
    cfg = CrossbarConfig(bits_per_cell=bits_per_cell, sigma=sigma, in_bits=None)
    clean, noisy = Crossbar(W), Crossbar(W)
    clean.program(CrossbarConfig(bits_per_cell=bits_per_cell, in_bits=None))
    noisy.program(cfg, torch.Generator().manual_seed(0))
    err_int = (noisy.effective_weight() - clean.effective_weight()) / clean.scale
    L = cfg.levels
    delta = (1 - 1 / cfg.on_off) / (L - 1)
    expected = sigma / delta * math.sqrt(2 * sum(L ** (2 * s) for s in range(cfg.n_slices)))
    assert err_int.std().item() == pytest.approx(expected, rel=0.03)
    assert err_int.mean().abs().item() < 0.05 * expected


def test_mlc_noisier_than_slc():
    lin = make_linear(out_f=128, in_f=256)
    x = torch.randn(64, 256)
    ref = F.linear(x, lin.weight)
    err = {}
    for name, b in (("SLC", 1), ("MLC2", 2), ("MLC3", 3)):
        xb = Crossbar(lin.weight)
        xb.program(CrossbarConfig(bits_per_cell=b, sigma=0.02, in_bits=None), torch.Generator().manual_seed(1))
        err[name] = rel_err(xb(x), ref)
    assert err["SLC"] < err["MLC2"] < err["MLC3"]


def test_seed_determinism():
    W = torch.randn(32, 64)
    cfg = CrossbarConfig(bits_per_cell=2, sigma=0.05, fault_rate=0.05)
    out = []
    for seed in (7, 7, 8):
        xb = Crossbar(W)
        xb.program(cfg, torch.Generator().manual_seed(seed))
        out.append(xb.effective_weight())
    assert torch.equal(out[0], out[1])
    assert not torch.equal(out[0], out[2])


def test_fault_rate_and_stuck_values():
    W = torch.randn(200, 200)
    g = torch.Generator().manual_seed(0)
    cfg = CrossbarConfig(bits_per_cell=2, fault_rate=0.1)
    d = torch.randint(0, 4, (200, 200)).float()
    cells = Crossbar._cells(d, cfg, g_min=0.01, delta=0.33, gen=g)
    clean = 0.01 + d * 0.33
    changed = (cells - clean).abs() > 1e-5
    assert changed.float().mean().item() == pytest.approx(0.1 * (1 - 0.25), abs=0.01)  # stuck at own level looks unchanged
    v = cells[changed]
    assert torch.all(((v - 0.01).abs() < 1e-6) | ((v - 1.0).abs() < 1e-6))

    # every cell stuck at G_max: both cells of each pair equal, so every digit reads 0
    xb = Crossbar(W)
    xb.program(CrossbarConfig(fault_rate=1.0, sa1_frac=1.0), g)
    assert xb.effective_weight().abs().max().item() == 0.0


@pytest.mark.parametrize("frac", [0.0, 0.3, 1.0])
def test_reram_linear_ideal_matches_linear(frac):
    """Full-rank SVD + any SLC/MLC split is lossless when the cells are ideal."""
    lin = make_linear(out_f=40, in_f=96)
    x = torch.randn(8, 96)

    def top_frac(S):
        mask = torch.zeros_like(S, dtype=torch.bool)
        mask[: math.ceil(frac * S.numel())] = True
        return mask

    layer = ReRAMLinear.from_linear(lin, top_frac)
    program_model(nn.Sequential(layer), {"SLC": CrossbarConfig(w_bits=24, **IDEAL),
                                         "MLC": CrossbarConfig(bits_per_cell=2, w_bits=24, **IDEAL)}, seed=0)
    assert rel_err(layer(x), lin(x)) < 1e-5
    assert rel_err(layer.effective_weight(), lin.weight.detach()) < 1e-5
    assert set(layer.tiers) == {"SLC", "MLC"} - ({"SLC"} if frac == 0 else set()) - ({"MLC"} if frac == 1 else set())


def test_slc_and_mlc_agree_without_noise():
    """With 8-bit cells and no noise, the split does not change the output."""
    lin = make_linear(out_f=40, in_f=96)
    x = torch.randn(8, 96)
    outs = []
    for frac in (0.0, 0.5, 1.0):
        layer = ReRAMLinear.from_linear(lin, lambda S: torch.arange(S.numel()) < frac * S.numel())
        program_model(nn.Sequential(layer), {"SLC": CrossbarConfig(bits_per_cell=1, in_bits=8),
                                             "MLC": CrossbarConfig(bits_per_cell=2, in_bits=8)}, seed=0)
        outs.append(layer(x))
    torch.testing.assert_close(outs[0], outs[2], rtol=1e-5, atol=1e-5)
    # A split changes the per-row U scales and the DAC range of h, so only
    # 8-bit quantisation differences remain.
    assert rel_err(outs[1], outs[0]) < 0.02
