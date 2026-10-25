!pip install -q hf_xet
# Do NOT `pip install -U datasets` here: it swaps pyarrow under the running kernel and
# breaks with "pyarrow.lib.Array size changed". Kaggle's preinstalled datasets is enough.
#
# Fault-aware mapping and unified rank-to-slot mapping (C1) for the SVD-tiered transformers
# (single Kaggle notebook cell).
#
# In an SVD layer W = U (sigma V^T) the k components can be stored in any order: component r occupies
# one row of the first crossbar (sigma_r v_r) and one input row of the second (u_r), and permuting the
# components together leaves the layer output unchanged. Once a chip has been tested and its stuck
# cells are known, we can therefore choose which component goes in which physical slot, at no hardware
# or runtime cost. The cost of putting component r in slot s is the expected rise in loss,
#     C[r, s] = sum over the entries e of r:  F_e * E[(weight error of e in slot s)^2]
# with F_e the Fisher information of the stored entry. The weight error has two parts: the exact error
# from the stuck cells of slot s (known after testing) and the programming noise of the slot's tier
# (SLC or MLC). The assignment is solved optimally with the Hungarian algorithm.
#
# Placements (20% of components in SLC)
#   all_slc            every component in SLC, stored in index order (no mapping)
#   all_slc_remap      every component in SLC, order chosen to avoid stuck cells
#   magnitude          largest singular values in SLC, index order (the report's recommended placement)
#   magnitude_remap    same SLC set, order within each tier chosen to avoid stuck cells   (fault-aware mapping)
#   c1                 tier AND slot of every component chosen in one assignment          (unified, C1)
# For c1 the noise part uses the chip's programming noise, and at least 2% (the low end of reported
# values) so that the tier choice stays sensible on a chip with faults but no noise.
#
# Experiments: stuck cells at several rates (no noise), stuck cells with 5% programming noise, and
# clustered / bit-line fault patterns. Every placement sees the same chips (same fault maps and noise).
#
# Setup: GPU T4, Internet on, add the svd-tiering notebook as input, paste into ONE cell.
# Run EXPERIMENT = "smoke" first (a few minutes), then "all" with Save & Run All.
# Output: /kaggle/working/mapping_results/<experiment>/<model>/{sweep.csv, sweep_summary.csv}
import copy
import csv
import json
import math
import shutil
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, pstdev

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

EXPERIMENT = "smoke"        # "smoke" | "all" | "fault" | "fault_noise" | "pattern"
TIME_BUDGET_H = 11.0

MODELS = {
    "distilbert-sst2": "distilbert-base-uncased-finetuned-sst-2-english",
    "bert-base-sst2": "textattack/bert-base-uncased-SST-2",
}
CONFIG = {"slc_bits": 1, "mlc_bits": 2, "batch_size": 128, "eval_n": None, "fisher_n": 2048,
          "slc_frac": 0.2, "design_sigma": 0.02}
OUT = Path("/kaggle/working/mapping_results") if Path("/kaggle/working").exists() else Path("mapping_results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_START = time.time()

def log(msg):
    print(f"[{time.time() - T_START:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- crossbar
@dataclass(frozen=True)
class CrossbarConfig:
    bits_per_cell: int = 1
    sigma: float = 0.0                 # programming noise / G_max
    level_sigma: tuple | None = None   # per-level multiplier on sigma, len = 2**bits_per_cell
    fault_rate: float = 0.0            # expected share of stuck cells
    sa1_frac: float = 0.5              # share of stuck cells stuck at G_max (rest at G_min)
    fault_mode: str = "uniform"        # "uniform" | "cluster" (k x k blocks) | "line" (whole bit line)
    cluster: int = 3
    read_sigma: float = 0.0            # read noise / G_max, fresh at every read
    ir_ratio: float = 0.0              # R_wire per cell segment / R_on
    on_off: float = 100.0
    w_bits: int = 8
    in_bits: int = 8
    input_mode: str = "dac"            # "dac" (whole input at once) | "bit_serial"
    adc_bits: int | None = None        # None = ideal readout
    tile_rows: int = 128
    tile_cols: int = 128

    @property
    def levels(self):
        return 2 ** self.bits_per_cell

    @property
    def n_slices(self):
        return math.ceil((self.w_bits - 1) / self.bits_per_cell)


def quantize_symmetric(x, bits, dim=None):
    qmax = 2 ** (bits - 1) - 1
    absmax = x.abs().amax() if dim is None else x.abs().amax(dim=dim, keepdim=True)
    scale = absmax.clamp(min=1e-12) / qmax
    return torch.round(x / scale).clamp(-qmax, qmax), scale


def fault_masks(shape, cfg, gen, device):
    """Boolean masks (stuck at G_max, stuck at G_min) for one [out, in] cell array."""
    rand = lambda: torch.rand(shape, generator=gen, device=device)
    if cfg.fault_mode == "uniform":
        stuck, sa1 = rand() < cfg.fault_rate, rand() < cfg.sa1_frac
        return stuck & sa1, stuck & ~sa1
    if cfg.fault_mode == "cluster":
        k = cfg.cluster
        seed_rate = cfg.fault_rate / (k * k)

        def grow(seeds):
            return F.max_pool2d(seeds.float()[None, None], k, stride=1, padding=k // 2)[0, 0].bool()

        m1 = grow(rand() < seed_rate * cfg.sa1_frac)
        m0 = grow(rand() < seed_rate * (1 - cfg.sa1_frac)) & ~m1
        return m1, m0
    if cfg.fault_mode == "line":
        # one bit line = one output column inside one row tile; a broken line fails as a whole
        out, n_in = shape
        n_tiles = math.ceil(n_in / cfg.tile_rows)
        line = torch.rand((out, n_tiles), generator=gen, device=device) < cfg.fault_rate
        sa1 = torch.rand((out, n_tiles), generator=gen, device=device) < cfg.sa1_frac
        expand = lambda m: m.repeat_interleave(cfg.tile_rows, dim=1)[:, :n_in]
        return expand(line & sa1), expand(line & ~sa1)
    raise ValueError(cfg.fault_mode)




class Crossbar(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.detach().clone().float())
        for name in ("scale", "w_hat", "digits"):
            self.register_buffer(name, None, persistent=False)
        self.cfg = None
        self.fs = None             # fixed ADC full scale per slice (set by calibration)
        self.calibrating = False
        self.fs_obs = None

    def segments(self, device):
        """Wire segments between each cell and its driver (row start) and sense amp (column end)."""
        o = torch.arange(self.out_features, device=device) % self.cfg.tile_cols
        i = torch.arange(self.in_features, device=device) % self.cfg.tile_rows
        return (o[:, None] + 1) + (self.cfg.tile_rows - 1 - i)[None, :]

    @torch.no_grad()
    def program(self, cfg, gen=None, faults=None):
        """faults: optional known stuck-cell map, a list over slices of {"pos": (m1, m0), "neg": (m1, m0)};
        when given it replaces random fault drawing (used by the fault-aware mappings)."""
        self.cfg = cfg
        dev = self.weight.device
        q, self.scale = quantize_symmetric(self.weight, cfg.w_bits, dim=1)
        mag = {"pos": q.clamp(min=0), "neg": (-q).clamp(min=0)}
        L = cfg.levels
        g_min = 1.0 / cfg.on_off
        delta = (1.0 - g_min) / (L - 1)
        nseg = self.segments(dev) if cfg.ir_ratio > 0 else None
        lvl = None if cfg.level_sigma is None else torch.tensor(cfg.level_sigma, device=dev, dtype=torch.float32)
        digits = []
        for s in range(cfg.n_slices):
            g = {}
            for k, m in mag.items():
                d = torch.div(m, L ** s, rounding_mode="floor") % L
                c = g_min + d * delta
                if cfg.sigma > 0:
                    sig = cfg.sigma if lvl is None else cfg.sigma * lvl[d.long()]
                    c = (c + sig * torch.randn(c.shape, generator=gen, device=dev)).clamp(min=0)
                if faults is not None:
                    m1, m0 = faults[s][k]
                    c = torch.where(m1, torch.ones_like(c), torch.where(m0, torch.full_like(c, g_min), c))
                elif cfg.fault_rate > 0:
                    m1, m0 = fault_masks(c.shape, cfg, gen, dev)
                    c = torch.where(m1, torch.ones_like(c), torch.where(m0, torch.full_like(c, g_min), c))
                if nseg is not None:
                    c = c / (1 + cfg.ir_ratio * c * nseg)
                g[k] = c
            digits.append((g["pos"] - g["neg"]) / delta)
        self.delta = delta
        self.w_hat = sum(L ** s * d for s, d in enumerate(digits)) * self.scale
        self.lsum2 = sum(L ** (2 * s) for s in range(cfg.n_slices))
        self.digits = torch.stack(digits) if self.needs_slow() or self.calibrating else None

    def needs_slow(self):
        return self.cfg.adc_bits is not None or self.cfg.input_mode == "bit_serial"

    def read_noise_k(self):
        return math.sqrt(2) * self.cfg.read_sigma / self.delta     # two cells per weight

    def _fast(self, xi):
        y = F.linear(xi, self.w_hat)
        if self.cfg.read_sigma > 0:
            std = self.read_noise_k() * math.sqrt(self.lsum2) * xi.norm(dim=-1, keepdim=True)
            y = y + torch.randn_like(y) * std * self.scale.squeeze(1)
        return y

    def _planes(self, xt):
        if self.cfg.input_mode != "bit_serial":
            return [(1.0, xt)]
        # sign-magnitude: bit k of |x| applied with the sign of x (+1 / 0 / -1 read voltage).
        # Two's complement would switch on every bit of a small negative input, so read noise
        # and ADC range would be set by the sign bit rather than by the input's size.
        mag, sign = xt.abs().long(), torch.sign(xt)
        return [(2.0 ** k, ((mag >> k) & 1).float() * sign) for k in range(self.cfg.in_bits - 1)]

    def _slow(self, xi):
        cfg = self.cfg
        R = cfg.tile_rows
        n_tiles = math.ceil(self.in_features / R)
        pad = n_tiles * R - self.in_features
        xt = F.pad(xi, (0, pad)).unflatten(-1, (n_tiles, R))
        planes = self._planes(xt)
        qmax = None if cfg.adc_bits is None else 2 ** (cfg.adc_bits - 1) - 1
        y = 0
        for s, d in enumerate(self.digits):
            dt = F.pad(d, (0, pad)).unflatten(-1, (n_tiles, R))
            for wk, p in planes:
                part = torch.einsum("...tr,otr->...ot", p, dt)
                if cfg.read_sigma > 0:
                    std = self.read_noise_k() * (p * p).sum(-1).sqrt().unsqueeze(-2)
                    part = part + torch.randn_like(part) * std
                if self.calibrating:
                    self.fs_obs[s] = max(self.fs_obs[s], part.abs().amax().item())
                elif qmax is not None:
                    step = max(self.fs[s], 1e-12) / qmax
                    part = torch.round(part / step).clamp(-qmax, qmax) * step
                y = y + wk * cfg.levels ** s * part.sum(-1)
        return y * self.scale.squeeze(1)

    def forward(self, x):
        xi, x_scale = quantize_symmetric(x, self.cfg.in_bits)
        y = self._slow(xi) if (self.needs_slow() or self.calibrating) else self._fast(xi)
        return y * x_scale


class ReRAMLinear(nn.Module):
    """y = U_slc (SVh_slc x) + U_mlc (SVh_mlc x) + bias, each factor its own crossbar."""

    def __init__(self, U, SVh, slc_mask, bias):
        super().__init__()
        self.tiers = nn.ModuleDict()
        for tier, mask in (("SLC", slc_mask), ("MLC", ~slc_mask)):
            idx = mask.nonzero().flatten()
            if idx.numel():
                self.tiers[tier] = nn.Sequential(Crossbar(SVh[idx]), Crossbar(U[:, idx]))
        self.register_buffer("bias", None if bias is None else bias.detach().clone().float())

    @classmethod
    def from_orders(cls, U, SVh, orders, bias):
        """orders: {tier: component indices in physical slot order}. Slot j of a tier is row j of its
        first crossbar (sigma*v) and row j of the inputs of its second crossbar (u)."""
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.tiers = nn.ModuleDict()
        for tier, idx in orders.items():
            if idx.numel():
                self.tiers[tier] = nn.Sequential(Crossbar(SVh[idx]), Crossbar(U[:, idx]))
        self.register_buffer("bias", None if bias is None else bias.detach().clone().float())
        return self

    def program(self, cfgs, gen, faults=None):
        """faults: optional {tier: (faults of crossbar 1, faults of crossbar 2)}."""
        for tier, pair in self.tiers.items():
            for i, xbar in enumerate(pair):
                xbar.program(cfgs[tier], gen, None if faults is None else faults[tier][i])

    def forward(self, x):
        y = sum(pair(x) for pair in self.tiers.values())
        return y if self.bias is None else y + self.bias


def crossbars(model):
    return [m for m in model.modules() if isinstance(m, Crossbar)]


def program_model(model, cfgs, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    torch.manual_seed(seed)                       # read noise uses the global generator
    for m in model.modules():
        if isinstance(m, ReRAMLinear):
            m.program(cfgs, gen)



# ---------------------------------------------------------------- self-tests
def selftest():
    """Checks the new simulator pieces on random matrices. Raises if any check fails."""
    log("[self-test] simulator checks on random 300x260 matrices")
    torch.manual_seed(0)
    gen = lambda: torch.Generator(device=DEVICE).manual_seed(1)
    W = torch.randn(300, 260, device=DEVICE)
    x = torch.randn(64, 260, device=DEVICE)
    results = []

    def check(name, ok, detail):
        results.append(ok)
        log(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    q, sc = quantize_symmetric(W, 8, dim=1)
    xi, xs = quantize_symmetric(x, 8)
    ref = F.linear(xi, q * sc) * xs
    for bits in (1, 2):
        cb = Crossbar(W).to(DEVICE)
        cb.program(CrossbarConfig(bits_per_cell=bits), gen())
        # g_min offset cancels in the differential pair, so the noise-free crossbar is exact
        check(f"ideal crossbar = INT8 matmul ({bits} b/cell)", rel(cb(x), ref) < 1e-4, f"rel err {rel(cb(x), ref):.1e}")
        for mode in ("dac", "bit_serial"):
            cfg = CrossbarConfig(bits_per_cell=bits, input_mode=mode)
            cb.calibrating, cb.fs_obs = True, [0.0] * cfg.n_slices     # keeps per-slice digits, no ADC
            cb.program(cfg, gen())
            y = cb._slow(xi) * xs
            cb.calibrating = False
            check(f"{mode} slice-by-slice readout = ideal ({bits} b/cell)", rel(y, ref) < 1e-4, f"rel err {rel(y, ref):.1e}")

    # bit-serial decomposition is exact
    xi = quantize_symmetric(x, 8)[0]
    cb = Crossbar(W).to(DEVICE)
    cb.program(CrossbarConfig(input_mode="bit_serial"), gen())
    rebuilt = sum(wk * p for wk, p in cb._planes(xi))
    check("sign-magnitude bit planes rebuild the input", torch.equal(rebuilt, xi), "exact")

    # fixed ADC: calibrate, then a 16-bit ADC is near-lossless and a 4-bit ADC is not
    for mode in ("dac", "bit_serial"):
        cfg = CrossbarConfig(bits_per_cell=2, input_mode=mode, adc_bits=16)
        cb = Crossbar(W).to(DEVICE)
        cb.calibrating, cb.fs_obs = True, [0.0] * cfg.n_slices
        cb.program(cfg, gen())
        cb(x)
        cb.calibrating, cb.fs = False, list(cb.fs_obs)
        e16 = rel(cb(x), ref)
        cb.program(replace(cfg, adc_bits=4), gen())
        e4 = rel(cb(x), ref)
        check(f"fixed ADC ({mode}): 16 bit ~ lossless, 4 bit lossy", e16 < 1e-3 < e4, f"16b {e16:.1e}, 4b {e4:.1e}")

    # read noise: measured spread matches the formula, in both readout paths
    for mode in ("dac", "bit_serial"):
        cfg = CrossbarConfig(bits_per_cell=2, input_mode=mode, read_sigma=0.02)
        cb = Crossbar(W).to(DEVICE)
        cb.program(cfg, gen())
        meas = (torch.stack([cb(x) for _ in range(20)]) - ref).std().item()
        if mode == "dac":
            pred_i = cb.read_noise_k() * math.sqrt(cb.lsum2) * xi.norm(dim=-1, keepdim=True)
        else:
            planes = cb._planes(xi)
            pred_i = cb.read_noise_k() * torch.sqrt(sum(wk ** 2 * (p * p).sum(-1, keepdim=True) for wk, p in planes)
                                                    * cb.lsum2)
        pred = (pred_i * cb.scale.squeeze(1) * xs).pow(2).mean().sqrt().item()
        check(f"read noise spread matches formula ({mode})", abs(meas / pred - 1) < 0.15,
              f"measured {meas:.4f}, predicted {pred:.4f}")

    # fault modes give the requested share of stuck cells
    for mode in ("uniform", "cluster", "line"):
        cfg = CrossbarConfig(fault_rate=0.02, fault_mode=mode)
        m1, m0 = fault_masks((1024, 1024), cfg, gen(), DEVICE)
        frac = (m1 | m0).float().mean().item()
        check(f"{mode} faults hit ~2% of cells", 0.014 < frac < 0.026, f"{100 * frac:.2f}%")

    # IR drop: zero resistance changes nothing; positive resistance shrinks the weights
    cb = Crossbar(W).to(DEVICE)
    cb.program(CrossbarConfig(ir_ratio=0.0), gen())
    w0 = cb.w_hat.clone()
    cb.program(CrossbarConfig(ir_ratio=1e-3), gen())
    shrink = (cb.w_hat.abs().sum() / w0.abs().sum()).item()
    check("IR drop shrinks stored weights", 0.5 < shrink < 1.0, f"|W| ratio {shrink:.3f}")

    # level-dependent noise: uniform multipliers equal plain sigma
    cb = Crossbar(W).to(DEVICE)
    cb.program(CrossbarConfig(bits_per_cell=2, sigma=0.05), gen())
    a = cb.w_hat.clone()
    cb.program(CrossbarConfig(bits_per_cell=2, sigma=0.05, level_sigma=(1, 1, 1, 1)), gen())
    check("level noise with all-1 multipliers = plain noise", torch.allclose(a, cb.w_hat), "same chip")

    if not all(results):
        raise RuntimeError("simulator self-test failed; see FAIL lines above")
    log(f"[self-test] all {len(results)} checks passed")


# ---------------------------------------------------------------- model loading
class SVDLinear(nn.Module):
    """Shell matching svd_tiering's SVDLinear so svd_model.pt loads into it."""

    def __init__(self, m, k, n, has_bias):
        super().__init__()
        self.rank = k
        self.U = nn.Parameter(torch.zeros(m, k))
        self.S = nn.Parameter(torch.zeros(k))
        self.Vh = nn.Parameter(torch.zeros(k, n))
        self.bias = nn.Parameter(torch.zeros(m)) if has_bias else None

    def forward(self, x):
        return F.linear(F.linear(x, self.Vh) * self.S, self.U, self.bias)


def replace_modules(model, build):
    for name, new in build.items():
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, new)


def find_svd_outputs(key):
    for p in Path("/kaggle/input").rglob("tier_map.json") if Path("/kaggle/input").exists() else []:
        if p.parent.name == key:
            return p.parent
    for z in Path("/kaggle/input").rglob("outputs.zip") if Path("/kaggle/input").exists() else []:
        dest = Path("/kaggle/working/svd_outputs")
        if not (dest / key / "tier_map.json").exists():
            log(f"  unzipping {z} -> {dest}")
            zipfile.ZipFile(z).extractall(dest)
        if (dest / key / "tier_map.json").exists():
            return dest / key
    raise FileNotFoundError(f"no {key}/tier_map.json under /kaggle/input; attach the svd-tiering notebook")


def load_svd_model(key, src):
    tok = AutoTokenizer.from_pretrained(MODELS[key])
    model = AutoModelForSequenceClassification.from_pretrained(MODELS[key])
    tier_map = json.loads((src / "tier_map.json").read_text())
    shells = {name: SVDLinear(e["out_features"], e["rank"], e["in_features"],
                              model.get_submodule(name).bias is not None)
              for name, e in tier_map.items()}
    replace_modules(model, shells)
    model.load_state_dict(torch.load(src / "svd_model.pt", map_location="cpu"))
    return model.eval(), tok, shells



def sst2_batches(tok, split, n, seed):
    ds = load_dataset("nyu-mll/glue", "sst2", split=split)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    ds = ds.map(lambda b: tok(b["sentence"], truncation=True, max_length=128),
                batched=True, remove_columns=["sentence", "idx"]).rename_column("label", "labels")
    loader = DataLoader(ds, batch_size=CONFIG["batch_size"], collate_fn=DataCollatorWithPadding(tok))
    return [{k: v.to(DEVICE) for k, v in b.items()} for b in loader]


@torch.no_grad()
def evaluate(model, batches):
    correct = total = 0
    for b in batches:
        logits = model(**b).logits
        correct += (logits.argmax(-1) == b["labels"]).sum().item()
        total += b["labels"].numel()
    return correct / total


# ---------------------------------------------------------------- ranking scores
def calib_sentences(n):
    # same calibration sentences as the SVD pipeline: SST-2 train, shuffle seed 1
    return load_dataset("nyu-mll/glue", "sst2", split="train").shuffle(seed=1).select(range(n))




# ---------------------------------------------------------------- Fisher information of every stored entry
def entry_fisher(model, layers, tok):
    """Empirical Fisher (sum of squared per-sentence gradients) of every stored entry, per layer:
    F1 [k, n] for the first crossbar (entries sigma_r * v_rj) and F2 [m, k] for the second (u_ir)."""
    n = CONFIG["fisher_n"]
    ds = calib_sentences(n)
    model.to(DEVICE).eval()
    params = [p for l in layers.values() for p in (l.U, l.Vh)]
    req = {p: p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    acc = [torch.zeros_like(p) for p in params]
    for i, ex in enumerate(ds):
        enc = tok(ex["sentence"], truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        loss = model(**enc, labels=torch.tensor([ex["label"]], device=DEVICE)).loss
        for a, g in zip(acc, torch.autograd.grad(loss, params)):
            a += g.detach() ** 2
        if (i + 1) % 256 == 0 or i + 1 == n:
            log(f"    fisher sentence {i + 1}/{n}")
    for p, r in req.items():
        p.requires_grad_(r)
    model.cpu()
    out = {}
    for j, (name, l) in enumerate(layers.items()):
        S = l.S.detach().float().to(DEVICE)
        # stored value sigma_r * v_rj, so dL/d(stored) = (dL/dVh_rj) / sigma_r
        out[name] = (acc[2 * j + 1] / S[:, None].pow(2).clamp(min=1e-12), acc[2 * j])
    return out


# ---------------------------------------------------------------- mapping cost
def tier_cfg(tier, base):
    return replace(base, bits_per_cell=CONFIG["slc_bits"] if tier == "SLC" else CONFIG["mlc_bits"])


def cell_digits(x, scale, cfg):
    """Digits stored in each cell: [*x.shape, slices, 2] for the positive and negative cell."""
    qmax = 2 ** (cfg.w_bits - 1) - 1
    q = torch.round(x / scale).clamp(-qmax, qmax)
    mag = torch.stack([q.clamp(min=0), (-q).clamp(min=0)], -1)
    L = cfg.levels
    return torch.stack([torch.div(mag, L ** t, rounding_mode="floor") % L for t in range(cfg.n_slices)], -2).to(torch.uint8)


def draw_faults(shape, cfg, gen):
    """Stuck-cell map of one physical crossbar: list over slices of {"pos": (m1, m0), "neg": (m1, m0)}."""
    if cfg.fault_rate <= 0:
        return None
    return [{k: fault_masks(shape, cfg, gen, DEVICE) for k in ("pos", "neg")} for _ in range(cfg.n_slices)]


def stuck_error(gstar, d, cfg, t):
    g_min = 1.0 / cfg.on_off
    delta = (1.0 - g_min) / (cfg.levels - 1)
    return (gstar - (g_min + d.float() * delta)) / delta * cfg.levels ** t     # in units of the 8-bit step


def fault_cost(cfg, faults1, faults2, d1, sc1, F1, d2, sc2, F2):
    """Fisher-weighted squared weight error from the known stuck cells, for every candidate component
    (rows, in the order of d1 / F1) in every physical slot of one tier (columns).
    d1 [R, n, S, 2], sc1 [R], F1 [R, n]: first crossbar (slot = row). d2 [m, R, S, 2], sc2 [m], F2 [m, R]:
    second crossbar (slot = input row)."""
    R = d1.shape[0]
    n_slots = faults1[0]["pos"][0].shape[0] if faults1 is not None else faults2[0]["pos"][0].shape[1]
    cost1 = torch.zeros(R, n_slots, device=DEVICE)
    cost2 = torch.zeros(R, n_slots, device=DEVICE)
    chunk = max(1, 16_000_000 // R)
    g_min = 1.0 / cfg.on_off
    for t in range(cfg.n_slices):
        for p, key in enumerate(("pos", "neg")):
            if faults1 is not None:
                m1, m0 = faults1[t][key]
                xy = (m1 | m0).nonzero()
                for c in range(0, len(xy), chunk):
                    s_c, j_c = xy[c:c + chunk].T
                    gstar = torch.where(m1[s_c, j_c], 1.0, g_min)
                    err = stuck_error(gstar, d1[:, j_c, t, p], cfg, t)
                    cost1.index_add_(1, s_c, F1[:, j_c] * err ** 2)
            if faults2 is not None:
                m1, m0 = faults2[t][key]
                xy = (m1 | m0).nonzero()
                for c in range(0, len(xy), chunk):
                    i_c, s_c = xy[c:c + chunk].T
                    gstar = torch.where(m1[i_c, s_c], 1.0, g_min)
                    err = stuck_error(gstar[:, None], d2[i_c, :, t, p], cfg, t) * sc2[i_c, None]
                    cost2.index_add_(1, s_c, (F2[i_c] * err ** 2).T)
    return cost1 * sc1[:, None] ** 2 + cost2


def noise_cost(cfg, sigma, sc1, F1, sc2, F2):
    """Expected Fisher-weighted damage from programming noise sigma in this tier, per component."""
    g_min = 1.0 / cfg.on_off
    delta = (1.0 - g_min) / (cfg.levels - 1)
    kappa = 2 * (sigma / delta) ** 2 * sum(cfg.levels ** (2 * t) for t in range(cfg.n_slices))
    return kappa * (sc1 ** 2 * F1.sum(1) + (sc2[:, None] ** 2 * F2).sum(0))


def assign(cost):
    rows, cols = linear_sum_assignment(cost.double().cpu().numpy())
    return torch.as_tensor(rows), torch.as_tensor(cols)


def layer_orders(policy, ctx, faults, cfgs, sigma, stats):
    """Component indices per tier, in physical slot order."""
    U, SVh, F1, F2, mag_rank = ctx["U"], ctx["SVh"], ctx["F1"], ctx["F2"], ctx["mag_rank"]
    k = SVh.shape[0]
    sc1 = SVh.abs().amax(1) / 127
    if policy.startswith("all_slc"):
        groups = {"SLC": torch.arange(k)}
    else:
        slc = torch.zeros(k, dtype=torch.bool)
        slc[mag_rank[:ctx["k_slc"]]] = True
        groups = {"SLC": slc.nonzero().flatten(), "MLC": (~slc).nonzero().flatten()}
    if policy in ("all_slc", "magnitude"):
        return groups

    def tier_cost(tier, comps, scale_comps):
        cfg = cfgs[tier]
        sc2 = U[:, scale_comps].abs().amax(1).clamp(min=1e-12) / 127
        c = fault_cost(cfg, faults[tier][0], faults[tier][1], cell_digits(SVh[comps], sc1[comps, None], cfg),
                       sc1[comps], F1[comps], cell_digits(U[:, comps], sc2[:, None], cfg), sc2, F2[:, comps])
        return c, sc2

    if policy in ("all_slc_remap", "magnitude_remap"):
        orders = {}
        for tier, comps in groups.items():
            if faults[tier][0] is None:
                orders[tier] = comps
                continue
            c, _ = tier_cost(tier, comps, comps)
            rows, cols = assign(c)
            order = torch.empty_like(comps)
            order[cols] = comps[rows]
            orders[tier] = order
            stats["before"] += c.diagonal().sum().item()
            stats["after"] += c[rows, cols].sum().item()
        return orders

    # c1: tier and slot of every component in one assignment
    allc = torch.arange(k)
    s = max(sigma, CONFIG["design_sigma"])
    blocks = []
    for tier in ("SLC", "MLC"):
        c = torch.zeros(k, ctx["n_slots"][tier], device=DEVICE)
        sc2 = U.abs().amax(1).clamp(min=1e-12) / 127
        if faults[tier][0] is not None:
            c, sc2 = tier_cost(tier, allc, allc)
        c = c + noise_cost(cfgs[tier], s, sc1, F1, sc2, F2)[:, None]
        blocks.append(c)
    rows, cols = assign(torch.cat(blocks, 1))
    n_slc = ctx["n_slots"]["SLC"]
    slot = torch.empty(k, dtype=torch.long)
    slot[rows] = cols
    orders = {"SLC": torch.empty(n_slc, dtype=torch.long), "MLC": torch.empty(k - n_slc, dtype=torch.long)}
    in_slc = slot < n_slc
    orders["SLC"][slot[in_slc]] = allc[in_slc]
    orders["MLC"][slot[~in_slc] - n_slc] = allc[~in_slc]
    stats["moved"] += len(set(orders["SLC"].tolist()) ^ set(groups["SLC"].tolist())) // 2
    stats["slc"] += n_slc
    return orders


# ---------------------------------------------------------------- building a chip
def layer_context(layers, fisher, mag, frac, all_slc):
    ctxs = {}
    for name, l in layers.items():
        k = l.rank
        k_slc = k if all_slc else math.ceil(frac * k)
        F1, F2 = fisher[name]
        ctxs[name] = {"U": l.U.detach().float().to(DEVICE), "SVh": (l.S[:, None] * l.Vh).detach().float().to(DEVICE),
                      "bias": l.bias, "F1": F1, "F2": F2, "k_slc": k_slc,
                      "mag_rank": torch.argsort(mag[name], descending=True),
                      "n_slots": {"SLC": k_slc, "MLC": k - k_slc}}
    return ctxs


def build_chip(model, ctxs, policy, cfg_base, seed, stats):
    """Draw this chip's fault maps, choose the mapping, then program every crossbar."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    torch.manual_seed(seed)
    cfgs = {t: tier_cfg(t, cfg_base) for t in ("SLC", "MLC")}
    built = {}
    for name, ctx in ctxs.items():
        m, n = ctx["U"].shape[0], ctx["SVh"].shape[1]
        faults = {}
        for tier, slots in ctx["n_slots"].items():
            if slots:
                faults[tier] = (draw_faults((slots, n), cfgs[tier], gen), draw_faults((m, slots), cfgs[tier], gen))
        orders = layer_orders(policy, ctx, faults, cfgs, cfg_base.sigma, stats)
        lin = ReRAMLinear.from_orders(ctx["U"], ctx["SVh"], {t: o.to(DEVICE) for t, o in orders.items()}, ctx["bias"])
        lin.to(DEVICE).program(cfgs, gen, faults)
        built[name] = lin
    replace_modules(model, built)


# ---------------------------------------------------------------- self-test of the mapping cost
def mapping_selftest():
    """The predicted stuck-cell cost must match the actual Fisher-weighted error of a programmed chip,
    and the Hungarian assignment must not increase it."""
    log("[mapping self-test] predicted vs actual stuck-cell damage on a random 60x48x70 layer")
    torch.manual_seed(0)
    k, n, m = 48, 70, 60
    U, SVh = torch.randn(m, k, device=DEVICE), torch.randn(k, n, device=DEVICE)
    F1, F2 = torch.rand(k, n, device=DEVICE), torch.rand(m, k, device=DEVICE)
    ok = True
    for bits in (1, 2):
        cfg = CrossbarConfig(bits_per_cell=bits, fault_rate=0.01)
        gen = torch.Generator(device=DEVICE).manual_seed(3)
        f1, f2 = draw_faults((k, n), cfg, gen), draw_faults((m, k), cfg, gen)
        sc1, sc2 = SVh.abs().amax(1) / 127, U.abs().amax(1) / 127
        cost = fault_cost(cfg, f1, f2, cell_digits(SVh, sc1[:, None], cfg), sc1, F1,
                          cell_digits(U, sc2[:, None], cfg), sc2, F2)
        x1, x2 = Crossbar(SVh).to(DEVICE), Crossbar(U).to(DEVICE)
        x1.program(replace(cfg, fault_rate=0.0), gen, f1)
        x2.program(replace(cfg, fault_rate=0.0), gen, f2)
        q1, s1 = quantize_symmetric(SVh, 8, dim=1)
        q2, s2 = quantize_symmetric(U, 8, dim=1)
        actual = (F1 * (x1.w_hat - q1 * s1) ** 2).sum() + (F2 * (x2.w_hat - q2 * s2) ** 2).sum()
        pred = cost.diagonal().sum()
        rows, cols = assign(cost)
        better = cost[rows, cols].sum() <= pred + 1e-9
        good = abs(pred / actual - 1).item() < 0.1 and better
        ok &= good
        log(f"  {'PASS' if good else 'FAIL'}  {bits} b/cell: predicted {pred:.4g}, actual {actual:.4g}; "
            f"after assignment {cost[rows, cols].sum():.4g}")
    if not ok:
        raise RuntimeError("mapping self-test failed")


# ---------------------------------------------------------------- experiments
POLICIES = ["all_slc", "all_slc_remap", "magnitude", "magnitude_remap", "c1"]
EXPERIMENTS = {
    "fault": {"param": "fault_rate", "values": [0.002, 0.005, 0.01, 0.02], "fixed": {}, "seeds": 10},
    "fault_noise": {"param": "fault_rate", "values": [0.002, 0.005, 0.01], "fixed": {"sigma": 0.05}, "seeds": 10},
    "pattern": {"param": "fault_mode", "values": ["cluster", "line"], "fixed": {"fault_rate": 0.01}, "seeds": 10},
}
if EXPERIMENT == "smoke":
    CONFIG.update(eval_n=128, fisher_n=64)
    OUT = OUT.with_name(OUT.name + "_smoke")
    for e in EXPERIMENTS.values():
        e.update(values=e["values"][:1], seeds=1)
    RUN, RUN_MODELS = list(EXPERIMENTS), ["distilbert-sst2"]
else:
    RUN = list(EXPERIMENTS) if EXPERIMENT == "all" else [EXPERIMENT]
    RUN_MODELS = list(MODELS)


def label(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def run_model(key):
    log(f"========== {key}")
    src = find_svd_outputs(key)
    digital, tok, layers = load_svd_model(key, src)
    mag = {n: s.float().cpu() for n, s in torch.load(src / "importance.pt", map_location="cpu")["magnitude"].items()}
    batches = sst2_batches(tok, "validation", CONFIG["eval_n"], 0)
    log(f"  {len(layers)} layers, {sum(len(b['labels']) for b in batches)} validation sentences")
    log("  Fisher information of every stored entry")
    fisher = entry_fisher(digital, layers, tok)
    ctx_hybrid = layer_context(layers, fisher, mag, CONFIG["slc_frac"], False)
    ctx_all = layer_context(layers, fisher, mag, CONFIG["slc_frac"], True)
    model = copy.deepcopy(digital).to(DEVICE).eval()
    for name in RUN:
        e = EXPERIMENTS[name]
        out = OUT / name / key
        out.mkdir(parents=True, exist_ok=True)
        n_points = len(POLICIES) * len(e["values"]) * e["seeds"]
        log(f"  ######## {name}: {len(POLICIES)} placements x {len(e['values'])} values of {e['param']} "
            f"x {e['seeds']} chips = {n_points} evaluations, fixed {e['fixed'] or 'nothing else'}")
        rows, done, t0 = [], 0, time.time()
        with open(out / "sweep.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["policy", "slc_frac", "param", "value", "seed", "acc"])
            for v in e["values"]:
                base = replace(CrossbarConfig(), **{**e["fixed"], e["param"]: v})
                for policy in POLICIES:
                    if (time.time() - T_START) / 3600 > TIME_BUDGET_H:
                        log("  !! time budget reached, stopping")
                        return
                    frac = 1.0 if policy.startswith("all_slc") else CONFIG["slc_frac"]
                    accs, stats = [], defaultdict(float)
                    for seed in range(e["seeds"]):
                        build_chip(model, ctx_all if policy.startswith("all_slc") else ctx_hybrid, policy, base, seed, stats)
                        acc = evaluate(model, batches)
                        accs.append(acc)
                        rows.append((policy, frac, label(v), acc))
                        w.writerow([policy, frac, e["param"], label(v), seed, f"{acc:.6f}"])
                        done += 1
                    fh.flush()
                    extra = ""
                    if stats["before"] > 0:
                        extra = f"  stuck-cell damage cut by {100 * (1 - stats['after'] / stats['before']):.0f}%"
                    if stats["slc"] > 0:
                        extra = f"  {100 * stats['moved'] / stats['slc']:.1f}% of SLC slots changed vs magnitude"
                    eta = (time.time() - t0) / done * (n_points - done)
                    log(f"    [{done:4d}/{n_points}] {e['param']}={label(v):8s} {policy:16s} acc {mean(accs):.4f} "
                        f"+- {pstdev(accs):.4f} (min {min(accs):.4f}){extra}  eta {eta / 60:.0f} min")
        groups = defaultdict(list)
        for p, f, v, a in rows:
            groups[(p, f, v)].append(a)
        with open(out / "sweep_summary.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["policy", "slc_frac", "param", "value", "acc_mean", "acc_std", "acc_min", "n"])
            for (p, f, v), a in groups.items():
                w.writerow([p, f, e["param"], v, f"{mean(a):.6f}", f"{pstdev(a):.6f}", f"{min(a):.6f}", len(a)])
        log(f"  summary {name} ({key}): mean accuracy %")
        vals = [label(v) for v in e["values"]]
        log(f"    {'placement':18s} " + " ".join(f"{v:>9s}" for v in vals))
        for p in POLICIES:
            log(f"    {p:18s} " + " ".join(f"{100 * mean(groups[(p, 1.0 if p.startswith('all_slc') else CONFIG['slc_frac'], v)]):9.2f}"
                                          if (p, 1.0 if p.startswith('all_slc') else CONFIG['slc_frac'], v) in groups else "        -" for v in vals))
        shutil.make_archive(str(OUT), "zip", OUT)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- main
log(f"device: {DEVICE} {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else ''}  torch {torch.__version__}")
log(f"experiments: {RUN}, models: {RUN_MODELS}, output: {OUT}")
selftest()
mapping_selftest()
for key in RUN_MODELS:
    run_model(key)
shutil.make_archive(str(OUT), "zip", OUT)
log(f"finished in {(time.time() - T_START) / 3600:.1f} h; download {OUT}.zip from the Output panel")
