!pip install -q hf_xet
# Do NOT `pip install -U datasets` here: it swaps pyarrow under the running kernel and
# breaks with "pyarrow.lib.Array size changed". Kaggle's preinstalled datasets is enough.
#
# ReRAM crossbar experiments for the SVD-tiered transformers (single Kaggle notebook cell).
#
# Loads the fine-tuned SVD models produced by svd_tiering/kaggle_svd_tiering.py, places every SVD
# component in a reliable 1-bit (SLC) or dense 2-bit (MLC) crossbar according to a placement policy,
# simulates the chip cell by cell, and measures SST-2 validation accuracy over many simulated chips.
#
# Placements (names as in the results report)
#   all_slc / all_mlc   every component in SLC / MLC (best and cheapest chip)
#   random              a random share of components in SLC (control)
#   magnitude           components with the largest singular values sigma_r (recommended)
#   gradient            HyFlexPIM-style: sum over batches of 32 of |dL/dsigma_r|, dropout on
#   fisher              expected loss increase if the component is left in MLC: sum over every stored
#                       entry (row sigma_r*v_r, column u_r) of Fisher(entry) x that entry's noise variance
#
# Experiments (one parameter swept at a time; 20% of components in SLC unless the SLC share is swept)
#   slc_share   SLC share 5-50% at programming noise sigma = 3, 5, 7.5%     20 chips
#   noise       programming noise sigma = 0-10%                             5 chips
#   fault       stuck-at cells 0-5%                                         20 chips
#   read        read noise 0-5%                                             5 chips
#   level       middle MLC levels 1.5x / 2x noisier at sigma = 5%           10 chips
#   cluster     1% stuck cells: scattered / 3x3 clusters / whole bit lines  10 chips
#   irdrop      wire resistance R_wire / R_on per segment                   1 chip (deterministic)
#   adc         fixed ADC 3-8 bits with bit-serial inputs                   1 chip (deterministic)
#   adc_noise   fixed ADC at sigma = 5%, DistilBERT only                    3 chips
#
# Cell model: 8-bit signed weights per row, differential pair (G+, G-), bit-sliced into base-2^b digits
# (SLC b=1: 7 cells per sign, MLC b=2: 4 cells). Digit d -> g = g_min + d*delta, G_max = 1, on/off 100.
# Crossbar rows = layer inputs, columns = outputs, 128 x 128 tiles. Programming noise g += sigma*N(0,1);
# stuck cells read G_max or G_min; read noise is fresh at every read; IR drop g / (1 + r*g*segments);
# the ADC range is calibrated once on a noise-free chip and then fixed. Runs in fp32 (fp16 overflows).
# A chip is one random draw (seed); every policy sees the same chips, so policies compare directly.
#
# Setup on Kaggle: GPU T4, Internet on, add the svd-tiering notebook as input, paste into ONE cell.
# Set EXPERIMENT below ("smoke" first, then one experiment name or "all"); Save & Run All for long runs.
# Output: /kaggle/working/results/<experiment>/<model>/{sweep.csv, sweep_summary.csv} and results.zip
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
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

EXPERIMENT = "smoke"        # "smoke" | "all" | one of the experiment names above
TIME_BUDGET_H = 11.0        # Kaggle stops sessions at 12 h; later experiments are skipped after this

MODELS = {
    "distilbert-sst2": "distilbert-base-uncased-finetuned-sst-2-english",
    "bert-base-sst2": "textattack/bert-base-uncased-SST-2",
}
SCORE_NAMES = ["gradient", "fisher"]       # scores computed in this script
CONFIG = {
    "slc_bits": 1, "mlc_bits": 2, "batch_size": 128, "eval_n": None,
    "fisher_n": 2048, "grad_batch": 32, "adc_calib_n": 64,
}
OUT = Path("/kaggle/working/results") if Path("/kaggle/working").exists() else Path("results")
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
    def program(self, cfg, gen=None):
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
                if cfg.fault_rate > 0:
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

    def program(self, cfgs, gen):
        for tier, pair in self.tiers.items():
            for xbar in pair:
                xbar.program(cfgs[tier], gen)

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


def tier_masks(layers, importance, policy, frac):
    """Boolean SLC mask per layer: the top ceil(frac * k) components by the policy's score."""
    if policy == "all_slc":
        return {n: torch.ones(l.rank, dtype=torch.bool) for n, l in layers.items()}
    if policy == "all_mlc":
        return {n: torch.zeros(l.rank, dtype=torch.bool) for n, l in layers.items()}
    if policy == "random":
        g = torch.Generator().manual_seed(0)
        scores = {n: torch.rand(l.rank, generator=g) for n, l in layers.items()}
    else:
        scores = importance[policy]
    masks = {}
    for n, s in scores.items():
        mask = torch.zeros_like(s, dtype=torch.bool)
        mask[torch.topk(s, math.ceil(frac * s.numel())).indices] = True
        masks[n] = mask
    return masks


def to_reram(model, layers, masks):
    built = {}
    for name, svd in layers.items():
        U, SVh = svd.U.detach(), (svd.S[:, None] * svd.Vh).detach()
        built[name] = ReRAMLinear(U, SVh, masks[name], svd.bias)
    replace_modules(model, built)


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


@torch.no_grad()
def calibrate_adc(model, cfgs, calib):
    """Fix each crossbar's ADC full scale per slice: the largest partial sum seen on a
    noise-free chip over the calibration sentences."""
    xbars = crossbars(model)
    for cb in xbars:
        cb.calibrating = True
    clean = {t: replace(c, sigma=0.0, fault_rate=0.0, read_sigma=0.0, adc_bits=None) for t, c in cfgs.items()}
    program_model(model, clean, 0)
    for cb in xbars:
        cb.fs_obs = [0.0] * cb.cfg.n_slices
    for b in calib:
        model(**b)
    for cb in xbars:
        cb.calibrating, cb.fs = False, list(cb.fs_obs)
        cb.digits = None
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()




# ---------------------------------------------------------------- ranking scores
def calib_sentences(n):
    # same calibration sentences as the SVD pipeline: SST-2 train, shuffle seed 1
    return load_dataset("nyu-mll/glue", "sst2", split="train").shuffle(seed=1).select(range(n))


def fisher_scores(model, layers, tok):
    """Fisher score: one pass over single sentences accumulating the empirical Fisher information
    (sum of squared per-sentence gradients) of every stored U and Vh entry, then
        score_r = sum_j F(SVh_rj) * s_r^2 + sum_i F(U_ir) * t_i^2
    where s_r and t_i are the 8-bit scales of the crossbar rows holding the entries: fixed cell noise
    becomes weight noise in proportion to that scale, so Var(noise) ~ scale^2."""
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
    comp, row_share = {}, []
    for j, (name, l) in enumerate(layers.items()):
        FU, FVh = (a.cpu() for a in acc[2 * j: 2 * j + 2])
        S, U, Vh = (l.S.detach().float().cpu(), l.U.detach().float().cpu(), l.Vh.detach().float().cpu())
        # crossbar 1 stores SVh_rj = S_r Vh_rj (dL/dSVh = dL/dVh / S_r), 8-bit scale per row r;
        # crossbar 2 stores U, 8-bit scale per output row i. Cell noise -> weight noise ~ that scale.
        F_svh = FVh / S[:, None].pow(2).clamp(min=1e-12)
        row = (S[:, None] * Vh).abs().amax(1).pow(2) * F_svh.sum(1)
        col = (U.abs().amax(1).pow(2)[:, None] * FU).sum(0)
        comp[name] = row + col
        row_share.append((row.sum() / (row + col).sum().clamp(min=1e-30)).item())
    log(f"    fisher: {100 * mean(row_share):.0f}% of the score from the sigma*v rows, "
        f"{100 - 100 * mean(row_share):.0f}% from the u columns")
    return comp


def batch_gradient_scores(model, layers, tok):
    """HyFlexPIM-style: sum over batches of |dL_batch / dsigma_r|, dropout on (training mode)."""
    ds = calib_sentences(CONFIG["fisher_n"])
    ds = ds.map(lambda b: tok(b["sentence"], truncation=True, max_length=128),
                batched=True, remove_columns=["sentence", "idx"]).rename_column("label", "labels")
    loader = DataLoader(ds, batch_size=CONFIG["grad_batch"], collate_fn=DataCollatorWithPadding(tok))
    torch.manual_seed(0)
    model.to(DEVICE).train()
    S = [l.S for l in layers.values()]
    req = {p: p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for s in S:
        s.requires_grad_(True)
    score = [torch.zeros_like(s) for s in S]
    for i, b in enumerate(loader):
        loss = model(**{k: v.to(DEVICE) for k, v in b.items()}).loss
        for a, g in zip(score, torch.autograd.grad(loss, S)):
            a += g.detach().abs()
    for p, r in req.items():
        p.requires_grad_(r)
    model.eval().cpu()
    log(f"    gradient over {len(loader)} batches")
    return {name: s.cpu() for name, s in zip(layers, score)}


def agreement(scores, frac=0.2):
    mag = scores["magnitude"]
    ranks = lambda x: x.argsort().argsort().float()
    for m in SCORE_NAMES:
        rho, ov = [], []
        for name, ref in mag.items():
            rho.append(torch.corrcoef(torch.stack([ranks(ref), ranks(scores[m][name])]))[0, 1].item())
            k = math.ceil(frac * ref.numel())
            a = set(torch.topk(ref, k).indices.tolist())
            b = set(torch.topk(scores[m][name], k).indices.tolist())
            ov.append(len(a & b) / k)
        log(f"    {m:17s} vs magnitude: rank correlation {mean(rho):.2f}, top-20% overlap {100 * mean(ov):.0f}%")


def compute_scores(key):
    path = OUT / "scores" / f"{key}_scores.pt"
    src = find_svd_outputs(key)
    model, tok, layers = load_svd_model(key, src)
    mag = torch.load(src / "importance.pt", map_location="cpu")["magnitude"]
    scores = {"magnitude": {n: s.float().cpu() for n, s in mag.items()}}
    if path.exists():
        scores.update(torch.load(path))
        log(f"  loaded saved scores from {path}")
    else:
        log("  Fisher pass")
        scores["fisher"] = fisher_scores(model, layers, tok)
        log("  gradient pass")
        scores["gradient"] = batch_gradient_scores(model, layers, tok)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({m: scores[m] for m in SCORE_NAMES}, path)
    agreement(scores)
    return scores


# ---------------------------------------------------------------- experiments
def tier_cfgs(param, value, fixed):
    cfgs = {"SLC": CrossbarConfig(bits_per_cell=CONFIG["slc_bits"], **fixed),
            "MLC": CrossbarConfig(bits_per_cell=CONFIG["mlc_bits"], **fixed)}
    if param == "mlc_level_sigma":
        cfgs["MLC"] = replace(cfgs["MLC"], level_sigma=tuple(value))
    else:
        cfgs = {t: replace(c, **{param: value}) for t, c in cfgs.items()}
    if param == "adc_bits" and value is None:
        cfgs = {t: replace(c, input_mode="dac") for t, c in cfgs.items()}   # same result, much faster
    return cfgs


def is_random(cfgs):
    return any(c.sigma > 0 or c.fault_rate > 0 or c.read_sigma > 0 for c in cfgs.values())


def label(v):
    if v is None:
        return "ideal"
    if isinstance(v, (tuple, list)):
        return "x" + "/".join(f"{a:g}" for a in v)
    return f"{v:g}" if isinstance(v, float) else str(v)


BOTH = list(MODELS)
# random is a control: run where it was reported (noise, SLC share, stuck cells), not for the chip effects
BASIC = ["all_slc", "all_mlc", "random", "magnitude", "gradient", "fisher"]
CHIP = ["all_slc", "all_mlc", "magnitude", "gradient", "fisher"]
# Batch sizes: inputs are quantised to 8 bits with one scale per batch, so accuracy depends slightly on
# batch composition. 128 (basic sweeps) and 64 (realistic-chip effects) reproduce the reported numbers.
EXPERIMENTS = {
    "slc_share": {"param": "sigma", "values": [0.03, 0.05, 0.075], "fixed": {},
                  "fracs": [0.05, 0.1, 0.2, 0.3, 0.5], "seeds": 20, "batch": 128, "models": BOTH,
                  "policies": BASIC},
    "noise": {"param": "sigma", "values": [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1], "fixed": {},
              "fracs": [0.2], "seeds": 5, "batch": 128, "models": BOTH, "policies": BASIC},
    "fault": {"param": "fault_rate", "values": [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05], "fixed": {},
              "fracs": [0.2], "seeds": 20, "batch": 128, "models": BOTH, "policies": BASIC},
    "read": {"param": "read_sigma", "values": [0.0, 0.005, 0.01, 0.02, 0.03, 0.05], "fixed": {},
             "fracs": [0.2], "seeds": 5, "batch": 64, "models": BOTH, "policies": CHIP},
    "level": {"param": "mlc_level_sigma", "values": [(1, 1, 1, 1), (1, 1.5, 1.5, 1), (1, 2, 2, 1)],
              "fixed": {"sigma": 0.05}, "fracs": [0.2], "seeds": 10, "batch": 64, "models": BOTH, "policies": CHIP},
    "cluster": {"param": "fault_mode", "values": ["uniform", "cluster", "line"], "fixed": {"fault_rate": 0.01},
                "fracs": [0.2], "seeds": 10, "batch": 64, "models": BOTH, "policies": CHIP},
    "irdrop": {"param": "ir_ratio", "values": [0.0, 1e-4, 3e-4, 1e-3, 3e-3], "fixed": {},
               "fracs": [0.2], "seeds": 1, "batch": 64, "models": BOTH, "policies": CHIP},
    "adc": {"param": "adc_bits", "values": [None, 8, 7, 6, 5, 4, 3], "fixed": {"input_mode": "bit_serial"},
            "fracs": [0.2], "seeds": 1, "batch": 64, "models": BOTH, "policies": CHIP},
    "adc_noise": {"param": "adc_bits", "values": [None, 8, 6, 4], "fixed": {"input_mode": "bit_serial", "sigma": 0.05},
                  "fracs": [0.2], "seeds": 3, "batch": 64, "models": ["distilbert-sst2"], "policies": CHIP},
}
if EXPERIMENT == "smoke":
    # tiny version of every experiment: DistilBERT, 128 sentences, 1 chip, 2 sweep values
    CONFIG.update(eval_n=128, fisher_n=64, adc_calib_n=16)
    OUT = OUT.with_name(OUT.name + "_smoke")
    for e in EXPERIMENTS.values():
        e.update(values=e["values"][:2], fracs=e["fracs"][:1], seeds=1, models=["distilbert-sst2"])
    EXPERIMENTS["adc"]["values"] = [None, 6]
    RUN = list(EXPERIMENTS)
elif EXPERIMENT == "all":
    RUN = list(EXPERIMENTS)
else:
    RUN = [EXPERIMENT]


def over_budget():
    return (time.time() - T_START) / 3600 > TIME_BUDGET_H


def run_experiment(name, e, key, scores):
    out = OUT / name / key
    out.mkdir(parents=True, exist_ok=True)
    CONFIG["batch_size"] = e["batch"]
    src = find_svd_outputs(key)
    digital, tok, layers = load_svd_model(key, src)
    batches = sst2_batches(tok, "validation", CONFIG["eval_n"], 0)
    needs_adc = any(tier_cfgs(e["param"], v, e["fixed"])["SLC"].adc_bits is not None for v in e["values"])
    calib = sst2_batches(tok, "train", CONFIG["adc_calib_n"], 2) if needs_adc else None
    jobs = []
    for p in e["policies"]:
        jobs += [(p, 1.0)] if p == "all_slc" else [(p, 0.0)] if p == "all_mlc" else [(p, f) for f in e["fracs"]]
    n_seeds = {label(v): e["seeds"] if is_random(tier_cfgs(e["param"], v, e["fixed"])) else 1 for v in e["values"]}
    n_points = len(jobs) * sum(n_seeds.values())
    log(f"  {key}: {len(jobs)} placements x {len(e['values'])} values of {e['param']} = {n_points} evaluations "
        f"({sum(len(b['labels']) for b in batches)} sentences, batch {e['batch']})")
    rows, done, t0 = [], 0, time.time()
    with open(out / "sweep.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "slc_frac", "param", "value", "seed", "acc"])
        for policy, frac in jobs:
            if over_budget():
                log(f"  !! time budget of {TIME_BUDGET_H} h reached: skipping the rest of {name}")
                break
            model = copy.deepcopy(digital)
            masks = tier_masks(layers, scores, policy, frac)
            to_reram(model, layers, masks)
            model.to(DEVICE).eval()
            n_slc = sum(int(m.sum()) for m in masks.values())
            log(f"    placement {policy} {100 * frac:.0f}% SLC ({n_slc}/{sum(m.numel() for m in masks.values())} components)")
            if needs_adc:
                calibrate_adc(model, tier_cfgs(e["param"], next(v for v in e["values"] if v is not None),
                                               e["fixed"]), calib)
            for v in e["values"]:
                cfgs, accs = tier_cfgs(e["param"], v, e["fixed"]), []
                for seed in range(n_seeds[label(v)]):
                    program_model(model, cfgs, seed)
                    acc = evaluate(model, batches)
                    accs.append(acc)
                    rows.append((policy, frac, label(v), acc))
                    w.writerow([policy, frac, e["param"], label(v), seed, f"{acc:.6f}"])
                    done += 1
                fh.flush()
                eta = (time.time() - t0) / max(done, 1) * (n_points - done)
                log(f"    [{done:4d}/{n_points}] {policy:16s} {100 * frac:3.0f}% SLC  {e['param']}={label(v):10s} "
                    f"acc {mean(accs):.4f} +- {pstdev(accs):.4f} (min {min(accs):.4f})  eta {eta / 60:.0f} min")
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    groups = defaultdict(list)
    for p, f, v, a in rows:
        groups[(p, f, v)].append(a)
    with open(out / "sweep_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "slc_frac", "param", "value", "acc_mean", "acc_std", "acc_min", "n"])
        for (p, f, v), a in groups.items():
            w.writerow([p, f, e["param"], v, f"{mean(a):.6f}", f"{pstdev(a):.6f}", f"{min(a):.6f}", len(a)])
    log(f"  summary ({key}, {name}): mean accuracy %")
    values = [label(v) for v in e["values"]]
    log(f"    {'placement':24s} " + " ".join(f"{v:>10s}" for v in values))
    for p, f in jobs:
        cells = [groups.get((p, f, v)) for v in values]
        log(f"    {p + f' {100 * f:.0f}%':24s} " + " ".join("         -" if c is None else f"{100 * mean(c):10.2f}" for c in cells))


# ---------------------------------------------------------------- main
log(f"device: {DEVICE} {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else ''}  torch {torch.__version__}")
log(f"experiments: {RUN}, output: {OUT}")
selftest()
models_needed = [k for k in MODELS if any(k in EXPERIMENTS[n]["models"] for n in RUN)]
SCORES = {}
log(f"[1/2] ranking scores for {models_needed}")
for key in models_needed:
    log(f"  ===== {key}")
    SCORES[key] = compute_scores(key)
log("[2/2] experiments")
skipped = []
for i, name in enumerate(RUN, 1):
    e = EXPERIMENTS[name]
    log(f"################ {name} ({i}/{len(RUN)}), elapsed {(time.time() - T_START) / 3600:.1f} h")
    if over_budget():
        skipped.append(name)
        log(f"  !! time budget reached, skipping {name}")
        continue
    for key in e["models"]:
        try:
            run_experiment(name, e, key, SCORES[key])
        except Exception as ex:          # keep going with the next experiment
            skipped.append(f"{name}/{key}")
            log(f"  !! {name} / {key} failed: {type(ex).__name__}: {ex}")
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    shutil.make_archive(str(OUT), "zip", OUT)
    log(f"  results so far zipped to {OUT}.zip")
log(f"finished in {(time.time() - T_START) / 3600:.1f} h. skipped or failed: {skipped or 'none'}")
