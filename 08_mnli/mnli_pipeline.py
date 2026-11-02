!pip install -q hf_xet psutil
# Do NOT `pip install -U datasets` here: it swaps pyarrow under the running kernel and
# breaks with "pyarrow.lib.Array size changed". Kaggle's preinstalled datasets is enough.
#
# MNLI version of the whole study, in ONE Kaggle notebook cell:
#   1. SVD + recovery fine-tune      (as 01_svd_tiering/svd_tiering.py)
#   2. ranking scores                 magnitude, gradient (HyFlexPIM batch), fisher
#   3. slc_share                      SLC share 5-50% at programming noise sigma = 5%      20 chips
#   4. fault                          stuck-at cells 0-5%, 20% SLC                          20 chips
#   5. mapping                        fault-aware mapping (remap + C1), 20% SLC             10 chips
# Simulator, placements, scores and mapping are the same code as steps 02-06 of this repository.
# Evaluation: a fixed random subset of 2,000 MNLI validation_matched pairs (shuffle seed 0).
#
# Models: typeform/distilbert-base-uncased-mnli, textattack/bert-base-uncased-MNLI. Their label order
# differs from GLUE (0 entailment, 1 neutral, 2 contradiction), so the script measures the label
# permutation on the evaluation subset, logs it, and maps every GLUE label into the model's order.
#
# Setup: GPU T4, Internet on, paste into ONE cell. MODE = "smoke" first (~5 min), then "full" with
# Save & Run All (~6 h, see 08_mnli/README.md). Every stage is resumable: if the session stops, add this
# notebook's output as input to a new run and it continues where it stopped (finished chips are kept).
# Output: /kaggle/working/mnli_results/{svd,scores,slc_share,fault,mapping}/..., log.txt,
#         stage_times.csv and mnli_results.zip (everything except the svd_model.pt weights).
import copy
import csv
import gc
import json
import math
import os
import re
import shutil
import time
import traceback
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, pstdev

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODE = "smoke"              # "smoke" | "full"
TIME_BUDGET_H = 11.0        # Kaggle stops sessions at 12 h; the run stops cleanly after this

MODELS = {
    "distilbert-mnli": {"hf_id": "typeform/distilbert-base-uncased-mnli",
                        "target": r"^distilbert\.transformer\.layer\.\d+\."},
    "bert-base-mnli": {"hf_id": "textattack/bert-base-uncased-MNLI",
                       "target": r"^bert\.encoder\.layer\.\d+\."},
}
RUN_MODELS = list(MODELS)
GLUE_LABELS = ["entailment", "neutral", "contradiction"]
CONFIG = {
    # data
    "eval_split": "validation_matched", "eval_n": 2000, "eval_seed": 0, "eval_full": True, "max_len": 128,
    # SVD + recovery fine-tune (MNLI train has 392,702 pairs; a 100k subset keeps the run in one session)
    "train_n": 100_000, "epochs": 2, "lr": 2e-5, "ft_batch": 32, "amp": True, "log_every": 250,
    # scores and simulator
    "fisher_n": 2048, "grad_batch": 32, "sim_batch": 128,
    "slc_bits": 1, "mlc_bits": 2, "slc_frac": 0.2, "design_sigma": 0.02,
}
BASIC = ["all_slc", "all_mlc", "random", "magnitude", "gradient", "fisher"]
SWEEPS = {
    "slc_share": {"param": "sigma", "values": [0.05], "fixed": {}, "fracs": [0.05, 0.1, 0.2, 0.3, 0.5],
                  "seeds": 20, "policies": BASIC},
    "fault": {"param": "fault_rate", "values": [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05], "fixed": {},
              "fracs": [0.2], "seeds": 20, "policies": BASIC},
}
MAPPING = {"param": "fault_rate", "values": [0.002, 0.005, 0.01, 0.02], "seeds": 10,
           "policies": ["all_slc", "all_slc_remap", "magnitude", "magnitude_remap", "c1"]}

OUT = Path("/kaggle/working/mnli_results") if Path("/kaggle/working").exists() else Path("mnli_results")
if MODE == "smoke":
    CONFIG.update(eval_n=128, eval_full=False, train_n=512, epochs=1, fisher_n=64, log_every=4)
    for e in SWEEPS.values():
        e.update(values=e["values"][:2], fracs=e["fracs"][:1], seeds=1)
    MAPPING.update(values=MAPPING["values"][:1], seeds=1)
    RUN_MODELS = ["distilbert-mnli"]
    OUT = OUT.with_name(OUT.name + "_smoke")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = CONFIG["amp"] and DEVICE.type == "cuda"
T_START = time.time()
STAGE_TIMES = []
OUT.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT / "log.txt"


def log(msg):
    line = f"[{time.time() - T_START:7.1f}s] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as fh:
        fh.write(line + "\n")


def mem(short=False):
    vm = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss / 2 ** 30
    if short:
        s = f"RAM {rss:.1f}G"
        return s + (f" GPU {torch.cuda.memory_allocated() / 2 ** 30:.1f}G" if DEVICE.type == "cuda" else "")
    s = f"RAM: python {rss:.1f} GB, free {vm.available / 2 ** 30:.1f} of {vm.total / 2 ** 30:.1f} GB"
    if DEVICE.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        s += (f" | GPU: {torch.cuda.memory_allocated() / 2 ** 30:.1f} GB in use, "
              f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} of {total:.1f} GB")
    return s


def free():
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    if psutil.virtual_memory().available < 2 * 2 ** 30:
        log(f"  !! low RAM: {mem()}")


class BudgetStop(Exception):
    pass


def check_budget():
    if (time.time() - T_START) / 3600 > TIME_BUDGET_H:
        raise BudgetStop(f"time budget of {TIME_BUDGET_H} h reached")


@contextmanager
def stage(key, name):
    t = time.time()
    log(f"---- {key} | {name} | {mem()}")
    try:
        yield
    finally:
        STAGE_TIMES.append((key, name, round(time.time() - t, 1)))
        free()
        log(f"---- {key} | {name} finished in {(time.time() - t) / 60:.1f} min | {mem()}")


# ---------------------------------------------------------------- crossbar simulator (as reram_sim / mapping)
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
        self.fs = None
        self.calibrating = False
        self.fs_obs = None

    def segments(self, device):
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
        return math.sqrt(2) * self.cfg.read_sigma / self.delta

    def _fast(self, xi):
        y = F.linear(xi, self.w_hat)
        if self.cfg.read_sigma > 0:
            std = self.read_noise_k() * math.sqrt(self.lsum2) * xi.norm(dim=-1, keepdim=True)
            y = y + torch.randn_like(y) * std * self.scale.squeeze(1)
        return y

    def _planes(self, xt):
        if self.cfg.input_mode != "bit_serial":
            return [(1.0, xt)]
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
        """orders: {tier: component indices in physical slot order}."""
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.tiers = nn.ModuleDict()
        for tier, idx in orders.items():
            if idx.numel():
                self.tiers[tier] = nn.Sequential(Crossbar(SVh[idx]), Crossbar(U[:, idx]))
        self.register_buffer("bias", None if bias is None else bias.detach().clone().float())
        return self

    def program(self, cfgs, gen, faults=None):
        for tier, pair in self.tiers.items():
            for i, xbar in enumerate(pair):
                xbar.program(cfgs[tier], gen, None if faults is None else faults[tier][i])

    def forward(self, x):
        y = sum(pair(x) for pair in self.tiers.values())
        return y if self.bias is None else y + self.bias


def program_model(model, cfgs, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, ReRAMLinear):
            m.program(cfgs, gen)


# ---------------------------------------------------------------- SVD layer (as svd_tiering)
class SVDLinear(nn.Module):
    """W ~= U diag(S) Vh. Crossbars: x -> [S Vh (k x in)] -> [U (out x k)] -> y."""

    def __init__(self, U, S, Vh, bias):
        super().__init__()
        self.in_features, self.out_features, self.rank = Vh.shape[1], U.shape[0], S.shape[0]
        self.U = nn.Parameter(U.contiguous())
        self.S = nn.Parameter(S.contiguous())
        self.Vh = nn.Parameter(Vh.contiguous())
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

    @classmethod
    def from_linear(cls, linear, rank):
        U, S, Vh = torch.linalg.svd(linear.weight.data.float(), full_matrices=False)
        k = min(rank, S.shape[0])
        bias = linear.bias.data.float() if linear.bias is not None else None
        return cls(U[:, :k], S[:k], Vh[:k], bias)

    @classmethod
    def shell(cls, m, k, n, has_bias):
        return cls(torch.zeros(m, k), torch.zeros(k), torch.zeros(k, n), torch.zeros(m) if has_bias else None)

    def forward(self, x):
        return F.linear(F.linear(x, self.Vh) * self.S, self.U, self.bias)


def replace_modules(model, build):
    for name, new in build.items():
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, new)


def factorize(model, key):
    pattern = re.compile(MODELS[key]["target"])
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and pattern.match(n)]
    layers = {name: SVDLinear.from_linear(lin, (lin.out_features * lin.in_features) // (lin.out_features + lin.in_features))
              for name, lin in targets}
    replace_modules(model, layers)
    for m, n, k in sorted({(l.out_features, l.in_features, l.rank) for l in layers.values()}):
        log(f"    W {m}x{n} -> U {m}x{k} + S Vh {k}x{n}  (rank {k} of {min(m, n)})")
    return layers


# ---------------------------------------------------------------- MNLI data
def mnli(split):
    return load_dataset("nyu-mll/glue", "mnli", split=split)


def eval_subset(n):
    ds = mnli(CONFIG["eval_split"])
    return ds if n is None or n >= len(ds) else ds.shuffle(seed=CONFIG["eval_seed"]).select(range(n))


def calib_pairs(n):
    # calibration pairs for the scores: MNLI train, shuffle seed 1 (as SST-2)
    return mnli("train").shuffle(seed=1).select(range(n))


def encode(tok, ds, perm):
    """Tokenise (premise, hypothesis) pairs; labels mapped from GLUE order into the model's order."""
    def f(b):
        enc = tok(b["premise"], b["hypothesis"], truncation=True, max_length=CONFIG["max_len"])
        enc["labels"] = [perm[l] for l in b["label"]]
        return enc
    return ds.map(f, batched=True, remove_columns=ds.column_names)


def gpu_batches(tok, ds, perm, bs):
    """Batches preloaded on the GPU, sorted by length (much less padding; same order every run)."""
    enc = encode(tok, ds, perm)
    lengths = [len(x) for x in enc["input_ids"]]
    enc = enc.select(sorted(range(len(enc)), key=lengths.__getitem__))
    loader = DataLoader(enc, batch_size=bs, collate_fn=DataCollatorWithPadding(tok))
    batches = [{k: v.to(DEVICE) for k, v in b.items()} for b in loader]
    log(f"    {len(enc)} pairs, {len(batches)} batches of {bs}, mean length {mean(lengths):.0f} tokens")
    return batches


@torch.no_grad()
def evaluate(model, batches, amp=False):
    model.eval()
    correct = total = 0
    for b in batches:
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=amp and USE_AMP):
            logits = model(**b).logits
        correct += (logits.argmax(-1) == b["labels"]).sum().item()
        total += b["labels"].numel()
    return correct / total


@torch.no_grad()
def find_label_perm(model, tok, ds):
    """perm[g] = model output index for GLUE label g, chosen to maximise agreement on ds."""
    model.to(DEVICE).eval()
    conf = torch.zeros(3, 3)
    for b in gpu_batches(tok, ds, [0, 1, 2], 128):
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            pred = model(**b).logits.argmax(-1)
        for g, p in zip(b["labels"].tolist(), pred.tolist()):
            conf[g, p] += 1
    _, cols = linear_sum_assignment(-conf.numpy())
    perm = [int(c) for c in cols]
    rows3 = [0, 1, 2]
    names = {i: str(n).lower() for i, n in model.config.id2label.items()}
    by_name = [next((i for i, n in names.items() if n.startswith(g[:4])), None) for g in GLUE_LABELS]
    log(f"    config id2label {model.config.id2label}")
    log(f"    accuracy with labels as-is {conf.trace() / conf.sum():.4f}, with measured permutation "
        f"{conf[rows3, perm].sum() / conf.sum():.4f}; GLUE label -> model index {perm}")
    if None not in by_name and by_name != perm:
        log(f"    !! config names suggest {by_name}, the data says {perm}; using the data")
    if conf[rows3, perm].sum() / conf.sum() < 0.5:
        raise RuntimeError("baseline below 50% even after label permutation: wrong checkpoint?")
    return perm


# ---------------------------------------------------------------- stage 1: SVD + recovery fine-tune
def finetune(model, loader, epochs, lr):
    steps = epochs * len(loader)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, total_iters=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    step, running, t0 = 0, 0.0, time.time()
    for epoch in range(epochs):
        for batch in loader:
            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                loss = model(**{k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            running += loss.item()
            if step % CONFIG["log_every"] == 0 or step == steps:
                n = CONFIG["log_every"] if step % CONFIG["log_every"] == 0 else step % CONFIG["log_every"]
                rate = (time.time() - t0) / step
                log(f"    epoch {epoch + 1}/{epochs} step {step}/{steps}  loss {running / n:.4f}  "
                    f"lr {sched.get_last_lr()[0]:.2e}  {rate:.3f} s/step  eta {rate * (steps - step) / 60:.0f} min  "
                    f"| {mem(True)}")
                running = 0.0
    model.eval()


def stage_svd(key):
    d = OUT / "svd" / key
    if (d / "summary.json").exists():
        log(f"    already done ({d / 'summary.json'}), skipping")
        return
    d.mkdir(parents=True, exist_ok=True)
    hf_id = MODELS[key]["hf_id"]
    log(f"    loading {hf_id}")
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id).to(DEVICE)
    log(f"    {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters | {mem(True)}")
    val = eval_subset(CONFIG["eval_n"])
    log("    label order")
    perm = find_label_perm(model, tok, val)
    log(f"    evaluation subset ({CONFIG['eval_split']})")
    eval_b = gpu_batches(tok, val, perm, 128)
    full_b = None
    if CONFIG["eval_full"]:
        log(f"    full {CONFIG['eval_split']} (digital accuracies only)")
        full_b = gpu_batches(tok, eval_subset(None), perm, 128)
    acc = {"baseline": evaluate(model, eval_b, amp=True)}
    if full_b:
        acc["baseline_full"] = evaluate(model, full_b, amp=True)
    log(f"    baseline acc {acc}")

    log("    SVD factorisation, k = floor(mn/(m+n))")
    layers = factorize(model, key)
    model.to(DEVICE)
    acc["svd_truncated"] = evaluate(model, eval_b, amp=True)
    log(f"    {len(layers)} layers factorised; acc {acc['svd_truncated']:.4f} | {mem(True)}")

    log(f"    training data: {CONFIG['train_n']} MNLI train pairs (shuffle seed 0)")
    train = encode(tok, mnli("train").shuffle(seed=0).select(range(CONFIG["train_n"])), perm)
    loader = DataLoader(train, batch_size=CONFIG["ft_batch"], shuffle=True, generator=torch.Generator().manual_seed(0),
                        collate_fn=DataCollatorWithPadding(tok), num_workers=2, pin_memory=True)
    log(f"    recovery fine-tune: {CONFIG['epochs']} epochs x {len(loader)} steps, lr {CONFIG['lr']}, amp {USE_AMP}")
    torch.manual_seed(0)
    finetune(model, loader, CONFIG["epochs"], CONFIG["lr"])
    acc["svd_finetuned"] = evaluate(model, eval_b, amp=True)
    if full_b:
        acc["svd_finetuned_full"] = evaluate(model, full_b, amp=True)
    log(f"    after fine-tune {acc}")

    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, d / "svd_model.pt")
    (d / "tier_map.json").write_text(json.dumps(
        {n: {"out_features": l.out_features, "in_features": l.in_features, "rank": l.rank} for n, l in layers.items()},
        indent=1))
    summary = {"model": key, "hf_id": hf_id, "label_perm": perm, "accuracy": acc, "num_layers": len(layers),
               "components": sum(l.rank for l in layers.values()), "train_n": CONFIG["train_n"],
               "epochs": CONFIG["epochs"], "eval_n": CONFIG["eval_n"]}
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"    wrote svd_model.pt, tier_map.json, summary.json to {d}")
    del model, layers, eval_b, full_b, train, loader


def load_svd(key):
    d = OUT / "svd" / key
    hf_id = MODELS[key]["hf_id"]
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id)
    dims = json.loads((d / "tier_map.json").read_text())
    shells = {name: SVDLinear.shell(e["out_features"], e["rank"], e["in_features"],
                                    model.get_submodule(name).bias is not None) for name, e in dims.items()}
    replace_modules(model, shells)
    model.load_state_dict(torch.load(d / "svd_model.pt", map_location="cpu"))
    summary = json.loads((d / "summary.json").read_text())
    return model.eval(), tok, shells, summary["label_perm"], summary


# ---------------------------------------------------------------- stage 2: ranking scores
def entry_fisher(model, layers, tok, perm):
    """Empirical Fisher (sum of squared per-pair gradients) of every stored entry, per layer:
    F1 [k, n] for the first crossbar (entries sigma_r * v_rj) and F2 [m, k] for the second (u_ir)."""
    n = CONFIG["fisher_n"]
    ds = calib_pairs(n)
    model.to(DEVICE).eval()
    params = [p for l in layers.values() for p in (l.U, l.Vh)]
    req = {p: p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    acc = [torch.zeros_like(p) for p in params]
    t0 = time.time()
    for i, ex in enumerate(ds):
        enc = tok(ex["premise"], ex["hypothesis"], truncation=True, max_length=CONFIG["max_len"],
                  return_tensors="pt").to(DEVICE)
        loss = model(**enc, labels=torch.tensor([perm[ex["label"]]], device=DEVICE)).loss
        for a, g in zip(acc, torch.autograd.grad(loss, params)):
            a += g.detach() ** 2
        if (i + 1) % 256 == 0 or i + 1 == n:
            rate = (time.time() - t0) / (i + 1)
            log(f"    fisher pair {i + 1}/{n}  eta {rate * (n - i - 1) / 60:.1f} min | {mem(True)}")
    for p, r in req.items():
        p.requires_grad_(r)
    model.cpu()
    out = {}
    for j, (name, l) in enumerate(layers.items()):
        S = l.S.detach().float().to(DEVICE)
        out[name] = (acc[2 * j + 1] / S[:, None].pow(2).clamp(min=1e-12), acc[2 * j])
    del acc
    free()
    return out


def component_fisher(layers, EF):
    """score_r = sum_j F(SVh_rj) s_r^2 + sum_i F(U_ir) t_i^2, s_r / t_i the 8-bit row scales."""
    comp, row_share = {}, []
    for name, l in layers.items():
        F1, F2 = EF[name]
        U = l.U.detach().float().to(DEVICE)
        SVh = (l.S[:, None] * l.Vh).detach().float().to(DEVICE)
        row = SVh.abs().amax(1).pow(2) * F1.sum(1)
        col = (U.abs().amax(1).pow(2)[:, None] * F2).sum(0)
        comp[name] = (row + col).cpu()
        row_share.append((row.sum() / (row + col).sum().clamp(min=1e-30)).item())
    log(f"    fisher: {100 * mean(row_share):.0f}% of the score from the sigma*v rows, "
        f"{100 - 100 * mean(row_share):.0f}% from the u columns")
    return comp


def batch_gradient_scores(model, layers, tok, perm):
    """HyFlexPIM-style: sum over batches of |dL_batch / dsigma_r|, dropout on (training mode)."""
    ds = encode(tok, calib_pairs(CONFIG["fisher_n"]), perm)
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
    for b in loader:
        loss = model(**{k: v.to(DEVICE) for k, v in b.items()}).loss
        for a, g in zip(score, torch.autograd.grad(loss, S)):
            a += g.detach().abs()
    for p, r in req.items():
        p.requires_grad_(r)
    model.eval().cpu()
    log(f"    gradient over {len(loader)} batches of {CONFIG['grad_batch']}")
    return {name: s.cpu() for name, s in zip(layers, score)}


def agreement(scores, frac=0.2):
    mag = scores["magnitude"]
    ranks = lambda x: x.argsort().argsort().float()
    for m in ("gradient", "fisher"):
        rho, ov = [], []
        for name, ref in mag.items():
            rho.append(torch.corrcoef(torch.stack([ranks(ref), ranks(scores[m][name])]))[0, 1].item())
            k = math.ceil(frac * ref.numel())
            ov.append(len(set(torch.topk(ref, k).indices.tolist()) & set(torch.topk(scores[m][name], k).indices.tolist())) / k)
        log(f"    {m:9s} vs magnitude: rank correlation {mean(rho):.2f}, top-20% overlap {100 * mean(ov):.0f}%")


def stage_scores(key, model, layers, tok, perm, fisher_cache):
    path = OUT / "scores" / f"{key}_scores.pt"
    scores = {"magnitude": {n: l.S.detach().abs().float().cpu() for n, l in layers.items()}}
    if path.exists():
        scores.update(torch.load(path))
        log(f"    loaded saved scores from {path}")
    else:
        log(f"    Fisher pass over {CONFIG['fisher_n']} single pairs")
        scores["fisher"] = component_fisher(layers, fisher_cache())
        log("    gradient pass")
        scores["gradient"] = batch_gradient_scores(model, layers, tok, perm)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({m: scores[m] for m in ("gradient", "fisher")}, path)
        log(f"    saved {path}")
    agreement(scores)
    return scores


# ---------------------------------------------------------------- resumable sweep file
def label(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


class Sweep:
    """sweep.csv that keeps rows from an earlier (stopped) session and skips chips already measured."""

    def __init__(self, path, param):
        self.path, self.param, self.rows = path, param, []
        if path.exists():
            with open(path) as fh:
                for r in csv.DictReader(fh):
                    self.rows.append((r["policy"], r["slc_frac"], r["value"], int(r["seed"]), float(r["acc"])))
        self.done = {r[:4] for r in self.rows}
        new = not path.exists()
        self.fh = open(path, "a", newline="")
        self.w = csv.writer(self.fh)
        if new:
            self.w.writerow(["policy", "slc_frac", "param", "value", "seed", "acc"])

    def has(self, policy, frac, value, seed):
        return (policy, f"{frac:g}", label(value), seed) in self.done

    def add(self, policy, frac, value, seed, acc):
        row = (policy, f"{frac:g}", label(value), seed, acc)
        self.rows.append(row)
        self.done.add(row[:4])
        self.w.writerow([policy, f"{frac:g}", self.param, label(value), seed, f"{acc:.6f}"])
        self.fh.flush()

    def close(self):
        self.fh.close()

    def summarise(self, title, values):
        groups = defaultdict(list)
        for p, f, v, _, a in self.rows:
            groups[(p, f, v)].append(a)
        with open(self.path.with_name("sweep_summary.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["policy", "slc_frac", "param", "value", "acc_mean", "acc_std", "acc_min", "n"])
            for (p, f, v), a in groups.items():
                w.writerow([p, f, self.param, v, f"{mean(a):.6f}", f"{pstdev(a):.6f}", f"{min(a):.6f}", len(a)])
        vals = [label(v) for v in values]
        log(f"    summary {title}: mean accuracy % ({self.param})")
        log(f"      {'placement':22s} " + " ".join(f"{v:>8s}" for v in vals))
        for p, f in dict.fromkeys((p, f) for p, f, _ in groups):
            cells = [groups.get((p, f, v)) for v in vals]
            log(f"      {p + ' ' + f'{100 * float(f):.0f}%':22s} "
                + " ".join("       -" if c is None else f"{100 * mean(c):8.2f}" for c in cells))


# ---------------------------------------------------------------- stages 3-4: slc_share and fault
def tier_cfgs(param, value, fixed):
    return {t: CrossbarConfig(bits_per_cell=CONFIG["slc_bits"] if t == "SLC" else CONFIG["mlc_bits"],
                              **{**fixed, param: value}) for t in ("SLC", "MLC")}


def is_random(cfgs):
    return any(c.sigma > 0 or c.fault_rate > 0 or c.read_sigma > 0 for c in cfgs.values())


def tier_masks(layers, scores, policy, frac):
    if policy == "all_slc":
        return {n: torch.ones(l.rank, dtype=torch.bool) for n, l in layers.items()}
    if policy == "all_mlc":
        return {n: torch.zeros(l.rank, dtype=torch.bool) for n, l in layers.items()}
    if policy == "random":
        g = torch.Generator().manual_seed(0)
        s_all = {n: torch.rand(l.rank, generator=g) for n, l in layers.items()}
    else:
        s_all = scores[policy]
    masks = {}
    for n, s in s_all.items():
        mask = torch.zeros_like(s, dtype=torch.bool)
        mask[torch.topk(s, math.ceil(frac * s.numel())).indices] = True
        masks[n] = mask
    return masks


def to_reram(model, layers, masks):
    replace_modules(model, {name: ReRAMLinear(l.U.detach(), (l.S[:, None] * l.Vh).detach(), masks[name], l.bias)
                            for name, l in layers.items()})


def run_sweep(name, key, digital, layers, scores, batches):
    e = SWEEPS[name]
    out = OUT / name / key
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in e["policies"]:
        jobs += [(p, 1.0)] if p == "all_slc" else [(p, 0.0)] if p == "all_mlc" else [(p, f) for f in e["fracs"]]
    n_seeds = {label(v): e["seeds"] if is_random(tier_cfgs(e["param"], v, e["fixed"])) else 1 for v in e["values"]}
    sw = Sweep(out / "sweep.csv", e["param"])
    todo_all = [(p, f, v, s) for p, f in jobs for v in e["values"] for s in range(n_seeds[label(v)])
                if not sw.has(p, f, v, s)]
    n_points = len(jobs) * sum(n_seeds.values())
    log(f"    {len(jobs)} placements x {len(e['values'])} values of {e['param']} = {n_points} chips "
        f"({n_points - len(todo_all)} already measured), fixed {e['fixed'] or 'nothing else'}")
    done, t0 = 0, time.time()
    try:
        for policy, frac in jobs:
            if not any(t[0] == policy and t[1] == frac for t in todo_all):
                continue
            check_budget()
            model = copy.deepcopy(digital)
            masks = tier_masks(layers, scores, policy, frac)
            to_reram(model, layers, masks)
            model.to(DEVICE).eval()
            n_slc = sum(int(m.sum()) for m in masks.values())
            log(f"    placement {policy} {100 * frac:.0f}% SLC ({n_slc}/{sum(m.numel() for m in masks.values())} "
                f"components) | {mem(True)}")
            for v in e["values"]:
                cfgs, accs = tier_cfgs(e["param"], v, e["fixed"]), []
                for seed in range(n_seeds[label(v)]):
                    if sw.has(policy, frac, v, seed):
                        continue
                    program_model(model, cfgs, seed)
                    acc = evaluate(model, batches)
                    sw.add(policy, frac, v, seed, acc)
                    accs.append(acc)
                    done += 1
                if accs:
                    eta = (time.time() - t0) / done * (len(todo_all) - done)
                    log(f"    [{done:4d}/{len(todo_all)}] {policy:10s} {100 * frac:3.0f}% SLC  {e['param']}={label(v):7s} "
                        f"acc {mean(accs):.4f} +- {pstdev(accs):.4f} (min {min(accs):.4f})  "
                        f"{(time.time() - t0) / done:.1f} s/chip  eta {eta / 60:.0f} min | {mem(True)}")
                check_budget()
            del model
            free()
    finally:
        sw.close()
        sw.summarise(f"{name} ({key})", e["values"])


# ---------------------------------------------------------------- stage 5: fault-aware mapping (as 06_fault_aware_mapping)
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
    if cfg.fault_rate <= 0:
        return None
    return [{k: fault_masks(shape, cfg, gen, DEVICE) for k in ("pos", "neg")} for _ in range(cfg.n_slices)]


def stuck_error(gstar, d, cfg, t):
    g_min = 1.0 / cfg.on_off
    delta = (1.0 - g_min) / (cfg.levels - 1)
    return (gstar - (g_min + d.float() * delta)) / delta * cfg.levels ** t


def fault_cost(cfg, faults1, faults2, d1, sc1, F1, d2, sc2, F2):
    """Fisher-weighted squared weight error from the known stuck cells, for every candidate component
    (rows) in every physical slot of one tier (columns)."""
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


def layer_contexts(layers, EF, frac):
    """(hybrid contexts at frac SLC, all-SLC contexts); both share the same tensors."""
    hybrid, all_slc = {}, {}
    for name, l in layers.items():
        k = l.rank
        k_slc = math.ceil(frac * k)
        F1, F2 = EF[name]
        base = {"U": l.U.detach().float().to(DEVICE), "SVh": (l.S[:, None] * l.Vh).detach().float().to(DEVICE),
                "bias": l.bias, "F1": F1, "F2": F2, "mag_rank": torch.argsort(l.S.detach().abs().float().cpu(), descending=True)}
        hybrid[name] = {**base, "k_slc": k_slc, "n_slots": {"SLC": k_slc, "MLC": k - k_slc}}
        all_slc[name] = {**base, "k_slc": k, "n_slots": {"SLC": k, "MLC": 0}}
    return hybrid, all_slc


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


def run_mapping(key, digital, layers, batches, fisher_cache):
    e = MAPPING
    out = OUT / "mapping" / key
    out.mkdir(parents=True, exist_ok=True)
    sw = Sweep(out / "sweep.csv", e["param"])
    frac_of = lambda p: 1.0 if p.startswith("all_slc") else CONFIG["slc_frac"]
    todo_all = [(p, v, s) for v in e["values"] for p in e["policies"] for s in range(e["seeds"])
                if not sw.has(p, frac_of(p), v, s)]
    n_points = len(e["policies"]) * len(e["values"]) * e["seeds"]
    log(f"    {len(e['policies'])} placements x {len(e['values'])} values of {e['param']} x {e['seeds']} chips "
        f"= {n_points} chips ({n_points - len(todo_all)} already measured)")
    if not todo_all:
        sw.close()
        sw.summarise(f"mapping ({key})", e["values"])
        return
    ctx_hybrid, ctx_all = layer_contexts(layers, fisher_cache(), CONFIG["slc_frac"])
    model = copy.deepcopy(digital).to(DEVICE).eval()
    log(f"    contexts on GPU | {mem(True)}")
    done, t0 = 0, time.time()
    try:
        for v in e["values"]:
            base = replace(CrossbarConfig(), **{e["param"]: v})
            for policy in e["policies"]:
                frac = frac_of(policy)
                seeds = [s for s in range(e["seeds"]) if not sw.has(policy, frac, v, s)]
                if not seeds:
                    continue
                check_budget()
                accs, stats = [], defaultdict(float)
                for seed in seeds:
                    build_chip(model, ctx_all if policy.startswith("all_slc") else ctx_hybrid, policy, base, seed, stats)
                    acc = evaluate(model, batches)
                    sw.add(policy, frac, v, seed, acc)
                    accs.append(acc)
                    done += 1
                extra = ""
                if stats["before"] > 0:
                    extra = f"  stuck-cell damage cut by {100 * (1 - stats['after'] / stats['before']):.0f}%"
                if stats["slc"] > 0:
                    extra = f"  {100 * stats['moved'] / stats['slc']:.1f}% of SLC slots changed vs magnitude"
                eta = (time.time() - t0) / done * (len(todo_all) - done)
                log(f"    [{done:4d}/{len(todo_all)}] {e['param']}={label(v):6s} {policy:16s} acc {mean(accs):.4f} "
                    f"+- {pstdev(accs):.4f} (min {min(accs):.4f}){extra}  {(time.time() - t0) / done:.1f} s/chip  "
                    f"eta {eta / 60:.0f} min | {mem(True)}")
    finally:
        sw.close()
        sw.summarise(f"mapping ({key})", e["values"])
        del model, ctx_hybrid, ctx_all
        free()


# ---------------------------------------------------------------- self-tests
def selftest():
    log("[self-test] simulator and mapping cost on random matrices")
    torch.manual_seed(0)
    gen = lambda: torch.Generator(device=DEVICE).manual_seed(1)
    W = torch.randn(300, 260, device=DEVICE)
    x = torch.randn(64, 260, device=DEVICE)
    q, sc = quantize_symmetric(W, 8, dim=1)
    xi, xs = quantize_symmetric(x, 8)
    ref = F.linear(xi, q * sc) * xs
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    ok = True
    for bits in (1, 2):
        cb = Crossbar(W).to(DEVICE)
        cb.program(CrossbarConfig(bits_per_cell=bits), gen())
        err = rel(cb(x), ref)
        ok &= err < 1e-4
        log(f"  {'PASS' if err < 1e-4 else 'FAIL'}  ideal crossbar = INT8 matmul ({bits} b/cell): rel err {err:.1e}")
    m1, m0 = fault_masks((1024, 1024), CrossbarConfig(fault_rate=0.02), gen(), DEVICE)
    frac = (m1 | m0).float().mean().item()
    ok &= 0.014 < frac < 0.026
    log(f"  {'PASS' if 0.014 < frac < 0.026 else 'FAIL'}  uniform faults hit ~2% of cells: {100 * frac:.2f}%")
    k, n, m = 48, 70, 60
    U, SVh = torch.randn(m, k, device=DEVICE), torch.randn(k, n, device=DEVICE)
    F1, F2 = torch.rand(k, n, device=DEVICE), torch.rand(m, k, device=DEVICE)
    for bits in (1, 2):
        cfg = CrossbarConfig(bits_per_cell=bits, fault_rate=0.01)
        g = torch.Generator(device=DEVICE).manual_seed(3)
        f1, f2 = draw_faults((k, n), cfg, g), draw_faults((m, k), cfg, g)
        sc1, sc2 = SVh.abs().amax(1) / 127, U.abs().amax(1) / 127
        cost = fault_cost(cfg, f1, f2, cell_digits(SVh, sc1[:, None], cfg), sc1, F1,
                          cell_digits(U, sc2[:, None], cfg), sc2, F2)
        x1, x2 = Crossbar(SVh).to(DEVICE), Crossbar(U).to(DEVICE)
        x1.program(replace(cfg, fault_rate=0.0), g, f1)
        x2.program(replace(cfg, fault_rate=0.0), g, f2)
        q1, s1 = quantize_symmetric(SVh, 8, dim=1)
        q2, s2 = quantize_symmetric(U, 8, dim=1)
        actual = (F1 * (x1.w_hat - q1 * s1) ** 2).sum() + (F2 * (x2.w_hat - q2 * s2) ** 2).sum()
        pred = cost.diagonal().sum()
        rows, cols = assign(cost)
        good = abs(pred / actual - 1).item() < 0.1 and cost[rows, cols].sum() <= pred + 1e-9
        ok &= bool(good)
        log(f"  {'PASS' if good else 'FAIL'}  mapping cost {bits} b/cell: predicted {pred:.4g}, actual {actual:.4g}, "
            f"after assignment {cost[rows, cols].sum():.4g}")
    if not ok:
        raise RuntimeError("self-test failed; see FAIL lines above")


# ---------------------------------------------------------------- driver
def restore_previous():
    """Continue a stopped run: copy an earlier session's results (attached as input) into OUT."""
    src = Path("/kaggle/input")
    if not src.exists() or any(OUT.glob("svd/*/summary.json")):
        return
    for d in src.glob(f"**/{OUT.name}"):
        if d.is_dir() and any(d.glob("svd/*/summary.json")):
            log(f"resuming: copying earlier results from {d}")
            shutil.copytree(d, OUT, dirs_exist_ok=True, copy_function=shutil.copyfile,
                            ignore=shutil.ignore_patterns("log.txt", "*.zip", "stage_times.csv"))
            return


def zip_results():
    with open(OUT / "stage_times.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "stage", "seconds"])
        w.writerows(STAGE_TIMES)
    z = OUT.parent / f"{OUT.name}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in OUT.rglob("*"):
            if p.is_file() and p.name != "svd_model.pt":
                zf.write(p, p.relative_to(OUT.parent))
    log(f"  results zipped to {z} (svd_model.pt weights stay in {OUT / 'svd'})")


def run_model(key):
    log(f"==================== {key} ({MODELS[key]['hf_id']})")
    with stage(key, "[1/5] SVD + recovery fine-tune"):
        stage_svd(key)
    digital, tok, layers, perm, summary = load_svd(key)
    log(f"  loaded SVD model: {summary['num_layers']} layers, {summary['components']} components, "
        f"label permutation {perm}, digital accuracies {summary['accuracy']}")
    fisher = {}

    def fisher_cache():
        if "EF" not in fisher:
            fisher["EF"] = entry_fisher(digital, layers, tok, perm)
        return fisher["EF"]

    with stage(key, "[2/5] ranking scores"):
        scores = stage_scores(key, digital, layers, tok, perm, fisher_cache)
    log(f"  simulator evaluation batches ({CONFIG['eval_n']} {CONFIG['eval_split']} pairs, batch {CONFIG['sim_batch']})")
    batches = gpu_batches(tok, eval_subset(CONFIG["eval_n"]), perm, CONFIG["sim_batch"])
    log(f"  digital SVD model, fp32: acc {evaluate(digital.to(DEVICE), batches):.4f}")
    digital.cpu()
    free()
    with stage(key, "[3/5] slc_share, sigma = 5%"):
        run_sweep("slc_share", key, digital, layers, scores, batches)
    zip_results()
    with stage(key, "[4/5] fault rate"):
        run_sweep("fault", key, digital, layers, scores, batches)
    zip_results()
    with stage(key, "[5/5] fault-aware mapping"):
        run_mapping(key, digital, layers, batches, fisher_cache)
    del digital, layers, batches, fisher
    free()


log(f"device: {DEVICE} {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else '(no GPU: enable the accelerator)'}"
    f"  torch {torch.__version__}  mode {MODE}")
log(f"output {OUT} | {mem()}")
log(f"config {CONFIG}")
log(f"models {RUN_MODELS}; sweeps {[(n, e['values'], e['seeds']) for n, e in SWEEPS.items()]}; "
    f"mapping {MAPPING['values']} x {MAPPING['seeds']} chips")
restore_previous()
selftest()
failed = []
for key in RUN_MODELS:
    try:
        run_model(key)
    except BudgetStop as ex:
        log(f"!! {ex}: stopping. To continue, add this notebook's output as input to a new run and run again.")
        failed.append(f"{key} (stopped)")
        break
    except Exception as ex:
        log(f"!! {key} failed: {type(ex).__name__}: {ex}\n{traceback.format_exc()}")
        failed.append(key)
    finally:
        free()
        zip_results()
log("stage times (min): " + ", ".join(f"{k} {s.split(']')[0]}] {t / 60:.1f}" for k, s, t in STAGE_TIMES))
log(f"finished in {(time.time() - T_START) / 3600:.2f} h; failed or stopped: {failed or 'none'}. "
    f"Download {OUT}.zip from the Output panel")
