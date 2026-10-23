!pip install -q hf_xet
# Do NOT `pip install -U datasets` here: it swaps pyarrow under the running kernel and
# breaks with "pyarrow.lib.Array size changed". Kaggle's preinstalled datasets is enough.
# Kaggle GPU version of the ReRAM noise sweep (single notebook cell).
#
# Setup:
#   1. Settings -> Accelerator: GPU T4 x2 (or P100), Internet: On.
#   2. Add Input -> your earlier SVD-tiering notebook (its Output holds outputs/<model>/ or
#      outputs.zip), or upload svd_tiering/outputs.zip as a Dataset. The script searches
#      /kaggle/input for <model>/tier_map.json and unzips outputs.zip if needed.
#   3. Paste this whole file into ONE notebook cell and run it.
#
# Per model: Step 1 baselines (FP32 and INT8 accuracy of the original, non-SVD model);
# load the SVD factors and importance scores from Script 1 (plus Fisher importance, computed
# here); then for each SLC placement (policy x SLC fraction), each sigma (programming noise
# as a fraction of G_max), each fault rate and each chip seed: program every crossbar with
# simulated ReRAM cells and evaluate SST-2 validation accuracy.
# Choose the experiment with EXPERIMENT below (smoke / noise / slc_frac / fault).
# Results: /kaggle/working/reram_results/<experiment>/<model>/{sweep.csv, sweep_summary.csv,
#          sweep.png, config.json}
#
# Cell model (same as reram_sim/reram_sim/crossbar.py, which has the local tests):
#   8-bit signed weights per row, differential pair (G+, G-), bit-sliced into
#   base-2^b digits (SLC b=1, MLC b=2). Digit d -> g = g_min + d*delta, g_max = 1.
#   Noise g += sigma*N(0,1); stuck-at faults set g to G_max or G_min.
#   Ideal readout merges the slices; with adc_bits each slice and 128-row tile is
#   quantised by an ideally calibrated ADC before shift-and-add.
#   Runs in fp32 on purpose: fp16 overflows the integer-domain partial sums.
import copy
import csv
import json
import math
import shutil
import time
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, pstdev

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

# ---------------------------------------------------------------- config
# Pick ONE experiment here; its preset (below) overrides the matching CONFIG keys.
#   "smoke"    quick check that everything runs (DistilBERT, 128 sentences, 1 chip, ~1 min)
#   "noise"    plan task 6: accuracy vs programming noise sigma, 20% SLC
#   "slc_frac" plan task 7: accuracy vs SLC fraction at sigma = 3, 5, 7.5%
#   "fault"    plan task 8: accuracy vs stuck-at fault rate, 20% SLC, no noise
EXPERIMENT = "fault"

CONFIG = {
    "models": ["distilbert-sst2", "bert-base-sst2"],
    # all_slc / all_mlc ignore slc_fracs; the others put the top slc_frac of each layer in SLC
    "policies": ["all_slc", "all_mlc", "gradient", "magnitude", "fisher", "random"],
    "slc_fracs": [0.2],
    "sigmas": [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1],   # sigma / G_max
    "fault_rates": [0.0],
    "sa1_frac": 0.5,           # share of stuck cells stuck at G_max
    "seeds": 20,               # simulated chips per point
    "fisher_n": 2048,          # SST-2 train sentences for Fisher importance (same set as Script 1's calibration)
    "slc_bits": 1,
    "mlc_bits": 2,
    "w_bits": 8,
    "in_bits": 8,              # DAC precision; None = ideal inputs
    "adc_bits": None,          # e.g. 8; None = ideal readout (much faster)
    "tile_rows": 128,
    "eval_n": None,            # None = all 872 validation sentences
    "batch_size": 128,
    "out": Path("/kaggle/working/reram_results") if Path("/kaggle/working").exists() else Path("reram_results"),
}

PRESETS = {
    "smoke": {"models": ["distilbert-sst2"], "slc_fracs": [0.1, 0.2], "sigmas": [0.0, 0.05],
              "seeds": 1, "eval_n": 128, "fisher_n": 64},
    "noise": {"slc_fracs": [0.2], "sigmas": [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1], "fault_rates": [0.0]},
    "slc_frac": {"policies": ["all_slc", "all_mlc", "gradient", "magnitude", "fisher", "random"],
                 "slc_fracs": [0.05, 0.1, 0.2, 0.3, 0.5], "sigmas": [0.03, 0.05, 0.075], "fault_rates": [0.0]},
    "fault": {"policies": ["all_slc", "all_mlc", "gradient", "magnitude", "fisher", "random"],
              "slc_fracs": [0.2], "sigmas": [0.0], "fault_rates": [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]},
}
CONFIG.update(PRESETS[EXPERIMENT])

MODELS = {
    "distilbert-sst2": "distilbert-base-uncased-finetuned-sst-2-english",
    "bert-base-sst2": "textattack/bert-base-uncased-SST-2",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_START = time.time()


def log(msg):
    print(f"[{time.time() - T_START:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- crossbar
@dataclass(frozen=True)
class CrossbarConfig:
    bits_per_cell: int = 1
    sigma: float = 0.0
    fault_rate: float = 0.0
    sa1_frac: float = 0.5
    on_off: float = 100.0
    w_bits: int = 8
    in_bits: int | None = 8
    adc_bits: int | None = None
    tile_rows: int = 128

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


class Crossbar(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.detach().clone().float())
        for name in ("scale", "w_hat", "digits"):
            self.register_buffer(name, None, persistent=False)
        self.cfg = None

    @torch.no_grad()
    def program(self, cfg, gen=None):
        self.cfg = cfg
        q, self.scale = quantize_symmetric(self.weight, cfg.w_bits, dim=1)
        mag = {"pos": q.clamp(min=0), "neg": (-q).clamp(min=0)}
        L = cfg.levels
        g_min = 1.0 / cfg.on_off
        delta = (1.0 - g_min) / (L - 1)
        digits = []
        for s in range(cfg.n_slices):
            g = {k: self._cells(torch.div(m, L ** s, rounding_mode="floor") % L, cfg, g_min, delta, gen)
                 for k, m in mag.items()}
            digits.append((g["pos"] - g["neg"]) / delta)
        if cfg.adc_bits is None:
            self.w_hat, self.digits = sum(L ** s * d for s, d in enumerate(digits)) * self.scale, None
        else:
            self.w_hat, self.digits = None, torch.stack(digits)

    @staticmethod
    def _cells(d, cfg, g_min, delta, gen):
        g = g_min + d * delta
        if cfg.sigma > 0:
            g = (g + cfg.sigma * torch.randn(g.shape, generator=gen, device=g.device)).clamp(min=0)
        if cfg.fault_rate > 0:
            stuck = torch.rand(g.shape, generator=gen, device=g.device) < cfg.fault_rate
            sa1 = torch.rand(g.shape, generator=gen, device=g.device) < cfg.sa1_frac
            g = torch.where(stuck, torch.where(sa1, torch.ones_like(g), torch.full_like(g, g_min)), g)
        return g

    def forward(self, x):
        cfg = self.cfg
        x_scale = 1.0
        if cfg.in_bits is not None:
            x, x_scale = quantize_symmetric(x, cfg.in_bits)
        if cfg.adc_bits is None:
            return F.linear(x, self.w_hat) * x_scale
        R = cfg.tile_rows
        n_tiles = math.ceil(self.in_features / R)
        pad = n_tiles * R - self.in_features
        xt = F.pad(x, (0, pad)).unflatten(-1, (n_tiles, R))
        qmax = 2 ** (cfg.adc_bits - 1) - 1
        y = 0
        for s, d in enumerate(self.digits):
            dt = F.pad(d, (0, pad)).unflatten(-1, (n_tiles, R))
            part = torch.einsum("...tr,otr->...ot", xt, dt)
            step = part.abs().flatten(0, -2).amax(0).clamp(min=1e-12) / qmax
            y = y + cfg.levels ** s * (torch.round(part / step).clamp(-qmax, qmax) * step).sum(-1)
        return y * self.scale.squeeze(1) * x_scale


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


class Int8Linear(nn.Module):
    """Plain INT8 layer for the Step 1 baseline: no SVD, no cells, no noise.
    Weights: per-row symmetric w_bits; inputs: per-tensor symmetric in_bits (same as Crossbar)."""

    def __init__(self, linear):
        super().__init__()
        q, scale = quantize_symmetric(linear.weight.detach().float(), CONFIG["w_bits"], dim=1)
        self.register_buffer("weight", q * scale)
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone().float())

    def forward(self, x):
        x_scale = 1.0
        if CONFIG["in_bits"] is not None:
            x, x_scale = quantize_symmetric(x, CONFIG["in_bits"])
        return F.linear(x, self.weight) * x_scale + (0 if self.bias is None else self.bias)


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


def program_model(model, cfgs, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    for m in model.modules():
        if isinstance(m, ReRAMLinear):
            m.program(cfgs, gen)


# ---------------------------------------------------------------- data / model loading
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
    local = Path("../svd_tiering/outputs") / key            # when run outside Kaggle
    if (local / "tier_map.json").exists():
        return local
    raise FileNotFoundError(f"no {key}/tier_map.json under /kaggle/input; attach the SVD-tiering "
                            "notebook output or upload svd_tiering/outputs.zip as a dataset")


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


def select_slc(scores, frac):
    """Top ceil(frac*k) components per layer -> SLC (same rule as svd_tiering, scope=layer)."""
    masks = {}
    for name, s in scores.items():
        mask = torch.zeros_like(s, dtype=torch.bool)
        mask[torch.topk(s, math.ceil(frac * s.numel())).indices] = True
        masks[name] = mask
    return masks


def tier_masks(layers, importance, policy, frac):
    if policy == "all_slc":
        return {n: torch.ones(l.rank, dtype=torch.bool) for n, l in layers.items()}
    if policy == "all_mlc":
        return {n: torch.zeros(l.rank, dtype=torch.bool) for n, l in layers.items()}
    if policy == "random":
        g = torch.Generator().manual_seed(0)
        return select_slc({n: torch.rand(l.rank, generator=g) for n, l in layers.items()}, frac)
    return select_slc({n: s.cpu() for n, s in importance[policy].items()}, frac)


def to_reram(model, layers, masks):
    built = {}
    for name, svd in layers.items():
        U, SVh = svd.U.detach(), (svd.S[:, None] * svd.Vh).detach()
        built[name] = ReRAMLinear(U, SVh, masks[name], svd.bias)
    replace_modules(model, built)


def sst2_batches(tok):
    ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
    if CONFIG["eval_n"] is not None and CONFIG["eval_n"] < len(ds):
        ds = ds.shuffle(seed=0).select(range(CONFIG["eval_n"]))
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


FRAC_POLICIES = ("gradient", "magnitude", "fisher", "random")


def sweep_jobs():
    """(policy, slc_frac) pairs to run. all_slc / all_mlc run once (frac 1 / 0)."""
    jobs = [("all_slc", 1.0)] if "all_slc" in CONFIG["policies"] else []
    jobs += [("all_mlc", 0.0)] if "all_mlc" in CONFIG["policies"] else []
    jobs += [(p, f) for p in CONFIG["policies"] if p in FRAC_POLICIES for f in CONFIG["slc_fracs"]]
    return jobs


def fisher_importance(model, layers, tok):
    """Empirical Fisher I_r = sum_n (dL_n/dσ_r)^2 over single sentences (no sign cancellation,
    unlike |sum_n dL_n/dσ_r|). Same calibration sentences as Script 1 (train, shuffle seed 1)."""
    n = CONFIG["fisher_n"]
    ds = load_dataset("nyu-mll/glue", "sst2", split="train").shuffle(seed=1).select(range(n))
    model.to(DEVICE).eval()
    S = [l.S for l in layers.values()]
    req = {p: p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for s in S:
        s.requires_grad_(True)
    fisher = [torch.zeros_like(s) for s in S]
    for i, ex in enumerate(ds):
        enc = tok(ex["sentence"], truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        loss = model(**enc, labels=torch.tensor([ex["label"]], device=DEVICE)).loss
        for f, g in zip(fisher, torch.autograd.grad(loss, S)):
            f += g.detach() ** 2
        if (i + 1) % 256 == 0 or i + 1 == n:
            log(f"  fisher sentence {i + 1}/{n}")
    for p, r in req.items():
        p.requires_grad_(r)
    model.cpu()
    return {name: f.cpu() for name, f in zip(layers, fisher)}


def overlap_report(importance, frac):
    """How many SLC components gradient / magnitude / fisher agree on (top-frac per layer)."""
    names = [m for m in ("gradient", "magnitude", "fisher") if m in importance]
    masks = {m: select_slc(importance[m], frac) for m in names}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ma, mb = masks[names[a]], masks[names[b]]
            both = sum(int((ma[l] & mb[l]).sum()) for l in ma)
            total = sum(int(ma[l].sum()) for l in ma)
            log(f"  top-{100 * frac:.0f}% overlap {names[a]} vs {names[b]}: {both}/{total} ({100 * both / total:.1f}%)")


def plot(summary, digital, int8, path, key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr0 = CONFIG["fault_rates"][0]
    if len(CONFIG["slc_fracs"]) > 1:
        # accuracy vs SLC fraction, one panel per sigma; all_mlc / all_slc are the 0% / 100% ends
        sig = CONFIG["sigmas"]
        fig, axes = plt.subplots(1, len(sig), figsize=(5 * len(sig), 4), squeeze=False)
        for ax, s in zip(axes[0], sig):
            at = {(r["policy"], r["slc_frac"]): r for r in summary if r["sigma"] == s and r["fault_rate"] == fr0}
            ends = [at[e] for e in (("all_mlc", 0.0), ("all_slc", 1.0)) if e in at]
            for p in [p for p in CONFIG["policies"] if p in FRAC_POLICIES]:
                pts = sorted([r for (q, _), r in at.items() if q == p] + ends, key=lambda r: r["slc_frac"])
                ax.errorbar([100 * r["slc_frac"] for r in pts], [r["acc_mean"] for r in pts],
                            yerr=[r["acc_std"] for r in pts], marker="o", capsize=3, label=p)
            ax.axhline(digital, color="gray", ls="--", lw=1, label="digital SVD model")
            ax.axhline(int8, color="black", ls=":", lw=1, label="INT8 original (no SVD)")
            ax.set_xlabel("components in SLC (%)")
            ax.set_ylabel("SST-2 accuracy")
            ax.set_title(f"{key}, sigma = {100 * s:g}% G_max")
            ax.grid(alpha=0.3)
        axes[0][0].legend()
    else:
        sweep_sigma = len(CONFIG["sigmas"]) > 1
        x_key = "sigma" if sweep_sigma else "fault_rate"
        fig, ax = plt.subplots(figsize=(6, 4))
        by_policy = defaultdict(list)
        for r in summary:
            if sweep_sigma and r["fault_rate"] != fr0:
                continue
            by_policy[r["policy"]].append(r)
        for policy, rows in by_policy.items():
            rows.sort(key=lambda r: r[x_key])
            ax.errorbar([100 * r[x_key] for r in rows], [r["acc_mean"] for r in rows],
                        yerr=[r["acc_std"] for r in rows], marker="o", capsize=3, label=policy)
        ax.axhline(digital, color="gray", ls="--", lw=1, label="digital SVD model")
        ax.axhline(int8, color="black", ls=":", lw=1, label="INT8 original (no SVD)")
        ax.set_xlabel("sigma / G_max (%)" if sweep_sigma else "stuck-at fault rate (%)")
        ax.set_ylabel("SST-2 accuracy")
        ax.set_title(f"{key}, {100 * CONFIG['slc_fracs'][0]:g}% SLC")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()
    log(f"  plot -> {path}")


def print_tables(summary):
    """Mean accuracy tables. slc_frac sweep: one table per sigma (rows policy, cols SLC %).
    Otherwise: rows (policy, SLC %), cols sigma or fault rate."""
    at = {(r["policy"], r["slc_frac"], r["sigma"], r["fault_rate"]): r["acc_mean"] for r in summary}
    if len(CONFIG["slc_fracs"]) > 1:
        fr0 = CONFIG["fault_rates"][0]
        fracs = [0.0] + CONFIG["slc_fracs"] + [1.0]
        for s in CONFIG["sigmas"]:
            log(f"  sigma {100 * s:g}%   SLC% " + " ".join(f"{100 * f:>6.0f}" for f in fracs))
            for p in [p for p in CONFIG["policies"] if p in FRAC_POLICIES]:
                vals = [at.get(("all_mlc", 0.0, s, fr0)) if f == 0 else at.get(("all_slc", 1.0, s, fr0)) if f == 1
                        else at.get((p, f, s, fr0)) for f in fracs]
                log(f"  {p:17s} " + " ".join("     -" if v is None else f"{100 * v:6.2f}" for v in vals))
        return
    sweep_sigma = len(CONFIG["sigmas"]) > 1
    xs = CONFIG["sigmas"] if sweep_sigma else CONFIG["fault_rates"]
    log(f"  {'sigma' if sweep_sigma else 'fault'} %         " + " ".join(f"{100 * x:>6.1f}" for x in xs))
    for p, f in sweep_jobs():
        vals = [at.get((p, f, x, CONFIG["fault_rates"][0]) if sweep_sigma else (p, f, CONFIG["sigmas"][0], x)) for x in xs]
        log(f"  {p:9s} {100 * f:3.0f}% SLC " + " ".join(f"{100 * v:6.2f}" for v in vals))


# ---------------------------------------------------------------- Step 1 baselines
def run_baselines(key, names, batches):
    """FP32 and INT8 accuracy of the original model. INT8 quantises the same encoder layers
    that later go on crossbars; embeddings, LayerNorm and the head stay FP32 in both."""
    model = AutoModelForSequenceClassification.from_pretrained(MODELS[key]).eval().to(DEVICE)
    fp32 = evaluate(model, batches)
    log(f"  FP32 original acc {fp32:.4f}")
    replace_modules(model, {n: Int8Linear(model.get_submodule(n)) for n in names})
    log(f"  quantised {len(names)} encoder layers to INT8 (w {CONFIG['w_bits']}b per row, "
        f"in {CONFIG['in_bits']}b per tensor)")
    int8 = evaluate(model, batches)
    log(f"  INT8 original acc {int8:.4f}  (drop {100 * (fp32 - int8):+.2f} points)")
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return {"fp32_acc": fp32, "int8_acc": int8}


# ---------------------------------------------------------------- sweep
def run_model(key):
    out = CONFIG["out"] / EXPERIMENT / key
    out.mkdir(parents=True, exist_ok=True)
    log(f"========== {key} (experiment: {EXPERIMENT}) ==========")

    log("[1/7] locating SVD-tiering outputs")
    src = find_svd_outputs(key)
    log(f"  {src}")

    log("[2/7] loading SVD model and SST-2 validation")
    digital_model, tok, layers = load_svd_model(key, src)
    importance = torch.load(src / "importance.pt", map_location="cpu")
    batches = sst2_batches(tok)
    log(f"  {len(layers)} factorised layers, {sum(len(b['labels']) for b in batches)} sentences")

    log("[3/7] Step 1 baselines on the original (non-SVD) model")
    baseline = run_baselines(key, list(layers), batches)

    log("[4/7] digital reference (SVD model, FP32)")
    digital = evaluate(digital_model.to(DEVICE), batches)
    digital_model.cpu()
    log(f"  digital SVD model acc {digital:.4f}")

    log("[5/7] importance scores")
    importance = {m: {n: s.cpu().float() for n, s in d.items()} for m, d in importance.items()}
    log(f"  loaded from Script 1: {list(importance)}")
    if "fisher" in CONFIG["policies"]:
        log(f"  computing Fisher importance on {CONFIG['fisher_n']} sentences")
        importance["fisher"] = fisher_importance(digital_model, layers, tok)
        torch.save(importance["fisher"], out / "fisher_importance.pt")
    for frac in CONFIG["slc_fracs"]:
        overlap_report(importance, frac)

    base = dict(w_bits=CONFIG["w_bits"], in_bits=CONFIG["in_bits"], adc_bits=CONFIG["adc_bits"],
                tile_rows=CONFIG["tile_rows"], sa1_frac=CONFIG["sa1_frac"])
    tier_base = {"SLC": CrossbarConfig(bits_per_cell=CONFIG["slc_bits"], **base),
                 "MLC": CrossbarConfig(bits_per_cell=CONFIG["mlc_bits"], **base)}
    with open(out / "config.json", "w") as f:
        json.dump({"experiment": EXPERIMENT, "config": {k: str(v) for k, v in CONFIG.items()}, "source": str(src),
                   "crossbar": {t: asdict(c) for t, c in tier_base.items()}, "digital_acc": digital,
                   **baseline}, f, indent=2)

    jobs = sweep_jobs()
    n_points = len(jobs) * len(CONFIG["sigmas"]) * len(CONFIG["fault_rates"]) * CONFIG["seeds"]
    log(f"[6/7] sweep: {len(jobs)} placements x {len(CONFIG['sigmas'])} sigmas x "
        f"{len(CONFIG['fault_rates'])} fault rates x {CONFIG['seeds']} chips = {n_points} evaluations")
    rows, done, t0 = [], 0, time.time()
    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "slc_frac", "sigma", "fault_rate", "seed", "acc"])
        for policy, frac in jobs:
            model = copy.deepcopy(digital_model)
            masks = tier_masks(layers, importance, policy, frac)
            to_reram(model, layers, masks)
            model.to(DEVICE)
            n_slc = sum(int(m.sum()) for m in masks.values())
            log(f"  placement {policy} {100 * frac:.0f}%: {n_slc}/{sum(m.numel() for m in masks.values())} "
                f"components in SLC")
            for fault in CONFIG["fault_rates"]:
                for sigma in CONFIG["sigmas"]:
                    accs = []
                    for seed in range(CONFIG["seeds"]):
                        program_model(model, {t: replace(c, sigma=sigma, fault_rate=fault)
                                              for t, c in tier_base.items()}, seed)
                        acc = evaluate(model, batches)
                        accs.append(acc)
                        rows.append({"policy": policy, "slc_frac": frac, "sigma": sigma, "fault_rate": fault,
                                     "seed": seed, "acc": acc})
                        w.writerow([policy, frac, sigma, fault, seed, f"{acc:.6f}"])
                        done += 1
                    f.flush()
                    eta = (time.time() - t0) / done * (n_points - done)
                    log(f"    [{done:5d}/{n_points}] {policy:9s} {100 * frac:3.0f}% SLC  sigma {sigma:.3f} "
                        f"fault {fault:.3f}  acc {mean(accs):.4f} +- {pstdev(accs):.4f} "
                        f"(min {min(accs):.4f})  eta {eta / 60:.0f} min")
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    log("[7/7] summary")
    log(f"  FP32 original {baseline['fp32_acc']:.4f} | INT8 original {baseline['int8_acc']:.4f} | "
        f"SVD digital {digital:.4f}")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["policy"], r["slc_frac"], r["sigma"], r["fault_rate"])].append(r["acc"])
    summary = []
    for (p, fr, s, fault), a in groups.items():
        a_sorted = sorted(a)
        summary.append({"policy": p, "slc_frac": fr, "sigma": s, "fault_rate": fault, "acc_mean": mean(a),
                        "acc_std": pstdev(a), "acc_p5": a_sorted[int(0.05 * (len(a) - 1))],
                        "acc_min": a_sorted[0], "n": len(a)})
    with open(out / "sweep_summary.csv", "w", newline="") as f:
        dw = csv.DictWriter(f, fieldnames=list(summary[0]))
        dw.writeheader()
        dw.writerows(summary)
    print_tables(summary)
    plot(summary, digital, baseline["int8_acc"], out / "sweep.png", key)


log(f"device: {DEVICE} {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else ''}")
log(f"torch {torch.__version__}")
log(f"experiment: {EXPERIMENT}")
log(f"config: {CONFIG}")
for key in CONFIG["models"]:
    run_model(key)

log("zipping results")
shutil.make_archive(str(CONFIG["out"]), "zip", CONFIG["out"])
log(f"done. download {CONFIG['out']}.zip from the Output panel")
