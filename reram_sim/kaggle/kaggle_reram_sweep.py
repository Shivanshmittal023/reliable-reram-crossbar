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
# Results: /kaggle/working/reram_results/<model>/ and /kaggle/working/reram_results.zip
#
# Per model: load the SVD factors, then for each tiering policy (all-SLC, all-MLC,
# gradient top-20%, magnitude top-20%, random 20%), each sigma (programming noise as a
# fraction of G_max), each fault rate and each chip seed: program every crossbar with
# simulated ReRAM cells and evaluate SST-2 validation accuracy.
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
CONFIG = {
    "models": ["distilbert-sst2", "bert-base-sst2"],
    "policies": ["all_slc", "all_mlc", "gradient", "magnitude", "random"],
    "slc_frac": 0.2,
    "sigmas": [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1],   # sigma / G_max
    "fault_rates": [0.0],      # e.g. [0.0, 0.001, 0.01, 0.05, 0.1] for a stuck-at sweep
    "sa1_frac": 0.5,           # share of stuck cells stuck at G_max
    "seeds": 5,                # simulated chips per point
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


def plot(summary, digital, path, key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sweep_sigma = len(CONFIG["sigmas"]) > 1
    x_key = "sigma" if sweep_sigma else "fault_rate"
    fig, ax = plt.subplots(figsize=(6, 4))
    by_policy = defaultdict(list)
    for r in summary:
        if sweep_sigma and r["fault_rate"] != CONFIG["fault_rates"][0]:
            continue
        by_policy[r["policy"]].append(r)
    for policy, rows in by_policy.items():
        rows.sort(key=lambda r: r[x_key])
        ax.errorbar([100 * r[x_key] for r in rows], [r["acc_mean"] for r in rows],
                    yerr=[r["acc_std"] for r in rows], marker="o", capsize=3, label=policy)
    ax.axhline(digital, color="gray", ls="--", lw=1, label="digital SVD model")
    ax.set_xlabel("sigma / G_max (%)" if sweep_sigma else "stuck-at fault rate (%)")
    ax.set_ylabel("SST-2 accuracy")
    ax.set_title(key)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()
    log(f"  plot -> {path}")


# ---------------------------------------------------------------- sweep
def run_model(key):
    out = CONFIG["out"] / key
    out.mkdir(parents=True, exist_ok=True)
    log(f"========== {key} ==========")

    log("[1/5] locating SVD-tiering outputs")
    src = find_svd_outputs(key)
    log(f"  {src}")

    log("[2/5] loading SVD model and SST-2 validation")
    digital_model, tok, layers = load_svd_model(key, src)
    importance = torch.load(src / "importance.pt", map_location="cpu")
    batches = sst2_batches(tok)
    log(f"  {len(layers)} factorised layers, {sum(len(b['labels']) for b in batches)} sentences")

    log("[3/5] digital reference")
    digital = evaluate(digital_model.to(DEVICE), batches)
    digital_model.cpu()
    log(f"  digital SVD model acc {digital:.4f}")

    base = dict(w_bits=CONFIG["w_bits"], in_bits=CONFIG["in_bits"], adc_bits=CONFIG["adc_bits"],
                tile_rows=CONFIG["tile_rows"], sa1_frac=CONFIG["sa1_frac"])
    tier_base = {"SLC": CrossbarConfig(bits_per_cell=CONFIG["slc_bits"], **base),
                 "MLC": CrossbarConfig(bits_per_cell=CONFIG["mlc_bits"], **base)}
    with open(out / "config.json", "w") as f:
        json.dump({"config": {k: str(v) for k, v in CONFIG.items()}, "source": str(src),
                   "crossbar": {t: asdict(c) for t, c in tier_base.items()}, "digital_acc": digital}, f, indent=2)

    n_points = len(CONFIG["policies"]) * len(CONFIG["sigmas"]) * len(CONFIG["fault_rates"]) * CONFIG["seeds"]
    log(f"[4/5] sweep: {n_points} evaluations")
    rows, done = [], 0
    with open(out / "noise_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "sigma", "fault_rate", "seed", "acc"])
        for policy in CONFIG["policies"]:
            model = copy.deepcopy(digital_model)
            masks = tier_masks(layers, importance, policy, CONFIG["slc_frac"])
            to_reram(model, layers, masks)
            model.to(DEVICE)
            n_slc = sum(int(m.sum()) for m in masks.values())
            log(f"  policy {policy}: {n_slc}/{sum(m.numel() for m in masks.values())} components in SLC")
            for fault in CONFIG["fault_rates"]:
                for sigma in CONFIG["sigmas"]:
                    accs = []
                    for seed in range(CONFIG["seeds"]):
                        program_model(model, {t: replace(c, sigma=sigma, fault_rate=fault)
                                              for t, c in tier_base.items()}, seed)
                        acc = evaluate(model, batches)
                        accs.append(acc)
                        rows.append({"policy": policy, "sigma": sigma, "fault_rate": fault, "seed": seed, "acc": acc})
                        w.writerow([policy, sigma, fault, seed, f"{acc:.6f}"])
                        done += 1
                    f.flush()
                    log(f"    [{done:4d}/{n_points}] {policy:9s} sigma {sigma:.3f} fault {fault:.3f}  "
                        f"acc {mean(accs):.4f} +- {pstdev(accs):.4f}")
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    log("[5/5] summary")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["policy"], r["sigma"], r["fault_rate"])].append(r["acc"])
    summary = [{"policy": p, "sigma": s, "fault_rate": fr, "acc_mean": mean(a), "acc_std": pstdev(a), "n": len(a)}
               for (p, s, fr), a in groups.items()]
    with open(out / "noise_sweep_summary.csv", "w", newline="") as f:
        dw = csv.DictWriter(f, fieldnames=list(summary[0]))
        dw.writeheader()
        dw.writerows(summary)
    # accuracy table: rows = policy, columns = sigma (first fault rate)
    sig = CONFIG["sigmas"]
    log("  policy     " + " ".join(f"{s:>7.3f}" for s in sig))
    for p in CONFIG["policies"]:
        accs = {r["sigma"]: r["acc_mean"] for r in summary if r["policy"] == p and r["fault_rate"] == CONFIG["fault_rates"][0]}
        log(f"  {p:10s} " + " ".join(f"{accs[s]:7.4f}" for s in sig))
    plot(summary, digital, out / "noise_sweep.png", key)


log(f"device: {DEVICE} {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else ''}")
log(f"torch {torch.__version__}")
log(f"config: {CONFIG}")
for key in CONFIG["models"]:
    run_model(key)

log("zipping results")
shutil.make_archive(str(CONFIG["out"]), "zip", CONFIG["out"])
log(f"done. download {CONFIG['out']}.zip from the Output panel")
