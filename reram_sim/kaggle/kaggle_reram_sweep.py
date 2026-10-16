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
# Per model: load the SVD factors, then run each experiment in CONFIG["experiments"]
# (A: HyFlexPIM noise vs SLC rate, B: cell noise vs SLC rate, C: cell-noise sweep,
# D: stuck-at fault sweep). Every point = one tiering policy x SLC rate x sigma x
# fault rate x chip seed: program all crossbars, evaluate SST-2 validation accuracy.
# Outputs per experiment: <name>.csv (every chip), <name>_summary.csv, <name>.png.
# Runtime: 505 evaluations per model (~25 min DistilBERT, ~50 min BERT on a T4, estimated);
# drop experiments or lower "seeds" to shorten.
#
# Cell model (same as reram_sim/reram_sim/crossbar.py, which has the local tests):
#   8-bit signed weights per row, differential pair (G+, G-), bit-sliced into
#   base-2^b digits (SLC b=1, MLC b=2). Digit d -> g = g_min + d*delta, g_max = 1.
#   Noise g += sigma*N(0,1) ("cell"), or w*(1+sigma*N(0,1)) per weight ("multiplicative",
#   HyFlexPIM eq. 5); stuck-at faults set g to G_max or G_min.
#   Ideal readout merges the slices; with adc_bits each slice and tile_rows tile is
#   quantised by an ideally calibrated ADC before shift-and-add.
#   Runs in fp32 on purpose: fp16 overflows the integer-domain partial sums.
import copy
import csv
import json
import math
import shutil
import time
import traceback
import warnings
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, pstdev

import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

# ---------------------------------------------------------------- config
# Values follow HyFlexPIM (Song et al., ISCA 2025):
#   * noise: W~ = W (1 + eta), eta ~ N(0, 0.025^2), MLC weights only (SLC noise-free),
#     calibrated to a measured 2-bit MLC bit error rate of 4.04% (Fan et al.)
#   * SLC (protection) rates in Fig. 12: 0, 5, 10, 30, 40, 50, 100 %
#   * RRAM on/off 150, 64-row arrays, 1-bit DAC, 6-b (SLC) / 7-b (MLC) ADC
# In our per-cell conductance model, sigma/G_max = 0.095 gives the same 4.04% MLC
# bit error rate (SLC: ~1e-7), so the cell-noise sweep is centred on 0.095.
# HyFlexPIM has no stuck-at faults; the fault sweep uses the project plan's 0.1-20 %.
SIGMA_CAL = 0.095
CONFIG = {
    "models": ["distilbert-sst2", "bert-base-sst2"],
    "experiments": [
        # A. HyFlexPIM reproduction: accuracy vs SLC rate under their own noise model.
        {"name": "A_hyflexpim_slc_rate", "x": "slc_frac", "noise_model": "multiplicative",
         "noisy_tiers": ["MLC"], "sigmas": [0.025],
         "policies": ["all_slc", "all_mlc", "gradient", "magnitude", "random"],
         "slc_fracs": [0.05, 0.1, 0.2, 0.3, 0.5], "fault_rates": [0.0]},
        # B. Same, with per-cell conductance noise at the BER-matched sigma (both tiers noisy).
        {"name": "B_cell_slc_rate", "x": "slc_frac", "noise_model": "cell",
         "noisy_tiers": ["SLC", "MLC"], "sigmas": [SIGMA_CAL],
         "policies": ["all_slc", "all_mlc", "gradient", "magnitude", "random"],
         "slc_fracs": [0.05, 0.1, 0.2, 0.3, 0.5], "fault_rates": [0.0]},
        # C. Noise sweep around the calibrated point at a 20 % SLC rate.
        {"name": "C_cell_noise_sweep", "x": "sigma", "noise_model": "cell",
         "noisy_tiers": ["SLC", "MLC"], "sigmas": [0.0, 0.02, 0.04, 0.06, 0.08, SIGMA_CAL, 0.12],
         "policies": ["all_slc", "all_mlc", "gradient", "magnitude", "random"],
         "slc_fracs": [0.2], "fault_rates": [0.0]},
        # D. Stuck-at fault sweep (no programming noise), project plan Table 2 range.
        {"name": "D_fault_sweep", "x": "fault_rate", "noise_model": "cell",
         "noisy_tiers": ["SLC", "MLC"], "sigmas": [0.0],
         "policies": ["all_slc", "all_mlc", "gradient", "random"],
         "slc_fracs": [0.2], "fault_rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]},
    ],
    "sa1_frac": 0.5,           # share of stuck cells stuck at G_max
    "seeds": 5,                # simulated chips per point
    "slc_bits": 1,
    "mlc_bits": 2,
    "w_bits": 8,
    "in_bits": 8,              # DAC precision; None = ideal inputs
    "adc_bits": None,          # e.g. 7; None = ideal readout (much faster)
    "tile_rows": 64,           # HyFlexPIM arrays are 64 rows x 128 columns
    "on_off": 150.0,
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


# Keep the cell output readable: only our own log lines, no HF progress bars or warnings.
transformers.logging.set_verbosity_error()
datasets.disable_progress_bars()
warnings.filterwarnings("ignore")
CONFIG["out"].mkdir(parents=True, exist_ok=True)
LOG_FILE = open(CONFIG["out"] / "run.log", "a", encoding="utf-8")


def log(msg):
    """Print with elapsed time and append to <out>/run.log (survives a truncated notebook view)."""
    line = f"[{time.time() - T_START:7.1f}s] {msg}"
    print(line, flush=True)
    LOG_FILE.write(line + "\n")
    LOG_FILE.flush()


def fmt_time(sec):
    return f"{sec / 60:.1f} min" if sec >= 60 else f"{sec:.0f} s"


# ---------------------------------------------------------------- crossbar
@dataclass(frozen=True)
class CrossbarConfig:
    bits_per_cell: int = 1
    sigma: float = 0.0
    noise_model: str = "cell"      # "cell" | "multiplicative" (HyFlexPIM eq. 5)
    fault_rate: float = 0.0
    sa1_frac: float = 0.5
    on_off: float = 150.0
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
        multiplicative = cfg.noise_model == "multiplicative"
        cell_cfg = replace(cfg, sigma=0.0) if multiplicative else cfg
        digits = []
        for s in range(cfg.n_slices):
            g = {k: self._cells(torch.div(m, L ** s, rounding_mode="floor") % L, cell_cfg, g_min, delta, gen)
                 for k, m in mag.items()}
            digits.append((g["pos"] - g["neg"]) / delta)
        if multiplicative and cfg.sigma > 0:
            eta = 1 + cfg.sigma * torch.randn(q.shape, generator=gen, device=q.device)
            digits = [d * eta for d in digits]
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


def plot(summary, digital, exp, path, key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = exp["x"]
    fig, ax = plt.subplots(figsize=(6, 4))
    by_policy = defaultdict(list)
    for r in summary:
        by_policy[r["policy"]].append(r)
    for i, (policy, rows) in enumerate(by_policy.items()):
        rows.sort(key=lambda r: r[x])
        if len({r[x] for r in rows}) == 1:          # all_slc / all_mlc in an SLC-rate sweep
            ax.axhline(rows[0]["acc_mean"], ls=":", lw=1.2, color=f"C{i}", label=policy)
            continue
        ax.errorbar([100 * r[x] for r in rows], [r["acc_mean"] for r in rows],
                    yerr=[r["acc_std"] for r in rows], marker="o", capsize=3, color=f"C{i}", label=policy)
    ax.axhline(digital, color="gray", ls="--", lw=1, label="digital SVD model")
    ax.set_xlabel({"sigma": "noise sigma (%)", "slc_frac": "SLC rate (%)",
                   "fault_rate": "stuck-at fault rate (%)"}[x])
    ax.set_ylabel("SST-2 accuracy")
    ax.set_title(f"{key}: {exp['name']}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()
    log(f"    plot -> {path}")


def grid(exp):
    """(policy, slc_frac) pairs; all_slc / all_mlc ignore slc_frac, so they run once."""
    for policy in exp["policies"]:
        for frac in ([None] if policy in ("all_slc", "all_mlc") else exp["slc_fracs"]):
            yield policy, frac


def run_experiment(exp, key, digital_model, layers, importance, batches, digital, out):
    log(f"  ---- experiment {exp['name']} (x = {exp['x']}, noise_model = {exp['noise_model']}, "
        f"noisy tiers = {exp['noisy_tiers']})")
    base = dict(w_bits=CONFIG["w_bits"], in_bits=CONFIG["in_bits"], adc_bits=CONFIG["adc_bits"],
                tile_rows=CONFIG["tile_rows"], on_off=CONFIG["on_off"], sa1_frac=CONFIG["sa1_frac"],
                noise_model=exp["noise_model"])
    tier_base = {"SLC": CrossbarConfig(bits_per_cell=CONFIG["slc_bits"], **base),
                 "MLC": CrossbarConfig(bits_per_cell=CONFIG["mlc_bits"], **base)}
    pairs = list(grid(exp))
    n_points = len(pairs) * len(exp["sigmas"]) * len(exp["fault_rates"]) * CONFIG["seeds"]
    log(f"    {len(pairs)} policy/SLC-rate pairs x {len(exp['sigmas'])} sigmas x {len(exp['fault_rates'])} "
        f"fault rates x {CONFIG['seeds']} chips = {n_points} evaluations")
    rows, done, t_exp = [], 0, time.time()
    with open(out / f"{exp['name']}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "slc_frac", "sigma", "fault_rate", "seed", "acc"])
        for i_pair, (policy, frac) in enumerate(pairs, 1):
            t_build = time.time()
            model = copy.deepcopy(digital_model)
            masks = tier_masks(layers, importance, policy, frac)
            to_reram(model, layers, masks)
            model.to(DEVICE)
            real_frac = sum(int(m.sum()) for m in masks.values()) / sum(m.numel() for m in masks.values())
            mem = f", GPU mem {torch.cuda.memory_allocated() / 2**30:.2f} GB" if DEVICE.type == "cuda" else ""
            log(f"    pair {i_pair}/{len(pairs)}: {policy}, {real_frac:.1%} of components in SLC "
                f"(built in {fmt_time(time.time() - t_build)}{mem})")
            for fault in exp["fault_rates"]:
                for sigma in exp["sigmas"]:
                    cfgs = {t: replace(c, sigma=sigma if t in exp["noisy_tiers"] else 0.0, fault_rate=fault)
                            for t, c in tier_base.items()}
                    accs = []
                    for seed in range(CONFIG["seeds"]):
                        program_model(model, cfgs, seed)
                        acc = evaluate(model, batches)
                        accs.append(acc)
                        rows.append({"policy": policy, "slc_frac": real_frac, "sigma": sigma,
                                     "fault_rate": fault, "seed": seed, "acc": acc})
                        w.writerow([policy, f"{real_frac:.4f}", sigma, fault, seed, f"{acc:.6f}"])
                        done += 1
                    f.flush()
                    elapsed = time.time() - t_exp
                    eta = elapsed / done * (n_points - done)
                    log(f"    [{done:4d}/{n_points}] {policy:9s} slc {real_frac:5.1%} sigma {sigma:.3f} "
                        f"fault {fault:.3f}  acc {mean(accs):.4f} +- {pstdev(accs):.4f} "
                        f"(min {min(accs):.4f}; chips {' '.join(f'{a:.3f}' for a in accs)})  ETA {fmt_time(eta)}")
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    groups = defaultdict(list)
    for r in rows:
        groups[(r["policy"], r["slc_frac"], r["sigma"], r["fault_rate"])].append(r["acc"])
    summary = [{"policy": p, "slc_frac": sf, "sigma": s, "fault_rate": fr, "acc_mean": mean(a),
                "acc_std": pstdev(a), "acc_min": min(a), "n": len(a)}
               for (p, sf, s, fr), a in groups.items()]
    with open(out / f"{exp['name']}_summary.csv", "w", newline="") as f:
        dw = csv.DictWriter(f, fieldnames=list(summary[0]))
        dw.writeheader()
        dw.writerows(summary)
    log_table(summary, exp, digital)
    log(f"    experiment {exp['name']} done in {fmt_time(time.time() - t_exp)}")
    plot(summary, digital, exp, out / f"{exp['name']}.png", key)


def log_table(summary, exp, digital):
    """Mean accuracy table: one row per policy, one column per x value."""
    x = exp["x"]
    xs = sorted({r[x] for r in summary if r["policy"] not in ("all_slc", "all_mlc")} or {r[x] for r in summary})
    log(f"    summary ({x}; mean acc over {CONFIG['seeds']} chips; digital {digital:.4f}):")
    log("      " + f"{'policy':10s}" + "".join(f"{v:>9.3f}" for v in xs))
    for policy in exp["policies"]:
        rows = {round(r[x], 4): r["acc_mean"] for r in summary if r["policy"] == policy}
        if policy in ("all_slc", "all_mlc") and x == "slc_frac":
            log("      " + f"{policy:10s}" + f"{next(iter(rows.values())):>9.4f}  (all x)")
            continue
        log("      " + f"{policy:10s}" + "".join(f"{rows[round(v, 4)]:>9.4f}" if round(v, 4) in rows
                                                  else f"{'-':>9s}" for v in xs))


def run_model(key):
    out = CONFIG["out"] / key
    out.mkdir(parents=True, exist_ok=True)
    log(f"========== {key} ==========")

    log("[1/4] locating SVD-tiering outputs")
    src = find_svd_outputs(key)
    log(f"  {src}")

    log("[2/4] loading SVD model and SST-2 validation")
    digital_model, tok, layers = load_svd_model(key, src)
    importance = torch.load(src / "importance.pt", map_location="cpu")
    batches = sst2_batches(tok)
    log(f"  {len(layers)} factorised layers, {sum(len(b['labels']) for b in batches)} sentences")

    log("[3/4] digital reference")
    digital = evaluate(digital_model.to(DEVICE), batches)
    digital_model.cpu()
    log(f"  digital SVD model acc {digital:.4f}")
    with open(out / "config.json", "w") as f:
        json.dump({"config": {k: str(v) for k, v in CONFIG.items()}, "source": str(src),
                   "digital_acc": digital}, f, indent=2)

    log(f"[4/4] {len(CONFIG['experiments'])} experiments")
    for exp in CONFIG["experiments"]:
        try:
            run_experiment(exp, key, digital_model, layers, importance, batches, digital, out)
        except Exception:
            # Log and move on, so one failure does not lose the other experiments.
            log(f"  !! experiment {exp['name']} failed:\n{traceback.format_exc()}")
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()


log(f"device: {DEVICE} {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else ''}")
log(f"torch {torch.__version__}")
log("config:\n" + json.dumps(CONFIG, indent=1, default=str))
for key in CONFIG["models"]:
    try:
        run_model(key)
    except Exception:
        log(f"!! model {key} failed:\n{traceback.format_exc()}")

log("output files:")
for f in sorted(CONFIG["out"].rglob("*")):
    if f.is_file():
        log(f"  {f.relative_to(CONFIG['out'])}  ({f.stat().st_size / 1024:.1f} KB)")
log("zipping results")
LOG_FILE.flush()
shutil.make_archive(str(CONFIG["out"]), "zip", CONFIG["out"])
log(f"done in {fmt_time(time.time() - T_START)}. download {CONFIG['out']}.zip from the Output panel")
