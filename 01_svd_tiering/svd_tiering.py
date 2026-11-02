!pip install -q hf_xet
# Do NOT `pip install -U datasets` here: it swaps pyarrow under the running kernel and
# breaks with "pyarrow.lib.Array size changed". Kaggle's preinstalled datasets is enough.
# Kaggle GPU version of the SVD + SLC/MLC tiering pipeline (single notebook cell).
#
# Setup: Settings -> Accelerator: GPU T4 x2 (or P100), Internet: On.
# Paste this whole file into ONE notebook cell and run it.
# Results: /kaggle/working/outputs/<model>/ and /kaggle/working/outputs.zip
#
# Pipeline (HyFlexPIM, Song et al. ISCA 2025), per model:
#   1. load SST-2 fine-tuned transformer, evaluate baseline
#   2. factorise every encoder Linear as W ~= U · (ΣVᵀ), k = floor(mn/(m+n))
#   3. fine-tune (8 epochs of SST-2, see CONFIG) to recover truncation loss
#   4. I_r = |dL/dσ_r| on a calibration set
#   5. top-p components -> SLC, rest -> MLC; write per-layer files
#   6. verify SLC + MLC sum back to the factorised weight, write summary
import csv
import json
import math
import re
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

# ---------------------------------------------------------------- config
CONFIG = {
    "models": ["distilbert-sst2", "bert-base-sst2"],
    "rank_mode": "param",      # "param": k=floor(mn/(m+n)) ; "full": k=min(m,n), lossless
    "slc_frac": 0.2,           # fraction of components placed in SLC
    "scope": "layer",          # "layer": top-p per layer ; "global": top-p across all layers
    "metric": "gradient",      # "gradient": |dL/dσ| ; "magnitude": σ
    "finetune_epochs": 8,
    "lr": 2e-5,
    "train_n": None,           # None = full SST-2 train (67,349 sentences)
    "calib_n": 2048,
    "batch_size": 32,
    "amp": True,               # fp16 autocast for fine-tune / eval (importance stays fp32)
    "log_every": 200,
    "seed": 0,
    "out": Path("/kaggle/working/outputs") if Path("/kaggle/working").exists() else Path("outputs"),
}

MODELS = {
    "distilbert-sst2": {"hf_id": "distilbert-base-uncased-finetuned-sst-2-english",
                        "target": r"^distilbert\.transformer\.layer\.\d+\."},
    "bert-base-sst2": {"hf_id": "textattack/bert-base-uncased-SST-2",
                       "target": r"^bert\.encoder\.layer\.\d+\."},
}
TIERS = ("SLC", "MLC")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = CONFIG["amp"] and DEVICE.type == "cuda"
T_START = time.time()


def log(msg):
    print(f"[{time.time() - T_START:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- SVD layer
class SVDLinear(nn.Module):
    """W ~= U · diag(S) · Vh. Crossbars: x -> [ΣVᵀ (k x in)] -> [U (out x k)] -> y.
    S is its own parameter so I_r = |dL/dσ_r| is read from S.grad."""

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

    def forward(self, x):
        return F.linear(F.linear(x, self.Vh) * self.S, self.U, self.bias)

    @torch.no_grad()
    def crossbar_factors(self):
        """(U, ΣVᵀ) on CPU in fp32: the two matrices programmed into crossbars."""
        return self.U.detach().float().cpu().clone(), (self.S[:, None] * self.Vh).detach().float().cpu().clone()

    @torch.no_grad()
    def effective_weight(self):
        U, SVh = self.crossbar_factors()
        return U @ SVh


def factorize(model, key, rank_mode):
    pattern = re.compile(MODELS[key]["target"])
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and pattern.match(n)]
    layers = {}
    for name, linear in targets:
        m, n = linear.out_features, linear.in_features
        k = (m * n) // (m + n) if rank_mode == "param" else min(m, n)
        svd = SVDLinear.from_linear(linear, k)
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, svd)
        layers[name] = svd
    shapes = sorted({(l.out_features, l.in_features, l.rank) for l in layers.values()})
    for m, n, k in shapes:
        log(f"  W {m}x{n} -> U {m}x{k} + ΣVᵀ {k}x{n}  (rank {k} of {min(m, n)})")
    return layers


# ---------------------------------------------------------------- data
def sst2_loader(tok, split, n, batch_size, shuffle, seed=0, max_len=128):
    ds = load_dataset("nyu-mll/glue", "sst2", split=split)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    ds = ds.map(lambda b: tok(b["sentence"], truncation=True, max_length=max_len),
                batched=True, remove_columns=["sentence", "idx"]).rename_column("label", "labels")
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g,
                        collate_fn=DataCollatorWithPadding(tok), num_workers=2, pin_memory=True)
    log(f"  SST-2 {split}: {len(ds)} sentences, {len(loader)} batches of {batch_size}")
    return loader


def to_dev(batch):
    return {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}


def autocast():
    return torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = to_dev(batch)
        with autocast():
            logits = model(**batch).logits
        correct += (logits.argmax(-1) == batch["labels"]).sum().item()
        total += batch["labels"].numel()
    return correct / total


# ---------------------------------------------------------------- fine-tune + importance
def finetune(model, loader, epochs, lr):
    steps = epochs * len(loader)
    if steps <= 0:
        log("  fine-tune skipped (finetune_epochs=0)")
        return
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, total_iters=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    step, running = 0, 0.0
    for epoch in range(epochs):
        for batch in loader:
            with autocast():
                loss = model(**to_dev(batch)).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            running += loss.item()
            if step % CONFIG["log_every"] == 0 or step == steps:
                n = CONFIG["log_every"] if step % CONFIG["log_every"] == 0 else step % CONFIG["log_every"]
                log(f"  epoch {epoch + 1}/{epochs} step {step}/{steps}  avg loss {running / n:.4f}  "
                    f"lr {sched.get_last_lr()[0]:.2e}")
                running = 0.0


def gradient_importance(model, layers, loader):
    """I_r = |dL/dσ_r|, gradient of loss summed over calibration set (fp32, eval mode, no dropout)."""
    model.eval()
    model.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        model(**to_dev(batch)).loss.backward()
        if (i + 1) % 20 == 0 or i + 1 == len(loader):
            log(f"  calibration batch {i + 1}/{len(loader)}")
    scores = {n: l.S.grad.detach().abs().float().cpu().clone() for n, l in layers.items()}
    model.zero_grad(set_to_none=True)
    return scores


def magnitude_importance(layers):
    return {n: l.S.detach().abs().float().cpu().clone() for n, l in layers.items()}


# ---------------------------------------------------------------- tiering
def select_slc(scores, frac, scope):
    if scope == "layer":
        masks = {}
        for name, s in scores.items():
            mask = torch.zeros_like(s, dtype=torch.bool)
            mask[torch.topk(s, math.ceil(frac * s.numel())).indices] = True
            masks[name] = mask
        return masks
    flat = torch.cat(list(scores.values()))
    thr = torch.topk(flat, math.ceil(frac * flat.numel())).values.min()
    return {name: s >= thr for name, s in scores.items()}


def save_tiers(out_dir, layers, scores, masks):
    for tier in TIERS:
        (out_dir / tier).mkdir(parents=True, exist_ok=True)
    tier_map = {}
    for name, layer in layers.items():
        U, SVh = layer.crossbar_factors()
        entry = {"out_features": layer.out_features, "in_features": layer.in_features, "rank": layer.rank}
        for tier, mask in (("SLC", masks[name]), ("MLC", ~masks[name])):
            idx = mask.nonzero().flatten()
            torch.save({"indices": idx, "U": U[:, idx].contiguous(), "SVh": SVh[idx].contiguous(),
                        "importance": scores[name][idx]}, out_dir / tier / f"{name}.pt")
            entry[tier] = {"count": idx.numel(), "indices": idx.tolist(),
                           "U_shape": [layer.out_features, idx.numel()],
                           "SVh_shape": [idx.numel(), layer.in_features]}
        tier_map[name] = entry
        log(f"  {name:55s} rank {layer.rank:4d}  SLC {entry['SLC']['count']:4d}  MLC {entry['MLC']['count']:4d}")
    with open(out_dir / "tier_map.json", "w") as f:
        json.dump(tier_map, f, indent=1)
    return tier_map


def reconstruct_weight(out_dir, name):
    """W_eff = U_slc @ SVh_slc + U_mlc @ SVh_mlc."""
    parts = [torch.load(out_dir / t / f"{name}.pt") for t in TIERS]
    return sum(p["U"] @ p["SVh"] for p in parts)


# ---------------------------------------------------------------- run
def run(key):
    cfg = CONFIG
    torch.manual_seed(cfg["seed"])
    out_dir = cfg["out"] / key
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hf_id = MODELS[key]["hf_id"]
    log(f"========== {key} ({hf_id}) ==========")

    log("[1/6] loading model, tokenizer, SST-2")
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id).to(DEVICE)
    log(f"  params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    eval_loader = sst2_loader(tok, "validation", None, 128, shuffle=False)
    train_loader = sst2_loader(tok, "train", cfg["train_n"], cfg["batch_size"], shuffle=True, seed=cfg["seed"])
    calib_loader = sst2_loader(tok, "train", cfg["calib_n"], cfg["batch_size"], shuffle=False, seed=cfg["seed"] + 1)
    acc = {"baseline": evaluate(model, eval_loader)}
    log(f"  baseline acc {acc['baseline']:.4f}")

    log(f"[2/6] SVD factorisation (rank_mode={cfg['rank_mode']})")
    layers = factorize(model, key, cfg["rank_mode"])
    model.to(DEVICE)
    acc["svd_truncated"] = evaluate(model, eval_loader)
    log(f"  factorised {len(layers)} layers; acc {acc['svd_truncated']:.4f}")

    log(f"[3/6] recovery fine-tune ({cfg['finetune_epochs']} epoch, lr {cfg['lr']}, amp={USE_AMP})")
    finetune(model, train_loader, cfg["finetune_epochs"], cfg["lr"])
    acc["svd_finetuned"] = evaluate(model, eval_loader)
    log(f"  after fine-tune acc {acc['svd_finetuned']:.4f}")

    log(f"[4/6] importance (metric={cfg['metric']}, calib_n={cfg['calib_n']})")
    grad_scores = gradient_importance(model, layers, calib_loader)
    mag_scores = magnitude_importance(layers)
    scores = grad_scores if cfg["metric"] == "gradient" else mag_scores
    torch.save({"gradient": grad_scores, "magnitude": mag_scores}, out_dir / "importance.pt")

    log(f"[5/6] SLC/MLC split (slc_frac={cfg['slc_frac']}, scope={cfg['scope']}) -> {out_dir}")
    masks = select_slc(scores, cfg["slc_frac"], cfg["scope"])
    tier_map = save_tiers(out_dir, layers, scores, masks)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_dir / "svd_model.pt")
    with open(out_dir / "importance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "rank_idx", "sigma", "grad_importance", "tier"])
        for name, layer in layers.items():
            sigma = layer.S.detach().float().cpu()
            for r in range(layer.rank):
                w.writerow([name, r, f"{sigma[r].item():.6g}", f"{grad_scores[name][r].item():.6g}",
                            "SLC" if masks[name][r] else "MLC"])
    log("  wrote SLC/, MLC/, tier_map.json, importance.pt, importance.csv, svd_model.pt")

    log("[6/6] verify SLC + MLC == U·diag(S)·Vh, write summary")
    max_err = max((reconstruct_weight(out_dir, n) - l.effective_weight()).abs().max().item()
                  for n, l in layers.items())
    log(f"  reconstruction max abs error {max_err:.3e}")
    n_slc = sum(e["SLC"]["count"] for e in tier_map.values())
    n_all = sum(e["rank"] for e in tier_map.values())
    slc_p = sum(e["SLC"]["count"] * (e["out_features"] + e["in_features"]) for e in tier_map.values())
    all_p = sum(e["rank"] * (e["out_features"] + e["in_features"]) for e in tier_map.values())
    summary = {
        "model": key, "hf_id": hf_id, "device": str(DEVICE), "config": {k: str(v) for k, v in cfg.items()},
        "accuracy": acc, "num_layers": len(layers),
        "components": {"SLC": n_slc, "MLC": n_all - n_slc, "slc_fraction": n_slc / n_all},
        "weights": {"SLC": slc_p, "MLC": all_p - slc_p, "slc_fraction": slc_p / all_p},
        "reconstruction_max_abs_error": max_err, "runtime_s": round(time.time() - t0, 1),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("accuracy", "components", "weights", "runtime_s")}, indent=2))
    del model, layers
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    log(f"device: {DEVICE} {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else '(no GPU: enable accelerator)'}")
    log(f"torch {torch.__version__}")
    for key in CONFIG["models"]:
        run(key)
    log("zipping outputs")
    shutil.make_archive(str(CONFIG["out"]), "zip", CONFIG["out"])
    log(f"done. download {CONFIG['out']}.zip from the Output panel")
