"""SST-2 accuracy of the SVD-tiered transformer on simulated ReRAM, swept over noise.

For every tiering policy, sigma (programming noise, fraction of G_max), fault rate
and chip seed: program all crossbars, evaluate on SST-2 validation, log one row.

  python scripts/noise_sweep.py --model distilbert-sst2
  python scripts/noise_sweep.py --model bert-base-sst2 --sigmas 0,0.02,0.05 --seeds 2 --eval-n 256
  python scripts/noise_sweep.py --model distilbert-sst2 --fault-rates 0,0.01,0.05 --sigmas 0

Writes results/<model>/{noise_sweep.csv, noise_sweep_summary.csv, config.json, noise_sweep.png}.
"""
import argparse
import copy
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, pstdev

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reram_sim import CrossbarConfig, program_model  # noqa: E402
from reram_sim.hf import (DEFAULT_OUTPUTS, MODELS, POLICIES, evaluate, load_svd_model,  # noqa: E402
                          sst2_loader, tier_masks, to_reram)

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def floats(s: str) -> list[float]:
    return [float(v) for v in s.split(",")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS, help="svd_tiering outputs root")
    p.add_argument("--policies", default=",".join(POLICIES))
    p.add_argument("--slc-frac", type=float, default=0.2)
    p.add_argument("--sigmas", type=floats, default=floats("0,0.01,0.02,0.03,0.05,0.075,0.1"))
    p.add_argument("--fault-rates", type=floats, default=[0.0])
    p.add_argument("--sa1-frac", type=float, default=0.5)
    p.add_argument("--seeds", type=int, default=3, help="simulated chips per point")
    p.add_argument("--slc-bits", type=int, default=1)
    p.add_argument("--mlc-bits", type=int, default=2)
    p.add_argument("--w-bits", type=int, default=8)
    p.add_argument("--in-bits", type=int, default=8)
    p.add_argument("--adc-bits", type=int, default=None)
    p.add_argument("--tile-rows", type=int, default=128)
    p.add_argument("--eval-n", type=int, default=None, help="SST-2 validation samples (default all 872)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results", type=Path, default=ROOT / "results")
    return p.parse_args()


def plot(summary: list[dict], path: Path, digital: float, fault_rates: list[float]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib missing, skipping plot")
        return
    x_key = "sigma" if len({r["sigma"] for r in summary}) > 1 else "fault_rate"
    fig, ax = plt.subplots(figsize=(6, 4))
    by_policy = defaultdict(list)
    for r in summary:
        if x_key == "sigma" and r["fault_rate"] != fault_rates[0]:
            continue
        by_policy[r["policy"]].append(r)
    for policy, rows in by_policy.items():
        rows.sort(key=lambda r: r[x_key])
        xs = [100 * r[x_key] for r in rows]
        ax.errorbar(xs, [r["acc_mean"] for r in rows], yerr=[r["acc_std"] for r in rows],
                    marker="o", capsize=3, label=policy)
    ax.axhline(digital, color="gray", ls="--", lw=1, label="digital SVD model")
    ax.set_xlabel("σ / G_max (%)" if x_key == "sigma" else "stuck-at fault rate (%)")
    ax.set_ylabel("SST-2 accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    log(f"plot -> {path}")


def main():
    args = parse_args()
    out = args.results / args.model
    out.mkdir(parents=True, exist_ok=True)
    policies = args.policies.split(",")
    base = dict(w_bits=args.w_bits, in_bits=args.in_bits, adc_bits=args.adc_bits,
                tile_rows=args.tile_rows, sa1_frac=args.sa1_frac)
    tier_base = {"SLC": CrossbarConfig(bits_per_cell=args.slc_bits, **base),
                 "MLC": CrossbarConfig(bits_per_cell=args.mlc_bits, **base)}
    n_points = len(policies) * len(args.sigmas) * len(args.fault_rates) * args.seeds
    log(f"device {args.device}; {len(policies)} policies x {len(args.sigmas)} sigmas x "
        f"{len(args.fault_rates)} fault rates x {args.seeds} seeds = {n_points} evaluations")

    log(f"[1/4] loading {args.model} SVD model from {args.outputs / args.model}")
    src = args.outputs / args.model
    digital_model, tok, layers, _ = load_svd_model(args.model, src)
    importance = torch.load(src / "importance.pt")
    digital_model.to(args.device)
    eval_loader = sst2_loader(tok, "validation", args.eval_n, args.batch_size, shuffle=False)
    eval_batches = [{k: v.to(args.device) for k, v in b.items()} for b in eval_loader]
    log(f"  {len(layers)} factorised layers, {sum(len(b['labels']) for b in eval_batches)} eval sentences")

    log("[2/4] digital reference (no ReRAM)")
    digital = evaluate(digital_model, eval_batches)
    log(f"  digital SVD model acc {digital:.4f}")

    with open(out / "config.json", "w") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()},
                   "crossbar": {t: asdict(c) for t, c in tier_base.items()},
                   "digital_acc": digital}, f, indent=2)

    log("[3/4] sweep")
    rows, done = [], 0
    csv_path = out / "noise_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "sigma", "fault_rate", "seed", "acc"])
        for policy in policies:
            # Each policy starts from the same SVD factors; only the SLC/MLC split differs.
            model = copy.deepcopy(digital_model)
            masks = tier_masks(layers, importance, policy, args.slc_frac, seed=0)
            to_reram(model, layers, masks)
            n_slc = sum(int(m.sum()) for m in masks.values())
            n_all = sum(m.numel() for m in masks.values())
            log(f"  policy {policy}: {n_slc}/{n_all} components in SLC")
            for fault in args.fault_rates:
                for sigma in args.sigmas:
                    accs = []
                    for seed in range(args.seeds):
                        cfgs = {t: replace(c, sigma=sigma, fault_rate=fault) for t, c in tier_base.items()}
                        program_model(model, cfgs, seed)
                        acc = evaluate(model, eval_batches)
                        accs.append(acc)
                        rows.append({"policy": policy, "sigma": sigma, "fault_rate": fault, "seed": seed, "acc": acc})
                        w.writerow([policy, sigma, fault, seed, f"{acc:.6f}"])
                        f.flush()
                        done += 1
                    log(f"    [{done:4d}/{n_points}] {policy:9s} sigma {sigma:.3f} fault {fault:.3f}  "
                        f"acc {mean(accs):.4f} ± {pstdev(accs):.4f}")
            del model

    log("[4/4] summary")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["policy"], r["sigma"], r["fault_rate"])].append(r["acc"])
    summary = [{"policy": p, "sigma": s, "fault_rate": fr, "acc_mean": mean(a), "acc_std": pstdev(a), "n": len(a)}
               for (p, s, fr), a in groups.items()]
    with open(out / "noise_sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for r in summary:
        log(f"  {r['policy']:9s} sigma {r['sigma']:.3f} fault {r['fault_rate']:.3f}  "
            f"{r['acc_mean']:.4f} ± {r['acc_std']:.4f}")
    plot(summary, out / "noise_sweep.png", digital, args.fault_rates)
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
