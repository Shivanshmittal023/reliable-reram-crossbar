"""SVD factorisation + gradient importance + SLC/MLC tier split for one model.

Pipeline (HyFlexPIM, Song et al. ISCA 2025):
  1. load SST-2 fine-tuned transformer, evaluate baseline
  2. factorise every encoder Linear as W ~= U · (ΣVᵀ)
  3. short fine-tune to recover truncation loss
  4. I_r = |dL/dσ_r| on a calibration set
  5. top-p components -> SLC, rest -> MLC; write per-layer files

Usage:
  python scripts/run_svd_tiering.py --model distilbert-sst2
  python scripts/run_svd_tiering.py --model bert-base-sst2 --slc-frac 0.1 --finetune-steps 100
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from svd_tiering import MODELS, factorize, load_model, reconstruct_weight, save_tiers, select_slc  # noqa: E402
from svd_tiering.data import evaluate, sst2_loader  # noqa: E402
from svd_tiering.importance import finetune, gradient_importance, magnitude_importance  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--rank-mode", choices=["param", "full"], default="param",
                   help="param: k=floor(mn/(m+n)) (HyFlexPIM); full: k=min(m,n), lossless")
    p.add_argument("--slc-frac", type=float, default=0.2, help="fraction of components placed in SLC")
    p.add_argument("--scope", choices=["layer", "global"], default="layer")
    p.add_argument("--metric", choices=["gradient", "magnitude"], default="gradient")
    p.add_argument("--finetune-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--train-n", type=int, default=8000, help="SST-2 train subset for fine-tuning")
    p.add_argument("--calib-n", type=int, default=512, help="SST-2 train samples for importance")
    p.add_argument("--eval-n", type=int, default=None, help="SST-2 validation samples (default all 872)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "outputs")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = args.out / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    model, tok = load_model(args.model)
    eval_loader = sst2_loader(tok, "validation", args.eval_n, 64, shuffle=False)
    train_loader = sst2_loader(tok, "train", args.train_n, args.batch_size, shuffle=True, seed=args.seed)
    calib_loader = sst2_loader(tok, "train", args.calib_n, args.batch_size, shuffle=False, seed=args.seed + 1)

    acc = {"baseline": evaluate(model, eval_loader)}
    print(f"[{args.model}] baseline acc {acc['baseline']:.4f}")

    layers = factorize(model, args.model, args.rank_mode)
    acc["svd_truncated"] = evaluate(model, eval_loader)
    print(f"[{args.model}] factorised {len(layers)} layers ({args.rank_mode}); acc {acc['svd_truncated']:.4f}")

    finetune(model, train_loader, args.finetune_steps, args.lr)
    acc["svd_finetuned"] = evaluate(model, eval_loader)
    print(f"[{args.model}] after fine-tune acc {acc['svd_finetuned']:.4f}")

    grad_scores = gradient_importance(model, layers, calib_loader)
    mag_scores = magnitude_importance(layers)
    scores = grad_scores if args.metric == "gradient" else mag_scores
    torch.save({"gradient": grad_scores, "magnitude": mag_scores}, out_dir / "importance.pt")

    masks = select_slc(scores, args.slc_frac, args.scope)
    tier_map = save_tiers(out_dir, layers, scores, masks)
    torch.save(model.state_dict(), out_dir / "svd_model.pt")

    # Per-component table: one row per (layer, rank index).
    with open(out_dir / "importance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "rank_idx", "sigma", "grad_importance", "tier"])
        for name, layer in layers.items():
            for r in range(layer.rank):
                w.writerow([name, r, f"{layer.S[r].item():.6g}", f"{grad_scores[name][r].item():.6g}",
                            "SLC" if masks[name][r] else "MLC"])

    # Check: SLC + MLC files sum back to the factorised weight exactly.
    max_err = max((reconstruct_weight(out_dir, n) - l.effective_weight()).abs().max().item()
                  for n, l in layers.items())
    print(f"[{args.model}] SLC+MLC reconstruction max abs error {max_err:.3e}")

    n_slc = sum(e["SLC"]["count"] for e in tier_map.values())
    n_all = sum(e["rank"] for e in tier_map.values())
    slc_params = sum(e["SLC"]["count"] * (e["out_features"] + e["in_features"]) for e in tier_map.values())
    all_params = sum(e["rank"] * (e["out_features"] + e["in_features"]) for e in tier_map.values())
    summary = {
        "model": args.model, "hf_id": MODELS[args.model]["hf_id"], "args": {k: str(v) for k, v in vars(args).items()},
        "accuracy": acc, "num_layers": len(layers),
        "components": {"SLC": n_slc, "MLC": n_all - n_slc, "slc_fraction": n_slc / n_all},
        "weights": {"SLC": slc_params, "MLC": all_params - slc_params, "slc_fraction": slc_params / all_params},
        "reconstruction_max_abs_error": max_err, "runtime_s": round(time.time() - t0, 1),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("accuracy", "components", "weights")}, indent=2))


if __name__ == "__main__":
    main()
