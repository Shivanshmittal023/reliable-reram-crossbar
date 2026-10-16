"""Step 1a: train the FP32 baseline (docs/step1_decisions.md, training recipes).

    python -m scripts.train_fp32 --out /content/drive/MyDrive/reram/resnet20_fp32_seed0.pt

Saves after every epoch; re-running the same command resumes.
"""
import argparse
import os

import torch

from models import build_model
from utils.common import (cifar10_loaders, evaluate, get_device, load_checkpoint, run_meta,
                          save_checkpoint, setup, train_one_epoch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="resnet20")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA, FP32 training only)")
    p.add_argument("--out", default=None, help="default: checkpoints/<arch>_fp32_seed<seed>.pt")
    args = p.parse_args()
    out = args.out or f"checkpoints/{args.arch}_fp32_seed{args.seed}.pt"
    resume_path = out + ".last"

    setup(args.seed, deterministic=False)
    device = get_device(allow_mps=True)
    train_loader, test_loader, _ = cifar10_loaders(args.data_dir, args.batch_size, args.workers, args.seed)

    model = build_model(args.arch).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True,
                          weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    meta = run_meta(args.arch, args.seed)
    start = 0

    if os.path.exists(resume_path):
        model, ckpt = load_checkpoint(resume_path, device)
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True,
                              weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start = ckpt["epoch"] + 1
        print(f"resumed from {resume_path} at epoch {start}")

    for epoch in range(start, args.epochs):
        loss, train_acc = train_one_epoch(model, train_loader, opt, device, scaler)
        sched.step()
        msg = f"epoch {epoch + 1:3d}/{args.epochs}  loss {loss:.4f}  train {train_acc:.2f}%"
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            msg += f"  test {evaluate(model, test_loader, device):.2f}%"
        print(msg, flush=True)
        save_checkpoint(resume_path, model, args.arch, False, meta, epoch=epoch,
                        optimizer=opt.state_dict(), scheduler=sched.state_dict())

    acc = evaluate(model, test_loader, device)
    meta["fp32_acc"] = acc
    save_checkpoint(out, model, args.arch, False, meta)
    print(f"FP32 test accuracy: {acc:.2f}%  ->  {out}")


if __name__ == "__main__":
    main()
