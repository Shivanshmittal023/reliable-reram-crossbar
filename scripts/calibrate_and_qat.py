"""Step 1b: FP32 checkpoint -> fold BN -> calibrate -> QAT -> INT8 golden export.

    python -m scripts.calibrate_and_qat --fp32 checkpoints/resnet20_fp32_seed0.pt

Produces checkpoints/<arch>_qat_seed<seed>.pt (float, folded, for later
fine-tuning) and checkpoints/<arch>_int8_seed<seed>.{pt,npz} (golden model),
and appends a row to results/baseline_accuracy.csv.
"""
import argparse
import os

import torch

from quant.export import build_int_model, export_int8, load_int8
from quant.fold_bn import fold_bn
from quant.prepare import calibrate, freeze_int, set_mode
from utils.common import (append_result, cifar10_loaders, evaluate, get_device, load_checkpoint,
                          run_meta, save_checkpoint, setup, train_one_epoch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp32", required=True, help="FP32 checkpoint from train_fp32")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-5)
    p.add_argument("--calib-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--no-results", action="store_true", help="do not append to results csv")
    args = p.parse_args()

    setup(args.seed, deterministic=True)
    device = get_device()
    train_loader, test_loader, calib_loader = cifar10_loaders(
        args.data_dir, args.batch_size, args.workers, args.seed)

    fp32, ckpt = load_checkpoint(args.fp32, device)
    arch = ckpt["arch"]
    meta = run_meta(arch, args.seed)
    meta["fp32_checkpoint"] = os.path.basename(args.fp32)
    meta["fp32_acc"] = evaluate(fp32, test_loader, device)
    print(f"FP32            {meta['fp32_acc']:.2f}%")

    # D6: fold BN and check the folded model computes the same function.
    model = fold_bn(fp32)
    x, _ = next(iter(test_loader))
    with torch.no_grad():
        diff = (fp32.eval()(x.to(device)) - model(x.to(device))).abs().max().item()
    meta["folded_acc"] = evaluate(model, test_loader, device)
    print(f"BN folded       {meta['folded_acc']:.2f}%   (max logit diff {diff:.2e})")
    if diff > 1e-3:
        raise RuntimeError("BN folding changed the model output")

    # D5: calibrate activation scales once, then freeze them.
    calibrate(model, calib_loader, args.calib_batches, device)
    meta["ptq_acc"] = evaluate(model, test_loader, device)
    print(f"INT8 PTQ        {meta['ptq_acc']:.2f}%   (before QAT)")

    # QAT: fake-quant weights with STE, activation scales frozen.
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    for epoch in range(args.epochs):
        loss, train_acc = train_one_epoch(model, train_loader, opt, device)
        sched.step()
        print(f"qat epoch {epoch + 1:2d}/{args.epochs}  loss {loss:.4f}  train {train_acc:.2f}%", flush=True)
    meta["qat_fakequant_acc"] = evaluate(model, test_loader, device)
    print(f"INT8 QAT (fake) {meta['qat_fakequant_acc']:.2f}%")

    os.makedirs(args.out_dir, exist_ok=True)
    save_checkpoint(os.path.join(args.out_dir, f"{arch}_qat_seed{args.seed}.pt"), model, arch, True, meta)

    # D10: export, then measure accuracy of the integer reference loaded *from the file*.
    freeze_int(model)
    stem = os.path.join(args.out_dir, f"{arch}_int8_seed{args.seed}")
    export_int8(model, arch, meta, stem)
    int_model = build_int_model(load_int8(stem + ".pt"), device)
    meta["int8_acc"] = evaluate(int_model, test_loader, device)
    print(f"INT8 integer    {meta['int8_acc']:.2f}%   (golden)")
    export_int8(int_model, arch, meta, stem)   # rewrite with the final accuracies in meta

    gap = abs(meta["int8_acc"] - meta["qat_fakequant_acc"])
    if gap > 0.05:
        print(f"WARNING: integer reference and fake-quant differ by {gap:.2f}% (expected <= 0.05%)")
    if meta["fp32_acc"] - meta["int8_acc"] > 0.5:
        print("WARNING: INT8 is more than 0.5% below FP32")
    if not args.no_results:
        append_result(meta)
    print(f"golden export -> {stem}.pt / .npz")


if __name__ == "__main__":
    main()
