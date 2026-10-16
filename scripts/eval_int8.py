"""Evaluate the INT8 golden model straight from its export (D10).

    python -m scripts.eval_int8 --export checkpoints/resnet20_int8_seed0.pt
"""
import argparse

import torch

from quant.export import build_int_model, load_int8
from utils.common import cifar10_loaders, evaluate, get_device, setup


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    setup(0, deterministic=True)
    device = torch.device("cpu") if args.cpu else get_device()
    blob = load_int8(args.export)
    model = build_int_model(blob, device)
    _, test_loader, _ = cifar10_loaders(args.data_dir, args.batch_size, args.workers)

    acc = evaluate(model, test_loader, device)
    expected = blob["meta"].get("int8_acc")
    print(f"{blob['arch']} INT8 integer reference: {acc:.2f}%  (export says {expected})")

    x, _ = next(iter(test_loader))
    with torch.no_grad():
        same = torch.equal(model(x.to(device)), model(x.to(device)))
    print(f"deterministic logits: {same}")
    if expected is not None and abs(acc - expected) > 1e-9:
        print("NOTE: differs from the stored accuracy (different device/torch version?)")


if __name__ == "__main__":
    main()
