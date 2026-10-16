"""Shared helpers: determinism (D11, D16), data (D2, D15), evaluation, checkpoints."""
import csv
import datetime
import os
import random
import subprocess

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from models import build_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- setup

def setup(seed, deterministic=True):
    """Seed everything and disable TF32 (D11). deterministic=False allows the
    cuDNN autotuner (FP32 training only)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device(allow_mps=False):
    """CUDA if available; MPS only for float training (no float64 support)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def run_meta(arch, seed):
    return {
        "arch": arch,
        "seed": seed,
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------- data

def cifar10_loaders(data_dir, batch_size=128, workers=2, seed=0):
    """Train (augmented), test, and calibration (train images, no augmentation)
    loaders. No mean/std normalisation: pixels stay k/255 (D2)."""
    plain = T.ToTensor()
    augment = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=augment)
    calib_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=plain)
    test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=plain)
    g = torch.Generator().manual_seed(seed)
    kw = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available())
    train = torch.utils.data.DataLoader(train_set, shuffle=True, drop_last=True, generator=g, **kw)
    calib = torch.utils.data.DataLoader(calib_set, shuffle=True,
                                        generator=torch.Generator().manual_seed(seed), **kw)
    test = torch.utils.data.DataLoader(test_set, shuffle=False, **kw)
    return train, test, calib


# --------------------------------------------------------------------------- train / eval

@torch.no_grad()
def evaluate(model, loader, device):
    """Top-1 accuracy in percent."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = correct = seen = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(x)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * y.numel()
        correct += (out.argmax(1) == y).sum().item()
        seen += y.numel()
    return total_loss / seen, 100.0 * correct / seen


# --------------------------------------------------------------------------- checkpoints

def save_checkpoint(path, model, arch, folded, meta, **extra):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"arch": arch, "folded": folded, "meta": meta,
                "state_dict": model.state_dict(), **extra}, tmp)
    os.replace(tmp, path)   # atomic: a Colab disconnect never leaves a half-written file


def load_checkpoint(path, device="cpu"):
    """Return (model, checkpoint dict). The model is in float ('fp') mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], folded=ckpt["folded"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), ckpt


RESULTS_FIELDS = ["arch", "seed", "fp32_acc", "folded_acc", "ptq_acc", "qat_fakequant_acc",
                  "int8_acc", "git_commit", "date"]


def append_result(row, path=os.path.join(REPO_ROOT, "results", "baseline_accuracy.csv")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
