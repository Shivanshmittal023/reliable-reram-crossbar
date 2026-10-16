"""Short recovery fine-tuning and per-component importance scores."""
import itertools

import torch

from .svd_linear import SVDLinear


def finetune(model, loader, steps: int, lr: float) -> list[float]:
    """Recover accuracy lost to SVD truncation (HyFlexPIM: 1-3 epochs; here step-limited)."""
    if steps <= 0:
        return []
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step, batch in enumerate(itertools.islice(itertools.cycle(loader), steps)):
        loss = model(**batch).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 20 == 0 or step == steps - 1:
            print(f"  finetune step {step + 1}/{steps}  loss {loss.item():.4f}", flush=True)
    return losses


def gradient_importance(model, layers: dict[str, SVDLinear], loader) -> dict[str, torch.Tensor]:
    """I_r = |dL/dσ_r|, with L the loss summed over the calibration set.

    Gradients are accumulated over all calibration batches before taking the
    absolute value, so I_r is the magnitude of the full-dataset gradient.
    Dropout is off (eval mode) so the score is deterministic.
    """
    model.eval()
    model.zero_grad(set_to_none=True)
    for batch in loader:
        model(**batch).loss.backward()
    scores = {name: layer.S.grad.detach().abs().clone() for name, layer in layers.items()}
    model.zero_grad(set_to_none=True)
    return scores


def magnitude_importance(layers: dict[str, SVDLinear]) -> dict[str, torch.Tensor]:
    """I_r = σ_r (rank/magnitude baseline from HyFlexPIM's comparison)."""
    return {name: layer.S.detach().abs().clone() for name, layer in layers.items()}
