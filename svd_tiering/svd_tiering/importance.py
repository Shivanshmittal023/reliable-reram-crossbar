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
    """I_r = mean over calibration batches of |dL_b/dσ_r|.

    HyFlexPIM (Alg. 1, Fig. 11) ranks singular values by absolute gradient
    stored during fine-tuning. The absolute value is taken per batch: after
    fine-tuning the model sits near a minimum, so the signed gradient summed
    over the whole set cancels to noise and no longer reflects sensitivity.
    Dropout is off (eval mode) so the score is deterministic.
    """
    model.eval()
    scores = {name: torch.zeros_like(layer.S) for name, layer in layers.items()}
    n = 0
    for batch in loader:
        model.zero_grad(set_to_none=True)
        model(**batch).loss.backward()
        for name, layer in layers.items():
            scores[name] += layer.S.grad.detach().abs()
        n += 1
    model.zero_grad(set_to_none=True)
    return {name: s / max(n, 1) for name, s in scores.items()}


def magnitude_importance(layers: dict[str, SVDLinear]) -> dict[str, torch.Tensor]:
    """I_r = σ_r (rank/magnitude baseline from HyFlexPIM's comparison)."""
    return {name: layer.S.detach().abs().clone() for name, layer in layers.items()}
