"""Integration with the svd_tiering outputs (skipped when they are not on disk)."""
import pytest
import torch

from reram_sim import CrossbarConfig, program_model
from reram_sim.hf import DEFAULT_OUTPUTS, load_svd_model, tier_masks, to_reram

KEY = "distilbert-sst2"
OUT = DEFAULT_OUTPUTS / KEY
pytestmark = pytest.mark.skipif(not (OUT / "svd_model.pt").exists(), reason="svd_tiering outputs missing")


@pytest.fixture(scope="module")
def loaded():
    return load_svd_model(KEY, OUT)


def test_gradient_policy_matches_saved_tiers(loaded):
    _, _, layers, tier_map = loaded
    importance = torch.load(OUT / "importance.pt")
    masks = tier_masks(layers, importance, "gradient", 0.2)
    for name, mask in masks.items():
        assert mask.nonzero().flatten().tolist() == tier_map[name]["SLC"]["indices"]


def test_ideal_reram_matches_svd_model(loaded):
    model, tok, layers, _ = loaded
    batch = tok(["a gripping, beautifully shot film", "dull and far too long"], padding=True, return_tensors="pt")
    with torch.no_grad():
        ref = model(**batch).logits
    importance = torch.load(OUT / "importance.pt")
    to_reram(model, layers, tier_masks(layers, importance, "gradient", 0.2))
    ideal = CrossbarConfig(w_bits=24, in_bits=None)
    program_model(model, {"SLC": ideal, "MLC": ideal}, seed=0)
    with torch.no_grad():
        torch.testing.assert_close(model(**batch).logits, ref, rtol=1e-3, atol=1e-3)
