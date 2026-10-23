"""Model registry and SVD factorisation of the transformer encoder layers."""
import re

import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .svd_linear import SVDLinear, param_preserving_rank

# Both are already fine-tuned on GLUE SST-2, so gradient importance is meaningful
# without training a task head first. `target` selects the encoder Linear layers;
# embeddings and the classifier head stay digital and are not factorised.
MODELS = {
    "distilbert-sst2": {
        "hf_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "target": r"^distilbert\.transformer\.layer\.\d+\.",
    },
    "bert-base-sst2": {
        "hf_id": "textattack/bert-base-uncased-SST-2",
        "target": r"^bert\.encoder\.layer\.\d+\.",
    },
}


def load_model(key: str):
    spec = MODELS[key]
    tokenizer = AutoTokenizer.from_pretrained(spec["hf_id"])
    model = AutoModelForSequenceClassification.from_pretrained(spec["hf_id"])
    return model, tokenizer


def factorize(model: nn.Module, key: str, rank_mode: str) -> dict[str, SVDLinear]:
    """Replace every target nn.Linear with an SVDLinear in place.

    rank_mode: "param" -> k = floor(mn/(m+n)) (HyFlexPIM, parameter-preserving, lossy)
               "full"  -> k = min(m, n)        (exact, no truncation)
    Returns {layer_name: SVDLinear}.
    """
    pattern = re.compile(MODELS[key]["target"])
    targets = [(name, mod) for name, mod in model.named_modules()
               if isinstance(mod, nn.Linear) and pattern.match(name)]
    layers = {}
    for name, linear in targets:
        m, n = linear.out_features, linear.in_features
        k = param_preserving_rank(m, n) if rank_mode == "param" else min(m, n)
        svd = SVDLinear.from_linear(linear, k)
        parent_name, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child, svd)
        layers[name] = svd
    return layers
