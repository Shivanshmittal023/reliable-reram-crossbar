# SVD + importance-based SLC/MLC tiering (Step 2 / 5b)

Implements the HyFlexPIM (Song et al., ISCA 2025) weight split for two
open transformers fine-tuned on GLUE SST-2:

| key               | Hugging Face model                                  | encoder Linear layers |
|-------------------|-----------------------------------------------------|-----------------------|
| `distilbert-sst2` | `distilbert-base-uncased-finetuned-sst-2-english`   | 36 (6 blocks x 6)     |
| `bert-base-sst2`  | `textattack/bert-base-uncased-SST-2`                | 72 (12 blocks x 6)    |

The ReRAM crossbar layers are out of scope here; this produces the weights
they consume.

## Pipeline

1. Each encoder `nn.Linear` is replaced by `SVDLinear`: `W ≈ U · diag(S) · Vh`,
   rank `k = floor(mn/(m+n))` (parameter-preserving) or `k = min(m,n)` with `--rank-mode full`.
   On hardware: `x -> [ΣVᵀ (k x in)] -> [U (out x k)] -> y`, two crossbars back to back.
2. Short fine-tune to recover truncation loss (`--finetune-steps`).
3. Importance `I_r = |∂L/∂σ_r|`, gradient accumulated over a calibration set
   (`--metric magnitude` uses `σ_r` instead, for comparison).
4. Top `--slc-frac` components (per layer, or `--scope global`) go to **SLC**, rest to **MLC**.
   Component `r` = column `r` of `U` + row `r` of `ΣVᵀ`; both move together.

## Run

```
pip install -r requirements.txt
python scripts/run_svd_tiering.py --model distilbert-sst2
python scripts/run_svd_tiering.py --model bert-base-sst2
```

## Output: `outputs/<model>/`

```
SLC/<layer>.pt      important components   {"indices", "U" (out x k_slc), "SVh" (k_slc x in), "importance"}
MLC/<layer>.pt      remaining components   {"indices", "U" (out x k_mlc), "SVh" (k_mlc x in), "importance"}
tier_map.json       per layer: shapes, rank, SLC/MLC counts and indices
importance.pt       {"gradient": {layer: I}, "magnitude": {layer: sigma}}
importance.csv      one row per component: layer, rank_idx, sigma, grad_importance, tier
svd_model.pt        full factorised + fine-tuned state_dict (embeddings, LayerNorm, head, biases)
summary.json        accuracies (baseline / truncated / fine-tuned), SLC/MLC counts
```

Layer output on hardware: `y = U_slc (SVh_slc x) + U_mlc (SVh_mlc x) + bias`.
`indices` are positions in the original σ-descending order, so the tiers merge back exactly;
the script checks `U_slc@SVh_slc + U_mlc@SVh_mlc == U·diag(S)·Vh` per layer.

Loading one layer for the simulator:

```python
from svd_tiering import load_layer
t = load_layer(Path("outputs/distilbert-sst2"), "distilbert.transformer.layer.0.attention.q_lin")
t["SLC"]["U"], t["SLC"]["SVh"], t["MLC"]["U"], t["MLC"]["SVh"]
```

Biases, embeddings, LayerNorm and the classifier head are not tiered; take them from `svd_model.pt`.
