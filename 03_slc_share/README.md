# Step 3: Placement scores and SLC share

**Report:** Sections IV-C (placement scores) and VIII-C (Fig. 3, Tables IV–V), IX-A (why magnitude works).

`slc_share.py` computes the three placement scores and measures accuracy against the share of components
in SLC, at three noise levels:

| Score | Definition |
|---|---|
| `magnitude` | \|σ_r\| (no data needed) |
| `gradient` | HyFlexPIM-style: Σ over batches of 32 of \|∂L/∂σ_r\|, dropout on |
| `fisher` | expected loss increase in MLC: Σ over every stored entry of Fisher(entry) × s², where s is the 8-bit scale of the crossbar column holding it |

| Experiment | Swept | Fixed | Chips per point |
|---|---|---|---|
| `slc_share` | SLC share 5, 10, 20, 30, 50% | σ = 3, 5, 7.5% | 20 |

All scores use the same 2,048 SST-2 training sentences. Every placement sees the same chips.

## Run
Kaggle, GPU T4, Internet on, the step 1 notebook as input. `EXPERIMENT = "smoke"`, then `"all"`.

## Results
`results/slc_share/<model>/sweep.csv` and `sweep_summary.csv`. BERT-Base at σ = 7.5%, mean accuracy (%):

| Placement | 5% SLC | 10% | 20% | 30% | 50% |
|---|---|---|---|---|---|
| Magnitude | 79.4 | 81.8 | 84.1 | 85.5 | 85.6 |
| Fisher | 80.0 | 82.2 | 84.2 | 85.1 | 85.8 |
| Gradient | 67.5 | 79.0 | 81.2 | 82.4 | 83.2 |
| Random | 53.6 | 54.2 | 55.2 | 57.5 | 67.3 |

(all-MLC 53.0, all-SLC 85.7). Paired over all 300 chips, Fisher − magnitude is +0.4 ± 0.7 sentences
(BERT-Base) and −0.6 ± 0.7 (DistilBERT) out of 872; the gradient is behind by 17.3 ± 2.0 and 7.8 ± 1.1.
With magnitude ranking, 5–10% SLC matches all-SLC at σ = 5%.
