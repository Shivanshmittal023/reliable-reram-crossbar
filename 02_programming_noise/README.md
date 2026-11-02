# Step 2: Programming noise

**Report:** Section VIII-B (Fig. 2) and the error-free check of Section VIII-A.

`programming_noise.py` places 20% of the SVD components in SLC by each policy and measures SST-2
accuracy as the programming noise σ grows. It contains the full crossbar simulator (see the top-level
README) and computes the placement scores itself. Steps 2–5 share the same code and differ only in the
experiments they run.

| Experiment | Swept | Chips per point | Placements |
|---|---|---|---|
| `noise` | σ = 0, 1, 2, 3, 5, 7.5, 10% of G_max | 5 | all-SLC, all-MLC, random, magnitude, gradient, Fisher (20% SLC) |

## Run
Kaggle, GPU T4, Internet on, the step 1 notebook as input. Run with `EXPERIMENT = "smoke"` first, then
`"all"`.

## Results
`results/noise/<model>/sweep.csv` (one row per chip) and `sweep_summary.csv`. BERT-Base, mean accuracy (%):

| Placement | σ = 0 | 3% | 5% | 7.5% | 10% |
|---|---|---|---|---|---|
| All SLC | 89.9 | 89.1 | 87.8 | 85.9 | 79.7 |
| All MLC | 90.0 | 83.9 | 66.3 | 52.2 | 51.3 |
| Magnitude | 89.7 | 90.0 | 88.6 | 83.5 | 72.0 |
| Fisher | 89.1 | 90.0 | 89.0 | 84.5 | 73.7 |
| Gradient | 89.8 | 90.0 | 88.0 | 79.3 | 67.7 |
| Random | 90.0 | 87.7 | 78.8 | 58.8 | 49.1 |

All-MLC fails from σ = 5%. Magnitude and Fisher with 20% SLC recover almost all of the loss up to 5%;
the gradient falls behind beyond that. With no noise, every placement is within about a point of the
digital SVD model.
