# Step 4: Stuck cells

**Report:** Section VIII-D (Fig. 4).

`stuck_cells.py` makes a share p of all cells stuck, half at G_max and half at G_min, with no
programming noise, and measures accuracy with 20% SLC.

| Experiment | Swept | Chips per point | Placements |
|---|---|---|---|
| `fault` | p = 0, 0.1, 0.2, 0.5, 1, 2, 5% | 20 | all-SLC, all-MLC, random, magnitude, gradient, Fisher |

## Run
Kaggle, GPU T4, Internet on, the step 1 notebook as input. `EXPERIMENT = "smoke"`, then `"all"`.

## Results
`results/fault/<model>/sweep.csv` and `sweep_summary.csv`. BERT-Base, mean accuracy (%):

| Placement | 0 | 0.1% | 0.2% | 0.5% | 1% | 2% |
|---|---|---|---|---|---|---|
| All SLC | 89.9 | 88.3 | 85.6 | 79.0 | 62.7 | 51.4 |
| All MLC | 90.0 | 58.6 | 52.6 | 50.8 | 51.2 | 49.6 |
| Magnitude | 89.7 | 89.6 | 88.6 | 76.8 | 57.9 | 51.0 |
| Fisher | 89.1 | 89.7 | 88.4 | 75.6 | 55.5 | 49.5 |
| Gradient | 89.8 | 89.4 | 87.5 | 71.0 | 52.8 | 50.2 |

Stuck cells break every placement: all-MLC fails at 0.1%, and all-SLC loses 14–27 points at 1%. Whether
a chip survives depends on where its stuck cells fall, which motivates fault-aware mapping (step 6).
