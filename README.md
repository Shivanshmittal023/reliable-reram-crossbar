# Reliable ReRAM Crossbar Accelerators for Transformers

CS5290 Memory-Centric Computer Architecture course project. We study how to store a transformer on a
ReRAM crossbar that mixes reliable one-bit cells (SLC) with dense two-bit cells (MLC). Each weight
matrix is split with SVD into components, the most important components go to SLC, and we simulate
the chip cell by cell to measure accuracy under realistic hardware errors.

Models: **BERT-Base** and **DistilBERT**, fine-tuned on **SST-2** (sentiment classification).

## Repository layout

```
svd_tiering/   step 1: SVD factorisation, fine-tuning and component scores (Kaggle script + outputs)
reram_sim/     step 2: crossbar simulator and every experiment in the report (one Kaggle script + results)
docs/          project plan and results report (PDF and LaTeX)
```

## Reproducing the results
Everything runs on Kaggle (GPU T4, Internet on). Each script is one self-contained notebook cell.

1. **`svd_tiering/kaggle_svd_tiering.py`**: factorises both models and fine-tunes them. Its notebook
   output (`outputs/<model>/svd_model.pt`, `importance.pt`, `tier_map.json`) is the input to step 2.
2. **`reram_sim/kaggle_reram_experiments.py`**: add the step 1 notebook as input, then set
   `EXPERIMENT` to `"smoke"` (a quick check), an experiment name, or `"all"`. The script self-tests the
   simulator, computes the placement scores, and writes `results/<experiment>/<model>/sweep.csv`.

Model weights (`*.pt`) are not stored in the repository; regenerate them with step 1.

## Main findings
- All-MLC storage fails at realistic programming noise; a hybrid SLC/MLC layout is necessary.
- Placing the components with the largest singular values in SLC (magnitude ranking) is as good as the
  best placement we could construct: a Fisher-information estimate of each component's expected damage
  gives the same accuracy, while a HyFlexPIM-style gradient is clearly weaker at high noise. With
  10–30% of components in SLC, magnitude ranking matches an all-SLC chip and saves 30–40% of the
  memory cells.
- The conclusions hold under read noise, level-dependent noise, a fixed 6-bit ADC and realistic IR drop.
- Stuck-at faults break every placement at rates below those reported for real arrays, which
  motivates fault-aware mapping (the next step).

See [`docs/results_report.pdf`](docs/results_report.pdf) for the full analysis.
