# ReRAM crossbar simulator and experiments

`kaggle_reram_experiments.py` is a single Kaggle notebook cell. It contains a bit-accurate crossbar
simulator, the placement scores, and every experiment in the results report.

## Simulator
- 8-bit signed weights per row, stored on a pair of crossbars (positive and negative parts) and split
  across cells: 7 SLC cells (1 bit each) or 4 MLC cells (2 bits each) per sign.
- 128×128 tiles, on/off ratio 100.
- Programming noise σ, as a fraction of the maximum conductance.
- Stuck-at cells: scattered, in 3×3 clusters, or as whole bit lines.
- Read noise, fresh at every read.
- Level-dependent MLC noise.
- Static IR drop.
- A fixed, calibrated ADC with bit-serial inputs.

Every run starts with self-tests that check the simulator against exact results.

## Placements
The names match the results report.

| Policy | Which components go to SLC |
|---|---|
| `all_slc`, `all_mlc` | all of them / none of them |
| `random` | a random share (control) |
| `magnitude` | the largest singular values σ_r (recommended) |
| `gradient` | HyFlexPIM-style: \|∂L/∂σ_r\| summed over batches of 32 |
| `fisher` | the largest expected loss increase if left in MLC: Fisher information × noise variance, summed over every stored entry of the component |

## Experiments
`slc_share`, `noise`, `fault`, `read`, `level`, `cluster`, `irdrop`, `adc`, `adc_noise`. They are
listed at the top of the script. Every placement is evaluated on the same simulated chips (seeds), so
results compare directly.

## Run
Kaggle, GPU T4, Internet on. Add the `svd_tiering` notebook output as input. Set
`EXPERIMENT = "smoke"` for a quick check, then an experiment name or `"all"`, and use Save & Run All.

## Results
`results/<experiment>/<model>/sweep.csv` holds one row per chip:
`policy, slc_frac, param, value, seed, acc`. `sweep_summary.csv` gives the mean, standard deviation,
worst chip and count for each point. The report's tables are built from these files.
