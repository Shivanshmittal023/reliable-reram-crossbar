# ReRAM crossbar simulator (Steps 2–5)

This package runs the SVD-tiered transformers from `../svd_tiering` on simulated ReRAM crossbars. It then measures SST-2 accuracy as programming noise and stuck-at faults increase.

## Layout

```
reram_sim/crossbar.py    Crossbar: one weight matrix on simulated cells (quantise, bit-slice, noise, faults, DAC/ADC)
reram_sim/layer.py       ReRAMLinear: y = U_slc(SVh_slc x) + U_mlc(SVh_mlc x) + b; program_model()
reram_sim/hf.py          loads svd_tiering outputs, tier policies, swaps SVDLinear -> ReRAMLinear
scripts/noise_sweep.py   local CLI sweep (CPU: use for small smoke runs only)
tests/                   pytest: toy matrices, toy MLP, nn.TransformerEncoderLayer, DistilBERT integration
kaggle/kaggle_reram_sweep.py   self-contained single-cell GPU version for the full sweep
```

## Cell model

- **Quantisation.** Weights are quantised per row to signed `w_bits` (8) integers. Each integer is split into a differential pair `q = q⁺ − q⁻`.
- **Bit-slicing.** Each magnitude is split into base-`L` digits, where `L = 2^bits_per_cell`. SLC uses 1 bit per cell, so 7 slices. MLC uses 2 bits per cell, so 4 slices.
- **Conductance.** Each digit `d` is programmed as conductance `g = g_min + d·Δ`, where `Δ = (G_max − g_min)/(L−1)`. Conductances are normalised so that `G_max = 1`, and `on_off = 150` (HyFlexPIM).
- **Programming noise.** `noise_model="cell"` (default) adds `g += σ·G_max·N(0,1)` to every cell, then clamps at ≥ 0. The same σ costs MLC more than SLC, because MLC levels are spaced closer together. `noise_model="multiplicative"` instead reproduces HyFlexPIM eq. (5): each weight becomes `w·(1 + σ·N(0,1))`.
- **Stuck-at faults.** A cell is stuck with probability `fault_rate`. A stuck cell is set to `G_max` (fraction `sa1_frac` of stuck cells) or to `G_min` (the rest).
- **Readout.** The read-out digit is `(g⁺ − g⁻)/Δ`. Slices are combined by shift-and-add.
- **DAC.** Inputs use an `in_bits` (8) DAC with a per-tensor scale.
- **ADC.** The ADC is optional (`adc_bits`). It quantises each slice and each 128-row tile separately, with an ideally calibrated range.
- **Chip seed.** One seeded generator programs the whole model, so each seed represents one simulated chip.

## Tiering policies

All policies use the same factorised weights. Only the SLC/MLC split changes:

- `all_slc`: every component in SLC.
- `all_mlc`: every component in MLC.
- `gradient`: top 20% by absolute gradient `|∂L/∂σ|` in SLC (HyFlexPIM Alg. 1 step 4). The Kaggle sweep computes it as the mean over calibration batches of the per-batch absolute gradient.
- `gradient_net` (Kaggle sweep only): `|Σ_b ∂L_b/∂σ|`, the score the first run used. It is still the `gradient` entry in `importance.pt` from the existing SVD run, so the local package uses it until `svd_tiering` is rerun. Near the fine-tuned minimum the signed per-batch gradients cancel, so this score is mostly noise.
- `rank`: top 20% by σ in SLC (HyFlexPIM Fig. 13 "rank-based").
- `random`: a random 20% in SLC.

## Local (CPU): smoke only

```
python -m pytest tests -q                    # 24 tests, ~1 min
python scripts/noise_sweep.py --model distilbert-sst2 --eval-n 128 --sigmas 0,0.05,0.1 --seeds 1
```

On CPU, each sweep point takes about 20 s (8.5 s of that is programming the chip). Run the full sweep on Kaggle.

## Kaggle (GPU): full sweep

Sweep values follow HyFlexPIM (Song et al., ISCA 2025):

- **HyFlexPIM noise.** `W(1+η)` with η ~ N(0, 0.025²), on MLC weights only. It is calibrated to a measured 2-bit MLC bit error rate of 4.04%.
- **Matching σ for our cell model.** σ/G_max = 0.095 gives the same 4.04% MLC bit error rate. At that σ the SLC bit error rate is about 1e-7.
- **Hardware.** SLC rates 5–50%, on/off 150, 64-row arrays.
- **Faults.** HyFlexPIM has none, so the fault sweep uses the project plan's 0.1–20%.

| experiment | x-axis | noise | what it shows |
|---|---|---|---|
| A_hyflexpim_slc_rate | SLC rate 5–50% | multiplicative σ=0.025, MLC only | reproduction of HyFlexPIM Fig. 12/13 |
| B_cell_slc_rate | SLC rate 5–50% | cell σ=0.095, both tiers | same question under per-cell noise |
| C_cell_noise_sweep | σ 0–0.12 | cell, 20% SLC | accuracy vs noise around the calibrated point |
| D_fault_sweep | fault rate 0.05–5% | none, 20% SLC | stuck-at fault tolerance |
| E_sa1_share | share of stuck cells at G_max | none, 20% SLC, 0.2% and 1% faults | whether stuck-at-G_max or stuck-at-G_min does the damage |

A stuck-at-G_max MLC cell reads as digit 3. In the top slice, only digits 0–1 are used, so this fault adds 192 to a weight with a full scale of 127.

1. Accelerator GPU, Internet on.
2. Add the SVD-tiering notebook's output as an input. Alternatively, upload `svd_tiering/outputs.zip` as a dataset.
3. Paste `kaggle/kaggle_reram_sweep.py` into one cell and run it. The defaults are 505 evaluations per model (5 chips each). Edit `CONFIG["experiments"]` or `"seeds"` to shorten.
4. Download `reram_results.zip`. Per model and per experiment it holds `<name>.csv` (every chip), `<name>_summary.csv` (mean, std, min) and `<name>.png`.
