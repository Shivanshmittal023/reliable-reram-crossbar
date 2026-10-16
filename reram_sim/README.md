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
- **Conductance.** Each digit `d` is programmed as conductance `g = g_min + d·Δ`, where `Δ = (G_max − g_min)/(L−1)`. Conductances are normalised so that `G_max = 1`, and `on_off = 100`.
- **Programming noise.** Noise is added as `g += σ·G_max·N(0,1)`, then clamped at ≥ 0. The same σ costs MLC more than SLC, because MLC levels are spaced closer together.
- **Stuck-at faults.** A cell is stuck with probability `fault_rate`. A stuck cell is set to `G_max` (fraction `sa1_frac` of stuck cells) or to `G_min` (the rest).
- **Readout.** The read-out digit is `(g⁺ − g⁻)/Δ`. Slices are combined by shift-and-add.
- **DAC.** Inputs use an `in_bits` (8) DAC with a per-tensor scale.
- **ADC.** The ADC is optional (`adc_bits`). It quantises each slice and each 128-row tile separately, with an ideally calibrated range.
- **Chip seed.** One seeded generator programs the whole model, so each seed represents one simulated chip.

## Tiering policies

All policies use the same factorised weights. Only the SLC/MLC split changes:

- `all_slc`: every component in SLC.
- `all_mlc`: every component in MLC.
- `gradient`: top 20% by `|∂L/∂σ|` in SLC. This is identical to `svd_tiering/outputs/<model>/SLC`.
- `magnitude`: top 20% by σ in SLC.
- `random`: a random 20% in SLC.

## Local (CPU): smoke only

```
python -m pytest tests -q                    # 23 tests, ~1 min
python scripts/noise_sweep.py --model distilbert-sst2 --eval-n 128 --sigmas 0,0.05,0.1 --seeds 1
```

On CPU, each sweep point takes about 20 s (8.5 s of that is programming the chip). Run the full sweep on Kaggle.

## Kaggle (GPU): full sweep

1. Accelerator GPU, Internet on.
2. Add the SVD-tiering notebook's output as an input. Alternatively, upload `svd_tiering/outputs.zip` as a dataset.
3. Paste `kaggle/kaggle_reram_sweep.py` into one cell and run it. Edit `CONFIG` at the top if needed. The defaults cover both models, 5 policies, 7 σ values, and 5 chips.
4. Download `reram_results.zip`. It holds, per model: `noise_sweep.csv`, `noise_sweep_summary.csv`, `config.json` and `noise_sweep.png`.

For a fault sweep, set `"sigmas": [0.0]` and `"fault_rates": [0, 0.001, 0.01, 0.05, 0.1]`.
