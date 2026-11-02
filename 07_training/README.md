# Step 7: Noise-, fault- and tier-aware training

**Report:** Sections VI (method) and VIII-G (Tables VII–VIII).

`training.py` is a single Kaggle notebook cell. It fine-tunes the SVD models from step 1
on simulated chips and then tests them with magnitude placement (20% SLC), with and without fault-aware
mapping.

## How the training works
On every training step, each SVD factor is programmed onto a freshly drawn chip. This uses the same
`Crossbar` code as every evaluation: 8-bit rows, differential cell pairs, bit slices, programming noise
and stuck cells. The loss is computed with the damaged weights, and the update goes to the clean weights
(straight-through estimator). Training runs for 2 epochs (AdamW, learning rate 1e-5, batch size 32). The
training chip has σ = 5% programming noise and, where stated, 0.5% stuck cells.

| Variant | Chip simulated during training |
|---|---|
| `nat_noise` | every component on SLC cells, noise only (noise-only training) |
| `nat_fault` | every component on SLC cells, noise + stuck cells (noise + fault training) |
| `tier_aware` | top 20% of components by \|σ\| on SLC cells, the rest on MLC, noise + stuck cells; the SLC set is refreshed every 200 steps (tier-aware training) |

The variants differ **only in the training chip**. Every trained model is tested on the same hybrid chip
(20% SLC). The tier-unaware variants train on SLC cells because an all-MLC training chip at this noise
and fault rate is at chance accuracy, and a first attempt on it only learned to output 50/50 (loss stuck at
about 0.69).

## Experiments
10 chips per point, on the same chips as `06_fault_aware_mapping/results/` (same seeds and layer shapes give
the same fault maps and noise, so results compare chip by chip with that folder).

| Experiment | Setting | Placements |
|---|---|---|
| `fault` | 0.2–2% stuck cells, no noise | `magnitude`, `magnitude_remap` |
| `fault_noise` | 0.2–1% stuck cells, σ = 5% | `magnitude`, `magnitude_remap` |
| `pattern` | 1% stuck cells, clusters or bit lines | `magnitude`, `magnitude_remap` |
| `noise` | σ = 0, 5%, 7.5%, no stuck cells | `magnitude` |

`magnitude_remap` is magnitude placement with fault-aware mapping. The untrained model (`base`) was re-run
on the noise sweep and at 0.5% stuck cells. That re-run reproduced all 40 chip results of
`06_fault_aware_mapping/results/` exactly, which confirms that the chips are identical.

## Results
`results/<variant>/<experiment>/<model>/{sweep.csv, sweep_summary.csv}`, with variants `nat_noise`,
`nat_fault`, `tier_aware` and `base`.

At 1% stuck cells, with fault-aware mapping (BERT-Base / DistilBERT accuracy %):

| | BERT-Base | DistilBERT |
|---|---|---|
| no training (mapping only) | 65.1 | 67.0 |
| noise-only training | 67.6 | 68.2 |
| noise + fault training | 78.6 | 80.7 |
| **tier-aware training** | **82.9** | **84.7** |
| all-SLC chip with mapping (reference) | 75.8 | 84.8 |

- **Stuck cells must be injected during training.** Noise-only training gives no consistent gain.
- **Training and mapping add up.**
- **Tier-aware training is best where errors are dense** (1% stuck cells, clusters, lines, added noise).
  At 0.2% stuck cells it is slightly behind, because on an error-free chip it costs 0.3 / 1.3 points.
- **With tier-aware training and mapping, the hybrid matches or beats an all-SLC chip with mapping** up to
  1% scattered stuck cells, using 34% fewer cells.

See Section VIII-F/G of [`report/report.pdf`](../report/report.pdf).

## Run on Kaggle
1. GPU T4, Internet on. Add the step 1 notebook output as input.
2. Paste `training.py` into one cell. Run it first with `STAGE = "smoke"` (a few minutes). Every
   self-test must say PASS: the simulator checks, the mapping check, and the training-chip checks (the
   noise-free training chip equals the evaluation chip exactly, and the gradients reach the weights).
3. Set `STAGE = "all"` and use Save & Run All. That takes about 7–8 GPU hours (about 4.5 h of training and
   3 h of testing).
4. If the run stops early, start a new version with `STAGE = "eval"` and attach the earlier version as
   input. It reuses the trained models (`nat_models/`) and only runs the tests.
