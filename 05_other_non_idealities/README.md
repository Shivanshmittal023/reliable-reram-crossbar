# Step 5: Other non-idealities

**Report:** Section VIII-E (Table VI; DistilBERT in the appendix).

`other_non_idealities.py` adds the remaining hardware non-idealities one at a time, with 20% SLC for the hybrids:

| Experiment | Effect | Chips per point |
|---|---|---|
| `read` | read noise σ_r = 0.5–5%, fresh at every read | 5 |
| `level` | middle MLC levels 1.5× / 2× noisier, at σ = 5% | 10 |
| `cluster` | 1% stuck cells: scattered, 3×3 clusters, whole bit lines | 10 |
| `irdrop` | static IR drop, wire resistance r = 10⁻⁴–3×10⁻³ of R_on per segment | 1 (deterministic) |
| `adc` | fixed calibrated ADC, 3–8 bits, bit-serial inputs | 1 (deterministic) |
| `adc_noise` | the same ADC at σ = 5% (DistilBERT only) | 3 |

These experiments use evaluation batches of 64; the ADC range is calibrated once on 64 training
sentences on an error-free chip and then fixed.

## Run
Kaggle, GPU T4, Internet on, the step 1 notebook as input. `EXPERIMENT = "smoke"`, then `"all"` (the ADC
experiments read every input bit and slice separately and take most of the time).

## Results
`results/<experiment>/<model>/sweep.csv` and `sweep_summary.csv`. BERT-Base, mean accuracy (%):

| Condition | All SLC | All MLC | Magnitude | Fisher | Gradient |
|---|---|---|---|---|---|
| Read noise 5% | 80.0 | 53.0 | 82.5 | 82.7 | 76.6 |
| σ = 5%, middle levels ×2 | 88.2 | 62.5 | 88.4 | 88.4 | 87.8 |
| 1% stuck, bit lines | 57.0 | 50.4 | 50.4 | 52.2 | 51.0 |
| ADC 6 bits | 89.3 | 83.0 | 90.0 | 90.5 | 89.6 |
| IR drop r = 3×10⁻³ | 58.5 | 79.1 | 76.8 | 72.8 | 76.4 |

Read noise and level-dependent noise behave like programming noise; a 6-bit ADC suffices for the
hybrids; IR drop matters only at large wire resistance and hurts SLC most; stuck cells remain the
limiting effect.
