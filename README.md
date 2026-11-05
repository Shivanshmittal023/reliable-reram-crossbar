# Reliable Hybrid SLC/MLC ReRAM Crossbars for Transformers

**Component placement, fault-aware mapping and noise-aware training.** Course project for CS5290
Memory-Centric Computer Architecture, IIT Guwahati, by Shivansh Mittal, Sarvesh Joshi, Thumati Naveen
Chandra Chowdary and Aryan Chakravorty.

A ReRAM crossbar computes matrix–vector products where the weights are stored, but its cells are
imprecise and some are stuck. HyFlexPIM (ISCA 2025) stores the most important SVD components of each
transformer weight matrix in reliable one-bit cells (SLC) and the rest in dense two-bit cells (MLC). We
re-examine this design in a bit-accurate, cell-level simulator. The questions are which components to
protect, which hardware effects still break the chip, and how to recover accuracy without extra hardware.

**Report:** [`report/report.pdf`](report/report.pdf) (LaTeX source in [`report/`](report/)).

Models: BERT-Base and DistilBERT on SST-2 (full study) and MNLI (placement and mapping).

## Main findings
- **A hybrid layout is necessary.** All-MLC storage falls to 52–60% on SST-2 at 7.5% programming noise.
- **Rank components by singular value.** Magnitude ranking matches a Fisher-information estimate of each
  component's expected damage, chip by chip, and beats a HyFlexPIM-style gradient by 2–12 points at
  7.5% noise. The reason is that each crossbar column (output) has its own 8-bit scale, so the weight error a fixed
  cell noise causes grows with the singular value. With magnitude ranking, 5–10% of components in SLC
  match an all-SLC chip at 5% noise.
- **Stuck cells are the limiting effect.** Read noise, level-dependent noise, a 6-bit ADC and moderate
  IR drop leave the hybrid within about a point of all-SLC; stuck cells break every placement.
- **Fault-aware mapping.** The components of an SVD layer can be stored in any order, so a tested chip's
  stuck cells can be avoided by a Fisher-weighted assignment of components to slots. This needs no extra
  hardware and raises BERT-Base at 0.5% stuck cells from 76.6% to 85.6%. Choosing the tier per chip as
  well (joint mapping) adds nothing reliable.
- **Tier-aware training.** Training on simulated chips helps only when stuck cells are injected.
  Training each component under the errors of its own tier, together with mapping, raises BERT-Base at
  1% stuck cells from 65.1% to 82.9%. The hybrid then matches or beats an all-SLC chip with mapping up to
  1% scattered stuck cells, using 34% fewer cells.
- **MNLI** confirms the placement and mapping results. It also exposed a simulator limitation: one
  8-bit scale for the vector between a tier's two crossbars costs up to 11 points even on an error-free
  chip.

## Repository layout
The folders follow the order of the work and of the report. Each step has one self-contained Kaggle
script, a README (method, settings, how to run, findings) and its `results/`.

| Step | Folder | Script | What it does | Report |
|---|---|---|---|---|
| 1 | [`01_svd_tiering/`](01_svd_tiering/) | `svd_tiering.py` | SVD factorisation and fine-tuning of both SST-2 models | IV-A, VIII-A |
| 2 | [`02_programming_noise/`](02_programming_noise/) | `programming_noise.py` | accuracy against programming noise, all placements | VIII-B |
| 3 | [`03_slc_share/`](03_slc_share/) | `slc_share.py` | placement scores (magnitude, gradient, Fisher) and accuracy against the SLC share | IV-C, VIII-C |
| 4 | [`04_stuck_cells/`](04_stuck_cells/) | `stuck_cells.py` | accuracy against the share of stuck cells | VIII-D |
| 5 | [`05_other_non_idealities/`](05_other_non_idealities/) | `other_non_idealities.py` | read noise, level noise, fault patterns, IR drop, fixed ADC | VIII-E |
| 6 | [`06_fault_aware_mapping/`](06_fault_aware_mapping/) | `fault_aware_mapping.py` | fault-aware and joint mapping on tested chips | V, VIII-F |
| 7 | [`07_training/`](07_training/) | `training.py` | noise-, fault- and tier-aware training, tested with fault-aware mapping | VI, VIII-G |
| 8 | [`08_mnli/`](08_mnli/) | `mnli_pipeline.py` | steps 1, 3, 4 and 6 repeated on MNLI | IX |

Other folders:

```
report/       final report: report.pdf and its LaTeX source (main.tex, sections/, refs.bib, data/)
submission/   submission documents: final_report.pdf (10 pages), baseline_results.pdf and
              contribution_results.pdf, with their LaTeX sources and plot data
docs/         project plan (revision 2) and an earlier paper draft
```

Results are always stored as `results/<experiment>/<model>/sweep.csv` (one row per simulated chip:
`policy, slc_frac, param, value, seed, acc`) and `sweep_summary.csv` (mean, standard deviation and minimum
over chips). Every placement, mapping and trained model is evaluated on the same chips (same seeds), so
results compare chip by chip, also across steps.

## The crossbar simulator
Steps 2–8 contain the same simulator (each script is one Kaggle cell, so the code is repeated):
- Inputs on the rows (word lines), outputs on the columns (bit lines). 8-bit signed weights, one scale
  per output column, on a differential pair of cells, split into bit
  slices: 7 SLC cells (1 bit) or 4 MLC cells (2 bits) per sign; G_max/G_min = 100; 128×128 tiles.
- Programming noise σ·G_max per cell (fixed per chip), read noise (fresh at every read), level-dependent
  MLC noise, stuck cells (scattered, 3×3 clusters, whole bit lines), static IR drop, and a fixed
  calibrated ADC with bit-serial inputs.
- Each SVD tier is two crossbars, x → [σVᵀ] → [U]; placements choose which components use SLC.
- Every run starts with self-tests against exact integer arithmetic and stops if one fails.

## Reproducing the results
Everything runs on Kaggle (GPU T4, Internet on). Each script is pasted into one notebook cell, has a
quick `smoke` mode, and runs its self-tests before any experiment. Run step 1 first; steps 2–7 take its
notebook output as input and are independent of each other. Step 8 is self-contained.

| Step | Approximate run time on a T4 |
|---|---|
| 1 | 50 min |
| 2–4 | under 2 h each |
| 5 | about 5 h (mostly the ADC experiments, which read every input bit and slice separately) |
| 6 | 3 h |
| 7 | 7–8 h |
| 8 | 6 h |

Model weights (`*.pt`) and Kaggle archives (`*.zip`) are not stored; regenerate them with the scripts.

## Building the report
```
cd report
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
It needs the IEEEtran, pgfplots, siunitx, cleveref and colortbl packages. Plots are drawn from
`report/data/`.
