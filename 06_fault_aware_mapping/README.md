# Step 6: Fault-aware mapping and joint mapping

**Report:** Sections V (method) and VIII-F (Table VII).

`fault_aware_mapping.py` is a single Kaggle notebook cell. It uses the same crossbar simulator as
steps 2–5 and adds a mapping step that takes a tested chip's stuck cells into account.

## Idea
In an SVD layer the k components can be stored in **any order**: component r is one row of the first
crossbar (σ_r·v_r) and one input row of the second (u_r), and moving both together leaves the layer
output unchanged. After a chip is tested, its stuck cells are known, so we can choose which component
goes into which physical slot. This needs no extra hardware and no retraining.

The cost of putting component r in slot s is its expected rise in loss: the Fisher information of every
stored entry × the squared weight error that entry gets in slot s. The error comes from the slot's
stuck cells (known after testing) and from the programming noise of the slot's tier. The Hungarian
algorithm finds the assignment with the lowest total cost. A component is always moved as a whole.

## Placements
| Policy | What is chosen |
|---|---|
| `all_slc` | everything in SLC, index order |
| `all_slc_remap` | everything in SLC, slots chosen to avoid stuck cells |
| `magnitude` | largest singular values in SLC (20%), index order |
| `magnitude_remap` | same SLC set, slots within each tier chosen to avoid stuck cells (**fault-aware mapping**) |
| `c1` | tier and slot of every component chosen together, per chip (**joint mapping**) |

All placements are tested on the same chips (same fault maps and noise), so they can be compared
chip by chip.

## Experiments
10 chips each, 20% SLC for the hybrid placements.

| Experiment | Setting |
|---|---|
| `fault` | 0.2–2% stuck cells, no programming noise |
| `fault_noise` | 0.2–1% stuck cells with 5% programming noise |
| `pattern` | 1% stuck cells, in 3×3 clusters or as whole bit lines |

Results are in `results/<experiment>/<model>/sweep.csv` (one row per chip) and `sweep_summary.csv`
(mean, std and minimum over chips), in the same format as steps 2–5.

## Run on Kaggle
1. New notebook, GPU T4 and Internet on. Add the step 1 notebook output as input.
2. Paste `fault_aware_mapping.py` into one cell. With `EXPERIMENT = "smoke"` it runs the simulator
   self-tests and a mapping self-test (predicted stuck-cell damage must match the damage measured on a
   programmed crossbar), then a tiny sweep.
3. Set `EXPERIMENT = "all"` and Save & Run All (about 3 GPU hours). Download `mapping_results.zip`.

## Findings
- **Fault-aware mapping works.** On the same chips, `magnitude_remap` beats `magnitude` at every fault
  rate, pattern and model, by 1.3–13.9 points. At 0.5% stuck cells: BERT 76.6% → 85.6%,
  DistilBERT 77.9% → 85.1%. Gains are largest for clustered and line faults.
- **Joint mapping (`c1`) gives no reliable extra gain.** Against `magnitude_remap` it wins in some settings and loses in
  others (from −5.3 to +7.0 points), so choosing the SLC set per chip adds nothing over magnitude.
- **Hybrid vs all-SLC.** Up to 0.2% stuck cells `magnitude_remap` matches `all_slc_remap` with about
  34% fewer cells. From 0.5% upwards `all_slc_remap` is clearly more robust; tier-aware training (step 7) closes this gap.
