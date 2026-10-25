# Mapping techniques

A mapping decides where each SVD component is physically stored on the chip: in which tier (SLC or
MLC) and in which slot (row) of that tier. The placements in `reram_sim/` fix the tier by a score and
keep the components in index order. The techniques here improve on that, each in its own subfolder
with one Kaggle script, a README and its results.

All techniques use the same simulator, the `svd_tiering` notebook output as input, and the same result
format as `reram_sim/results/` (`results/<experiment>/<model>/sweep.csv` and `sweep_summary.csv`).

| Folder | Technique | Main result |
|---|---|---|
| [`fault_aware/`](fault_aware/) | Fault-aware mapping: slot order chosen around the chip's known stuck cells; C1: tier and slot chosen together per chip | Remapping recovers most of the stuck-cell loss (BERT, 0.5% stuck: 76.6% → 85.6%); C1 adds no reliable gain |
