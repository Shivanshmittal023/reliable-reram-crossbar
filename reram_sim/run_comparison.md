# Run 1 vs run 2 of `kaggle/kaggle_reram_sweep.py`

Run 1 is `logs.txt` (commit before `e09b5dd`, 57.9 min, experiments A-D).
Run 2 is `logs2.txt` / `reram_sim2.zip` (commit `e09b5dd`, 82.3 min, experiments A-E).
Run 2 results are written up in `results.txt`. Both runs: T4, SST-2 validation (872 sentences),
mean over 5 simulated chips. Digital SVD reference: DistilBERT 0.8911, BERT-base 0.9014.

`reram_sim2.zip` holds the raw run 2 output (per-experiment CSV, summary CSV, PNG, `config.json`,
`importance_scores.pt`, `run.log`). Its `run.log` matches `logs2.txt` line for line.

## Unchanged

`all_slc`, `all_mlc` and `random` are identical to 4 decimals in every table.
Run 1 `magnitude` is the same policy as run 2 `rank` (renamed); its numbers match exactly.

## Changed: `gradient` policy

Run 2 `gradient` scores each singular value by the mean per-batch |dL/dsigma|.
Run 1 `gradient` used |sum of the gradient|, which cancels near the fine-tuned minimum.
Run 2 keeps the old score as `gradient_net`; it matches run 1 `gradient` exactly.

| Result                        | Run 1 `gradient` | Run 2 `gradient` |
|-------------------------------|------------------|------------------|
| A, DistilBERT, 20% SLC        | 0.8908           | 0.8869           |
| B, DistilBERT, 20% SLC        | 0.6846           | 0.7433           |
| B, DistilBERT, 50% SLC        | 0.8037           | 0.8275           |
| B, BERT-base, 20% SLC         | 0.6482           | 0.6498           |
| B, BERT-base, 50% SLC         | 0.6995           | 0.7005           |
| D, DistilBERT, 0.5% faults    | 0.7142           | 0.7300           |
| D, BERT-base, 0.5% faults     | 0.5101           | 0.6190           |

- A: differences are within chip-to-chip noise (about +-0.5 points).
- B: DistilBERT gains about 6 points at 20% SLC; BERT-base is unchanged.
  `rank` still beats `gradient` at every SLC rate on both models.
- C: `gradient` improves at high sigma but stays below `rank`.
- D: `gradient` is better at 0.5% faults on both models.

## Changed: experiment setup

- Policies: `magnitude` renamed `rank`; `gradient_net` added to B and C.
- D fault grid: run 1 used 0, 0.1, 0.5, 1, 2, 5, 10, 20%. Run 2 uses 0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5%.
  Every policy is at chance beyond 5% in both runs, so nothing is lost.
- New experiment E: share of stuck cells at G_max (SA1 share 0, 0.5, 1.0 at 0.2% and 1% faults).
  With SA1 share 0 every policy stays within about 1 point of digital, even `all_mlc` at 1% faults.
  The damage comes from cells stuck at G_max.
- Run time grows from 57.9 to 82.3 min because of E, the extra policy and the calibration step.
