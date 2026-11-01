# MNLI: second dataset

`kaggle_mnli_pipeline.py` repeats the core of the SST-2 study on MNLI in one Kaggle cell.

| Stage | What | Chips |
|---|---|---|
| 1 | SVD (k = floor(mn/(m+n))) + recovery fine-tune, 100k train pairs, 2 epochs | - |
| 2 | Ranking scores: magnitude, gradient (HyFlexPIM batch), fisher | - |
| 3 | `slc_share`: SLC share 5/10/20/30/50 % at sigma = 5 % | 20 per point |
| 4 | `fault`: stuck-at rate 0-5 %, 20 % SLC | 20 per point |
| 5 | `mapping`: all_slc(_remap), magnitude(_remap), c1 at fault 0.2-2 % | 10 per point |

IR drop, ADC, fault patterns and other noise levels are skipped.

Models: `typeform/distilbert-base-uncased-mnli` and `textattack/bert-base-uncased-MNLI`.
The script measures each model's label order on the evaluation subset and maps the GLUE labels into it.
The permutation is logged and saved as `label_perm` in `svd/<model>/summary.json`.

Evaluation uses a fixed random 2,000-pair subset of `validation_matched` (shuffle seed 0).
The digital baseline and the fine-tuned SVD model are also scored on all 9,815 pairs.

## Run

1. Kaggle notebook, GPU T4, Internet on. Paste the file into one cell.
2. Set `MODE = "smoke"` and run it (about 5-10 min). Check that the self-tests pass and the label permutation looks right.
3. Set `MODE = "full"` and use Save & Run All.
4. Download `mnli_results.zip` from the Output panel.

If the session stops (time budget of 11 h, or a crash), add the notebook's output as input to a new run and run again.
Finished stages and measured chips are kept and skipped.

## Output

`mnli_results/`
- `svd/<model>/summary.json`, `tier_map.json`, `svd_model.pt` (weights, not in the zip)
- `scores/<model>_scores.pt`
- `slc_share/<model>/`, `fault/<model>/`, `mapping/<model>/`: `sweep.csv` (one row per chip) and `sweep_summary.csv`
- `log.txt`, `stage_times.csv`

## Runtime estimate (T4)

MNLI pairs are about twice as long as SST-2 sentences after length sorting, so one chip takes about 4 s (DistilBERT) or 8 s (BERT-Base).

| Stage | DistilBERT | BERT-Base |
|---|---|---|
| SVD + fine-tune (6,250 steps) | ~19 min | ~32 min |
| Scores | ~2 min | ~4 min |
| slc_share (440 chips) | ~29 min | ~59 min |
| fault (726 chips) | ~48 min | ~97 min |
| mapping (200 chips) | ~22 min | ~45 min |
| Total | ~2 h | ~4 h |

Expected total is about 6 h (range 4.5-9 h). This fits one 12 h session.
