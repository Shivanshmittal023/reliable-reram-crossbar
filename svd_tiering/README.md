# SVD tiering

`kaggle_svd_tiering.py` prepares the two SST-2 transformers for the crossbar, following HyFlexPIM
(Song et al., ISCA 2025).

| Key | Hugging Face model | Factorised layers |
|---|---|---|
| `distilbert-sst2` | `distilbert-base-uncased-finetuned-sst-2-english` | 36 |
| `bert-base-sst2` | `textattack/bert-base-uncased-SST-2` | 72 |

For each model the script:
1. Evaluates the original model.
2. Factorises every encoder linear layer as W ≈ U · diag(σ) · Vᵀ, keeping k = ⌊mn/(m+n)⌋ components
   (384 for a 768×768 layer), so the parameter count stays the same.
3. Fine-tunes for 8 epochs to recover the truncation loss.
4. Saves the factorised model and the singular values (magnitude scores) used by `reram_sim`.

On the chip each layer becomes two crossbars, x → [σVᵀ] → [U] → y. Component r is row r of σVᵀ
together with column r of U.

## Run
Kaggle, GPU T4, Internet on. Paste the script into one cell and run it. The output is written to
`/kaggle/working/outputs/<model>/` and zipped.

## Results (SST-2 validation accuracy)

| Model | Original | SVD, before fine-tuning | SVD, after fine-tuning |
|---|---|---|---|
| BERT-Base | 92.43% | 89.33% | 90.14% |
| DistilBERT | 91.06% | 89.68% | 89.11% |

`outputs/<model>/` holds `summary.json`, `tier_map.json` (layer shapes and ranks) and `importance.csv`
(one row per component). The weight files (`svd_model.pt`, `importance.pt`) are produced by the run
and are not stored here. `logs/` contains the Kaggle run logs.
