# Step 1: SVD tiering

**Report:** Section IV-A (factorisation and fine-tuning) and VIII-A (cost of SVD).

`svd_tiering.py` prepares the two SST-2 transformers for the crossbar, following HyFlexPIM
(Song et al., ISCA 2025). Its Kaggle notebook output is the input of steps 2–7.

| Key | Hugging Face model | Factorised layers |
|---|---|---|
| `distilbert-sst2` | `distilbert-base-uncased-finetuned-sst-2-english` | 36 |
| `bert-base-sst2` | `textattack/bert-base-uncased-SST-2` | 72 |

For each model the script:
1. Evaluates the original model.
2. Factorises every encoder linear layer as W ≈ U · diag(σ) · Vᵀ, keeping k = ⌊mn/(m+n)⌋ components
   (384 for a 768×768 layer, 614 for a 768×3072 layer), so the parameter count stays the same.
3. Fine-tunes for 8 epochs on the full SST-2 training set (AdamW, learning rate 2×10⁻⁵, batch 32) to
   recover the truncation loss.
4. Saves the factorised model and the singular values, which are the magnitude scores.

On the chip each layer becomes two crossbars, x → [σVᵀ] → [U] → y. Component r is row r of σVᵀ
together with column r of U.

## Run
Kaggle, GPU T4, Internet on. Paste the script into one cell and run it (about 50 min for both models).
The output is written to `/kaggle/working/outputs/<model>/` and zipped.

## Results
SST-2 validation accuracy (%):

| Model | Original FP32 | SVD, before fine-tuning | SVD, after fine-tuning |
|---|---|---|---|
| BERT-Base | 92.43 | 89.33 | 90.14 |
| DistilBERT | 91.06 | 89.68 | 89.11 |

- `results/outputs/<model>/`: `summary.json` (accuracies, settings), `tier_map.json` (layer shapes and
  ranks), `importance.csv` (one row per component). The weight files (`svd_model.pt`, `importance.pt`)
  are produced by the run and are not stored.
- `results/logs/`: the Kaggle run logs.
