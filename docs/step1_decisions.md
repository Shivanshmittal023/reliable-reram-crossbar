# Step 1: Baseline Models and Quantisation, Design Decisions

**Status:** Agreed baseline. Change a decision only by editing this file and telling the team.
**Scope:** Step 1 of the project plan: FP32 training, INT8 quantisation-aware training (QAT), and export of the INT8 reference models.

---

## Why this document exists

Step 1 is ordinary deep learning, but the choices made here are hard to change later:

- **Step 2** (the crossbar simulator) must reproduce the INT8 model *bit for bit*, so it needs an exact integer definition of the model.
- **Step 4** (non-idealities) and **Step 5** (mapping and training) all start from the checkpoints produced here.
- **Step 7** (bit-flip attack) attacks the integer weights defined here.
- **Step 8** (FPGA) implements the same arithmetic in hardware.

Each decision below gives the choice, the reason, and the later step it protects.

---

## Summary table

| # | Topic | Decision |
|---|---|---|
| D1 | First model | ResNet-20 (CIFAR variant, option-A shortcuts, 0.27M weights); other models later |
| D2 | Input pipeline | No mean/std normalisation; the image enters as raw UINT8 pixels (scale 1/255) |
| D3 | Weight format | Signed INT8 in **[-127, 127]**, symmetric (zero-point 0), **one scale per output channel** |
| D4 | Activation format | **Unsigned UINT8** [0, 255], symmetric (zero-point 0), one scale per tensor |
| D5 | Activation scales | Calibrated once (99.99th percentile), then **frozen** during QAT |
| D6 | BatchNorm | **Folded into conv weights and bias** after FP32 training, *before* QAT |
| D7 | What goes on the crossbar | **Every conv and FC layer** (all 20 in ResNet-20), including the first and last |
| D8 | What stays digital | Bias, scaling, residual add, ReLU, pooling, requantisation, softmax |
| D9 | Quantisation points | One UINT8 quantiser on the input of every crossbar layer; block outputs are stored as UINT8 |
| D10 | Golden model | The **integer-path model** (`eval_int8.py`), *not* the fake-quant training model |
| D11 | Exact arithmetic | Accumulators computed in float64 (or int64); TF32 disabled everywhere |
| D12 | Rounding | `torch.round` (round half to even) everywhere; clamp after rounding |
| D13 | Weight-to-crossbar layout | Conv weight `[out, in, kh, kw]` is flattened to the matrix `[in·kh·kw, out]`; rows are inputs, columns are output channels |
| D14 | Export format | `.pt` (PyTorch) plus `.npz` (NumPy, for PYNQ), with a per-layer manifest |
| D15 | Data | Train on the full 50k CIFAR-10 training set; report on the 10k test set; calibrate on training data |
| D16 | Reproducibility | Seed 0 for the golden models; deterministic flags on; git hash and versions saved in every checkpoint |
| D17 | Transformer track | Provisional: INT8 encoder linear layers on the crossbar, attention matmuls digital, signed activations |

---

## D1. Start with ResNet-20 only

**Decision:** ResNet-20 for CIFAR-10 (He et al. 2016): 3 stages of 16/32/64 channels, 3 basic blocks per stage, **option-A shortcuts**. A downsampling shortcut is a stride-2 subsample plus zero-padding of the extra channels, so it has no parameters. ResNet-18 (CIFAR variant), VGG-16-BN and BERT follow the same rules later.

**Why:**
- It's small (0.27M weights, matching the plan's FPGA budget), and FP32 training on Colab takes about an hour.
- Option-A shortcuts have no parameters, so every weight lives in a 3×3 conv or the FC layer, and there is one less layer type to map.
- The same model is used in Steps 2 to 8, so getting it right once pays off everywhere.

**Target:** FP32 ≈ 91.5–92.5% test accuracy.

---

## D2. Feed raw pixels, with no mean/std normalisation

**Decision:** Input = `ToTensor()` only, so pixels are `k/255` for an integer `k` in [0, 255]. No `Normalize(mean, std)`. Augmentation: random crop 32 with padding 4, plus horizontal flip.

**Why:**
- The input is then **already exact UINT8** with scale 1/255, so the first conv can sit on the crossbar like every other layer (see D7). There is no special case for a signed first layer.
- Folding normalisation into conv1 instead breaks zero-padding: a padded 0 in normalised space is not pixel 0. Skipping normalisation avoids that.
- BatchNorm right after conv1 makes normalisation unnecessary for accuracy; the expected cost is under 0.1%.

---

## D3. Weights: signed INT8 [-127, 127], symmetric, per output channel

**Decision:**
```
s_w[c] = max(|W[c, ...]|) / 127            # one scale per output channel c
w_int  = clamp(round(W / s_w[c]), -127, 127)
```
We never use -128.

**Why:**
- **Symmetric range:** with differential pairs (Step 2), the weight is `G⁺ − G⁻`. A symmetric range means a 7-bit magnitude plus a sign, with no odd extra code. It also keeps the bit-flip attack (Step 7) and the stuck-at analysis free of edge cases.
- **Zero-point 0:** a weight of 0 is integer 0, which is a cell pair at the same conductance. It needs no offset correction in the periphery.
- **Per output channel:** one output channel is one crossbar column, and the column's ADC result is scaled digitally anyway. A per-column scale is therefore free in hardware and gives better accuracy than one scale per layer.
- **Max-abs (no clipping):** at 8 bits, max-abs is near lossless. Weight *clipping* belongs to noise-aware training (Step 5a, AIHWKIT recipe) and will be added there, not here.

**Left open on purpose:** the bit encoding (differential magnitude with 7 slices in SLC / 4 in MLC, or offset encoding) is a Step 2 decision. The [-127, 127] range supports both.

---

## D4. Activations: unsigned UINT8, symmetric, per tensor

**Decision:**
```
x_int = clamp(round(x / s_x), 0, 255)       # one s_x per quantisation point
```

**Why:**
- **Unsigned:** Step 2 applies inputs **one bit per cycle** (bit-serial). With unsigned inputs every bit has positive weight 2^t, the simplest and most common scheme (ISAAC). Every crossbar input in ResNet-20 is non-negative: the pixels (D2), ReLU outputs and the average-pool output.
- **Zero-point 0:** conv zero-padding is then exactly integer 0, so im2col in the simulator and the FPGA needs no special padding value.
- **Per tensor, not per channel:** each crossbar row receives one input. A per-row input scale would need a separate digital multiply per row before the DAC. A per-tensor scale is one multiply after the ADC.
- **8 bits:** this gives 8 input cycles. Step 4 can sweep lower input precision, which needs retraining at that precision.

---

## D5. Activation scales: calibrate, then freeze

**Decision:**
1. Run 20 training batches (no augmentation) through the folded FP32 model.
2. Set each `s_x = percentile_99.99(x) / 255`.
3. Freeze all `s_x` for the whole of QAT. Weight scales (D3) are recomputed every forward pass during QAT and frozen at export.

**Why:**
- At 8 bits a calibrated, frozen scale loses almost nothing. Learned step sizes (LSQ/PACT) add hyperparameters and instability for no gain.
- Frozen scales make the training model and the exported model agree by construction.
- **Later:** if Step 4 sweeps 4-bit inputs, switch that experiment to LSQ. The `ActQuant` module should allow a learnable scale, but defaults to frozen.

---

## D6. Fold BatchNorm before QAT

**Decision:**
1. Train FP32 with BatchNorm.
2. Fold each BN into the preceding conv using the running statistics:
   ```
   W' = W · γ / √(σ² + ε)          (per output channel)
   b' = β − γ·μ / √(σ² + ε)
   ```
3. Run QAT on the folded model (no BN layers left), and quantise `W'`.

**Why:**
- A crossbar stores only W. BN must become part of W (scale) and the digital bias, so we must quantise **the weights the hardware actually holds**.
- Fold-then-QAT is simpler than simulating BN folding during training (Jacob et al. 2018), and works well when fine-tuning from a converged model.
- **Unit test:** the folded FP32 model must give the same logits as the unfolded one (within 1e-5).

---

## D7. Every conv and FC layer goes on the crossbar

**Decision:** all 19 conv layers and the final FC layer of ResNet-20 are crossbar layers, and all are quantised to INT8.

**Why:**
- Many papers keep the first and last layers digital and at high precision. That makes the accuracy-under-noise numbers look better, but it hides exactly the layers Step 4 should measure. With D2 and D4 there is no technical reason to exclude them.
- If we later want the "first/last layer digital" setting as an ablation, it is a config flag, `digital_layers: [conv1, fc]`. Running that ablation is cheap; adding crossbar layers back later is not.

---

## D8. What stays digital (the "periphery")

**Decision:** these operations run digitally in **float32**, using shared code in `quant/periphery.py` and the model's `forward`. Two exceptions are computed in float64 and rounded to float32 once: the dequantisation of the accumulator (`acc` can exceed float32's exact range, see D11) and the average pool (so the result doesn't depend on device or reduction order).

| Operation | Where |
|---|---|
| Dequantise accumulator: `acc · s_x · s_w[c]` | after every crossbar layer |
| Add bias `b'[c]` | after every crossbar layer |
| Residual add (including option-A zero-pad/subsample) | end of each block |
| ReLU | after BN-folded conv and after the residual add |
| Global average pool (8×8) | before FC |
| Requantise to UINT8 (D4) | before the next crossbar layer |
| Softmax / argmax | output |

**Why:**
- The reference model (D10) and the simulator (Step 2) are **the same model**. In `int` mode each `QConv2d` / `QLinear` calls `mac_backend(layer, x_int)` if one is set, otherwise the exact MAC. The Step 2 simulator is just a different `mac_backend`, so bit-exactness is a test of the MAC alone.
- float32 is simple. For Step 8, the periphery can run on the KV260's ARM cores (the same NumPy code), or be converted to fixed-point (`M · 2^-n` multipliers) in one place without touching anything else.

---

## D9. Where the quantisers sit

**Decision:** the dataflow of one ResNet-20 basic block:

```
x (UINT8, scale s_in)
 ├─► conv1 [crossbar] ─► ×s,+b ─► ReLU ─► Q (UINT8) ─► conv2 [crossbar] ─► ×s,+b ─┐
 └──────────────── shortcut (dequant x; option-A pad/subsample) ─────────────────► (+) ─► ReLU ─► Q (UINT8) ─► next block
```

- The image enters as UINT8 (D2).
- Each block output is quantised **once**. That UINT8 tensor is both the next conv's input and the next residual.
- After average pooling, one more quantiser feeds the FC layer.

**Why:** on real hardware, activations between layers sit in 8-bit buffers. Quantising once per buffer is realistic, keeps the residual path consistent with what the hardware stores, and gives every crossbar layer exactly one input scale.

---

## D10. The golden model is the integer-path model

**Decision:** there are two INT8 versions of each model:

| Version | Used for | Arithmetic |
|---|---|---|
| Fake-quant model (`QConv2d` / `QLinear` in training mode) | QAT only | float: `conv(x_q·s_x, w_q·s_w)` |
| **Integer reference** (`eval_int8.py`) | **All reporting and Step 2 verification** | `acc = Σ x_int·w_int` (exact) → periphery (D8) |

**Why:** the two are mathematically equal, but float rounding in the fake-quant path can very rarely push a value across a rounding boundary at the next quantiser. The integer path has one unambiguous answer, so it is the one the simulator and the FPGA must match. The two versions should agree in accuracy within 0.05%.

---

## D11. Exact integer arithmetic: float64 accumulators, no TF32

**Decision:**
- The integer reference computes `acc` with `conv2d` / `matmul` on **float64** tensors that hold integer values, then casts to int64. An im2col + int64 matmul on CPU is an acceptable alternative.
- At the top of every script:
  ```python
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  ```
- No mixed precision (AMP) in QAT or in evaluation. AMP is allowed only for FP32 *training*.

**Why:**
- float32 is exact for integers only up to 2²⁴ ≈ 16.8M. The worst-case ResNet-20 accumulator is 576 × 255 × 127 ≈ **18.7M**, already above that. For ResNet-18 it is about 150M.
- TF32 (on by default on the A100/L4 GPUs that Colab sometimes assigns) keeps only 10 mantissa bits, which would silently break bit-exactness.
- float64 is exact up to 2⁵³, far beyond any accumulator here.

---

## D12. Rounding convention

**Decision:** use `torch.round` (round half to even) for every quantiser, then clamp. NumPy's `np.round` behaves the same, so PYNQ code matches. Any HLS code in Step 8 must implement round-half-to-even explicitly.

**Why:** one convention everywhere removes a whole class of off-by-one mismatches between the reference, the simulator and the FPGA.

---

## D13. How a layer becomes a crossbar matrix

**Decision:**
```
W_xbar = w_int.reshape(out_ch, in_ch * kh * kw).T      # shape [rows = in·kh·kw, cols = out_ch]
```
The row order is PyTorch's flatten order `(in_ch, kh, kw)`, which matches `torch.nn.functional.unfold` (im2col). FC weights `[out, in]` become `[in, out]`.

**Why:** fixing this now means Step 2 tiling (128×128), the Step 5 row/column permutations and fault maps, and the Step 8 weight loader all refer to the same row and column indices.

**Sizes for ResNet-20:** the largest layer is 576×64, which is 5 row-tiles of 128 (the last one partly filled). The FC layer is 64×10.

---

## D14. Export format

**Decision:** `quant/export.py` (called at the end of `scripts/calibrate_and_qat.py`) writes `resnet20_int8_seed0.pt` and `resnet20_int8_seed0.npz` containing:

```
format_version, arch
meta:       seed, git_commit, torch_version, date, fp32_acc, folded_acc, ptq_acc,
            qat_fakequant_acc, int8_acc
act_scales: {quantiser name: scale}  e.g. "q_in": 1/255, "layer1.0.q1": ...
layers:     ordered list, one entry per crossbar layer (forward order):
    name        "layer1.0.conv1"
    type        "conv" | "linear"
    w_int       int8   [out, in, kh, kw]   (or [out, in])
    w_scale     float32 [out]
    bias        float32 [out]
    in_scale    float32 scalar   (scale of this layer's UINT8 input)
    in_quant    name of the quantiser that produced the input
    stride, padding
```

The graph (residuals, option-A shortcuts) is implied by `arch` and defined once in `models/<arch>.py` (`forward` and `crossbar_layers()`). In the `.npz`, arrays are stored as `<layer>/w_int`, `<layer>/w_scale` and `<layer>/bias`, and everything else is a JSON string in `header_json`.

**Why:**
- `.pt` is for the PyTorch simulator. `.npz` is readable on the KV260's ARM/PYNQ without PyTorch.
- `w_int` is the single source of truth: Steps 2, 4, 5, 7 and 8 all load from it.

**Storage:** the ResNet-20 INT8 export (about 0.3 MB) **is committed to git** as `checkpoints/resnet20_int8_seed0.*`, so every teammate uses the identical golden model. FP32 checkpoints and the large models (ResNet-18, VGG-16, BERT) live on the shared Google Drive.

---

## D15. Data splits

**Decision:**
- Train on all 50k CIFAR-10 training images.
- Report on the 10k test set.
- Calibration (D5) uses training images.
- We do **no** hyperparameter tuning, so no validation split is needed. The recipes below are standard.

**Why:** our baseline numbers stay directly comparable with published CIFAR-10 numbers and with HyFlexPIM. Later steps that need data, such as gradient importance (Step 5b) and the bit-flip attack batch (Step 7), will state which subset they use.

---

## D16. Reproducibility

**Decision:**
- The golden models use **seed 0**.
- `torch.backends.cudnn.deterministic = True` and `benchmark = False` for QAT and evaluation.
- Every checkpoint stores its git commit, the torch version and its accuracy.
- Seeds for simulated chips (Step 4) are a separate seed space and are **never** reused for training.

**Why:** "20 simulated chips" only means something if everyone starts from the same weights and can replay a chip by its seed.

---

## D17. Transformer track (provisional)

Read the HyFlexPIM code and align with its quantisation before finalising this section.

- **Model:** `bert-base-uncased`, fine-tuned on SST-2 and MRPC (or the HyFlexPIM checkpoints, if provided).
- **On the crossbar:** encoder linear layers Q, K, V, O, FFN1 and FFN2, with INT8 weights per output channel (same as D3).
- **Digital:** embeddings, LayerNorm, softmax, GELU, and the two attention matmuls (QKᵀ and attention·V). Those multiply activations by activations, so there are no stored weights to put on a crossbar.
- **Difference from the CNNs:** LayerNorm outputs are **signed**, so Transformer inputs are signed INT8 [-127, 127]. The bit-serial input then uses two's complement, with the MSB cycle weighted −2⁷. The Step 2 simulator must support both signed and unsigned inputs.

---

## Training recipes

| | FP32 (ResNet-20) | QAT (ResNet-20) |
|---|---|---|
| Starting point | random init | folded FP32 checkpoint (D6) |
| Optimiser | SGD, momentum 0.9, Nesterov | SGD, momentum 0.9 |
| Learning rate | 0.1, cosine to 0 | 1e-3, cosine to 0 |
| Weight decay | 5e-4 | 5e-5 |
| Epochs | 200 | 30 |
| Batch size | 128 | 128 |
| Augmentation | crop 32 / pad 4, flip | same |
| Precision | AMP allowed | float32, TF32 off |
| Expected test accuracy | 91.5–92.5% | within 0.5% of FP32 |

Gradients pass through rounding with the straight-through estimator (STE): `x + (round(x) − x).detach()`.

---

## Code layout for Step 1

```
reliable-reram-crossbar/
├── models/
│   ├── __init__.py             build_model(arch, folded)
│   └── resnet_cifar.py         ResNet-20 (option A), quantiser placement (D9), crossbar_layers()
├── quant/
│   ├── quantizers.py           weight quantiser (D3), ActQuant (D4, D5), STE, rounding (D12)
│   ├── qlayers.py              QConv2d, QLinear: modes fp / fakequant / int, mac_backend hook
│   ├── periphery.py            D8; shared with the Step 2 simulator
│   ├── fold_bn.py              D6
│   ├── prepare.py              set_mode, calibrate, freeze_int
│   └── export.py               D13, D14: export_int8, load_int8, build_int_model
├── utils/common.py             seeds and TF32 (D11, D16), CIFAR-10 loaders (D2, D15), train/eval, checkpoints
├── scripts/
│   ├── train_fp32.py           Step 1a (resumes automatically)
│   ├── calibrate_and_qat.py    Step 1b: fold, calibrate, QAT, export, evaluate the integer reference
│   └── eval_int8.py            re-evaluate a golden export (D10)
├── tests/
│   ├── test_quantizers.py      ranges, symmetry, STE, exact pixel input
│   ├── test_fold_bn.py         folded == unfolded logits
│   ├── test_int_reference.py   int == fake-quant, deterministic, float64 MAC == int64 im2col, backend hook
│   └── test_export_roundtrip.py
├── checkpoints/                ResNet-20 INT8 golden export (committed)
├── notebooks/colab_runner.ipynb
└── results/baseline_accuracy.csv
```

Run everything from the repo root: `python -m scripts.<name>` and `python -m pytest`.

---

## Definition of done for Step 1 (CNN track)

- [ ] ResNet-20 FP32 reaches ≥ 91.5% on the CIFAR-10 test set.
- [ ] The folded model matches the unfolded model (`test_fold_bn` passes).
- [ ] The INT8 integer reference is within 0.5% of FP32.
- [ ] The integer reference and the fake-quant model agree within 0.05%, and the integer reference gives identical logits on two runs.
- [ ] The golden export `.pt` and `.npz` are committed, and `results/baseline_accuracy.csv` is filled in.
- [ ] Every teammate can reproduce the INT8 accuracy from the committed export on Colab with one notebook cell.

## Explicitly *not* part of Step 1

No bit-slicing, no ADC, no noise, no SVD, no mapping. All of those are Step 2 onward, and they build on the formats fixed here.
