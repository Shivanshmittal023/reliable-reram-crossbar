# Reliable and Secure ReRAM Crossbar Accelerators for DNNs

CS5290 Memory-Centric Computer Architecture course project: a noise-, fault- and attack-aware hybrid SLC/MLC in-memory computing study, built on HyFlexPIM (ISCA 2025).

## Status

- **Step 1: baseline models and INT8 quantisation.** Code is done; the decisions are in [docs/step1_decisions.md](docs/step1_decisions.md).
- Step 2 (bit-accurate crossbar simulator) plugs into the `mac_backend` hook of `quant/qlayers.py`.

## Quick start

On Colab, open `notebooks/colab_runner.ipynb` with a GPU runtime and run all cells.

Locally:

```bash
pip install -r requirements.txt
python -m pytest -q                                   # ~3 s, CPU, no dataset needed
python -m scripts.train_fp32 --amp                    # FP32 ResNet-20 (200 epochs)
python -m scripts.calibrate_and_qat --fp32 checkpoints/resnet20_fp32_seed0.pt
python -m scripts.eval_int8 --export checkpoints/resnet20_int8_seed0.pt
```

Run all commands from the repo root.

Using the golden model in later steps:

```python
from quant.export import load_int8, build_int_model
model = build_int_model(load_int8("checkpoints/resnet20_int8_seed0.pt"))   # integer reference
for name, layer, _ in model.crossbar_layers():
    W = layer.crossbar_matrix()          # int8 [rows = in*kh*kw, cols = out]
```
