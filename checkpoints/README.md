# Checkpoints

Only the INT8 golden exports are committed here (docs/step1_decisions.md, D14):

- `resnet20_int8_seed0.pt`: for the PyTorch reference and simulator (`quant.export.load_int8`)
- `resnet20_int8_seed0.npz`: the same data for NumPy / PYNQ on the KV260 (`header_json` + `<layer>/w_int|w_scale|bias`)

FP32 and QAT checkpoints, and all larger models, live on the shared Drive under `MyDrive/reram/checkpoints`.
