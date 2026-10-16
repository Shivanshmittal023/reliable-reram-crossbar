import json

import numpy as np
import torch

from quant.export import build_int_model, export_int8, load_int8
from tests.conftest import random_images


def test_export_roundtrip_is_bit_identical(int_model, tmp_path):
    stem = str(tmp_path / "resnet20_int8_test")
    export_int8(int_model, "resnet20", {"seed": 0}, stem)
    loaded = build_int_model(load_int8(stem + ".pt"))
    x = random_images(16, seed=6)
    with torch.no_grad():
        assert torch.equal(int_model(x), loaded(x))


def test_export_contents(int_model, tmp_path):
    stem = str(tmp_path / "e")
    blob = export_int8(int_model, "resnet20", {}, stem)
    assert len(blob["layers"]) == 20                          # 19 conv + fc (D7)
    for L in blob["layers"]:
        assert L["w_int"].dtype == torch.int8
        assert L["in_scale"] == blob["act_scales"][L["in_quant"]]
    assert abs(blob["act_scales"]["q_in"] - 1 / 255) < 1e-9   # D2

    npz = np.load(stem + ".npz")
    header = json.loads(str(npz["header_json"]))
    assert [L["name"] for L in header["layers"]] == [L["name"] for L in blob["layers"]]
    for L in blob["layers"]:
        assert np.array_equal(npz[f"{L['name']}/w_int"], L["w_int"].numpy())
