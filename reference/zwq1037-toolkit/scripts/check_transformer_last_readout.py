"""Check the generated notebook's final model and its scheduled loss on CPU."""

import ast
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data/interim/kaggle_kernels/multistream_factorized_transformer_dev/run_v25_last_readout.py"
tree = ast.parse(path.read_text(encoding="utf-8"))
names = {"ConvBlock", "_FactorizedStreamEncoder", "_JointMultiStreamStaticModel",
         "cosine_loss", "cosine_weight_for_epoch"}
nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
         and node.name in names]
namespace = {"torch": torch, "nn": nn, "F": F, "np": np,
             "MARKET_FEATURES": list(range(11)), "TX_FEATURES": list(range(7)),
             "ORDER_FEATURES": list(range(10)), "MARKET_LEN": 200,
             "FLOW_LEN": 60, "STATIC_FEATURE_COUNT": 379}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
weights = [namespace["cosine_weight_for_epoch"](epoch) for epoch in range(1, 7)]
assert weights == [0.65] * 6
model = namespace["_JointMultiStreamStaticModel"]()
inputs = [torch.randn(4, 200, 11), torch.randn(4, 60, 7),
          torch.randn(4, 60, 10), torch.randn(4, 379)]
target = torch.randn(4)
prediction = model(*inputs)
assert prediction.shape == (4,) and torch.isfinite(prediction).all()
loss = 0.35 * F.smooth_l1_loss(prediction, target) + 0.65 * namespace["cosine_loss"](prediction, target)
loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(parameter.grad).all()
    for parameter in model.parameters() if parameter.grad is not None)
notebook = json.loads(path.with_suffix(".ipynb").read_text(encoding="utf-8"))
assert "".join(notebook["cells"][0]["source"]) == path.read_text(encoding="utf-8")
print(json.dumps({"prediction_shape": list(prediction.shape), "finite_loss_and_gradients": True,
                  "cosine_weights": weights, "notebook_matches_script": True}))

# 精确验证末有效位置，覆盖内部缺口及尾部 padding。
# Verify the last valid position with internal gaps and trailing padding.
encoder = namespace["_FactorizedStreamEncoder"](96, 5, dropout=0.0)
encoder.projection = nn.Identity()
encoder.conv5 = nn.Identity()
encoder.conv3 = nn.Identity()
class IdentityEncoder(nn.Module):
    def forward(self, x, src_key_padding_mask=None): return x
encoder.encoder = IdentityEncoder()
class ZeroAttention(nn.Module):
    def forward(self, x): return torch.zeros(*x.shape[:2], 1)
encoder.attention = ZeroAttention()
encoder.position.data.zero_()
values = torch.zeros(2,5,96)
values[0,0]=1; values[0,2]=3
values[1,1]=2; values[1,3]=4
actual=encoder(values)
assert torch.allclose(actual[0], torch.full((96,), 0.8*2+0.2*3))
assert torch.allclose(actual[1], torch.full((96,), 0.8*3+0.2*4))
print("Last-valid readout and masked padding passed")
