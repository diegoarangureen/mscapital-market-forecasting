"""Exercise window boundaries, empty streams, and gradients without Kaggle data."""
import ast
import warnings
from pathlib import Path

import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "data/interim/kaggle_kernels/multistream_factorized_transformer_dev/run_v13_multiwindow10.py"

torch.set_num_threads(2)
warnings.filterwarnings("ignore", message="enable_nested_tensor.*")
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
selected = []
for name in ("ConvBlock", "_FactorizedStreamEncoder", "_JointMultiStreamStaticModel"):
    selected.append([node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name][-1])
scope = {"torch": torch, "nn": nn, "MARKET_LEN": 200, "FLOW_LEN": 60, "MARKET_SECONDS": 600.0, "FLOW_SECONDS": 60.0, "MARKET_FEATURES": range(11), "TX_FEATURES": range(7), "ORDER_FEATURES": range(10), "STATIC_FEATURE_COUNT": 379}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), scope)
model = scope["_JointMultiStreamStaticModel"]().cuda().eval()
assert model.market.window_mask.sum(1).tolist() == [200, 20, 5]
assert model.transaction.window_mask.sum(1).tolist() == [60, 15, 5]
inputs = [torch.randn(3, 200, 11, device="cuda"), torch.zeros(3, 60, 7, device="cuda"), torch.zeros(3, 60, 10, device="cuda"), torch.randn(3, 379, device="cuda")]
inputs[0][0] = 0
inputs[1][1, -1] = 1
inputs[2][2, -6] = 1
pooled, missing = model.order(inputs[2])
assert missing[2].tolist() == [False, False, True]
assert torch.equal(pooled[2, 2], torch.zeros_like(pooled[2, 2]))
prediction = model(*inputs)
assert prediction.shape == (3,) and torch.isfinite(prediction).all()
prediction.square().mean().backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
print("PASS: window lengths, boundary exclusion, empty streams, output shape, finite gradients")
