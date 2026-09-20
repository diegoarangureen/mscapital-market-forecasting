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
path = ROOT / "data/interim/kaggle_kernels/multistream_factorized_transformer_dev/run_v24_loss_schedule.py"
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
assert weights == [0.65, 0.65, 0.65, 0.50, 0.35, 0.20]
model = namespace["_JointMultiStreamStaticModel"]()
inputs = [torch.randn(4, 200, 11), torch.randn(4, 60, 7),
          torch.randn(4, 60, 10), torch.randn(4, 379)]
target = torch.randn(4)
prediction = model(*inputs)
assert prediction.shape == (4,) and torch.isfinite(prediction).all()
loss = 0.8 * F.smooth_l1_loss(prediction, target) + 0.2 * namespace["cosine_loss"](prediction, target)
loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(parameter.grad).all()
    for parameter in model.parameters() if parameter.grad is not None)
notebook = json.loads(path.with_suffix(".ipynb").read_text(encoding="utf-8"))
assert "".join(notebook["cells"][0]["source"]) == path.read_text(encoding="utf-8")
print(json.dumps({"prediction_shape": list(prediction.shape), "finite_loss_and_gradients": True,
                  "cosine_weights": weights, "notebook_matches_script": True}))
