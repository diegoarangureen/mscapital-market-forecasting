"""Verify that v27 StrictBaseModel can strictly load EXP020-style checkpoints."""

from __future__ import annotations

import ast
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/interim/kaggle_kernels/multistream_factorized_transformer_dev/run_v27_market_conditioned_event_residual_full.py"
LOCAL_STATE = ROOT / "data/interim/tree_experiments/EXP-TRANSFORMER-020-FACTORIZED-EMA/train059_valid6270_ex66/raw_best_state.pt"

tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
wanted_classes = {"ConvBlock", "_FactorizedStreamEncoder", "StrictBaseModel"}
wanted_functions = {"canonical_state"}
selected = []
seen_classes = set()
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name in wanted_classes and node.name not in seen_classes:
        selected.append(node)
        seen_classes.add(node.name)
    elif isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
        selected.append(node)

namespace = {
    "torch": torch,
    "nn": nn,
    "MARKET_FEATURES": [None] * 11,
    "TX_FEATURES": [None] * 7,
    "ORDER_FEATURES": [None] * 10,
    "MARKET_LEN": 200,
    "FLOW_LEN": 60,
    "STATIC_FEATURE_COUNT": 379,
}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)

state = torch.load(LOCAL_STATE, map_location="cpu", weights_only=False)
model = namespace["StrictBaseModel"]()
incompatible = model.load_state_dict(namespace["canonical_state"](state), strict=True)
wrapped = {"module." + key: value for key, value in state.items()}
model.load_state_dict(namespace["canonical_state"](wrapped), strict=True)

parameter_count = sum(parameter.numel() for parameter in model.parameters())
print(
    {
        "status": "passed",
        "selected_classes": sorted(seen_classes),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "parameter_count": parameter_count,
        "plain_checkpoint_keys": len(state),
        "dataparallel_checkpoint_keys": len(wrapped),
    }
)

