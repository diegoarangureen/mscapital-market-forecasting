"""Shape, finite-value, and gradient smoke test for the synchronized TCN."""
import ast
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
PREPARER = PROJECT / "scripts" / "prepare_sync_tcn_v16.py"


class ConvBlock(nn.Module):
    def __init__(self, d_model, kernel=5, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, values):
        left = torch.cat([values[:, :1], values[:, :-1]], dim=1)
        right = torch.cat([values[:, 1:], values[:, -1:]], dim=1)
        local_average = (left + 2.0 * values + right) * 0.25
        return values + self.net(self.norm(local_average))


def load_architecture():
    tree = ast.parse(PREPARER.read_text(encoding="utf-8-sig"))
    architecture = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id == "ARCHITECTURE"
        and isinstance(node.value, ast.Constant)
    )
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "ConvBlock": ConvBlock,
        "MARKET_FEATURES": [f"m{i}" for i in range(12)],
        "TX_FEATURES": [f"t{i}" for i in range(8)],
        "ORDER_FEATURES": [f"o{i}" for i in range(10)],
        "MARKET_LEN": 200,
        "STATIC_FEATURE_COUNT": 379,
    }
    exec(architecture, namespace)
    return namespace["_JointMultiStreamStaticModel"]


def main():
    torch.manual_seed(7)
    model_class = load_architecture()
    model = model_class(d_model=32, dropout=0.0)
    market = torch.randn(3, 200, 12)
    transaction = torch.randn(3, 60, 8)
    order = torch.randn(3, 60, 10)
    static = torch.randn(3, 379)
    market[1, :150] = 0
    transaction[1, :35] = 0
    order[1, :35] = 0
    market[2] = 0
    transaction[2] = 0
    order[2] = 0
    output = model(market, transaction, order, static)
    assert output.shape == (3,), output.shape
    assert torch.isfinite(output).all(), output
    output.square().mean().backward()
    gradient_parameters = sum(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    assert gradient_parameters > 0
    print(f"sync_tcn_smoke_ok shape={tuple(output.shape)} finite_grad_params={gradient_parameters}")


if __name__ == "__main__":
    main()
