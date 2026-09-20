"""Fast local smoke checks for the predictive-pretraining Kaggle experiment."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT = Path(__file__).resolve().parents[1]
PREPARE_PATH = PROJECT / "scripts/prepare_predictive_pretrain_v18.py"
CACHE = (
    PROJECT
    / "data/interim/kaggle_kernels/multistream_transformer_dev/output_v1/transformer_cnn_grid"
)


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
        return values + self.net(self.norm(values))


def load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_v18", PREPARE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def architecture_smoke(module):
    namespace = {
        "torch": torch,
        "nn": nn,
        "ConvBlock": ConvBlock,
        "MARKET_FEATURES": list(range(11)),
        "TX_FEATURES": list(range(7)),
        "ORDER_FEATURES": list(range(10)),
        "MARKET_LEN": 200,
        "FLOW_LEN": 60,
        "STATIC_FEATURE_COUNT": 379,
    }
    exec(module.ARCHITECTURE, namespace)
    model = namespace["_JointMultiStreamStaticModel"](
        d_model=24, nhead=4, nlayers=1, dropout=0.0
    )
    market = torch.randn(2, 200, 11)
    transaction = torch.randn(2, 60, 7)
    order = torch.randn(2, 60, 10)
    static = torch.randn(2, 379)
    market[:, -5:] = 0.0
    transaction[:, -15:] = 0.0
    order[:, -15:] = 0.0
    pretrain_output = model(
        market, transaction, order, static=None, pretrain=True
    )
    main_output = model(market, transaction, order, static=static)
    (pretrain_output.square().mean() + main_output.square().mean()).backward()
    assert pretrain_output.shape == (2,)
    assert main_output.shape == (2,)
    assert torch.isfinite(pretrain_output).all()
    assert torch.isfinite(main_output).all()
    print("architecture_smoke=passed")


def target_smoke():
    label = pd.read_feather(PROJECT / "data/raw/label.feather")
    month_column = "month" if "month" in label.columns else "month_id"
    months = label[month_column].to_numpy()
    n_rows = len(label)
    count = np.memmap(
        CACHE / "train_v2_market_count_200.mmap",
        dtype=np.float16,
        mode="r",
        shape=(n_rows, 200),
    )
    market = np.memmap(
        CACHE / "train_v2_market_200x11.mmap",
        dtype=np.float16,
        mode="r",
        shape=(n_rows, 200, 11),
    )
    candidates = np.flatnonzero(months <= 59)
    if candidates.size > 100_000:
        positions = np.linspace(0, candidates.size - 1, 100_000, dtype=np.int64)
        candidates = candidates[positions]
    counts = np.asarray(count[candidates], dtype=np.float32)
    visible = counts[:, :195] > 0
    hidden = counts[:, 195:] > 0
    valid = visible.any(axis=1) & hidden.any(axis=1)
    rows = candidates[valid]
    visible_last = 194 - np.argmax(visible[valid, ::-1], axis=1)
    hidden_last = 199 - np.argmax(hidden[valid, ::-1], axis=1)
    start_mid = np.asarray(market[rows, visible_last, 0], dtype=np.float32) + 1.0
    end_mid = np.asarray(market[rows, hidden_last, 0], dtype=np.float32) + 1.0
    endpoint_valid = (start_mid > 0.1) & (end_mid > 0.1)
    hidden_return = end_mid[endpoint_valid] / start_mid[endpoint_valid] - 1.0
    hidden_return = hidden_return[
        np.isfinite(hidden_return) & (np.abs(hidden_return) <= 0.10)
    ]
    final_coverage = hidden_return.size / candidates.size
    assert final_coverage > 0.80
    assert np.isfinite(hidden_return).all()
    assert float(np.std(hidden_return)) > 1e-7
    print(
        "target_smoke=passed "
        f"quote_coverage={valid.mean():.6f} "
        f"final_coverage={final_coverage:.6f} "
        f"mean={hidden_return.mean():.8f} "
        f"std={hidden_return.std():.8f} "
        f"p99_abs={np.quantile(np.abs(hidden_return), 0.99):.8f}"
    )


if __name__ == "__main__":
    prepare_module = load_prepare_module()
    architecture_smoke(prepare_module)
    target_smoke()
