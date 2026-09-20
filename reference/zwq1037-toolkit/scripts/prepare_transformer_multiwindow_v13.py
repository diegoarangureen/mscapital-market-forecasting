"""Package a ten-token multiwindow Transformer in the existing Kaggle notebook."""
from __future__ import annotations

import ast
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
KERNEL = PROJECT / "data/interim/kaggle_kernels/multistream_factorized_transformer_dev"

ARCHITECTURE = '''
class _FactorizedStreamEncoder(nn.Module):
    def __init__(self, input_dim, length, windows, d_model=96, nhead=4, nlayers=2, dropout=0.15):
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, length, d_model))
        self.conv5 = ConvBlock(d_model, kernel=5, dropout=dropout)
        self.conv3 = ConvBlock(d_model, kernel=3, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.attention = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        window_mask = torch.stack([
            torch.arange(length) >= length - steps for steps in windows
        ])
        self.register_buffer("window_mask", window_mask)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, values):
        values = values.contiguous()
        padding_mask = values.abs().sum(-1) == 0
        # 全空样本允许编码一个位置，池化仍用原mask，避免全mask产生NaN。
        # Unmask a sentinel only for encoding; original masks govern pooling.
        safe_padding = padding_mask.clone()
        safe_padding[padding_mask.all(dim=1), -1] = False
        tokens = self.projection(values) + self.position
        tokens = self.conv5(tokens)
        tokens = self.conv3(tokens)
        tokens = self.encoder(tokens, src_key_padding_mask=safe_padding)
        observed = (~padding_mask)[:, None, :] & self.window_mask[None, :, :]
        missing = ~observed.any(dim=-1)
        logits = self.attention(tokens).squeeze(-1)[:, None, :]
        logits = logits.expand(-1, 3, -1).masked_fill(~observed, -1e4)
        weights = torch.softmax(logits, dim=-1) * observed.to(logits.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        summaries = torch.einsum("bwl,bld->bwd", weights, tokens)
        return summaries, missing


class _JointMultiStreamStaticModel(nn.Module):
    """Retain three windows per stream before mixing ten summary tokens."""
    def __init__(self, d_model=96, nhead=4, nlayers=2, dropout=0.15):
        super().__init__()
        market_windows = [MARKET_LEN, round(60 / (MARKET_SECONDS / MARKET_LEN)), round(15 / (MARKET_SECONDS / MARKET_LEN))]
        flow_windows = [FLOW_LEN, round(15 / (FLOW_SECONDS / FLOW_LEN)), round(5 / (FLOW_SECONDS / FLOW_LEN))]
        self.market = _FactorizedStreamEncoder(len(MARKET_FEATURES), MARKET_LEN, market_windows, d_model, nhead, nlayers, dropout)
        self.transaction = _FactorizedStreamEncoder(len(TX_FEATURES), FLOW_LEN, flow_windows, d_model, nhead, nlayers, dropout)
        self.order = _FactorizedStreamEncoder(len(ORDER_FEATURES), FLOW_LEN, flow_windows, d_model, nhead, nlayers, dropout)
        self.static_adapter = nn.Sequential(
            nn.Linear(STATIC_FEATURE_COUNT, 192), nn.SiLU(), nn.LayerNorm(192),
            nn.Dropout(dropout), nn.Linear(192, d_model), nn.SiLU(),
        )
        self.source_embedding = nn.Parameter(torch.zeros(1, 4, d_model))
        self.window_embedding = nn.Parameter(torch.zeros(1, 3, d_model))
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=1)
        self.source_attention = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.source_embedding, std=0.02)
        nn.init.normal_(self.window_embedding, std=0.02)

    def forward(self, market, transaction, order, static):
        inputs = [market, transaction, order]
        encoders = [self.market, self.transaction, self.order]
        summaries, missing_masks = [], []
        for index, (encoder, values) in enumerate(zip(encoders, inputs)):
            pooled, missing = encoder(values)
            summaries.append(pooled + self.source_embedding[:, index:index+1] + self.window_embedding)
            missing_masks.append(missing)
        summaries.append(self.static_adapter(static)[:, None] + self.source_embedding[:, 3:4])
        missing_masks.append(torch.zeros(static.shape[0], 1, dtype=torch.bool, device=static.device))
        summaries = torch.cat(summaries, dim=1)
        missing = torch.cat(missing_masks, dim=1)
        summaries = self.fusion(summaries, src_key_padding_mask=missing)
        logits = self.source_attention(summaries).squeeze(-1).masked_fill(missing, -1e4)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return self.head((summaries * weights).sum(dim=1)).squeeze(-1)
'''


def main():
    source = (KERNEL / "run_v7_baseline.py").read_text(encoding="utf-8")
    start = source.rfind("class _FactorizedStreamEncoder(nn.Module):")
    end = source.index('if __name__ == "__main__":', start)
    source = source[:start] + ARCHITECTURE + "\n" + source[end:]
    source = source.replace('EPOCHS", "6"', 'EPOCHS", "4"')
    source = source.replace('"multistream_factorized_transformer_dev"', '"multistream_factorized_transformer_multiwindow10_dev"')
    source = source.replace('WORK_DIR", "/kaggle/working/factorized_transformer"', 'WORK_DIR", "/kaggle/working/factorized_transformer_multiwindow10"')
    # 预测也走双卡封装，避免推理时绕过DataParallel。
    # Keep DataParallel active during inference as well as training.
    source = source.replace('        if hasattr(model, "parallel") and isinstance(model.parallel, nn.DataParallel):\n            inference_model = model.parallel.module\n', '')
    ast.parse(source)
    (KERNEL / "run_v13_multiwindow10.py").write_text(source, encoding="utf-8")
    (KERNEL / "run.py").write_text(source, encoding="utf-8")
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    for name in ("run.ipynb", "run_v13_multiwindow10.ipynb"):
        (KERNEL / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    metadata_path = KERNEL / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["title"] = "MultiStream Factorized Transformer MultiWindow Dev"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"code": str(KERNEL / "run_v13_multiwindow10.py"), "token_count": 10, "epochs": 4, "same_kernel": metadata["id"]}))


if __name__ == "__main__":
    main()
