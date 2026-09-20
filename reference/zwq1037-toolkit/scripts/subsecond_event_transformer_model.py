"""Old market/static Transformer with two explicit-length event GRU encoders."""
import ast
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ROOT = Path(__file__).resolve().parents[1]


class SubsecondEventEncoder(nn.Module):
    def __init__(self, input_dim, d_model=96, dropout=0.15):
        super().__init__()
        self.input_dim = input_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model),
            nn.SiLU(), nn.Dropout(dropout),
        )
        self.gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
        self.attention = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.pool = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.SiLU(), nn.Dropout(dropout),
        )

    def forward(self, values):
        # 有效位置列由缓存长度构造，与归一化后的特征值无关。
        # The validity column comes from stored lengths, never normalized values.
        valid = values[:, :, self.input_dim] > 0.5
        lengths = valid.sum(dim=1)
        projected = self.projection(values[:, :, :self.input_dim].contiguous())
        packed = pack_padded_sequence(
            projected, lengths.clamp_min(1).cpu(), batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        tokens, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=values.shape[1],
        )
        last_index = (lengths - 1).clamp_min(0)
        last = tokens[torch.arange(len(tokens), device=tokens.device), last_index]
        mean = (tokens * valid[:, :, None]).sum(dim=1) / lengths.clamp_min(1)[:, None]
        logits = self.attention(tokens).squeeze(-1).masked_fill(~valid, -1e4)
        weights = torch.softmax(logits, dim=1) * valid.to(logits.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        attended = (tokens * weights[:, :, None]).sum(dim=1)
        pooled = self.pool(torch.cat([last, mean, attended], dim=1))
        return pooled * (lengths > 0)[:, None].to(pooled.dtype)


def make_model():
    reference = ROOT / 'data/interim/kaggle_kernels/multistream_factorized_transformer_dev/run_v7_baseline.py'
    source = reference.read_text(encoding='utf-8')
    tree = ast.parse(source)
    classes = {node.name: ast.unparse(node) for node in tree.body if isinstance(node, ast.ClassDef)}
    namespace = {
        'torch': torch, 'nn': nn,
        'MARKET_FEATURES': list(range(11)), 'TX_FEATURES': list(range(7)),
        'ORDER_FEATURES': list(range(10)), 'MARKET_LEN': 200, 'FLOW_LEN': 60,
        'STATIC_FEATURE_COUNT': 379, 'SubsecondEventEncoder': SubsecondEventEncoder,
    }
    exec(classes['ConvBlock'], namespace)
    encoder = classes['_FactorizedStreamEncoder']
    encoder = encoder.replace(
        'tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)',
        'safe_mask = padding_mask.clone()\n        safe_mask[padding_mask.all(dim=1), -1] = False\n        tokens = self.encoder(tokens, src_key_padding_mask=safe_mask.contiguous())',
    )
    exec(encoder, namespace)
    joint = classes['_JointMultiStreamStaticModel']
    joint = joint.replace(
        '_FactorizedStreamEncoder(len(TX_FEATURES), FLOW_LEN, d_model, nhead, nlayers, dropout)',
        'SubsecondEventEncoder(9, d_model, dropout)',
    ).replace(
        '_FactorizedStreamEncoder(len(ORDER_FEATURES), FLOW_LEN, d_model, nhead, nlayers, dropout)',
        'SubsecondEventEncoder(15, d_model, dropout)',
    )
    if joint.count('SubsecondEventEncoder(') != 2:
        raise RuntimeError('Expected exactly two replaced event encoders')
    exec(joint, namespace)
    torch.backends.mha.set_fastpath_enabled(False)
    return namespace['_JointMultiStreamStaticModel']()


if __name__ == '__main__':
    torch.set_num_threads(2)
    model = make_model()
    market = torch.randn(3, 200, 11)
    transaction = torch.randn(3, 129, 10)
    order = torch.randn(3, 257, 16)
    transaction[:, :, -1] = 0
    order[:, :, -1] = 0
    transaction[0, :10, -1] = 1
    transaction[1, :129, -1] = 1
    order[0, :50, -1] = 1
    order[1, :257, -1] = 1
    static = torch.randn(3, 379)
    output = model(market, transaction, order, static)
    assert output.shape == (3,) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print('event_gru_transformer_shape_gradient_empty_stream=passed')
