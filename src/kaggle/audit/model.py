"""Champion architecture, extracted unchanged; shared by CPU/CUDA/XLA."""
import numpy as np
import torch
import torch.nn as nn

class ScalingLayer(nn.Module):
    def __init__(self, n_ens, n_features):
        super().__init__(); self.scale = nn.Parameter(torch.ones(n_ens, n_features))
    def forward(self, x): return x * self.scale[None, :, :]

class PBLDEmbedding(nn.Module):
    def __init__(self, n_ens, n_features, hidden_dim=16, out_dim=4, freq_scale=1.0):
        super().__init__()
        self.n_ens = n_ens; self.n_features = n_features; self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) * (1.0 / np.sqrt(hidden_dim)))
        self.b2 = nn.Parameter(torch.randn(n_ens, n_features, out_dim - 1))
        self.act = nn.GELU()
        nn.init.uniform_(self.b1, -np.pi, np.pi)
    def forward(self, x):
        b = x.shape[0]
        periodic = torch.cos(2 * np.pi * (x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0)))
        t = torch.einsum('bnfh,nfhd->bnfd', periodic, self.w2)
        t = self.act(t + self.b2.unsqueeze(0))
        return torch.cat([x.unsqueeze(-1), t], dim=-1).view(b, self.n_ens, -1)

class NTPLinear(nn.Module):
    def __init__(self, n_ens, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None
    def forward(self, x):
        x = torch.einsum('bni,nio->bno', x, self.weight) / np.sqrt(self.in_features)
        return x + self.bias if self.bias is not None else x

class RealMLP(nn.Module):
    def __init__(self, n_numerical, n_ens=8):
        super().__init__()
        act = nn.GELU
        self.n_ens = n_ens
        self.num_embed = PBLDEmbedding(n_features=n_numerical, hidden_dim=24, out_dim=3,
                                       freq_scale=1.0, n_ens=n_ens)
        total_dim = n_numerical * 3
        self.dropout = nn.Dropout(0.01)
        self.shared = nn.Sequential(
            nn.LayerNorm(total_dim),
            ScalingLayer(n_ens, total_dim),
            NTPLinear(n_ens, total_dim, 512), act(), self.dropout,
            NTPLinear(n_ens, 512, 512), act(), self.dropout,
            NTPLinear(n_ens, 512, 128), act(), self.dropout,
        )
        self.reg_head = NTPLinear(n_ens, 128, 1)
        mask = torch.ones(n_ens, total_dim, dtype=torch.bool)
        for i in range(n_ens):
            mask[i, i::n_ens // 2] = False
        self.register_buffer('feature_mask', mask)
    def forward(self, x_num):
        x = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x = self.num_embed(x)
        x = x * self.feature_mask.unsqueeze(0).float()
        f = self.shared(x)
        return self.reg_head(f).squeeze(-1)   # (batch, n_ens)

class EMA:
    def __init__(self, model, decay=0.998):
        self.model = model; self.decay = decay
        self.ema_state = {n: p.data.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
    def update(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.ema_state[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)
    def apply(self):
        orig = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                orig[n] = p.data.clone().detach(); p.data.copy_(self.ema_state[n])
        return orig
    def restore(self, orig):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in orig: p.data.copy_(orig[n])


def flat_anneal(v, progress, flat_ratio=0.5):
    if progress < flat_ratio: return v
    return v * (1 - (progress - flat_ratio) / (1 - flat_ratio))

