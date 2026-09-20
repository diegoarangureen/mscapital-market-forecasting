"""Prepare strict dev of a physically synchronized recent three-stream TCN."""
import ast
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
KERNEL = PROJECT/'data/interim/kaggle_kernels/multistream_factorized_transformer_dev'

ARCHITECTURE = r'''
class _LongMarketEncoder(nn.Module):
    def __init__(self, d_model=96, dropout=0.15):
        super().__init__()
        self.projection = nn.Linear(len(MARKET_FEATURES), d_model)
        self.position = nn.Parameter(torch.zeros(1, MARKET_LEN, d_model))
        self.conv5 = ConvBlock(d_model, kernel=5, dropout=dropout)
        self.conv3 = ConvBlock(d_model, kernel=3, dropout=dropout)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4,
            dim_feedforward=d_model*4, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.attention = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, values):
        padding = values.abs().sum(-1) == 0
        safe = padding.clone()
        safe[padding.all(dim=1), -1] = False
        tokens = self.projection(values) + self.position
        tokens = self.encoder(self.conv3(self.conv5(tokens)),
                              src_key_padding_mask=safe.contiguous())
        logits = self.attention(tokens).squeeze(-1).masked_fill(padding, -1e4)
        weights = torch.softmax(logits, dim=1) * (~padding).to(logits.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("bl,bld->bd", weights, tokens), padding.all(dim=1)


class _CausalTCNBlock(nn.Module):
    """Dilated causal temporal mixing without Kaggle T4 Conv1d kernels."""
    def __init__(self, width, dilation, dropout=0.15):
        super().__init__()
        self.dilation = dilation
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Linear(width * 3, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        residual = tokens
        x = self.norm(tokens)
        dilation = self.dilation
        past_one = torch.cat([torch.zeros_like(x[:, :dilation]), x[:, :-dilation]], dim=1)
        two_dilation = 2 * dilation
        past_two = torch.cat([torch.zeros_like(x[:, :two_dilation]), x[:, :-two_dilation]], dim=1)
        mixed = torch.cat([x, past_one, past_two], dim=-1)
        return residual + self.dropout(self.net(mixed))


class _RecentSynchronizedTCN(nn.Module):
    """Align each recent market 3-second bar with three 1-second flow bins."""
    def __init__(self, d_model=96, dropout=0.15):
        super().__init__()
        stream_width = 32
        self.market_projection = nn.Linear(len(MARKET_FEATURES), stream_width)
        self.transaction_projection = nn.Linear(3*len(TX_FEATURES)+3, stream_width)
        self.order_projection = nn.Linear(3*len(ORDER_FEATURES)+3, stream_width)
        self.joint_projection = nn.Linear(3*stream_width+7, d_model)
        self.position = nn.Parameter(torch.zeros(1, 20, d_model))
        self.blocks = nn.ModuleList([
            _CausalTCNBlock(d_model, dilation, dropout) for dilation in (1,2,4,8)
        ])
        self.attention = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, market, transaction, order):
        recent_market = market[:, -20:].contiguous()
        tx = transaction.reshape(transaction.shape[0], 20, 3, transaction.shape[-1])
        od = order.reshape(order.shape[0], 20, 3, order.shape[-1])
        market_present = (recent_market.abs().sum(-1) > 0)
        tx_present = (tx.abs().sum(-1) > 0)
        od_present = (od.abs().sum(-1) > 0)
        tx_input = torch.cat([tx.flatten(2), tx_present.to(tx.dtype)], dim=-1)
        od_input = torch.cat([od.flatten(2), od_present.to(od.dtype)], dim=-1)
        presence = torch.cat([market_present[...,None].to(market.dtype),
                              tx_present.to(market.dtype), od_present.to(market.dtype)], dim=-1)
        joint = torch.cat([self.market_projection(recent_market),
                           self.transaction_projection(tx_input),
                           self.order_projection(od_input), presence], dim=-1)
        tokens = self.joint_projection(joint) + self.position
        for block in self.blocks:
            tokens = block(tokens)
        padding = ~(market_present | tx_present.any(-1) | od_present.any(-1))
        logits = self.attention(tokens).squeeze(-1).masked_fill(padding, -1e4)
        weights = torch.softmax(logits, dim=1) * (~padding).to(logits.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("bl,bld->bd", weights, tokens), padding.all(dim=1)


class _JointMultiStreamStaticModel(nn.Module):
    def __init__(self, d_model=96, dropout=0.15):
        super().__init__()
        self.long_market = _LongMarketEncoder(d_model, dropout)
        self.recent_joint = _RecentSynchronizedTCN(d_model, dropout)
        self.static_adapter = nn.Sequential(nn.Linear(STATIC_FEATURE_COUNT,192),
            nn.SiLU(), nn.LayerNorm(192), nn.Dropout(dropout),
            nn.Linear(192,d_model), nn.SiLU())
        self.source_embedding = nn.Parameter(torch.zeros(1,3,d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=4,
            dim_feedforward=d_model*2,dropout=dropout,activation="gelu",
            batch_first=True,norm_first=True)
        self.fusion = nn.TransformerEncoder(layer,num_layers=1)
        self.source_attention = nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,1))
        self.head = nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,d_model),
            nn.GELU(),nn.Dropout(dropout),nn.Linear(d_model,1))
        nn.init.normal_(self.source_embedding,std=0.02)

    def forward(self, market, transaction, order, static):
        long_summary,long_missing = self.long_market(market)
        recent_summary,recent_missing = self.recent_joint(market,transaction,order)
        static_summary = self.static_adapter(static)
        summaries = torch.stack([long_summary,recent_summary,static_summary],dim=1)
        summaries = summaries + self.source_embedding
        missing = torch.stack([long_missing,recent_missing,
            torch.zeros_like(long_missing)],dim=1)
        summaries = self.fusion(summaries,src_key_padding_mask=missing.contiguous())
        logits = self.source_attention(summaries).squeeze(-1).masked_fill(missing,-1e4)
        weights = torch.softmax(logits,dim=1).unsqueeze(-1)
        return self.head((summaries*weights).sum(dim=1)).squeeze(-1)
'''


def main():
    source = (KERNEL/'run_v14_multiwindow10_evalfix.py').read_text(encoding='utf-8')
    start = source.rfind('class _FactorizedStreamEncoder(nn.Module):')
    end = source.index('if __name__ == "__main__":', start)
    source = source[:start] + ARCHITECTURE + '\n\n' + source[end:]
    source = source.replace('factorized_transformer_multiwindow10',
                            'synchronized_recent_tcn')
    source = source.replace('best_transformer_cnn.pt','best_sync_tcn.pt')
    ast.parse(source)
    for name in ('run_v16_sync_tcn_dev.py','run.py'):
        (KERNEL/name).write_text(source,encoding='utf-8')
    notebook={'cells':[{'id':'sync-tcn-dev','cell_type':'code','execution_count':None,
        'metadata':{},'outputs':[],'source':source.splitlines(keepends=True)}],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                    'language_info':{'name':'python','version':'3.11'}},
        'nbformat':4,'nbformat_minor':5}
    (KERNEL/'run.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    metadata_path=KERNEL/'kernel-metadata.json'
    metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    metadata['id']='zwq1037/multistream-factorized-transformer-multiwindow-dev'
    metadata['title']='MultiStream Factorized Transformer MultiWindow Dev'
    metadata_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Prepared synchronized recent TCN dev in existing dual-T4 cached notebook')

if __name__=='__main__': main()

