import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

# ============ 固定随机种子 ============
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"随机种子已固定为: {seed}")

set_seed(2026)

onehotmax = 10
n_ens = 16
embed_dim = 6
LR = 1e-3
epochs = 10
train_bs = 256
eval_bs = 256
target_col = 'y'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
# ==CELL==
#BASE_PATH = '/kaggle/input/competitions/ms-capital-real-financial-market-forecasting'
#train=pd.read_csv("/kaggle/input/notebooks/yunsuxiaozi/rfmf-0723data/train.csv").sort_values('sample_id')
#test=pd.read_csv("/kaggle/input/notebooks/yunsuxiaozi/rfmf-0723data/test.csv").sort_values('sample_id')
import pandas as pd
import numpy as np

BASE_PATH = '/kaggle/input/competitions/ms-capital-real-financial-market-forecasting'

# ===== 读取两个版本 =====
# 版本1: 不等间隔（0723）
train = pd.read_csv("/kaggle/input/notebooks/yunsuxiaozi/rfmf-0726data/train.csv").sort_values('sample_id')
test = pd.read_csv("/kaggle/input/notebooks/yunsuxiaozi/rfmf-0726data/test.csv").sort_values('sample_id')



from scipy.stats import spearmanr

def filter_high_correlation(df, target_col, corr_threshold=0.95, method='pearson'):
    """
    基于相关性筛选特征：对相关性高于阈值的特征对，保留与target相关性更高的特征
    
    Parameters:
    -----------
    df : pd.DataFrame
        包含特征和目标变量的数据框
    target_col : str
        目标变量列名
    corr_threshold : float, default=0.95
        相关性阈值，超过此值的特征对将被处理
    method : str, default='pearson'
        相关性计算方法，可选 'pearson', 'spearman', 'kendall'
    
    Returns:
    --------
    drop_cols : list
        需要删除的特征列名列表
    """
    
    # 1. 分离特征和目标
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    # 2. 计算特征与目标变量的相关性（用于后续择优保留）
    target_corr = {}
    for col in feature_cols:
        try:
            if method == 'pearson':
                corr = X[col].corr(y, method='pearson')
            elif method == 'spearman':
                corr = X[col].corr(y, method='spearman')
            else:
                corr = X[col].corr(y, method='kendall')
            # 处理NaN（比如全为常量的特征）
            target_corr[col] = abs(corr) if not np.isnan(corr) else -1
        except:
            target_corr[col] = -1  # 计算失败则排最后
    
    # 3. 计算特征间的相关系数矩阵
    if method == 'pearson':
        corr_matrix = X.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = X.corr(method='spearman')
    else:
        corr_matrix = X.corr(method='kendall')
    
    # 4. 遍历上三角矩阵，找出高相关的特征对
    high_corr_pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= corr_threshold:
                high_corr_pairs.append((cols[i], cols[j], abs(corr_val)))
    
    # 5. 按相关系数从高到低排序（先处理相关性最强的对子）
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # 6. 贪心策略：对每一对，保留与target相关性更高的特征
    drop_cols = set()
    for col1, col2, _ in high_corr_pairs:
        # 如果任一特征已被标记删除，跳过（因为另一特征可能已经保留了）
        if col1 in drop_cols or col2 in drop_cols:
            continue
        
        # 比较与target的相关性，删掉相关性较低的那个
        if target_corr[col1] >= target_corr[col2]:
            drop_cols.add(col2)
        else:
            drop_cols.add(col1)
    
    # 7. 额外检查：如果某个特征与target相关性为0或极低，但还保留着，也建议删除
    # （这一步是可选的安全措施，根据业务需要决定是否启用）
    for col in feature_cols:
        if col not in drop_cols and target_corr[col] < 0.0001:
            # 如果特征和target几乎无关，且它没有因为高相关被删，也标记为删除
            # 注意：这里阈值设得很低，防止误删
            drop_cols.add(col)
    
    return list(drop_cols)


drop=filter_high_correlation(train, 'target', corr_threshold=0.9, method='pearson')


drop+=[c for c in train.columns if train[c].nunique()==1]#+['x_sharpe_like', 't_sec_rowcount_weighted_15', 'o_sec_cancel_weighted_60', 'o_vol_120', 't_sec_vol_weighted_15', 't_sec_vol_60', 't_sec_row_count_30', 'x_t_price_trend_30', 'o_sec_vol_30', 't_sv_30', 't_sec_vol_near_far_ratio_30', 'o_sec_rowcount_weighted_60', 'x_sp_imb', 'm_mid_ewm_120', 'm_vol_short', 'o_sec_vol_15', 'm_vol_short_long_ratio', 'o_sec_cancel_new_ratio_45', 't_vol_weighted_15', 'o_n_30', 't_price_weighted_30', 'o_n_120', 't_transaction_count', 'o_cancel_ratio_first_half_vs_second', 'o_sec_vol_45', 't_buy_ratio_30', 'o_sec_row_count_60', 'x_tx_order_count_ratio', 'x_sec_o_max_missing_ratio', 'o_sec_cancel_new_ratio_60', 'o_order_count', 'o_n_15', 'x_t_vol_weight_ratio_15', 'm_mid_mean_180', 'x_t_vol_trend_15', 'x_sec_trans_order_vol_ratio', 'o_sec_has_data_60', 't_n_120', 't_price_weighted_60', 'x_sec_tx_order_activity_ratio', 'm_sp_mean_180', 't_n_15', 'o_vol_weighted_30', 'o_cancel_weighted_30', 't_sec_price_mean_60', 'o_sec_price_mean_60', 'o_sec_cancel_count_60', 't_sec_total_row_count', 'x_sec_o_cancel_trend_15', 't_sec_row_count_45', 't_sec_buy_ratio_60', 't_sec_vol_weighted_30', 't_sec_has_data_ratio', 'o_sec_has_data_45', 'm_mid_mean', 'o_sec_price_weighted_30', 'x_m_mid_long_short_diff', 'o_cancel_weighted_60', 't_sv_sum', 'o_sec_buy_ratio_60', 'o_vol_weighted_15', 'm_sp_last', 'm_mid_mean_60', 'm_vol_long', 'o_vol_weighted_60', 'x_ofi_ewm_short_long', 't_vol_30', 't_vol_weighted_30', 't_sec_row_count_60', 'o_sec_has_data_count', 't_price_momentum_10', 'o_sec_vol_sum', 't_sec_vol_45', 'o_sec_total_row_count', 'x_o_cancel_trend_15', 't_sec_vol_weighted_60', 'o_sec_vol_weighted_60', 'o_cancel_ratio_30', 'x_m_rv_60_180_ratio', 't_vol_15', 't_value_weighted_60', 't_buy_ratio', 'm_imb_weighted_60', 'o_sec_price_weighted_60', 't_sec_price_std_60', 'x_t_price_trend_15', 't_sec_vol_sum', 'o_cancel_weighted_15', 'o_sec_vol_60', 'o_cancel_ratio', 'o_add_ratio', 'o_sec_row_count_45', 't_sec_has_data_60', 't_sec_price_weighted_60', 'x_sec_t_max_missing_ratio']
print(len(drop))
train.drop(drop,axis=1,inplace=True)
test.drop(drop,axis=1,inplace=True)

for col in test.select_dtypes(include=[np.number]):
    if (col!='sample_id') and (train[col].nunique()>100):
        # 计算分位数
        quantiles = np.linspace(0, 1, 41)
        bins = train[col].dropna().quantile(quantiles)
        
        # 去除重复值、NaN，并排序
        bins = bins.dropna().unique()
        bins = np.sort(bins)
            
        train[col] = pd.cut(train[col], bins, include_lowest=True, labels=False)
        test[col] = pd.cut(test[col], bins, include_lowest=True, labels=False)
        test.loc[test[col].isna(), col] = 20 # 超出范围的处理


print(test.shape)
# ==CELL==
def reduce_mem_usage(df, verbose=True):
    """减少 DataFrame 内存占用"""
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage before: {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #     df[col] = df[col].astype(np.float16)
                # el
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage after: {end_mem:.2f} MB")
        print(f"Reduced: {100 * (start_mem - end_mem) / start_mem:.1f}%")
    
    return df

train = reduce_mem_usage(train).fillna(0)
test = reduce_mem_usage(test).fillna(0)
train.head()
# ==CELL==
# 预处理类
from sklearn.base import BaseEstimator, TransformerMixin

class RobustScaleSmoothClipTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        assert isinstance(X, np.ndarray)
        self._median = np.median(X, axis=-2)
        quant_diff = np.quantile(X, 0.75, axis=-2) - np.quantile(X, 0.25, axis=-2)
        idxs = quant_diff == 0.0
        quant_diff[idxs] = 0.5 * (np.max(X, axis=-2)[idxs] - np.min(X, axis=-2)[idxs])
        factors = 1.0 / (quant_diff + 1e-30)
        factors[quant_diff == 0.0] = 0.0
        self._factors = factors
        return self

    def transform(self, X, y=None):
        x_scaled = self._factors[None, :] * (X - self._median[None, :])
        return x_scaled / np.sqrt(1 + (x_scaled / 3) ** 2)
# ==CELL==
class ScalingLayer(nn.Module):
    def __init__(self, n_ens: int, n_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale[None, :, :]
class CategoricalFeatureLayer(nn.Module):
    def __init__(self, n_ens: int, cat_dims, embed_dim=8, device=None):
        super().__init__()
        self.n_ens = n_ens
        self.device = device
        self.cat_dims = cat_dims
        
        self.onehot_features = []
        self.embed_features = []
        self.embed_dims = []
        
        self.embed_offsets = []
        
        for i, dim in enumerate(cat_dims):
            if dim <= onehotmax:
                self.onehot_features.append(i)
            else:
                self.embed_features.append(i)
                self.embed_dims.append(dim)
        
        if self.embed_features:
            total_vocab = sum(self.embed_dims) * n_ens
            self.combined_emb = nn.Embedding(total_vocab, embed_dim, padding_idx=0)
            
            offset = 0
            for dim in self.embed_dims:
                self.embed_offsets.append(offset)
                offset += dim
            self.per_ens_offset = sum(self.embed_dims)
    
    def forward(self, x):
        batch_size, n_ens, n_cat = x.shape
        features = []

        if self.onehot_features:
            onehot_x = x[:, :, self.onehot_features]
            onehot_dims = [self.cat_dims[i] for i in self.onehot_features]
            total_onehot_dim = sum(onehot_dims)
            
            onehot_encoded = torch.zeros(batch_size, n_ens, total_onehot_dim, device=x.device)
            start = 0
            for idx, dim in enumerate(onehot_dims):
                pos = onehot_x[:, :, idx:idx+1].long()
                onehot_encoded.scatter_(2, pos + start, 1.0)
                start += dim
            
            features.append(onehot_encoded)
        
        # Embedding特征 - 向量化版本
        if self.embed_features:
            batch_size, n_ens, n_cat = x.shape
            n_embed_feat = len(self.embed_features)
            
            # 取出需要embedding的特征
            embed_x = x[:, :, self.embed_features].long()  # (batch, n_ens, n_embed_feat)
            
            # 计算每个样本在合并表中的索引
            # 为每个ensemble和每个特征计算偏移
            ens_offset = torch.arange(n_ens, device=x.device) * self.per_ens_offset  # (n_ens,)
            feat_offset = torch.tensor(self.embed_offsets, device=x.device)  # (n_embed_feat,)
            indices = embed_x + feat_offset.unsqueeze(0).unsqueeze(0) + ens_offset.unsqueeze(0).unsqueeze(-1)
            
            # 一次embedding lookup
            embedded = self.combined_emb(indices)  # (batch, n_ens, n_embed_feat, embed_dim)
            
            # 展平特征维度
            embedded = embedded.view(batch_size, n_ens, -1)
            features.append(embedded)
        
        return torch.cat(features, dim=2)


class PBLDEmbedding(nn.Module):
    def __init__(self, n_ens: int, n_features: int, hidden_dim: int = 16, out_dim: int = 4, 
                 freq_scale: float = 0.1):
        super().__init__()
        self.n_ens = n_ens
        self.n_features = n_features
        self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) * (1.0 / np.sqrt(hidden_dim)))
        self.b2 = nn.Parameter(torch.randn(n_ens, n_features, out_dim - 1))
        
        self.act = nn.GELU()
        nn.init.uniform_(self.b1, -np.pi, np.pi)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_expanded = x.unsqueeze(-1)
        w1_expanded = self.w1.unsqueeze(0)
        b1_expanded = self.b1.unsqueeze(0)
        periodic = torch.cos(2 * np.pi * (x_expanded * w1_expanded + b1_expanded))
        transformed = torch.einsum('b n f h, n f h d -> b n f d', periodic, self.w2)
        transformed = self.act(transformed + self.b2.unsqueeze(0))
        result = torch.cat([x.unsqueeze(-1), transformed], dim=-1)
        
        return result.view(batch_size, self.n_ens, -1)


class NTPLinear(nn.Module):
    def __init__(
        self, n_ens: int, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        x = torch.einsum('b n i, n i o -> b n o', x, self.weight) / np.sqrt(self.in_features)
        if self.bias is not None:
            x = x + self.bias
        return x
# ==CELL==
import math
from copy import deepcopy


def flat_anneal(init_value, progress, flat_ratio=0.5):
    """先平坦再衰减的学习率调度"""
    if progress < flat_ratio:
        return init_value
    else:
        decay_progress = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (1 - decay_progress)


def cosine_anneal(init_value, progress):
    """余弦退火"""
    return init_value * (math.cos(math.pi * progress) + 1) / 2


class EMA:
    """指数移动平均 (Exponential Moving Average)"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.ema_state = None
        self._init()
    
    def _init(self):
        self.ema_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema_state[name] = param.data.clone().detach()
    
    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.ema_state[name].mul_(self.decay).add_(
                        param.data, alpha=1.0 - self.decay
                    )
    
    def apply(self):
        """将模型参数替换为EMA版本"""
        original_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                original_state[name] = param.data.clone().detach()
                param.data.copy_(self.ema_state[name])
        return original_state
    
    def restore(self, original_state):
        """恢复原始模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in original_state:
                param.data.copy_(original_state[name])
# ==CELL==
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== RQ-Kmeans编码器 ====================
class RQKMeansEncoder:
    def __init__(self, n_layers=4, codebook_size=5):
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.codebooks = []
        
    def fit(self, y):
        residuals = y.copy().reshape(-1, 1)
        for layer in range(self.n_layers):
            kmeans = KMeans(n_clusters=self.codebook_size, random_state=42, n_init=10)
            kmeans.fit(residuals)
            self.codebooks.append(kmeans)
            codes = kmeans.predict(residuals)
            centroids = kmeans.cluster_centers_[codes].reshape(-1, 1)
            residuals = residuals - centroids
        return self
    
    def encode(self, y):
        residuals = y.copy().reshape(-1, 1)
        all_codes = []
        for kmeans in self.codebooks:
            codes = kmeans.predict(residuals)
            all_codes.append(codes)
            centroids = kmeans.cluster_centers_[codes].reshape(-1, 1)
            residuals = residuals - centroids
        return np.stack(all_codes, axis=1)
    
    def decode(self, codes):
        y_reconstructed = np.zeros(len(codes))
        for layer, kmeans in enumerate(self.codebooks):
            layer_codes = codes[:, layer]
            centroids = kmeans.cluster_centers_[layer_codes].reshape(-1)
            y_reconstructed += centroids
        return y_reconstructed
    
    def get_vocab_sizes(self):
        return [self.codebook_size] * self.n_layers


class RealMLP_RQ(nn.Module):
    def __init__(self, output_dim=1, cat_dims=[], n_numerical=None,
                 n_ens=8, embed_dim=4, n_rq_layers=4, rq_vocab_size=5):
        super().__init__()
        act = nn.GELU
        self.n_ens = n_ens
        self.embed_dim = embed_dim
        self.n_rq_layers = n_rq_layers
        self.rq_vocab_size = rq_vocab_size
        
        self.cate = CategoricalFeatureLayer(n_ens=self.n_ens, cat_dims=cat_dims, 
                                            embed_dim=self.embed_dim, device=device)
        
        self.num_embed = PBLDEmbedding(
            n_features=n_numerical, hidden_dim=24, 
            out_dim=3,
            freq_scale=1.0, 
            n_ens=self.n_ens
        )
        num_emb_dim = n_numerical * 3
        cat_emb_dim = sum([c if c <= onehotmax else self.embed_dim for c in cat_dims])
        total_dim = int(num_emb_dim + cat_emb_dim)

        self.dropout = nn.Dropout(0.01)
        self.shared = nn.Sequential(
            nn.LayerNorm(total_dim),
            ScalingLayer(n_ens=self.n_ens, n_features=total_dim),
            NTPLinear(n_ens=self.n_ens, in_features=total_dim, out_features=512),
            act(), self.dropout,
            NTPLinear(n_ens=self.n_ens, in_features=512, out_features=512),
            act(), self.dropout,
            NTPLinear(n_ens=self.n_ens, in_features=512, out_features=128),
            act(), self.dropout,
        )
        
        self.code_heads = nn.ModuleList([
            NTPLinear(n_ens=self.n_ens, in_features=128, out_features=rq_vocab_size)
            for _ in range(n_rq_layers)
        ])
        
        self.reg_head = NTPLinear(n_ens=self.n_ens, in_features=128, out_features=1)
        
        # 预计算mask，避免每次forward都重新计算
        self.register_buffer('feature_mask', self._create_mask(total_dim))
        
    def _create_mask(self, n_features):
        """创建mask: 对于(n_ens, n_features)，模型i丢弃特征 i, i+n_ens, i+2*n_ens, ..."""
        mask = torch.ones(self.n_ens, n_features, dtype=torch.bool)
        for i in range(self.n_ens):
            # 第i个模型丢弃索引为 i, i+n_ens, i+2*n_ens, ... 的特征
            mask[i, i::self.n_ens//2] = False
        return mask
        
    def forward(self, x_num, x_cat, return_codes=False):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        
        x_num = self.num_embed(x_num)
        x_cat = self.cate(x_cat)
        
        combined = torch.cat([x_num, x_cat], dim=2)  # (batch, n_ens, total_dim)
        
        # 应用mask：将被丢弃的特征置为0
        mask_expanded = self.feature_mask.unsqueeze(0).expand(combined.shape[0], -1, -1)
        combined = combined * mask_expanded.float()
        
        features = self.shared(combined)
        
        code_logits = [head(features) for head in self.code_heads]
        reg_pred = self.reg_head(features)
        
        if return_codes:
            return code_logits, reg_pred
        else:
            return reg_pred.mean(dim=1)
# ==CELL==
# ============ 准备数据 ============
from sklearn.preprocessing import LabelEncoder
import math
from copy import deepcopy

target_col = 'target'

# 识别分类特征和数值特征
CATS = [c for c in train.columns if train[c].dtype == 'object' or train[c].dtype.name == 'category' or train[c].nunique() <= 10]
NUMS = [c for c in train.columns if c not in CATS + ['sample_id', target_col]]

print(f"分类特征: {CATS}")
print(f"数值特征: {NUMS}")

# 处理分类特征
for c in CATS:
    mapping={v:i for i,v in enumerate(train[c].unique())}
    train[c] = train[c].map(mapping)
    test[c] = test[c].map(mapping).fillna(0)

# 获取分类特征的维度
cat_dims = [train[c].nunique() for c in CATS]

rssc = RobustScaleSmoothClipTransform()
rssc.fit(train[NUMS].values)
train[NUMS] = rssc.fit_transform(train[NUMS].values)
test[NUMS] = rssc.transform(test[NUMS].values)

print(f"分类特征维度: {cat_dims}")
print(f"数值特征数量: {len(NUMS)}")


# ============ 划分训练集和验证集 ============
train_size = 800000

X_num_train = train[NUMS].iloc[:train_size].values
X_cat_train = train[CATS].iloc[:train_size].values
y_train = train[target_col].iloc[:train_size].round(4).values

X_num_val = train[NUMS].iloc[train_size:].values
X_cat_val = train[CATS].iloc[train_size:].values
y_val = train[target_col].iloc[train_size:].values

print(f"训练集: {X_num_train.shape}, 验证集: {X_num_val.shape}")

# ============ 准备RQ编码器 ============
rq_encoder = RQKMeansEncoder(n_layers=3, codebook_size=3)
rq_encoder.fit(y_train.reshape(-1, 1))

# 获取训练集的RQ编码
y_train_rq_codes = rq_encoder.encode(y_train.reshape(-1, 1))  # [train_size, n_layers]

# 转换为 Tensor
X_num_train_tensor = torch.tensor(X_num_train, dtype=torch.float32).to(device)
X_cat_train_tensor = torch.tensor(X_cat_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
y_train_rq_tensor = torch.tensor(y_train_rq_codes, dtype=torch.long).to(device)

X_num_val_tensor = torch.tensor(X_num_val, dtype=torch.float32).to(device)
X_cat_val_tensor = torch.tensor(X_cat_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)


# ============ 评估函数 ============
def cosine_similarity_score(y_pred, y_true):
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    
    pred_centered = y_pred - y_pred.mean()
    true_centered = y_true - y_true.mean()
    
    cos_sim = (pred_centered * true_centered).sum() / (np.linalg.norm(pred_centered) + 1e-8) / (np.linalg.norm(true_centered) + 1e-8)
    return float(cos_sim)


def evaluate_model(model, x_num, x_cat, y_true, batch_size=2048):
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(x_num), batch_size):
            batch_x_num = x_num[i:i+batch_size]
            batch_x_cat = x_cat[i:i+batch_size]
            pred = model(batch_x_num, batch_x_cat, return_codes=False).mean(dim=1).squeeze()
            all_preds.append(pred.cpu())
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    cos_score = cosine_similarity_score(all_preds, y_true_np)
    return cos_score, all_preds


def compute_loss_with_rq(y_pred, y_true, code_logits, y_codes, lambda_cos=0.01, lambda_rq=0.1):
    if y_pred.dim() == 3:
        y_pred = y_pred.squeeze(-1)
    
    batch_size, n_ens = y_pred.shape
    y_true_expanded = y_true.unsqueeze(1).expand(-1, n_ens)
    
    y_pred_flat = y_pred.reshape(-1)
    y_true_flat = y_true_expanded.reshape(-1)
    
    # ========== 加权MSE ==========
    abs_true = torch.abs(y_true_flat)
    sample_weights = torch.where(abs_true > 0.001, 0.5, 1.0)
    mse_loss = (sample_weights * (y_pred_flat - y_true_flat) ** 2).mean()
    
    # ========== 余弦损失 ==========
    pred_centered = y_pred_flat - y_pred_flat.mean()
    true_centered = y_true_flat - y_true_flat.mean()
    cos_sim = (pred_centered * true_centered).sum() / (pred_centered.norm() + 1e-8) / (true_centered.norm() + 1e-8)
    cos_loss = 1 - cos_sim
    
    # ========== RQ分类损失 ==========
    rq_loss = 0
    y_codes_expanded = y_codes.unsqueeze(1).expand(-1, n_ens, -1)  # [batch, n_ens, n_layers]
    
    for layer, logits in enumerate(code_logits):
        labels = y_codes_expanded[:, :, layer].reshape(-1)
        logits_flat = logits.reshape(-1, logits.size(-1))
        rq_loss += F.cross_entropy(logits_flat, labels, reduction='mean')
    
    rq_loss = rq_loss / len(code_logits)
    
    total_loss = mse_loss + lambda_cos * cos_loss + lambda_rq * rq_loss
    
    return total_loss, cos_sim, mse_loss, rq_loss


# ============ 工具函数：分组学习率、EMA、学习率调度 ============
def get_parameter_groups(model):
    """为回归模型分组参数"""
    scale_p = []
    pbld_p = []
    first_linear_p = []
    other_w_p = []
    bias_p = []
    
    # 找到第一个线性层的weight
    first_linear_weight_id = None
    for name, param in model.named_parameters():
        if 'shared.0.weight' in name:  # ScalingLayer是shared[0]
            first_linear_weight_id = id(param)
            break
    
    for name, param in model.named_parameters():
        if 'scale' in name:
            scale_p.append(param)
        elif 'num_embed' in name:
            pbld_p.append(param)
        elif first_linear_weight_id is not None and id(param) == first_linear_weight_id:
            first_linear_p.append(param)
        elif 'bias' in name:
            bias_p.append(param)
        else:
            other_w_p.append(param)
    
    return scale_p, pbld_p, first_linear_p, other_w_p, bias_p


def flat_anneal(init_value, progress, flat_ratio=0.5):
    """先平坦再衰减的学习率调度"""
    if progress < flat_ratio:
        return init_value
    else:
        decay_progress = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (1 - decay_progress)


class EMA:
    """指数移动平均 (Exponential Moving Average)"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.ema_state = None
        self._init()
    
    def _init(self):
        self.ema_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema_state[name] = param.data.clone().detach()
    
    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.ema_state[name].mul_(self.decay).add_(
                        param.data, alpha=1.0 - self.decay
                    )
    
    def apply(self):
        """将模型参数替换为EMA版本，返回原始参数"""
        original_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                original_state[name] = param.data.clone().detach()
                param.data.copy_(self.ema_state[name])
        return original_state
    
    def restore(self, original_state):
        """恢复原始模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in original_state:
                param.data.copy_(original_state[name])


# ============ 模型初始化 ============
model = RealMLP_RQ(
    output_dim=1,
    cat_dims=cat_dims,
    n_numerical=len(NUMS),
    n_ens=n_ens,
    embed_dim=embed_dim,
    n_rq_layers=2,
    rq_vocab_size=3,
).to(device)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# ============ 分组学习率 ============
scale_p, pbld_p, first_linear_p, other_w_p, bias_p = get_parameter_groups(model)

optimizer = torch.optim.AdamW([
    {'params': scale_p,        'lr': LR * 20.0,  'weight_decay': 1e-2 * 0.1},
    {'params': pbld_p,         'lr': LR * 0.093, 'weight_decay': 1e-2},
    {'params': first_linear_p, 'lr': LR * 1.0,   'weight_decay': 1e-2 * 0.1},
    {'params': other_w_p,      'lr': LR,         'weight_decay': 1e-2},
    {'params': bias_p,         'lr': LR * 0.1,   'weight_decay': 1e-2 * 0.5},
], betas=(0.9, 0.98))

# ============ EMA ============
use_ema = True
ema_decay = 0.998
ema = EMA(model, decay=ema_decay) if use_ema else None

# ============ 训练 ============
print("开始训练...")
best_cos = -1.0
best_model_state = None

total_steps = (len(y_train_tensor) + train_bs - 1) // train_bs * epochs

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_cos = 0
    epoch_mse = 0
    epoch_rq = 0
    num_batches = 0
    
    perm = torch.randperm(len(y_train_tensor))
    X_num_shuffled = X_num_train_tensor[perm]
    X_cat_shuffled = X_cat_train_tensor[perm]
    y_shuffled = y_train_tensor[perm]
    y_rq_shuffled = y_train_rq_tensor[perm]
    
    for i in range(0, len(y_train_tensor), train_bs):
        # 计算当前进度 (用于学习率衰减)
        batch_idx = i // train_bs
        global_step = epoch * ((len(y_train_tensor) + train_bs - 1) // train_bs) + batch_idx
        progress = min(global_step / total_steps, 1.0)
        
        # ============ 动态更新学习率 ============
        optimizer.param_groups[0]['lr'] = flat_anneal(LR * 20.0, progress)
        optimizer.param_groups[1]['lr'] = flat_anneal(LR * 0.093, progress)
        optimizer.param_groups[2]['lr'] = flat_anneal(LR * 1.0, progress)
        optimizer.param_groups[3]['lr'] = flat_anneal(LR, progress)
        optimizer.param_groups[4]['lr'] = flat_anneal(LR * 0.1, progress)
        
        batch_x_num = X_num_shuffled[i:i+train_bs]
        batch_x_cat = X_cat_shuffled[i:i+train_bs]
        batch_y = y_shuffled[i:i+train_bs]
        batch_y_rq = y_rq_shuffled[i:i+train_bs]

        # 标签平滑噪声
        noise_std = 0.005 * (1 - progress) 
        batch_y_noisy = batch_y + torch.randn_like(batch_y) * noise_std
        
        optimizer.zero_grad()
        
        # 前向传播（返回分类logits和回归预测）
        code_logits, y_pred = model(batch_x_num, batch_x_cat, return_codes=True)
        
        # 计算损失
        loss, cos_sim, mse_loss, rq_loss = compute_loss_with_rq(
            y_pred, batch_y_noisy, code_logits, batch_y_rq,
            lambda_cos=0.01, lambda_rq=flat_anneal(0.1, progress)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # ============ 更新 EMA ============
        if use_ema and ema is not None:
            ema.update()
        
        epoch_loss += loss.item()
        epoch_cos += cos_sim.item()
        epoch_mse += mse_loss.item()
        epoch_rq += rq_loss.item()
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    avg_cos = epoch_cos / num_batches
    avg_mse = epoch_mse / num_batches
    avg_rq = epoch_rq / num_batches
    
    # ============ 验证时使用 EMA 模型 ============
    original_state = None
    if use_ema and ema is not None:
        original_state = ema.apply()
    
    val_cos, val_pred = evaluate_model(model, X_num_val_tensor, X_cat_val_tensor, y_val_tensor)
    
    # 恢复原始模型
    if use_ema and original_state is not None:
        ema.restore(original_state)
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Train Cos: {avg_cos:.6f} | MSE: {avg_mse:.6f} | RQ: {avg_rq:.6f} | Val Cos: {val_cos:.6f}")
    
    if val_cos > best_cos:
        best_cos = val_cos
        # 保存 EMA 模型
        state_to_save = ema.ema_state if use_ema and ema is not None else model.state_dict()
        best_model_state = {k: v.cpu().clone() for k, v in state_to_save.items()}
        print(f"  ✅ 新最佳: {best_cos:.6f}")

print(f"\n训练完成！最佳验证余弦相似度: {best_cos:.6f}")
# ==CELL==
# ============ 7. 用全量数据重新训练（参数完全一致） ============
print("\n使用全量数据重新训练...")

X_num_full = train[NUMS].values
X_cat_full = train[CATS].values
y_full = train[target_col].round(4).values

X_num_full_tensor = torch.tensor(X_num_full, dtype=torch.float32).to(device)
X_cat_full_tensor = torch.tensor(X_cat_full, dtype=torch.float32).to(device)
y_full_tensor = torch.tensor(y_full, dtype=torch.float32).to(device)

# 全量数据的RQ编码
y_full_rq_codes = rq_encoder.encode(y_full.reshape(-1, 1))
y_full_rq_tensor = torch.tensor(y_full_rq_codes, dtype=torch.long).to(device)

final_model = RealMLP_RQ(
    output_dim=1,
    cat_dims=cat_dims,
    n_numerical=len(NUMS),
    n_ens=n_ens,
    embed_dim=embed_dim,
    n_rq_layers=2,
    rq_vocab_size=3,
).to(device)

# 加载最佳模型权重（已经是EMA版本）
#final_model.load_state_dict(best_model_state)

# 重新初始化优化器（使用相同分组）
scale_p, pbld_p, first_linear_p, other_w_p, bias_p = get_parameter_groups(final_model)
optimizer_final = torch.optim.AdamW([
    {'params': scale_p,        'lr': LR * 20.0,  'weight_decay': 1e-2 * 0.1},
    {'params': pbld_p,         'lr': LR * 0.093, 'weight_decay': 1e-2},
    {'params': first_linear_p, 'lr': LR * 1.0,   'weight_decay': 1e-2 * 0.1},
    {'params': other_w_p,      'lr': LR,         'weight_decay': 1e-2},
    {'params': bias_p,         'lr': LR * 0.1,   'weight_decay': 1e-2 * 0.5},
], betas=(0.9, 0.98))

# ============ EMA ============
ema_final = EMA(final_model, decay=ema_decay) if use_ema else None

# ============ 计算总步数 ============
total_steps_full = (len(y_full_tensor) + train_bs - 1) // train_bs * epochs

for epoch in range(epochs):
    final_model.train()
    epoch_loss = 0
    num_batches = 0
    
    perm = torch.randperm(len(y_full_tensor))
    X_num_shuffled = X_num_full_tensor[perm]
    X_cat_shuffled = X_cat_full_tensor[perm]
    y_shuffled = y_full_tensor[perm]
    y_rq_shuffled = y_full_rq_tensor[perm]
    
    for i in range(0, len(y_full_tensor), train_bs):
        # ============ 计算精细进度 ============
        batch_idx = i // train_bs
        global_step = epoch * ((len(y_full_tensor) + train_bs - 1) // train_bs) + batch_idx
        progress = min(global_step / total_steps_full, 1.0)
        
        # ============ 动态更新学习率 ============
        optimizer_final.param_groups[0]['lr'] = flat_anneal(LR * 20.0, progress)
        optimizer_final.param_groups[1]['lr'] = flat_anneal(LR * 0.093, progress)
        optimizer_final.param_groups[2]['lr'] = flat_anneal(LR * 1.0, progress)
        optimizer_final.param_groups[3]['lr'] = flat_anneal(LR, progress)
        optimizer_final.param_groups[4]['lr'] = flat_anneal(LR * 0.1, progress)
        
        batch_x_num = X_num_shuffled[i:i+train_bs]
        batch_x_cat = X_cat_shuffled[i:i+train_bs]
        batch_y = y_shuffled[i:i+train_bs]
        batch_y_rq = y_rq_shuffled[i:i+train_bs]

        noise_std = 0.005 * (1 - progress) 
        batch_y_noisy = batch_y + torch.randn_like(batch_y) * noise_std
        
        optimizer_final.zero_grad()
        code_logits, y_pred = final_model(batch_x_num, batch_x_cat, return_codes=True)
        
        loss, cos_sim, mse_loss, rq_loss = compute_loss_with_rq(
            y_pred, batch_y_noisy, code_logits, batch_y_rq,
            lambda_cos=0.01, lambda_rq=0.1
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
        optimizer_final.step()
        
        # ============ 更新 EMA ============
        if use_ema and ema_final is not None:
            ema_final.update()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    print(f"全量训练 Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

# ============ 推理前应用EMA ============
if use_ema and ema_final is not None:
    ema_final.apply()  # 将模型参数替换为EMA平滑版本
    print("✅ 已应用EMA参数进行推理")

print("全量训练完成！")
# ==CELL==
# ============ 8. 推理 ============
print("\n开始推理测试集...")

# 准备测试数据
X_num_test = test[NUMS].values
X_cat_test = test[CATS].values

X_num_test_tensor = torch.tensor(X_num_test, dtype=torch.float32).to(device)
X_cat_test_tensor = torch.tensor(X_cat_test, dtype=torch.float32).to(device)

final_model.eval()
test_preds = []

with torch.no_grad():
    for i in range(0, len(X_num_test_tensor), eval_bs):
        batch_x_num = X_num_test_tensor[i:i+eval_bs]
        batch_x_cat = X_cat_test_tensor[i:i+eval_bs]
        y_pred = final_model(batch_x_num, batch_x_cat, return_codes=False)  # [batch, n_ens, 1]
        y_pred_mean = y_pred.mean(dim=1).squeeze().cpu().numpy()
        test_preds.append(y_pred_mean)

test_preds = np.concatenate(test_preds)
print(f"测试集预测完成，共 {len(test_preds)} 条")

# ============ 9. 提交 ============
sample_submission = pd.read_csv(f'{BASE_PATH}/submission.csv')
sample_submission['prediction'] = test_preds
sample_submission.to_csv('submission.csv', index=False)
print("提交文件已保存: submission.csv")
print(sample_submission.head())