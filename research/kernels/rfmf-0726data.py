import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
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
embed_dim = 8
LR = 1e-3
epochs = 5
train_bs = 256
eval_bs = 256
target_col = 'y'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
# ==CELL==
import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path

BASE_PATH = '/kaggle/input/competitions/ms-capital-real-financial-market-forecasting'
OUTPUT_PATH = '/kaggle/working/processed_data'  # 可写目录


def resample_to_secondly(
    df: pl.DataFrame,
    time_col: str = 'seconds_before_predict',
    group_col: str = 'sample_id',
    max_time: int = 60,
) -> pl.DataFrame:
    """
    将非等间隔数据重采样为按秒等间隔
    时间保持倒序：59, 58, 57, ..., 0
    """
    
    # 1. 四舍五入到最近整数秒
    df = df.with_columns(
        pl.col(time_col).round(0).cast(pl.Int32).alias('_second_tmp')
    )
    
    # 2. 过滤掉超出范围的
    df = df.filter(
        (pl.col('_second_tmp') >= 0) & (pl.col('_second_tmp') <= max_time)
    )
    
    # 3. 按 sample_id + _second_tmp 聚合
    agg_exprs = []
    
    for col in df.columns:
        if col in [group_col, time_col, '_second_tmp']:
            continue
        
        dtype = df[col].dtype
        
        if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
            agg_exprs.append(pl.col(col).sum().alias(col))
        elif dtype in [pl.Float32, pl.Float64]:
            agg_exprs.append(pl.col(col).mean().alias(col))
        else:
            agg_exprs.append(pl.col(col).first().alias(col))
    
    agg_exprs.append(pl.count().alias('row_count'))
    
    aggregated = df.group_by([group_col, '_second_tmp']).agg(agg_exprs)
    
    # 4. 生成完整笛卡尔积
    sample_ids = df[group_col].unique().to_list()
    all_seconds = list(range(max_time + 1))
    
    cartesian = pl.DataFrame({
        group_col: [sid for sid in sample_ids for _ in all_seconds],
        '_second_tmp': [s for _ in sample_ids for s in all_seconds],
    })
    
    # 5. 左连接
    result = cartesian.join(
        aggregated,
        on=[group_col, '_second_tmp'],
        how='left'
    )
    
    # 6. 恢复 seconds_before_predict
    result = result.with_columns(
        pl.col('_second_tmp').cast(pl.Float32).alias(time_col)
    )
    
    result = result.drop('_second_tmp')
    
    # 7. 按 sample_id 升序，seconds_before_predict 降序（倒序：59→0）
    result = result.sort(
        by=[group_col, time_col],
        descending=[False, True]
    )
    
    return result


def process_transaction(mode='train'):
    """处理 transaction 文件"""
    print(f"处理 transaction ({mode})...")
    
    df = pl.read_ipc(f'{BASE_PATH}/{mode}/transaction.feather', memory_map=False)
    print(f"  原始行数: {df.height}")
    print(f"  原始时间范围: {df['seconds_before_predict'].min():.3f} ~ {df['seconds_before_predict'].max():.3f}")
    
    result = resample_to_secondly(df)
    
    # 保存到 /kaggle/working/
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    output_file = f'{OUTPUT_PATH}/{mode}_transaction_secondly.feather'
    result.write_ipc(output_file)
    print(f"  保存到: {output_file}")
    print(f"  输出行数: {result.height}")
    print(f"  输出列: {result.columns}")
    
    return result


def process_order(mode='train'):
    """处理 order 文件"""
    print(f"处理 order ({mode})...")
    
    df = pl.read_ipc(f'{BASE_PATH}/{mode}/order.feather', memory_map=False)
    print(f"  原始行数: {df.height}")
    print(f"  原始时间范围: {df['seconds_before_predict'].min():.3f} ~ {df['seconds_before_predict'].max():.3f}")
    
    result = resample_to_secondly(df)
    
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    output_file = f'{OUTPUT_PATH}/{mode}_order_secondly.feather'
    result.write_ipc(output_file)
    print(f"  保存到: {output_file}")
    print(f"  输出行数: {result.height}")
    print(f"  输出列: {result.columns}")
    
    return result


def main():
    """处理所有文件"""
    print("=" * 60)
    print("开始处理 transaction 和 order 为按秒等间隔")
    print(f"输出目录: {OUTPUT_PATH}")
    print("=" * 60)
    
    # 创建输出目录
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    
    for mode in ['train', 'test']:
        print(f"\n--- {mode} 集 ---")
        process_transaction(mode)
        process_order(mode)
    
    print("\n" + "=" * 60)
    print("✅ 所有文件处理完成！")
    print(f"输出文件在: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
# ==CELL==
# import requests

# API_KEY = "sk-6d9c321dd72e4e30ad7c970e6451317b"

# def ask(prompt):
#     response = requests.post(
#         "https://api.deepseek.com/chat/completions",
#         headers={
#             "Authorization": f"Bearer {API_KEY}",
#             "Content-Type": "application/json"
#         },
#         json={
#             "model": "deepseek-chat",
#             "messages": [{"role": "user", "content": prompt}],
#             "stream": False
#         },
#         timeout=30
#     )
#     return response.json()["choices"][0]["message"]["content"]

# question="""
# 1+1=
# """

# # 一行调用
# print(ask(question))
# ==CELL==
import polars as pl
import numpy as np
import pandas as pd

BASE_PATH = '/kaggle/input/competitions/ms-capital-real-financial-market-forecasting'
PROCESSED_PATH = '/kaggle/working/processed_data'

# 所有需要删除的特征（合并你给的两个列表）
DROP_FEATURES = {
    't_px_last', 'o_cancel_order_ratio', 'o_cancel_ratio', 
    'o_cancel_ratio_15', 'o_cancel_ratio_45', 'o_cancel_ratio_120', 
    'x_cancel_spread', 't_vol_120', 't_sv_120', 't_sd_120', 
    't_lv_mean_120', 't_buy_ratio_120', 't_px_std_120', 't_avg_signed_vol_120', 
    'o_market_order_ratio', 'o_sv_120', 'o_av_120', 'o_buy_ratio_120', 
    'o_market_ratio_120', 'x_t_signed_ratio', 'm_spread_ratio_15', 
    'm_value_imbalance_mean', 'm_spread_ratio_60', 'm_spread_ratio_180', 
    'm_spread_ratio_mean', 'o_avg_signed_vol', 't_sd_45', 
    't_buy_sell_cnt_ratio', 'o_limit_order_ratio', 't_avg_signed_vol_45', 
    't_sd_15', 'x_vwap_vs_mid', 't_px_rms', 't_avg_signed_vol_15', 
    'm_spread_ewm_120', 'm_spread_ewm_30', 'm_txv_sum_60',
    
    # 你给的额外删除列表
    'm_sp_mean', 'o_sec_has_data_30', 't_sec_vol_sum', 't_sv_15', 'o_sec_has_data_60', 
    't_sec_rowcount_weighted_60', 'o_sec_sgn_weighted_30', 't_sec_row_count_15', 
    'm_sp_mean_180', 't_n_45', 'o_av_weighted_15', 'm_vol_short_long_ratio', 
    'x_sec_tx_order_activity_ratio', 'o_sec_price_weighted_15', 't_sec_vol_30', 
    't_sec_price_std_30', 'm_txv_sum_180', 't_price_weighted_60', 'o_sec_vol_60', 
    'o_sec_price_weighted_60', 'x_t_vol_weight_ratio_15', 'o_sv_45', 
    't_sec_buy_ratio_60', 'm_mid_mean_60', 't_value_weighted_15', 't_sec_has_data_30', 
    'o_cancel_weighted_15', 't_sv_sum', 'o_sec_vol_weighted_60', 'm_mid_range', 
    'x_o_cancel_trend_15', 't_sec_has_data_count', 'x_ofi_ewm_short_long', 
    'o_sv_weighted_15', 'o_vol_120', 't_sec_rowcount_weighted_15', 't_sv_30', 
    'o_n_15', 'o_av_45', 'x_sec_o_max_missing_ratio', 'x_t_price_trend_30', 
    'm_imb_ewm_120', 't_sec_has_data_60', 'x_t_signed_weighted_30', 'o_vol_weighted_30', 
    'm_mid_weighted_300', 't_sec_buy_ratio_30', 't_price_momentum_10', 'x_sharpe_like', 
    't_sec_has_data_ratio', 't_large_buy_95', 'o_cancel_ratio', 'o_sec_row_count_45', 
    'o_sec_cancel_weighted_60', 't_vol_weighted_60', 'o_sec_cancel_count_30', 
    't_sec_rowcount_weighted_30', 't_sec_vol_weighted_60', 't_price_weighted_30', 
    'm_imb_weighted_300', 't_sv_weighted_30', 't_buy_ratio_30', 'o_sec_sgn_weighted_60', 
    't_sec_vol_near_far_ratio_15', 't_vol_weighted_30', 'o_sv_sum', 't_large_sell_90', 
    'x_m_mid_long_short_diff', 'o_sec_has_data_45', 't_sec_buy_ratio', 
    'o_cancel_weighted_30', 't_value_weighted_60', 't_sec_sgn_weighted_30', 
    'x_imb_ewm_short_long', 'm_imb_weighted_60', 'x_m_rv_60_180_ratio', 
    't_sec_price_std', 'o_vol_weighted_15', 't_sec_price_mean_60', 't_sd_sum', 
    't_sec_total_row_count', 'o_buy_ratio', 't_n_15', 'o_sec_price_mean_45', 
    'o_sec_cancel_weighted_15', 't_sec_vol_weighted_30', 'o_sec_price_weighted_30', 
    'm_ofi_weighted_60', 't_lv_mean_45', 'o_sec_vol_sum', 't_sec_has_data_45', 
    'o_sec_cancel_count_15', 't_sec_buy_ratio_45', 'o_av_weighted_60', 
    't_sec_price_weighted_15', 't_n_30', 'o_sec_cancel_count', 'x_tx_order_count_ratio', 
    'o_sec_vol_mean', 't_sec_row_count_30', 'o_n_120', 't_sec_sgn_weighted_60', 
    'o_sec_rowcount_weighted_15', 't_sec_vol_weighted_15', 'm_mid_mean_180', 
    't_sec_price_weighted_60', 'o_vol_45', 'o_sv_weighted_60', 'm_sp_last', 
    'o_sec_rowcount_weighted_30', 't_sv_weighted_60', 'o_cancel_ratio_first_half_vs_second', 
    't_sec_price_mean', 'm_imb_mean', 't_transaction_count', 't_sec_price_std_60', 
    't_sec_price_mean_30', 't_vol_weighted_15', 'o_sec_rowcount_weighted_60', 
    't_px_std_45', 'o_vol_30', 'x_t_signed_ratio_15', 'o_sec_buy_ratio_60', 
    'o_sec_cancel_new_ratio_45', 'o_n_30', 'o_sec_has_data_count', 'o_sec_price_mean_60', 
    't_sec_price_weighted_30', 'o_sec_buy_ratio_45', 't_sd_30', 
    'x_sec_trans_order_vol_ratio', 'o_sec_cancel_weighted_30', 't_buy_ratio', 
    'o_sec_cancel_new_ratio_60', 't_vol_15', 't_sec_buy_ratio_15', 'm_mid_std_60', 
    'x_o_signed_weighted_30', 'x_t_price_trend_15', 'o_sec_vol_weighted_15', 
    'o_av_sum', 'o_order_count', 'o_cancel_weighted_60', 't_sec_vol_45', 
    'x_sec_o_cancel_trend_15', 'o_sec_row_count_60', 't_transaction_rate', 
    'm_vol_short', 't_n_120', 'x_t_vol_trend_30', 't_sec_price_range', 
    'o_sv_weighted_30', 'o_sec_vol_15', 't_sec_price_std_15', 'o_sec_cancel_count_60', 
    't_sec_vol_60', 'o_add_ratio', 't_vol_45', 'o_av_30', 'o_sec_buy_ratio_30', 
    'o_sec_total_row_count', 'o_sec_has_data_ratio', 't_sec_row_count_45', 
    'o_order_rate', 't_sec_price_std_45', 't_vol_sum', 't_sec_vol_std', 
    'm_mid_ewm_120', 'o_vol_weighted_60', 'o_sec_row_count_30', 'm_vol_long', 
    'o_sec_vol_30', 'x_sp_imb', 't_vol_30', 'm_mid_mean', 'o_sec_buy_ratio', 
    'o_sec_buy_ratio_15', 'o_sec_vol_weighted_30', 'o_cancel_ratio_30', 
    'o_sec_vol_45', 't_sec_row_count_60'
}


def get_data(mode='train', return_pandas=True, start_id=None, end_id=None):
    """
    读取 5 个文件并构造所有特征，最后统一删除不需要的特征
    """
    
    # ============================================================
    # 1. 读取原始 transaction（不等间隔）
    # ============================================================
    trans = pl.read_ipc(f'{BASE_PATH}/{mode}/transaction.feather', memory_map=False)
    print("1. 读取 transaction (原始不等间隔)")
    if start_id is not None or end_id is not None:
        trans = trans.filter(
            (pl.col('sample_id') >= (start_id if start_id is not None else 0)) &
            (pl.col('sample_id') < (end_id if end_id is not None else pl.col('sample_id').max()+1))
        )
    print(f"   transaction 行数: {trans.height}")

    # diff/shift特征
    for gap in [1, 7, 14, 21, 30, 60, 120]:
        trans = trans.with_columns(
            pl.col('seconds_before_predict').diff(gap).over('sample_id').alias(f'seconds_before_predict_diff{gap}')
        )
        for c in ['price', 'volume', 'side']:
            trans = trans.with_columns([
                pl.col(c).diff(gap).over('sample_id').alias(f'{c}_diff{gap}'),
                pl.col(c).shift(gap).over('sample_id').alias(f'{c}_shift{gap}'),
            ])
            if c == 'price':
                trans = trans.with_columns([
                    pl.col(f'{c}_diff{gap}').diff(gap).over('sample_id').alias(f'{c}_diff2_{gap}'),
                    (pl.col(f'{c}_diff{gap}') / (pl.col(c).shift(gap).over('sample_id') + 1e-8)).alias(f'{c}_diff_ratio_{gap}'),
                ])

    # 基础列
    trans = trans.with_columns([
        (pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0)).cast(pl.Float32).alias("_sgn"),
        pl.col("volume").log1p().alias("_lv"),
    ])
    trans = trans.with_columns([
        (pl.col("_sgn") * pl.col("volume")).alias("_sv"),
        (pl.col("_sgn") * pl.col("price") * pl.col("volume")).alias("_sd"),
    ])

    trans = trans.with_columns([
        (pl.col("volume").filter(pl.col("side") == 0).sum().over("sample_id") / 
         (pl.col("volume").filter(pl.col("side") == 1).sum().over("sample_id") + 1e-8)).alias("t_buy_sell_vol_ratio"),
    ])

    for quantile in [90, 95]:
        threshold = trans.select(pl.col("volume").quantile(quantile/100)).item()
        trans = trans.with_columns((pl.col("volume") > threshold).cast(pl.Boolean).alias(f"_is_large_{quantile}"))
        trans = trans.with_columns([
            pl.when(pl.col(f"_is_large_{quantile}") & (pl.col("side") == 0)).then(1.0).otherwise(0.0).sum().over("sample_id").alias(f"t_large_buy_{quantile}"),
            pl.when(pl.col(f"_is_large_{quantile}") & (pl.col("side") == 1)).then(1.0).otherwise(0.0).sum().over("sample_id").alias(f"t_large_sell_{quantile}"),
        ])

    # 时间加权
    for w in [15, 30, 60]:
        weight = (-pl.col("seconds_before_predict") / w).exp()
        trans = trans.with_columns([
            (weight * pl.col("volume")).sum().over("sample_id").alias(f"t_vol_weighted_{w}"),
            (weight * pl.col("_sv")).sum().over("sample_id").alias(f"t_sv_weighted_{w}"),
            (weight * pl.col("price")).sum().over("sample_id").alias(f"t_price_weighted_{w}"),
            (weight * pl.col("volume") * pl.col("price")).sum().over("sample_id").alias(f"t_value_weighted_{w}"),
        ])

    trans = trans.with_columns([
        pl.col("seconds_before_predict").diff().over("sample_id").abs().alias("_time_gap"),
        pl.col("seconds_before_predict").rank(method='ordinal').over("sample_id").alias("_time_rank"),
    ])

    trans = trans.with_columns([
        pl.col("_time_rank").alias("_event_idx"),
        pl.col("_time_rank").max().over("sample_id").alias("_total_events"),
    ])

    tx_agg_exprs = [
        pl.col("volume").sum().alias("t_vol_sum"),
        pl.col("_sv").sum().alias("t_sv_sum"),
        pl.col("_sd").sum().alias("t_sd_sum"),
        pl.col("_lv").mean().alias("t_lv_mean"),
        pl.col("price").std().alias("t_px_std"),
        (pl.col("_sgn") > 0).mean().alias("t_buy_ratio"),
        ((pl.col("price") * pl.col("volume")).sum() / (pl.col("volume").sum() + 1e-8)).alias("t_vwap"),
        ((pl.col("price").last() - pl.col("price").shift(10).first()) / (pl.col("price").shift(10).first() + 1e-8)).alias("t_price_momentum_10"),
        pl.col("price").pct_change().std().alias("t_price_volatility"),
        ((pl.col("price").max() - pl.col("price").min()) / (pl.col("price").mean() + 1e-8)).alias("t_price_range_ratio"),
        (pl.col("_sv") / (pl.col("volume") + 1e-8)).mean().alias("t_avg_signed_vol"),
        pl.col("price").skew().alias("t_px_skew"),
        pl.col("t_buy_sell_vol_ratio").first().alias("t_buy_sell_vol_ratio"),
        pl.col("t_large_buy_90").first().alias("t_large_buy_90"),
        pl.col("t_large_sell_90").first().alias("t_large_sell_90"),
        pl.col("t_large_buy_95").first().alias("t_large_buy_95"),
        pl.col("t_large_sell_95").first().alias("t_large_sell_95"),
        # 时间加权
        pl.col("t_vol_weighted_15").first().alias("t_vol_weighted_15"),
        pl.col("t_vol_weighted_30").first().alias("t_vol_weighted_30"),
        pl.col("t_vol_weighted_60").first().alias("t_vol_weighted_60"),
        pl.col("t_sv_weighted_15").first().alias("t_sv_weighted_15"),
        pl.col("t_sv_weighted_30").first().alias("t_sv_weighted_30"),
        pl.col("t_sv_weighted_60").first().alias("t_sv_weighted_60"),
        pl.col("t_price_weighted_15").first().alias("t_price_weighted_15"),
        pl.col("t_price_weighted_30").first().alias("t_price_weighted_30"),
        pl.col("t_price_weighted_60").first().alias("t_price_weighted_60"),
        pl.col("t_value_weighted_15").first().alias("t_value_weighted_15"),
        pl.col("t_value_weighted_30").first().alias("t_value_weighted_30"),
        pl.col("t_value_weighted_60").first().alias("t_value_weighted_60"),
        # 时间间隔
        pl.col("_time_gap").mean().alias("t_avg_time_gap"),
        pl.col("_time_gap").std().alias("t_time_gap_std"),
        pl.col("_time_gap").max().alias("t_max_time_gap"),
        (pl.col("seconds_before_predict").max() - pl.col("seconds_before_predict").min()).alias("t_time_range"),
        pl.col("_time_rank").max().alias("t_transaction_count"),
        (pl.col("_time_rank").max() / 60.0).alias("t_transaction_rate"),
        # 事件顺序特征
        ((pl.col("price").filter(pl.col("_event_idx") <= pl.col("_total_events") * 0.5).mean() /
          (pl.col("price").filter(pl.col("_event_idx") > pl.col("_total_events") * 0.5).mean() + 1e-8) - 1)
         .first().alias("t_price_first_half_vs_second")),
        ((pl.col("_sgn").filter(pl.col("_event_idx") <= pl.col("_total_events") * 0.5).mean() -
          pl.col("_sgn").filter(pl.col("_event_idx") > pl.col("_total_events") * 0.5).mean())
         .first().alias("t_buy_ratio_first_half_vs_second")),
        (pl.col("volume").filter(pl.col("_event_idx") <= pl.col("_total_events") * 0.33).sum() /
         (pl.col("volume").filter(pl.col("_event_idx") > pl.col("_total_events") * 0.67).sum() + 1e-8))
         .first().alias("t_vol_first_third_vs_last"),
        (pl.col("price").last() - pl.col("price").first()).first().alias("t_price_change_first_to_last"),
        (pl.col("_time_gap").filter(pl.col("_event_idx") <= pl.col("_total_events") * 0.5).mean() /
         (pl.col("_time_gap").filter(pl.col("_event_idx") > pl.col("_total_events") * 0.5).mean() + 1e-8))
         .first().alias("t_gap_first_half_vs_second"),
    ]

    for w in [15, 30, 45, 120]:
        cond = pl.col("seconds_before_predict") <= w
        tx_agg_exprs.append(pl.col("volume").filter(cond).sum().alias(f"t_vol_{w}"))
        tx_agg_exprs.append(pl.col("_sv").filter(cond).sum().alias(f"t_sv_{w}"))
        tx_agg_exprs.append(pl.col("_sd").filter(cond).sum().alias(f"t_sd_{w}"))
        tx_agg_exprs.append(pl.col("_lv").filter(cond).mean().alias(f"t_lv_mean_{w}"))
        tx_agg_exprs.append((pl.col("_sgn").filter(cond) > 0).mean().alias(f"t_buy_ratio_{w}"))
        tx_agg_exprs.append(pl.col("_sgn").filter(cond).len().alias(f"t_n_{w}"))
        tx_agg_exprs.append(pl.col("price").filter(cond).std().alias(f"t_px_std_{w}"))
        tx_agg_exprs.append((pl.col("_sv").filter(cond) / (pl.col("volume").filter(cond) + 1e-8)).mean().alias(f"t_avg_signed_vol_{w}"))
        if w == 15:
            tx_agg_exprs.append(pl.col("price").filter(cond).skew().alias(f"t_px_skew_{w}"))
            tx_agg_exprs.append(((pl.col("price").filter(cond).max() - pl.col("price").filter(cond).min()) / (pl.col("price").filter(cond).mean() + 1e-8)).alias(f"t_price_range_ratio_{w}"))

    trans_result = trans.group_by('sample_id').agg(tx_agg_exprs).sort('sample_id')


    # ============================================================
    # 2. 读取原始 order（不等间隔）
    # ============================================================
    order = pl.read_ipc(f'{BASE_PATH}/{mode}/order.feather', memory_map=False)
    print("2. 读取 order (原始不等间隔)")
    if start_id is not None or end_id is not None:
        order = order.filter(
            (pl.col('sample_id') >= (start_id if start_id is not None else 0)) &
            (pl.col('sample_id') < (end_id if end_id is not None else pl.col('sample_id').max()+1))
        )
    print(f"   order 行数: {order.height}")

    for gap in [1, 7, 14, 21, 30, 60, 120]:
        for c in ['price', 'volume']:
            diff_col = f'order_{c}_diff{gap}'
            shift_col = f'order_{c}_shift{gap}'
            order = order.with_columns([
                pl.col(c).diff(gap).over('sample_id').alias(diff_col),
                pl.col(c).shift(gap).over('sample_id').alias(shift_col),
            ])
            if c == 'price':
                order = order.with_columns(pl.col(diff_col).diff(gap).over('sample_id').alias(f'order_{c}_diff2_{gap}'))
        order = order.with_columns([
            pl.col('side').diff(gap).over('sample_id').alias(f'order_side_diff{gap}'),
            pl.col('side').shift(gap).over('sample_id').alias(f'order_side_shift{gap}'),
            pl.col('order_action').diff(gap).over('sample_id').alias(f'order_action_diff{gap}'),
            pl.col('order_action').shift(gap).over('sample_id').alias(f'order_action_shift{gap}'),
            pl.col('seconds_before_predict').diff(gap).over('sample_id').alias(f'order_seconds_diff{gap}'),
        ])
        order = order.with_columns(pl.col(f'order_action_diff{gap}').diff(gap).over('sample_id').alias(f'order_action_diff2_{gap}'))

    order = order.with_columns([
        (pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0)).cast(pl.Float32).alias("_sgn"),
        (pl.when(pl.col("order_action") == 0).then(1.0).otherwise(-1.0)).cast(pl.Float32).alias("_act"),
        (pl.col("order_action") == 0).cast(pl.Float32).alias("_is_market"),
        (pl.col("order_action") == 1).cast(pl.Float32).alias("_is_limit"),
        (pl.col("order_action") == 2).cast(pl.Float32).alias("_is_cancel"),
    ])
    order = order.with_columns([
        (pl.col("_sgn") * pl.col("volume")).alias("_sv"),
        (pl.col("_act") * pl.col("volume")).alias("_av"),
    ])

    for w in [15, 30, 60]:
        weight = (-pl.col("seconds_before_predict") / w).exp()
        order = order.with_columns([
            (weight * pl.col("volume")).sum().over("sample_id").alias(f"o_vol_weighted_{w}"),
            (weight * pl.col("_sv")).sum().over("sample_id").alias(f"o_sv_weighted_{w}"),
            (weight * pl.col("_av")).sum().over("sample_id").alias(f"o_av_weighted_{w}"),
            (weight * (pl.col("_is_cancel"))).sum().over("sample_id").alias(f"o_cancel_weighted_{w}"),
        ])

    order = order.with_columns([
        pl.col("seconds_before_predict").diff().over("sample_id").abs().alias("_order_gap"),
        pl.col("seconds_before_predict").rank(method='ordinal').over("sample_id").alias("_order_rank"),
    ])

    order = order.with_columns([
        pl.col("_order_rank").alias("_event_idx"),
        pl.col("_order_rank").max().over("sample_id").alias("_total_events"),
    ])

    order_agg_exprs = [
        pl.col("volume").sum().alias("o_vol_sum"),
        pl.col("_sv").sum().alias("o_sv_sum"),
        pl.col("_av").sum().alias("o_av_sum"),
        (pl.col("_sgn") > 0).mean().alias("o_buy_ratio"),
        (pl.col("_act") > 0).mean().alias("o_add_ratio"),
        pl.col("price").std().alias("o_px_std"),
        pl.col("price").skew().alias("o_px_skew"),
        pl.col("volume").filter(pl.col("side") == 0).sum().alias("o_bid_depth"),
        pl.col("volume").filter(pl.col("side") == 1).sum().alias("o_ask_depth"),
        (pl.col("_is_market").mean()).alias("o_market_ratio"),
        (pl.col("_is_cancel").mean()).alias("o_cancel_ratio"),
        # 时间加权
        pl.col("o_vol_weighted_15").first().alias("o_vol_weighted_15"),
        pl.col("o_vol_weighted_30").first().alias("o_vol_weighted_30"),
        pl.col("o_vol_weighted_60").first().alias("o_vol_weighted_60"),
        pl.col("o_sv_weighted_15").first().alias("o_sv_weighted_15"),
        pl.col("o_sv_weighted_30").first().alias("o_sv_weighted_30"),
        pl.col("o_sv_weighted_60").first().alias("o_sv_weighted_60"),
        pl.col("o_av_weighted_15").first().alias("o_av_weighted_15"),
        pl.col("o_av_weighted_30").first().alias("o_av_weighted_30"),
        pl.col("o_av_weighted_60").first().alias("o_av_weighted_60"),
        pl.col("o_cancel_weighted_15").first().alias("o_cancel_weighted_15"),
        pl.col("o_cancel_weighted_30").first().alias("o_cancel_weighted_30"),
        pl.col("o_cancel_weighted_60").first().alias("o_cancel_weighted_60"),
        # 时间间隔
        pl.col("_order_gap").mean().alias("o_avg_time_gap"),
        pl.col("_order_gap").std().alias("o_time_gap_std"),
        pl.col("_order_rank").max().alias("o_order_count"),
        (pl.col("_order_rank").max() / 60.0).alias("o_order_rate"),
        # 事件顺序特征
        ((pl.col("price").filter(pl.col("_event_idx") <= pl.col("_total_events") * 0.5).mean() /
          (pl.col("price").filter(pl.col("_event_idx") > pl.col("_total_events") * 0.5).mean() + 1e-8) - 1)
         .first().alias("o_price_first_half_vs_second")),
        (pl.col("volume").filter(pl.col("_event_idx") <= pl.col("_total_events") * 0.33).sum() /
         (pl.col("volume").filter(pl.col("_event_idx") > pl.col("_total_events") * 0.67).sum() + 1e-8))
         .first().alias("o_vol_first_third_vs_last"),
        ((pl.col("_is_cancel").filter(pl.col("_event_idx") <= pl.col("_total_events") * 0.5).mean() -
          pl.col("_is_cancel").filter(pl.col("_event_idx") > pl.col("_total_events") * 0.5).mean())
         .first().alias("o_cancel_ratio_first_half_vs_second")),
    ]

    for w in [15, 30, 45, 120]:
        cond = pl.col("seconds_before_predict") <= w
        order_agg_exprs.append(pl.col("volume").filter(cond).sum().alias(f"o_vol_{w}"))
        order_agg_exprs.append(pl.col("_sv").filter(cond).sum().alias(f"o_sv_{w}"))
        order_agg_exprs.append(pl.col("_av").filter(cond).sum().alias(f"o_av_{w}"))
        order_agg_exprs.append((pl.col("_sgn").filter(cond) > 0).mean().alias(f"o_buy_ratio_{w}"))
        order_agg_exprs.append(pl.col("_sgn").filter(cond).len().alias(f"o_n_{w}"))
        order_agg_exprs.append(pl.col("_is_market").filter(cond).mean().alias(f"o_market_ratio_{w}"))
        order_agg_exprs.append(pl.col("_is_cancel").filter(cond).mean().alias(f"o_cancel_ratio_{w}"))

    order_result = order.group_by('sample_id').agg(order_agg_exprs).sort('sample_id')


    # ============================================================
    # 3. 读取等间隔 transaction
    # ============================================================
    trans_sec = pl.read_ipc(f'{PROCESSED_PATH}/{mode}_transaction_secondly.feather', memory_map=False)
    print("3. 读取 transaction (等间隔)")
    if start_id is not None or end_id is not None:
        trans_sec = trans_sec.filter(
            (pl.col('sample_id') >= (start_id if start_id is not None else 0)) &
            (pl.col('sample_id') < (end_id if end_id is not None else pl.col('sample_id').max()+1))
        )
    print(f"   trans_sec 行数: {trans_sec.height}")

    trans_sec = trans_sec.with_columns([
        (pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0)).cast(pl.Float32).alias("_sgn"),
        pl.col("volume").log1p().alias("_lv"),
        (pl.col("row_count") > 0).cast(pl.Int32).alias("_has_data"),
    ])

    t_sec_agg_exprs = [
        pl.col("volume").sum().alias("t_sec_vol_sum"),
        pl.col("volume").mean().alias("t_sec_vol_mean"),
        pl.col("volume").std().alias("t_sec_vol_std"),
        pl.col("price").mean().alias("t_sec_price_mean"),
        pl.col("price").std().alias("t_sec_price_std"),
        pl.col("price").min().alias("t_sec_price_min"),
        pl.col("price").max().alias("t_sec_price_max"),
        (pl.col("price").max() - pl.col("price").min()).alias("t_sec_price_range"),
        (pl.col("_sgn") > 0).mean().alias("t_sec_buy_ratio"),
        pl.col("_lv").mean().alias("t_sec_lv_mean"),
        pl.col("row_count").sum().alias("t_sec_total_row_count"),
        pl.col("_has_data").sum().alias("t_sec_has_data_count"),
        (pl.col("_has_data").sum() / 61.0).alias("t_sec_has_data_ratio"),
    ]

    # 最大连续缺失
    trans_sec = trans_sec.with_columns((pl.col("_has_data") == 0).cast(pl.Int32).alias("_missing"))
    trans_sec = trans_sec.with_columns(pl.col("_missing").rle_id().over("sample_id").alias("_rle_id"))
    missing_stats = trans_sec.group_by(["sample_id", "_rle_id"]).agg([
        pl.col("_missing").sum().alias("consecutive_missing_len")
    ])
    max_missing = missing_stats.group_by("sample_id").agg([
        pl.col("consecutive_missing_len").max().alias("t_sec_max_consecutive_missing")
    ])
    trans_sec = trans_sec.join(max_missing, on="sample_id", how="left")

    # 价格趋势
    trans_sec = trans_sec.with_columns(
        pl.col("seconds_before_predict").rank("ordinal").over("sample_id").alias("_time_idx")
    )
    trans_sec = trans_sec.with_columns([
        ((pl.col("price") - pl.col("price").mean().over("sample_id")) * 
         (pl.col("_time_idx") - pl.col("_time_idx").mean().over("sample_id"))).alias("_cov_num"),
        ((pl.col("_time_idx") - pl.col("_time_idx").mean().over("sample_id")) ** 2).alias("_time_var"),
    ])
    t_sec_agg_exprs.append(
        (pl.col("_cov_num").sum() / (pl.col("_time_var").sum() + 1e-8)).alias("t_sec_price_trend")
    )

    # 价格自相关
    trans_sec = trans_sec.with_columns([
        pl.col("price").shift(1).over("sample_id").alias("_price_lag1"),
        pl.col("price").shift(5).over("sample_id").alias("_price_lag5"),
    ])
    
    trans_sec = trans_sec.with_columns([
        ((pl.col("price") - pl.col("price").mean().over("sample_id")) * 
         (pl.col("_price_lag1") - pl.col("_price_lag1").mean().over("sample_id"))).alias("_autocorr_num1"),
        ((pl.col("price") - pl.col("price").mean().over("sample_id")) ** 2).alias("_autocorr_den1"),
        ((pl.col("price") - pl.col("price").mean().over("sample_id")) * 
         (pl.col("_price_lag5") - pl.col("_price_lag5").mean().over("sample_id"))).alias("_autocorr_num5"),
        ((pl.col("price") - pl.col("price").mean().over("sample_id")) ** 2).alias("_autocorr_den5"),
    ])

    t_sec_agg_exprs.extend([
        (pl.col("_autocorr_num1").sum() / (pl.col("_autocorr_den1").sum() + 1e-8)).alias("t_sec_autocorr_lag1"),
        (pl.col("_autocorr_num5").sum() / (pl.col("_autocorr_den5").sum() + 1e-8)).alias("t_sec_autocorr_lag5"),
    ])

    # 时间加权
    for w in [15, 30, 60]:
        weight = (-pl.col("seconds_before_predict") / w).exp()
        trans_sec = trans_sec.with_columns([
            (weight * pl.col("volume")).sum().over("sample_id").alias(f"t_sec_vol_weighted_{w}"),
            (weight * pl.col("price")).sum().over("sample_id").alias(f"t_sec_price_weighted_{w}"),
            (weight * pl.col("_sgn")).sum().over("sample_id").alias(f"t_sec_sgn_weighted_{w}"),
            (weight * pl.col("row_count")).sum().over("sample_id").alias(f"t_sec_rowcount_weighted_{w}"),
        ])
        t_sec_agg_exprs.extend([
            pl.col(f"t_sec_vol_weighted_{w}").first().alias(f"t_sec_vol_weighted_{w}"),
            pl.col(f"t_sec_price_weighted_{w}").first().alias(f"t_sec_price_weighted_{w}"),
            pl.col(f"t_sec_sgn_weighted_{w}").first().alias(f"t_sec_sgn_weighted_{w}"),
            pl.col(f"t_sec_rowcount_weighted_{w}").first().alias(f"t_sec_rowcount_weighted_{w}"),
        ])

    # 窗口聚合
    for w in [15, 30, 45, 60]:
        cond = pl.col("seconds_before_predict") <= w
        t_sec_agg_exprs.extend([
            pl.col("volume").filter(cond).sum().alias(f"t_sec_vol_{w}"),
            pl.col("price").filter(cond).mean().alias(f"t_sec_price_mean_{w}"),
            pl.col("price").filter(cond).std().alias(f"t_sec_price_std_{w}"),
            (pl.col("_sgn").filter(cond) > 0).mean().alias(f"t_sec_buy_ratio_{w}"),
            pl.col("row_count").filter(cond).sum().alias(f"t_sec_row_count_{w}"),
            pl.col("_has_data").filter(cond).sum().alias(f"t_sec_has_data_{w}"),
        ])

    # 前段 vs 后段
    for w in [15, 30, 60]:
        mid = w / 2
        near_cond = pl.col("seconds_before_predict") <= mid
        far_cond = (pl.col("seconds_before_predict") <= w) & (pl.col("seconds_before_predict") > mid)
        t_sec_agg_exprs.extend([
            (pl.col("price").filter(near_cond).mean() / (pl.col("price").filter(far_cond).mean() + 1e-8) - 1).alias(f"t_sec_price_near_far_diff_{w}"),
            (pl.col("volume").filter(near_cond).sum() / (pl.col("volume").filter(far_cond).sum() + 1e-8)).alias(f"t_sec_vol_near_far_ratio_{w}"),
            (pl.col("row_count").filter(near_cond).sum() / (pl.col("row_count").filter(far_cond).sum() + 1e-8)).alias(f"t_sec_rowcount_near_far_ratio_{w}"),
        ])

    t_sec_result = trans_sec.group_by('sample_id').agg(t_sec_agg_exprs).sort('sample_id')


    # ============================================================
    # 4. 读取等间隔 order
    # ============================================================
    order_sec = pl.read_ipc(f'{PROCESSED_PATH}/{mode}_order_secondly.feather', memory_map=False)
    print("4. 读取 order (等间隔)")
    if start_id is not None or end_id is not None:
        order_sec = order_sec.filter(
            (pl.col('sample_id') >= (start_id if start_id is not None else 0)) &
            (pl.col('sample_id') < (end_id if end_id is not None else pl.col('sample_id').max()+1))
        )
    print(f"   order_sec 行数: {order_sec.height}")

    order_sec = order_sec.with_columns([
        (pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0)).cast(pl.Float32).alias("_sgn"),
        (pl.col("row_count") > 0).cast(pl.Int32).alias("_has_data"),
        (pl.col("order_action") == 0).cast(pl.Int32).alias("_is_new"),
        (pl.col("order_action") == 1).cast(pl.Int32).alias("_is_cancel"),
        (pl.col("order_action") == 2).cast(pl.Int32).alias("_is_other"),
    ])

    o_sec_agg_exprs = [
        pl.col("volume").sum().alias("o_sec_vol_sum"),
        pl.col("volume").mean().alias("o_sec_vol_mean"),
        pl.col("price").mean().alias("o_sec_price_mean"),
        pl.col("price").std().alias("o_sec_price_std"),
        (pl.col("_sgn") > 0).mean().alias("o_sec_buy_ratio"),
        pl.col("row_count").sum().alias("o_sec_total_row_count"),
        pl.col("_has_data").sum().alias("o_sec_has_data_count"),
        (pl.col("_has_data").sum() / 61.0).alias("o_sec_has_data_ratio"),
        pl.col("_is_new").sum().alias("o_sec_new_count"),
        pl.col("_is_cancel").sum().alias("o_sec_cancel_count"),
        (pl.col("_is_cancel").sum() / (pl.col("_is_new").sum() + 1e-8)).alias("o_sec_cancel_new_ratio"),
        pl.col("volume").filter(pl.col("_is_cancel") == 1).sum().alias("o_sec_cancel_volume"),
        pl.col("volume").filter(pl.col("_is_new") == 1).sum().alias("o_sec_new_volume"),
    ]

    # 连续缺失
    order_sec = order_sec.with_columns((pl.col("_has_data") == 0).cast(pl.Int32).alias("_missing"))
    order_sec = order_sec.with_columns(pl.col("_missing").rle_id().over("sample_id").alias("_rle_id"))
    missing_stats_o = order_sec.group_by(["sample_id", "_rle_id"]).agg([
        pl.col("_missing").sum().alias("consecutive_missing_len")
    ])
    max_missing_o = missing_stats_o.group_by("sample_id").agg([
        pl.col("consecutive_missing_len").max().alias("o_sec_max_consecutive_missing")
    ])
    order_sec = order_sec.join(max_missing_o, on="sample_id", how="left")

    # 时间加权
    for w in [15, 30, 60]:
        weight = (-pl.col("seconds_before_predict") / w).exp()
        order_sec = order_sec.with_columns([
            (weight * pl.col("volume")).sum().over("sample_id").alias(f"o_sec_vol_weighted_{w}"),
            (weight * pl.col("price")).sum().over("sample_id").alias(f"o_sec_price_weighted_{w}"),
            (weight * pl.col("_sgn")).sum().over("sample_id").alias(f"o_sec_sgn_weighted_{w}"),
            (weight * pl.col("row_count")).sum().over("sample_id").alias(f"o_sec_rowcount_weighted_{w}"),
            (weight * pl.col("_is_cancel")).sum().over("sample_id").alias(f"o_sec_cancel_weighted_{w}"),
        ])
        o_sec_agg_exprs.extend([
            pl.col(f"o_sec_vol_weighted_{w}").first().alias(f"o_sec_vol_weighted_{w}"),
            pl.col(f"o_sec_price_weighted_{w}").first().alias(f"o_sec_price_weighted_{w}"),
            pl.col(f"o_sec_sgn_weighted_{w}").first().alias(f"o_sec_sgn_weighted_{w}"),
            pl.col(f"o_sec_rowcount_weighted_{w}").first().alias(f"o_sec_rowcount_weighted_{w}"),
            pl.col(f"o_sec_cancel_weighted_{w}").first().alias(f"o_sec_cancel_weighted_{w}"),
        ])

    # 窗口聚合
    for w in [15, 30, 45, 60]:
        cond = pl.col("seconds_before_predict") <= w
        o_sec_agg_exprs.extend([
            pl.col("volume").filter(cond).sum().alias(f"o_sec_vol_{w}"),
            pl.col("price").filter(cond).mean().alias(f"o_sec_price_mean_{w}"),
            (pl.col("_sgn").filter(cond) > 0).mean().alias(f"o_sec_buy_ratio_{w}"),
            pl.col("row_count").filter(cond).sum().alias(f"o_sec_row_count_{w}"),
            pl.col("_has_data").filter(cond).sum().alias(f"o_sec_has_data_{w}"),
            pl.col("_is_cancel").filter(cond).sum().alias(f"o_sec_cancel_count_{w}"),
            (pl.col("_is_cancel").filter(cond).sum() / (pl.col("_is_new").filter(cond).sum() + 1e-8)).alias(f"o_sec_cancel_new_ratio_{w}"),
        ])

    # 前段 vs 后段
    for w in [15, 30, 60]:
        mid = w / 2
        near_cond = pl.col("seconds_before_predict") <= mid
        far_cond = (pl.col("seconds_before_predict") <= w) & (pl.col("seconds_before_predict") > mid)
        o_sec_agg_exprs.extend([
            (pl.col("volume").filter(near_cond).sum() / (pl.col("volume").filter(far_cond).sum() + 1e-8)).alias(f"o_sec_vol_near_far_ratio_{w}"),
            (pl.col("row_count").filter(near_cond).sum() / (pl.col("row_count").filter(far_cond).sum() + 1e-8)).alias(f"o_sec_rowcount_near_far_ratio_{w}"),
            (pl.col("_is_cancel").filter(near_cond).sum() / (pl.col("_is_cancel").filter(far_cond).sum() + 1e-8)).alias(f"o_sec_cancel_near_far_ratio_{w}"),
        ])

    o_sec_result = order_sec.group_by('sample_id').agg(o_sec_agg_exprs).sort('sample_id')


    # ============================================================
    # 5. 读取 market
    # ============================================================
    print("5. 读取 market")
    market = pl.read_ipc(f'{BASE_PATH}/{mode}/market.feather', memory_map=False)
    if start_id is not None or end_id is not None:
        market = market.filter(
            (pl.col('sample_id') >= (start_id if start_id is not None else 0)) &
            (pl.col('sample_id') < (end_id if end_id is not None else pl.col('sample_id').max()+1))
        )
    print(f"   market 行数: {market.height}")

    market = market.with_columns([
        ((pl.col("ask_price_1") + pl.col("bid_price_1")) * 0.5).alias("_mid"),
        (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("_sp"),
        ((pl.col("ask_price_1") - pl.col("bid_price_1")) / ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2 + 1e-8)).alias("_spread_ratio"),
        ((pl.col("ask_price_2") - pl.col("ask_price_1")) / (pl.col("ask_volume_2") - pl.col("ask_volume_1") + 1e-8)).alias("_ask_slope"),
        ((pl.col("bid_price_1") - pl.col("bid_price_2")) / (pl.col("bid_volume_1") - pl.col("bid_volume_2") + 1e-8)).alias("_bid_slope"),
        ((pl.col("ask_volume_1") - pl.col("bid_volume_1")) / (pl.col("ask_volume_1") + pl.col("bid_volume_1") + 1.0)).alias("_imb"),
        (pl.col("ask_volume_1") + pl.col("bid_volume_1")).alias("_depth"),
    ])
    market = market.with_columns([
        pl.col("_mid").diff().over("sample_id").fill_null(0.0).alias("_dmid"),
        (pl.col("ask_volume_1").diff().over("sample_id").fill_null(0.0) - pl.col("bid_volume_1").diff().over("sample_id").fill_null(0.0)).alias("_ofi"),
        pl.col("_mid").pct_change().abs().alias("_mid_ret_abs"),
    ])

    for w in [60, 300]:
        weight = (-pl.col("seconds_before_predict") / w).exp()
        market = market.with_columns([
            (weight * pl.col("_mid")).sum().over("sample_id").alias(f"m_mid_weighted_{w}"),
            (weight * pl.col("_imb")).sum().over("sample_id").alias(f"m_imb_weighted_{w}"),
            (weight * pl.col("_ofi")).sum().over("sample_id").alias(f"m_ofi_weighted_{w}"),
            (weight * pl.col("transaction_volume")).sum().over("sample_id").alias(f"m_vol_weighted_{w}"),
        ])

    market = market.with_columns([
        pl.col("_mid").pct_change().abs().rolling_mean(window_size=5).over("sample_id").alias("_vol_5"),
        pl.col("_mid").pct_change().abs().rolling_mean(window_size=20).over("sample_id").alias("_vol_20"),
    ])

    market_agg_exprs = [
        pl.col("_mid").last().alias("m_mid_last"),
        pl.col("_mid").mean().alias("m_mid_mean"),
        pl.col("_mid").std().alias("m_mid_std"),
        (pl.col("_mid").max() - pl.col("_mid").min()).alias("m_mid_range"),
        pl.col("_sp").last().alias("m_sp_last"),
        pl.col("_sp").mean().alias("m_sp_mean"),
        pl.col("_imb").last().alias("m_imb_last"),
        pl.col("_imb").mean().alias("m_imb_mean"),
        pl.col("_imb").std().alias("m_imb_std"),
        (pl.col("_dmid") ** 2).sum().sqrt().alias("m_rv"),
        pl.col("_ofi").sum().alias("m_ofi_sum"),
        pl.col("_ask_slope").mean().alias("m_ask_slope_mean"),
        pl.col("_bid_slope").mean().alias("m_bid_slope_mean"),
        pl.col("_mid").skew().alias("m_mid_skew"),
        pl.col("m_mid_weighted_60").first().alias("m_mid_weighted_60"),
        pl.col("m_mid_weighted_300").first().alias("m_mid_weighted_300"),
        pl.col("m_imb_weighted_60").first().alias("m_imb_weighted_60"),
        pl.col("m_imb_weighted_300").first().alias("m_imb_weighted_300"),
        pl.col("m_ofi_weighted_60").first().alias("m_ofi_weighted_60"),
        pl.col("m_ofi_weighted_300").first().alias("m_ofi_weighted_300"),
        pl.col("m_vol_weighted_60").first().alias("m_vol_weighted_60"),
        pl.col("m_vol_weighted_300").first().alias("m_vol_weighted_300"),
        pl.col("_vol_5").mean().alias("m_vol_short"),
        pl.col("_vol_20").mean().alias("m_vol_long"),
        (pl.col("_vol_5").mean() / (pl.col("_vol_20").mean() + 1e-8)).alias("m_vol_short_long_ratio"),
    ]

    for w in [60, 180]:
        cond = pl.col("seconds_before_predict") <= w
        market_agg_exprs.append(pl.col("_mid").filter(cond).mean().alias(f"m_mid_mean_{w}"))
        market_agg_exprs.append(pl.col("_mid").filter(cond).std().alias(f"m_mid_std_{w}"))
        market_agg_exprs.append(pl.col("_sp").filter(cond).mean().alias(f"m_sp_mean_{w}"))
        market_agg_exprs.append(pl.col("_imb").filter(cond).mean().alias(f"m_imb_mean_{w}"))
        market_agg_exprs.append(((pl.col("_dmid").filter(cond)) ** 2).sum().sqrt().alias(f"m_rv_{w}"))
        market_agg_exprs.append(pl.col("_ofi").filter(cond).sum().alias(f"m_ofi_sum_{w}"))
        market_agg_exprs.append(pl.col("transaction_volume").filter(cond).sum().alias(f"m_txv_sum_{w}"))
        market_agg_exprs.append(pl.col("_spread_ratio").filter(cond).mean().alias(f"m_spread_ratio_{w}"))
        if w == 60:
            market_agg_exprs.append((pl.col("_mid").filter(cond).max() - pl.col("_mid").filter(cond).min()).alias(f"m_mid_range_{w}"))

    for tau in [120]:
        w_expr = pl.col("seconds_before_predict").mul(-1.0 / tau).exp()
        market_agg_exprs.append(((w_expr * pl.col("_mid")).sum() / (w_expr.sum() + 1e-8)).alias(f"m_mid_ewm_{tau}"))
        market_agg_exprs.append(((w_expr * pl.col("_imb")).sum() / (w_expr.sum() + 1e-8)).alias(f"m_imb_ewm_{tau}"))
        market_agg_exprs.append(((w_expr * pl.col("_ofi")).sum() / (w_expr.sum() + 1e-8)).alias(f"m_ofi_ewm_{tau}"))
        market_agg_exprs.append(((w_expr * pl.col("_spread_ratio")).sum() / (w_expr.sum() + 1e-8)).alias(f"m_spread_ewm_{tau}"))

    market_result = market.group_by('sample_id').agg(market_agg_exprs).sort('sample_id')


    # ============================================================
    # 6. 合并所有数据
    # ============================================================
    print("6. 合并数据")
    result = trans_result.join(order_result, how='left', on='sample_id', suffix='_order')
    result = result.join(t_sec_result, how='left', on='sample_id', suffix='_sec')
    result = result.join(o_sec_result, how='left', on='sample_id', suffix='_sec_o')
    result = result.join(market_result, how='left', on='sample_id', suffix='_market')


    # ============================================================
    # 7. 交叉特征
    # ============================================================
    print("7. 生成交叉特征")
    result = result.with_columns([
        (pl.col("m_sp_mean") * pl.col("m_imb_mean")).alias("x_sp_imb"),
        (pl.col("o_sv_sum") / (pl.col("o_vol_sum") + 1.0)).alias("x_o_signed_ratio"),
        (pl.col("m_ofi_ewm_120") - pl.col("m_ofi_weighted_60")).alias("x_ofi_ewm_short_long"),
        (pl.col("m_imb_ewm_120") - pl.col("m_imb_weighted_60")).alias("x_imb_ewm_short_long"),
        (pl.col("t_sv_15") / (pl.col("t_vol_15") + 1.0)).alias("x_t_signed_ratio_15"),
        (pl.col("m_rv_60") / (pl.col("m_rv") + 1e-8)).alias("x_rv_60_over_full"),
        (pl.col("t_vol_sum") / (pl.col("o_vol_sum") + 1e-8)).alias("x_trans_order_vol_ratio"),
        ((pl.col("t_vwap") - pl.col("m_mid_last")) / (pl.col("m_mid_last") + 1e-8)).alias("x_vwap_mid_ratio"),
        (pl.col("m_rv_60") / (pl.col("m_rv_180") + 1e-8)).alias("x_rv_60_180_ratio"),
        (pl.col("t_buy_ratio") - pl.col("o_buy_ratio")).alias("x_trans_order_buy_diff"),
        ((pl.col("t_large_buy_90") - pl.col("t_large_sell_90")) / (pl.col("t_n_15") + 1e-8)).alias("x_large_trade_imbalance"),
        (pl.col("m_bid_slope_mean") - pl.col("m_ask_slope_mean")).alias("x_slope_imbalance"),
        (pl.col("t_buy_ratio_15") - pl.col("o_buy_ratio_15")).alias("x_trans_order_buy_diff_15"),
        (pl.col("t_vol_15") / (pl.col("t_vol_45") + 1e-8)).alias("x_t_vol_15_45_ratio"),
        (pl.col("t_price_momentum_10") / (pl.col("t_price_volatility") + 1e-8)).alias("x_sharpe_like"),
        (pl.col("m_mid_last") - pl.col("m_mid_mean")) / (pl.col("m_mid_std") + 1e-8).alias("x_mid_zscore"),
        # 时间交叉
        (pl.col("t_vol_weighted_15") / (pl.col("t_vol_15") + 1e-8)).alias("x_t_vol_weight_ratio_15"),
        (pl.col("t_vol_weighted_30") / (pl.col("t_vol_30") + 1e-8)).alias("x_t_vol_weight_ratio_30"),
        (pl.col("o_vol_weighted_15") / (pl.col("o_vol_15") + 1e-8)).alias("x_o_vol_weight_ratio_15"),
        (pl.col("o_vol_weighted_30") / (pl.col("o_vol_30") + 1e-8)).alias("x_o_vol_weight_ratio_30"),
        # 等间隔交易特征
        (pl.col("t_sec_price_near_far_diff_15")).alias("x_t_price_trend_15"),
        (pl.col("t_sec_price_near_far_diff_30")).alias("x_t_price_trend_30"),
        (pl.col("t_sec_vol_near_far_ratio_15") - 1.0).alias("x_t_vol_trend_15"),
        (pl.col("t_sec_vol_near_far_ratio_30") - 1.0).alias("x_t_vol_trend_30"),
        (pl.col("o_sec_cancel_near_far_ratio_15") - 1.0).alias("x_o_cancel_trend_15"),
        (pl.col("t_transaction_rate") / (pl.col("o_order_rate") + 1e-8)).alias("x_tx_order_rate_ratio"),
        (pl.col("t_transaction_count") / (pl.col("o_order_count") + 1e-8)).alias("x_tx_order_count_ratio"),
        (pl.col("t_time_gap_std") / (pl.col("t_avg_time_gap") + 1e-8)).alias("x_t_gap_stability"),
        (pl.col("o_time_gap_std") / (pl.col("o_avg_time_gap") + 1e-8)).alias("x_o_gap_stability"),
        (pl.col("m_mid_weighted_60") - pl.col("m_mid_weighted_300")).alias("x_m_mid_long_short_diff"),
        (pl.col("m_vol_weighted_60") / (pl.col("m_vol_weighted_300") + 1e-8)).alias("x_m_vol_long_short_ratio"),
        (pl.col("m_ofi_weighted_60") - pl.col("m_ofi_weighted_300")).alias("x_m_ofi_long_short_diff"),
        (pl.col("m_vol_short_long_ratio") - 1.0).alias("x_m_vol_regime_change"),
        (pl.col("m_rv_60") / (pl.col("m_rv_180") + 1e-8)).alias("x_m_rv_60_180_ratio"),
        # 使用替代特征
        (pl.col("t_sv_weighted_15") / (pl.col("t_vol_weighted_15") + 1e-8)).alias("x_t_signed_weighted_15"),
        (pl.col("t_sv_weighted_30") / (pl.col("t_vol_weighted_30") + 1e-8)).alias("x_t_signed_weighted_30"),
        (pl.col("o_sv_weighted_15") / (pl.col("o_vol_weighted_15") + 1e-8)).alias("x_o_signed_weighted_15"),
        (pl.col("o_sv_weighted_30") / (pl.col("o_vol_weighted_30") + 1e-8)).alias("x_o_signed_weighted_30"),
        ((pl.col("t_sv_weighted_15") / (pl.col("t_vol_weighted_15") + 1e-8)) -
         (pl.col("t_sv_15") / (pl.col("t_vol_15") + 1e-8))).alias("x_t_signed_weight_diff_15"),
        ((pl.col("o_sv_weighted_15") / (pl.col("o_vol_weighted_15") + 1e-8)) -
         (pl.col("o_sv_15") / (pl.col("o_vol_15") + 1e-8))).alias("x_o_signed_weight_diff_15"),
        (pl.col("m_imb_weighted_60") - pl.col("m_imb_weighted_300")).alias("x_m_imb_trend"),
        # 等间隔交叉特征
        (pl.col("t_sec_vol_sum") / (pl.col("o_sec_vol_sum") + 1e-8)).alias("x_sec_trans_order_vol_ratio"),
        (pl.col("t_sec_price_mean") - pl.col("m_mid_mean")).alias("x_sec_price_mid_diff"),
        (pl.col("t_sec_has_data_ratio") * pl.col("m_imb_mean")).alias("x_sec_has_data_imb"),
        (pl.col("o_sec_cancel_new_ratio") * pl.col("m_sp_mean")).alias("x_sec_cancel_spread"),
        (pl.col("t_sec_has_data_ratio") - pl.col("o_sec_has_data_ratio")).alias("x_sec_tx_order_data_ratio_diff"),
        (1.0 - pl.col("t_sec_has_data_ratio")).alias("x_sec_t_max_missing_ratio"),
        (1.0 - pl.col("o_sec_has_data_ratio")).alias("x_sec_o_max_missing_ratio"),
        (pl.col("t_sec_price_trend") / (pl.col("t_sec_price_std") + 1e-8)).alias("x_sec_price_trend_normalized"),
        (pl.col("t_sec_price_near_far_diff_15") - pl.col("t_sec_price_near_far_diff_30")).alias("x_sec_price_trend_acceleration"),
        (pl.col("o_sec_cancel_near_far_ratio_15") - 1.0).alias("x_sec_o_cancel_trend_15"),
        (pl.col("t_sec_total_row_count") / (pl.col("o_sec_total_row_count") + 1e-8)).alias("x_sec_tx_order_activity_ratio"),
    ])


    # ============================================================
    # 8. 删除不需要的特征
    # ============================================================
    print(f"8. 删除 {len(DROP_FEATURES)} 个特征")
    # 只删除存在于result中的特征
    cols_to_drop = [col for col in DROP_FEATURES if col in result.columns]
    result = result.drop(cols_to_drop)


    # ============================================================
    # 9. label
    # ============================================================
    print("9. 加载 label")
    if mode == 'train':
        label = pl.read_ipc(f'{BASE_PATH}/train/label.feather', memory_map=False)
        if start_id is not None or end_id is not None:
            label = label.filter(
                (pl.col('sample_id') >= (start_id if start_id is not None else 0)) &
                (pl.col('sample_id') <= (end_id if end_id is not None else pl.col('sample_id').max()))
            )
        result = result.join(label.select(['sample_id', 'target']), on='sample_id', how='left')

    result = result.unique()
    print(f"   最终结果: {result.shape}")

    if return_pandas:
        return result.to_pandas()
    return result


# ===== 主程序 =====
if __name__ == "__main__":
    print("开始处理训练集...")
    train = pd.concat([
        get_data('train', start_id=i*40000, end_id=(i+1)*40000)
        for i in range(31)
    ], axis=0)
    last = get_data('train', start_id=1240000)
    train = pd.concat([train, last], axis=0)
    
    print("开始处理测试集...")
    test = pd.concat([
        get_data('test', start_id=i*40000, end_id=(i+1)*40000)
        for i in range(16)
    ], axis=0)
    last = get_data('test', start_id=640000)
    test = pd.concat([test, last], axis=0)
    
    train.to_csv('train.csv', index=None)
    test.to_csv('test.csv', index=None)
    print(f"训练集: {train.shape}")
    print(f"测试集: {test.shape}")