"""Create the static single-variable TabM397 experiment from TabM032."""
import ast
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
source = (PROJECT/'scripts/exp_tabm_032_market385_k32.py').read_text(encoding='utf-8')
source = source.replace('"""Single-variable TabM385 validation for k=32 versus k=16."""',
'''"""Append the 12 exact-name-new Yunsu features to frozen TabM385-k32."""''')
source = source.replace('EXPERIMENT_ID = "EXP-TABM-032-MARKET385-K32"',
                        'EXPERIMENT_ID = "EXP-TABM-037-UNION397-K32"')
anchor = 'FEATURE_COLUMNS = "mid_return_180 mid_return_60 mid_return_20 realized_volatility_20 realized_volatility_60 mid_momentum_60".split()\n'
addition = '''PUBLIC_CACHE = PROJECT_DIR / "data/interim/public_features/rfmf_0726data/numpy_cache"
PUBLIC_EXTRA_COLUMNS = [
    "t_avg_signed_vol", "t_large_buy_90", "t_large_sell_95",
    "t_avg_time_gap", "t_time_gap_std", "t_max_time_gap",
    "m_imb_last", "m_imb_std", "m_sp_mean_60", "m_imb_mean_60",
    "x_large_trade_imbalance", "x_sec_price_mid_diff",
]
'''
assert anchor in source
source = source.replace(anchor, anchor+addition)
anchor = '    del state, order_frame, labels\n    gc.collect()\n\n'
addition = '''    public_names = json.loads((PUBLIC_CACHE / "feature_columns.json").read_text(encoding="utf-8"))
    if len(PUBLIC_EXTRA_COLUMNS) != 12 or not set(PUBLIC_EXTRA_COLUMNS).issubset(public_names):
        raise AssertionError("Expected 12 public-only feature columns")
    public_memmap = np.load(PUBLIC_CACHE / "features.npy", mmap_mode="r")
    if public_memmap.shape != (len(sample_ids), len(public_names)):
        raise AssertionError("Public feature cache rows do not align")
    public_indices = [public_names.index(name) for name in PUBLIC_EXTRA_COLUMNS]
    public_extra = np.asarray(public_memmap[:, public_indices], dtype=np.float32)
    del public_memmap

'''
assert anchor in source
source = source.replace(anchor, anchor+addition)
source = source.replace('if len(set([*base_columns, *xs_columns, *ORDER_COLUMNS, *FEATURE_COLUMNS])) != 385:',
                        'if len(set([*base_columns, *xs_columns, *ORDER_COLUMNS, *FEATURE_COLUMNS, *PUBLIC_EXTRA_COLUMNS])) != 397:')
source = source.replace('raise AssertionError("Expected 385 unique candidate columns.")',
                        'raise AssertionError("Expected 397 unique candidate columns.")')
source = source.replace('/ "EXP-TABM-031-MARKET-TRAJECTORY6"',
                        '/ "EXP-TABM-032-MARKET385-K32"')
source = source.replace('np.concatenate([base, xs40, order_values, state_values], axis=1)',
                        'np.concatenate([base, xs40, order_values, state_values, public_extra], axis=1)')
source = source.replace('if candidate_features.shape[1] != 385:', 'if candidate_features.shape[1] != 397:')
source = source.replace('f"{fold_name}_relative319_xs40_order20_market_trajectory6"',
                        'f"{fold_name}_tabm385_plus_public12"')
source = source.replace('"baseline": "TabM385 k16, seed42, 15 epochs",',
                        '"baseline": "TabM385 k32, seed42, 15 epochs",')
source = source.replace('"candidate": "TabM385 with only k changed from 16 to 32",',
                        '"candidate": "TabM385 k32 plus 12 exact-name-new Yunsu features",')
source = source.replace('"baseline_feature_count": 385,\n        "candidate_feature_count": 385,',
                        '"baseline_feature_count": 385,\n        "candidate_feature_count": 397,')
source = source.replace('"single_variable_change": "k=16 to k=32",\n        "added_features": [],',
                        '"single_variable_change": "append public-only 12 features",\n        "added_features": PUBLIC_EXTRA_COLUMNS,')
source = source.replace('passed = no_window_decline and average_delta >= 0.0010',
                        'passed = no_window_decline and average_delta >= 0.0007')
source = source.replace('"pass_threshold": 0.0010,', '"pass_threshold": 0.0007,')
source = source.replace('"confirm_with_seed137" if passed else "reject_feature_family"',
                        '"paired_early_window" if passed else "reject_union_features"')
ast.parse(source)
output = PROJECT/'scripts/exp_tabm_037_union397_k32.py'
output.write_text(source, encoding='utf-8')
print(f'prepared {output}')
