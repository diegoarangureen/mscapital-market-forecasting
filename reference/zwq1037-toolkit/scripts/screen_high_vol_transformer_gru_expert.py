"""Pre-screen a fixed high-volatility Transformer/GRU expert before GPU training."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/submission_metadata/high_vol_transformer_gru_prescreen_20260919.json"


def cosine(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(y @ p / (np.linalg.norm(y) * np.linalg.norm(p) + 1e-30))


def score(y, p, months):
    selection = np.isin(months, [62, 63, 64, 65])
    forward = np.isin(months, [67, 68, 69, 70])
    return {
        "selection": cosine(y[selection], p[selection]),
        "forward": cosine(y[forward], p[forward]),
        "all": cosine(y, p),
        "monthly": {str(month): cosine(y[months == month], p[months == month]) for month in sorted(set(months))},
    }


labels = pd.read_feather(ROOT / "data/raw/label.feather", columns=["sample_id", "month"])
rv = pd.read_feather(
    ROOT / "data/processed/train_market_microstructure_features.feather",
    columns=["sample_id", "realized_volatility_60"],
)
train_rv = labels.loc[labels.month.le(59), ["sample_id"]].merge(rv, on="sample_id", how="left", validate="one_to_one")
finite_train = train_rv.realized_volatility_60.to_numpy(np.float64)
finite_train = finite_train[np.isfinite(finite_train)]
threshold = float(np.quantile(finite_train, 0.75))
median = float(np.median(finite_train))

transformer = pd.read_feather(
    ROOT / "data/interim/tree_experiments/EXP-TRANSFORMER-020-FACTORIZED-EMA/train059_valid6270_ex66/validation_predictions.feather"
)[["sample_id", "month", "target", "raw"]]
gru = pd.read_csv(
    ROOT / "data/interim/kaggle_outputs/multistream_factorized_gru_timeaware_dev/factorized_gru/validation_predictions.csv",
    usecols=["sample_id", "prediction"],
).rename(columns={"prediction": "gru"})
frame = transformer.merge(gru, on="sample_id", how="inner", validate="one_to_one")
frame = frame.merge(rv, on="sample_id", how="left", validate="one_to_one")
if len(frame) != 140_806:
    raise AssertionError(len(frame))

y = frame.target.to_numpy(np.float64)
months = frame.month.to_numpy()
selection = np.isin(months, [62, 63, 64, 65])
base = frame.raw.to_numpy(np.float64)
other = frame.gru.to_numpy(np.float64)
base /= np.sqrt(np.mean(base[selection] ** 2))
other /= np.sqrt(np.mean(other[selection] ** 2))
volatility = np.nan_to_num(frame.realized_volatility_60.to_numpy(np.float64), nan=median, posinf=median, neginf=median)
high = volatility >= threshold

predictions = {
    "transformer": base,
    "global_gru20": 0.8 * base + 0.2 * other,
    "high_vol_gru20": np.where(high, 0.8 * base + 0.2 * other, base),
}
metrics = {name: score(y, prediction, months) for name, prediction in predictions.items()}
baseline = metrics["transformer"]
conditional = metrics["high_vol_gru20"]
global_control = metrics["global_gru20"]
delta = {name: conditional[name] - baseline[name] for name in ("selection", "forward", "all")}
versus_global = {name: conditional[name] - global_control[name] for name in ("selection", "forward", "all")}
monthly_delta = {month: conditional["monthly"][month] - baseline["monthly"][month] for month in baseline["monthly"]}
passed = bool(
    delta["selection"] >= 0.0003
    and delta["forward"] >= 0.0003
    and versus_global["selection"] > 0
    and versus_global["forward"] > 0
    and min(monthly_delta.values()) >= -0.002
)
report = {
    "experiment": "HIGH-VOL-TRANSFORMER-GRU-PRESCREEN",
    "threshold_source": "train months0-59 q75 of realized_volatility_60",
    "threshold": threshold,
    "train_median": median,
    "validation_high_fraction": float(high.mean()),
    "metrics": metrics,
    "conditional_delta_vs_transformer": delta,
    "conditional_delta_vs_global20": versus_global,
    "monthly_delta_vs_transformer": monthly_delta,
    "passed": passed,
    "decision": "train a volatility-weighted Transformer expert only if passed",
}
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
