"""Cheap gate before spending GPU on a multiwindow + EMA rerun."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/interim/tree_experiments/EXP-TRANSFORMER-020-FACTORIZED-EMA/train059_valid6270_ex66/validation_predictions.feather"
MULTI = ROOT / "data/interim/kaggle_outputs/multistream_transformer_multiwindow_v14/factorized_transformer_multiwindow10/validation_predictions.csv"
OUT = ROOT / "outputs/submission_metadata/multiwindow_vs_transformer020_screen_20260919.json"


def cosine(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(y @ p / (np.linalg.norm(y) * np.linalg.norm(p) + 1e-30))


def score(y, p, month):
    selection = np.isin(month, [62, 63, 64, 65])
    forward = np.isin(month, [67, 68, 69, 70])
    return {
        "selection": cosine(y[selection], p[selection]),
        "forward": cosine(y[forward], p[forward]),
        "all": cosine(y, p),
        "monthly": {str(value): cosine(y[month == value], p[month == value]) for value in sorted(set(month))},
    }


base = pd.read_feather(BASE)[["sample_id", "month", "target", "raw"]]
multi = pd.read_csv(MULTI, usecols=["sample_id", "prediction"]).rename(columns={"prediction": "multiwindow"})
frame = base.merge(multi, on="sample_id", how="inner", validate="one_to_one")
if len(frame) != 140_806:
    raise AssertionError(len(frame))
y = frame.target.to_numpy(np.float64)
month = frame.month.to_numpy()
selection = np.isin(month, [62, 63, 64, 65])
raw = frame.raw.to_numpy(np.float64)
multiwindow = frame.multiwindow.to_numpy(np.float64)
raw /= np.sqrt(np.mean(raw[selection] ** 2))
multiwindow /= np.sqrt(np.mean(multiwindow[selection] ** 2))
baseline = score(y, raw, month)
rows = []
for fraction in [0.0, 0.20, 0.35, 0.50]:
    prediction = (1.0 - fraction) * raw + fraction * multiwindow
    current = score(y, prediction, month)
    current["multiwindow_fraction"] = fraction
    current["delta"] = {name: current[name] - baseline[name] for name in ("selection", "forward", "all")}
    current["monthly_delta"] = {key: current["monthly"][key] - baseline["monthly"][key] for key in baseline["monthly"]}
    rows.append(current)
selected = max(rows, key=lambda item: item["selection"])
passed = bool(
    selected["multiwindow_fraction"] > 0
    and selected["delta"]["selection"] >= 0.0005
    and selected["delta"]["forward"] >= 0.0005
    and selected["delta"]["all"] >= 0.0007
    and selected["monthly_delta"]["63"] >= -0.002
    and selected["monthly_delta"]["68"] >= -0.002
)
report = {
    "experiment": "MULTIWINDOW-VS-TRANSFORMER020-PRESCREEN",
    "baseline": baseline,
    "rows": rows,
    "selected_by_62_65": selected,
    "passed": passed,
    "decision": "train multiwindow with EMA only if passed",
}
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({"passed": passed, "selected": selected}, indent=2))
