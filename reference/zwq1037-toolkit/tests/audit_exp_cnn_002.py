from pathlib import Path

import numpy as np
import pandas as pd


def cosine_score(targets, predictions):
    numerator = np.dot(targets, predictions)
    denominator = np.linalg.norm(targets) * np.linalg.norm(predictions)
    return float(numerator / denominator)


project_dir = Path(__file__).resolve().parents[1]
baseline_path = project_dir / "data/interim/lesson_06c_full_cnn_runs/full_20260909_133512/best_valid_predictions.feather"
weighted_path = project_dir / "data/interim/lesson_06d_recency_weighted_cnn_runs/full_20260909_163406/best_valid_predictions.feather"

baseline = pd.read_feather(baseline_path)
weighted = pd.read_feather(weighted_path)
assert np.array_equal(baseline["sample_id"], weighted["sample_id"])
assert np.array_equal(baseline["target"], weighted["target"])

targets = weighted["target"].to_numpy(dtype=np.float64)
baseline_predictions = baseline["prediction"].to_numpy(dtype=np.float64)
weighted_predictions = weighted["prediction"].to_numpy(dtype=np.float64)
months = weighted["month"].to_numpy(dtype=np.int64)

baseline_cosine = cosine_score(targets, baseline_predictions)
weighted_cosine = cosine_score(targets, weighted_predictions)
print(f"baseline cosine = {baseline_cosine:.10f}")
print(f"weighted cosine = {weighted_cosine:.10f}")
print(f"overall delta = {weighted_cosine - baseline_cosine:+.10f}")
print(f"prediction correlation = {np.corrcoef(baseline_predictions, weighted_predictions)[0, 1]:.10f}")
print(f"target mean / std = {targets.mean():.10f} / {targets.std():.10f}")
print(f"weighted prediction mean / std = {weighted_predictions.mean():.10f} / {weighted_predictions.std():.10f}")
print(f"weighted prediction MSE = {np.mean((targets - weighted_predictions) ** 2):.12f}")
print(f"zero prediction MSE = {np.mean(targets ** 2):.12f}")
print(f"weighted prediction Pearson = {np.corrcoef(targets, weighted_predictions)[0, 1]:.10f}")
print("month, baseline, weighted, delta")

baseline_month_scores = []
weighted_month_scores = []
for month in np.unique(months):
    month_mask = months == month
    baseline_month = cosine_score(targets[month_mask], baseline_predictions[month_mask])
    weighted_month = cosine_score(targets[month_mask], weighted_predictions[month_mask])
    baseline_month_scores.append(baseline_month)
    weighted_month_scores.append(weighted_month)
    print(f"{month}, {baseline_month:.10f}, {weighted_month:.10f}, {weighted_month - baseline_month:+.10f}")

print(f"baseline macro monthly mean = {np.mean(baseline_month_scores):.10f}")
print(f"weighted macro monthly mean = {np.mean(weighted_month_scores):.10f}")
print(f"baseline negative months = {np.sum(np.asarray(baseline_month_scores) < 0)}")
print(f"weighted negative months = {np.sum(np.asarray(weighted_month_scores) < 0)}")
