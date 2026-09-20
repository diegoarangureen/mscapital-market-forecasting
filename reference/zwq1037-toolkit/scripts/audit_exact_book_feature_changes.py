"""Audit how exact book reconstructions differ from cached feature values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from build_train_market_microstructure_features import BOOK_FEATURE_COLUMNS


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    processed_dir = project_dir / "data" / "processed"
    cached = pd.read_feather(
        processed_dir / "train_market_microstructure_features.feather",
        columns=["sample_id", *BOOK_FEATURE_COLUMNS],
    ).sort_values("sample_id")
    exact = pd.read_feather(
        processed_dir / "train_market_book_features_exact.feather"
    ).sort_values("sample_id")
    if not np.array_equal(cached["sample_id"].to_numpy(), exact["sample_id"].to_numpy()):
        raise AssertionError("Cached and exact book rows are not aligned.")

    model = xgb.XGBRegressor()
    model.load_model(project_dir / "outputs" / "models" / "exp-tree-053r.json")
    booster = model.get_booster()
    gain = booster.get_score(importance_type="total_gain")
    split_count = booster.get_score(importance_type="weight")
    total_gain = sum(gain.values())

    rows: list[dict[str, float | str]] = []
    for feature in BOOK_FEATURE_COLUMNS:
        cached_values = cached[feature].to_numpy(dtype=np.float64)
        exact_values = exact[feature].to_numpy(dtype=np.float64)
        cached_finite = np.isfinite(cached_values)
        exact_finite = np.isfinite(exact_values)
        common = cached_finite & exact_finite
        if int(common.sum()) >= 2:
            correlation = float(np.corrcoef(cached_values[common], exact_values[common])[0, 1])
            mean_absolute_change = float(
                np.mean(np.abs(exact_values[common] - cached_values[common]))
            )
            cached_std = float(np.std(cached_values[common]))
            normalized_mae = mean_absolute_change / (cached_std + 1e-12)
        else:
            correlation = np.nan
            mean_absolute_change = np.nan
            normalized_mae = np.nan
        changed = ~np.isclose(
            cached_values,
            exact_values,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=True,
        )
        gain_share = float(gain.get(feature, 0.0) / total_gain) if total_gain else 0.0
        # 同时重视模型依赖度和重算后变化幅度，用于缩小实验范围。
        # Combine model reliance and reconstruction change to narrow the test budget.
        priority = gain_share * min(float(normalized_mae), 5.0) * float(changed.mean())
        rows.append(
            {
                "feature": feature,
                "correlation": correlation,
                "changed_fraction": float(changed.mean()),
                "mean_absolute_change": mean_absolute_change,
                "normalized_mae": normalized_mae,
                "cached_missing_fraction": float((~cached_finite).mean()),
                "exact_missing_fraction": float((~exact_finite).mean()),
                "model_total_gain_share": gain_share,
                "model_split_count": float(split_count.get(feature, 0.0)),
                "priority": priority,
            }
        )

    audit = pd.DataFrame(rows).sort_values("priority", ascending=False)
    output_path = (
        project_dir
        / "data"
        / "interim"
        / "exact_book_feature_change_audit.csv"
    )
    audit.to_csv(output_path, index=False)
    print(audit.head(20).to_string(index=False), flush=True)
    print(f"output={output_path}", flush=True)


if __name__ == "__main__":
    main()
