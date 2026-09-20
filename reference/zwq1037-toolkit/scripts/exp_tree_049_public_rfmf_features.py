"""Evaluate the public RFMF-0726 152-feature table under the EXP037 protocol."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)
from exp_tree_027_031_xgboost_capacity_regularization import (
    evaluate_and_save,
    fit_model,
    model_parameters,
)


EXPERIMENT_ID = "EXP-TREE-049"
PUBLIC_CSV_RELATIVE = Path(
    "data/interim/public_features/rfmf_0726data/train.csv"
)


def load_public_features(project_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load the public CSV as compact numeric columns and verify its schema."""

    csv_path = project_dir / PUBLIC_CSV_RELATIVE
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    feature_columns = [name for name in header if name not in {"sample_id", "target"}]
    if len(header) != 154 or len(feature_columns) != 152:
        raise AssertionError(
            f"Expected sample_id + 152 features + target, got {len(header)} columns."
        )

    # 使用 float32 读取，降低 2600 MB CSV 展开后的内存占用。
    # Read as float32 to limit the expanded memory footprint of the 2600 MB CSV.
    dtypes = {name: np.float32 for name in feature_columns + ["target"]}
    dtypes["sample_id"] = np.int32
    public = pd.read_csv(csv_path, dtype=dtypes)
    if len(public) != 1_257_637 or public["sample_id"].nunique() != len(public):
        raise AssertionError(f"Unexpected public feature table shape: {public.shape}.")
    if np.isinf(public[feature_columns].to_numpy(copy=False)).any():
        raise AssertionError("Public feature table contains infinity.")
    return public, feature_columns


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    public, feature_columns = load_public_features(project_dir)
    public = public.rename(columns={"target": "public_target"})
    labels = pd.read_feather(
        project_dir / "data" / "raw" / "label.feather",
        columns=["sample_id", "month", "target"],
    )
    model_data = public.merge(labels, on="sample_id", how="inner", validate="one_to_one")
    model_data = model_data.sort_values("sample_id").reset_index(drop=True)
    target_difference = np.max(
        np.abs(
            model_data["public_target"].to_numpy(dtype=np.float64)
            - model_data["target"].to_numpy(dtype=np.float64)
        )
    )
    if target_difference > 1e-5:
        raise AssertionError(
            f"Public target does not match the official label: {target_difference}."
        )
    model_data = model_data.drop(columns="public_target")
    del public, labels
    gc.collect()

    # 完全复用 EXP037 的月份划分、目标去均值和 XGBoost 参数。
    # Reuse EXP037's month split, centered target, and XGBoost parameters exactly.
    train_mask = model_data["month"] <= 59
    valid_mask = model_data["month"] >= 60
    centered_target = model_data.loc[train_mask, "target"].copy()
    centered_target -= float(centered_target.mean())
    validation_rows = model_data.loc[
        valid_mask, ["sample_id", "month", "target"]
    ].copy()
    baseline_predictions = pd.read_feather(
        project_dir / "outputs" / "predictions" / "exp-tree-037_valid.feather"
    )
    assert_prediction_alignment(baseline_predictions, validation_rows)

    parameters = model_parameters(n_estimators=800, colsample_bytree=0.8)
    model, training_seconds = fit_model(
        model_data,
        feature_columns,
        train_mask,
        centered_target,
        parameters,
    )
    predictions = model.predict(model_data.loc[valid_mask, feature_columns])
    evaluate_and_save(
        project_dir=project_dir,
        experiment_id=EXPERIMENT_ID,
        description=(
            "public RFMF-0726data 152 features only; EXP037 split, centered target, "
            "and XGBoost parameters"
        ),
        model=model,
        predictions=predictions,
        validation_rows=validation_rows,
        baseline_predictions=baseline_predictions,
        feature_columns=feature_columns,
        parameters=parameters,
        training_seconds=training_seconds,
        baseline_id="EXP-TREE-037",
    )


if __name__ == "__main__":
    main()
