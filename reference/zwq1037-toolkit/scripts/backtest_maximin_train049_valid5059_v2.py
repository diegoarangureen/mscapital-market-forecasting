"""Corrected gain-ranked rerun of the strict Maximin 50-59 backtest.

This wrapper maps XGBoost's NumPy feature keys (f0, f1, ...) back to the
EXP053R column names. It reuses the completed cosine-TabM and LightGBM
predictions, and retrains only the correlation-pruned TabM component.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import backtest_maximin_train049_valid5059 as experiment


EXPERIMENT_ID = "EXP-BACKTEST-003-MAXIMIN-GAINRANKED-TRAIN049-VALID5059"
ORIGINAL_SELECT = experiment.select_correlated_features


class _MappedBooster:
    def __init__(self, booster, feature_columns: list[str]) -> None:
        self.booster = booster
        self.feature_columns = feature_columns

    def get_score(self, importance_type: str):
        raw = self.booster.get_score(importance_type=importance_type)
        mapped: dict[str, float] = {}
        for key, value in raw.items():
            if key in self.feature_columns:
                mapped[key] = float(value)
            elif key.startswith("f") and key[1:].isdigit():
                index = int(key[1:])
                if index < len(self.feature_columns):
                    mapped[self.feature_columns[index]] = float(value)
        return mapped


class _MappedImportanceModel:
    def __init__(self, model, feature_columns: list[str]) -> None:
        self.model = model
        self.feature_columns = feature_columns

    def get_booster(self):
        return _MappedBooster(self.model.get_booster(), self.feature_columns)


def select_correlated_features(
    features,
    feature_columns,
    months,
    train_end,
    importance_model,
):
    mapped_model = _MappedImportanceModel(importance_model, feature_columns)
    return ORIGINAL_SELECT(
        features,
        feature_columns,
        months,
        train_end,
        mapped_model,
    )


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    old_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / experiment.EXPERIMENT_ID
    )
    new_dir = old_dir.parent / EXPERIMENT_ID
    new_dir.mkdir(parents=True, exist_ok=True)

    # These components are independent of correlation pruning and are safe to
    # reuse byte-for-byte from the completed strict temporal run.
    reusable = [
        "cosine_predictions.feather",
        "cosine_details.json",
        "lightgbm_predictions.feather",
        "lightgbm_details.json",
    ]
    for name in reusable:
        source = old_dir / name
        destination = new_dir / name
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)

    experiment.EXPERIMENT_ID = EXPERIMENT_ID
    experiment.select_correlated_features = select_correlated_features
    experiment.main()


if __name__ == "__main__":
    main()
