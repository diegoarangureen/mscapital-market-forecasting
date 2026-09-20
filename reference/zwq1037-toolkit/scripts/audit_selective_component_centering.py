"""Audit selectively centering only sources that benefited independently."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_multimodel_component_centering import (
    BLEND_WEIGHTS,
    SOURCE_FILES,
    centered_unit,
    cosine,
    unit,
)
from exp_tree_017_019_xgboost_target_and_monthly_transforms import (
    assert_prediction_alignment,
)


def evaluate(
    target: np.ndarray,
    prediction: np.ndarray,
    months: np.ndarray,
    baseline: np.ndarray,
) -> dict:
    month_values = np.sort(np.unique(months))
    monthly = np.asarray(
        [
            cosine(target[months == month], prediction[months == month])
            for month in month_values
        ]
    )
    primary_month_mask = (month_values >= 62) & (month_values != 66)
    primary = monthly[primary_month_mask]
    lomo = []
    for omitted_month in month_values[primary_month_mask]:
        mask = (months >= 62) & (months != 66) & (months != omitted_month)
        lomo.append(
            cosine(target[mask], prediction[mask])
            - cosine(target[mask], baseline[mask])
        )
    return {
        "overall": cosine(target, prediction),
        "no66": cosine(
            target[(months >= 62) & (months != 66)],
            prediction[(months >= 62) & (months != 66)],
        ),
        "recent": cosine(target[months >= 67], prediction[months >= 67]),
        "early": cosine(target[months <= 64], prediction[months <= 64]),
        "primary_std": float(primary.std(ddof=0)),
        "primary_worst": float(primary.min()),
        "primary_q25": float(np.quantile(primary, 0.25)),
        "primary_lomo_min": float(np.min(lomo)),
        "primary_lomo_mean": float(np.mean(lomo)),
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    prediction_dir = project_dir / "outputs" / "predictions"
    frames = {
        name: pd.read_feather(prediction_dir / file_name)
        for name, file_name in SOURCE_FILES.items()
    }
    reference = frames["tabm_original"]
    validation_rows = reference[["sample_id", "month", "target"]]
    target = reference["target"].to_numpy(dtype=np.float64)
    months = reference["month"].to_numpy()
    raw = {}
    centered = {}
    for name, frame in frames.items():
        assert_prediction_alignment(frame, validation_rows)
        values = frame["prediction"].to_numpy(dtype=np.float64)
        raw[name] = unit(values)
        centered[name] = centered_unit(values)

    baseline = sum(BLEND_WEIGHTS[name] * raw[name] for name in BLEND_WEIGHTS)
    center_sets = {
        "raw_blend005": set(),
        "center_tabm_xgb_lgbm": {"tabm_original", "xgboost", "lightgbm"},
        "center_tabm_xgb_lgbm_corr": {
            "tabm_original", "tabm_corrprune", "xgboost", "lightgbm"
        },
        "center_all": set(BLEND_WEIGHTS),
    }
    rows = []
    for variant, centered_names in center_sets.items():
        prediction = sum(
            BLEND_WEIGHTS[name]
            * (centered[name] if name in centered_names else raw[name])
            for name in BLEND_WEIGHTS
        )
        rows.append(
            {
                "variant": variant,
                "centered_sources": ",".join(sorted(centered_names)),
                **evaluate(target, prediction, months, baseline),
            }
        )

    run_dir = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / "EXP-POST-005-SELECTIVE-CENTERING"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(run_dir / "selective_centering_results.csv", index=False)
    (run_dir / "audit.json").write_text(
        json.dumps(
            {
                "weights": BLEND_WEIGHTS,
                "selection_basis": (
                    "center only sources whose independent overall/no66/recent "
                    "all improved in EXP-POST-004"
                ),
                "uses_labels_for_centering_value": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
