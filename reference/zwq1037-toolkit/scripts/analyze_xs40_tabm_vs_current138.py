"""Compare the validated XS40 TabM to the current 0.138 submission recipe."""

from pathlib import Path

import numpy as np
import pandas as pd

from analyze_conv_hybrid_transformer_full_blend import load_window
from exp_tabm_014_quantile_stage_curves import unit


PROJECT_DIR = Path(__file__).resolve().parents[1]
PREDICTION_DIR = (
    PROJECT_DIR / "data" / "interim" / "kaggle_kernels"
    / "relative319_xs_tabm_notebook" / "output_v1"
)


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.dot(target, prediction)
        / (np.linalg.norm(target) * np.linalg.norm(prediction))
    )


def main() -> None:
    rows = []
    for name, fold, start, end in (
        ("50-59", "train049_valid5059", 50, 59),
        ("62-70-no66", "train059_valid6070", 60, 70),
    ):
        current, components, target, months, _ = load_window(
            fold, start, end
        )
        tabm = pd.read_csv(
            PREDICTION_DIR / f"{fold}_predictions.csv"
        )
        tree_dir = PROJECT_DIR / "data" / "interim" / "tree_experiments"
        reference = pd.read_feather(
            tree_dir / "EXP-TABM-016-RELATIVE-SCALE-ADD12"
            / fold / "validation_predictions.feather",
            columns=["sample_id"],
        )
        if not np.array_equal(
            tabm["sample_id"].to_numpy(), reference["sample_id"].to_numpy()
        ):
            raise AssertionError("TabM OOF IDs differ.")
        transformer = (
            0.335 * components["hybrid42"]
            + 0.335 * components["hybrid137"]
            + 0.330 * components["conv42"]
        )
        current138 = 0.70 * unit(current) + 0.30 * unit(transformer)
        xs40 = tabm["relative319_xs40"].to_numpy(dtype=np.float64)
        baseline_tabm = tabm["relative319"].to_numpy(dtype=np.float64)
        mask = (
            np.ones(len(months), dtype=bool)
            if start == 50 else (months >= 62) & (months != 66)
        )
        target = target[mask]
        current138 = current138[mask]
        xs40 = xs40[mask]
        baseline_tabm = baseline_tabm[mask]
        for weight in (0.0, 0.10, 0.20, 0.30):
            candidate = (
                (1.0 - weight) * unit(current138)
                + weight * unit(xs40)
            )
            rows.append({
                "window": name,
                "xs40_weight": weight,
                "rows": int(mask.sum()),
                "current138": cosine(target, current138),
                "xs40": cosine(target, xs40),
                "tabm319": cosine(target, baseline_tabm),
                "blend": cosine(target, candidate),
                "delta": cosine(target, candidate)
                - cosine(target, current138),
                "corr_with_current138": float(np.corrcoef(
                    xs40, current138
                )[0, 1]),
            })
    output = pd.DataFrame(rows)
    output_path = (
        PROJECT_DIR / "outputs"
        / "xs40_tabm_current138_blend_comparison.csv"
    )
    output.to_csv(output_path, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()