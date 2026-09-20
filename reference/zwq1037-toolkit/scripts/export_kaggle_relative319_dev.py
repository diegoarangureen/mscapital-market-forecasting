"""Export the leakage-safe Relative319 training matrix for a private Kaggle run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import exp_tabm_001_exp053r_features as tabm
from exp_tabm_016_relative_scale_features import RELATIVE_COLUMNS, add_relative_features


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    output_dir = project_dir / "data" / "interim" / "kaggle_relative319_dev"
    output_dir.mkdir(parents=True, exist_ok=True)

    data, base_columns, _, _ = tabm.load_exp053r_data(project_dir)
    add_relative_features(project_dir, data)
    feature_columns = [*base_columns, *RELATIVE_COLUMNS]
    if len(feature_columns) != 319 or len(set(feature_columns)) != 319:
        raise AssertionError("Expected 319 unique Relative319 features.")

    features = data[feature_columns].to_numpy(dtype=np.float32, copy=True)
    sample_ids = data["sample_id"].to_numpy(dtype=np.int64, copy=True)
    months = data["month"].to_numpy(dtype=np.int16, copy=True)
    targets = data["target"].to_numpy(dtype=np.float32, copy=True)
    if features.shape != (len(sample_ids), 319):
        raise AssertionError("Unexpected Relative319 matrix shape.")
    if np.any(months[1:] < months[:-1]):
        raise AssertionError("Expected sample order to preserve month order.")

    np.save(output_dir / "features.npy", features, allow_pickle=False)
    np.save(output_dir / "sample_ids.npy", sample_ids, allow_pickle=False)
    np.save(output_dir / "months.npy", months, allow_pickle=False)
    np.save(output_dir / "targets.npy", targets, allow_pickle=False)
    (output_dir / "feature_columns.json").write_text(
        json.dumps(feature_columns, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "title": "MSCapital Relative319 Development Features",
        "id": "zwq1037/mscapital-relative319-dev",
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }
    (output_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "shape": list(features.shape),
                "months": [int(months.min()), int(months.max())],
                "bytes": int(features.nbytes),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
