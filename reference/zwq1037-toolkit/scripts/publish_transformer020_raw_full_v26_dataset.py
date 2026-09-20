"""Publish the verified v26 full Transformer checkpoint and CSV as a private Kaggle dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/interim/kaggle_results/transformer020_raw_full_v26"
STAGING = ROOT / "data/interim/kaggle_datasets/mscapital-transformer020-raw-full-v26"
CLI = r"D:\anaconda\envs\pytorch\Scripts\kaggle.exe"
DATASET_ID = "zwq1037/mscapital-transformer020-raw-full-v26"


def one(pattern: str) -> Path:
    matches = list(SOURCE.rglob(pattern))
    if len(matches) != 1:
        raise AssertionError(f"Expected one {pattern}, found {len(matches)}")
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checkpoint = one("factorized_transformer379_raw_full_e5.pt")
    prediction = one("standalone_factorized_transformer379_raw_full_e5.csv")
    result = one("result.json")
    result_data = json.loads(result.read_text(encoding="utf-8"))
    if result_data.get("status") != "complete":
        raise AssertionError("v26 result is not complete")
    if result_data.get("visible_gpu_count") != 2:
        raise AssertionError("v26 was not trained on dual T4")

    prediction_frame = pd.read_csv(prediction)
    if list(prediction_frame.columns) != ["sample_id", "prediction"]:
        raise AssertionError(
            f"Unexpected v26 CSV columns: {list(prediction_frame.columns)}"
        )
    if len(prediction_frame) != 647_896:
        raise AssertionError(f"Unexpected v26 CSV rows: {len(prediction_frame)}")
    if prediction_frame.sample_id.duplicated().any():
        raise AssertionError("v26 CSV contains duplicate sample IDs")
    if not np.isfinite(
        prediction_frame.prediction.to_numpy(np.float64)
    ).all():
        raise AssertionError("v26 CSV contains nonfinite predictions")

    checkpoint_data = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    required_checkpoint_keys = {"model", "target_scale", "epochs"}
    missing_checkpoint_keys = required_checkpoint_keys.difference(
        checkpoint_data
    )
    if missing_checkpoint_keys:
        raise AssertionError(
            f"v26 checkpoint missing keys: {sorted(missing_checkpoint_keys)}"
        )

    artifact_sha256 = {
        checkpoint.name: sha256(checkpoint),
        prediction.name: sha256(prediction),
        result.name: sha256(result),
    }

    STAGING.mkdir(parents=True, exist_ok=True)
    for source in (checkpoint, prediction, result):
        shutil.copy2(source, STAGING / source.name)
    metadata = {
        "title": "MSCapital Transformer020 Raw Full v26",
        "id": DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (STAGING / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    process = subprocess.run(
        [CLI, "datasets", "create", "-p", str(STAGING), "--dir-mode", "skip"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    print(process.stdout, flush=True)
    if process.returncode:
        raise RuntimeError(process.stdout + process.stderr)
    report = {
        "status": "created",
        "dataset": DATASET_ID,
        "checkpoint": checkpoint.name,
        "prediction": prediction.name,
        "result": result.name,
        "private_by_default": True,
        "csv_rows": len(prediction_frame),
        "artifact_sha256": artifact_sha256,
    }
    (
        ROOT
        / "outputs/submission_metadata/transformer020_raw_full_v26_dataset.json"
    ).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()


