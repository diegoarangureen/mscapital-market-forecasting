"""Verify v26 artifacts locally without uploading them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/interim/kaggle_results/transformer020_raw_full_v26/transformer_raw_full"
OUTPUT = ROOT / "outputs/submission_metadata/transformer020_raw_full_v26_local_verification.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    checkpoint = SOURCE / "factorized_transformer379_raw_full_e5.pt"
    prediction = SOURCE / "standalone_factorized_transformer379_raw_full_e5.csv"
    result_path = SOURCE / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(prediction)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {"model", "target_scale", "epochs"}
    report = {
        "status": "pending",
        "result_complete": result.get("status") == "complete",
        "visible_gpu_count": result.get("visible_gpu_count"),
        "csv_rows": len(frame),
        "csv_columns": list(frame.columns),
        "duplicate_ids": int(frame.sample_id.duplicated().sum()),
        "finite_predictions": bool(
            np.isfinite(frame.prediction.to_numpy(np.float64)).all()
        ),
        "checkpoint_required_keys_present": required.issubset(state),
        "checkpoint_epochs": int(state["epochs"]),
        "artifact_sha256": {
            checkpoint.name: digest(checkpoint),
            prediction.name: digest(prediction),
            result_path.name: digest(result_path),
        },
    }
    report["passed"] = bool(
        report["result_complete"]
        and report["visible_gpu_count"] == 2
        and report["csv_rows"] == 647_896
        and report["csv_columns"] == ["sample_id", "prediction"]
        and report["duplicate_ids"] == 0
        and report["finite_predictions"]
        and report["checkpoint_required_keys_present"]
        and report["checkpoint_epochs"] == 5
    )
    report["status"] = "passed" if report["passed"] else "failed"
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()


