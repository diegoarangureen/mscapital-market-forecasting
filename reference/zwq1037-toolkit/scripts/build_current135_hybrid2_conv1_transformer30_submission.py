from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = PROJECT_DIR / "outputs" / "submissions"
METADATA_DIR = PROJECT_DIR / "outputs" / "submission_metadata"
RUN_NAME = "current135_hybrid2_conv1_transformer30"

def unit(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    return (values - values.mean()) / values.std(ddof=0)

sources = {
    "current135": "gru_embedding_seed_ensemble_fulltrain.csv",
    "hybrid42": "joint_transformer_hybrid_epoch5_fulltrain.csv",
    "hybrid137": "joint_transformer_hybrid_seed137_epoch4_fulltrain.csv",
    "conv42": "joint_transformer_conv_hybrid_epoch5_fulltrain.csv",
}
frames = {name: pd.read_csv(SUBMISSION_DIR / filename) for name, filename in sources.items()}
sample_ids = frames["current135"]["sample_id"].to_numpy()
for name, frame in frames.items():
    if list(frame.columns) != ["sample_id", "prediction"]:
        raise AssertionError(f"Unexpected columns for {name}: {list(frame.columns)}")
    if not np.array_equal(frame["sample_id"].to_numpy(), sample_ids):
        raise AssertionError(f"sample_id order differs for {name}")
    if not np.isfinite(frame["prediction"].to_numpy()).all():
        raise AssertionError(f"Non-finite prediction in {name}")
transformer = (
    0.335 * unit(frames["hybrid42"]["prediction"].to_numpy())
    + 0.335 * unit(frames["hybrid137"]["prediction"].to_numpy())
    + 0.330 * unit(frames["conv42"]["prediction"].to_numpy())
)
prediction = 0.70 * unit(frames["current135"]["prediction"].to_numpy()) + 0.30 * unit(transformer)
submission = pd.DataFrame({"sample_id": sample_ids, "prediction": prediction})
output_path = SUBMISSION_DIR / f"{RUN_NAME}.csv"
submission.to_csv(output_path, index=False)
metadata = {
    "run_name": RUN_NAME,
    "sources": sources,
    "weights": {
        "current135": 0.70,
        "transformer_total": 0.30,
        "within_transformer": {"hybrid42": 0.335, "hybrid137": 0.335, "conv42": 0.330},
    },
    "local_scores": {"50_59": 0.156322, "60_70": 0.179385},
    "local_delta_vs_current135": {"50_59": 0.000882, "60_70": 0.002052},
    "submission_status": "prepared_not_uploaded",
    "rows": len(submission),
    "prediction_mean": float(prediction.mean()),
    "prediction_std": float(prediction.std(ddof=0)),
    "prediction_min": float(prediction.min()),
    "prediction_max": float(prediction.max()),
    "output_path": str(output_path),
}
METADATA_DIR.mkdir(parents=True, exist_ok=True)
(METADATA_DIR / f"{RUN_NAME}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(metadata, ensure_ascii=False, indent=2))
