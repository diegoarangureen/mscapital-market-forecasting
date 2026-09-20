"""Compare validation predictions under monthly and pooled XS40 inference groups."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "data" / "interim" / "kaggle_kernels" / "relative319_xs_tabm_notebook"))
import run_extracted as recipe
import exp_gru_003_strong_joint as gru


def cosine(target, prediction):
    a = np.asarray(target, dtype=np.float64)
    b = np.asarray(prediction, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    torch.set_num_threads(2)
    labels = pd.read_feather(PROJECT_DIR / "data" / "raw" / "label.feather", columns=["sample_id", "month", "target"]).sort_values("sample_id").reset_index(drop=True)
    base, columns = gru.load_relative319(PROJECT_DIR, labels)
    months = labels["month"].to_numpy()
    selected = (months >= 62) & (months <= 70)
    score_mask = months[selected] != 66
    valid_base = np.asarray(base[selected], dtype=np.float32)
    valid_months = months[selected]
    monthly_xs, names = recipe.add_relative_features(valid_base, valid_months, columns)
    pooled_xs, pooled_names = recipe.add_relative_features(valid_base, np.zeros(len(valid_base), dtype=np.int16), columns)
    equal_groups = (np.arange(len(valid_base), dtype=np.int64) * len(np.unique(valid_months)) // len(valid_base)).astype(np.int16)
    equal_xs, equal_names = recipe.add_relative_features(valid_base, equal_groups, columns)
    assert names == pooled_names
    run_dir = PROJECT_DIR / "data" / "interim" / "submissions" / "tabm_relative319_xs40_seed42_holdout059"
    saved = np.load(run_dir / "quantile_preprocessing.npz")
    device = torch.device("cuda")
    preprocessor = recipe.QuantilePreprocessor(saved["knots"], saved["medians"], saved["missing_columns"], device)
    model = recipe.make_model(preprocessor.output_dimension, device)
    model.load_state_dict(torch.load(run_dir / "model_holdout059.pt", map_location=device, weights_only=True))
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    indices = np.arange(len(valid_base))
    target = labels["target"].to_numpy(dtype=np.float32)
    target_std = float(target[months <= 59].std(dtype=np.float64))
    predictions = {}
    for name, xs in (("monthly", monthly_xs), ("pooled", pooled_xs), ("equal_blocks", equal_xs)):
        features = np.concatenate([valid_base, xs], axis=1)
        prediction = recipe.predict(model, features, indices, preprocessor, amp_dtype).astype(np.float64) * target_std
        predictions[name] = prediction
    valid_target = target[selected][score_mask]
    report = {
        "experiment": "XS40 monthly versus pooled validation inference",
        "training_months": "0-59",
        "validation_months": "62-70 excluding 66 for scoring",
        "monthly_cosine": cosine(valid_target, predictions["monthly"][score_mask]),
        "pooled_cosine": cosine(valid_target, predictions["pooled"][score_mask]),
        "equal_blocks_cosine": cosine(valid_target, predictions["equal_blocks"][score_mask]),
        "prediction_correlation": float(np.corrcoef(predictions["monthly"][score_mask], predictions["pooled"][score_mask])[0, 1]),
        "feature_count": int(valid_base.shape[1] + monthly_xs.shape[1]),
    }
    report["pooled_minus_monthly"] = report["pooled_cosine"] - report["monthly_cosine"]
    report["equal_blocks_minus_monthly"] = report["equal_blocks_cosine"] - report["monthly_cosine"]
    out = PROJECT_DIR / "data" / "interim" / "tree_experiments" / "XS40-GROUPING-MISMATCH"
    out.mkdir(parents=True, exist_ok=True)
    (out / "score_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

