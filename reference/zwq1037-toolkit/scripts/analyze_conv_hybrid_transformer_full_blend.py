from pathlib import Path

import numpy as np
import pandas as pd

from exp_tabm_011_quantile_preprocessing import evaluate
from exp_tabm_014_quantile_stage_curves import unit

PROJECT_DIR = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_DIR / "data" / "interim"


def load_window(fold_name: str, start_month: int, end_month: int):
    tree_dir = INTERIM_DIR / "tree_experiments"
    sequence_dir = INTERIM_DIR / "sequence_experiments"
    tabm = pd.read_feather(tree_dir / "EXP-TABM-016-RELATIVE-SCALE-ADD12" / fold_name / "validation_predictions.feather")
    tree42 = pd.read_feather(tree_dir / "EXP-TREE-071-GRU96-EMBEDDING" / fold_name / "validation_predictions.feather")
    tree137 = pd.read_feather(tree_dir / "EXP-TREE-072-GRU96-EMBEDDING-SEED137" / fold_name / "validation_predictions.feather")
    if start_month == 50:
        joint42 = pd.read_feather(sequence_dir / "EXP-GRU-003-STRONG-JOINT-DEV" / "joint_gru319" / "validation_predictions_epoch06.feather")
        joint137 = pd.read_feather(sequence_dir / "EXP-GRU-005-JOINT-SEED137" / fold_name / "joint_gru319" / "validation_predictions_epoch06.feather")
        hybrid42 = pd.read_feather(sequence_dir / "EXP-TRANSFORMER-006-HYBRID-LOSS-DEV" / "validation_predictions_epoch05.feather")
        hybrid137 = pd.read_feather(sequence_dir / "EXP-TRANSFORMER-008-HYBRID-LOSS-SEED137-DEV" / "validation_predictions_epoch04.feather")
        conv42 = pd.read_feather(sequence_dir / "EXP-TRANSFORMER-010-CONV-HYBRID-DEV" / "validation_predictions_epoch05.feather")
    else:
        joint42 = pd.read_feather(sequence_dir / "EXP-GRU-004-STRONG-JOINT-CONFIRM" / "joint_gru319" / "validation_predictions_epoch06.feather")
        joint137 = pd.read_feather(sequence_dir / "EXP-GRU-005-JOINT-SEED137" / fold_name / "joint_gru319" / "validation_predictions_epoch06.feather")
        hybrid42 = pd.read_feather(sequence_dir / "EXP-TRANSFORMER-007-HYBRID-LOSS-CONFIRM" / "validation_predictions_epoch05.feather")
        hybrid137 = pd.read_feather(sequence_dir / "EXP-TRANSFORMER-009-HYBRID-LOSS-SEED137-CONFIRM" / "validation_predictions_epoch04.feather")
        conv42 = pd.read_feather(sequence_dir / "EXP-TRANSFORMER-012-CONV-HYBRID-CONFIRM" / "validation_predictions_epoch05.feather")
    sample_ids = tabm["sample_id"].to_numpy()
    for frame in (tree42, tree137, joint42, joint137, hybrid42, hybrid137, conv42):
        assert np.array_equal(frame["sample_id"].to_numpy(), sample_ids)
    embedding = 0.5 * unit(tree42["gru_embedding_tree"].to_numpy(float)) + 0.5 * unit(tree137["gru_embedding_tree"].to_numpy(float))
    joint = 0.5 * unit(joint42["prediction"].to_numpy(float)) + 0.5 * unit(joint137["prediction"].to_numpy(float))
    tree = 0.75 * unit(tabm["tabm_trim1"].to_numpy(float)) + 0.20 * unit(tabm["xgboost"].to_numpy(float)) + 0.05 * unit(embedding)
    current = 0.90 * unit(tree) + 0.10 * unit(joint)
    predictions = {
        "hybrid42": unit(hybrid42["prediction"].to_numpy(float)),
        "hybrid137": unit(hybrid137["prediction"].to_numpy(float)),
        "conv42": unit(conv42["prediction"].to_numpy(float)),
    }
    target = tabm["target"].to_numpy(float)
    months = tabm["month"].to_numpy()
    baseline_score = evaluate(target, current, months, start_month, end_month)["overall"]
    return current, predictions, target, months, baseline_score


windows = {
    "50-59": load_window("train049_valid5059", 50, 59),
    "60-70": load_window("train059_valid6070", 60, 70),
}
rows = []
for conv_share in (0.0, 0.20, 0.33, 0.50, 0.67, 0.80, 1.0):
    ordinary_share = (1.0 - conv_share) / 2.0
    for append_weight in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        result = {"conv_share": conv_share, "append_weight": append_weight}
        for window_name, (current, predictions, target, months, baseline_score) in windows.items():
            transformer = ordinary_share * predictions["hybrid42"] + ordinary_share * predictions["hybrid137"] + conv_share * predictions["conv42"]
            combined = (1.0 - append_weight) * unit(current) + append_weight * unit(transformer)
            start_month, end_month = (50, 59) if window_name == "50-59" else (60, 70)
            score = evaluate(target, combined, months, start_month, end_month)["overall"]
            result[f"score_{window_name}"] = score
            result[f"delta_{window_name}"] = score - baseline_score
        result["min_delta"] = min(result["delta_50-59"], result["delta_60-70"])
        result["mean_delta"] = 0.5 * (result["delta_50-59"] + result["delta_60-70"])
        rows.append(result)
output = pd.DataFrame(rows).sort_values(["min_delta", "mean_delta"], ascending=False)
print(output.head(15).to_string(index=False))
output.to_csv(PROJECT_DIR / "outputs" / "conv_hybrid_weight_search.csv", index=False)

