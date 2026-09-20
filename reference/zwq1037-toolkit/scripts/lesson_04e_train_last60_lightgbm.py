import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation


def cosine_similarity_score(y_true, y_pred):
    """Compute cosine similarity with a zero-vector guard."""

    true_values = np.asarray(y_true, dtype=np.float64)
    predicted_values = np.asarray(y_pred, dtype=np.float64)
    numerator = np.dot(true_values, predicted_values)
    denominator = np.linalg.norm(true_values) * np.linalg.norm(predicted_values)

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def lightgbm_cosine_metric(y_true, y_pred):
    """Adapt cosine similarity to LightGBM's evaluation interface."""

    score = cosine_similarity_score(y_true, y_pred)
    return "cosine", score, True


# 读取固定V1特征、新增窗口特征和label。
# Read the fixed V1 features, new window features, and labels.
project_dir = Path(__file__).resolve().parents[1]
v1_path = project_dir / "data" / "processed" / "train_market_features.feather"
last60_path = (
    project_dir
    / "data"
    / "processed"
    / "train_market_last60_features_complete.feather"
)
label_path = project_dir / "data" / "raw" / "label.feather"

v1_data = pd.read_feather(v1_path)
last60_data = pd.read_feather(last60_path)
label_data = pd.read_feather(label_path)

# V1是完整样本主表；左连接保留最后60秒没有market记录的样本。
# V1 is the complete sample master; the left join retains samples with no market rows in the final 60 seconds.
model_data = (
    v1_data.merge(
        last60_data,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    .merge(
        label_data,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
)

# 没有窗口记录时行数为0，其他窗口统计保持NaN并交给LightGBM处理。
# When a window has no rows, its count is zero; other window statistics remain NaN for LightGBM.
model_data["last60_row_count"] = model_data["last60_row_count"].fillna(0)

assert len(model_data) == len(v1_data)

v1_feature_columns = v1_data.columns[1:].tolist()
last60_feature_columns = last60_data.columns[1:].tolist()
feature_columns = v1_feature_columns + last60_feature_columns

# 使用与EXP-001完全相同的forward validation。
# Use exactly the same forward validation as EXP-001.
train_condition = model_data["month"] <= 59
valid_condition = model_data["month"] >= 60

X_train = model_data.loc[train_condition, feature_columns].copy()
X_valid = model_data.loc[valid_condition, feature_columns].copy()
y_train = model_data.loc[train_condition, "target"].copy()
y_valid = model_data.loc[valid_condition, "target"].copy()

validation_output = model_data.loc[
    valid_condition,
    ["sample_id", "month"],
].copy()

del v1_data
del last60_data
del label_data
del model_data
gc.collect()

print(f"V1 feature count = {len(v1_feature_columns)}")
print(f"last60 feature count = {len(last60_feature_columns)}")
print(f"total feature count = {len(feature_columns)}")
print(f"X_train shape = {X_train.shape}")
print(f"X_valid shape = {X_valid.shape}")

# 运行前判断：你认为分数会提高、近似不变还是下降？为什么？
# Pre-run judgment: Will the score improve, remain similar, or decrease? Why?
# 你的判断：我认为会上升

# 参数、早停和验证指标与EXP-001保持一致，只改变输入特征。
# Parameters, early stopping, and validation metric match EXP-001; only features change.
model = LGBMRegressor(
    objective="regression",
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=100,
    reg_lambda=1.0,
    metric="None",
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

stopping_callback = early_stopping(
    stopping_rounds=200,
    first_metric_only=True,
    verbose=True,
)
logging_callback = log_evaluation(period=100)

training_start_time = time.perf_counter()

model.fit(
    X_train,
    y_train,
    eval_X=X_valid,
    eval_y=y_valid,
    eval_names=["forward_valid"],
    eval_metric=lightgbm_cosine_metric,
    callbacks=[stopping_callback, logging_callback],
)

training_seconds = time.perf_counter() - training_start_time
validation_predictions = model.predict(
    X_valid,
    num_iteration=model.best_iteration_,
)
validation_cosine = cosine_similarity_score(y_valid, validation_predictions)

baseline_cosine = 0.07782086012742553
score_change = validation_cosine - baseline_cosine

print()
print(f"best iteration = {model.best_iteration_}")
print(f"training seconds = {training_seconds:.2f}")
print(f"last60 validation cosine = {validation_cosine:.6f}")
print(f"EXP-001 baseline cosine = {baseline_cosine:.6f}")
print(f"score change = {score_change:+.6f}")

# 用Python写模型文本，避免LightGBM的C++文件接口处理中文路径失败。
# Let Python write model text to avoid Unicode-path problems in LightGBM's C++ file interface.
model_output_dir = project_dir / "outputs" / "models"
prediction_output_dir = project_dir / "outputs" / "predictions"
model_output_dir.mkdir(parents=True, exist_ok=True)
prediction_output_dir.mkdir(parents=True, exist_ok=True)

model_output_path = model_output_dir / "lightgbm_market_v1_last60.txt"
prediction_output_path = (
    prediction_output_dir / "lightgbm_market_v1_last60_valid.feather"
)

model_text = model.booster_.model_to_string(
    num_iteration=model.best_iteration_,
)
model_output_path.write_text(model_text, encoding="utf-8")

validation_output["target"] = y_valid.to_numpy()
validation_output["prediction"] = validation_predictions
validation_output.to_feather(prediction_output_path)

print(f"model saved to = {model_output_path}")
print(f"validation predictions saved to = {prediction_output_path}")

# 运行后回答：结果更支持A还是B？它是否足以证明60秒是最佳窗口？
# Post-run answer: Does the result support A or B, and does it prove 60 seconds is optimal?
# 你的回答：forward_valid's cosine: 0.104092，说明有一定提升，不过60s不一定是最佳窗口
