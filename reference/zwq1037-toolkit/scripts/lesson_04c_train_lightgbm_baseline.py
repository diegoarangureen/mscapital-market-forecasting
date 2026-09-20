import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation


def cosine_similarity_score(y_true, y_pred):
    """Compute the competition cosine similarity with a zero-vector guard."""

    # 转换为 float64 一维数组，降低大向量点积的数值误差。
    # Convert to one-dimensional float64 arrays to reduce large-vector dot-product error.
    true_values = np.asarray(y_true, dtype=np.float64)
    predicted_values = np.asarray(y_pred, dtype=np.float64)

    # 分子是点积，分母是两个向量长度的乘积。
    # The numerator is the dot product; the denominator is the product of vector norms.
    numerator = np.dot(true_values, predicted_values)
    denominator = np.linalg.norm(true_values) * np.linalg.norm(predicted_values)

    # 零向量没有方向；返回0以避免除零产生NaN。
    # A zero vector has no direction; return zero to avoid division by zero.
    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def lightgbm_cosine_metric(y_true, y_pred):
    """Adapt the competition score to LightGBM's custom-evaluation interface."""

    # LightGBM 需要：指标名、指标值、是否越大越好。
    # LightGBM expects the metric name, metric value, and whether higher is better.
    score = cosine_similarity_score(y_true, y_pred)
    return "cosine", score, True


# 找到项目目录和阶段4生成的完整特征、label。
# Locate the project and the full stage-four feature and label files.
project_dir = Path(__file__).resolve().parents[1]
feature_path = project_dir / "data" / "processed" / "train_market_features.feather"
label_path = project_dir / "data" / "raw" / "label.feather"

# 读取逐样本特征并按唯一 sample_id 连接 label。
# Read per-sample features and join labels by unique sample ID.
feature_data = pd.read_feather(feature_path)
label_data = pd.read_feather(label_path)
model_data = feature_data.merge(
    label_data,
    on="sample_id",
    how="inner",
    validate="one_to_one",
)

# 第一列 sample_id 只用于对齐；后面的40列才是模型输入。
# The first sample_id column is only for alignment; the following 40 columns are model inputs.
feature_columns = feature_data.columns[1:].tolist()

# 固定forward validation：月份0～59训练，月份60～70验证。
# Use fixed forward validation: months 0-59 for training and months 60-70 for validation.
train_condition = model_data["month"] <= 59
valid_condition = model_data["month"] >= 60

# 创建独立训练和验证对象；month、sample_id和target都不进入X。
# Create independent train/validation objects; month, sample_id, and target stay outside X.
X_train = model_data.loc[train_condition, feature_columns].copy()
X_valid = model_data.loc[valid_condition, feature_columns].copy()
y_train = model_data.loc[train_condition, "target"].copy()
y_valid = model_data.loc[valid_condition, "target"].copy()

# 保存验证样本身份，训练后与预测一起写入文件。
# Preserve validation identities so predictions can be saved with their sample IDs and months.
validation_output = model_data.loc[
    valid_condition,
    ["sample_id", "month"],
].copy()

# 大表切分完成后释放不再需要的对象，为LightGBM留出内存。
# Release no-longer-needed full tables after splitting to leave memory for LightGBM.
del feature_data
del label_data
del model_data
gc.collect()

print(f"X_train shape = {X_train.shape}")
print(f"X_valid shape = {X_valid.shape}")
print(f"train target shape = {y_train.shape}")
print(f"valid target shape = {y_valid.shape}")
print()

# 运行前预测：请在这里用中文写三句自己的判断。
# Pre-run prediction: write three brief predictions in Chinese here.
# 1.V1能得到正的验证余弦分数
# 2.会在5000轮前early stop
# 3.我觉得价格变化更重要

# TODO 1：使用讲义第2节的参数创建 LGBMRegressor。
# TODO 1: Create LGBMRegressor with the parameters in lesson section 2.
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

# TODO 2：创建 patience 为200轮的early-stopping回调。
# TODO 2: Create an early-stopping callback with a patience of 200 rounds.
stopping_callback = early_stopping(
    stopping_rounds=200,
    first_metric_only=True,
    verbose=True,
)

# 固定日志回调每100轮显示一次forward-valid余弦分数。
# The fixed logging callback displays forward-validation cosine every 100 rounds.
logging_callback = log_evaluation(period=100)

# 记录训练开始时间；perf_counter()适合测量经过的时长。
# Record the start time; perf_counter() is suitable for measuring elapsed duration.
training_start_time = time.perf_counter()

# TODO 3：训练模型，并传入验证集、自定义余弦指标和两个回调。
# TODO 3: Fit the model with the validation set, custom cosine metric, and both callbacks.
model.fit(
    X_train,
    y_train,
    eval_X=X_valid,
    eval_y=y_valid,
    eval_names=["forward_valid"],
    eval_metric=lightgbm_cosine_metric,
    callbacks=[stopping_callback, logging_callback],
)

# 训练结束时间减去开始时间，得到本次经过的秒数。
# Subtract start time from end time to obtain elapsed training seconds.
training_seconds = time.perf_counter() - training_start_time

# TODO 4A：使用 model.best_iteration_ 对 X_valid 调用 predict()。
# TODO 4A: Predict X_valid with model.best_iteration_.
validation_predictions = model.predict(
    X_valid,
    num_iteration=model.best_iteration_,
)

# TODO 4B：调用固定余弦函数，比较 y_valid 与验证预测。
# TODO 4B: Call the fixed cosine function on y_valid and the validation predictions.
validation_cosine = cosine_similarity_score(y_valid, validation_predictions)

print()
print(f"best iteration = {model.best_iteration_}")
print(f"training seconds = {training_seconds:.2f}")
print(f"validation prediction shape = {validation_predictions.shape}")
print(f"forward validation cosine = {validation_cosine:.6f}")

# 创建模型与预测输出目录；不会改动原始数据。
# Create model and prediction directories without modifying raw data.
model_output_dir = project_dir / "outputs" / "models"
prediction_output_dir = project_dir / "outputs" / "predictions"
model_output_dir.mkdir(parents=True, exist_ok=True)
prediction_output_dir.mkdir(parents=True, exist_ok=True)

model_output_path = model_output_dir / "lightgbm_market_v1.txt"
prediction_output_path = (
    prediction_output_dir / "lightgbm_market_v1_valid.feather"
)

# 先让LightGBM生成最佳轮数以内的模型文本；避免其C++写入接口无法处理中文路径。
# First generate best-iteration model text to avoid the C++ writer's Unicode-path limitation.
model_text = model.booster_.model_to_string(
    num_iteration=model.best_iteration_,
)

# 由Python以UTF-8写入模型文本；Python能够正确处理当前中文项目路径。
# Let Python write the UTF-8 model text because it handles the current Unicode project path.
model_output_path.write_text(model_text, encoding="utf-8")

# 将真实target与预测加入验证身份表，供后续误差分析和融合使用。
# Add targets and predictions to validation identities for later error analysis and ensembling.
validation_output["target"] = y_valid.to_numpy()
validation_output["prediction"] = validation_predictions
validation_output.to_feather(prediction_output_path)

print(f"model saved to = {model_output_path}")
print(f"validation predictions saved to = {prediction_output_path}")
