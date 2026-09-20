"""EXP-TABM-001: TabM on the leakage-safe EXP053R feature set.

The script uses months 0-49 / 50-59 to select the epoch count, then retrains
from scratch on months 0-59 and evaluates months 60-70.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tabm import TabM

from build_event_flow_features import TRANSACTION_FEATURE_COLUMNS
from build_market_last15_baseline_features import FEATURE_COLUMNS as LAST15_FEATURE_COLUMNS
from build_transaction_event_gap_features import FEATURE_COLUMNS as GAP_FEATURE_COLUMNS
from exp_tree_017_019_xgboost_target_and_monthly_transforms import assert_prediction_alignment
from exp_tree_024_026_xgboost_notebook_features import ORDER_MULTI_FEATURE_COLUMNS
from exp_tree_027_031_xgboost_capacity_regularization import (
    add_event_features,
    evaluate_and_save,
)
from exp_tree_052_053_xgboost_gpu_last15 import finalize_metadata
from train_full_xgboost_exp051_submission import (
    load_public_table,
    selected_public_features,
)
from train_full_xgboost_submissions import load_features


EXPERIMENT_ID = "EXP-TABM-001"
SEED = 42
K = 16
N_BLOCKS = 2
D_BLOCK = 256
DROPOUT = 0.1
BATCH_SIZE = 2048
EVAL_BATCH_SIZE = 4096
MAX_SCOUT_EPOCHS = 20
PATIENCE = 4
MIN_DELTA = 1.0e-5
LEARNING_RATE = 2.0e-3
WEIGHT_DECAY = 3.0e-4


def seed_everything(seed: int) -> None:
    """固定随机种子，确保实验可以复现。 / Seed all random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算竞赛使用的 cosine similarity。 / Compute cosine similarity."""
    numerator = float(np.dot(y_true.astype(np.float64), y_pred.astype(np.float64)))
    denominator = float(np.linalg.norm(y_true) * np.linalg.norm(y_pred))
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def load_exp053r_data(project_dir: Path) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """重建 EXP053R 的 307 个无泄露特征。 / Rebuild the 307 EXP053R features."""
    processed_dir = project_dir / "data" / "processed"
    public_features, dropped_public_features = selected_public_features(project_dir)

    model_data, base_features = load_features(project_dir, "train")
    model_data = add_event_features(project_dir, model_data)

    gap_data = pd.read_feather(
        processed_dir / "train_transaction_event_gap_features.feather"
    )
    last15_data = pd.read_feather(
        processed_dir / "train_market_last15_baseline_features.feather"
    )
    public_data = load_public_table(project_dir, "train", public_features).rename(
        columns={"target": "public_target"}
    )
    labels = pd.read_feather(
        project_dir / "data" / "raw" / "label.feather",
        columns=["sample_id", "month", "target"],
    )

    for extra_data in (gap_data, public_data, last15_data, labels):
        model_data = model_data.merge(extra_data, on="sample_id", how="inner", validate="one_to_one")

    model_data = model_data.sort_values("sample_id").reset_index(drop=True)
    target_difference = np.abs(
        model_data["public_target"].to_numpy(dtype=np.float64)
        - model_data["target"].to_numpy(dtype=np.float64)
    )
    if float(target_difference.max()) > 1.0e-5:
        raise ValueError("Public feature rows are not aligned with official labels.")
    model_data = model_data.drop(columns=["public_target"])

    model_data["x_rv_15_over_full"] = (
        model_data["m_rv_15"] / (model_data["m_rv"] + 1.0e-8)
    )
    feature_columns = (
        list(base_features)
        + list(TRANSACTION_FEATURE_COLUMNS)
        + list(ORDER_MULTI_FEATURE_COLUMNS)
        + list(GAP_FEATURE_COLUMNS)
        + list(public_features)
        + list(LAST15_FEATURE_COLUMNS)
    )
    if len(feature_columns) != 307 or len(set(feature_columns)) != 307:
        raise ValueError(
            f"Expected 307 unique EXP053R features, got {len(feature_columns)} "
            f"columns and {len(set(feature_columns))} unique columns."
        )
    missing_features = [name for name in feature_columns if name not in model_data.columns]
    if missing_features:
        raise KeyError(f"Missing model features: {missing_features[:10]}")
    return model_data, feature_columns, list(public_features), list(dropped_public_features)


def fit_preprocessor(
    features: np.ndarray, fit_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """只用训练月份拟合中位数和标准化参数。 / Fit preprocessing on train months only."""
    n_features = features.shape[1]
    medians = np.empty(n_features, dtype=np.float32)
    means = np.empty(n_features, dtype=np.float32)
    standard_deviations = np.empty(n_features, dtype=np.float32)
    has_missing = np.zeros(n_features, dtype=bool)
    fit_count = int(len(fit_indices))

    for column_index in range(n_features):
        values = features[fit_indices, column_index]
        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask].astype(np.float64, copy=False)
        has_missing[column_index] = not bool(finite_mask.all())

        if finite_values.size == 0:
            median = 0.0
            mean = 0.0
            standard_deviation = 1.0
        else:
            median = float(np.median(finite_values))
            missing_count = fit_count - int(finite_values.size)
            mean = float((finite_values.sum() + median * missing_count) / fit_count)
            squared_sum = float(np.square(finite_values - mean).sum())
            squared_sum += float(missing_count) * (median - mean) ** 2
            standard_deviation = float(np.sqrt(squared_sum / fit_count))
            if not np.isfinite(standard_deviation) or standard_deviation < 1.0e-6:
                standard_deviation = 1.0

        medians[column_index] = median
        means[column_index] = mean
        standard_deviations[column_index] = standard_deviation

    missing_columns = np.flatnonzero(has_missing).astype(np.int64)
    return medians, means, standard_deviations, missing_columns


class BatchPreprocessor:
    """在 GPU 上逐批填补和标准化，避免复制整张特征表。"""

    def __init__(
        self,
        medians: np.ndarray,
        means: np.ndarray,
        standard_deviations: np.ndarray,
        missing_columns: np.ndarray,
        device: torch.device,
    ) -> None:
        self.medians = torch.as_tensor(medians, device=device)
        self.means = torch.as_tensor(means, device=device)
        self.standard_deviations = torch.as_tensor(standard_deviations, device=device)
        self.missing_columns = torch.as_tensor(missing_columns, device=device)
        self.device = device

    @property
    def output_dimension(self) -> int:
        return int(self.medians.numel() + self.missing_columns.numel())

    def transform(self, raw_batch: np.ndarray) -> torch.Tensor:
        values = torch.as_tensor(raw_batch, dtype=torch.float32, device=self.device)
        missing_mask = ~torch.isfinite(values)
        values = torch.where(missing_mask, self.medians, values)
        values = (values - self.means) / self.standard_deviations
        values = torch.clamp(values, min=-10.0, max=10.0)
        if self.missing_columns.numel() > 0:
            missing_indicators = missing_mask.index_select(1, self.missing_columns).to(values.dtype)
            values = torch.cat([values, missing_indicators], dim=1)
        return values


def make_model(input_dimension: int, device: torch.device) -> TabM:
    model = TabM.make(
        n_num_features=input_dimension,
        cat_cardinalities=None,
        d_out=1,
        k=K,
        n_blocks=N_BLOCKS,
        d_block=D_BLOCK,
        dropout=DROPOUT,
        arch_type="tabm",
    )
    return model.to(device)


@torch.inference_mode()
def predict(
    model: TabM,
    features: np.ndarray,
    indices: np.ndarray,
    preprocessor: BatchPreprocessor,
) -> np.ndarray:
    model.eval()
    output = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), EVAL_BATCH_SIZE):
        end = min(start + EVAL_BATCH_SIZE, len(indices))
        batch_indices = indices[start:end]
        batch = preprocessor.transform(features[batch_indices])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # TabM 返回 k 个并行成员；预测时再平均。 / Average k members only at inference.
            batch_prediction = model(batch).squeeze(-1).mean(dim=1)
        output[start:end] = batch_prediction.float().cpu().numpy()
    return output


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def train_one_epoch(
    model: TabM,
    optimizer: torch.optim.Optimizer,
    features: np.ndarray,
    targets_scaled: np.ndarray,
    train_indices: np.ndarray,
    preprocessor: BatchPreprocessor,
    epoch: int,
) -> float:
    model.train()
    generator = np.random.default_rng(SEED + epoch)
    shuffled_indices = generator.permutation(train_indices)
    total_loss = 0.0
    total_count = 0

    for start in range(0, len(shuffled_indices), BATCH_SIZE):
        batch_indices = shuffled_indices[start : start + BATCH_SIZE]
        batch = preprocessor.transform(features[batch_indices])
        target = torch.as_tensor(
            targets_scaled[batch_indices], dtype=torch.float32, device=preprocessor.device
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            member_predictions = model(batch).squeeze(-1)
            # 每个成员分别学习目标，不能先平均再算损失。 / Train every member separately.
            loss = F.mse_loss(
                member_predictions.float(),
                target[:, None].expand_as(member_predictions),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_count = int(len(batch_indices))
        total_loss += float(loss.detach().cpu()) * batch_count
        total_count += batch_count

    return total_loss / total_count


def scout_epochs(
    features: np.ndarray,
    targets: np.ndarray,
    months: np.ndarray,
    run_dir: Path,
    device: torch.device,
) -> tuple[int, list[dict], float]:
    selection_path = run_dir / "epoch_selection.json"
    log_path = run_dir / "epoch_selection.csv"
    checkpoint_path = run_dir / "scout_checkpoint.pt"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        logs = pd.read_csv(log_path).to_dict(orient="records") if log_path.exists() else []
        print(f"Reusing selected epoch: {selection['best_epoch']}", flush=True)
        return int(selection["best_epoch"]), logs, float(selection["elapsed_seconds"])

    train_indices = np.flatnonzero(months <= 49)
    validation_indices = np.flatnonzero((months >= 50) & (months <= 59))
    medians, means, standard_deviations, missing_columns = fit_preprocessor(
        features, train_indices
    )
    preprocessor = BatchPreprocessor(
        medians, means, standard_deviations, missing_columns, device
    )
    target_mean = float(targets[train_indices].mean(dtype=np.float64))
    target_standard_deviation = float(targets[train_indices].std(dtype=np.float64))
    targets_scaled = ((targets - target_mean) / target_standard_deviation).astype(np.float32)

    seed_everything(SEED)
    model = make_model(preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    start_epoch = 1
    best_score = -np.inf
    best_epoch = 1
    stale_epochs = 0
    logs: list[dict] = []
    elapsed_before_resume = 0.0

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["next_epoch"])
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        logs = list(checkpoint["logs"])
        elapsed_before_resume = float(checkpoint["elapsed_seconds"])
        print(f"Resuming epoch selection at epoch {start_epoch}", flush=True)

    started_at = time.perf_counter()
    for epoch in range(start_epoch, MAX_SCOUT_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            optimizer,
            features,
            targets_scaled,
            train_indices,
            preprocessor,
            epoch,
        )
        validation_prediction = predict(
            model, features, validation_indices, preprocessor
        )
        validation_prediction *= target_standard_deviation
        validation_score = cosine_score(
            targets[validation_indices], validation_prediction
        )
        improved = validation_score > best_score + MIN_DELTA
        if improved:
            best_score = validation_score
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        elapsed_seconds = elapsed_before_resume + time.perf_counter() - started_at
        log_row = {
            "epoch": epoch,
            "train_mse": train_loss,
            "internal_cosine_50_59": validation_score,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "elapsed_seconds": elapsed_seconds,
        }
        logs.append(log_row)
        pd.DataFrame(logs).to_csv(log_path, index=False)
        atomic_torch_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "next_epoch": epoch + 1,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "logs": logs,
                "elapsed_seconds": elapsed_seconds,
            },
            checkpoint_path,
        )
        print(
            f"Scout epoch {epoch:02d}: loss={train_loss:.6f}, "
            f"cosine_50_59={validation_score:.7f}, best={best_score:.7f} "
            f"at {best_epoch}",
            flush=True,
        )
        if stale_epochs >= PATIENCE:
            break

    total_elapsed = elapsed_before_resume + time.perf_counter() - started_at
    selection_path.write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_score": best_score,
                "elapsed_seconds": total_elapsed,
                "train_months": "0-49",
                "validation_months": "50-59",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del model, optimizer, preprocessor
    torch.cuda.empty_cache()
    return best_epoch, logs, total_elapsed


def fit_formal_model(
    features: np.ndarray,
    targets: np.ndarray,
    months: np.ndarray,
    feature_columns: list[str],
    best_epoch: int,
    project_dir: Path,
    run_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    train_indices = np.flatnonzero(months <= 59)
    validation_indices = np.flatnonzero(months >= 60)
    medians, means, standard_deviations, missing_columns = fit_preprocessor(
        features, train_indices
    )
    preprocessor = BatchPreprocessor(
        medians, means, standard_deviations, missing_columns, device
    )
    target_mean = float(targets[train_indices].mean(dtype=np.float64))
    target_standard_deviation = float(targets[train_indices].std(dtype=np.float64))
    targets_scaled = ((targets - target_mean) / target_standard_deviation).astype(np.float32)

    preprocessing_path = run_dir / "preprocessing.npz"
    np.savez_compressed(
        preprocessing_path,
        medians=medians,
        means=means,
        standard_deviations=standard_deviations,
        missing_columns=missing_columns,
        feature_columns=np.asarray(feature_columns),
    )

    checkpoint_path = run_dir / "formal_checkpoint.pt"
    formal_log_path = run_dir / "formal_training.csv"
    seed_everything(SEED)
    model = make_model(preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    start_epoch = 1
    logs: list[dict] = []
    elapsed_before_resume = 0.0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["next_epoch"])
        logs = list(checkpoint["logs"])
        elapsed_before_resume = float(checkpoint["elapsed_seconds"])
        print(f"Resuming formal training at epoch {start_epoch}", flush=True)

    started_at = time.perf_counter()
    for epoch in range(start_epoch, best_epoch + 1):
        train_loss = train_one_epoch(
            model,
            optimizer,
            features,
            targets_scaled,
            train_indices,
            preprocessor,
            epoch,
        )
        elapsed_seconds = elapsed_before_resume + time.perf_counter() - started_at
        logs.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "elapsed_seconds": elapsed_seconds,
            }
        )
        pd.DataFrame(logs).to_csv(formal_log_path, index=False)
        atomic_torch_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "next_epoch": epoch + 1,
                "logs": logs,
                "elapsed_seconds": elapsed_seconds,
            },
            checkpoint_path,
        )
        print(
            f"Formal epoch {epoch:02d}/{best_epoch:02d}: loss={train_loss:.6f}",
            flush=True,
        )

    formal_elapsed = elapsed_before_resume + time.perf_counter() - started_at
    prediction = predict(model, features, validation_indices, preprocessor)
    prediction *= target_standard_deviation

    model_path = project_dir / "outputs" / "models" / f"{EXPERIMENT_ID.lower()}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.cpu()
    torch.save(
        {
            "state_dict": model_cpu.state_dict(),
            "model_parameters": {
                "k": K,
                "n_blocks": N_BLOCKS,
                "d_block": D_BLOCK,
                "dropout": DROPOUT,
                "arch_type": "tabm",
            },
            "input_dimension": preprocessor.output_dimension,
            "feature_columns": feature_columns,
            "target_mean": target_mean,
            "target_standard_deviation": target_standard_deviation,
        },
        model_path,
    )
    details = {
        "model_path": str(model_path.relative_to(project_dir)),
        "preprocessing_path": str(preprocessing_path.relative_to(project_dir)),
        "target_mean": target_mean,
        "target_standard_deviation": target_standard_deviation,
        "missing_indicator_count": int(len(missing_columns)),
        "input_dimension": preprocessor.output_dimension,
        "formal_training_seconds": formal_elapsed,
        "formal_train_rows": int(len(train_indices)),
        "formal_validation_rows": int(len(validation_indices)),
    }
    return prediction, details


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-TABM-001 requires a CUDA GPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("EXP-TABM-001 requires BF16 support on the CUDA GPU.")

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

    model_data, feature_columns, public_features, dropped_public_features = load_exp053r_data(project_dir)
    features = model_data[feature_columns].to_numpy(dtype=np.float32, copy=True)
    targets = model_data["target"].to_numpy(dtype=np.float32, copy=True)
    months = model_data["month"].to_numpy(dtype=np.int16, copy=True)
    sample_ids = model_data["sample_id"].to_numpy(copy=True)
    print(
        f"Loaded rows={len(model_data):,}, features={len(feature_columns)}, "
        f"memory={features.nbytes / 2**30:.2f} GiB",
        flush=True,
    )

    best_epoch, scout_logs, scout_seconds = scout_epochs(
        features, targets, months, run_dir, device
    )
    prediction, training_details = fit_formal_model(
        features,
        targets,
        months,
        feature_columns,
        best_epoch,
        project_dir,
        run_dir,
        device,
    )

    validation_mask = months >= 60
    validation_data = model_data.loc[
        validation_mask, ["sample_id", "month", "target"]
    ].reset_index(drop=True)
    if not np.array_equal(
        validation_data["sample_id"].to_numpy(), sample_ids[validation_mask]
    ):
        raise ValueError("Validation row alignment changed unexpectedly.")

    baseline_predictions = pd.read_feather(
        project_dir / "outputs" / "predictions" / "exp-tree-053r_valid.feather"
    )
    assert_prediction_alignment(baseline_predictions, validation_data)
    metadata = evaluate_and_save(
        project_dir=project_dir,
        experiment_id=EXPERIMENT_ID,
        description=(
            "TabM baseline on the exact 307-feature EXP053R schema; select epoch "
            "on months 50-59, retrain on 0-59, evaluate on 60-70"
        ),
        model=None,
        predictions=prediction,
        validation_rows=validation_data,
        baseline_predictions=baseline_predictions,
        feature_columns=feature_columns,
        parameters={
            "k": K,
            "n_blocks": N_BLOCKS,
            "d_block": D_BLOCK,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        training_seconds=training_details["formal_training_seconds"],
        baseline_id="EXP-TREE-053R",
        save_model=False,
    )
    finalize_metadata(
        metadata=metadata,
        project_dir=project_dir,
        experiment_id=EXPERIMENT_ID,
        baseline_predictions=baseline_predictions,
        public_features=public_features,
        dropped_public_features=dropped_public_features,
    )
    metadata.pop("xgboost_version", None)
    metadata.update(
        {
            "model_family": "TabM",
            "tabm_version": "0.0.3",
            "torch_version": torch.__version__,
            "training_device": "cuda-bfloat16",
            "cuda_device": torch.cuda.get_device_name(0),
            "target_centered": True,
            "target_standardized_for_training": True,
            "epoch_selection_train_months": "0-49",
            "epoch_selection_validation_months": "50-59",
            "selected_epoch": best_epoch,
            "epoch_selection_best_cosine": max(
                float(row["internal_cosine_50_59"]) for row in scout_logs
            ),
            "epoch_selection_seconds": scout_seconds,
            "parameters": {
                "k": K,
                "n_blocks": N_BLOCKS,
                "d_block": D_BLOCK,
                "dropout": DROPOUT,
                "batch_size": BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "optimizer": "AdamW",
                "precision": "bfloat16",
                "gradient_clip_norm": 1.0,
            },
            **training_details,
        }
    )
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
