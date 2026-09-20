"""Use frozen Joint-GRU sequence representations as XGBoost features."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor

import exp_gru_003_strong_joint as gru
from exp_gru_001_sequence import load_raw_statistics
from exp_gru_002_order_control import FullSequenceBatchBuilder
from exp_tabm_011_quantile_preprocessing import evaluate
from exp_tabm_014_quantile_stage_curves import unit
from exp_tabm_016_relative_scale_features import RELATIVE_COLUMNS
from exp_tree_017_019_xgboost_target_and_monthly_transforms import save_model_safely
from exp_tree_027_031_xgboost_capacity_regularization import model_parameters


EXPERIMENT_ID = "EXP-TREE-071-GRU96-EMBEDDING"
SEED = 42
EMBEDDING_BATCH_SIZE = 1024
FOLDS = {
    "train049_valid5059": {
        "train_end_month": 49,
        "valid_start_month": 50,
        "valid_end_month": 59,
        "encoder_run": "EXP-GRU-003-STRONG-JOINT-DEV",
    },
    "train059_valid6070": {
        "train_end_month": 59,
        "valid_start_month": 60,
        "valid_end_month": 70,
        "encoder_run": "EXP-GRU-004-STRONG-JOINT-CONFIRM",
    },
}


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    return float(np.dot(target, prediction) / denominator) if denominator else 0.0


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def add_window_metrics(
    metrics: dict,
    target: np.ndarray,
    prediction: np.ndarray,
    months: np.ndarray,
) -> None:
    if int(months.max()) >= 70:
        mask = (months >= 62) & (months != 66)
        metrics["months_62_70_without_66"] = cosine(target[mask], prediction[mask])


def load_encoder(
    checkpoint_path: Path,
    input_size: int,
    device: torch.device,
) -> gru.JointRegressor:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state"]
    static_input_size = int(state["static_adapter.0.weight"].shape[1])
    model = gru.JointRegressor(input_size, static_input_size, use_sequence=True).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.inference_mode()
def export_embeddings(
    *,
    cache: np.memmap,
    cache_dir: Path,
    encoder_run_dir: Path,
    checkpoint_path: Path,
    end_index: int,
    output_path: Path,
    device: torch.device,
) -> dict:
    manifest_path = output_path.with_suffix(".json")
    if output_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["shape"] == [end_index, gru.HIDDEN_SIZE]
            and manifest["checkpoint_sha256"] == sha256(checkpoint_path)
        ):
            print(f"reusing embeddings {output_path}", flush=True)
            return manifest

    fold_stats = np.load(encoder_run_dir / "fold_standardization.npz")
    raw_means, raw_scales = load_raw_statistics(cache_dir)
    builder = FullSequenceBatchBuilder(
        cache,
        raw_means,
        raw_scales,
        fold_stats["means"],
        fold_stats["scales"],
        device,
    )
    input_size = len(gru.RAW_CHANNEL_NAMES) + len(gru.DERIVED_CHANNEL_NAMES)
    model = load_encoder(checkpoint_path, input_size, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(end_index, gru.HIDDEN_SIZE)
    )
    started = time.perf_counter()
    for start in range(0, end_index, EMBEDDING_BATCH_SIZE):
        end = min(start + EMBEDDING_BATCH_SIZE, end_index)
        sequence = builder.make(start, end)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            representation = model.sequence_adapter(model.encoder(sequence))
        values = representation.float().cpu().numpy()
        if not np.isfinite(values).all():
            raise AssertionError("Non-finite GRU embedding.")
        output[start:end] = values
        if (start // EMBEDDING_BATCH_SIZE + 1) % 200 == 0:
            output.flush()
            print(f"embedding rows {end}/{end_index}", flush=True)
    output.flush()
    seconds = time.perf_counter() - started
    manifest = {
        "shape": [end_index, gru.HIDDEN_SIZE],
        "dtype": "float32",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "encoder_source": str(encoder_run_dir),
        "seconds": seconds,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del model, output
    torch.cuda.empty_cache()
    gc.collect()
    return manifest


def load_fold_predictions(
    project_dir: Path,
    fold_name: str,
    validation_ids: np.ndarray,
    encoder_run: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tabm_path = (
        project_dir / "data" / "interim" / "tree_experiments"
        / "EXP-TABM-016-RELATIVE-SCALE-ADD12" / fold_name
        / "validation_predictions.feather"
    )
    tabm_frame = pd.read_feather(tabm_path)
    tree_path = (
        project_dir / "data" / "interim" / "tree_experiments"
        / "EXP-TREE-069-RELATIVE319" / fold_name / "validation_predictions.feather"
    )
    tree_frame = pd.read_feather(tree_path)
    joint_path = (
        project_dir / "data" / "interim" / "sequence_experiments"
        / encoder_run / "joint_gru319" / "validation_predictions_epoch06.feather"
    )
    joint_frame = pd.read_feather(joint_path)
    for name, frame in (("tabm", tabm_frame), ("tree", tree_frame), ("joint", joint_frame)):
        if not np.array_equal(frame["sample_id"].to_numpy(), validation_ids):
            raise AssertionError(f"{name} prediction IDs differ for {fold_name}.")
    return (
        tabm_frame["tabm_trim1"].to_numpy(dtype=np.float64),
        tabm_frame["b001_trim1"].to_numpy(dtype=np.float64),
        tree_frame["xgb_relative319"].to_numpy(dtype=np.float64),
        joint_frame["prediction"].to_numpy(dtype=np.float64),
    )


def run_fold(
    *,
    project_dir: Path,
    run_dir: Path,
    fold_name: str,
    config: dict,
    cache: np.memmap,
    cache_dir: Path,
    labels: pd.DataFrame,
    features: np.ndarray,
    feature_columns: list[str],
    device: torch.device,
) -> dict:
    fold_dir = run_dir / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    result_path = fold_dir / "result.json"
    prediction_path = fold_dir / "validation_predictions.feather"
    if result_path.exists() and prediction_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    months = labels["month"].to_numpy()
    train_end = int(np.searchsorted(months, config["train_end_month"] + 1, side="left"))
    valid_start = int(np.searchsorted(months, config["valid_start_month"], side="left"))
    valid_end = int(np.searchsorted(months, config["valid_end_month"] + 1, side="left"))
    if train_end != valid_start:
        raise AssertionError("Expected contiguous forward split.")

    encoder_run_dir = (
        project_dir / "data" / "interim" / "sequence_experiments"
        / config["encoder_run"]
    )
    encoder_result = json.loads(
        (encoder_run_dir / "result.json").read_text(encoding="utf-8")
    )
    if encoder_result["train_months"] != f"0-{config['train_end_month']}":
        raise AssertionError("Encoder training boundary differs from the outer fold.")
    checkpoint_path = encoder_run_dir / "joint_gru319" / "model_epoch06.pt"
    embedding_path = fold_dir / "gru96_embeddings.npy"
    embedding_manifest = export_embeddings(
        cache=cache,
        cache_dir=cache_dir,
        encoder_run_dir=encoder_run_dir,
        checkpoint_path=checkpoint_path,
        end_index=valid_end,
        output_path=embedding_path,
        device=device,
    )
    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.shape != (valid_end, gru.HIDDEN_SIZE):
        raise AssertionError("Embedding shape differs.")

    train_static = features[:train_end]
    valid_static = features[valid_start:valid_end]
    train_embedding = np.asarray(embeddings[:train_end], dtype=np.float32)
    valid_embedding = np.asarray(embeddings[valid_start:valid_end], dtype=np.float32)
    train_matrix = np.concatenate([train_static, train_embedding], axis=1)
    valid_matrix = np.concatenate([valid_static, valid_embedding], axis=1)
    if train_matrix.shape[1] != 319 + gru.HIDDEN_SIZE:
        raise AssertionError("Expected 415 enhanced tree features.")
    target = labels["target"].to_numpy(dtype=np.float32)
    centered_target = target[:train_end].astype(np.float64)
    centered_target -= centered_target.mean()

    parameters = model_parameters(
        n_estimators=800,
        colsample_bytree=0.8,
        device="cuda",
        n_jobs=2,
        random_state=SEED,
    )
    model = XGBRegressor(**parameters)
    started = time.perf_counter()
    model.fit(train_matrix, centered_target)
    training_seconds = time.perf_counter() - started
    embedding_prediction = np.asarray(model.predict(valid_matrix), dtype=np.float64)
    if not np.isfinite(embedding_prediction).all():
        raise AssertionError("Non-finite enhanced tree prediction.")
    save_model_safely(model, fold_dir / "model.json")

    validation_rows = labels.iloc[valid_start:valid_end]
    validation_ids = validation_rows["sample_id"].to_numpy()
    validation_target = validation_rows["target"].to_numpy(dtype=np.float64)
    validation_months = validation_rows["month"].to_numpy()
    tabm_trim1, current_b001, relative319_tree, joint_prediction = load_fold_predictions(
        project_dir, fold_name, validation_ids, config["encoder_run"]
    )
    predictions = {
        "current_b001_trim1": current_b001,
        "relative319_tree": relative319_tree,
        "gru_embedding_tree": embedding_prediction,
        "relative319_tabm75_embedding_tree25": (
            0.75 * unit(tabm_trim1) + 0.25 * unit(embedding_prediction)
        ),
        "current_b001_90_joint10": (
            0.90 * unit(current_b001) + 0.10 * unit(joint_prediction)
        ),
    }
    predictions["embedding_tree_blend90_joint10"] = (
        0.90 * unit(predictions["relative319_tabm75_embedding_tree25"])
        + 0.10 * unit(joint_prediction)
    )
    metrics = {}
    for name, prediction in predictions.items():
        values = evaluate(
            validation_target,
            prediction,
            validation_months,
            config["valid_start_month"],
            config["valid_end_month"],
        )
        add_window_metrics(values, validation_target, prediction, validation_months)
        metrics[name] = values

    baseline_metrics = metrics["current_b001_trim1"]
    tree_baseline_metrics = metrics["relative319_tree"]
    for name in (
        "gru_embedding_tree",
        "relative319_tabm75_embedding_tree25",
        "current_b001_90_joint10",
        "embedding_tree_blend90_joint10",
    ):
        reference = (
            tree_baseline_metrics if name == "gru_embedding_tree" else baseline_metrics
        )
        for key in (
            "overall",
            "without_month_66",
            "months_62_70_without_66",
            "months_67_70",
            "monthly_std",
            "monthly_worst",
            "monthly_q25",
        ):
            if key in metrics[name] and key in reference:
                metrics[name][key + "_delta"] = metrics[name][key] - reference[key]

    output = validation_rows[["sample_id", "month", "target"]].copy()
    for name, prediction in predictions.items():
        output[name] = prediction
    output.to_feather(prediction_path)
    summary = []
    for name, values in metrics.items():
        summary.append({
            "candidate": name,
            **{key: value for key, value in values.items() if key not in {
                "monthly", "first_half_months", "second_half_months"
            }},
        })
    pd.DataFrame(summary).to_csv(fold_dir / "summary.csv", index=False)
    pd.DataFrame({
        "month": sorted(np.unique(validation_months)),
        **{
            name: [row["cosine"] for row in values["monthly"]]
            for name, values in metrics.items()
        },
    }).to_csv(fold_dir / "monthly_cosine.csv", index=False)

    result = {
        "fold": fold_name,
        "train_months": f"0-{config['train_end_month']}",
        "validation_months": f"{config['valid_start_month']}-{config['valid_end_month']}",
        "train_rows": train_end,
        "validation_rows": valid_end - valid_start,
        "base_feature_count": len(feature_columns),
        "embedding_feature_count": gru.HIDDEN_SIZE,
        "enhanced_feature_count": train_matrix.shape[1],
        "encoder_did_not_see_validation_labels": True,
        "tree_training_embeddings_are_in_sample_supervised_representations": True,
        "embedding_manifest": embedding_manifest,
        "parameters": parameters,
        "training_seconds": training_seconds,
        "metrics": metrics,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    del model, train_matrix, valid_matrix, train_embedding, valid_embedding, embeddings
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")
    device = torch.device("cuda")
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = (
        project_dir / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = project_dir / "data" / "processed" / "train_sequence_cache_v1"
    cache = np.load(cache_dir / "sequences.npy", mmap_mode="r")
    labels = pd.read_feather(
        project_dir / "data" / "raw" / "label.feather",
        columns=["sample_id", "month", "target"],
    ).sort_values("sample_id").reset_index(drop=True)
    if cache.shape != (len(labels), 14, 200):
        raise AssertionError("Sequence cache and labels differ.")
    features, feature_columns = gru.load_relative319(project_dir, labels)
    if len(feature_columns) != 319 or feature_columns[-len(RELATIVE_COLUMNS):] != RELATIVE_COLUMNS:
        raise AssertionError("Relative319 schema differs.")

    results = {}
    for fold_name, config in FOLDS.items():
        results[fold_name] = run_fold(
            project_dir=project_dir,
            run_dir=run_dir,
            fold_name=fold_name,
            config=config,
            cache=cache,
            cache_dir=cache_dir,
            labels=labels,
            features=features,
            feature_columns=feature_columns,
            device=device,
        )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "seed": SEED,
        "purpose": "test frozen Joint-GRU representations as incremental XGBoost features",
        "folds": results,
    }
    (run_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

