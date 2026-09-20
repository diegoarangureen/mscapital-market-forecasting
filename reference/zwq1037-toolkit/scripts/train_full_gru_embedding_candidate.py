"""Train the promoted two-seed Joint-GRU candidate and prepare a submission."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import torch
from torch import nn
from xgboost import XGBRegressor

import exp_gru_003_strong_joint as gru
import train_full_quantile_tabm_b001_submission as full_b001
from exp_gru_001_sequence import DERIVED_CHANNEL_NAMES, RAW_CHANNEL_NAMES, load_raw_statistics
from exp_gru_002_order_control import FullSequenceBatchBuilder, fit_fold_standardization
from exp_tabm_011_quantile_preprocessing import QuantileBatchPreprocessor, fit_quantile_knots
from exp_tabm_014_quantile_stage_curves import unit
from exp_tabm_016_relative_scale_features import RELATIVE_COLUMNS
from exp_tree_017_019_xgboost_target_and_monthly_transforms import save_model_safely
from exp_tree_027_031_xgboost_capacity_regularization import model_parameters
from train_full_quantile_tabm_relative319_submission import add_split_relative_features


RUN_NAME = "gru_embedding_seed_ensemble_fulltrain"
SEEDS = (42, 137)
EPOCHS = 6
BATCH_SIZE = 512
PREDICT_BATCH_SIZE = 1024


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def fit_preprocessor(
    features: np.ndarray,
    feature_columns: list[str],
    seed: int,
    output_path: Path,
    device: torch.device,
) -> QuantileBatchPreprocessor:
    if output_path.exists():
        saved = np.load(output_path)
        if saved["feature_columns"].astype(str).tolist() != feature_columns:
            raise AssertionError("Saved quantile feature schema differs.")
        knots = saved["knots"]
        medians = saved["medians"]
        missing_columns = saved["missing_columns"]
    else:
        indices = np.arange(len(features), dtype=np.int64)
        knots, medians, missing_columns = fit_quantile_knots(features, indices, seed)
        np.savez_compressed(
            output_path,
            knots=knots,
            medians=medians,
            missing_columns=missing_columns,
            feature_columns=np.asarray(feature_columns),
        )
    return QuantileBatchPreprocessor(knots, medians, missing_columns, device)


def train_joint_model(
    *,
    seed: int,
    model: gru.JointRegressor,
    builder: FullSequenceBatchBuilder,
    features: np.ndarray,
    preprocessor: QuantileBatchPreprocessor,
    target_scaled: np.ndarray,
    seed_dir: Path,
    device: torch.device,
) -> list[dict]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=gru.MAX_LEARNING_RATE, weight_decay=gru.WEIGHT_DECAY
    )
    checkpoint_path = seed_dir / "training_checkpoint.pt"
    training_path = seed_dir / "training_log.csv"
    logs: list[dict] = []
    start_epoch = 1
    elapsed_before = 0.0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        logs = list(checkpoint["logs"])
        start_epoch = int(checkpoint["next_epoch"])
        elapsed_before = float(checkpoint["elapsed_seconds"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        print(f"resuming seed={seed} at epoch {start_epoch}", flush=True)

    started = time.perf_counter()
    for epoch in range(start_epoch, EPOCHS + 1):
        rate = gru.learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = rate
        model.train()
        train_blocks = gru.blocks(0, len(features), BATCH_SIZE)
        np.random.default_rng(seed + epoch).shuffle(train_blocks)
        loss_sum = 0.0
        row_count = 0
        for start, end in train_blocks:
            sequence = builder.make(start, end)
            static = preprocessor.transform(features[start:end])
            target = torch.as_tensor(target_scaled[start:end], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(sequence, static)
                loss = nn.functional.mse_loss(prediction.float(), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = end - start
            loss_sum += float(loss.detach().cpu()) * count
            row_count += count
        elapsed = elapsed_before + time.perf_counter() - started
        log = {
            "epoch": epoch,
            "learning_rate": rate,
            "train_mse": loss_sum / row_count,
            "elapsed_seconds": elapsed,
        }
        logs.append(log)
        pd.DataFrame(logs).to_csv(training_path, index=False)
        atomic_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "next_epoch": epoch + 1,
                "logs": logs,
                "elapsed_seconds": elapsed,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
            },
            checkpoint_path,
        )
        print(
            f"joint seed={seed} epoch={epoch}/{EPOCHS} "
            f"lr={rate:.7f} loss={log['train_mse']:.7f}",
            flush=True,
        )
    return logs


@torch.inference_mode()
def export_split(
    *,
    model: gru.JointRegressor,
    builder: FullSequenceBatchBuilder,
    features: np.ndarray,
    preprocessor: QuantileBatchPreprocessor,
    target_scale: float,
    ids: np.ndarray,
    embedding_path: Path,
    prediction_path: Path,
) -> None:
    if embedding_path.exists() and prediction_path.exists():
        embeddings = np.load(embedding_path, mmap_mode="r")
        predictions = pd.read_feather(prediction_path, columns=["sample_id"])
        if embeddings.shape == (len(features), gru.HIDDEN_SIZE) and np.array_equal(
            predictions["sample_id"].to_numpy(), ids
        ):
            print(f"reusing {embedding_path.name}", flush=True)
            return
        raise AssertionError("Saved full-data export shape or IDs differ.")

    output = np.lib.format.open_memmap(
        embedding_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(features), gru.HIDDEN_SIZE),
    )
    prediction = np.empty(len(features), dtype=np.float32)
    model.eval()
    for block_index, (start, end) in enumerate(
        gru.blocks(0, len(features), PREDICT_BATCH_SIZE), start=1
    ):
        sequence = builder.make(start, end)
        static = preprocessor.transform(features[start:end])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sequence_representation = model.sequence_adapter(model.encoder(sequence))
            static_representation = model.static_adapter(static)
            standardized_prediction = model.head(
                torch.cat([static_representation, sequence_representation], dim=1)
            ).squeeze(1)
        output[start:end] = sequence_representation.float().cpu().numpy()
        prediction[start:end] = standardized_prediction.float().cpu().numpy() * target_scale
        if block_index % 200 == 0:
            output.flush()
            print(f"exported rows {end}/{len(features)}", flush=True)
    output.flush()
    del output
    pd.DataFrame({"sample_id": ids, "prediction": prediction}).to_feather(prediction_path)


def load_test_relative319(project_dir: Path) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    data, template, base_columns = full_b001.load_test_only(project_dir)
    add_split_relative_features(project_dir, data, "test")
    columns = [*base_columns, *RELATIVE_COLUMNS]
    if len(columns) != 319 or len(set(columns)) != 319:
        raise AssertionError("Expected 319 unique test features.")
    if not np.array_equal(data["sample_id"].to_numpy(), template["sample_id"].to_numpy()):
        raise AssertionError("Test feature IDs do not match submission order.")
    features = data[columns].to_numpy(dtype=np.float32, copy=True)
    del data
    gc.collect()
    return features, template, columns


def main() -> None:
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")
    gru.EPOCHS = EPOCHS
    device = torch.device("cuda")
    project_dir = Path(__file__).resolve().parents[1]
    run_dir = project_dir / "data" / "interim" / "submissions" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    train_cache_dir = project_dir / "data" / "processed" / "train_sequence_cache_v1"
    test_cache_dir = project_dir / "data" / "processed" / "test_sequence_cache_v1"
    train_cache = np.load(train_cache_dir / "sequences.npy", mmap_mode="r")
    test_cache = np.load(test_cache_dir / "sequences.npy", mmap_mode="r")
    labels = pd.read_feather(
        project_dir / "data" / "raw" / "label.feather",
        columns=["sample_id", "month", "target"],
    ).sort_values("sample_id").reset_index(drop=True)
    template = pd.read_csv(project_dir / "data" / "raw" / "submission.csv")
    if train_cache.shape != (len(labels), 14, 200):
        raise AssertionError("Train sequence cache and labels differ.")
    if test_cache.shape != (len(template), 14, 200):
        raise AssertionError("Test sequence cache and template differ.")
    if set(labels["month"].unique()) != set(range(71)):
        raise AssertionError("Full training data must contain months 0-70.")
    gru.check_cache_layout(train_cache)
    gru.check_cache_layout(test_cache)

    raw_means, raw_scales = load_raw_statistics(train_cache_dir)
    full_means, full_scales = fit_fold_standardization(
        train_cache, len(labels), run_dir / "full_standardization.npz"
    )
    train_builder = FullSequenceBatchBuilder(
        train_cache, raw_means, raw_scales, full_means, full_scales, device
    )
    test_builder = FullSequenceBatchBuilder(
        test_cache, raw_means, raw_scales, full_means, full_scales, device
    )
    train_features, feature_columns = gru.load_relative319(project_dir, labels)
    target = labels["target"].to_numpy(dtype=np.float32)
    target_mean = float(target.mean(dtype=np.float64))
    target_scale = float(target.std(dtype=np.float64))
    target_scaled = ((target - target_mean) / target_scale).astype(np.float32)
    input_size = len(RAW_CHANNEL_NAMES) + len(DERIVED_CHANNEL_NAMES)

    training_logs = {}
    for seed in SEEDS:
        seed_dir = run_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        preprocessor = fit_preprocessor(
            train_features,
            feature_columns,
            seed,
            seed_dir / "quantile_preprocessing.npz",
            device,
        )
        gru.SEED = seed
        gru.seed_everything(seed)
        model = gru.JointRegressor(input_size, preprocessor.output_dimension, True).to(device)
        gru.check_padding_invariance(model, input_size, device)
        training_logs[str(seed)] = train_joint_model(
            seed=seed,
            model=model,
            builder=train_builder,
            features=train_features,
            preprocessor=preprocessor,
            target_scaled=target_scaled,
            seed_dir=seed_dir,
            device=device,
        )
        export_split(
            model=model,
            builder=train_builder,
            features=train_features,
            preprocessor=preprocessor,
            target_scale=target_scale,
            ids=labels["sample_id"].to_numpy(),
            embedding_path=seed_dir / "train_embeddings.npy",
            prediction_path=seed_dir / "train_predictions.feather",
        )
        atomic_save(
            {
                "model_state": model.state_dict(),
                "seed": seed,
                "epochs": EPOCHS,
                "input_size": input_size,
                "static_input_size": preprocessor.output_dimension,
                "feature_columns": feature_columns,
                "target_mean": target_mean,
                "target_scale": target_scale,
            },
            seed_dir / "model_epoch06.pt",
        )
        del model, preprocessor
        torch.cuda.empty_cache()
        gc.collect()

    test_features, loaded_template, test_columns = load_test_relative319(project_dir)
    if feature_columns != test_columns:
        raise AssertionError("Train and test relative319 schemas differ.")
    if not np.array_equal(loaded_template["sample_id"].to_numpy(), template["sample_id"].to_numpy()):
        raise AssertionError("Loaded test template order differs.")
    for seed in SEEDS:
        seed_dir = run_dir / f"seed{seed}"
        preprocessor = fit_preprocessor(
            train_features,
            feature_columns,
            seed,
            seed_dir / "quantile_preprocessing.npz",
            device,
        )
        saved_model = torch.load(seed_dir / "model_epoch06.pt", map_location=device, weights_only=False)
        model = gru.JointRegressor(
            saved_model["input_size"], saved_model["static_input_size"], True
        ).to(device)
        model.load_state_dict(saved_model["model_state"])
        export_split(
            model=model,
            builder=test_builder,
            features=test_features,
            preprocessor=preprocessor,
            target_scale=target_scale,
            ids=template["sample_id"].to_numpy(),
            embedding_path=seed_dir / "test_embeddings.npy",
            prediction_path=seed_dir / "test_predictions.feather",
        )
        del model, preprocessor, saved_model
        torch.cuda.empty_cache()
        gc.collect()

    train_embedding42 = np.load(run_dir / "seed42" / "train_embeddings.npy", mmap_mode="r")
    train_embedding137 = np.load(run_dir / "seed137" / "train_embeddings.npy", mmap_mode="r")
    train_matrix = np.empty((len(labels), 319 + gru.HIDDEN_SIZE), dtype=np.float32)
    train_matrix[:, :319] = train_features
    train_matrix[:, 319:] = 0.5 * np.asarray(train_embedding42)
    train_matrix[:, 319:] += 0.5 * np.asarray(train_embedding137)
    del train_features, train_embedding42, train_embedding137, target_scaled
    gc.collect()

    tree_parameters = model_parameters(
        n_estimators=800,
        colsample_bytree=0.8,
        device="cuda",
        n_jobs=2,
        random_state=42,
    )
    tree = XGBRegressor(**tree_parameters)
    centered_target = target.astype(np.float64) - target_mean
    tree_started = time.perf_counter()
    tree.fit(train_matrix, centered_target)
    tree_seconds = time.perf_counter() - tree_started
    del train_matrix, centered_target
    gc.collect()
    save_model_safely(tree, project_dir / "outputs" / "models" / f"{RUN_NAME}_xgb.json")

    test_embedding42 = np.load(run_dir / "seed42" / "test_embeddings.npy", mmap_mode="r")
    test_embedding137 = np.load(run_dir / "seed137" / "test_embeddings.npy", mmap_mode="r")
    test_matrix = np.empty((len(template), 319 + gru.HIDDEN_SIZE), dtype=np.float32)
    test_matrix[:, :319] = test_features
    test_matrix[:, 319:] = 0.5 * np.asarray(test_embedding42)
    test_matrix[:, 319:] += 0.5 * np.asarray(test_embedding137)
    embedding_tree_prediction = np.asarray(tree.predict(test_matrix), dtype=np.float64)
    del test_matrix, test_features, test_embedding42, test_embedding137, tree
    gc.collect()

    existing = pd.read_feather(
        project_dir / "outputs" / "predictions"
        / "tabm_quantile_rel319_coslr15_tree053r_blend75_fulltrain_trim1_test.feather"
    )
    if not np.array_equal(existing["sample_id"].to_numpy(), template["sample_id"].to_numpy()):
        raise AssertionError("Existing full-data prediction IDs differ.")
    joint42 = pd.read_feather(run_dir / "seed42" / "test_predictions.feather")
    joint137 = pd.read_feather(run_dir / "seed137" / "test_predictions.feather")
    for frame in (joint42, joint137):
        if not np.array_equal(frame["sample_id"].to_numpy(), template["sample_id"].to_numpy()):
            raise AssertionError("Joint-GRU test prediction IDs differ.")

    joint_prediction = 0.5 * unit(joint42["prediction"].to_numpy(dtype=np.float64))
    joint_prediction += 0.5 * unit(joint137["prediction"].to_numpy(dtype=np.float64))
    tree_blend = (
        0.75 * unit(existing["tabm_prediction"].to_numpy(dtype=np.float64))
        + 0.20 * unit(existing["tree_prediction"].to_numpy(dtype=np.float64))
        + 0.05 * unit(embedding_tree_prediction)
    )
    final_prediction = 0.90 * unit(tree_blend) + 0.10 * unit(joint_prediction)
    if not np.isfinite(final_prediction).all():
        raise AssertionError("Final prediction contains invalid values.")

    prediction_dir = project_dir / "outputs" / "predictions"
    submission_dir = project_dir / "outputs" / "submissions"
    metadata_dir = project_dir / "outputs" / "submission_metadata"
    for directory in (prediction_dir, submission_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{RUN_NAME}_test.feather"
    submission_path = submission_dir / f"{RUN_NAME}.csv"
    pd.DataFrame(
        {
            "sample_id": template["sample_id"].to_numpy(),
            "tabm_prediction": existing["tabm_prediction"].to_numpy(),
            "old_tree_prediction": existing["tree_prediction"].to_numpy(),
            "embedding_tree_prediction": embedding_tree_prediction,
            "joint_seed42_prediction": joint42["prediction"].to_numpy(),
            "joint_seed137_prediction": joint137["prediction"].to_numpy(),
            "prediction": final_prediction,
        }
    ).to_feather(prediction_path)
    submission = template[["sample_id"]].copy()
    submission["prediction"] = final_prediction
    submission.to_csv(submission_path, index=False)

    metadata = {
        "run_name": RUN_NAME,
        "status": "complete",
        "submission_status": "prepared_not_uploaded",
        "training_months": "0-70",
        "train_rows": len(labels),
        "test_rows": len(template),
        "feature_count": 319,
        "embedding_feature_count": gru.HIDDEN_SIZE,
        "joint_seeds": list(SEEDS),
        "joint_epochs": EPOCHS,
        "tree_parameters": tree_parameters,
        "tree_training_seconds": tree_seconds,
        "tree_training_embeddings_are_in_sample_supervised_representations": True,
        "blend": {
            "tree_basket": {"tabm": 0.75, "old_tree": 0.20, "gru_embedding_tree": 0.05},
            "final": {"tree_basket": 0.90, "joint_gru_seed_ensemble": 0.10},
            "effective_weights": {"tabm": 0.675, "old_tree": 0.18, "gru_embedding_tree": 0.045, "joint_gru_seed_ensemble": 0.10},
        },
        "target_mean": target_mean,
        "target_scale": target_scale,
        "training_logs": training_logs,
        "prediction_path": str(prediction_path),
        "submission_path": str(submission_path),
        "prediction_mean": float(final_prediction.mean()),
        "prediction_std": float(final_prediction.std(ddof=0)),
        "prediction_min": float(final_prediction.min()),
        "prediction_max": float(final_prediction.max()),
    }
    (metadata_dir / f"{RUN_NAME}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
