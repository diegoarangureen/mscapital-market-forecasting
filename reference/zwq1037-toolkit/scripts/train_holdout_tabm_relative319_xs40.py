"""Fit the validated Relative319+XS40 TabM recipe on months 0-59."""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = (
    PROJECT_DIR / "data" / "interim" / "kaggle_kernels"
    / "relative319_xs_tabm_notebook"
)
sys.path.insert(0, str(REFERENCE_DIR))
import run_extracted as recipe
import exp_gru_003_strong_joint as gru
from train_full_gru_embedding_candidate import load_test_relative319


RUN_NAME = "tabm_relative319_xs40_seed42_holdout059"
RUN_DIR = PROJECT_DIR / "data" / "interim" / "submissions" / RUN_NAME


def main() -> None:
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    device = torch.device("cuda")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    labels = pd.read_feather(
        PROJECT_DIR / "data" / "raw" / "label.feather",
        columns=["sample_id", "month", "target"],
    ).sort_values("sample_id").reset_index(drop=True)
    train_base, train_columns = gru.load_relative319(PROJECT_DIR, labels)
    train_months = labels["month"].to_numpy()
    train_xs, xs_columns = recipe.add_relative_features(
        train_base, train_months, train_columns
    )
    train_features = np.concatenate([train_base, train_xs], axis=1)
    del train_base, train_xs
    gc.collect()

    test_base, template, test_columns = load_test_relative319(PROJECT_DIR)
    if test_columns != train_columns:
        raise AssertionError("Train and test Relative319 columns differ.")
    test_months = np.zeros(len(template), dtype=np.int16)
    test_xs, test_xs_columns = recipe.add_relative_features(
        test_base, test_months, test_columns
    )
    if test_xs_columns != xs_columns:
        raise AssertionError("Train and test XS40 columns differ.")
    test_features = np.concatenate([test_base, test_xs], axis=1)
    del test_base, test_xs
    gc.collect()
    if train_features.shape[1] != 359 or test_features.shape[1] != 359:
        raise AssertionError("Expected 359 features.")

    train_indices = np.flatnonzero(train_months <= 59).astype(np.int64)
    preprocessing_path = RUN_DIR / "quantile_preprocessing.npz"
    if preprocessing_path.exists():
        saved = np.load(preprocessing_path)
        knots, medians, missing_columns = (
            saved["knots"], saved["medians"], saved["missing_columns"]
        )
    else:
        knots, medians, missing_columns = recipe.fit_quantile_knots(
            train_features, train_indices
        )
        np.savez_compressed(
            preprocessing_path, knots=knots, medians=medians,
            missing_columns=missing_columns,
            feature_columns=np.asarray([*train_columns, *xs_columns]),
        )
    preprocessor = recipe.QuantilePreprocessor(
        knots, medians, missing_columns, device
    )
    target = labels["target"].to_numpy(dtype=np.float32)
    target_mean = float(target[train_indices].mean(dtype=np.float64))
    target_std = float(target[train_indices].std(dtype=np.float64))
    scaled_target = ((target - target_mean) / target_std).astype(np.float32)

    recipe.seed_everything(recipe.SEED)
    model = recipe.make_model(preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe.LEARNING_RATE,
        weight_decay=recipe.WEIGHT_DECAY,
    )
    amp_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    checkpoint_path = RUN_DIR / "training_checkpoint.pt"
    logs = []
    start_epoch = 1
    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        logs = checkpoint["logs"]
        start_epoch = checkpoint["next_epoch"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state"]]
        )
        print(f"resuming at epoch={start_epoch}", flush=True)

    for epoch in range(start_epoch, recipe.EPOCHS + 1):
        started = time.perf_counter()
        rate = recipe.learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = rate
        model.train()
        shuffled = np.random.default_rng(
            recipe.SEED + epoch
        ).permutation(train_indices)
        total_loss = 0.0
        total_count = 0
        for start in range(0, len(shuffled), recipe.BATCH_SIZE):
            indices = shuffled[start:start + recipe.BATCH_SIZE]
            batch = preprocessor.transform(train_features[indices])
            batch_target = torch.as_tensor(
                scaled_target[indices], dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                members = model(batch).squeeze(-1)
                loss = F.mse_loss(
                    members.float(), batch_target[:, None].expand_as(members)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(indices)
            total_loss += float(loss.detach().cpu()) * count
            total_count += count
        logs.append({
            "epoch": epoch, "learning_rate": rate,
            "train_mse": total_loss / total_count,
            "seconds": time.perf_counter() - started,
        })
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "next_epoch": epoch + 1,
            "logs": logs,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        }, temporary)
        temporary.replace(checkpoint_path)
        print(
            f"epoch={epoch}/{recipe.EPOCHS} "
            f"mse={logs[-1]['train_mse']:.7f} "
            f"seconds={logs[-1]['seconds']:.1f}",
            flush=True,
        )

    torch.save(model.state_dict(), RUN_DIR / "model_holdout059.pt")
    test_indices = np.arange(len(template), dtype=np.int64)
    prediction = recipe.predict(
        model, test_features, test_indices, preprocessor, amp_dtype
    ).astype(np.float64) * target_std
    if not np.isfinite(prediction).all():
        raise AssertionError("Nonfinite test prediction.")
    prediction_path = (
        PROJECT_DIR / "outputs" / "predictions" / f"{RUN_NAME}_test.feather"
    )
    submission_path = (
        PROJECT_DIR / "outputs" / "submissions" / f"{RUN_NAME}.csv"
    )
    pd.DataFrame({
        "sample_id": template["sample_id"].to_numpy(),
        "prediction": prediction,
    }).to_feather(prediction_path)
    pd.DataFrame({
        "sample_id": template["sample_id"].to_numpy(),
        "prediction": prediction,
    }).to_csv(submission_path, index=False)
    metadata = {
        "run_name": RUN_NAME,
        "seed": recipe.SEED,
        "epochs": recipe.EPOCHS,
        "feature_count": train_features.shape[1],
        "train_rows": len(train_indices),
        "train_months": "0-59",
        "heldout_months": "60-70",
        "test_rows": len(test_features),
        "validation_50_59": 0.15223842838833979,
        "validation_62_70_no66": 0.1562411575302188,
        "validation_control_62_70_no66": 0.15375214944787352,
        "source_notebook": "relative319_xs_tabm_notebook",
        "prediction_path": str(prediction_path),
        "submission_path": str(submission_path),
        "prediction_std": float(prediction.std()),
        "logs": logs,
        "submission_status": "prepared_not_uploaded",
    }
    metadata_path = (
        PROJECT_DIR / "outputs" / "submission_metadata" / f"{RUN_NAME}.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "submission_path": str(submission_path),
        "prediction_std": metadata["prediction_std"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

