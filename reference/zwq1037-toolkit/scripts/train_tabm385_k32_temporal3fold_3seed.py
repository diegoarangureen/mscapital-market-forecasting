"""Train a 3 temporal-fold x 3 seed TabM385-k32 test ensemble."""

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
from build_order_quote_position_features import FEATURE_COLUMNS as ORDER_COLUMNS
from train_full_gru_embedding_candidate import load_test_relative319


RUN_NAME = "tabm385_k32_temporal3fold_3seed"
RUN_DIR = PROJECT_DIR / "data" / "interim" / "submissions" / RUN_NAME
OUTPUT_DIR = PROJECT_DIR / "outputs" / "submissions"
PREDICTION_DIR = PROJECT_DIR / "outputs" / "predictions" / RUN_NAME
METADATA_PATH = PROJECT_DIR / "outputs" / "submission_metadata" / f"{RUN_NAME}.json"
FOLD_ENDS = (52, 57, 62)
SEEDS = (42, 137, 2026)
MARKET6_COLUMNS = (
    "mid_return_180 mid_return_60 mid_return_20 realized_volatility_20 "
    "realized_volatility_60 mid_momentum_60"
).split()


def make_model(input_dimension: int, device: torch.device):
    return recipe.TabM.make(
        n_num_features=input_dimension,
        cat_cardinalities=None,
        d_out=1,
        k=32,
        n_blocks=2,
        d_block=256,
        dropout=0.1,
        arch_type="tabm",
    ).to(device)


def load_features():
    labels = pd.read_feather(
        PROJECT_DIR / "data" / "raw" / "label.feather",
        columns=["sample_id", "month", "target"],
    ).sort_values("sample_id").reset_index(drop=True)
    train_base, train_columns = gru.load_relative319(PROJECT_DIR, labels)
    train_months = labels["month"].to_numpy(dtype=np.int16, copy=True)
    train_xs, xs_columns = recipe.add_relative_features(
        train_base, train_months, train_columns
    )
    train_order = pd.read_feather(
        PROJECT_DIR / "data" / "processed" / "train_order_quote_position_features.feather",
        columns=["sample_id", *ORDER_COLUMNS],
    ).sort_values("sample_id").reset_index(drop=True)
    train_market = pd.read_feather(
        PROJECT_DIR / "data" / "processed" / "train_market_microstructure_features.feather",
        columns=["sample_id", *MARKET6_COLUMNS],
    ).sort_values("sample_id").reset_index(drop=True)
    label_ids = labels["sample_id"].to_numpy()
    if not np.array_equal(label_ids, train_order["sample_id"].to_numpy()):
        raise AssertionError("Train order IDs do not align.")
    if not np.array_equal(label_ids, train_market["sample_id"].to_numpy()):
        raise AssertionError("Train market IDs do not align.")
    train_features = np.concatenate(
        [
            train_base,
            train_xs,
            train_order[ORDER_COLUMNS].to_numpy(dtype=np.float32),
            train_market[MARKET6_COLUMNS].to_numpy(dtype=np.float32),
        ],
        axis=1,
    )
    del train_base, train_xs, train_order, train_market
    gc.collect()

    test_base, template, test_columns = load_test_relative319(PROJECT_DIR)
    if test_columns != train_columns:
        raise AssertionError("Train and test Relative319 columns differ.")
    test_xs, test_xs_columns = recipe.add_relative_features(
        test_base, np.zeros(len(template), dtype=np.int16), test_columns
    )
    if test_xs_columns != xs_columns:
        raise AssertionError("Train and test XS40 columns differ.")
    test_order = pd.read_feather(
        PROJECT_DIR / "data" / "processed" / "test_order_quote_position_features.feather",
        columns=["sample_id", *ORDER_COLUMNS],
    ).sort_values("sample_id").reset_index(drop=True)
    test_market = pd.read_feather(
        PROJECT_DIR / "data" / "processed" / "test_market_microstructure_features.feather",
        columns=["sample_id", *MARKET6_COLUMNS],
    ).sort_values("sample_id").reset_index(drop=True)
    template_ids = template["sample_id"].to_numpy()
    if not np.array_equal(template_ids, test_order["sample_id"].to_numpy()):
        raise AssertionError("Test order IDs do not align.")
    if not np.array_equal(template_ids, test_market["sample_id"].to_numpy()):
        raise AssertionError("Test market IDs do not align.")
    test_features = np.concatenate(
        [
            test_base,
            test_xs,
            test_order[ORDER_COLUMNS].to_numpy(dtype=np.float32),
            test_market[MARKET6_COLUMNS].to_numpy(dtype=np.float32),
        ],
        axis=1,
    )
    if train_features.shape[1] != 385 or test_features.shape[1] != 385:
        raise AssertionError("Expected 385 features.")
    del test_base, test_xs, test_order, test_market
    gc.collect()
    feature_columns = [
        *train_columns, *xs_columns, *ORDER_COLUMNS, *MARKET6_COLUMNS
    ]
    return labels, train_features, test_features, template, feature_columns


def fit_or_load_preprocessor(
    fold_dir: Path,
    features: np.ndarray,
    train_indices: np.ndarray,
    feature_columns: list[str],
    device: torch.device,
):
    path = fold_dir / "quantile_preprocessing.npz"
    if path.exists():
        saved = np.load(path)
        knots = saved["knots"]
        medians = saved["medians"]
        missing_columns = saved["missing_columns"]
    else:
        knots, medians, missing_columns = recipe.fit_quantile_knots(
            features, train_indices
        )
        np.savez_compressed(
            path,
            knots=knots,
            medians=medians,
            missing_columns=missing_columns,
            feature_columns=np.asarray(feature_columns),
        )
    return recipe.QuantilePreprocessor(
        knots, medians, missing_columns, device
    )


def train_one_model(
    fold_end: int,
    seed: int,
    train_features: np.ndarray,
    test_features: np.ndarray,
    months: np.ndarray,
    target: np.ndarray,
    feature_columns: list[str],
    test_ids: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    fold_dir = RUN_DIR / f"train000_{fold_end:03d}"
    model_dir = fold_dir / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = PREDICTION_DIR / f"fold_end{fold_end}_seed{seed}.feather"
    result_path = model_dir / "result.json"
    if prediction_path.exists() and result_path.exists():
        frame = pd.read_feather(prediction_path).sort_values("sample_id")
        if not np.array_equal(frame["sample_id"].to_numpy(), test_ids):
            raise AssertionError("Saved prediction IDs do not align.")
        return (
            frame["prediction"].to_numpy(dtype=np.float64),
            json.loads(result_path.read_text(encoding="utf-8")),
        )

    train_indices = np.flatnonzero(months <= fold_end)
    preprocessor = fit_or_load_preprocessor(
        fold_dir, train_features, train_indices, feature_columns, device
    )
    target_mean = float(target[train_indices].mean(dtype=np.float64))
    target_std = float(target[train_indices].std(dtype=np.float64))
    scaled_target = ((target - target_mean) / target_std).astype(np.float32)

    recipe.seed_everything(seed)
    model = make_model(preprocessor.output_dimension, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=recipe.LEARNING_RATE,
        weight_decay=recipe.WEIGHT_DECAY,
    )
    amp_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    checkpoint_path = model_dir / "training_checkpoint.pt"
    logs: list[dict] = []
    start_epoch = 1
    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        logs = checkpoint["logs"]
        start_epoch = int(checkpoint["next_epoch"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state"]]
        )
        print(
            f"resume fold_end={fold_end} seed={seed} epoch={start_epoch}",
            flush=True,
        )

    for epoch in range(start_epoch, recipe.EPOCHS + 1):
        started = time.perf_counter()
        learning_rate = recipe.learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        shuffled = np.random.default_rng(seed + epoch).permutation(train_indices)
        total_loss = 0.0
        total_rows = 0
        for start in range(0, len(shuffled), recipe.BATCH_SIZE):
            indices = shuffled[start : start + recipe.BATCH_SIZE]
            batch = preprocessor.transform(train_features[indices])
            batch_target = torch.as_tensor(
                scaled_target[indices], dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                members = model(batch).squeeze(-1)
                loss = F.mse_loss(
                    members.float(),
                    batch_target[:, None].expand_as(members),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(indices)
            total_loss += float(loss.detach().cpu()) * count
            total_rows += count
        log = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_mse": total_loss / total_rows,
            "seconds": time.perf_counter() - started,
        }
        logs.append(log)
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "next_epoch": epoch + 1,
                "logs": logs,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
            },
            temporary,
        )
        temporary.replace(checkpoint_path)
        print(
            f"fold_end={fold_end} seed={seed} epoch={epoch:02d}/{recipe.EPOCHS} "
            f"mse={log['train_mse']:.7f} seconds={log['seconds']:.1f}",
            flush=True,
        )

    torch.save(model.state_dict(), model_dir / "model.pt")
    prediction = recipe.predict(
        model,
        test_features,
        np.arange(len(test_features), dtype=np.int64),
        preprocessor,
        amp_dtype,
    ).astype(np.float64) * target_std
    if not np.isfinite(prediction).all():
        raise AssertionError("Nonfinite test prediction.")
    pd.DataFrame(
        {"sample_id": test_ids, "prediction": prediction}
    ).to_feather(prediction_path)
    result = {
        "fold_end": fold_end,
        "seed": seed,
        "train_rows": int(len(train_indices)),
        "epochs": recipe.EPOCHS,
        "target_mean": target_mean,
        "target_std": target_std,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "prediction_path": str(prediction_path),
        "logs": logs,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del model, optimizer, scaler, preprocessor
    torch.cuda.empty_cache()
    gc.collect()
    return prediction, result


def main() -> None:
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    labels, train_features, test_features, template, feature_columns = load_features()
    months = labels["month"].to_numpy(dtype=np.int16, copy=True)
    target = labels["target"].to_numpy(dtype=np.float32, copy=True)
    test_ids = template["sample_id"].to_numpy(copy=True)
    device = torch.device("cuda")

    model_predictions: list[np.ndarray] = []
    model_results: list[dict] = []
    fold_predictions: dict[int, list[np.ndarray]] = {
        fold_end: [] for fold_end in FOLD_ENDS
    }
    for fold_end in FOLD_ENDS:
        for seed in SEEDS:
            prediction, result = train_one_model(
                fold_end,
                seed,
                train_features,
                test_features,
                months,
                target,
                feature_columns,
                test_ids,
                device,
            )
            model_predictions.append(prediction)
            model_results.append(result)
            fold_predictions[fold_end].append(prediction)

        fold_average = np.mean(fold_predictions[fold_end], axis=0)
        fold_path = OUTPUT_DIR / f"{RUN_NAME}_fold_end{fold_end}.csv"
        pd.DataFrame(
            {"sample_id": test_ids, "prediction": fold_average}
        ).to_csv(fold_path, index=False)
        print(
            f"saved fold_end={fold_end} three-seed average: {fold_path}",
            flush=True,
        )

    ensemble = np.mean(model_predictions, axis=0)
    submission_path = OUTPUT_DIR / f"{RUN_NAME}.csv"
    pd.DataFrame(
        {"sample_id": test_ids, "prediction": ensemble}
    ).to_csv(submission_path, index=False)
    correlation = np.corrcoef(np.column_stack(model_predictions), rowvar=False)
    metadata = {
        "run_name": RUN_NAME,
        "model": "official TabM385 k32",
        "fold_ends": list(FOLD_ENDS),
        "seeds": list(SEEDS),
        "model_count": len(model_predictions),
        "epochs_per_model": recipe.EPOCHS,
        "feature_count": len(feature_columns),
        "test_rows": len(test_ids),
        "prediction_mean": float(ensemble.mean()),
        "prediction_std": float(ensemble.std()),
        "pairwise_prediction_correlation": correlation.tolist(),
        "models": model_results,
        "submission_path": str(submission_path),
        "submission_status": "prepared_not_uploaded",
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "submission_path": str(submission_path),
                "models": len(model_predictions),
                "prediction_std": metadata["prediction_std"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
