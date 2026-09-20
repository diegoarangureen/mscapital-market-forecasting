"""Evaluate saved temporal RealMLP models on late chronological windows."""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import train_full_realmlp379_rq_corr095_blend_tabm as base


reference = base.reference
experiment = base.experiment
RUN_NAME = "realmlp379_rq16_temporal3fold_e8"
MODEL_DIR = PROJECT / "outputs" / "models" / RUN_NAME
OUTPUT_PATH = (
    PROJECT
    / "outputs"
    / "submission_metadata"
    / f"{RUN_NAME}_late_validation.json"
)


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(
        np.dot(target, prediction)
        / (np.linalg.norm(target) * np.linalg.norm(prediction) + 1e-30)
    )


def unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(values.std(), 1e-12)


def predict(model: torch.nn.Module, numerical: torch.Tensor, categorical: torch.Tensor) -> np.ndarray:
    parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(numerical), reference.EVAL_BATCH_SIZE):
            stop = min(start + reference.EVAL_BATCH_SIZE, len(numerical))
            part = model(numerical[start:stop], categorical[start:stop]).squeeze(-1)
            parts.append(part.detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def main() -> None:
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    train_features, target_map, feature_names = experiment.ensure_our379_cache()
    train_ids = np.load(base.SOURCE_CACHE / "sample_ids.npy", mmap_mode="r")
    months = np.asarray(
        np.load(base.SOURCE_CACHE / "months.npy", mmap_mode="r"), dtype=np.int16
    )
    target = np.asarray(target_map, dtype=np.float32).copy()
    raw_train = np.array(train_features, dtype=np.float32, copy=True)
    dummy_test = np.zeros((1, raw_train.shape[1]), dtype=np.float32)
    raw_train, dummy_test = base.fill_missing(raw_train, dummy_test)
    keep, kept_names, selection = base.low_memory_select(
        train_ids, raw_train, target, feature_names
    )
    selected_train = raw_train[:, keep].copy()
    selected_dummy = dummy_test[:, keep].copy()
    del raw_train, dummy_test
    gc.collect()
    train_num, _, train_cat, _, cat_dims, preprocessing = base.prepare_in_place(
        selected_train, selected_dummy
    )
    del selected_train, selected_dummy
    gc.collect()

    broad_mask = (months >= 62) & (months <= 70) & (months != 66)
    strict_mask = (months >= 63) & (months <= 70) & (months != 66)
    union_mask = broad_mask
    union_indices = np.flatnonzero(union_mask)
    valid_num = torch.from_numpy(train_num[union_indices]).to("cuda")
    valid_cat = torch.from_numpy(train_cat[union_indices]).to("cuda")
    valid_target = target[union_indices].astype(np.float64)
    valid_months = months[union_indices]
    del train_num, train_cat
    gc.collect()

    predictions: dict[int, np.ndarray] = {}
    per_fold = {}
    for fold_end in (52, 57, 62):
        path = MODEL_DIR / f"fold_end{fold_end}_seed{reference.SEED}.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = reference.RealMLPRQ(valid_num.shape[1], cat_dims).to("cuda")
        model.load_state_dict(state)
        fold_prediction = predict(model, valid_num, valid_cat)
        if not np.isfinite(fold_prediction).all():
            raise AssertionError(f"Nonfinite validation prediction for fold_end={fold_end}")
        predictions[fold_end] = fold_prediction
        broad_score = cosine(valid_target, fold_prediction)
        strict_rows = valid_months >= 63
        strict_score = cosine(valid_target[strict_rows], fold_prediction[strict_rows])
        per_fold[str(fold_end)] = {
            "broad_62_70_without_66": broad_score,
            "strict_63_70_without_66": strict_score,
            "prediction_std": float(fold_prediction.std()),
        }
        del model, state
        torch.cuda.empty_cache()

    ensemble_52_57 = np.mean([unit(predictions[52]), unit(predictions[57])], axis=0)
    ensemble_all = np.mean(
        [unit(predictions[52]), unit(predictions[57]), unit(predictions[62])], axis=0
    )
    strict_rows = valid_months >= 63
    report = {
        "run_name": RUN_NAME,
        "validation_note": (
            "Saved models only; no retraining. Feature selection and preprocessing reproduce "
            "the training script's shared-full-labeled-data convention."
        ),
        "kept_features": len(kept_names),
        "selection": selection,
        "preprocessing": preprocessing,
        "rows": {
            "broad_62_70_without_66": int(broad_mask.sum()),
            "strict_63_70_without_66": int(strict_mask.sum()),
        },
        "per_fold": per_fold,
        "ensemble_52_57": {
            "broad_62_70_without_66": cosine(valid_target, ensemble_52_57),
            "strict_63_70_without_66": cosine(
                valid_target[strict_rows], ensemble_52_57[strict_rows]
            ),
        },
        "ensemble_all3": {
            "strict_63_70_without_66": cosine(
                valid_target[strict_rows], ensemble_all[strict_rows]
            )
        },
        "pairwise_prediction_correlation": np.corrcoef(
            np.column_stack([predictions[52], predictions[57], predictions[62]]),
            rowvar=False,
        ).tolist(),
        "historical_realmlp_baseline_raw_62_70_without_66": 0.15466833910919217,
    }
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
