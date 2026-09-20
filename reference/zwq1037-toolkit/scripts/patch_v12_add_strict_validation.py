from pathlib import Path

path = Path(r"F:\深度学习\projects\mscapital_market_forecasting\data\interim\kaggle_kernels\multistream_factorized_transformer_dev\run_v12_gru_temporal3fold.py")
source = path.read_text(encoding="utf-8")
replacements = [
    (
'''def _unit_prediction(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(values.std(), 1e-12)
''',
'''def _unit_prediction(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(values.std(), 1e-12)


def _cosine_score(target, prediction):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(
        np.dot(target, prediction)
        / (np.linalg.norm(target) * np.linalg.norm(prediction) + 1e-30)
    )
'''
    ),
    (
'''    test_static = _load_test_static(test_ids)
    test_idx = np.arange(test_ids.size, dtype=np.int64)
    device = torch.device("cuda")
    fold_predictions, fold_results = [], []
''',
'''    test_static = _load_test_static(test_ids)
    test_idx = np.arange(test_ids.size, dtype=np.int64)
    strict_valid_idx = np.flatnonzero(
        (months >= 63) & (months <= 70) & (months != 66)
    )
    strict_valid_target = target[strict_valid_idx]
    device = torch.device("cuda")
    fold_predictions, fold_validation_predictions, fold_results = [], [], []
'''
    ),
    (
'''        del train_loader, train_ds
        gc.collect(); torch.cuda.empty_cache()
        _STATIC_FEATURES = test_static
''',
'''        del train_loader, train_ds
        gc.collect(); torch.cuda.empty_cache()

        _STATIC_FEATURES = train_static
        strict_valid_ds = GridDataset(
            train_arrays, strict_valid_idx, norm, target=None, target_scale=target_scale
        )
        strict_valid_loader = DataLoader(
            strict_valid_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
        strict_valid_prediction = (
            predict(model, strict_valid_loader, device) * target_scale
        )
        if (
            strict_valid_prediction.shape != (strict_valid_idx.size,)
            or not np.isfinite(strict_valid_prediction).all()
        ):
            raise AssertionError(f"Invalid fold_end={fold_end} validation predictions")
        strict_valid_cosine = _cosine_score(
            strict_valid_target, strict_valid_prediction
        )
        fold_validation_predictions.append(
            _unit_prediction(strict_valid_prediction)
        )
        print(
            f"fold_end={fold_end} strict_valid_cosine={strict_valid_cosine:.9f}",
            flush=True,
        )
        del strict_valid_loader, strict_valid_ds
        gc.collect(); torch.cuda.empty_cache()

        _STATIC_FEATURES = test_static
'''
    ),
    (
'''            "prediction_std": float(prediction.std()), "prediction_file": fold_path.name,
            "checkpoint_file": model_path.name, "training_log": logs,
''',
'''            "prediction_std": float(prediction.std()), "prediction_file": fold_path.name,
            "checkpoint_file": model_path.name,
            "strict_valid_rows": int(strict_valid_idx.size),
            "strict_valid_cosine": strict_valid_cosine,
            "training_log": logs,
'''
    ),
    (
'''    ensemble = np.mean(fold_predictions, axis=0)
    submission_path = WORK_DIR / "factorized_gru379_temporal3fold_e4.csv"
''',
'''    ensemble = np.mean(fold_predictions, axis=0)
    validation_ensemble = np.mean(fold_validation_predictions, axis=0)
    validation_ensemble_cosine = _cosine_score(
        strict_valid_target, validation_ensemble
    )
    submission_path = WORK_DIR / "factorized_gru379_temporal3fold_e4.csv"
'''
    ),
    (
'''        "loss": "0.35 SmoothL1 + 0.65 centered cosine",
        "ensemble_method": "equal average after per-fold z-standardization",
        "prediction_mean": float(ensemble.mean()), "prediction_std": float(ensemble.std()),
''',
'''        "loss": "0.35 SmoothL1 + 0.65 centered cosine",
        "ensemble_method": "equal average after per-fold z-standardization",
        "strict_validation_window": "months 63-70 excluding 66",
        "strict_validation_rows": int(strict_valid_idx.size),
        "strict_validation_ensemble_cosine": validation_ensemble_cosine,
        "prediction_mean": float(ensemble.mean()), "prediction_std": float(ensemble.std()),
'''
    ),
]
for old, new in replacements:
    if source.count(old) != 1:
        raise RuntimeError(f"Expected one match, found {source.count(old)} for {old[:80]!r}")
    source = source.replace(old, new, 1)
path.write_text(source, encoding="utf-8")
print(path)
