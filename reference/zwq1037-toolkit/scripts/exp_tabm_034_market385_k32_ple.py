"""Test piecewise-linear embeddings on the frozen TabM385 k32 recipe."""

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
    PROJECT_DIR
    / "data"
    / "interim"
    / "kaggle_kernels"
    / "relative319_xs_tabm_notebook"
)
sys.path.insert(0, str(REFERENCE_DIR))

import run_extracted as recipe
from rtdl_num_embeddings import PiecewiseLinearEmbeddings

import exp_gru_003_strong_joint as gru
from build_order_quote_position_features import FEATURE_COLUMNS as ORDER_COLUMNS


EXPERIMENT_ID = "EXP-TABM-034-MARKET385-K32-PLE32"
RUN_DIR = PROJECT_DIR / "data" / "interim" / "tree_experiments" / EXPERIMENT_ID
FEATURE_COLUMNS = (
    "mid_return_180 mid_return_60 mid_return_20 realized_volatility_20 "
    "realized_volatility_60 mid_momentum_60"
).split()
FOLDS = {
    "train059_valid6270_ex66": (59, 62, 70, "months_62_70_without_66"),
}
PLE_FIT_ROWS = 50_000
PLE_N_BINS = 32
PLE_D_EMBEDDING = 8


def save_model_result(result: dict, path: Path) -> None:
    """Save metrics and training metadata without the large prediction arrays."""

    compact = {
        key: value
        for key, value in result.items()
        if key not in {"validation_indices", "prediction"}
    }
    path.write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@torch.inference_mode()
def fit_ple_bins(
    features: np.ndarray,
    train_indices: np.ndarray,
    preprocessor: recipe.QuantilePreprocessor,
) -> list[torch.Tensor]:
    """Fit quantile bin edges on a fixed training-only sample.

    The existing quantile preprocessor can create constant columns. PLE requires
    at least two distinct edges, so a constant column receives a harmless
    interval around its observed value.
    """

    generator = np.random.default_rng(recipe.SEED + 34)
    sampled_indices = generator.choice(
        train_indices,
        size=min(PLE_FIT_ROWS, len(train_indices)),
        replace=False,
    )
    transformed = preprocessor.transform(features[sampled_indices]).float()
    probabilities = torch.linspace(
        0.0,
        1.0,
        PLE_N_BINS + 1,
        device=transformed.device,
        dtype=transformed.dtype,
    )
    bins: list[torch.Tensor] = []
    for column_index in range(transformed.shape[1]):
        column = transformed[:, column_index]
        edges = torch.quantile(column, probabilities).unique(sorted=True)
        if edges.numel() < 2:
            value = edges[0] if edges.numel() else torch.zeros((), device=column.device)
            edges = torch.stack([value - 1.0, value + 1.0])
        bins.append(edges)
    del transformed
    torch.cuda.empty_cache()
    return bins


def run_model_ple(
    name: str,
    features: np.ndarray,
    target: np.ndarray,
    months: np.ndarray,
    train_end: int,
    valid_start: int,
    valid_end: int,
    device: torch.device,
) -> dict:
    """Run the frozen k32 model with PLE as the only structural change."""

    train_indices = np.flatnonzero(months <= train_end)
    valid_indices = np.flatnonzero((months >= valid_start) & (months <= valid_end))
    knots, medians, missing_columns = recipe.fit_quantile_knots(
        features, train_indices
    )
    preprocessor = recipe.QuantilePreprocessor(
        knots, medians, missing_columns, device
    )
    bins = fit_ple_bins(features, train_indices, preprocessor)
    embeddings = PiecewiseLinearEmbeddings(
        bins,
        d_embedding=PLE_D_EMBEDDING,
        activation=True,
        version="B",
    )

    target_mean = float(target[train_indices].mean(dtype=np.float64))
    target_std = float(target[train_indices].std(dtype=np.float64))
    scaled_target = ((target - target_mean) / target_std).astype(np.float32)
    recipe.seed_everything(recipe.SEED)
    model = recipe.TabM.make(
        n_num_features=preprocessor.output_dimension,
        cat_cardinalities=None,
        d_out=1,
        num_embeddings=embeddings,
        k=32,
        n_blocks=2,
        d_block=256,
        dropout=0.1,
        arch_type="tabm",
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=recipe.LEARNING_RATE,
        weight_decay=recipe.WEIGHT_DECAY,
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    logs: list[dict] = []
    started = time.perf_counter()

    for epoch in range(1, recipe.EPOCHS + 1):
        rate = recipe.learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = rate
        model.train()
        shuffled = np.random.default_rng(recipe.SEED + epoch).permutation(
            train_indices
        )
        total_loss = 0.0
        total_count = 0
        for start in range(0, len(shuffled), recipe.BATCH_SIZE):
            batch_indices = shuffled[start : start + recipe.BATCH_SIZE]
            batch = preprocessor.transform(features[batch_indices])
            batch_target = torch.as_tensor(
                scaled_target[batch_indices], dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                member_predictions = model(batch).squeeze(-1)
                loss = F.mse_loss(
                    member_predictions.float(),
                    batch_target[:, None].expand_as(member_predictions),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(batch_indices)
            total_count += len(batch_indices)
        logs.append(
            {
                "epoch": epoch,
                "learning_rate": rate,
                "train_mse": total_loss / total_count,
            }
        )
        print(
            f"{name} epoch={epoch:02d} loss={logs[-1]['train_mse']:.7f}",
            flush=True,
        )

    prediction = recipe.predict(
        model, features, valid_indices, preprocessor, amp_dtype
    ).astype(np.float64)
    prediction *= target_std
    metrics = recipe.evaluate(
        target[valid_indices].astype(np.float64), prediction, months[valid_indices]
    )
    result = {
        "name": name,
        "feature_count": int(features.shape[1]),
        "preprocessed_feature_count": int(preprocessor.output_dimension),
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "ple_fit_rows": min(PLE_FIT_ROWS, len(train_indices)),
        "ple_n_bins": PLE_N_BINS,
        "ple_d_embedding": PLE_D_EMBEDDING,
        "ple_version": "B",
        "seconds": time.perf_counter() - started,
        "logs": logs,
        "metrics": metrics,
        "validation_indices": valid_indices,
        "prediction": prediction,
    }
    del model, optimizer, preprocessor, bins, embeddings, knots, medians
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    labels = pd.read_feather(
        PROJECT_DIR / "data" / "raw" / "label.feather",
        columns=["sample_id", "month", "target"],
    ).sort_values("sample_id").reset_index(drop=True)
    base, base_columns = gru.load_relative319(PROJECT_DIR, labels)
    months = labels["month"].to_numpy(dtype=np.int16, copy=True)
    target = labels["target"].to_numpy(dtype=np.float32, copy=True)
    sample_ids = labels["sample_id"].to_numpy(copy=True)
    xs40, xs_columns = recipe.add_relative_features(base, months, base_columns)

    state = pd.read_feather(
        PROJECT_DIR
        / "data"
        / "processed"
        / "train_market_microstructure_features.feather",
        columns=["sample_id", *FEATURE_COLUMNS],
    ).sort_values("sample_id").reset_index(drop=True)
    order_frame = pd.read_feather(
        PROJECT_DIR
        / "data"
        / "processed"
        / "train_order_quote_position_features.feather"
    ).sort_values("sample_id").reset_index(drop=True)
    if not np.array_equal(sample_ids, state["sample_id"].to_numpy()):
        raise AssertionError("Market feature sample IDs do not match labels.")
    if not np.array_equal(sample_ids, order_frame["sample_id"].to_numpy()):
        raise AssertionError("Order feature sample IDs do not match labels.")
    state_values = state[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    order_values = order_frame[ORDER_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    del state, order_frame, labels
    gc.collect()

    if len(set([*base_columns, *xs_columns, *ORDER_COLUMNS, *FEATURE_COLUMNS])) != 385:
        raise AssertionError("Expected 385 unique candidate columns.")
    candidate_features = np.concatenate(
        [base, xs40, order_values, state_values], axis=1
    )
    if candidate_features.shape[1] != 385:
        raise AssertionError("Expected 385 candidate columns.")

    all_results: dict[str, dict] = {}
    device = torch.device("cuda")
    for fold_name, (train_end, valid_start, valid_end, score_key) in FOLDS.items():
        fold_dir = RUN_DIR / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        baseline_dir = (
            PROJECT_DIR
            / "data"
            / "interim"
            / "tree_experiments"
            / "EXP-TABM-032-MARKET385-K32"
            / fold_name
        )
        baseline = json.loads(
            (baseline_dir / "candidate_result.json").read_text(encoding="utf-8")
        )
        baseline_predictions = pd.read_feather(
            baseline_dir / "validation_predictions.feather"
        ).sort_values("sample_id").reset_index(drop=True)
        validation_indices = np.flatnonzero(
            (months >= valid_start) & (months <= valid_end)
        )
        if not np.array_equal(
            sample_ids[validation_indices], baseline_predictions["sample_id"].to_numpy()
        ):
            raise AssertionError("Baseline validation rows do not match this fold.")
        baseline_prediction = baseline_predictions["candidate"].to_numpy(
            dtype=np.float64, copy=True
        )

        candidate = run_model_ple(
            f"{fold_name}_tabm385_k32_ple32",
            candidate_features,
            target,
            months,
            train_end,
            valid_start,
            valid_end,
            device,
        )
        if not np.array_equal(validation_indices, candidate["validation_indices"]):
            raise AssertionError("Candidate validation rows differ from baseline.")
        baseline_score = float(baseline["metrics"][score_key])
        candidate_score = float(candidate["metrics"][score_key])
        row = {
            "fold": fold_name,
            "score_key": score_key,
            "baseline": baseline_score,
            "candidate": candidate_score,
            "delta": candidate_score - baseline_score,
        }
        all_results[fold_name] = row
        pd.DataFrame(
            {
                "sample_id": sample_ids[validation_indices],
                "month": months[validation_indices],
                "target": target[validation_indices],
                "baseline": baseline_prediction,
                "candidate": candidate["prediction"],
            }
        ).to_feather(fold_dir / "validation_predictions.feather")
        save_model_result(baseline, fold_dir / "baseline_result.json")
        save_model_result(candidate, fold_dir / "candidate_result.json")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    average_delta = float(np.mean([row["delta"] for row in all_results.values()]))
    passed = average_delta >= 0.0010
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "baseline": "TabM385 k32 without numerical embeddings",
        "candidate": "TabM385 k32 with PLE32 d8 version-B",
        "single_variable_change": "PiecewiseLinearEmbeddings",
        "folds": all_results,
        "average_delta": average_delta,
        "pass_threshold": 0.0010,
        "passed": passed,
        "next_action": "seed137_confirmation" if passed else "blend_gate_or_stop",
    }
    (RUN_DIR / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
