"""Wait for Kaggle v23, download compact outputs, and test it in the owned-model blend."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLI = r"D:\anaconda\envs\pytorch\Scripts\kaggle.exe"
KERNEL = "zwq1037/multistream-factorized-transformer-multiwindow-dev"
VERSION = 23
OUT = ROOT / "data/interim/kaggle_results/linear_sequence_v23"
PLAN = ROOT / "outputs/submission_metadata/astra_sequential_plan_20260917.json"
NAMES = ["tabm", "realmlp", "transformer", "gru"]
OWNED4_WEIGHTS = np.asarray(
    [0.31819564837516784, 0.13030480191317811,
     0.19215218382533478, 0.35934736588631944],
    dtype=np.float64,
)


def fresh_token() -> str:
    result = subprocess.run(
        [CLI, "auth", "print-access-token"], capture_output=True, text=True, timeout=90
    )
    if result.returncode:
        raise RuntimeError("Kaggle OAuth refresh failed")
    return result.stdout.strip()


def command(arguments: list[str], token: str, timeout: int = 240) -> str:
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    result = subprocess.run(
        [CLI, *arguments], env=env, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode:
        raise RuntimeError("Kaggle operation failed: " + result.stdout + result.stderr)
    return result.stdout


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(target @ prediction / (np.linalg.norm(target) * np.linalg.norm(prediction) + 1e-30))


def fusion_analysis(candidate: pd.DataFrame) -> dict:
    tabm = pd.read_feather(
        ROOT / "data/interim/tree_experiments/EXP-TABM-032-MARKET385-K32/train059_valid6270_ex66/validation_predictions.feather"
    )[["sample_id", "candidate"]].rename(columns={"candidate": "tabm"})
    real_dir = ROOT / "data/interim/tree_experiments/EXP-REALMLP-009-OUR379-CORR095/train059_valid6270_ex66"
    realmlp = pd.DataFrame({
        "sample_id": np.load(real_dir / "validation_sample_ids.npy"),
        "realmlp": np.load(real_dir / "validation_predictions.npy"),
    })
    transformer = pd.read_csv(
        ROOT / "outputs/kaggle_transformer_v7_result/factorized_transformer/validation_predictions.csv",
        usecols=["sample_id", "prediction"],
    ).rename(columns={"prediction": "transformer"})
    gru = pd.read_csv(
        ROOT / "data/interim/kaggle_outputs/multistream_factorized_gru_timeaware_dev/factorized_gru/validation_predictions.csv",
        usecols=["sample_id", "prediction"],
    ).rename(columns={"prediction": "gru"})

    frame = candidate[["sample_id", "month", "target", "prediction"]].rename(
        columns={"prediction": "linear_sequence"}
    )
    expected_rows = len(frame)
    for source in (tabm, realmlp, transformer, gru):
        frame = frame.merge(source, on="sample_id", how="inner", validate="one_to_one")
    if len(frame) != expected_rows or expected_rows != 140_806:
        raise AssertionError(f"Validation alignment failed: {len(frame)} / {expected_rows}")

    selection = frame.month.le(65).to_numpy()
    forward = frame.month.ge(67).to_numpy()
    truth = frame.target.to_numpy(np.float64)
    old = frame[NAMES].to_numpy(np.float64)
    old_scale = np.sqrt(np.mean(old[selection] ** 2, axis=0))
    old_prediction = (old / old_scale) @ OWNED4_WEIGHTS
    new = frame.linear_sequence.to_numpy(np.float64)
    new_scale = float(np.sqrt(np.mean(new[selection] ** 2)))
    new = new / max(new_scale, 1e-12)

    rows = []
    for weight in (0.0, 0.05, 0.10, 0.15):
        prediction = (1.0 - weight) * old_prediction + weight * new
        rows.append({
            "linear_sequence_weight": weight,
            "selection_62_65": cosine(truth[selection], prediction[selection]),
            "forward_67_70": cosine(truth[forward], prediction[forward]),
            "monthly_forward": {
                str(month): cosine(truth[frame.month.eq(month)], prediction[frame.month.eq(month)])
                for month in range(67, 71)
            },
        })
    chosen = max(rows, key=lambda row: row["selection_62_65"])
    baseline = rows[0]
    monthly_delta = {
        month: chosen["monthly_forward"][month] - baseline["monthly_forward"][month]
        for month in baseline["monthly_forward"]
    }
    chosen["forward_delta"] = chosen["forward_67_70"] - baseline["forward_67_70"]
    chosen["monthly_delta"] = monthly_delta
    chosen["passed"] = bool(
        chosen["linear_sequence_weight"] > 0
        and chosen["forward_delta"] >= 0.0003
        and sum(value > 0 for value in monthly_delta.values()) >= 2
        and min(monthly_delta.values()) >= -0.0005
    )
    return {
        "rows": rows,
        "selected_by_62_65": chosen,
        "owned4_scale": old_scale.tolist(),
        "linear_sequence_scale": new_scale,
    }


def update_plan(status: str, **values) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    step = next(item for item in plan["steps"] if item["id"] == "linear_sequence_control")
    step.update({"status": status, **values})
    if status == "validation_complete":
        if values["passed"]:
            plan["next_action"] = (
                "After the other active experiment reaches a terminal state, compare both gates. "
                "If linear sequence remains the strongest passing candidate, prepare its full-train Kaggle version and CSV; do not submit."
            )
        else:
            plan["next_action"] = (
                "Keep the linear sequence control rejected. Let the active local TSMixer finish, verify its gate, "
                "then continue to RealMLP all379 only if TSMixer also fails."
            )
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    print(f"Continuation plan: {PLAN}", flush=True)
    token = fresh_token()
    time.sleep(20 * 60)
    deadline = time.monotonic() + 4 * 60 * 60
    errors = 0
    while time.monotonic() < deadline:
        try:
            status = command(["kernels", "status", KERNEL], token)
            errors = 0
        except Exception:
            errors += 1
            if errors >= 2:
                token = fresh_token()
                errors = 0
            time.sleep(10 * 60)
            continue
        print(status.strip(), flush=True)
        if "KernelWorkerStatus.ERROR" in status or "KernelWorkerStatus.CANCEL" in status:
            update_plan("remote_failed", actual_version=VERSION)
            raise RuntimeError("Kaggle v23 ended without successful completion")
        if "KernelWorkerStatus.COMPLETE" in status:
            break
        time.sleep(10 * 60)
    else:
        update_plan("wait_timeout", actual_version=VERSION)
        raise RuntimeError("Kaggle v23 wait deadline reached")

    OUT.mkdir(parents=True, exist_ok=True)
    output_text = command([
        "kernels", "output", f"{KERNEL}/{VERSION}", "-p", str(OUT), "--force",
        "--file-pattern", r"(score_only|result)\.json$|validation_predictions\.csv$",
    ], token, timeout=600)
    print(output_text.strip(), flush=True)
    score_paths = list(OUT.rglob("score_only.json"))
    prediction_paths = list(OUT.rglob("validation_predictions.csv"))
    if len(score_paths) != 1 or len(prediction_paths) != 1:
        raise AssertionError("Expected exactly one score file and one validation prediction file")
    score = json.loads(score_paths[0].read_text(encoding="utf-8"))
    fusion = fusion_analysis(pd.read_csv(prediction_paths[0]))
    score["owned4_fusion"] = fusion
    score_paths[0].write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
    chosen = fusion["selected_by_62_65"]
    update_plan(
        "validation_complete",
        actual_version=VERSION,
        score_file=str(score_paths[0]),
        passed=chosen["passed"],
        selected_weight=chosen["linear_sequence_weight"],
        forward_delta=chosen["forward_delta"],
    )
    print(json.dumps({"standalone": score.get("best_cosine"), "fusion": chosen}), flush=True)
    print(f"NEXT: read {PLAN} and continue the next already-authorized step.", flush=True)


if __name__ == "__main__":
    main()
