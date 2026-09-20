"""Wait for Kaggle v24 and download only the compact validation artifacts."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = r"D:\anaconda\envs\pytorch\Scripts\kaggle.exe"
KERNEL = "zwq1037/multistream-factorized-transformer-multiwindow-dev"
VERSION = 24
OUT = ROOT / "data/interim/kaggle_results/market_conditioned_event_v24"
PLAN = ROOT / "outputs/submission_metadata/goal_improve_beyond152_20260919.json"


def command(arguments: list[str], timeout: int = 240) -> str:
    result = subprocess.run([CLI, *arguments], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError("Kaggle operation failed: " + result.stdout + result.stderr)
    return result.stdout


def update_plan(status: str, **values) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    stage = next(item for item in plan["stages"] if item["id"] == "market_conditioned_event_residual")
    stage.update({"status": status, **values})
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    print(f"Continuation plan: {PLAN}", flush=True)
    time.sleep(15 * 60)
    deadline = time.monotonic() + 8 * 60 * 60
    while time.monotonic() < deadline:
        try:
            status = command(["kernels", "status", KERNEL])
        except Exception as exc:
            print(f"status unavailable: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(10 * 60)
            continue
        print(status.strip(), flush=True)
        if "KernelWorkerStatus.ERROR" in status or "KernelWorkerStatus.CANCEL" in status:
            update_plan("remote_failed", kaggle_version=VERSION)
            raise RuntimeError("Kaggle v24 ended without successful completion")
        if "KernelWorkerStatus.COMPLETE" in status:
            break
        time.sleep(10 * 60)
    else:
        update_plan("wait_timeout", kaggle_version=VERSION)
        raise RuntimeError("Kaggle v24 wait deadline reached")

    OUT.mkdir(parents=True, exist_ok=True)
    text = command([
        "kernels", "output", f"{KERNEL}/{VERSION}", "-p", str(OUT), "--force",
        "--file-pattern", r"score_only\.json$|validation_predictions\.feather$|failure\.json$",
    ], timeout=600)
    print(text.strip(), flush=True)
    failures = list(OUT.rglob("failure.json"))
    if failures:
        failure = json.loads(failures[0].read_text(encoding="utf-8"))
        update_plan("remote_failed", kaggle_version=VERSION, failure=failure)
        raise RuntimeError("Kaggle v24 artifact reports failure: " + json.dumps(failure))
    scores = list(OUT.rglob("score_only.json"))
    predictions = list(OUT.rglob("validation_predictions.feather"))
    if len(scores) != 1 or len(predictions) != 1:
        raise AssertionError(f"Expected one score and prediction artifact, got {len(scores)} and {len(predictions)}")
    result = json.loads(scores[0].read_text(encoding="utf-8"))
    update_plan(
        "validation_complete",
        kaggle_version=VERSION,
        passed=bool(result["passed"]),
        score_file=str(scores[0]),
        prediction_file=str(predictions[0]),
        best_delta=result["best"]["delta"],
    )
    print(json.dumps({"passed": result["passed"], "best": result["best"]}), flush=True)
    print(f"NEXT: read {PLAN}, verify both active experiment results, and execute the next authorized gate.", flush=True)


if __name__ == "__main__":
    main()
