"""Wait for GRU EMA Kaggle full train, download compact outputs, and update the continuation plan."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = Path(r"D:\anaconda\envs\pytorch\Scripts\kaggle.exe")
KERNEL = "zwq1037/multistream-gru-transformer-ema-full-train"
VERSION = 3
OUT = ROOT / "data/interim/kaggle_outputs/gru_ema_full_v3"
PLAN = ROOT / "outputs/submission_metadata/ema_suite_plan_20260918.json"
DEST = ROOT / "outputs/submissions/standalone_timeaware_gru379_ema0999_full_e5.csv"


def token(max_wait_seconds=7200):
    deadline = time.monotonic() + max_wait_seconds
    delay = 120
    while time.monotonic() < deadline:
        result = subprocess.run([str(CLI), "auth", "print-access-token"], capture_output=True, text=True, timeout=90)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        message = (result.stdout + result.stderr).strip().replace("\n", " ")
        print(f"Kaggle OAuth unavailable; retry in {delay}s: {message[-300:]}", flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 900)
    raise RuntimeError("Kaggle OAuth refresh retry deadline reached")


def run(args, access_token, timeout=240):
    env = os.environ.copy(); env["KAGGLE_API_TOKEN"] = access_token
    result = subprocess.run([str(CLI), *args], env=env, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def update(status, **values):
    plan = json.loads(PLAN.read_text(encoding="utf-8-sig"))
    stage = next(row for row in plan["stages"] if row["id"] == "timeaware_gru_ema")
    stage.update({"full_csv_status": status, **values})
    if status == "complete":
        plan["next_action"] = "Verify the downloaded GRU EMA CSV, then run the factorized Transformer full-data EMA version on the existing Kaggle notebook. Do not submit."
    elif status == "remote_failed":
        plan["next_action"] = "Inspect the GRU EMA Kaggle failure once, repair deterministic code/configuration errors if possible, then continue the already-authorized full CSV workflow. Do not submit."
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    access_token = token()
    time.sleep(20 * 60)
    deadline = time.monotonic() + 6 * 60 * 60
    failures = 0
    while time.monotonic() < deadline:
        try:
            status = run(["kernels", "status", KERNEL], access_token)
            failures = 0
        except Exception:
            failures += 1
            if failures >= 2:
                access_token = token(); failures = 0
            time.sleep(10 * 60); continue
        print(status.strip(), flush=True)
        if "KernelWorkerStatus.COMPLETE" in status:
            break
        if "KernelWorkerStatus.ERROR" in status or "KernelWorkerStatus.CANCEL" in status:
            update("remote_failed", kernel=KERNEL, version=VERSION)
            raise RuntimeError("Kaggle GRU EMA full train failed")
        time.sleep(10 * 60)
    else:
        update("wait_timeout", kernel=KERNEL, version=VERSION)
        raise RuntimeError("Kaggle GRU EMA wait deadline reached")
    OUT.mkdir(parents=True, exist_ok=True)
    print(run(["kernels", "output", f"{KERNEL}/{VERSION}", "-p", str(OUT), "--force", "--file-pattern", r"(score_only|result)\.json$|standalone_timeaware_gru379_ema0999_full_e5\.csv$"], access_token, timeout=900), flush=True)
    csvs = list(OUT.rglob("standalone_timeaware_gru379_ema0999_full_e5.csv"))
    scores = list(OUT.rglob("score_only.json"))
    if len(csvs) != 1 or len(scores) != 1:
        raise AssertionError(f"Expected one CSV and score file, got {len(csvs)} and {len(scores)}")
    import pandas as pd
    frame = pd.read_csv(csvs[0])
    if list(frame.columns) != ["sample_id", "prediction"] or len(frame) != 647896 or not frame["prediction"].map(lambda x: x == x and abs(x) != float("inf")).all():
        raise AssertionError("Downloaded GRU EMA CSV failed shape/schema/finite checks")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csvs[0], DEST)
    score = json.loads(scores[0].read_text(encoding="utf-8"))
    update("complete", kernel=KERNEL, version=VERSION, csv=str(DEST), score_file=str(scores[0]), visible_gpu_count=score.get("visible_gpu_count"))
    print(json.dumps({"csv": str(DEST), "rows": len(frame), "prediction_std": float(frame.prediction.std()), "score": score}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

