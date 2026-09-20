"""Wait for raw Transformer full train, download CSV, and build slot replacements."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = r"D:\anaconda\envs\pytorch\Scripts\kaggle.exe"
KERNEL = "zwq1037/multistream-factorized-transformer-multiwindow-dev"
VERSION = 26
OUT = ROOT / "data/interim/kaggle_results/transformer020_raw_full_v26"
PLAN = ROOT / "outputs/submission_metadata/goal_improve_beyond152_20260919.json"


def command(arguments: list[str], timeout: int = 240) -> str:
    result = subprocess.run([CLI, *arguments], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError("Kaggle operation failed: " + result.stdout + result.stderr)
    return result.stdout


def update_plan(status: str, **values) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    stage = next(item for item in plan["stages"] if item["id"] == "transformer020_raw_full")
    stage.update({"status": status, **values})
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    print(f"Continuation plan: {PLAN}", flush=True)
    time.sleep(20 * 60)
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
            raise RuntimeError("Transformer020 raw full v26 failed")
        if "KernelWorkerStatus.COMPLETE" in status:
            break
        time.sleep(10 * 60)
    else:
        update_plan("wait_timeout", kaggle_version=VERSION)
        raise RuntimeError("Transformer020 raw full v25 wait deadline reached")

    OUT.mkdir(parents=True, exist_ok=True)
    print(command([
        "kernels", "output", f"{KERNEL}/{VERSION}", "-p", str(OUT), "--force",
        "--file-pattern", r"score_only\.json$|result\.json$|standalone_factorized_transformer379_raw_full_e5\.csv$|factorized_transformer379_raw_full_e5\.pt$",
    ], timeout=900), flush=True)
    csv_files = list(OUT.rglob("standalone_factorized_transformer379_raw_full_e5.csv"))
    results = list(OUT.rglob("result.json"))
    checkpoints = list(OUT.rglob("factorized_transformer379_raw_full_e5.pt"))
    if len(csv_files) != 1 or len(results) != 1 or len(checkpoints) != 1:
        raise AssertionError(
            f"Expected one prediction, result, and checkpoint file, got "
            f"{len(csv_files)}, {len(results)}, and {len(checkpoints)}"
        )
    result = json.loads(results[0].read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("visible_gpu_count") != 2:
        raise AssertionError("Full training did not complete on T4 x2")

    built = {}
    for fraction in (0.25, 0.50):
        label = f"transformer020_raw{int(fraction * 100):02d}"
        process = subprocess.run([
            r"D:\anaconda\envs\pytorch\python.exe",
            str(ROOT / "scripts/build_current152_transformer_slot_replacement.py"),
            "--candidate", str(csv_files[0]), "--fraction", str(fraction), "--label", label,
        ], capture_output=True, text=True, timeout=900)
        print(process.stdout, flush=True)
        if process.returncode:
            raise RuntimeError("Slot replacement failed: " + process.stderr)
        metadata = ROOT / f"outputs/submission_metadata/current152_{label}_transformer_slot_replacement.json"
        built[str(fraction)] = json.loads(metadata.read_text(encoding="utf-8"))

    update_plan(
        "full_csv_complete",
        kaggle_version=VERSION,
        result_file=str(results[0]),
        standalone_csv=str(csv_files[0]),
        full_checkpoint=str(checkpoints[0]),
        blend_candidates=built,
        formal_submission=False,
    )
    print(json.dumps({"full_result": result, "blend_candidates": built}), flush=True)
    print(f"NEXT: read {PLAN}, compare v24, EXP022, and these CSVs; submit only the strongest authorized candidate.", flush=True)


if __name__ == "__main__":
    main()
