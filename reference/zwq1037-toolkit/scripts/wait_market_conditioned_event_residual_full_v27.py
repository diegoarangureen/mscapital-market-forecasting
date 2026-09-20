"""Wait for v27 full event residual, download compact artifacts, and build slot replacements."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = r"D:\anaconda\envs\pytorch\Scripts\kaggle.exe"
KERNEL = "zwq1037/multistream-factorized-transformer-multiwindow-dev"
VERSION = 27
OUT = ROOT / "data/interim/kaggle_results/market_conditioned_event_residual_full_v27"
PLAN = ROOT / "outputs/submission_metadata/goal_improve_beyond152_20260919.json"


def command(arguments: list[str], timeout: int = 240) -> str:
    result = subprocess.run(
        [CLI, *arguments], capture_output=True, text=True, timeout=timeout
    )
    if result.returncode:
        raise RuntimeError("Kaggle operation failed: " + result.stdout + result.stderr)
    return result.stdout


def update_plan(status: str, **values) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    stage = next(
        item for item in plan["stages"]
        if item["id"] == "market_conditioned_event_residual"
    )
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
            print(
                f"status unavailable: {type(exc).__name__}: {exc}", flush=True
            )
            time.sleep(10 * 60)
            continue
        print(status.strip(), flush=True)
        if (
            "KernelWorkerStatus.ERROR" in status
            or "KernelWorkerStatus.CANCEL" in status
        ):
            update_plan("full_remote_failed", kaggle_version=VERSION)
            raise RuntimeError("Market-conditioned event residual full v27 failed")
        if "KernelWorkerStatus.COMPLETE" in status:
            break
        time.sleep(10 * 60)
    else:
        update_plan("full_wait_timeout", kaggle_version=VERSION)
        raise RuntimeError("Event residual full v27 wait deadline reached")

    OUT.mkdir(parents=True, exist_ok=True)
    print(
        command(
            [
                "kernels",
                "output",
                f"{KERNEL}/{VERSION}",
                "-p",
                str(OUT),
                "--force",
                "--file-pattern",
                r"score_only\.json$|result\.json$|standalone_market_conditioned_event_residual_full\.csv$|market_conditioned_event_residual_full\.pt$",
            ],
            timeout=900,
        ),
        flush=True,
    )
    csv_files = list(
        OUT.rglob("standalone_market_conditioned_event_residual_full.csv")
    )
    results = list(OUT.rglob("result.json"))
    checkpoints = list(OUT.rglob("market_conditioned_event_residual_full.pt"))
    if len(csv_files) != 1 or len(results) != 1 or len(checkpoints) != 1:
        raise AssertionError(
            f"Expected one CSV, result, and checkpoint; got "
            f"{len(csv_files)}, {len(results)}, {len(checkpoints)}"
        )
    result = json.loads(results[0].read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("visible_gpu_count") != 2:
        raise AssertionError("Full event residual did not complete on T4 x2")
    if result.get("full_base_reproduction_cosine", 0.0) < 0.9999999:
        raise AssertionError("Full base reproduction failed")

    built = {}
    for fraction in (0.50, 1.00):
        label = f"event_residual_full{int(fraction * 100):03d}"
        process = subprocess.run(
            [
                r"D:\anaconda\envs\pytorch\python.exe",
                str(
                    ROOT
                    / "scripts/build_current152_transformer_slot_replacement.py"
                ),
                "--candidate",
                str(csv_files[0]),
                "--fraction",
                str(fraction),
                "--label",
                label,
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        print(process.stdout, flush=True)
        if process.returncode:
            raise RuntimeError(
                "Event residual slot replacement failed: " + process.stderr
            )
        metadata = (
            ROOT
            / f"outputs/submission_metadata/current152_{label}_transformer_slot_replacement.json"
        )
        built[str(fraction)] = json.loads(metadata.read_text(encoding="utf-8"))

    raw_root = ROOT / "data/interim/kaggle_results/transformer020_raw_full_v26"
    raw_csv_files = list(raw_root.rglob("standalone_factorized_transformer379_raw_full_e5.csv"))
    if len(raw_csv_files) != 1:
        raise AssertionError(
            f"Expected one downloaded v26 raw CSV, got {len(raw_csv_files)}"
        )
    joint_process = subprocess.run(
        [
            r"D:\anaconda\envs\pytorch\python.exe",
            str(ROOT / "scripts/build_current152_joint_transformer_slot.py"),
            "--raw-candidate",
            str(raw_csv_files[0]),
            "--event-candidate",
            str(csv_files[0]),
            "--raw-weight",
            "0.4",
            "--event-weight",
            "0.6",
            "--label",
            "transformer020raw40_event60",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    print(joint_process.stdout, flush=True)
    if joint_process.returncode:
        raise RuntimeError(
            "Joint raw-event slot replacement failed: " + joint_process.stderr
        )
    joint_metadata = (
        ROOT
        / "outputs/submission_metadata/current152_transformer020raw40_event60.json"
    )
    joint_candidate = json.loads(joint_metadata.read_text(encoding="utf-8"))

    update_plan(
        "full_csv_complete",
        kaggle_version=VERSION,
        result_file=str(results[0]),
        standalone_csv=str(csv_files[0]),
        full_checkpoint=str(checkpoints[0]),
        blend_candidates=built,
        joint_candidate=joint_candidate,
        formal_submission=False,
    )
    print(
        json.dumps({"full_result": result, "blend_candidates": built, "joint_candidate": joint_candidate}), flush=True
    )
    print(
        f"NEXT: read {PLAN}, compare v26 and v27 candidate geometry, then spend "
        "at most one authorized formal submission on the strongest candidate.",
        flush=True,
    )


if __name__ == "__main__":
    main()

