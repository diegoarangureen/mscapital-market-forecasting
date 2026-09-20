"""Prepare the monthwise-guarded four-source fulltrain submission."""

from __future__ import annotations

import json
from pathlib import Path

import prepare_four_source_tabm_tree_submission as prepare


OUTPUT_NAME = "tabm_four_source_guarded_50_25_05_xgb20_fulltrain"
WEIGHTS = {
    "tabm_original_centered": 0.50,
    "tabm_corrprune_raw": 0.25,
    "tabm_cosine_raw": 0.05,
    "xgboost_centered": 0.20,
}


def main() -> None:
    prepare.OUTPUT_NAME = OUTPUT_NAME
    prepare.WEIGHTS = WEIGHTS
    prepare.main()

    project_dir = Path(__file__).resolve().parents[1]
    metadata_path = (
        project_dir / "outputs" / "submission_metadata" / f"{OUTPUT_NAME}.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validation = json.loads(
        (
            project_dir
            / "data"
            / "interim"
            / "tree_experiments"
            / "EXP-BLEND-010"
            / "config.json"
        ).read_text(encoding="utf-8")
    )
    metadata["local_validation"] = {
        "overall": validation["overall_cosine"],
        "no66": validation["cosine_62_70_without_66"],
        "recent": validation["cosine_67_70"],
        "early": validation["cosine_60_64"],
        "primary_std": validation["primary_monthly_std"],
        "primary_worst": validation["primary_monthly_worst"],
        "primary_q25": validation["primary_monthly_q25"],
        "primary_lomo_min": validation["primary_leave_one_month_out_min_change"],
        "primary_lomo_mean": validation["primary_leave_one_month_out_mean_change"],
        "passes_core_gates": validation["passes_core_generalization_gates"],
        "passes_strict_gates": validation["passes_strict_stability_gates"],
    }
    metadata["selection"] = {
        "experiment_id": "EXP-BLEND-010",
        "rule": (
            "every month 62-70 except 66 must be non-decreasing versus the "
            "centered 80/20 baseline"
        ),
        "minimum_primary_month_gain": 0.00017719312275801813,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
