"""Repeat EXP-TABM-001 with seed 137 as the only changed variable."""

from __future__ import annotations

import json
from pathlib import Path

import exp_tabm_001_exp053r_features as experiment


EXPERIMENT_ID = "EXP-TABM-008-SEED137"
SEED = 137


def main() -> None:
    experiment.EXPERIMENT_ID = EXPERIMENT_ID
    experiment.SEED = SEED
    experiment.main()

    project_dir = Path(__file__).resolve().parents[1]
    config_path = (
        project_dir
        / "data"
        / "interim"
        / "tree_experiments"
        / EXPERIMENT_ID
        / "config.json"
    )
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    metadata["main_change"] = "random seed 42 -> 137; all other settings unchanged"
    metadata["seed"] = SEED
    metadata["baseline_seed"] = 42
    config_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
