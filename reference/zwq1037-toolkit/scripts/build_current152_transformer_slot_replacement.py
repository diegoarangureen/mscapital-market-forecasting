"""Replace a fixed fraction of the Transformer slot in the scored 0.152 soft gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_current151_lower_public_grid import PATHS, load, unit


ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ROOT / "data/interim/tree_experiments/EXP-BLEND-003-SHRINK-SOFTGATE/fusion_parameters.npz"
CURRENT = ROOT / "outputs/submissions/current152_owned_softgate25_transfer.csv"
GEOMETRY = ROOT / "outputs/submission_metadata/current151_theoretical_geometry_optimum.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")

    incumbent = pd.read_csv(CURRENT)
    ids = incumbent.sample_id.to_numpy()
    incumbent_prediction = incumbent.prediction.to_numpy(np.float64)
    old = load(PATHS["transformer_old"], ids)
    new = load(args.candidate, ids)
    old_z = unit(old)
    new_z = unit(new)

    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    weights = geometry["best_weights"]
    owned_names = ["tabm", "realmlp", "transformer_old", "gru"]
    owned_mass = float(sum(weights[name] for name in owned_names))
    state = np.load(PARAMETERS)
    anchors = state["anchors"]
    state_weights = state["state_weights"]
    rv = pd.read_feather(
        ROOT / "data/processed/test_market_microstructure_features.feather",
        columns=["sample_id", "realized_volatility_60"],
    )
    rv = pd.DataFrame({"sample_id": ids}).merge(rv, on="sample_id", how="left", validate="one_to_one")
    volatility = rv.realized_volatility_60.to_numpy(np.float64)
    median = float(state["rv_median"])
    volatility = np.nan_to_num(volatility, nan=median, posinf=median, neginf=median)
    sample_weights = np.column_stack([
        np.interp(volatility, anchors, state_weights[:, column])
        for column in range(4)
    ])
    if not np.allclose(sample_weights.sum(axis=1), 1):
        raise AssertionError("Owned soft-gate weights do not sum to one")

    transformer_weight = sample_weights[:, 2]
    delta = owned_mass * transformer_weight * args.fraction * (new_z - old_z)
    groups = np.arange(len(ids)) * 38 // len(ids)
    for group in range(38):
        mask = groups == group
        delta[mask] -= delta[mask].mean()
    prediction = incumbent_prediction + delta
    if len(ids) != 647_896 or not np.isfinite(prediction).all():
        raise AssertionError("Invalid replacement prediction")
    if max(abs(delta[groups == group].mean()) for group in range(38)) >= 1e-12:
        raise AssertionError("Group-common component changed")

    output = ROOT / f"outputs/submissions/current152_{args.label}_transformer_slot{int(args.fraction * 100):02d}.csv"
    pd.DataFrame({"sample_id": ids, "prediction": prediction}).to_csv(output, index=False)
    report = {
        "status": "prepared_not_submitted",
        "candidate": str(args.candidate),
        "output": str(output),
        "replacement_fraction": args.fraction,
        "owned_mass": owned_mass,
        "mean_effective_total_weight": float(np.mean(owned_mass * transformer_weight * args.fraction)),
        "correlation_with_scored_0p152": float(np.corrcoef(incumbent_prediction, prediction)[0, 1]),
        "delta_std": float(delta.std()),
        "rows": len(ids),
        "group_common_preserved": True,
        "formal_submission": False,
    }
    metadata = ROOT / f"outputs/submission_metadata/current152_{args.label}_transformer_slot_replacement.json"
    metadata.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
