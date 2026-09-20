"""Replace only the matching owned slots in the scored 0.151 incumbent."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_factorized_transformer_lb_candidates import build_common_component

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "outputs/submissions/final_gru_direct_t12_r03_x18_g03.csv"
OLD = {
    "transformer": PROJECT / "outputs/submissions/standalone_factorized_transformer379.csv",
    "realmlp": PROJECT / "outputs/submissions/realmlp379_rq16_temporal3fold_e8_fold_end57.csv",
}
WEIGHTS = {"transformer": 0.18, "realmlp": 0.03}


def zunit(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / x.std()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer", type=Path)
    parser.add_argument("--realmlp", type=Path)
    args = parser.parse_args()
    inputs = {name: path for name, path in vars(args).items() if path is not None}
    if not inputs:
        raise ValueError("At least one validated production prediction is required.")
    _, template, groups, _ = build_common_component()
    ids = template.sample_id.to_numpy()

    def load(path):
        frame = pd.read_csv(path)
        if list(frame.columns) != ["sample_id", "prediction"] or len(frame) != 647896:
            raise AssertionError(f"Invalid submission columns/rows: {path}")
        if not frame.sample_id.is_unique or not np.array_equal(frame.sample_id.to_numpy(), ids):
            raise AssertionError(f"Invalid IDs: {path}")
        prediction = frame.prediction.to_numpy(dtype=np.float64)
        if not np.isfinite(prediction).all():
            raise AssertionError(f"Nonfinite prediction: {path}")
        return prediction

    base = load(BASE)
    deltas, correlations = {}, {}
    for name, path in inputs.items():
        previous = load(OLD[name])
        current = load(path)
        correlations[name] = float(np.corrcoef(previous, current)[0, 1])
        deltas[name] = WEIGHTS[name] * (zunit(current) - zunit(previous))
    outputs = []
    names = list(inputs)
    for fractions in itertools.product((0.0, 0.5, 1.0), repeat=len(names)):
        if not any(fractions):
            continue
        delta = sum(fraction * deltas[name] for name, fraction in zip(names, fractions))
        for group in np.unique(groups):
            mask = groups == group
            delta[mask] -= delta[mask].mean()
        prediction = base + delta
        label = "_".join(f"{name}{int(fraction * 100)}" for name, fraction in zip(names, fractions) if fraction > 0)
        output = PROJECT / "outputs/submissions" / f"current151_structure_{label}.csv"
        pd.DataFrame({"sample_id": ids, "prediction": prediction}).to_csv(output, index=False)
        outputs.append({"file": str(output), "replacement_fractions": dict(zip(names, fractions)), "incumbent_correlation": float(np.corrcoef(base, prediction)[0, 1]), "delta_std": float(delta.std()), "rows": len(ids), "finite": bool(np.isfinite(prediction).all())})
    report = {"base": str(BASE), "base_submission_ref": 56270670, "base_lb": 0.151, "slot_weights": WEIGHTS, "new_inputs": {name: str(path) for name, path in inputs.items()}, "old_new_correlations": correlations, "outputs": outputs, "submission_status": "prepared_not_submitted"}
    path = PROJECT / "outputs/submission_metadata/current151_structure_replacements.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
