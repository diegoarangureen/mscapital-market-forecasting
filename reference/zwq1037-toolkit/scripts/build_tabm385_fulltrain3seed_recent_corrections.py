from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs/submissions"
META = PROJECT / "outputs/submission_metadata"

def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / x.std()

seed42 = pd.read_csv(OUT / "tabm_relative319_xs40_order20_market6_k32_seed42_fulltrain.csv")
ids = seed42.sample_id.to_numpy()
paths = [
    PROJECT / "outputs/predictions/tabm385_k32_fulltrain_additional_2seed/fold_end70_seed137.feather",
    PROJECT / "outputs/predictions/tabm385_k32_fulltrain_additional_2seed/fold_end70_seed2026.feather",
]
frames = [pd.read_feather(path).sort_values("sample_id") for path in paths]
for frame in frames:
    assert np.array_equal(ids, frame.sample_id.to_numpy())
full_members = [seed42.prediction.to_numpy(float)] + [frame.prediction.to_numpy(float) for frame in frames]
full3 = np.mean([unit(x) for x in full_members], axis=0)
full_path = OUT / "tabm385_k32_fulltrain_3seed.csv"
pd.DataFrame({"sample_id": ids, "prediction": full3}).to_csv(full_path, index=False)

temporal_frame = pd.read_csv(OUT / "tabm385_k32_temporal3fold_3seed.csv")
assert np.array_equal(ids, temporal_frame.sample_id.to_numpy())
temporal = unit(temporal_frame.prediction.to_numpy(float))
mixtures = {}
for full_weight in (0.10, 0.15, 0.20):
    prediction = (1.0 - full_weight) * temporal + full_weight * unit(full3)
    name = f"tabm385_k32_temporal3x3_full3seed_w{int(full_weight*100):02d}"
    path = OUT / f"{name}.csv"
    pd.DataFrame({"sample_id": ids, "prediction": prediction}).to_csv(path, index=False)
    mixtures[name] = {
        "temporal_weight": 1.0 - full_weight,
        "fulltrain_weight": full_weight,
        "path": str(path),
        "correlation_with_temporal": float(np.corrcoef(prediction, temporal)[0, 1]),
    }
correlations = np.corrcoef(np.column_stack([*map(unit, full_members), unit(full3), temporal]), rowvar=False)
report = {
    "run_name": "tabm385_k32_fulltrain3seed_recent_corrections",
    "fulltrain_seeds": [42, 137, 2026],
    "temporal_fold_ends": [52, 57, 62],
    "temporal_seeds": [42, 137, 2026],
    "known_public_lb": {"fulltrain_seed42": 0.134, "temporal3x3": 0.140},
    "fulltrain_3seed_path": str(full_path),
    "fulltrain_member_correlations": correlations[:3, :3].tolist(),
    "fulltrain3_vs_temporal3x3_correlation": float(correlations[3, 4]),
    "mixtures": mixtures,
    "submission_status": "prepared_not_submitted",
}
path = META / "tabm385_k32_fulltrain3seed_recent_corrections.json"
path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
