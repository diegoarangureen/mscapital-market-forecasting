"""Paired temporal evaluation; preserve separate metrics, never a total score."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# 在导入数值库前限制线程；评分不需要训练或 GPU。
# Limit threads before importing numerical libraries; evaluation needs no training.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import pyarrow as pa

pa.set_cpu_count(2)
pa.set_io_thread_count(2)

PANELS = {
    "cosine_50_59": list(range(50, 60)),
    "cosine_60_70_no66": [month for month in range(60, 71) if month != 66],
    "cosine_62_70_no66": [month for month in range(62, 71) if month != 66],
    "cosine_67_70": list(range(67, 71)),
    "cosine_60_70": list(range(60, 71)),
    "cosine_62_70": list(range(62, 71)),
}
PRIMARY = ("cosine_50_59", "cosine_60_70_no66", "cosine_62_70_no66", "cosine_67_70")
EPSILON = 1e-12


def cosine(target, prediction):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.ndim != 1 or target.shape != prediction.shape or not len(target):
        raise ValueError("Cosine requires matching nonempty 1D arrays.")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite target or prediction.")
    denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
    return float(np.dot(target, prediction) / denominator) if denominator else 0.0


def validate_frame(frame):
    required = ["sample_id", "month", "target", "prediction"]
    if not set(required).issubset(frame.columns) or frame.empty:
        raise ValueError("Missing prediction schema or empty frame.")
    if frame[required].isna().any().any() or frame.sample_id.duplicated().any():
        raise ValueError("Missing values or duplicate sample IDs.")
    if not np.isfinite(frame[["month", "target", "prediction"]].to_numpy()).all():
        raise ValueError("Nonfinite values.")
    if (frame.month != frame.month.astype(int)).any():
        raise ValueError("Month must be integral.")
    return frame[required].sort_values("sample_id").reset_index(drop=True)


def align_pair(baseline, candidate):
    baseline = validate_frame(baseline)
    candidate = validate_frame(candidate)
    for column in ("sample_id", "month", "target"):
        if not np.array_equal(baseline[column].to_numpy(), candidate[column].to_numpy()):
            raise ValueError(f"Baseline/candidate {column} mismatch; refusing an inner join.")
    return baseline, candidate


def evaluate_pair(baseline, candidate):
    baseline, candidate = align_pair(baseline, candidate)
    summary, monthly, leave_one = {}, [], []
    observed = set(baseline.month.unique())
    for panel, expected in PANELS.items():
        # 缺少月份必须显示不完整，不能把部分窗口冒充完整窗口。
        # Missing months invalidate the panel rather than silently shortening it.
        if not set(expected).issubset(observed):
            summary[panel] = None
            summary[panel + "_delta"] = None
            summary[panel + "_missing_months"] = sorted(set(expected) - observed)
            continue
        mask = baseline.month.isin(expected).to_numpy()
        base = baseline.loc[mask]
        cand = candidate.loc[mask]
        base_score = cosine(base.target, base.prediction)
        score = cosine(cand.target, cand.prediction)
        summary[panel] = score
        summary[panel + "_baseline"] = base_score
        summary[panel + "_delta"] = score - base_score
        summary[panel + "_rows"] = len(cand)
        base_monthly, candidate_monthly, loo_deltas = [], [], []
        for month in expected:
            b = base.loc[base.month == month]
            c = cand.loc[cand.month == month]
            bs, cs = cosine(b.target, b.prediction), cosine(c.target, c.prediction)
            base_monthly.append(bs)
            candidate_monthly.append(cs)
            monthly.append(dict(panel=panel, month=month, rows=len(c), baseline=bs, candidate=cs, delta=cs-bs))
            b = base.loc[base.month != month]
            c = cand.loc[cand.month != month]
            bs, cs = cosine(b.target, b.prediction), cosine(c.target, c.prediction)
            loo_deltas.append(cs-bs)
            leave_one.append(dict(panel=panel, excluded_month=month, rows=len(c), baseline=bs, candidate=cs, delta=cs-bs))
        for suffix, function in (
            ("monthly_std", lambda values: np.std(values, ddof=0)),
            ("monthly_q25", lambda values: np.quantile(values, .25)),
            ("monthly_worst", np.min),
        ):
            value = float(function(candidate_monthly))
            base_value = float(function(base_monthly))
            summary[panel + "_" + suffix] = value
            summary[panel + "_" + suffix + "_baseline"] = base_value
            summary[panel + "_" + suffix + "_delta"] = value - base_value
        deltas = np.asarray(candidate_monthly) - base_monthly
        summary[panel + "_improved_months"] = int(np.sum(deltas > EPSILON))
        summary[panel + "_month_count"] = len(expected)
        summary[panel + "_worst_month"] = int(expected[int(np.argmin(candidate_monthly))])
        summary[panel + "_loo_min_delta"] = float(min(loo_deltas))
        summary[panel + "_loo_positive_count"] = int(np.sum(np.asarray(loo_deltas) > EPSILON))
    return summary, monthly, leave_one


def apply_gates(summary, policy):
    complete = all(summary.get(metric) is not None for metric in PRIMARY)
    if not complete:
        return dict(complete=False, passes_gates=False, decision="incomplete")
    deltas = [summary[metric + "_delta"] for metric in PRIMARY]
    no_regression = all(delta >= -EPSILON for delta in deltas)
    meaningful_gain = any(delta >= policy["minimum_gain_one_primary"] for delta in deltas)
    loo_protected = all(summary[metric + "_loo_min_delta"] >= -policy["maximum_loo_regression"] for metric in PRIMARY)
    passed = no_regression and meaningful_gain and loo_protected
    return dict(complete=True, gate_primary_nonregression=no_regression,
                gate_meaningful_gain=meaningful_gain, gate_leave_one=loo_protected,
                passes_gates=passed, decision="eligible_for_confirmation" if passed else "fails_gates")


def pareto_labels(rows):
    # 只在同一对照组内比较四个主分数；不把随机种子不同的结果混排。
    # Compare four primary scores only within the same paired comparison group.
    for row in rows:
        if not row["complete"]:
            row["pareto_nondominated"] = None
            row["dominated_by"] = []
            continue
        dominated = []
        for other in rows:
            if other is row or other["group"] != row["group"] or not other["complete"]:
                continue
            differences = [other[key] - row[key] for key in PRIMARY]
            if all(value >= -EPSILON for value in differences) and any(value > EPSILON for value in differences):
                dominated.append(other["name"])
        row["dominated_by"] = dominated
        row["pareto_nondominated"] = not dominated
        row["eligible_and_nondominated"] = row["passes_gates"] and not dominated


def load_source(root, specification):
    frames, provenance = [], []
    for part in specification:
        path = (root / part["path"]).resolve()
        column = part["column"]
        frame = pd.read_feather(path, columns=["sample_id", "month", "target", column])
        frame = frame.rename(columns={column: "prediction"})
        expected = set(range(part["valid_start"], part["valid_end"] + 1))
        if set(frame.month.unique()) != expected or part["train_end"] >= part["valid_start"]:
            raise ValueError(f"Invalid declared forward fold: {path}")
        frames.append(frame)
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        provenance.append(dict(**part, absolute_path=str(path), sha256=digest, rows=len(frame)))
    return validate_frame(pd.concat(frames, ignore_index=True)), provenance


def run_manifest(manifest_path, output_dir):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = (manifest_path.parent / manifest["project_root"]).resolve()
    rows, monthly_rows, loo_rows, provenance = [], [], [], {}
    for group in manifest["groups"]:
        baseline, base_sources = load_source(root, group["baseline"])
        specifications = [{"name": group["baseline_name"], "sources": group["baseline"]}, *group["candidates"]]
        for specification in specifications:
            candidate, sources = load_source(root, specification["sources"])
            summary, monthly, leave_one = evaluate_pair(baseline, candidate)
            identity = dict(group=group["name"], name=specification["name"])
            gates = apply_gates(summary, manifest["policy"])
            rows.append(dict(**identity, **summary, **gates))
            monthly_rows.extend(dict(**identity, **item) for item in monthly)
            loo_rows.extend(dict(**identity, **item) for item in leave_one)
            provenance[group["name"] + "/" + specification["name"]] = dict(baseline=base_sources, candidate=sources)
    pareto_labels(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(output_dir / "monthly.csv", index=False)
    pd.DataFrame(loo_rows).to_csv(output_dir / "leave_one_month_out.csv", index=False)
    payload = dict(policy=manifest["policy"], primary=list(PRIMARY), rows=rows, provenance=provenance,
                   limitation="Historical retrospective audit; declared fold boundaries do not prove training provenance. No test-likeness selection or composite score.")
    (output_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(pd.DataFrame(rows)[["group", "name", *PRIMARY, "passes_gates", "pareto_nondominated"]].to_string(index=False))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run_manifest(arguments.manifest.resolve(), arguments.output.resolve())

