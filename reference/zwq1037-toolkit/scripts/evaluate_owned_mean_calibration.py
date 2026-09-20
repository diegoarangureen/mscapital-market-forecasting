"""Test the incumbent's owned mean calibration with train-only monthly fitting."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data/interim/kaggle_kernels/relative319_xs_tabm_notebook'))
import run_extracted as recipe


def cosine(y, p):
    y, p = torch.as_tensor(y, dtype=torch.float64), torch.as_tensor(p, dtype=torch.float64)
    return float(torch.dot(y, p) / (y.norm()*p.norm()).clamp_min(1e-30))


def main():
    torch.set_num_threads(1)
    directory = ROOT / 'data/interim/tree_experiments/EXP-BLEND-OWNED-20260917'
    labels = pd.read_feather(ROOT / 'data/raw/label.feather').sort_values('sample_id').reset_index(drop=True)
    cache = ROOT / 'data/interim/kaggle_relative319_dev'
    ids = np.load(cache / 'sample_ids.npy', mmap_mode='r')
    assert np.array_equal(labels.sample_id.to_numpy(), ids)
    features = np.load(cache / 'features.npy', mmap_mode='r')
    columns = json.loads((cache / 'feature_columns.json').read_text(encoding='utf-8'))
    selected = [columns.index(c) for c in recipe.TOP_FEATURES]
    months, target = labels.month.to_numpy(), labels.target.to_numpy(np.float64)
    summaries, means, keys = [], [], []
    for month in np.unique(months):
        rows = np.flatnonzero(months == month)
        values = np.asarray(features[rows][:, selected], np.float32)
        summaries.append(np.r_[np.nanmean(values, axis=0), np.nanstd(values, axis=0)])
        means.append(float(target[rows].mean()))
        keys.append(int(month))
    summaries = np.asarray(summaries, np.float64)
    means, keys = np.asarray(means), np.asarray(keys)
    fit = keys <= 59
    fill = np.nanmedian(summaries[fit], axis=0)
    summaries = np.where(np.isfinite(summaries), summaries, fill)
    scaler = StandardScaler().fit(summaries[fit])
    model = Ridge(alpha=100.).fit(scaler.transform(summaries[fit]), means[fit])
    predicted_mean = dict(zip(keys, model.predict(scaler.transform(summaries))))
    scale = float(target[months <= 59].std())
    meta = json.loads((ROOT / 'outputs/submission_metadata/owned_only_tabm33_gru20_realmlp28_transformer19.json').read_text(encoding='utf-8'))
    frame = pd.read_feather(directory / 'recovered_temporal/owned_temporal3fold_valid6570.feather')
    before = sum(weight*frame[name].to_numpy(np.float64)/meta['rms'][name] for name, weight in meta['weights'].items())
    centered, corrected = before.copy(), before.copy()
    for month in frame.month.unique():
        mask = frame.month.to_numpy() == month
        centered[mask] -= before[mask].mean()
        corrected[mask] = centered[mask] + predicted_mean[int(month)]/scale
    y = frame.target.to_numpy(np.float64)
    masks = {'selection_65_67': frame.month.to_numpy() <= 67,
             'forward_68_70': frame.month.to_numpy() >= 68,
             'all': np.ones(len(frame), bool)}
    masks.update({f'month_{int(m)}': frame.month.to_numpy() == m for m in frame.month.unique()})
    result = {name: {'raw': cosine(y[mask], before[mask]),
                     'centered': cosine(y[mask], centered[mask]),
                     'mean_corrected': cosine(y[mask], corrected[mask]),
                     'correction_delta': cosine(y[mask], corrected[mask])-cosine(y[mask], before[mask])}
              for name, mask in masks.items()}
    primary = [result[k]['correction_delta'] for k in ['selection_65_67', 'forward_68_70']]
    report = {'experiment': 'OWNED-MEAN-CALIBRATION', 'model': 'Ridge alpha100 fixed from previous incumbent',
              'fit_months': '0-59', 'purge_months': '60-61', 'validation_months': '65-70 ex66',
              'public_weight': 0, 'scores': result,
              'forecast_mean': {str(m): predicted_mean[int(m)] for m in frame.month.unique()},
              'actual_mean_diagnostic': {str(m): float(frame.loc[frame.month.eq(m), 'target'].mean()) for m in frame.month.unique()},
              'passed': bool(min(primary) > 0 and np.mean(primary) >= .0007),
              'limitation': 'Existing late-window diagnostic only; no new early confirmation or public submission.'}
    out = ROOT / 'outputs/submission_metadata/owned_mean_calibration_20260917.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
