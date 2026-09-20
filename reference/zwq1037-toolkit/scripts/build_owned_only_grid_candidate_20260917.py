"""Generate the frozen owned-only blend; never submit to Kaggle."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
BLEND = ROOT / 'data/interim/tree_experiments/EXP-BLEND-OWNED-20260917'
RECOVERED = BLEND / 'recovered_temporal'
NAME = 'owned_only_tabm33_gru20_realmlp28_transformer19'
WEIGHTS = {'tabm': .33, 'gru': .20, 'realmlp': .28, 'transformer': .19}
PATHS = {
    'tabm': 'outputs/submissions/tabm385_k32_temporal3fold_3seed.csv',
    'realmlp': 'outputs/submissions/realmlp379_rq16_corr095_e8_fulltrain.csv',
    'transformer': 'outputs/submissions/standalone_factorized_transformer379.csv',
}


def cosine(y, p):
    y, p = torch.as_tensor(y, dtype=torch.float64), torch.as_tensor(p, dtype=torch.float64)
    return float(torch.dot(y, p) / (y.norm() * p.norm()).clamp_min(1e-30))


def main():
    torch.set_num_threads(2)
    assert abs(sum(WEIGHTS.values()) - 1) < 1e-12
    validation = pd.read_feather(RECOVERED / 'owned_temporal3fold_valid6570.feather')
    fit_mask = validation.month.to_numpy() <= 67
    stats = json.loads((BLEND / 'temporal3fold/score_only.json').read_text(encoding='utf-8'))
    rms = stats['selection_rms']
    normalized, sources, ids = {}, {}, None

    def read_test(relative):
        nonlocal ids
        frame = pd.read_csv(ROOT / relative)
        assert frame.columns.tolist() == ['sample_id', 'prediction']
        assert len(frame) == 647896 and frame.sample_id.is_unique
        current = frame.sample_id.to_numpy()
        if ids is None:
            ids = current
        else:
            assert np.array_equal(current, ids)
        values = frame.prediction.to_numpy(np.float64)
        assert np.isfinite(values).all()
        return values

    for name, relative in PATHS.items():
        normalized[name] = read_test(relative) / rms[name]
        sources[name] = str(ROOT / relative)
    gru_parts, gru_norm = [], {}
    for fold in (52, 57, 62):
        relative = f'outputs/submissions/standalone_timeaware_three_stream_gru379_fold_end{fold}.csv'
        raw_test = read_test(relative)
        valid = pd.read_feather(RECOVERED / f'gru_fold{fold}.feather')
        valid = valid.set_index('sample_id').loc[validation.sample_id]
        raw_valid = valid.prediction.to_numpy(np.float64)
        mean, std = float(raw_valid[fit_mask].mean()), float(raw_valid[fit_mask].std())
        assert std > 1e-12
        gru_parts.append((raw_test - mean) / std)
        gru_norm[str(fold)] = {'mean': mean, 'std': std, 'source': str(ROOT / relative)}
    normalized['gru'] = np.mean(gru_parts, axis=0) / rms['gru']
    sources['gru'] = gru_norm
    prediction = sum(WEIGHTS[name] * normalized[name] for name in WEIGHTS)
    assert np.isfinite(prediction).all()
    path = ROOT / 'outputs/submissions' / (NAME + '.csv')
    pd.DataFrame({'sample_id': ids, 'prediction': prediction}).to_csv(path, index=False)
    check = pd.read_csv(path)
    assert np.array_equal(check.sample_id.to_numpy(), ids)
    assert np.isfinite(check.prediction.to_numpy()).all()

    val = sum(WEIGHTS[name] * validation[name].to_numpy(np.float64) / rms[name] for name in WEIGHTS)
    prior = {'tabm': 1/3, 'gru': 1/12, 'realmlp': 1/12, 'transformer': .5}
    old = sum(prior[name] * validation[name].to_numpy(np.float64) / rms[name] for name in prior)
    target = validation.target.to_numpy(np.float64)
    masks = {'selection_65_67': fit_mask, 'forward_68_70': ~fit_mask,
             'all_65_70_ex66': np.ones(len(validation), bool)}
    local = {name: {'candidate': cosine(target[mask], val[mask]),
                    'prior_owned_ratio': cosine(target[mask], old[mask]),
                    'delta': cosine(target[mask], val[mask])-cosine(target[mask], old[mask])}
             for name, mask in masks.items()}
    report = {'run_name': NAME, 'weights': WEIGHTS, 'tree_weight': 0, 'public_weight': 0,
              'sources': sources, 'normalization': 'Frozen validation-selection RMS for model families; GRU per-fold means/stds also frozen from months65/67. No additional common component.',
              'rms': {name: rms[name] for name in WEIGHTS},
              'weight_selection': 'Weights searched on temporal2fold predictions months62-65; checked unchanged on months67-70 and on the latest three-fold version.',
              'latest_threefold_local_scores': local,
              'early_confirmation': 'Skipped by explicit user instruction on 2026-09-17; local early training stopped and no early notebook uploaded.',
              'validation_deployment_note': 'TabM and GRU validation use the saved deployment checkpoints; RealMLP and Transformer use train<=59 development predictions for weight selection and their existing fulltrain test predictions for deployment.',
              'rows': len(check), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'prediction_mean': float(prediction.mean()), 'prediction_std': float(prediction.std()),
              'output_path': str(path), 'submission_status': 'prepared_not_submitted'}
    metadata = ROOT / 'outputs/submission_metadata' / (NAME + '.json')
    metadata.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'csv': str(path), 'weights': WEIGHTS, 'rows': len(check), 'local_scores': local,
                      'prediction_mean': report['prediction_mean'], 'prediction_std': report['prediction_std'],
                      'submission_status': report['submission_status']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
