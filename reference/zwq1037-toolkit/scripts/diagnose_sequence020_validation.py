"""Compare saved predictions with the strict old Transformer, without fitting."""
import json
import os
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'data/interim/tree_experiments/EXP-SEQUENCE-020-SUBSECOND-GRU-TRANSFORMER'


def cosine(x, y):
    x, y = torch.as_tensor(np.asarray(x, dtype=np.float64)), torch.as_tensor(np.asarray(y, dtype=np.float64))
    return float(torch.dot(x, y) / (x.norm() * y.norm()).clamp_min(1e-20))


def main():
    torch.set_num_threads(1)
    before = (RUN / 'score_only.json').read_bytes()
    state = json.loads(before)
    candidate = pd.read_feather(RUN / 'validation_predictions.feather')
    baseline_path = ROOT / 'outputs/kaggle_transformer_v7_result/factorized_transformer/validation_predictions.csv'
    old = pd.read_csv(baseline_path, usecols=['sample_id', 'prediction']).rename(columns={'prediction': 'old'})
    frame = candidate.merge(old, on='sample_id', how='left', validate='one_to_one')
    if before != (RUN / 'score_only.json').read_bytes():
        raise RuntimeError('Score changed during diagnosis; rerun for a consistent snapshot')
    assert len(frame) == 140806 and not frame['old'].isna().any()
    assert np.isfinite(frame[['target', 'prediction', 'old']].to_numpy()).all()
    assert sorted(frame['month'].unique().tolist()) == [62, 63, 64, 65, 67, 68, 69, 70]
    candidate_cosine = cosine(frame.prediction, frame.target)
    assert abs(candidate_cosine - state['best_cosine']) < 1e-9
    report = {'epochs_finished_at_snapshot': state['epochs_finished'],
              'candidate_best_cosine': candidate_cosine,
              'old_cosine': cosine(frame.old, frame.target), 'rows': len(frame),
              'prediction_correlation': cosine(frame.prediction - frame.prediction.mean(), frame.old - frame.old.mean()),
              'candidate_prediction_std': float(frame.prediction.std()),
              'old_prediction_std': float(frame.old.std()),
              'monthly': {}, 'diagnostic_only': True, 'weights_not_searched': True}
    for month, group in frame.groupby('month'):
        new, reference = cosine(group.prediction, group.target), cosine(group.old, group.target)
        report['monthly'][str(month)] = {'candidate': new, 'old': reference, 'delta': new-reference}
    path = ROOT / f"outputs/submission_metadata/subsecond_diagnostic_epoch{state['epochs_finished']}_20260917.json"
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
