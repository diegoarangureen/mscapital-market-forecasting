"""Summarize the two purged windows and a fixed 25-percent replacement."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
folders = {
    'late': 'EXP-TABM-042-MARKET-PATH-SPECTRAL24/train059_valid6270_ex66',
    'early': 'EXP-TABM-043-PATH-SPECTRAL24-EARLY/train047_valid5059',
}


def cosine(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.dot(prediction, target) / (np.linalg.norm(prediction) * np.linalg.norm(target)))


results = {}
for name, folder in folders.items():
    frame = pd.read_feather(root / 'data/interim/tree_experiments' / folder / 'validation_predictions.feather')
    if name == 'late':
        frame = frame[frame['month'] != 66]
    target = frame['target'].to_numpy()
    baseline = frame['baseline'].to_numpy()
    candidate = frame['candidate'].to_numpy()
    baseline_score = cosine(baseline, target)
    candidate_score = cosine(candidate, target)
    # 冻结尺度与替换比例，避免在验证分数上搜索权重。
    # Freeze scale and replacement fraction instead of searching validation weights.
    blended = 0.75 * baseline / baseline.std() + 0.25 * candidate / candidate.std()
    results[name] = {
        'baseline': baseline_score,
        'candidate': candidate_score,
        'delta': candidate_score - baseline_score,
        'correlation': float(np.corrcoef(baseline, candidate)[0, 1]),
        'fixed25_delta': cosine(blended, target) - baseline_score,
        'monthly': {
            str(int(month)): {
                'baseline': cosine(group['baseline'], group['target']),
                'candidate': cosine(group['candidate'], group['target']),
            }
            for month, group in frame.groupby('month')
        },
    }
summary = {
    'windows': results,
    'mean_delta': float(np.mean([value['delta'] for value in results.values()])),
    'both_nonnegative': all(value['delta'] >= 0 for value in results.values()),
}
summary['passed'] = summary['both_nonnegative'] and summary['mean_delta'] >= 0.0007
destination = root / 'outputs/submission_metadata/tabm409_path_spectral_paired.json'
destination.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
