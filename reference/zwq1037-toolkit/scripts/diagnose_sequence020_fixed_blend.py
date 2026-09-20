"""Fixed, predeclared 25% diagnostic; no weight search and no submission."""
import json
import os
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
NEW_PATH = ROOT / 'data/interim/tree_experiments/EXP-SEQUENCE-020-SUBSECOND-GRU-TRANSFORMER/validation_predictions.feather'
OLD_PATH = ROOT / 'outputs/kaggle_transformer_v7_result/factorized_transformer/validation_predictions.csv'


def cosine(prediction, target):
    prediction = torch.as_tensor(np.asarray(prediction, dtype=np.float64))
    target = torch.as_tensor(np.asarray(target, dtype=np.float64))
    return float(torch.dot(prediction, target) / (prediction.norm() * target.norm()).clamp_min(1e-20))


def main():
    torch.set_num_threads(1)
    new = pd.read_feather(NEW_PATH)
    old = pd.read_csv(OLD_PATH, usecols=['sample_id', 'prediction']).rename(columns={'prediction': 'old'})
    frame = new.merge(old, on='sample_id', validate='one_to_one')
    selection = frame.month.isin([62, 63, 64, 65]).to_numpy()
    forward = frame.month.isin([67, 68, 69, 70]).to_numpy()
    old_rms = float(np.sqrt(np.mean(np.square(frame.old.to_numpy()[selection], dtype=np.float64))))
    new_rms = float(np.sqrt(np.mean(np.square(frame.prediction.to_numpy()[selection], dtype=np.float64))))
    frame['mixed'] = 0.75 * frame.old / old_rms + 0.25 * frame.prediction / new_rms
    report = {'new_weight': 0.25, 'old_weight': 0.75,
              'ratio_predeclared': True, 'weights_searched': False,
              'normalization': 'RMS fit only on months62-65',
              'old_rms': old_rms, 'new_rms': new_rms,
              'candidate_epoch': 3, 'candidate_is_saved_best': True,
              'windows': {}, 'monthly': {}, 'submission_created': False}
    for name, mask in [('selection62_65', selection), ('forward67_70', forward), ('all62_70_ex66', np.ones(len(frame), dtype=bool))]:
        subset = frame.loc[mask]
        reference = cosine(subset.old, subset.target)
        mixed = cosine(subset.mixed, subset.target)
        report['windows'][name] = {'old': reference, 'mixed': mixed, 'delta': mixed-reference}
    for month, subset in frame.groupby('month'):
        reference = cosine(subset.old, subset.target)
        mixed = cosine(subset.mixed, subset.target)
        report['monthly'][str(month)] = {'old': reference, 'mixed': mixed, 'delta': mixed-reference}
    output = ROOT / 'outputs/submission_metadata/subsecond_epoch3_fixed25_blend_20260917.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
