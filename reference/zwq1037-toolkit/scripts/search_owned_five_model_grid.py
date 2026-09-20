"""Search five owned model families without any public prediction component."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['tabm', 'tree', 'gru', 'realmlp', 'transformer']
RUN = ROOT / 'data/interim/tree_experiments/EXP-BLEND-OWNED-20260917'


def compositions(total, count):
    if count == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for rest in compositions(total - first, count - 1):
                yield (first,) + rest


def load_existing():
    tabm = pd.read_feather(ROOT / 'data/interim/tree_experiments/EXP-TABM-032-MARKET385-K32/train059_valid6270_ex66/validation_predictions.feather')
    frame = tabm[['sample_id', 'month', 'target', 'candidate']].rename(columns={'candidate': 'tabm'})
    tree = pd.read_feather(ROOT / 'data/interim/tree_experiments/EXP-TREE-069-RELATIVE319/train059_valid6070/validation_predictions.feather')
    sources = {'tree': tree[['sample_id', 'xgb_relative319']].rename(columns={'xgb_relative319': 'tree'})}
    real_dir = ROOT / 'data/interim/tree_experiments/EXP-REALMLP-009-OUR379-CORR095/train059_valid6270_ex66'
    sources['realmlp'] = pd.DataFrame({'sample_id': np.load(real_dir / 'validation_sample_ids.npy'), 'realmlp': np.load(real_dir / 'validation_predictions.npy')})
    for name, path in {
        'gru': 'data/interim/kaggle_outputs/multistream_factorized_gru_timeaware_dev/factorized_gru/validation_predictions.csv',
        'transformer': 'outputs/kaggle_transformer_v7_result/factorized_transformer/validation_predictions.csv',
    }.items():
        sources[name] = pd.read_csv(ROOT / path, usecols=['sample_id', 'prediction']).rename(columns={'prediction': name})
    for name, source in sources.items():
        frame = frame.merge(source, on='sample_id', how='left', validate='one_to_one')
    frame = frame.loc[frame.month.between(62, 70) & frame.month.ne(66)].sort_values('sample_id').reset_index(drop=True)
    assert len(frame) == 140806
    assert frame.sample_id.is_unique
    assert np.isfinite(frame[NAMES + ['target']].to_numpy()).all()
    return frame


def score_terms(x, y, mask):
    a, b = x[mask], y[mask]
    return a.T @ a, a.T @ b, b.square().sum()


def scores(weights, terms):
    gram, cross, target_square = terms
    denominator = ((weights @ gram * weights).sum(dim=1) * target_square).clamp_min(1e-30).sqrt()
    return weights @ cross / denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path)
    parser.add_argument('--run-name', default='matched_train059')
    parser.add_argument('--selection-end', type=int, default=65)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = RUN / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    frame = pd.read_feather(args.input) if args.input else load_existing()
    months = frame.month.to_numpy()
    selection = months <= args.selection_end
    forward = months > args.selection_end
    assert selection.any() and forward.any()
    raw = torch.tensor(frame[NAMES].to_numpy(np.float64), dtype=torch.float64)
    target = torch.tensor(frame.target.to_numpy(np.float64), dtype=torch.float64)
    # 搜索期确定缩放，之后固定应用；保留预测均值以匹配原始cosine。
    # Fit RMS scales on selection months and preserve means for raw cosine.
    scale = raw[selection].square().mean(dim=0).sqrt().clamp_min(1e-12)
    x = raw / scale
    masks = {'selection': selection, 'forward': forward, 'all': np.ones(len(frame), bool)}
    masks.update({f'month_{int(m)}': months == m for m in np.unique(months)})
    terms = {name: score_terms(x, target, mask) for name, mask in masks.items()}
    coarse = torch.tensor(list(compositions(20, 5)), dtype=torch.float64) / 20
    selection_scores = scores(coarse, terms['selection'])
    top = torch.topk(selection_scores, min(3, len(coarse))).indices.tolist()
    fine = set()
    for index in top:
        center = (coarse[index] * 100).round().long().tolist()
        ranges = [range(max(0, v - 5), min(100, v + 5) + 1) for v in center[:4]]
        for first in itertools.product(*ranges):
            last = 100 - sum(first)
            if 0 <= last <= 100 and abs(last - center[4]) <= 5:
                fine.add(first + (last,))
    fine_weights = torch.tensor(sorted(fine), dtype=torch.float64) / 100
    weights = torch.cat([coarse, fine_weights], dim=0).unique(dim=0)
    selection_scores = scores(weights, terms['selection'])
    chosen_index = int(selection_scores.argmax())
    chosen = weights[chosen_index:chosen_index + 1]
    baseline = torch.tensor([[1/3, 0, 1/12, 1/12, 1/2]], dtype=torch.float64)
    equal = torch.full((1, 5), .2, dtype=torch.float64)
    metrics = {name: scores(weights, term).numpy() for name, term in terms.items()}
    grid = pd.DataFrame(weights.numpy(), columns=[f'weight_{n}' for n in NAMES])
    for name, values in metrics.items():
        grid[name] = values
    # 排序与选择只使用selection，forward结果只用来检查固定权重。
    # Never use forward scores to choose or refine weights.
    grid.sort_values('selection', ascending=False).to_csv(run / 'weight_grid.csv', index=False)
    reports = {}
    for name, vector in {'selected': chosen, 'prior_owned_ratio': baseline, 'equal': equal}.items():
        reports[name] = {'weights': dict(zip(NAMES, vector[0].tolist())), 'scores': {k: float(scores(vector, t)[0]) for k, t in terms.items()}}
    month_keys = [k for k in terms if k.startswith('month_')]
    deltas = {k: reports['selected']['scores'][k] - reports['prior_owned_ratio']['scores'][k] for k in terms}
    nearby = (weights - chosen).abs().max(dim=1).values <= .02 + 1e-12
    report = {
        'rows': len(frame), 'selection_months': sorted(map(int, np.unique(months[selection]))),
        'forward_months': sorted(map(int, np.unique(months[forward]))),
        'public_weight': 0, 'models': NAMES, 'candidate_count': len(weights),
        'normalization': 'RMS fitted on selection predictions; no mean subtraction',
        'selection_rms': dict(zip(NAMES, scale.tolist())),
        'recipes': reports, 'selected_delta_vs_prior_owned_ratio': deltas,
        'positive_months': sum(deltas[k] > 0 for k in month_keys), 'months': len(month_keys),
        'neighborhood_selection_min': float(selection_scores[nearby].min()),
        'individual_raw_cosine': {n: float(scores(torch.eye(5, dtype=torch.float64)[i:i+1], terms['all'])[0]) for i, n in enumerate(NAMES)},
        'forward_improved': deltas['forward'] > 0,
        'limitation': 'Historical validation months have been inspected before. Forward split is a diagnostic, not an untouched independent holdout. Default input uses train<=59 development models, not the deployed three-fold ensembles. Independent early-window verification remains required.',
        'submission_status': 'not_submitted',
    }
    frame['selected_blend'] = (x @ chosen[0]).numpy()
    frame.to_feather(run / 'aligned_validation_predictions.feather')
    (run / 'score_only.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['rows', 'candidate_count', 'recipes', 'selected_delta_vs_prior_owned_ratio', 'forward_improved']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
