"""Evaluate one prespecified recency ensemble; export only if all gates pass."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit.io import sha256, write_json
from audit.metrics import paired_panel, postprocess
from review_champion import labels, recovered_run, assert_seed_extension, export_final


def combine(records, seeds, half_life, split='score'):
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError('Invalid fixed half-life')
    expected = set(itertools.product(range(5), seeds))
    keys = [(j.fold, j.seed) for j, _ in records]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Incomplete or duplicate fold/seed coverage')
    if len({j.origin for j, _ in records}) != 1:
        raise ValueError('Cannot combine different temporal origins')
    ends = {j.fold: j.train_end for j, _ in records}
    if any(j.train_end != ends[j.fold] for j, _ in records):
        raise ValueError('Inconsistent train cutoffs')
    unnormalized = {f: 2 ** ((t - max(ends.values())) / half_life) for f, t in ends.items()}
    weights = {f: w / sum(unnormalized.values()) for f, w in unnormalized.items()}
    rows = records[0][1][split + '_rows']
    base = np.zeros(len(rows), dtype=np.float64)
    candidate = base.copy()
    for job, pred in records:
        if not np.array_equal(rows, pred[split + '_rows']):
            raise ValueError('Misaligned prediction rows')
        values = np.asarray(pred[split + '_pred'], dtype=np.float64)
        if values.shape != rows.shape or not np.isfinite(values).all():
            raise ValueError('Invalid model predictions')
        base += values
        candidate += values * weights[job.fold] / len(seeds)
    return rows, base / len(records), candidate, weights


def evaluate(args):
    cfg = json.loads(Path(args.config).read_text())
    if cfg['parameter_search'] or cfg['clip'] or not cfg['zero_nodata']:
        raise ValueError('This experiment must keep the prespecified baseline handling')
    dm, data = labels(args.labels)
    metas, records = [], []
    for root in args.confirm:
        meta, rec = recovered_run(root, dm, data)
        if meta['config']['mode'] != 'confirm':
            raise ValueError('Require checkpoint-external confirmation predictions')
        if metas:
            assert_seed_extension(metas[0], meta)
        metas.append(meta); records.extend(rec)
    if sorted({j.seed for j, _ in records}) != cfg['seeds'] or sorted({j.origin for j, _ in records}) != cfg['origins']:
        raise ValueError('Seed/origin set differs from frozen plan')
    n = len(data['y'])
    global_base, global_candidate = np.full(n, np.nan), np.full(n, np.nan)
    by_seed = {s: (np.full(n, np.nan), np.full(n, np.nan)) for s in cfg['seeds']}
    by_origin, weights = {}, None
    for origin in cfg['origins']:
        group = [(j, p) for j, p in records if j.origin == origin]
        rows, base, candidate, w = combine(group, cfg['seeds'], cfg['half_life_periods'])
        if np.isfinite(global_base[rows]).any():
            raise ValueError('Overlapping temporal score windows')
        if weights is not None and weights != w:
            raise ValueError('Weight rule changed between origins')
        weights = w
        nd = data['train_nodata'][rows]
        base, candidate = postprocess(base, None, nd), postprocess(candidate, None, nd)
        global_base[rows], global_candidate[rows] = base, candidate
        by_origin[str(origin)] = paired_panel(base, candidate, data['y'][rows], data['month'][rows])
        for seed in cfg['seeds']:
            sr, sb, sc, _ = combine([(j, p) for j, p in group if j.seed == seed], [seed], cfg['half_life_periods'])
            if not np.array_equal(sr, rows):
                raise ValueError('Seed row mismatch')
            by_seed[seed][0][rows] = postprocess(sb, None, nd)
            by_seed[seed][1][rows] = postprocess(sc, None, nd)
    rows = np.flatnonzero(np.isfinite(global_base))
    y, months = data['y'][rows], data['month'][rows]
    pooled = paired_panel(global_base[rows], global_candidate[rows], y, months)
    seed_panels = {str(s): paired_panel(b[rows], c[rows], y, months) for s, (b, c) in by_seed.items()}
    checks = {
        'each_origin_ge_0p002': all(v['delta'] >= cfg['min_delta_each_origin'] for v in by_origin.values()),
        'each_seed_pooled_ge_0p002': all(v['delta'] >= cfg['min_delta_each_seed_pooled'] for v in seed_panels.values()),
        'at_least_six_positive_months': sum(v > 0 for v in pooled['monthly_delta'].values()) >= cfg['min_positive_months'],
        'all_lomo_positive': all(v > 0 for v in pooled['lomo_delta'].values()),
    }
    passed = all(checks.values())
    result = {'name': cfg['name'], 'config': cfg, 'config_sha256': sha256(args.config),
              'reference_signatures': [m['run_signature'] for m in metas],
              'model_count': len(records), 'fold_weights': weights, 'paired': pooled,
              'by_origin': by_origin, 'by_seed': seed_panels, 'gate': dict(checks, passed=passed),
              'accelerator_hours': 0, 'submitted': False,
              'note': 'One fixed candidate; previously consulted windows. No fresh-holdout or LB claim.'}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if passed:
        final_meta, final_records = recovered_run(args.final, dm, data)
        if final_meta['config']['mode'] != 'final':
            raise ValueError('Wrong final prediction mode')
        control = export_final(args.final, args.template, out / 'control', dm, data)
        if control['artifacts']['flow31_no_clip']['sha256'] != cfg['baseline_sha256']:
            raise ValueError('Final baseline differs from the submitted 0.142 champion')
        rows, _, pred, fw = combine(final_records, cfg['seeds'], cfg['half_life_periods'], split='test')
        if fw != weights or not np.array_equal(rows, np.arange(len(data['test_ids']))):
            raise ValueError('Final rule/IDs differ from confirmation')
        pred = postprocess(pred, None, data['test_nodata'])
        template = pd.read_csv(args.template)
        target = next(c for c in template if c != 'sample_id')
        template[target] = pd.Series(pred, index=data['test_ids']).reindex(template.sample_id).to_numpy()
        path = out / (cfg['name'] + '.csv')
        template.to_csv(path, index=False, lineterminator='\n')
        result['export'] = {'filename': path.name, 'sha256': sha256(path), 'rows': len(template)}
    write_json(out / 'review.json', result)
    print(json.dumps({'weights': weights, 'baseline': pooled['base']['overall'],
                      'candidate': pooled['candidate']['overall'], 'delta': pooled['delta'],
                      'origins': {k: v['delta'] for k, v in by_origin.items()},
                      'seeds': {k: v['delta'] for k, v in seed_panels.items()}, 'gate': result['gate']}, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/postprocessing/recency20.json')
    p.add_argument('--labels', required=True); p.add_argument('--confirm', nargs='+', required=True)
    p.add_argument('--final', required=True); p.add_argument('--template', required=True)
    p.add_argument('--out', required=True)
    evaluate(p.parse_args())
