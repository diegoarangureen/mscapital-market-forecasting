"""Verify recovered predictions, review fixed postprocessing and export a candidate.

Uses no training features, performs no fit or parameter search, and never submits.
Additional confirmation directories may extend seeds only with identical source,
runtime, dataset and training recipe. Previously viewed windows remain reused.
"""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit.io import sha256, write_json
from audit.metrics import PredictionMean, panel, paired_panel, postprocess
from audit.protocol import jobs


def labels(root):
    root = Path(root)
    meta = json.loads((root / 'manifest.json').read_text())
    arrays = {}
    for name in ('y', 'month', 'train_ids', 'train_nodata', 'test_ids', 'test_nodata'):
        path = root / (name + '.npy')
        spec = meta['files'][path.name]
        if sha256(path) != spec['sha256']:
            raise ValueError('Label/ID checksum mismatch: ' + name)
        arrays[name] = np.load(path, allow_pickle=False)
        if list(arrays[name].shape) != spec['shape']:
            raise ValueError('Label/ID shape mismatch: ' + name)
    if not np.array_equal(arrays['train_ids'], np.arange(len(arrays['y']))):
        raise ValueError('Training IDs are not dense/aligned')
    return meta, arrays


def recovered_run(root, dataset_meta, data):
    root = Path(root)
    meta = json.loads((root / 'run_worker_00.json').read_text())
    if meta['dataset_fingerprint'] != dataset_meta['fingerprint']:
        raise ValueError('Run/dataset fingerprint mismatch')
    result = []
    for job in jobs(meta['config']):
        if job.arm != 'flow31':
            continue
        folder = root / 'jobs' / job.key
        done = json.loads((folder / 'done.json').read_text())
        path = folder / 'predictions.npz'
        if (done['run_signature'] != meta['run_signature'] or done['job'] != asdict(job)
                or sha256(path) != done['artifacts']['predictions.npz']):
            raise ValueError('Job provenance/checksum mismatch: ' + job.key)
        with np.load(path, allow_pickle=False) as z:
            p = {k: z[k] for k in z.files}
        _, es, score = job.masks(data['month'])
        for prefix, mask in [('es', es), ('score', score)]:
            rows = np.flatnonzero(mask)
            if not np.array_equal(p[prefix + '_rows'], rows):
                raise ValueError('Prediction row mismatch: ' + job.key)
            pred = p[prefix + '_pred']
            if pred.shape != rows.shape or not np.isfinite(pred).all():
                raise ValueError('Invalid prediction values: ' + job.key)
        if meta['config']['mode'] == 'final':
            if (not np.array_equal(p['test_rows'], np.arange(len(data['test_ids'])))
                    or p['test_pred'].shape != data['test_ids'].shape
                    or not np.isfinite(p['test_pred']).all()):
                raise ValueError('Test prediction mapping/value failure: ' + job.key)
        result.append((job, p))
    return meta, result


def assert_seed_extension(reference, other):
    fields = ('backend', 'torch_version', 'torch_xla_version', 'code_hashes', 'dataset_fingerprint')
    for field in fields:
        if reference.get(field) != other.get(field):
            raise ValueError('Incompatible seed extension: ' + field)
    ignored = {'seeds', 'arms', 'frozen_from'}
    recipe = lambda m: {k: v for k, v in m['config'].items() if k not in ignored}
    if recipe(reference) != recipe(other):
        raise ValueError('Incompatible seed-extension training recipe')


def review_confirm(runs, dataset_meta, data):
    all_jobs, manifests = [], []
    for root in runs:
        meta, records = recovered_run(root, dataset_meta, data)
        if meta['config']['mode'] != 'confirm':
            raise ValueError('Confirmation review requires external score windows')
        if manifests:
            assert_seed_extension(manifests[0], meta)
        manifests.append(meta)
        all_jobs.extend(records)
    if len({j.key for j, _ in all_jobs}) != len(all_jobs):
        raise ValueError('Duplicate confirmation models')
    n = len(data['y'])
    seeds = sorted({j.seed for j, _ in all_jobs})
    origins = sorted({j.origin for j, _ in all_jobs})
    vectors = {v: np.full(n, np.nan) for v in ('raw', 'clip_only', 'nodata_only', 'champion')}
    monthly = {}
    subsets = {','.join(map(str, s)): np.full(n, np.nan)
               for count in range(1, len(seeds) + 1) for s in itertools.combinations(seeds, count)}
    for origin in origins:
        records = [(j, p) for j, p in all_jobs if j.origin == origin]
        if {(j.fold, j.seed) for j, _ in records} != set(itertools.product(range(5), seeds)):
            raise ValueError('Incomplete five-fold/seed panel')
        es, score = PredictionMean(n), PredictionMean(n)
        by_seed = {s: PredictionMean(n) for s in seeds}
        for job, p in records:
            es.add(job.key, p['es_rows'], p['es_pred'])
            score.add(job.key, p['score_rows'], p['score_pred'])
            by_seed[job.seed].add(job.key, p['score_rows'], p['score_pred'])
        rows = np.flatnonzero(score.count)
        if np.isfinite(vectors['raw'][rows]).any() or not np.all(score.count[rows] == len(records)):
            raise ValueError('Overlapping windows or incomplete coverage')
        q = manifests[0]['config']['clip_quantiles']
        bounds = np.quantile(es.mean()[es.count > 0], q).tolist() if q else None
        raw = score.mean()[rows]
        nd = data['train_nodata'][rows]
        values = {'raw': raw, 'clip_only': postprocess(raw, bounds),
                  'nodata_only': postprocess(raw, None, nd), 'champion': postprocess(raw, bounds, nd)}
        for name, pred in values.items():
            vectors[name][rows] = pred
        y, m = data['y'][rows], data['month'][rows]
        monthly[str(origin)] = {'bounds': bounds, 'model_count': len(records),
            'variants': {name: panel(pred, y, m) for name, pred in values.items()},
            'no_clip_minus_champion': paired_panel(values['champion'], values['nodata_only'], y, m)}
        for key, vector in subsets.items():
            included = list(map(int, key.split(',')))
            pred = np.mean([by_seed[s].mean()[rows] for s in included], axis=0)
            vector[rows] = postprocess(pred, None, nd)
    rows = np.flatnonzero(np.isfinite(vectors['raw']))
    y, m = data['y'][rows], data['month'][rows]
    report = {'seeds': seeds, 'models': len(all_jobs), 'origins': monthly,
              'run_signatures': [v['run_signature'] for v in manifests],
              'variants': {name: panel(v[rows], y, m) for name, v in vectors.items()},
              'no_clip_minus_champion': paired_panel(vectors['champion'][rows], vectors['nodata_only'][rows], y, m),
              'fixed_seed_subsets_no_clip': {key: panel(v[rows], y, m) for key, v in subsets.items()},
              'note': 'Checkpoint-external but previously consulted windows; neither fresh holdout nor LB improvement.'}
    if set(seeds) == {2026, 42, 123}:
        report['third_seed_increment_no_clip'] = paired_panel(
            subsets['42,2026'][rows], subsets['42,123,2026'][rows], y, m)
    return report


def export_final(root, template_path, out, dataset_meta, data):
    meta, records = recovered_run(root, dataset_meta, data)
    if meta['config']['mode'] != 'final' or meta['config']['seeds'] != [2026, 42, 123]:
        raise ValueError('Expected the frozen three-seed champion')
    if len(records) != 15:
        raise ValueError('Expected all fifteen champion models')
    test = PredictionMean(len(data['test_ids']))
    es = PredictionMean(len(data['y']))
    for job, p in records:
        test.add(job.key, p['test_rows'], p['test_pred'])
        es.add(job.key, p['es_rows'], p['es_pred'])
    if not np.all(test.count == 15):
        raise ValueError('Incomplete final ensemble')
    raw = test.mean()
    bounds = np.quantile(es.mean()[es.count > 0], meta['config']['clip_quantiles']).tolist()
    champion = postprocess(raw, bounds, data['test_nodata'])
    candidate = postprocess(raw, None, data['test_nodata'])
    template = pd.read_csv(template_path)
    cols = [c for c in template if c != 'sample_id']
    ids = data['test_ids']
    if ('sample_id' not in template or len(cols) != 1 or template.sample_id.duplicated().any()
            or len(set(ids.tolist())) != len(ids) or set(template.sample_id) != set(ids.tolist())):
        raise ValueError('Submission template ID/schema mismatch')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, pred in [('champion_reconstructed', champion), ('flow31_no_clip', candidate)]:
        df = template.copy()
        df[cols[0]] = pd.Series(pred, index=ids).reindex(df.sample_id).to_numpy()
        path = out / (name + '.csv')
        df.to_csv(path, index=False, lineterminator='\n')
        check = pd.read_csv(path)
        if not np.array_equal(check.sample_id, template.sample_id) or not np.isfinite(check[cols[0]]).all():
            raise ValueError('Export roundtrip failed')
        artifacts[name] = {'filename': path.name, 'sha256': sha256(path), 'rows': len(df),
                           'mean': float(pred.mean()), 'std': float(pred.std()),
                           'min': float(pred.min()), 'max': float(pred.max())}
    # Historical champion hash was recorded as a 16-character prefix.
    if not artifacts['champion_reconstructed']['sha256'].startswith('c2e5e71c03bcc75c'):
        raise ValueError('Reconstructed champion differs from the submitted artifact')
    return {'source_signature': meta['run_signature'], 'models': 15, 'bounds': bounds,
            'changed_rows': int(np.count_nonzero(candidate != champion)),
            'nodata_rows': int(np.count_nonzero(data['test_nodata'])), 'artifacts': artifacts,
            'submitted': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', required=True); p.add_argument('--confirm', nargs='+', required=True)
    p.add_argument('--final'); p.add_argument('--template'); p.add_argument('--out', required=True)
    a = p.parse_args()
    dm, data = labels(a.labels)
    report = {'confirmation': review_confirm(a.confirm, dm, data)}
    if a.final:
        if not a.template:
            p.error('--final requires --template')
        report['export'] = export_final(a.final, a.template, a.out, dm, data)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'review.json', report)
    c = report['confirmation']
    print(json.dumps({'seeds': c['seeds'], 'models': c['models'],
                      'no_clip_score': c['variants']['nodata_only']['overall'],
                      'champion_score': c['variants']['champion']['overall'],
                      'delta': c['no_clip_minus_champion']['delta'],
                      'export': report.get('export')}, indent=2))
