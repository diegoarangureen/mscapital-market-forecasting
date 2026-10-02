from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest

from audit.io import sha256
from audit.metrics import cosine
from audit.protocol import jobs
from review_champion import assert_seed_extension, review_confirm


def make_run(root, seeds, data):
    root.mkdir()
    meta = {'config': {'mode': 'confirm', 'arms': ['flow31'], 'seeds': seeds,
                      'folds': list(range(5)), 'clip_quantiles': [.1, .9]},
            'dataset_fingerprint': 'test-data', 'run_signature': str(seeds),
            'backend': 'xla', 'torch_version': 'test', 'torch_xla_version': 'test',
            'code_hashes': {'trainer': 'frozen'}}
    (root / 'run_worker_00.json').write_text(json.dumps(meta))
    for j in jobs(meta['config']):
        folder = root / 'jobs' / j.key
        folder.mkdir(parents=True)
        _, es, score = j.masks(data['month'])
        er, sr = np.flatnonzero(es), np.flatnonzero(score)
        pred = data['y'] * (1 + j.seed / 10000) + .01 * j.fold
        np.savez(folder / 'predictions.npz', es_rows=er, es_pred=pred[er],
                 score_rows=sr, score_pred=pred[sr])
        done = {'job': asdict(j), 'run_signature': meta['run_signature'],
                'artifacts': {'predictions.npz': sha256(folder / 'predictions.npz')}}
        (folder / 'done.json').write_text(json.dumps(done))
    return meta


def test_third_seed_extension_averages_all_models_and_rejects_partial(tmp_path):
    month = np.repeat(np.arange(71), 2)
    y = np.sin(np.arange(len(month)))
    data = {'month': month, 'y': y, 'train_nodata': np.zeros(len(y), bool)}
    dm = {'fingerprint': 'test-data'}
    a, b = tmp_path / 'old', tmp_path / 'extension'
    make_run(a, [2026, 42], data)
    make_run(b, [123], data)
    report = review_confirm([a, b], dm, data)
    rows = (month >= 62) & (month <= 70)
    expected = y[rows] * (1 + np.mean([2026, 42, 123]) / 10000) + .02
    assert report['models'] == 30
    assert report['variants']['raw']['overall'] == pytest.approx(cosine(expected, y[rows]))
    assert 'third_seed_increment_no_clip' in report
    next((b / 'jobs').glob('*/done.json')).unlink()
    with pytest.raises(FileNotFoundError):
        review_confirm([a, b], dm, data)


def test_seed_extension_rejects_changed_runtime_or_training_recipe():
    a = {'config': {'seeds': [1], 'arms': ['flow31'], 'noise_std': .005},
         'backend': 'xla', 'torch_version': 'same', 'code_hashes': {'trainer': 'same'}}
    b = deepcopy(a); b['config']['seeds'] = [2]
    assert_seed_extension(a, b)
    b['config']['noise_std'] = 0
    with pytest.raises(ValueError, match='recipe'):
        assert_seed_extension(a, b)
    b = deepcopy(a); b['backend'] = 'cuda'
    with pytest.raises(ValueError, match='backend'):
        assert_seed_extension(a, b)


def test_review_rejects_corrupted_predictions(tmp_path):
    month = np.repeat(np.arange(71), 2)
    data = {'month': month, 'y': np.sin(np.arange(len(month))),
            'train_nodata': np.zeros(len(month), bool)}
    root = tmp_path / 'run'
    make_run(root, [42], data)
    path = next((root / 'jobs').glob('*/predictions.npz'))
    path.write_bytes(path.read_bytes() + b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        review_confirm([root], {'fingerprint': 'test-data'}, data)
