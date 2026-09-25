import hashlib
import json
import os
from pathlib import Path
import numpy as np
from . import VERSION


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def save_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def save_feature_pack(out, name, split, matrix, names):
    out = Path(out)
    if matrix.ndim != 2 or matrix.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError('Invalid feature schema')
    if not np.isfinite(matrix).all():
        raise ValueError('Feature pack contains NaN/Inf; fix its builder')
    files = {f'{name}_{split}.npy': matrix,
             f'{name}_{split}_names.npy': np.asarray(names, dtype=str),
             f'{name}_{split}_ids.npy': np.arange(len(matrix), dtype=np.int64)}
    for filename, data in files.items():
        np.save(out / filename, data)
    write_json(out / f'{name}_{split}_manifest.json', {
        'audit_version': VERSION, 'pack': name, 'split': split, 'shape': list(matrix.shape),
        'chronology': 'seconds_before_predict descending; ties retain input order',
        'sample_complete_chunks': True, 'split_fitted_statistics': False,
        'files': {filename: sha256(out / filename) for filename in files},
    })
