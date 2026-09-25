"""Prepare aligned immutable matrices once on CPU. See TRAINING_AGENT.md."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from audit.io import sha256, write_json, signature
from audit import VERSION

DROP = [301, 267, 268, 299, 257]


def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Use an empty preparation directory; never overwrite a published dataset')
    files, sources = {}, {}

    def source(path, pickle=False):
        path = Path(path)
        sources[str(path.resolve())] = sha256(path)
        return np.load(path, mmap_mode=None if pickle else 'r', allow_pickle=pickle)

    def save(name, array):
        path = out / name
        np.save(path, array)
        files[name] = {'sha256': sha256(path), 'shape': list(array.shape)}

    data, x21, x22, public = map(Path, [args.data, args.x21, args.x22, args.public])
    y, month = source(data/'full_y.npy'), source(data/'full_month.npy')
    label_path = Path(args.labels)
    sources[str(label_path.resolve())] = sha256(label_path)
    labels = pd.read_feather(label_path).sort_values('sample_id', kind='stable')
    ids = labels.sample_id.to_numpy()
    if not np.array_equal(ids, np.arange(len(y))):
        raise ValueError('Raw labels must prove dense row = sample_id mapping')
    if not np.array_equal(labels.month.to_numpy(), month) or not np.array_equal(labels.target.to_numpy().astype(np.float32), y.astype(np.float32)):
        raise ValueError('Base labels/months do not match raw labels by sample_id')
    if not np.isfinite(y).all() or len(month) != len(y):
        raise ValueError('Invalid labels')
    save('y.npy', np.asarray(y, dtype=np.float32))
    save('month.npy', np.asarray(month))
    del labels
    full_names = [str(n) for n in source(data/'full_names.npy', pickle=True)]
    columns = None
    for split in ('train', 'test') if args.include_test else ('train',):
        base, a, b = [source(folder/f'{prefix}_{split}.npy') for folder, prefix in
                      ((data, 'full'), (x21, 'X21'), (x22, 'X22'))]
        n = len(base)
        csv_path = public/f'{split}.csv'
        sources[str(csv_path.resolve())] = sha256(csv_path)
        table = pd.read_csv(csv_path).sort_values('sample_id', kind='stable').reset_index(drop=True)
        if not np.array_equal(table.sample_id.to_numpy(), np.arange(n)) or len(a) != n or len(b) != n:
            raise ValueError(f'{split}: row alignment failure')
        excluded = {'sample_id', 'target', 'label', 'month', 'id'}
        fc = [c for c in table.columns if c not in excluded and str(table[c].dtype) in ('float64','float32','int64','int32')]
        if split == 'train':
            columns = fc
        elif fc != columns:
            raise ValueError('Public train/test column schema differs')
        if len(fc) != 152 or base.shape[1] + a.shape[1] + b.shape[1] - len(DROP) + len(fc) != 450:
            raise ValueError('Champion schema must contain exactly 450 columns (152 public)')
        names = full_names + [f'X21:{i}' for i in range(a.shape[1])] + [f'X22:{i}' for i in range(b.shape[1])]
        keep = np.delete(np.arange(len(names)), DROP)
        names = [names[i] for i in keep] + [f'public:{c}' for c in fc]
        path = out/f'base_{split}.npy'
        matrix = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(n,450))
        for start in range(0, n, 32768):
            sl = slice(start, start+32768)
            own = np.concatenate([base[sl], a[sl], b[sl]], axis=1)[:,keep]
            block = np.concatenate([own, table.iloc[sl][fc].to_numpy(dtype=np.float32)], axis=1)
            matrix[sl] = np.nan_to_num(block, nan=0., posinf=0., neginf=0.)
        matrix.flush()
        files[path.name] = {'sha256': sha256(path), 'shape': list(matrix.shape)}
        nodata = ((base[:, full_names.index('X2:tx_n')] == 0) &
                  (base[:, full_names.index('X2:mk_nbars')] == 0))
        save(f'{split}_ids.npy', np.arange(n, dtype=np.int64))
        save(f'{split}_nodata.npy', nodata)
        save('base_names.npy', np.asarray(names))
        del matrix, table, base, a, b
        gc.collect()
        for folder, pack in ((args.x30, 'X30v2'), (args.x31, 'X31v2')):
            if not folder:
                continue
            folder = Path(folder)
            meta_path = folder/f'{pack}_{split}_manifest.json'
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            if meta['pack'] != pack or meta['split'] != split or meta['audit_version'] != VERSION:
                raise ValueError('Feature version mismatch')
            for filename, digest in meta['files'].items():
                if Path(filename).name != filename or sha256(folder/filename) != digest:
                    raise ValueError('Feature manifest checksum mismatch')
            xx = source(folder/f'{pack}_{split}.npy')
            xid = source(folder/f'{pack}_{split}_ids.npy')
            xn = source(folder/f'{pack}_{split}_names.npy')
            if not np.array_equal(xid, np.arange(n)) or len(xx) != n or not np.isfinite(xx).all():
                raise ValueError('Feature pack alignment/finiteness failure')
            count = 36 if pack == 'X30v2' else 18
            if xx.shape[1] != count or len(xn) != count:
                raise ValueError('Unexpected pack schema')
            names_path = out/f'{pack}_names.npy'
            if names_path.exists() and not np.array_equal(np.load(names_path), xn):
                raise ValueError('Feature train/test names differ')
            save(f'{pack}_{split}.npy', xx)
            save(f'{pack}_names.npy', xn)
    manifest = {'audit_version': VERSION, 'files': files, 'sources': sources,
                'row_contract': 'dense sample_id; raw train labels/months checked; X21/X22 use original row-indexed builder convention',
                'public_columns': columns, 'drop_indices': DROP}
    manifest['fingerprint'] = signature(manifest)
    write_json(out/'manifest.json', manifest)
    print(f'PREPARED {out} fingerprint={manifest["fingerprint"]}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('data', 'x21', 'x22', 'public', 'labels', 'out'):
        p.add_argument('--'+flag, required=True)
    p.add_argument('--x30'); p.add_argument('--x31')
    p.add_argument('--include-test', action='store_true')
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
