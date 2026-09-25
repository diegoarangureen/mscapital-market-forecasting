import json
from pathlib import Path
import numpy as np
from .io import sha256, signature
from . import VERSION


def family_indices(names):
    names = [str(n) for n in names]
    flow = [i for i,n in enumerate(names) if 'ofi' in n or n.startswith('x30_ord') or
            n.startswith('x30_qi') or 'depth_ratio' in n]
    vol_names = {'x30_rv_60','x30_rv_600','x30_semi_ratio','x30_parkinson',
                 'x30_vol_of_vol','x30_signvol_60','x30_spread_rms','x30_spread_drift'}
    vol = [i for i,n in enumerate(names) if n in vol_names]
    price = [i for i in range(len(names)) if i not in flow and i not in vol]
    if (len(names),len(flow),len(vol),len(price)) != (36,18,8,10):
        raise ValueError('X30 schema does not match family definitions')
    return {'flow':flow, 'vol':vol, 'price':price, 'flow_price':sorted(flow+price)}


class Dataset:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        m = dict(self.manifest)
        fingerprint = m.pop('fingerprint')
        if fingerprint != signature(m) or m['audit_version'] != VERSION:
            raise ValueError('Dataset manifest changed or incompatible version')
        for name, entry in m['files'].items():
            if Path(name).name != name or sha256(self.root/name) != entry['sha256']:
                raise ValueError(f'Dataset checksum mismatch: {name}')
        self.y, self.months = self.load('y.npy'), self.load('month.npy')
        self.ids = self.load('train_ids.npy')
        if len(self.y) != len(self.ids) or len(self.months) != len(self.ids):
            raise ValueError('Dataset label alignment failure')
        if not np.array_equal(self.ids,np.arange(len(self.ids))) or not np.isfinite(self.y).all():
            raise ValueError('Invalid prepared labels/sample IDs')
        if self.load('base_train.npy').shape != (len(self.ids),len(self.load('base_names.npy'))):
            raise ValueError('Base feature schema mismatch')

    def load(self, name):
        if name not in self.manifest['files']:
            raise ValueError(f'Missing prepared artifact: {name}')
        return np.load(self.root/name, mmap_mode='r', allow_pickle=False)

    def features(self, arm, split, rows):
        base = np.asarray(self.load(f'base_{split}.npy')[rows], dtype=np.float32)
        if arm == 'base':
            if not np.isfinite(base).all():
                raise ValueError('Nonfinite base features')
            return base
        x30 = self.load(f'X30v2_{split}.npy')
        names = self.load('X30v2_names.npy')
        groups = family_indices(names)
        if arm == 'x30':
            indices = list(range(36))
        elif arm == 'flow31':
            indices = groups['flow']
        elif arm in groups:
            indices = groups[arm]
        else:
            raise ValueError(f'Unknown arm: {arm}')
        blocks = [base, np.asarray(x30[rows])[:, indices]]
        if arm == 'flow31':
            blocks.append(np.asarray(self.load(f'X31v2_{split}.npy')[rows]))
        result = np.concatenate(blocks, axis=1)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite assembled features')
        return result

    def names(self, arm):
        base = self.load('base_names.npy').tolist()
        if arm == 'base':
            return base
        names = self.load('X30v2_names.npy').tolist()
        groups = family_indices(names)
        indices = range(36) if arm == 'x30' else groups['flow' if arm == 'flow31' else arm]
        return base + [names[i] for i in indices] + (self.load('X31v2_names.npy').tolist() if arm == 'flow31' else [])
