"""Extract the exact v7 market grid builder for local test inference."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
reference = ROOT / 'data/interim/kaggle_kernels/multistream_factorized_transformer_dev/run_v7_baseline.py'
tree = ast.parse(reference.read_text(encoding='utf-8'))
functions = {node.name: ast.unparse(node) for node in tree.body if isinstance(node, ast.FunctionDef)}
names = ['build_indexer', 'map_ids', 'iter_batches', 'col_np', 'open_memmap',
         'feature_paths', 'market_bin', 'add_at_3d', 'clean_feature_block', 'build_market_grid']
body = '\n\n'.join(functions[name] for name in names)
header = """from pathlib import Path
import json
import os
import time
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/raw'
CACHE_DIR = ROOT / 'data/processed/subsecond_fulltrain_grid'
BATCH_SIZE_ARROW = 65536
GRID_VERSION = 'v2'
MARKET_LEN = 200
MARKET_SECONDS = 600.0
MARKET_FEATURES = list(range(11))
FLOW_LEN = 60
TX_FEATURES = list(range(7))
ORDER_FEATURES = list(range(10))
EPS = 1e-6
"""
footer = """def main():
    sample_ids = pd.read_csv(DATA / 'submission.csv', usecols=['sample_id'])['sample_id'].to_numpy(dtype=np.int64)
    if len(sample_ids) != 647896 or np.any(sample_ids[1:] <= sample_ids[:-1]):
        raise AssertionError('Unexpected or unsorted test sample IDs')
    paths = build_market_grid('test', sample_ids)
    expected = len(sample_ids) * MARKET_LEN * len(MARKET_FEATURES) * np.dtype(np.float16).itemsize
    if paths['market'].stat().st_size != expected:
        raise AssertionError('Test market mmap size mismatch')
    manifest = {'rows': len(sample_ids), 'market_shape': [len(sample_ids), MARKET_LEN, len(MARKET_FEATURES)],
                'dtype': 'float16', 'market_bytes': paths['market'].stat().st_size,
                'count_bytes': paths['market_count'].stat().st_size,
                'source': str(DATA / 'test/market.feather'), 'builder': 'exact functions extracted from run_v7_baseline.py'}
    (CACHE_DIR / 'test_market_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == '__main__':
    main()
"""
script = '\n\n'.join((header, body, footer))
destination = ROOT / 'scripts/build_test_market_grid_v2.py'
compile(script, str(destination), 'exec')
destination.write_text(script, encoding='utf-8')
print(destination)
