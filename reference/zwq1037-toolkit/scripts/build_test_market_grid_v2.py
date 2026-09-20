from pathlib import Path
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


def build_indexer(sample_ids):
    max_id = int(sample_ids.max())
    if max_id <= 25000000:
        indexer = np.full(max_id + 1, -1, dtype=np.int32)
        indexer[sample_ids] = np.arange(sample_ids.size, dtype=np.int32)
        return ('direct', indexer)
    return ('dict', {int(s): i for i, s in enumerate(sample_ids)})

def map_ids(batch_ids, indexer):
    kind, payload = indexer
    if kind == 'direct':
        out = np.full(batch_ids.size, -1, dtype=np.int32)
        ok = (batch_ids >= 0) & (batch_ids < payload.size)
        out[ok] = payload[batch_ids[ok]]
        return out
    return np.fromiter((payload.get(int(x), -1) for x in batch_ids), dtype=np.int32, count=batch_ids.size)

def iter_batches(path, columns):
    if ds is not None:
        try:
            dataset = ds.dataset(str(path), format='ipc')
            scanner = dataset.scanner(columns=columns, batch_size=BATCH_SIZE_ARROW, use_threads=False)
            for i, batch in enumerate(scanner.to_batches(), start=1):
                yield (batch, i, None)
            return
        except Exception as exc:
            print(f'  dataset scanner fallback for {path.name}: {type(exc).__name__}: {exc}', flush=True)
    source = pa.memory_map(str(path), 'r')
    try:
        try:
            reader = ipc.RecordBatchFileReader(source)
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                try:
                    batch = batch.select(columns)
                except Exception:
                    pass
                yield (batch, i + 1, reader.num_record_batches)
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.open_stream(source)
            for i, batch in enumerate(reader, start=1):
                try:
                    batch = batch.select(columns)
                except Exception:
                    pass
                yield (batch, i, None)
    finally:
        source.close()

def col_np(batch, name, dtype=None):
    arr = batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr

def open_memmap(path, shape, dtype=np.float16, mode='w+'):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)

def feature_paths(split):
    return {'market': CACHE_DIR / f'{split}_{GRID_VERSION}_market_{MARKET_LEN}x{len(MARKET_FEATURES)}.mmap', 'market_count': CACHE_DIR / f'{split}_{GRID_VERSION}_market_count_{MARKET_LEN}.mmap', 'tx': CACHE_DIR / f'{split}_{GRID_VERSION}_tx_{FLOW_LEN}x{len(TX_FEATURES)}.mmap', 'tx_count': CACHE_DIR / f'{split}_{GRID_VERSION}_tx_count_{FLOW_LEN}.mmap', 'order': CACHE_DIR / f'{split}_{GRID_VERSION}_order_{FLOW_LEN}x{len(ORDER_FEATURES)}.mmap', 'order_count': CACHE_DIR / f'{split}_{GRID_VERSION}_order_count_{FLOW_LEN}.mmap'}

def market_bin(seconds):
    bins = MARKET_LEN - 1 - np.floor(seconds / (MARKET_SECONDS / MARKET_LEN)).astype(np.int32)
    return np.clip(bins, 0, MARKET_LEN - 1)

def add_at_3d(arr, row, col, feat, values):
    values = np.nan_to_num(values, nan=0.0, posinf=50.0, neginf=-50.0)
    values = np.clip(values, -50.0, 50.0)
    np.add.at(arr, (row, col, feat), values.astype(arr.dtype, copy=False))

def clean_feature_block(x, clip=50.0):
    x = np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(x, -clip, clip)

def build_market_grid(split, sample_ids):
    n = sample_ids.size
    paths = feature_paths(split)
    feat = open_memmap(paths['market'], (n, MARKET_LEN, len(MARKET_FEATURES)), np.float16)
    cnt = open_memmap(paths['market_count'], (n, MARKET_LEN), np.float16)
    feat[:] = 0.0
    cnt[:] = 0.0
    indexer = build_indexer(sample_ids)
    columns = ['sample_id', 'seconds_before_predict', 'transaction_avgprice', 'transaction_volume', 'transaction_count', 'ask_price_1', 'ask_price_2', 'bid_price_1', 'bid_price_2', 'ask_volume_1', 'ask_volume_2', 'bid_volume_1', 'bid_volume_2']
    t0 = time.time()
    rows_seen = 0
    for batch, batch_no, total_batches in iter_batches(DATA / split / 'market.feather', columns):
        sid = col_np(batch, 'sample_id', np.int64)
        row = map_ids(sid, indexer)
        ok = row >= 0
        if not np.all(ok):
            row = row[ok]
        sec = col_np(batch, 'seconds_before_predict', np.float32)
        ask1 = col_np(batch, 'ask_price_1', np.float32)
        ask2 = col_np(batch, 'ask_price_2', np.float32)
        bid1 = col_np(batch, 'bid_price_1', np.float32)
        bid2 = col_np(batch, 'bid_price_2', np.float32)
        askv1 = col_np(batch, 'ask_volume_1', np.float32)
        askv2 = col_np(batch, 'ask_volume_2', np.float32)
        bidv1 = col_np(batch, 'bid_volume_1', np.float32)
        bidv2 = col_np(batch, 'bid_volume_2', np.float32)
        txpx = col_np(batch, 'transaction_avgprice', np.float32)
        txv = col_np(batch, 'transaction_volume', np.float32)
        txc = col_np(batch, 'transaction_count', np.float32)
        if not np.all(ok):
            sec = sec[ok]
            ask1 = ask1[ok]
            ask2 = ask2[ok]
            bid1 = bid1[ok]
            bid2 = bid2[ok]
            askv1 = askv1[ok]
            askv2 = askv2[ok]
            bidv1 = bidv1[ok]
            bidv2 = bidv2[ok]
            txpx = txpx[ok]
            txv = txv[ok]
            txc = txc[ok]
        col = market_bin(sec)
        mid = (ask1 + bid1) * 0.5
        depth1 = askv1 + bidv1
        depth2 = depth1 + askv2 + bidv2
        imb1 = (bidv1 - askv1) / (depth1 + EPS)
        imb2 = (bidv1 + bidv2 - askv1 - askv2) / (depth2 + EPS)
        micro = (ask1 * bidv1 + bid1 * askv1) / (depth1 + EPS)
        values = [mid - 1.0, txpx - 1.0, (ask1 - bid1) / (mid + EPS), (ask2 - bid2) / (mid + EPS), imb1, imb2, micro / (mid + EPS) - 1.0, (ask2 - ask1 + (bid1 - bid2)) / (mid + EPS), np.log1p(depth2), np.log1p(txv), np.log1p(txc)]
        np.add.at(cnt, (row, col), 1.0)
        for j, val in enumerate(values):
            add_at_3d(feat, row, col, j, val)
        rows_seen += batch.num_rows
        if batch_no % 200 == 0:
            print(f'  {split}/market batch={batch_no}, rows={rows_seen:,}, elapsed={time.time() - t0:.1f}s', flush=True)
        del batch, sid, row, ok, sec, ask1, ask2, bid1, bid2, askv1, askv2, bidv1, bidv2, txpx, txv, txc
    mask = cnt > 0
    for j in range(len(MARKET_FEATURES)):
        channel = feat[:, :, j]
        channel[mask] = channel[mask] / cnt[mask]
        channel[~mask] = 0.0
        feat[:, :, j] = clean_feature_block(channel)
    feat.flush()
    cnt.flush()
    print(f'  {split}/market grid done, rows={rows_seen:,}, time={time.time() - t0:.1f}s', flush=True)
    return paths

def main():
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
