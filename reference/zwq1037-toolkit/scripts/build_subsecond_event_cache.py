"""Build bounded event-time caches, merging ties and preserving a prefix summary."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]


def encode_quantities(quantities):
    result = quantities.copy()
    for start in range(0, quantities.shape[1], 3):
        volume = quantities[:, start]
        result[:, start] = np.log1p(volume)
        result[:, start + 1] = np.log1p(quantities[:, start + 1])
        result[:, start + 2] = np.divide(
            quantities[:, start + 2], volume,
            out=np.ones(len(volume)), where=volume > 0,
        ) - 1.0
    return np.clip(result, -20.0, 20.0)


def write_complete_samples(columns, sample_ids, output, lengths, limit, source):
    sid, seconds, price, volume, side = columns[:5]
    action = columns[5] if source == 'order' else np.zeros(len(sid), dtype=np.int8)
    valid = (seconds >= 0) & (seconds <= 60) & np.isfinite(price) & (price > 0)
    valid &= (volume >= 0) & (side >= 0) & (side <= 1) & (action >= 0) & (action <= 1)
    sid, seconds, price, volume, side, action = [
        values[valid] for values in (sid, seconds, price, volume, side, action)
    ]
    if len(sid) == 0:
        return 0
    ordering = np.lexsort((-seconds, sid))
    sid, seconds, price, volume, side, action = [
        values[ordering] for values in (sid, seconds, price, volume, side, action)
    ]
    event_start = np.r_[0, np.flatnonzero((sid[1:] != sid[:-1]) | (seconds[1:] != seconds[:-1])) + 1]
    group_sid = sid[event_start]
    group_seconds = seconds[event_start]
    categories = 4 if source == 'order' else 2
    category = action * 2 + side
    quantities = np.zeros((len(event_start), categories * 3), dtype=np.float64)
    for group in range(categories):
        present = category == group
        group_volume = np.where(present, volume, 0).astype(np.float64)
        quantities[:, group * 3] = np.add.reduceat(group_volume, event_start)
        quantities[:, group * 3 + 1] = np.add.reduceat(present.astype(np.float64), event_start)
        quantities[:, group * 3 + 2] = np.add.reduceat(group_volume * price, event_start)

    sample_start = np.r_[0, np.flatnonzero(group_sid[1:] != group_sid[:-1]) + 1]
    sample_count = np.diff(np.r_[sample_start, len(group_sid)])
    unique_ids = group_sid[sample_start]
    global_rows = np.searchsorted(sample_ids, unique_ids)
    if np.any(global_rows >= len(sample_ids)) or not np.array_equal(sample_ids[global_rows], unique_ids):
        raise AssertionError('Unexpected sample ID in event source')
    if np.any(lengths[global_rows] != 0):
        raise AssertionError('Sample was split between completed blocks')
    local_rows = np.repeat(np.arange(len(sample_count)), sample_count)
    ranks = np.arange(len(group_sid)) - np.repeat(sample_start, sample_count)
    older_count = np.maximum(sample_count - limit, 0)
    older = ranks < older_count[local_rows]
    keep = ~older
    positions = ranks[keep] - older_count[local_rows[keep]] + (older_count[local_rows[keep]] > 0)
    target_rows = global_rows[local_rows[keep]]
    features = np.zeros((keep.sum(), categories * 3 + 3), dtype=np.float64)
    features[:, :categories * 3] = encode_quantities(quantities[keep])
    delta = np.r_[0.0, np.maximum(group_seconds[:-1] - group_seconds[1:], 0.0)]
    delta[sample_start] = 0.0
    features[:, -3] = np.log1p(delta[keep])
    features[:, -2] = group_seconds[keep] / 60.0
    output[target_rows, positions] = features.astype(np.float16)

    truncated = np.flatnonzero(older_count > 0)
    if truncated.size:
        prefix_quantities = np.zeros((len(sample_count), categories * 3), dtype=np.float64)
        np.add.at(prefix_quantities, local_rows[older], quantities[older])
        prefix_seconds = np.bincount(local_rows[older], weights=group_seconds[older], minlength=len(sample_count))
        prefix = np.zeros((len(truncated), categories * 3 + 3), dtype=np.float64)
        prefix[:, :categories * 3] = encode_quantities(prefix_quantities[truncated])
        prefix[:, -2] = prefix_seconds[truncated] / older_count[truncated] / 60.0
        prefix[:, -1] = 1.0
        output[global_rows[truncated], 0] = prefix.astype(np.float16)
    lengths[global_rows] = np.minimum(sample_count, limit) + (older_count > 0)
    return len(sample_count)


def build_source(split, source, sample_ids, directory):
    limit = 128 if source == 'transaction' else 256
    categories = 2 if source == 'transaction' else 4
    feature_count = categories * 3 + 3
    output_path = directory / f'{source}_events.npy'
    output = np.lib.format.open_memmap(
        output_path, mode='w+', dtype=np.float16,
        shape=(len(sample_ids), limit + 1, feature_count),
    )
    output[:] = 0
    lengths = np.zeros(len(sample_ids), dtype=np.int16)
    names = ['sample_id', 'seconds_before_predict', 'price', 'volume', 'side']
    if source == 'order':
        names.append('order_action')
    scanner = ds.dataset(ROOT / f'data/raw/{split}/{source}.feather', format='ipc').scanner(
        columns=names, batch_size=131_072, use_threads=False,
    )
    pending = None
    completed = 0
    last_seen_id = -1
    for batch in scanner.to_batches():
        columns = [batch[name].to_numpy(zero_copy_only=False) for name in names]
        sid = columns[0]
        if len(sid) == 0:
            continue
        if np.any(sid[1:] < sid[:-1]) or sid[0] < last_seen_id:
            raise AssertionError('Source must be globally sorted by sample_id')
        last_seen_id = int(sid[-1])
        within = sid <= sample_ids[-1]
        stop_after = not within.all()
        columns = [values[within] for values in columns]
        if pending is not None:
            columns = [np.concatenate([left, right]) for left, right in zip(pending, columns)]
        if len(columns[0]) == 0:
            break
        sid = columns[0]
        boundary = len(sid) if stop_after else int(np.searchsorted(sid, sid[-1], side='left'))
        if boundary:
            completed += write_complete_samples(
                [values[:boundary] for values in columns], sample_ids,
                output, lengths, limit, source,
            )
        pending = None if stop_after else [values[boundary:].copy() for values in columns]
        if completed and completed // 50_000 != (completed - max(1, boundary // 100)) // 50_000:
            print(f'{source} cached_samples={completed:,}', flush=True)
        if stop_after:
            break
    if pending is not None and len(pending[0]):
        completed += write_complete_samples(pending, sample_ids, output, lengths, limit, source)
    output.flush()
    np.save(directory / f'{source}_lengths.npy', lengths)
    metadata = {
        'shape': list(output.shape), 'dtype': 'float16',
        'nonempty_rows': int((lengths > 0).sum()),
        'limit': limit, 'prefix_summary': True, 'padding': 'right',
        'feature_order': 'per category log1p(volume), log1p(count), vwap-1; log1p(real_dt), seconds/60, prefix_flag',
        'categories': ['buy', 'sell'] if source == 'transaction' else ['buy_new', 'sell_new', 'buy_cancel', 'sell_cancel'],
        'completed_samples': completed,
    }
    (directory / f'{source}_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({source: metadata}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    parser.add_argument('--source', choices=['transaction', 'order', 'all'], default='all')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    frame = pd.read_feather(ROOT / 'data/raw/label.feather', columns=['sample_id']) if args.split == 'train' else pd.read_csv(ROOT / 'data/raw/submission.csv', usecols=['sample_id'])
    sample_ids = np.sort(frame['sample_id'].to_numpy())
    if args.limit:
        sample_ids = sample_ids[:args.limit]
    suffix = f'_smoke{args.limit}' if args.limit else ''
    directory = ROOT / f'data/processed/{args.split}_subsecond_event_cache_v1{suffix}'
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / 'sample_ids.npy', sample_ids)
    for source in ['transaction', 'order'] if args.source == 'all' else [args.source]:
        build_source(args.split, source, sample_ids, directory)


if __name__ == '__main__':
    main()
