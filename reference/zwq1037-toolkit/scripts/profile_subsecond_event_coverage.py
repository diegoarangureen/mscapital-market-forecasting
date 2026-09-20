"""Profile the exact-timestamp event representation before allocating caches."""
import json
import os
from pathlib import Path

os.environ.setdefault('POLARS_MAX_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import polars as pl

root = Path(__file__).resolve().parents[1]
labels = pd.read_feather(root / 'data/raw/label.feather', columns=['sample_id', 'month'])
selected_parts = []
for _, group in labels.groupby('month'):
    ids = group['sample_id'].to_numpy()
    selected_parts.append(ids[np.linspace(0, len(ids) - 1, min(50, len(ids)), dtype=np.int64)])
selected = np.unique(np.concatenate(selected_parts)).tolist()
results = {}
for source, limit in [('transaction', 128), ('order', 256)]:
    events = (
        pl.scan_ipc(root / f'data/raw/train/{source}.feather')
        .select('sample_id', 'seconds_before_predict', 'volume')
        .filter(pl.col('sample_id').is_in(selected))
        .filter((pl.col('seconds_before_predict') >= 0) & (pl.col('seconds_before_predict') <= 60))
        .collect(engine='streaming')
    )
    # 完全相同时间戳合并，不使用文件行序虚构同时间事件的先后。
    # Aggregate equal timestamps instead of treating file order as chronology.
    grouped = events.group_by('sample_id', 'seconds_before_predict').agg(
        pl.col('volume').cast(pl.Float64).sum().alias('volume'),
        pl.len().alias('raw_count'),
    ).sort(['sample_id', 'seconds_before_predict'], descending=[False, True])
    profiles = []
    for frame in grouped.partition_by('sample_id', maintain_order=True):
        seconds = frame['seconds_before_predict'].to_numpy()
        volume = frame['volume'].to_numpy()
        counts = frame['raw_count'].to_numpy()
        start = max(0, len(seconds) - limit)
        total_volume = volume.sum()
        full_span = seconds[0] - seconds[-1]
        kept_span = seconds[start] - seconds[-1]
        profiles.append({
            'sample_id': int(frame['sample_id'][0]),
            'unique_timestamps': len(seconds),
            'raw_events': int(counts.sum()),
            'kept_volume_share': float(volume[start:].sum() / total_volume) if total_volume > 0 else 1.0,
            'kept_span_seconds': float(kept_span),
            'full_span_seconds': float(full_span),
            'span_share': float(kept_span / full_span) if full_span > 0 else 1.0,
            'needs_prefix_summary': len(seconds) > limit,
        })
    profile = pd.DataFrame(profiles)
    active_cut = float(profile['raw_events'].quantile(0.9))
    active = profile[profile['raw_events'] >= active_cut]
    results[source] = {
        'sample_rows': len(profile),
        'limit': limit,
        'unique_count_quantiles': profile['unique_timestamps'].quantile([.5, .9, .99]).to_dict(),
        'truncated_share': float(profile['needs_prefix_summary'].mean()),
        'overall_volume_share_mean': float(profile['kept_volume_share'].mean()),
        'active_top10_volume_share_mean': float(active['kept_volume_share'].mean()),
        'active_top10_span_quantiles': active['kept_span_seconds'].quantile([.1, .5, .9]).to_dict(),
        'active_top10_span_share_mean': float(active['span_share'].mean()),
    }
    destination = root / f'data/interim/{source}_subsecond_coverage_profile.feather'
    profile.to_feather(destination)
    print(json.dumps({source: results[source]}, indent=2), flush=True)
    del events, grouped, profile
output = root / 'outputs/submission_metadata/subsecond_event_coverage_profile.json'
output.write_text(json.dumps(results, indent=2), encoding='utf-8')
