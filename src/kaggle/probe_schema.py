# X30 prep v2 (light): exact feather schemas + bounded row stats. v1 was cancelled mid-run
# (full num_rows scans over all batches of multi-GB feathers - too slow). v2: no full scans.
# Output: /kaggle/working/schema_report.txt (downloaded as kernel output - stdout not API-retrievable).
import os, json
import numpy as np
import pyarrow as pa
BASE = os.environ.get('BASE', '/kaggle/input/competitions/ms-capital-real-financial-market-forecasting')
SPLIT = os.environ.get('SPLIT', 'train')
rep = {}
for name in ['market', 'order', 'transaction']:
    reader = pa.ipc.open_file(f'{BASE}/{SPLIT}/{name}.feather')
    schema = [(f.name, str(f.type)) for f in reader.schema]
    nb = reader.num_record_batches
    b0 = reader.get_batch(0)
    rep[name] = {'columns': schema, 'num_record_batches': nb,
                 'rows_in_batch0': b0.num_rows, 'est_rows': nb * b0.num_rows}
# market: sbp range + rows per sample over first 6 batches only
sbp_min = 1e18; sbp_max = -1e18; counts = {}
reader = pa.ipc.open_file(f'{BASE}/{SPLIT}/market.feather')
for i in range(min(reader.num_record_batches, 6)):
    b = reader.get_batch(i)
    sbp = b.column('seconds_before_predict').to_numpy(zero_copy_only=False)
    sid = b.column('sample_id').to_numpy(zero_copy_only=False)
    sbp_min = min(sbp_min, float(np.min(sbp))); sbp_max = max(sbp_max, float(np.max(sbp)))
    for s in np.unique(sid): counts[int(s)] = counts.get(int(s), 0) + int((sid==s).sum())
c = np.array(list(counts.values()))
rep['market_stats'] = {'sbp_min': sbp_min, 'sbp_max': sbp_max,
                       'rows_per_sample_min': int(c.min()), 'rows_per_sample_med': float(np.median(c)),
                       'rows_per_sample_max': int(c.max()), 'n_samples_seen': len(counts)}
# order_action / side value sets (first 4 batches each)
for name, col in [('order','order_action'), ('order','side'), ('transaction','side')]:
    reader = pa.ipc.open_file(f'{BASE}/{SPLIT}/{name}.feather')
    vals = set()
    for i in range(min(reader.num_record_batches, 4)):
        v = reader.get_batch(i).column(col).to_numpy(zero_copy_only=False)
        vals.update(np.unique(v).tolist())
    rep[f'{name}_{col}_values'] = sorted([str(x) for x in vals])[:20]
with open('/kaggle/working/schema_report.txt', 'w') as f:
    f.write(json.dumps(rep, indent=1))
print('DONE')
