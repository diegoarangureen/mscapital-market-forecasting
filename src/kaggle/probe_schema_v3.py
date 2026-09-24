# X30 prep v3 (metadata-only): pa.ipc.open_file reads ONLY the feather footer - zero data
# reads, finishes in seconds. v1 cancelled mid-run (unknown cause), v2 slow on get_batch.
# Delivers the one fact the X30 builder still needs: exact column list (L3+ depth?).
import os, json
import pyarrow as pa
BASE = os.environ.get('BASE', '/kaggle/input/competitions/ms-capital-real-financial-market-forecasting')
SPLIT = os.environ.get('SPLIT', 'train')
rep = {}
for name in ['market', 'order', 'transaction']:
    reader = pa.ipc.open_file(f'{BASE}/{SPLIT}/{name}.feather')
    rep[name] = {'columns': [(f.name, str(f.type)) for f in reader.schema],
                 'num_record_batches': reader.num_record_batches}
with open('/kaggle/working/schema_report_v3.txt', 'w') as f:
    f.write(json.dumps(rep, indent=1))
print('DONE')
