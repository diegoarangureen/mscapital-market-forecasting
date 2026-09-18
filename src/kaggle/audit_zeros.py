# Audit: fraction of market rows with zero ask1/bid1 (empty levels) per split + per-sample incidence.
import pyarrow as pa, numpy as np, json
DATA = '/kaggle/input/competitions/ms-capital-real-financial-market-forecasting'
out = {}
for split in ['train','test']:
    src = pa.memory_map(f'{DATA}/{split}/market.feather','r')
    reader = pa.ipc.open_file(src)
    ia = reader.schema.get_field_index('ask_price_1'); ib = reader.schema.get_field_index('bid_price_1')
    isid = reader.schema.get_field_index('sample_id')
    tot=0; z=0; zb=0; zboth=0
    hasz = None
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i)
        a = b.column(ia).to_numpy(zero_copy_only=False)
        d = b.column(ib).to_numpy(zero_copy_only=False)
        sid = b.column(isid).to_numpy(zero_copy_only=False).astype(np.int64)
        za = a==0; zb_ = d==0
        tot += len(a); z += int(za.sum()); zb += int(zb_.sum()); zboth += int((za&zb_).sum())
        ns = int(sid.max())+1
        if hasz is None or len(hasz) < ns:
            h = np.zeros(max(ns, len(hasz) if hasz is not None else 0), np.int64)
            if hasz is not None: h[:len(hasz)] = hasz
            hasz = h
        hasz += np.bincount(sid[za|zb_], minlength=len(hasz))
        del a, d, sid
    out[split] = dict(rows=int(tot), ask0=z, bid0=zb, both0=zboth,
                      pct_zero_side=(z+zb-zboth)/tot, pct_samples_with_zero=float((hasz>0).mean()))
    print(split, json.dumps(out[split]), flush=True)
json.dump(out, open('/kaggle/working/audit_zeros.json','w'))
