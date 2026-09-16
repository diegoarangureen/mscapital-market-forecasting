import os, traceback, json, time
t0=time.time()
try:
    import numpy as np, torch
    print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), flush=True)
    for base in ('/kaggle/input/datasets/diegoaranguren/mscapital-matrices',
                 '/kaggle/input/datasets/diegoaranguren/mscapital-seq2',
                 '/kaggle/input'):
        try:
            print(base, '->', sorted(os.listdir(base))[:12], flush=True)
        except Exception as e:
            print(base, 'LISTFAIL', e, flush=True)
    SQ='/kaggle/input/datasets/diegoaranguren/mscapital-seq2'
    MT='/kaggle/input/datasets/diegoaranguren/mscapital-matrices'
    X = np.memmap(f'{SQ}/seq2_train.f16', dtype=np.float16, mode='r', shape=(1257637,120,8))
    print('memmap ok', X.shape, X.dtype, flush=True)
    y = np.load(f'{MT}/full_y.npy'); m = np.load(f'{MT}/full_month.npy')
    print('y', y.shape, 'month', m.shape, flush=True)
    sample = np.asarray(X[:256], dtype=np.float32)
    print('sample', sample.shape, float(np.isfinite(sample).mean()), flush=True)
    json.dump({'ok': True, 'elapsed': time.time()-t0}, open('/kaggle/working/probe.json','w'))
    print('PROBE_OK', flush=True)
except Exception:
    tb = traceback.format_exc()
    print(tb, flush=True)
    try: open('/kaggle/working/probe_error.txt','w').write(tb)
    except Exception: pass
