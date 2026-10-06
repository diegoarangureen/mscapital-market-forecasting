"""Kaggle GPU wrapper prefix for final-mode kernels: resolves the prepared dataset mount, then runs the bundled audited trainer."""
import sys, pathlib
def _find(slug, marker):
    root = pathlib.Path('/kaggle/input')
    for p in sorted(root.rglob(marker)):
        if slug in str(p): return p.parent
    for p in sorted(root.rglob(marker)): return p.parent
    raise FileNotFoundError(f'{slug}/{marker} under /kaggle/input')
if __name__ == '__main__':
    data = _find('mscapital-prepared-v2','manifest.json')
    print('DATA',data,flush=True)
    sys.argv += ['--data',str(data),'--out','/kaggle/working/run','--device','cuda','--max-hours','MAXH']
