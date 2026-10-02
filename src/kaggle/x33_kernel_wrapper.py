"""Kaggle GPU wrapper prefix for X33 screen kernels: resolves dataset mounts robustly, then runs the bundled audited trainer."""
import sys, pathlib
def _find(slug, marker):
    for root in (pathlib.Path('/kaggle/input'),):
        for p in sorted(root.rglob(marker)):
            if slug in str(p): return p.parent
    for p in sorted(pathlib.Path('/kaggle/input').rglob(marker)): return p.parent
    raise FileNotFoundError(f'{slug}/{marker} under /kaggle/input')
if __name__ == '__main__':
    data = _find('mscapital-prepared-v2','manifest.json')
    packs = _find('mscapital-x32-packs','X32CTX_train_manifest.json')
    print('DATA',data,'PACKS',packs,flush=True)
    sys.argv += ['--data',str(data),'--packs',str(packs),'--out','/kaggle/working/run','--device','cuda','--max-hours','2.0']
