from pathlib import Path
import json
import numpy as np
import pandas as pd
root=Path('.')
meta_path=root/'outputs/submission_metadata/tabm385_k32_temporal5fold_3seed_ema0999.json'
meta=json.loads(meta_path.read_text(encoding='utf-8'))
keys=['equal_ema000','equal_ema025','equal_ema050','equal_ema075','equal_ema100','linear_recent_ema100','last3_equal_ema000','last3_equal_ema100','last3_recent_ema100']
by={r['key']:r for r in meta['candidates']}
rows=[{k:r[k] for k in ('key','selection_63_65','forward_67_70','all_63_70_ex66','robust_score')} for key in keys if (r:=by.get(key))]
top=sorted(meta['candidates'],key=lambda r:r['robust_score'],reverse=True)[:8]
checks=[]
for name,path_text in meta['outputs'].items():
 p=Path(path_text); x=pd.read_csv(p)
 checks.append({'name':name,'path':str(p.resolve()),'rows':len(x),'columns':list(x.columns),'unique_ids':int(x.sample_id.nunique()),'finite':bool(np.isfinite(x.prediction).all()),'mean':float(x.prediction.mean()),'std':float(x.prediction.std()),'min':float(x.prediction.min()),'max':float(x.prediction.max())})
old=pd.read_csv(root/'outputs/submissions/tabm385_k32_temporal3fold_3seed.csv').sort_values('sample_id')
new={name:pd.read_csv(path).sort_values('sample_id') for name,path in meta['outputs'].items()}
corr={name:float(np.corrcoef(old.prediction,new[name].prediction)[0,1]) for name in new}
model_dirs=list((root/'data/interim/submissions/tabm385_k32_temporal5fold_3seed_ema0999').glob('train*/seed_*/result.json'))
print(json.dumps({'model_results':len(model_dirs),'selected':meta['selected'],'relevant':rows,'top8':top,'csv_checks':checks,'correlation_with_old_public_3x3':corr},ensure_ascii=False,indent=2))
