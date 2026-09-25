"""Recover a seed-averaged validation report from historical per-model OOF tensors."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit.metrics import PredictionMean, panel
from audit.io import write_json,save_npz


def recover(models, target, month, reports, arm=None):
    if models.ndim!=2 or models.shape[1]!=len(target) or len(target)!=len(month) or len(reports)!=len(models):
        raise ValueError('Historical model/label/report shapes differ')
    arms={r.get('arm','base') for r in reports}
    if arm is None and len(arms)>1:
        raise ValueError('Specify --arm; never average different feature variants')
    arm=arm or next(iter(arms))
    if arm not in arms:
        raise ValueError('Requested arm absent')
    mean=PredictionMean(len(target))
    for i,r in enumerate(reports):
        if r.get('arm','base')!=arm:
            continue
        rows=np.flatnonzero(np.isfinite(models[i]))
        mean.add(f'{arm}_{r["seed"]}_{r["fold"]}',rows,models[i,rows])
    pred=mean.mean(); seen=mean.count>0
    if not seen.any():
        raise ValueError('No predictions for requested arm')
    result={'arm':arm,'models':len(mean.models),'independent_outer_score':False,
            'note':'Checkpoint-selected validation; per-row counts are NOT the test ensemble size.',
            'all_seen':panel(pred[seen],target[seen],month[seen]),
            'min_models_per_seen_row':int(mean.count[seen].min()),
            'max_models_per_seen_row':int(mean.count[seen].max())}
    for lo,hi in [(61,70),(66,70),(40,64)]:
        rows=seen&(month>=lo)&(month<=hi)
        result[f'{lo}-{hi}']=panel(pred[rows],target[rows],month[rows]) if rows.any() else None
    return result,pred,mean.count


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('models','y','months','fold-report','out'): p.add_argument('--'+name,required=True)
    p.add_argument('--arm')
    a=p.parse_args()
    report=json.loads(Path(a.fold_report).read_text(encoding='utf-8'))
    if isinstance(report,dict): report=report['folds']
    result,pred,count=recover(np.load(a.models,mmap_mode='r'),np.load(a.y),np.load(a.months),report,a.arm)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    write_json(out/'rescored.json',result)
    save_npz(out/'oof_mean.npz',pred=pred,count=count)
    print(out/'rescored.json')
