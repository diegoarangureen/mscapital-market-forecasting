"""Compare complete X34 screens with exactly one angular-loss configuration change."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit.io import write_json
from audit.metrics import paired_panel, postprocess
from review_champion import labels, recovered_run


def review(args):
    dm, data = labels(args.labels)
    bm, br = recovered_run(args.control, dm, data)
    cm, cr = recovered_run(args.candidate, dm, data)
    for key in ('backend','torch_version','torch_xla_version','code_hashes','dataset_fingerprint'):
        if bm.get(key) != cm.get(key):
            raise ValueError('Incomparable provenance: ' + key)
    changes = {k: [bm['config'].get(k),cm['config'].get(k)]
               for k in set(bm['config'])|set(cm['config']) if bm['config'].get(k)!=cm['config'].get(k)}
    if changes != {args.difference: ['noisy','clean'] if args.difference=='angular_target' else ['members','mean']}:
        raise ValueError('Expected only the declared angular-loss change')
    cfg = bm['config']
    if (cfg['mode']!='screen' or cfg['folds']!=[0,4] or cfg['seeds']!=[2026,42]
            or cfg['clip_quantiles'] is not None or cfg['target_winsor'] is not None):
        raise ValueError('Unexpected screen protocol/postprocessing')
    base, cand = {j.key:(j,p) for j,p in br}, {j.key:(j,p) for j,p in cr}
    if base.keys()!=cand.keys() or len(base)!=4:
        raise ValueError('Incomplete/mismatched screen panel')
    singles, folds, rows_all, bp_all, cp_all = {}, {}, [], [], []
    for fold in cfg['folds']:
        bp,cp = [],[]
        rows = None
        for seed in cfg['seeds']:
            key=f'flow31_o70_f{fold}_s{seed}'
            j,p=base[key]; _,q=cand[key]
            if not np.array_equal(p['score_rows'],q['score_rows']):
                raise ValueError('Control/candidate row mismatch')
            if rows is not None and not np.array_equal(rows,p['score_rows']):
                raise ValueError('Seed row mismatch')
            rows=p['score_rows']; nd=data['train_nodata'][rows]
            b=postprocess(p['score_pred'],None,nd); c=postprocess(q['score_pred'],None,nd)
            singles[key]=paired_panel(b,c,data['y'][rows],data['month'][rows])
            bp.append(b); cp.append(c)
        b,c=np.mean(bp,axis=0),np.mean(cp,axis=0)
        folds[str(fold)]=paired_panel(b,c,data['y'][rows],data['month'][rows])
        rows_all.append(rows); bp_all.append(b); cp_all.append(c)
    rr=np.concatenate(rows_all)
    if len(np.unique(rr))!=len(rr): raise ValueError('Overlapping fold rows')
    pooled=paired_panel(np.concatenate(bp_all),np.concatenate(cp_all),data['y'][rr],data['month'][rr])
    gate={'each_single_ge_0p002':all(v['delta']>=.002 for v in singles.values()),
          'each_fold_ensemble_ge_0p002':all(v['delta']>=.002 for v in folds.values()),
          'majority_months_positive':sum(v>0 for v in pooled['monthly_delta'].values())>len(pooled['monthly_delta'])/2,
          'all_lomo_positive':all(v>0 for v in pooled['lomo_delta'].values())}
    gate['passed']=all(gate.values())
    result={'complete':True,'changed_config':changes,'gate':gate,'paired':pooled,
            'by_fold':folds,'by_fold_seed':singles,'independent_outer_score':False,
            'reference_signatures':[bm['run_signature'],cm['run_signature']],
            'note':'ES-selected checkpoint windows, not independent confirmation. A passing screen needs temporal confirmation. No automatic submission.'}
    write_json(args.out,result)
    print(json.dumps({'delta':pooled['delta'],'gate':gate,'folds':{k:v['delta'] for k,v in folds.items()}},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('labels','control','candidate','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--difference',choices=['angular_target','angular_aggregation'],required=True)
    review(p.parse_args())
