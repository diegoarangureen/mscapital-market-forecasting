"""Paired comparison of separate loss/control runs on the same protocol and backend."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit.dataset import Dataset
from audit.io import write_json
from audit.metrics import paired_panel


def compare(args):
    def read(folder,arm):
        folder=Path(folder)
        meta=json.loads(next(folder.glob('run_worker_*.json')).read_text(encoding='utf-8'))
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if not summary['complete']:
            raise ValueError('Incomplete panel')
        with np.load(folder/f'aggregate_{arm}.npz') as p:
            return meta,p['rows'].copy(),p['processed'].copy()
    b,rows,bp=read(args.base,args.base_arm)
    c,cr,cp=read(args.candidate,args.candidate_arm)
    for key in ('backend','torch_version','dataset_fingerprint','code_hashes'):
        if b[key]!=c[key]:
            raise ValueError(f'Incomparable runs: {key}')
    for key in ('mode','seeds','folds','origins','metric','clip_quantiles'):
        if b['config'].get(key)!=c['config'].get(key):
            raise ValueError(f'Incomparable protocol: {key}')
    if not np.array_equal(rows,cr):
        raise ValueError('Different evaluation rows')
    data=Dataset(args.data)
    if data.manifest['fingerprint']!=b['dataset_fingerprint']:
        raise ValueError('Different dataset')
    result=paired_panel(bp,cp,data.y[rows],data.months[rows],b['config']['metric'])
    result['changed_config']={k:[b['config'].get(k),c['config'].get(k)] for k in set(b['config'])|set(c['config']) if b['config'].get(k)!=c['config'].get(k)}
    result['independent_outer_score']=b['config']['mode']=='confirm'
    write_json(args.out,result)
    print('Delta:',result['delta'],'report:',args.out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','base','candidate','out'): p.add_argument('--'+name,required=True)
    p.add_argument('--base-arm',default='base'); p.add_argument('--candidate-arm',default='base')
    compare(p.parse_args())
