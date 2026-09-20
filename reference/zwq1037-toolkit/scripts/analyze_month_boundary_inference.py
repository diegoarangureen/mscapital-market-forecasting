"""Evaluate whether month boundaries can be recovered without using month labels."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
FEATURE_PATH = PROJECT / 'data' / 'interim' / 'kaggle_relative319_dev' / 'features.npy'
COLUMN_PATH = FEATURE_PATH.parent / 'feature_columns.json'
LABEL_PATH = PROJECT / 'data' / 'raw' / 'label.feather'
FEATURE_NAMES = [
    'ask_price_1_mean','bid_price_1_mean','mid_price_1_mean','ask_price_1_last','bid_price_1_last',
    'last60_ask_price_1_mean','last60_bid_price_1_mean','ask_price_2_mean','bid_price_2_mean',
    'transaction_avgprice_mean','transaction_avgprice_last_valid','spread_1_mean','spread_2_mean',
    'book_volume_imbalance_1','last60_book_volume_imbalance_1','book_imbalance_1_mean_robust',
    'book_imbalance_2_mean_robust','total_book_imbalance_mean','microprice_displacement_mean',
    'ofi_1_mean','ofi_1_last','ofi_2_mean','multilevel_ofi_mean','log_total_depth_mean',
    'trade_volume_imbalance_60','trade_volume_imbalance_20','trade_count_imbalance_60',
    'new_order_volume_imbalance_10','net_order_pressure_30','last_trade_seconds_before_predict',
]
WINDOWS=(128,256,512,1024)
SEARCH_RADIUS=700


def rolling_change(values, window):
    n,d=values.shape
    cs=np.vstack([np.zeros((1,d),dtype=np.float64),np.cumsum(values,dtype=np.float64,axis=0)])
    pos=np.arange(window,n-window+1)
    left=(cs[pos]-cs[pos-window])/window
    right=(cs[pos+window]-cs[pos])/window
    score=np.sqrt(np.mean((right-left)**2,axis=1))
    full=np.full(n,np.nan,dtype=np.float64)
    full[pos]=score
    return full


def main():
    columns=json.loads(COLUMN_PATH.read_text(encoding='utf-8'))
    idx=[columns.index(x) for x in FEATURE_NAMES]
    mm=np.load(FEATURE_PATH,mmap_mode='r')
    values=np.asarray(mm[:,idx],dtype=np.float32)
    sample=np.linspace(0,len(values)-1,100000,dtype=np.int64)
    med=np.nanmedian(values[sample],axis=0)
    q25=np.nanpercentile(values[sample],25,axis=0)
    q75=np.nanpercentile(values[sample],75,axis=0)
    scale=np.maximum(q75-q25,1e-6)
    values=np.nan_to_num((values-med)/scale,nan=0.0,posinf=8.0,neginf=-8.0)
    values=np.clip(values,-8,8)
    labels=pd.read_feather(LABEL_PATH,columns=['sample_id','month']).sort_values('sample_id')
    counts=labels.groupby('month',sort=True).size().to_numpy()
    boundaries=np.cumsum(counts)[:-1]
    report={'feature_count':len(idx),'boundary_count':len(boundaries),'search_radius':SEARCH_RADIUS,'windows':{}}
    for window in WINDOWS:
        score=rolling_change(values,window)
        inferred=[]
        for boundary in boundaries:
            lo=max(window,boundary-SEARCH_RADIUS); hi=min(len(score)-window,boundary+SEARCH_RADIUS+1)
            best=lo+int(np.nanargmax(score[lo:hi]))
            inferred.append(best)
        errors=np.asarray(inferred)-boundaries
        report['windows'][str(window)]={
            'median_abs_error':float(np.median(np.abs(errors))),
            'mean_abs_error':float(np.mean(np.abs(errors))),
            'p90_abs_error':float(np.quantile(np.abs(errors),0.9)),
            'max_abs_error':int(np.max(np.abs(errors))),
            'within_50':float(np.mean(np.abs(errors)<=50)),
            'within_100':float(np.mean(np.abs(errors)<=100)),
            'errors':errors.tolist(),
        }
    out=PROJECT/'data'/'interim'/'month_boundary_inference'
    out.mkdir(parents=True,exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    compact={w:{k:v for k,v in x.items() if k!='errors'} for w,x in report['windows'].items()}
    print(json.dumps(compact,indent=2))

if __name__=='__main__': main()
