import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from prepare_audited import prepare
from audit.dataset import Dataset
from rescore_legacy_oof import recover


def test_prepare_checks_ids_and_reproduces_450_schema(tmp_path):
    folders={k:tmp_path/k for k in ['data','x21','x22','public']}
    for f in folders.values(): f.mkdir()
    n=8
    full=np.arange(n*246,dtype=np.float32).reshape(n,246)
    np.save(folders['data']/'full_train.npy',full)
    names=['X2:tx_n','X2:mk_nbars']+[f'f{i}' for i in range(244)]
    np.save(folders['data']/'full_names.npy',np.array(names))
    y=np.arange(n,dtype=np.float32)*.001; m=np.arange(n)
    np.save(folders['data']/'full_y.npy',y); np.save(folders['data']/'full_month.npy',m)
    a=np.ones((n,30),np.float32); b=np.ones((n,27),np.float32)*2
    np.save(folders['x21']/'X21_train.npy',a); np.save(folders['x22']/'X22_train.npy',b)
    labels=tmp_path/'labels.feather'
    pd.DataFrame({'sample_id':np.arange(n),'target':y,'month':m}).to_feather(labels)
    public=pd.DataFrame({f'p{i}':np.arange(n,dtype=float)+i for i in range(152)})
    public.insert(0,'sample_id',np.arange(n))
    public.iloc[::-1].to_csv(folders['public']/'train.csv',index=False)
    args=argparse.Namespace(**{k:str(v) for k,v in folders.items()},labels=str(labels),out=str(tmp_path/'prepared'),x30=None,x31=None,include_test=False)
    prepare(args)
    d=Dataset(args.out)
    expected=np.concatenate([np.delete(np.concatenate([full,a,b],axis=1),[301,267,268,299,257],axis=1),public.drop(columns='sample_id').to_numpy()],axis=1)
    np.testing.assert_array_equal(d.features('base','train',np.arange(n)),expected)
    assert len(d.names('base'))==450
    public.loc[0,'sample_id']=1; public.to_csv(folders['public']/'train.csv',index=False)
    args.out=str(tmp_path/'bad')
    with pytest.raises(ValueError,match='row alignment'):
        prepare(args)


def test_historical_tensor_recovery_averages_and_separates_arms():
    models=np.array([[1.,2.,np.nan],[3.,4.,np.nan],[100.,100.,np.nan]])
    y=np.array([1.,2.,3.]); m=np.array([66,67,70])
    reports=[{'arm':'base','seed':1,'fold':5},{'arm':'base','seed':2,'fold':5},{'arm':'x30','seed':1,'fold':5}]
    with pytest.raises(ValueError,match='Specify'):
        recover(models,y,m,reports)
    result,pred,count=recover(models,y,m,reports,'base')
    np.testing.assert_equal(pred,[2.,3.,np.nan]); np.testing.assert_equal(count,[2,2,0])
    assert result['models']==2 and not result['independent_outer_score']
