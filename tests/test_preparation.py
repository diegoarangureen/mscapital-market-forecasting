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


def test_pack_arms_extend_flow31_with_verified_packs(tmp_path):
    import json
    from audit.io import sha256, signature, write_json, save_feature_pack
    from audit import VERSION
    n, nt = 8, 5
    root = tmp_path/'prepared'; root.mkdir()
    rng = np.random.default_rng(0)
    base_tr = rng.normal(size=(n, 450)).astype(np.float32)
    base_te = rng.normal(size=(nt, 450)).astype(np.float32)
    x30 = rng.normal(size=(n, 36)).astype(np.float32)
    x31 = rng.normal(size=(n, 18)).astype(np.float32)
    files = {}
    def save(name, arr):
        np.save(root/name, arr); files[name] = {'sha256': sha256(root/name), 'shape': list(arr.shape)}
    save('y.npy', np.zeros(n, np.float32)); save('month.npy', np.arange(n))
    save('train_ids.npy', np.arange(n, dtype=np.int64))
    save('base_train.npy', base_tr); save('base_test.npy', base_te)
    save('base_names.npy', np.array([f'b{i}' for i in range(450)]))
    x30names = (['ofi_a']*6 + [f'x30_ord_b{i}' for i in range(6)] + [f'x30_qi_c{i}' for i in range(6)]
                + ['x30_rv_60','x30_rv_600','x30_semi_ratio','x30_parkinson','x30_vol_of_vol',
                   'x30_signvol_60','x30_spread_rms','x30_spread_drift'] + [f'x30_p{i}' for i in range(10)])
    save('X30v2_train.npy', x30); save('X30v2_names.npy', np.array(x30names))
    save('X31v2_train.npy', x31); save('X31v2_names.npy', np.array([f'x31_{i}' for i in range(18)]))
    m = {'audit_version': VERSION, 'files': files, 'sources': {}, 'row_contract': 't', 'public_columns': [], 'drop_indices': []}
    m['fingerprint'] = signature(m)
    write_json(root/'manifest.json', m)
    packs = tmp_path/'packs'; packs.mkdir()
    ctx = rng.normal(size=(n, 12)).astype(np.float32)
    save_feature_pack(packs, 'X32CTX', 'train', ctx, [f'x32c_{i}' for i in range(12)])
    d = Dataset(root, packs_root=packs)
    rows = np.arange(n)
    flow31 = d.features('flow31', 'train', rows)
    ctx_out = d.features('flow31ctx', 'train', rows)
    assert ctx_out.shape == (n, 486+12)
    np.testing.assert_array_equal(ctx_out[:, :486], flow31)
    np.testing.assert_array_equal(ctx_out[:, 486:], ctx)
    assert d.names('flow31ctx')[-12:] == [f'x32c_{i}' for i in range(12)]
    with pytest.raises(ValueError):
        Dataset(root).features('flow31ctx', 'train', rows)
    # tampered pack must fail verification
    arr = np.load(packs/'X32CTX_train.npy'); arr[0, 0] += 1
    np.save(packs/'X32CTX_train.npy', arr)
    with pytest.raises(ValueError, match='checksum'):
        Dataset(root, packs_root=packs).features('flow31ctx', 'train', rows)
