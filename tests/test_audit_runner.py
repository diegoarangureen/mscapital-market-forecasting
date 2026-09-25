import argparse
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import train_audited as runner
from audit import VERSION
from audit.io import signature,sha256,write_json
from audit.dataset import Dataset
from summarize_audited import summarize
from bundle_audited import bundle


@pytest.fixture
def dataset(tmp_path):
    out=tmp_path/'data'; out.mkdir()
    rng=np.random.default_rng(9)
    month=np.repeat(np.arange(71),2)
    x=rng.normal(size=(len(month),6)).astype(np.float32)
    y=(.003*x[:,0]+rng.normal(0,.002,len(month))).astype(np.float32)
    arrays={'base_train.npy':x,'base_test.npy':rng.normal(size=(5,6)).astype(np.float32),
            'base_names.npy':np.array([f'f{i}' for i in range(6)]),
            'y.npy':y,'month.npy':month,'train_ids.npy':np.arange(len(month)),
            'test_ids.npy':np.arange(5),'train_nodata.npy':np.zeros(len(month),bool),
            'test_nodata.npy':np.array([False,True,False,False,False])}
    for name,a in arrays.items():
        np.save(out/name,a)
    m={'audit_version':VERSION,'files':{name:{'shape':list(a.shape),'sha256':sha256(out/name)} for name,a in arrays.items()}}
    m['fingerprint']=signature(m); write_json(out/'manifest.json',m)
    return out


def config_file(tmp_path,**overrides):
    cfg={'mode':'screen','arms':['base'],'seeds':[7,11],'folds':[4],
         'epochs':2,'n_ens':2,'batch_size':64,'noise_std':.001,'patience':3,
         'metric':'cosine','metric_verified':False,**overrides}
    path=tmp_path/'config.json'; write_json(path,cfg); return path


def args(config,data,out):
    return runner.parser().parse_args(['--config',str(config),'--data',str(data),'--out',str(out),'--device','cpu'])


def summarize_args(data,out,template=None):
    return argparse.Namespace(data=str(data),out=str(out),submission_template=template)


def test_real_model_two_seed_run_correct_mean_and_idempotent_resume(tmp_path,dataset):
    torch.set_num_threads(1)
    config=config_file(tmp_path)
    out=tmp_path/'run'; a=args(config,dataset,out)
    runner.run(a)
    result=summarize(summarize_args(dataset,out))
    assert result['complete'] and not result['independent_outer_score']
    paths=sorted((out/'jobs').glob('*/predictions.npz'))
    pp=[np.load(p)['score_pred'] for p in paths]
    with np.load(out/'aggregate_base.npz') as p:
        np.testing.assert_allclose(p['raw'],np.mean(np.stack(pp).astype(np.float64),axis=0),rtol=0,atol=0)
    digest=[sha256(p) for p in paths]
    runner.run(a)
    assert [sha256(p) for p in paths]==digest
    cfg=json.loads(config.read_text()); cfg['noise_std']=0; write_json(config,cfg)
    with pytest.raises(ValueError,match='another recipe'):
        runner.run(a)


def test_epoch_checkpoint_resume_matches_uninterrupted(tmp_path,dataset,monkeypatch):
    torch.set_num_threads(1)
    config=config_file(tmp_path,seeds=[7],epochs=3)
    full=tmp_path/'full'; interrupted=tmp_path/'interrupted'
    runner.run(args(config,dataset,full))
    original=runner.atomic_checkpoint
    class Interrupt(Exception): pass
    def interrupt(path,state,device):
        original(path,state,device)
        if path.name=='last.pt' and state['next_epoch']==1:
            raise Interrupt()
    monkeypatch.setattr(runner,'atomic_checkpoint',interrupt)
    with pytest.raises(Interrupt):
        runner.run(args(config,dataset,interrupted))
    monkeypatch.setattr(runner,'atomic_checkpoint',original)
    runner.run(args(config,dataset,interrupted))
    p1=np.load(next((full/'jobs').glob('*/predictions.npz')))
    p2=np.load(next((interrupted/'jobs').glob('*/predictions.npz')))
    for key in p1.files:
        np.testing.assert_array_equal(p1[key],p2[key])


class Tiny(torch.nn.Module):
    def __init__(self,n_features,n_ens):
        super().__init__(); self.proj=torch.nn.Linear(n_features,n_ens)
    def forward(self,x): return self.proj(x)


def test_full_nested_ensemble_counts_and_partial_rejection(tmp_path,dataset,monkeypatch):
    torch.set_num_threads(1); monkeypatch.setattr(runner,'RealMLP',Tiny)
    config=config_file(tmp_path,mode='confirm',seeds=[7],folds=[0,1,2,3,4],epochs=1)
    out=tmp_path/'nested'; a=args(config,dataset,out)
    a.job_key='base_o59_f0_s7'; runner.run(a)
    incomplete=summarize(summarize_args(dataset,out))
    assert not incomplete['complete'] and len(incomplete['missing'])==9
    a.job_key=None; runner.run(a)
    result=summarize(summarize_args(dataset,out))
    assert result['complete'] and result['independent_outer_score']
    assert result['arms']['base']['model_count_by_origin']=={'59':5,'64':5}
    assert '66' in result['arms']['base']['processed']['monthly']


def test_final_export_reorders_ids_and_zeroes_no_data(tmp_path,dataset,monkeypatch):
    import pandas as pd
    torch.set_num_threads(1); monkeypatch.setattr(runner,'RealMLP',Tiny)
    config=config_file(tmp_path,mode='final',seeds=[7],folds=[0,1,2,3,4],epochs=1,
                       metric_verified=True,metric_source='Synthetic test scorer')
    out=tmp_path/'final'; runner.run(args(config,dataset,out))
    template=tmp_path/'template.csv'
    pd.DataFrame({'sample_id':[4,1,3,0,2],'target':0.}).to_csv(template,index=False)
    summarize(summarize_args(dataset,out,str(template)))
    result=pd.read_csv(out/'submission.csv')
    assert result.sample_id.tolist()==[4,1,3,0,2]
    assert result.set_index('sample_id').loc[1,'target']==0
    with np.load(out/'aggregate_base.npz') as p:
        np.testing.assert_allclose(result.target.to_numpy(),p['test_processed'][[4,1,3,0,2]])


def test_dataset_checksum_and_final_metric_guard(tmp_path,dataset):
    config=config_file(tmp_path,mode='final',folds=[0,1,2,3,4])
    with pytest.raises(ValueError,match='official scorer'):
        runner.read_config(config)
    np.save(dataset/'y.npy',np.zeros(142))
    with pytest.raises(ValueError,match='checksum'):
        Dataset(dataset)


def test_standalone_bundle_plan(tmp_path):
    import subprocess,sys
    config=config_file(tmp_path)
    output=tmp_path/'standalone.py'; bundle('train_audited.py',output,config)
    p=subprocess.run([sys.executable,str(output),'--plan'],capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    assert json.loads(p.stdout)['models']==2
