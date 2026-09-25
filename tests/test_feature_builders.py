import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest

ROOT=Path(__file__).resolve().parents[1]


def raw_fixture(root):
    folder=root/'train'; folder.mkdir(parents=True)
    sid=[0,0,0,0,0,1,1,2,2]
    times=[65,35,30,10,0,20,0,0,0]
    market={'sample_id':sid,'seconds_before_predict':times}
    for name in ['ask_price_1','bid_price_1','ask_price_2','bid_price_2']:
        market[name]=[1.+(0.01 if 'ask' in name else -.01)+i*.001 for i in range(9)]
    for name in ['ask_volume_1','bid_volume_1','ask_volume_2','bid_volume_2']:
        market[name]=[10+i*(2 if 'bid' in name else 1) for i in range(9)]
    market['ask_price_1'][5]=0.; market['ask_volume_1'][5]=0.
    market.update(transaction_avgprice=[1.]*9,transaction_volume=[3.]*9,transaction_count=[1.]*9)
    tx={'sample_id':[0,0,0,0,1,2,2], 'seconds_before_predict':[60,40,20,0,0,10,0],
        'price':[1.,1.01,1.,1.03,1.,1.,1.01], 'volume':[1.,2.,3.,4.,100.,3.,5.],
        'side':[0,1,0,1,0,1,0]}
    order={'sample_id':[0,0,0,1,2,2], 'seconds_before_predict':[60,30,0,0,10,0],
           'price':[1.]*6, 'volume':[1.,2.,3.,4.,5.,6.], 'side':[0,1,0,1,0,1],
           'order_action':[0,1,0,0,1,0]}
    for name,table in [('market',market),('transaction',tx),('order',order)]:
        feather.write_feather(pa.table(table),folder/f'{name}.feather',chunksize=3)


@pytest.mark.parametrize('number,count',[(30,36),(31,18)])
def test_full_build_chunk_invariance_and_empty_samples(tmp_path,number,count):
    raw=tmp_path/'raw'; raw_fixture(raw)
    results=[]
    for chunk in [1,4,100]:
        out=tmp_path/f'out{chunk}'; out.mkdir()
        env={**os.environ,'BASE':str(raw),'OUT':str(out),'NS':'4','SPLIT':'train','CHUNK_ROWS':str(chunk)}
        p=subprocess.run([sys.executable,str(ROOT/f'src/kaggle/build_x{number}.py')],env=env,capture_output=True,text=True)
        assert p.returncode==0,p.stdout+p.stderr
        x=np.load(out/f'X{number}v2_train.npy')
        assert x.shape==(4,count) and np.isfinite(x).all()
        assert (out/f'X{number}v2_train_manifest.json').exists()
        results.append(x)
        names=np.load(out/f'X{number}v2_train_names.npy').tolist()
        if number==31:
            assert x[0,names.index('x31_ord_gap_mean')]==60
    for x in results[1:]:
        np.testing.assert_array_equal(x,results[0])
