from pathlib import Path
import numpy as np
import pandas as pd
from exp_tabm_011_quantile_preprocessing import evaluate
from exp_tabm_014_quantile_stage_curves import unit

P=Path(__file__).resolve().parents[1]
D=P/'data'/'interim'

def load(fold, start, end):
    tabm=pd.read_feather(D/'tree_experiments'/'EXP-TABM-016-RELATIVE-SCALE-ADD12'/fold/'validation_predictions.feather')
    t42=pd.read_feather(D/'tree_experiments'/'EXP-TREE-071-GRU96-EMBEDDING'/fold/'validation_predictions.feather')
    t137=pd.read_feather(D/'tree_experiments'/'EXP-TREE-072-GRU96-EMBEDDING-SEED137'/fold/'validation_predictions.feather')
    if start==50:
        j42=pd.read_feather(D/'sequence_experiments'/'EXP-GRU-003-STRONG-JOINT-DEV'/'joint_gru319'/'validation_predictions_epoch06.feather')
        j137=pd.read_feather(D/'sequence_experiments'/'EXP-GRU-005-JOINT-SEED137'/fold/'joint_gru319'/'validation_predictions_epoch06.feather')
        h42=pd.read_feather(D/'sequence_experiments'/'EXP-TRANSFORMER-006-HYBRID-LOSS-DEV'/'validation_predictions_epoch05.feather')
        h137=pd.read_feather(D/'sequence_experiments'/'EXP-TRANSFORMER-008-HYBRID-LOSS-SEED137-DEV'/'validation_predictions_epoch04.feather')
    else:
        j42=pd.read_feather(D/'sequence_experiments'/'EXP-GRU-004-STRONG-JOINT-CONFIRM'/'joint_gru319'/'validation_predictions_epoch06.feather')
        j137=pd.read_feather(D/'sequence_experiments'/'EXP-GRU-005-JOINT-SEED137'/fold/'joint_gru319'/'validation_predictions_epoch06.feather')
        h42=pd.read_feather(D/'sequence_experiments'/'EXP-TRANSFORMER-007-HYBRID-LOSS-CONFIRM'/'validation_predictions_epoch05.feather')
        h137=pd.read_feather(D/'sequence_experiments'/'EXP-TRANSFORMER-009-HYBRID-LOSS-SEED137-CONFIRM'/'validation_predictions_epoch04.feather')
    ids=tabm.sample_id.to_numpy()
    for x in (t42,t137,j42,j137,h42,h137): assert np.array_equal(x.sample_id.to_numpy(),ids)
    emb=.5*unit(t42.gru_embedding_tree.to_numpy(float))+.5*unit(t137.gru_embedding_tree.to_numpy(float))
    joint=.5*unit(j42.prediction.to_numpy(float))+.5*unit(j137.prediction.to_numpy(float))
    tree=.75*unit(tabm.tabm_trim1.to_numpy(float))+.20*unit(tabm.xgboost.to_numpy(float))+.05*unit(emb)
    current=.90*unit(tree)+.10*unit(joint)
    h=.5*unit(h42.prediction.to_numpy(float))+.5*unit(h137.prediction.to_numpy(float)); y=tabm.target.to_numpy(float); m=tabm.month.to_numpy()
    base=evaluate(y,current,m,start,end)['overall']
    rows=[]
    for w in (.025,.05,.10,.15,.20):
        append=(1-w)*unit(current)+w*unit(h)
        replace=.90*unit(tree)+(0.10-w)*unit(joint)+w*unit(h) if w<=.10 else None
        rows.append({'window':f'{start}-{end}','type':'append','w':w,'score':evaluate(y,append,m,start,end)['overall'],'delta':evaluate(y,append,m,start,end)['overall']-base})
        if replace is not None: rows.append({'window':f'{start}-{end}','type':'replace_gru','w':w,'score':evaluate(y,replace,m,start,end)['overall'],'delta':evaluate(y,replace,m,start,end)['overall']-base})
    return base,rows

rows=[]
for args in [('train049_valid5059',50,59),('train059_valid6070',60,70)]:
    base,r=load(*args); print(args[1],args[2],'current',base); rows+=r
print(pd.DataFrame(rows).to_string(index=False))

