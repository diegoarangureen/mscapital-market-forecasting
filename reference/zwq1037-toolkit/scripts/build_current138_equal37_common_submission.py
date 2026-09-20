"""Build the deployable 37-block month-common correction of the current 0.138 submission."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
PROJECT=Path(__file__).resolve().parents[1]
REF=PROJECT/'data'/'interim'/'kaggle_kernels'/'relative319_xs_tabm_notebook';sys.path.insert(0,str(REF))
import run_extracted as recipe
from train_full_gru_embedding_candidate import load_test_relative319
RUN='current138_equal37_common_ridge100'
ALPHA=100.0
GROUPS=37
def summary(values,groups,target=None):
 rows=[]; ys=[]; keys=[]
 for g in np.unique(groups):
  ix=np.flatnonzero(groups==g); v=np.asarray(values[ix],float)
  rows.append(np.r_[np.nanmean(v,0),np.nanstd(v,0)]); keys.append(int(g))
  if target is not None: ys.append(float(target[ix].mean()))
 return np.asarray(rows),np.asarray(ys),np.asarray(keys)
def main():
 labels=pd.read_feather(PROJECT/'data'/'raw'/'label.feather',columns=['sample_id','month','target']).sort_values('sample_id').reset_index(drop=True)
 train=np.load(PROJECT/'data'/'interim'/'kaggle_relative319_dev'/'features.npy',mmap_mode='r'); columns=json.loads((PROJECT/'data'/'interim'/'kaggle_relative319_dev'/'feature_columns.json').read_text()); ix=[columns.index(c) for c in recipe.TOP_FEATURES]
 tx,ty,_=summary(train[:,ix],labels.month.to_numpy(),labels.target.to_numpy(float)); med=np.nanmedian(tx,0); tx=np.where(np.isfinite(tx),tx,med); scaler=StandardScaler().fit(tx); model=Ridge(alpha=ALPHA).fit(scaler.transform(tx),ty)
 test,template,test_columns=load_test_relative319(PROJECT); assert test_columns==columns
 group=(np.arange(len(test),dtype=np.int64)*GROUPS//len(test)).astype(np.int16); vx,_,keys=summary(test[:,ix],group); vx=np.where(np.isfinite(vx),vx,med); common=dict(zip(keys,model.predict(scaler.transform(vx))))
 source=PROJECT/'outputs'/'submissions'/'current135_hybrid2_conv1_transformer30.csv'; frame=pd.read_csv(source); assert np.array_equal(frame.sample_id.to_numpy(),template.sample_id.to_numpy()); pred=frame.prediction.to_numpy(dtype=np.float64); target_scale=float(labels.target.to_numpy(float).std())
 before=pred.copy()
 for g in np.unique(group):
  z=group==g; pred[z]=pred[z]-pred[z].mean()+common[int(g)]/target_scale
 out_path=PROJECT/'outputs'/'submissions'/f'{RUN}.csv'; pd.DataFrame({'sample_id':frame.sample_id,'prediction':pred}).to_csv(out_path,index=False)
 report={'run_name':RUN,'source_submission':str(source),'method':'37 equal sequential groups; replace each prediction group mean with Ridge(alpha=100) input-only common target component / full target std','rows':len(pred),'group_count':GROUPS,'group_size_min':int(np.bincount(group).min()),'group_size_max':int(np.bincount(group).max()),'source_correlation':float(np.corrcoef(before,pred)[0,1]),'source_std':float(before.std()),'prediction_std':float(pred.std()),'prediction_mean':float(pred.mean()),'common_normalized_min':float(min(common.values())/target_scale),'common_normalized_max':float(max(common.values())/target_scale),'local_scores':{'first_raw':0.155996,'first_candidate':0.158018,'first_delta':0.002021,'second_raw':0.157262,'second_candidate':0.158458,'second_delta':0.001196},'submission_status':'prepared_not_submitted','output_path':str(out_path)}
 (PROJECT/'outputs'/'submission_metadata'/f'{RUN}.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__':main()
