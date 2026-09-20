"""Re-run existing XS40 checkpoints with 37 equal sequential test groups."""
from __future__ import annotations
import gc,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
PROJECT=Path(__file__).resolve().parents[1]
REF=PROJECT/'data'/'interim'/'kaggle_kernels'/'relative319_xs_tabm_notebook'
sys.path.insert(0,str(REF))
import run_extracted as recipe
from train_full_gru_embedding_candidate import load_test_relative319

RUNS={
 'fulltrain':('tabm_relative319_xs40_seed42_fulltrain','model_fulltrain.pt'),
 'holdout059':('tabm_relative319_xs40_seed42_holdout059','model_holdout059.pt'),
}

def main():
 torch.set_num_threads(2); device=torch.device('cuda')
 base,template,columns=load_test_relative319(PROJECT)
 groups=(np.arange(len(base),dtype=np.int64)*37//len(base)).astype(np.int16)
 counts=np.bincount(groups,minlength=37)
 xs,xs_columns=recipe.add_relative_features(base,groups,columns)
 features=np.concatenate([base,xs],axis=1); del base,xs; gc.collect()
 out={}
 for key,(run_name,model_file) in RUNS.items():
  run_dir=PROJECT/'data'/'interim'/'submissions'/run_name
  saved=np.load(run_dir/'quantile_preprocessing.npz')
  pre=recipe.QuantilePreprocessor(saved['knots'],saved['medians'],saved['missing_columns'],device)
  model=recipe.make_model(pre.output_dimension,device)
  model.load_state_dict(torch.load(run_dir/model_file,map_location=device,weights_only=True))
  meta=json.loads((PROJECT/'outputs'/'submission_metadata'/f'{run_name}.json').read_text(encoding='utf-8'))
  # Existing metadata records prediction scale, while target standard deviation is recoverable from old/new predictions.
  old=pd.read_feather(PROJECT/'outputs'/'predictions'/f'{run_name}_test.feather').prediction.to_numpy(dtype=np.float64)
  raw=recipe.predict(model,features,np.arange(len(features)),pre,torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16).astype(np.float64)
  # The checkpoint predicts scaled targets; infer the exact saved training target scale from old prediction/raw pooled prediction.
  pooled_xs,_=recipe.add_relative_features(features[:,:319],np.zeros(len(features),dtype=np.int16),columns)
  pooled_features=np.concatenate([features[:,:319],pooled_xs],axis=1)
  pooled_raw=recipe.predict(model,pooled_features,np.arange(len(features)),pre,torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16).astype(np.float64)
  scale=float(np.dot(old,pooled_raw)/np.dot(pooled_raw,pooled_raw))
  pred=raw*scale
  new_name=f'{run_name}_equal37'
  pred_path=PROJECT/'outputs'/'predictions'/f'{new_name}_test.feather'
  sub_path=PROJECT/'outputs'/'submissions'/f'{new_name}.csv'
  pd.DataFrame({'sample_id':template.sample_id.to_numpy(),'prediction':pred}).to_feather(pred_path)
  pd.DataFrame({'sample_id':template.sample_id.to_numpy(),'prediction':pred}).to_csv(sub_path,index=False)
  out[key]={'source':run_name,'scale':scale,'prediction_std':float(pred.std()),'old_correlation':float(np.corrcoef(pred,old)[0,1]),'submission_path':str(sub_path)}
  del model,pre,raw,pooled_raw,pred,old; torch.cuda.empty_cache(); gc.collect()
 report={'status':'complete','test_group_count':37,'group_size_min':int(counts.min()),'group_size_max':int(counts.max()),'runs':out}
 path=PROJECT/'outputs'/'submission_metadata'/'xs40_equal37_candidates.json'; path.write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__': main()
