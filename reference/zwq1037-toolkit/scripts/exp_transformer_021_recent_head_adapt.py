"""Paired recent-window versus all-history head adaptation for the strong factorized Transformer."""
from __future__ import annotations
import json
import os
import random
import time
from pathlib import Path

os.environ['OMP_NUM_THREADS']='2'; os.environ['MKL_NUM_THREADS']='2'; os.environ['OPENBLAS_NUM_THREADS']='2'; os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import exp_transformer_020_factorized_ema_dev as base

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'data/interim/tree_experiments/EXP-TRANSFORMER-021-RECENT-HEAD-ADAPT'
CHECKPOINT=ROOT/'data/interim/tree_experiments/EXP-TRANSFORMER-020-FACTORIZED-EMA/train059_valid6270_ex66/raw_best_state.pt'
SEED=2026
LR=2e-5
ADAPT_START_MONTH=36
MIX_WEIGHT=0.25
BATCH_SIZE=256


def trainable_head(model):
    for parameter in model.parameters(): parameter.requires_grad_(False)
    model.source_embedding.requires_grad_(True)
    for module in (model.fusion, model.source_attention, model.head):
        for parameter in module.parameters(): parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def adapt(model, loader, namespace, device, max_batches, label):
    parameters=trainable_head(model)
    optimizer=torch.optim.AdamW(parameters,lr=LR,weight_decay=1e-4)
    model.train(); total=0.0; rows=0; started=time.time()
    for batch_no,batch in enumerate(loader,1):
        if batch_no>max_batches: break
        market,transaction,order,static,target=batch
        inputs=[x.to(device,non_blocking=True).contiguous() for x in (market,transaction,order,static)]
        target=target.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction=torch.nan_to_num(model(*inputs))
        loss=0.35*F.smooth_l1_loss(prediction,target)+0.65*namespace['cosine_loss'](prediction,target)
        loss.backward(); nn.utils.clip_grad_norm_(parameters,1.0); optimizer.step()
        total+=float(loss.detach())*len(target); rows+=len(target)
        if batch_no%500==0: print(json.dumps({'branch':label,'batch':batch_no,'max_batches':max_batches,'loss':total/max(rows,1)}),flush=True)
    return {'branch':label,'batches':min(max_batches,batch_no),'rows':rows,'loss':total/max(rows,1),'seconds':time.time()-started}


def block_scores(target,months,prediction):
    return base.scores(target,months,prediction)


def deltas(candidate,reference):
    result={k:candidate[k]-reference[k] for k in ('all_ex66','62_65','67_70')}
    result['monthly']={m:candidate['monthly'][m]-reference['monthly'][m] for m in reference['monthly']}
    return result


def main():
    torch.set_num_threads(2)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    RUN.mkdir(parents=True,exist_ok=True)
    ids=np.load(base.CACHE_META/'sample_ids.npy',mmap_mode='r')
    months=np.load(base.CACHE_META/'months.npy',mmap_mode='r')
    target=np.load(base.CACHE_META/'targets.npy',mmap_mode='r')
    static=np.load(base.STATIC_PATH,mmap_mode='r')
    arrays=base.open_arrays(len(ids))
    all_train=np.flatnonzero(months<=59)
    recent_train=np.flatnonzero((months>=ADAPT_START_MONTH)&(months<=59))
    valid=np.flatnonzero((months>=62)&(months<=70))
    target_scale=float(np.std(target[all_train]))
    namespace,norm=base.extract_runtime(static)
    namespace['_STATIC_NORM']=namespace['_compute_static_norm'](static,all_train)
    dataset=namespace['GridDataset']
    def loader(rows,seed,shuffle=True):
        generator=torch.Generator().manual_seed(seed)
        return DataLoader(dataset(arrays,rows,norm,target=target,target_scale=target_scale),batch_size=BATCH_SIZE,shuffle=shuffle,generator=generator if shuffle else None,num_workers=0,pin_memory=True,drop_last=shuffle)
    recent_loader=loader(recent_train,SEED+101)
    all_loader=loader(all_train,SEED+101)
    valid_loader=DataLoader(dataset(arrays,valid,norm,target=target,target_scale=target_scale),batch_size=512,shuffle=False,num_workers=0,pin_memory=True)
    max_batches=len(recent_loader)
    device=torch.device('cuda')
    initial=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    def fresh():
        base.seed_all(SEED); model=namespace['_JointMultiStreamStaticModel']().to(device); model.load_state_dict(initial); return model
    model=fresh(); base_prediction=base.predict(model,valid_loader,device,target_scale); del model; torch.cuda.empty_cache()
    control=fresh(); control_log=adapt(control,all_loader,namespace,device,max_batches,'all_history_control'); control_prediction=base.predict(control,valid_loader,device,target_scale); torch.save(base.cpu_state(control),RUN/'control_state.pt'); del control; torch.cuda.empty_cache()
    recent=fresh(); recent_log=adapt(recent,recent_loader,namespace,device,max_batches,'recent_36_59'); recent_prediction=base.predict(recent,valid_loader,device,target_scale); torch.save(base.cpu_state(recent),RUN/'recent_state.pt'); del recent; torch.cuda.empty_cache()
    truth=np.asarray(target[valid],np.float64); valid_months=np.asarray(months[valid])
    base_scores=block_scores(truth,valid_months,base_prediction)
    control_scores=block_scores(truth,valid_months,control_prediction)
    recent_scores=block_scores(truth,valid_months,recent_prediction)
    control_blend=(1-MIX_WEIGHT)*base_prediction+MIX_WEIGHT*control_prediction
    recent_blend=(1-MIX_WEIGHT)*base_prediction+MIX_WEIGHT*recent_prediction
    control_blend_scores=block_scores(truth,valid_months,control_blend)
    recent_blend_scores=block_scores(truth,valid_months,recent_blend)
    control_delta=deltas(control_blend_scores,base_scores); recent_delta=deltas(recent_blend_scores,base_scores)
    forward_months=('67','68','69','70')
    passed=bool(recent_delta['62_65']>=0 and recent_delta['67_70']>=0.0005 and recent_delta['67_70']>=control_delta['67_70']+0.0002 and sum(recent_delta['monthly'][m]>0 for m in forward_months)>=2 and min(recent_delta['monthly'][m] for m in forward_months)>=-0.0005)
    np.savez_compressed(RUN/'validation_predictions.npz',sample_id=np.asarray(ids[valid]),month=valid_months,target=truth,base=base_prediction,control=control_prediction,recent=recent_prediction,control_blend=control_blend,recent_blend=recent_blend)
    summary={'experiment':'EXP-TRANSFORMER-021-RECENT-HEAD-ADAPT','checkpoint':str(CHECKPOINT),'frozen':'market/transaction/order/static adapters','trained':'source embedding, fusion, source attention, head','adapt_months':'36-59','control_months':'0-59 sampled for same batch count','learning_rate':LR,'mix_weight':MIX_WEIGHT,'max_batches':max_batches,'logs':{'control':control_log,'recent':recent_log},'scores':{'base':base_scores,'control':control_scores,'recent':recent_scores,'control_blend':control_blend_scores,'recent_blend':recent_blend_scores},'deltas_vs_base':{'control_blend':control_delta,'recent_blend':recent_delta},'gate':'recent mix: selection>=0, forward>=+0.0005, >=control+0.0002, >=2/4 forward months positive, worst forward month>=-0.0005','passed':passed,'next_action':'full-train recent adapter and CSV only if passed'}
    (RUN/'score_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
